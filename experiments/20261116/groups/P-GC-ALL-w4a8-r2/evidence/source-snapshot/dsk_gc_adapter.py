"""Opt-in SM90 GC-MoE integration. No patch is installed on import."""
import os
import types
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import gc_moe

_BUFFERS = {}

@triton.jit
def _stage_gc(X, IDS, W, Q, SF, GI, GW, M:tl.constexpr, H:tl.constexpr,
              K:tl.constexpr, CHUNK:tl.constexpr, RANK:tl.constexpr,
              XS:tl.constexpr, IS:tl.constexpr, WS:tl.constexpr):
    row=tl.program_id(0); block=tl.program_id(1); source=RANK*CHUNK+row
    col=block*128+tl.arange(0,128)
    value=tl.load(X+source*XS+col,source<M,other=0).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(value),0),1.e-4)/448.
    tl.store(Q+row*H+col,(value/scale).to(tl.float8e4nv))
    tl.store(SF+row*(H//128)+block,scale)
    if block==0:
        j=tl.arange(0,8)
        idx=tl.load(IDS+source*IS+j,(source<M)&(j<K),other=0)
        weight=tl.load(W+source*WS+j,(source<M)&(j<K),other=0).to(tl.float32)
        tl.store(GI+row*K+j,idx,j<K);tl.store(GW+row*K+j,weight,j<K)

@triton.jit
def _gc_finalize(R,S,O,N:tl.constexpr,SCALE:tl.constexpr,HAS_SHARED:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    r=tl.load(R+i,i<N,other=0).to(tl.float32)
    # Match native BF16 in-place multiply, then BF16 shared addition.
    r=(r*SCALE).to(tl.bfloat16).to(tl.float32)
    if HAS_SHARED:r=r+tl.load(S+i,i<N,other=0).to(tl.float32)
    tl.store(O+i,r,i<N)

def finalize(routed,shared,scale):
    assert routed.dtype==torch.bfloat16 and routed.is_contiguous()
    assert shared is None or (shared.shape==routed.shape and shared.dtype==routed.dtype and shared.is_contiguous())
    out=torch.empty_like(routed)
    n=routed.numel()
    if n:_gc_finalize[(triton.cdiv(n,512),)](routed,shared if shared is not None else routed,out,n,float(scale),shared is not None,512,enable_fp_fusion=False)
    return out

class GCExecution:
    """Replicated TP rows -> unique EP ownership -> GC -> replicated rows."""
    def __init__(self,weights,group,max_rows=4096,hidden=5120,intermediate=2304,experts=384,topk=6):
        self.group=group;self.rank=dist.get_rank(group);self.ranks=dist.get_world_size(group)
        self.hidden=hidden;self.topk=topk;self.max_rows=max_rows;self.weights=weights
        assert self.ranks==8 and experts==384
        capacity=triton.cdiv(max_rows,self.ranks)
        key=(id(group),capacity,hidden,intermediate,experts,topk)
        if key not in _BUFFERS:
            _BUFFERS[key]=gc_moe.get_symm_buffer_for_gc_moe_sm90(group,experts,capacity,topk,hidden,intermediate,activation='swiglu')
        self.buffer=_BUFFERS[key]
    def run(self,x,ids,weights,gather=None):
        m=x.shape[0];h=self.hidden
        assert x.dtype==torch.bfloat16 and x.ndim==2 and x.shape[1]==h and x.stride(1)==1
        assert ids.shape==weights.shape==(m,self.topk) and ids.stride(1)==weights.stride(1)==1
        assert 0<=m<=self.max_rows
        if m==0:return torch.empty_like(x)
        chunk=triton.cdiv(m,self.ranks);b=self.buffer
        _stage_gc[(chunk,h//128)](x,ids,weights,b.x,b.x_sf,b.topk_idx,b.topk_weights,m,h,self.topk,chunk,self.rank,x.stride(0),ids.stride(0),weights.stride(0),num_warps=4,enable_fp_fusion=False)
        y=torch.empty((chunk,h),device=x.device,dtype=torch.bfloat16)
        gc_moe.sm90_fp8_fp4_gc_moe(y,*self.weights,b,recipe=(1,1,32),activation='swiglu',activation_clamp=10.,fast_math=False,num_tokens_bound=chunk)
        if gather is None:
            result=torch.empty((chunk*self.ranks,h),device=x.device,dtype=x.dtype)
            dist.all_gather_into_tensor(result,y,group=self.group)
        else:result=gather(y)
        return result[:m]

class _KernelContract:
    can_overlap_shared_experts=False
    is_monolithic=False
    fused_experts=types.SimpleNamespace(consumes_expert_mask=False)
    @staticmethod
    def output_is_reduced():return True

_INSTALLED=False

def install():
    global _INSTALLED
    if _INSTALLED:return
    assert os.environ.get('DSK_GC_MOE','0')=='1'
    assert os.environ.get('DSK_HUMMING_A8','0')=='0' and os.environ.get('DSK_HUMMING_FINALIZE','0')=='0'
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
    from vllm.models.deepseek_v41 import quant_config as cfg
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
    from vllm.model_executor.utils import replace_parameter
    from vllm.distributed import get_ep_group,get_tp_group
    from vllm.utils.torch_utils import direct_register_custom_op
    class GCMethod(Mxfp4MoEMethod):
        _dsk_gc=True
        def __init__(self,moe):
            # Keep upstream checkpoint loader, replacing only expert execution.
            FusedMoEMethodBase.__init__(self,moe)
            self.weight_dtype='mxfp4';self.activation_key=None;self.max_capture_size=moe.max_capture_size
            self.moe_kernel=_KernelContract()
            assert moe.num_experts==384 and moe.hidden_dim==5120 and moe.intermediate_size==2304
            p=moe.moe_parallel_config
            assert p.ep_size==8 and p.dp_size==1 and p.sp_size==1 and p.pcp_size==1 and not p.enable_eplb
            assert not moe.skip_final_all_reduce and not moe.has_bias
        @property
        def supports_eplb(self):return False
        @property
        def topk_indices_dtype(self):return torch.int64
        @property
        def is_monolithic(self):return False
        @property
        def skip_forward_padding(self):return True
        @property
        def has_unpadded_output(self):return True
        def maybe_roundup_sizes(self,hidden_size,intermediate_size_per_partition,act_dtype,moe_parallel_config):
            return hidden_size,intermediate_size_per_partition
        def get_fused_moe_quant_config(self,layer):return None
        def process_weights_after_loading(self,layer):
            assert layer.w13_weight.shape==(48,4608,2560) and layer.w2_weight.shape==(48,5120,1152)
            converted=gc_moe.transform_weights_for_gc_moe_sm90((layer.w13_weight,layer.w13_weight_scale),(layer.w2_weight,layer.w2_weight_scale))
            for prefix,pair in zip(['w13','w2'],converted):
                replace_parameter(layer,prefix+'_weight',pair[0]);replace_parameter(layer,prefix+'_weight_scale',pair[1])
            self.execution=GCExecution(((layer.w13_weight,layer.w13_weight_scale),(layer.w2_weight,layer.w2_weight_scale)),get_ep_group().device_group,max_rows=self.moe.max_num_tokens)
            assert get_ep_group().rank_in_group==get_tp_group().rank_in_group
        def apply(self,layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input):
            assert not layer.apply_router_weight_on_input
            return self.execution.run(x,topk_ids,topk_weights,gather=lambda y:get_tp_group().all_gather(y,dim=0))
    native=cfg.Mxfp4MoEMethod
    def choose(moe,*args,**kwargs):
        if moe.num_experts==384 and moe.hidden_dim==5120:return GCMethod(moe)
        return native(moe,*args,**kwargs)
    cfg.Mxfp4MoEMethod=choose
    if os.environ.get('DSK_GC_FINALIZE','0')=='1':
        def op(routed:torch.Tensor,shared:torch.Tensor|None,scale:float)->torch.Tensor:return finalize(routed,shared,scale)
        def fake(routed,shared,scale):return torch.empty_like(routed)
        direct_register_custom_op(op_name='dsk_gc_output_finalize',op_func=op,mutates_args=[],fake_impl=fake)
        original=MoERunner._maybe_apply_routed_scale_to_output
        def fused(self,shared,routed):
            if getattr(self._quant_method,'_dsk_gc',False):
                assert self.routed_output_transform is None
                return None,torch.ops.vllm.dsk_gc_output_finalize(routed,shared,float(self.routed_scaling_factor))
            return original(self,shared,routed)
        MoERunner._maybe_apply_routed_scale_to_output=fused
    _INSTALLED=True
