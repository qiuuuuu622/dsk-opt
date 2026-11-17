"""Default-off, bounded Humming grouped finalize + BF16 scale/shared fusion.

No patch on import. TP8/EP8, replicated rows, no DBO/transforms only.
Shared output is peeked without consuming its destructive getter, and its
aux stream event is awaited at the finalizer, preserving compute overlap.
"""
import contextvars,os,json,pathlib
_HITS={}
import torch
_ACTIVE=contextvars.ContextVar('dsk_full_finalize',default=None)
_INSTALLED=False

def eligible_call(runner,x):
    return getattr(runner,'_dsk_full_finalize',False) and x.dtype==torch.bfloat16 and x.ndim==2 and 0<x.shape[0]<=128

def install():
    global _INSTALLED
    assert os.environ.get('DSK_HUMMING_FULL_FINALIZE','0')=='1'
    if _INSTALLED:return
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
    from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe as hm
    from vllm_dsv41_moe import ops
    original_apply=MoERunner._apply_quant_method
    original_scale=MoERunner._maybe_apply_routed_scale_to_output
    original_grouped=hm.moe_unpermute
    def grouped(*args,**kwargs):
        ctx=_ACTIVE.get()
        if ctx is None:return original_grouped(*args,**kwargs)
        runner=ctx['runner'];shared=runner._shared_experts
        assert not ctx['hit'],'Expected one grouped finalizer per layer call'
        if ctx['overlap']:shared.wait()
        # .output consumes the value; leave it for the normal runner boundary.
        value=shared._output[shared._output_idx]
        out=kwargs.get('out',args[0] if args else None)
        assert value is not None and value.shape==out.shape and value.dtype==torch.bfloat16
        assert value.stride(1)==1
        ops.moe_unpermute(*args,**kwargs,shared=value,routed_scaling_factor=float(runner.routed_scaling_factor))
        ctx['hit']=True
        directory=os.environ.get('DSK_FULL_FINALIZE_EVIDENCE_DIR')
        name=getattr(runner,'layer_name',None)
        if directory and name and name not in _HITS:
            import torch.distributed as dist
            _HITS[name]={'rows':out.shape[0],'scale':float(runner.routed_scaling_factor),'shared_fused':True,'async_wait_at_finalizer':ctx['overlap'],'before_EP_reduce':True}
            path=pathlib.Path(directory);path.mkdir(parents=True,exist_ok=True)
            (path/f'branch-rank{dist.get_rank()}.json').write_text(json.dumps(_HITS,indent=2))
    def apply(self,hidden_states,router_logits,shared_experts_input,input_ids=None,shared_experts_overlapping=False):
        if not eligible_call(self,hidden_states):
            return original_apply(self,hidden_states,router_logits,shared_experts_input,input_ids,shared_experts_overlapping)
        self._maybe_apply_shared_experts(shared_experts_input,SharedExpertsOrder.NO_OVERLAP)
        assert not self.routed_experts.quant_method.is_monolithic
        weights,ids=self.router.select_experts(hidden_states=hidden_states,router_logits=router_logits,topk_indices_dtype=self._quant_method.topk_indices_dtype,input_ids=input_ids)
        ctx={'runner':self,'overlap':shared_experts_overlapping,'hit':False}
        token=_ACTIVE.set(ctx)
        try:
            fused=self.routed_experts.forward_modular(x=hidden_states,topk_weights=weights,topk_ids=ids,shared_experts=self._shared_experts,shared_experts_input=shared_experts_input)
            assert ctx['hit'],'Eligible layer failed to select full finalizer; refusing double scaling'
            return self._shared_experts.output,fused
        finally:_ACTIVE.reset(token)
    def scale(self,shared_output,fused_output):
        if eligible_call(self,fused_output):
            assert shared_output is not None
            return None,fused_output
        return original_scale(self,shared_output,fused_output)
    hm.moe_unpermute=grouped
    MoERunner._apply_quant_method=apply
    MoERunner._maybe_apply_routed_scale_to_output=scale
    _INSTALLED=True

def enable_target(runner):
    """Fail closed on unsupported execution contracts before any graph capture."""
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingGroupedExperts
    c=runner.moe_config
    assert c.num_experts==384 and c.hidden_dim==5120
    runner.routed_experts._ensure_moe_quant_config_init()
    assert c.dp_size==1 and not c.is_sequence_parallel
    assert c.moe_parallel_config.ep_size==8 and c.moe_parallel_config.pcp_size==1
    assert not runner.enable_dbo and runner.routed_input_transform is None and runner.routed_output_transform is None
    assert runner._shared_experts is not None and not runner._fused_output_is_reduced
    assert not runner.do_naive_dispatch_combine
    assert not runner._quant_method.is_monolithic
    kernel=runner._quant_method.moe_kernel
    assert isinstance(kernel.fused_experts,HummingGroupedExperts)
    assert not kernel.can_overlap_shared_experts
    runner._dsk_full_finalize=True
