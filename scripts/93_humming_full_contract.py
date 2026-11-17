"""Exercise actual patched MoERunner boundaries using synthetic expert outputs.
Does not claim full model or GEMM correctness; compares native unpermute,
BF16 multiply/add and 8-rank reduction with fused path in eager/Graph.
"""
import os,pathlib,json,types,torch,torch.distributed as dist
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe as hm
before=hm.moe_unpermute
import dsk_humming_full_finalize as opt
assert hm.moe_unpermute is before,'Import must not patch'
os.environ['DSK_HUMMING_FULL_FINALIZE']='1';opt.install()
class Shared:
 def __init__(self):self._output=[None];self._output_idx=0;self.stream=torch.cuda.Stream();self.event=torch.cuda.Event();self.waits=0
 @property
 def output(self):
  out=self._output[0];assert out is not None;self._output[0]=None;return out
 def wait(self):self.waits+=1;self.event.wait(torch.cuda.current_stream())
 def start(self,x,async_):
  assert self._output[0] is None
  if async_:
   self.stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(self.stream):self._output[0]=x*0.375;self.event.record(self.stream)
  else:self._output[0]=x*0.375
class Runner:
 def __init__(self,m,scale,enabled):
  self._dsk_full_finalize=enabled;self.routed_scaling_factor=scale;self._shared_experts=Shared()
  self._quant_method=types.SimpleNamespace(topk_indices_dtype=torch.int64)
  self.routed_experts=types.SimpleNamespace(quant_method=types.SimpleNamespace(is_monolithic=False),forward_modular=self.experts)
  self.router=types.SimpleNamespace(select_experts=lambda **kw:(self.w,self.ids))
 def _maybe_apply_shared_experts(self,x,order):
  if not self.async_:self._shared_experts.start(x,False)
 def experts(self,**kw):
  out=torch.empty_like(kw['x'])
  hm.moe_unpermute(out=out,permuted_hidden_states=self.down,topk_weights=self.w,inv_permuted_idx=self.inv,expert_first_token_offset=self.offsets)
  return out
 def run(self,x,async_):
  self.async_=async_
  if async_:self._shared_experts.start(x,True)
  shared,out=MoERunner._apply_quant_method(self,x,None,x,shared_experts_overlapping=async_)
  shared,out=MoERunner._maybe_apply_routed_scale_to_output(self,shared,out)
  if shared is not None:out=shared+out
  dist.all_reduce(out)
  return out
outdir=pathlib.Path('/root/dsk_opt_goal_20261116/results/humming-full-contract');outdir.mkdir(parents=True,exist_ok=True);rows=[]
for m in [1,7,64,128,129]:
 for async_ in [False,True]:
  for scale in [1.,1.5]:
   torch.manual_seed(2111+rank+m)
   x=torch.randn(m,5120,device='cuda',dtype=torch.bfloat16);down=torch.randn(m*6,5120,device='cuda',dtype=torch.bfloat16)
   w=torch.rand(m,6,device='cuda');ids=torch.zeros(m,6,device='cuda',dtype=torch.int64);inv=torch.randperm(m*6,device='cuda').reshape(m,6).int();inv[:,::3]=m*6
   offsets=torch.tensor([0,m*6],device='cuda',dtype=torch.int64)
   a=Runner(m,scale,False);b=Runner(m,scale,True)
   for obj in [a,b]:obj.down=down;obj.w=w;obj.ids=ids;obj.inv=inv;obj.offsets=offsets
   for _ in range(3):expected=a.run(x,async_);actual=b.run(x,async_)
   torch.cuda.synchronize();assert torch.equal(expected,actual),(rank,m,async_,scale,'eager')
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):expected=a.run(x,async_);actual=b.run(x,async_)
   for step in range(10):
    x.add_(0.0078125);down.add_(0.015625);g.replay();torch.cuda.synchronize()
    assert torch.equal(expected,actual),(rank,m,async_,scale,step)
   assert a._shared_experts._output[0] is b._shared_experts._output[0] is None
   rows.append({'rows':m,'async_shared':async_,'scale':scale,'eager_equal':True,'graph_steps':10,'all_reduce_equal':True,'native_fallback':m>128})
   del g
(outdir/f'rank{rank}.json').write_text(json.dumps({'rank':rank,'passed':True,'cases':rows,'scope':__doc__},indent=2));print('PASS',rank,flush=True);dist.destroy_process_group()
