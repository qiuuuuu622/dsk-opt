"""Compare grouped finalize + BF16 scale/shared boundaries against native kernels."""
import json,os,pathlib,sys
import torch
sys.path.insert(0,'/root/dsk_opt_goal_20261116/src')
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_unpermute as native
from vllm_dsv41_moe.ops import moe_unpermute as candidate
rank=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
torch.manual_seed(20261116+rank)
root=pathlib.Path('/root/dsk_opt_goal_20261116/results/humming-finalize')
root.mkdir(parents=True,exist_ok=True)
with (root/f'rank{rank}.jsonl').open('x') as log:
 for m in [1,2,7,8,9,32,64,127,128,129,256,512]:
  for k in [3,6]:
   for invalid in [False,True]:
    h=5120; capacity=m*k
    x=torch.randn(capacity,h,device='cuda',dtype=torch.bfloat16)
    w=torch.rand(m,k,device='cuda',dtype=torch.float32)
    ids=torch.randperm(capacity,device='cuda',dtype=torch.int64).reshape(m,k).to(torch.int32)
    valid=capacity if not invalid else capacity//2
    if invalid: ids[:,::2]=capacity
    offsets=torch.tensor([0,valid],device='cuda',dtype=torch.int64)
    shared=torch.randn(m*2,h,device='cuda',dtype=torch.bfloat16)[::2]
    a=torch.full((m,h),float('nan'),device='cuda',dtype=torch.bfloat16); b=torch.empty_like(a)
    for fusion in [False,True]:
     def ref():
      native(a,x,w,ids,offsets)
      if fusion: a.mul_(1.5); a.add_(shared)
     def opt():
      candidate(b,x,w,ids,offsets,shared=shared if fusion else None,routed_scaling_factor=1.5 if fusion else 1.)
     ref(); opt(); torch.cuda.synchronize()
     for mode in ['eager','graph']:
      if mode=='graph':
       stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
       with torch.cuda.stream(stream):
        for _ in range(3): ref(); opt()
       torch.cuda.current_stream().wait_stream(stream)
       graph=torch.cuda.CUDAGraph()
       with torch.cuda.graph(graph): ref(); opt()
       x.add_(0.125); graph.replay(); torch.cuda.synchronize()
      equal=torch.equal(a,b)
      record={'rank':rank,'m':m,'k':k,'invalid':invalid,'shared_fusion':fusion,'mode':mode,'equal':equal,'max_abs':float((a.float()-b.float()).abs().max()),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
      log.write(json.dumps(record)+'\n');log.flush()
      if not equal or not record['finite']: raise AssertionError(record)
 print(f'rank{rank}: 192 grouped finalize cases passed',flush=True)
