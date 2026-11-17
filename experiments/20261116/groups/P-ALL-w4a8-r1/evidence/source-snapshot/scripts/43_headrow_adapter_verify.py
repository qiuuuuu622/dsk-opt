"""Eight-rank correctness of serving adapter, including changing graph inputs."""
import json,os,pathlib,types,time
import torch
import torch.distributed as dist
from dsk_headrow_adapter import HeadRowAdapter
from benchmarks.kernels.head_exchange.baseline import packed_cache,load_flashmla,measure,check_reference
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
r=pathlib.Path('/root/dsk_opt_goal_20261116')
fm,_=load_flashmla(r/'evidence/flash_mla_interface.py')
sink=torch.full((64,),-torch.inf,device='cuda',dtype=torch.float32)
sink[:8]=torch.linspace(-2,3,8,device='cuda')+rank*.25
adapter=HeadRowAdapter([types.SimpleNamespace(attn_sink=sink)],dist.group.WORLD,384)
adapter.original=fm.flash_mla_with_kvcache
reports=[]
for rows in [6,12,64,129,384]:
 for ratio in [0,1,2]:
  torch.manual_seed(20261116+rows+ratio)
  cache,indices,sd=packed_cache(rows,128,128,64)
  extra,ei,ed=None,None,None
  if ratio:extra,ei,ed=packed_cache(rows,512,4096//ratio,64//ratio)
  lengths=torch.full((rows,),128,device='cuda',dtype=torch.int32)
  el=torch.full((rows,),512,device='cuda',dtype=torch.int32) if ratio else None
  torch.manual_seed(100+rank)
  q=torch.zeros(rows,1,64,512,device='cuda',dtype=torch.bfloat16)
  q[:,:,:8]=torch.randn(rows,1,8,512,device='cuda',dtype=torch.bfloat16)
  baseline=torch.empty_like(q);out=torch.empty_like(q)
  def kwargs(dest):
   return dict(q=q,k_cache=cache,block_table=None,cache_seqlens=None,
    head_dim_v=512,tile_scheduler_metadata=fm.get_mla_metadata()[0],
    is_fp8_kvcache=True,indices=indices,topk_length=lengths,
    softmax_scale=512**-.5,attn_sink=sink,extra_k_cache=extra,
    extra_indices_in_kvcache=ei,extra_topk_length=el,out=dest)
  def native():adapter.original(**kwargs(baseline))
  def optimized():adapter(**kwargs(out))
  native();optimized();torch.cuda.synchronize()
  torch.testing.assert_close(out[:,:,:8],baseline[:,:,:8],rtol=.02,atol=.002)
  for _ in range(3):optimized()
  torch.cuda.synchronize();dist.barrier()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):optimized()
  errors=[]; oracle_checks=[]
  for step in range(20):
   q[:,:,:8].add_(.015625)
   lengths.copy_((torch.arange(rows,device='cuda')*19+step*11).remainder(129).int())
   if el is not None:el.copy_((torch.arange(rows,device='cuda')*79+step*29).remainder(513).int())
   native();graph.replay();torch.cuda.synchronize()
   ref=baseline[:,:,:8].float();value=out[:,:,:8].float()
   need_oracle=torch.tensor([not torch.allclose(value,ref,rtol=.02,atol=.002)],device='cuda',dtype=torch.int32)
   dist.all_reduce(need_oracle,op=dist.ReduceOp.MAX)
   if need_oracle.item():
    # Different FlashMLA split/reduction shapes can differ slightly. Do not
    # relax the numerical tolerance: check each path against FP32 attention.
    bcheck=check_reference(q,sink,baseline,sd,ed,lengths,el)
    ocheck=check_reference(q,sink,out,sd,ed,lengths,el)
    captured=out.clone();optimized();torch.cuda.synchronize()
    torch.testing.assert_close(out[:,:,:8],captured[:,:,:8],rtol=0,atol=0)
    oracle_checks.append({'step':step,'native':bcheck,'optimized':ocheck,'graph_eager_exact':True})
   err=((value-ref).norm()/ref.norm().clamp_min(1e-12)).item();assert err<.01;errors.append(err)
  timings={}
  for name,fn in [('native',native),('adapter_full_chain',optimized)]:
   for _ in range(3):fn()
   timings[name]=measure(fn,5,3)
  reports.append({'rows':rows,'ratio':ratio,'changed_graph_steps':20,'max_nrmse':max(errors),'fp32_oracle_checks':oracle_checks,'timings':timings})
  print(json.dumps({'rank':rank,'rows':rows,'ratio':ratio,'passed':True,'oracle_checks':len(oracle_checks)}),flush=True)
  del graph
p=r/'results/headrow-adapter';p.mkdir(exist_ok=True)
(p/f'rank{rank}.json').write_text(json.dumps({'passed':True,'rank':rank,'cases':reports},indent=2))
print(json.dumps({'rank':rank,'passed_cases':len(reports),'max_nrmse':max(x['max_nrmse'] for x in reports)}),flush=True)
dist.destroy_process_group()
