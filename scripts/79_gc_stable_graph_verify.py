"""Eight-rank adapter checks against the direct native GC API, not E2E."""
import os,json,pathlib,torch,torch.distributed as dist
import gc_moe
from dsk_gc_adapter import GCExecution,finalize
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl');group=dist.group.WORLD
outdir=pathlib.Path('/root/dsk_opt_goal_20261116/results/gc-service-adapter-stable-graph');outdir.mkdir(exist_ok=True,parents=True)
torch.manual_seed(811+rank)
def make_weights(seed):
 torch.manual_seed(seed+rank)
 raw=[]
 for n,k in [(4608,5120),(5120,2304)]:
  packed=torch.randint(0,256,(48,n,k//2),device='cuda',dtype=torch.uint8)
  scales=torch.randint(117,123,(48,n,k//32),device='cuda',dtype=torch.uint8)
  raw.append((packed,scales))
 return gc_moe.transform_weights_for_gc_moe_sm90(*raw)
a=GCExecution(make_weights(100),group);b=GCExecution(make_weights(200),group)
assert a.buffer is b.buffer and a.output is b.output
rows=[1,7,8,9,63,64,65,129,384,4096];results=[]
def inputs(m,seed):
 torch.manual_seed(seed)
 x=torch.randn((m,5120),device='cuda',dtype=torch.bfloat16)
 ids=torch.randint(0,384,(m,6),device='cuda',dtype=torch.int64)
 weights=torch.softmax(torch.randn((m,6),device='cuda'),dim=1)
 return x,ids,weights

def reference(execution,x,ids,w):
 m=x.shape[0];c=(m+7)//8;start=rank*c;valid=max(0,min(c,m-start));buf=execution.buffer
 local=torch.zeros((c,5120),device='cuda',dtype=torch.bfloat16)
 li=torch.zeros((c,6),device='cuda',dtype=torch.int64);lw=torch.zeros((c,6),device='cuda')
 if valid:local[:valid].copy_(x[start:start+valid]);li[:valid].copy_(ids[start:start+valid]);lw[:valid].copy_(w[start:start+valid])
 f=local.float().view(c,40,128);sf=f.abs().amax(-1).clamp_min(1e-4)/448.
 q=(f/sf.unsqueeze(-1)).to(torch.float8_e4m3fn).view(c,5120)
 buf.x[:c].copy_(q);buf.x_sf[:c].copy_(sf);buf.topk_idx[:c].copy_(li);buf.topk_weights[:c].copy_(lw)
 y=torch.empty_like(local)
 gc_moe.sm90_fp8_fp4_gc_moe(y,*execution.weights,buf,recipe=(1,1,32),activation='swiglu',activation_clamp=10.,fast_math=False,num_tokens_bound=c)
 output=torch.empty((c*8,5120),device='cuda',dtype=torch.bfloat16);dist.all_gather_into_tensor(output,y,group=group)
 return output[:m]

for m in rows:
 x,i,w=inputs(m,400+m)
 for label,e in [('a',a),('b',b)]:
  expected=reference(e,x,i,w);actual=e.run(x,i,w)
  torch.cuda.synchronize()
  max_abs=(actual.float()-expected.float()).abs().max().item()
  assert torch.equal(actual,expected),(rank,m,label,max_abs)
  shared=torch.randn_like(actual);fused=finalize(actual,shared,1.5)
  expected_final=(actual*1.5)+shared
  assert torch.equal(fused,expected_final),(rank,m,label,'finalize')
  results.append({'rows':m,'weights':label,'native_gc_bitwise_equal':True,'scale_shared_bitwise_equal':True})
 print('rank',rank,'rows',m,'passed',flush=True)
# Two differently-weighted layers reuse one workspace in the same graph.
x,ids,w=inputs(65,900)
for _ in range(3):a.run(x,ids,w);b.run(x,ids,w)
torch.cuda.synchronize();dist.barrier()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):ya=a.run(x,ids,w);yb=b.run(x,ids,w)
for step in range(20):
 nx,ni,nw=inputs(65,1000+step);x.copy_(nx);ids.copy_(ni);w.copy_(nw)
 expected_a=reference(a,x,ids,w);expected_b=reference(b,x,ids,w)
 g.replay();torch.cuda.synchronize()
 assert torch.equal(ya,expected_a) and torch.equal(yb,expected_b),(rank,step,'graph')
report={'rank':rank,'passed':True,'cases':results,'graph_steps':20,'graph_weight_sets':2,'scope':'replicated-row sharding, native GC equivalence, fused scale/shared and multi-layer shared workspace Graph'}
(outdir/f'rank{rank}.json').write_text(json.dumps(report,indent=2));print('PASS',rank,flush=True)
dist.barrier();del g;torch.cuda.synchronize();dist.destroy_process_group()
