"""Verify integration workspace binding/reuse against validated compact operator.
Not an independent arithmetic reference; native score accuracy is tested in 25.
"""
import json,os,pathlib,types
import torch
import torch.distributed as dist
from dsk_indexer_adapter import DecodeWorkspace
from vllm_dsv41_indexer import IndexerParallel
from vllm_dsv41_indexer.workspace import CandidateState,CompactIndexer
from benchmarks.kernels.dsv41_indexer.inputs import make_inputs
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
parallel=IndexerParallel(8,rank);pool=DecodeWorkspace(384,parallel)
root=pathlib.Path('/root/dsk_opt_goal_20261116/results/indexer-pool-spec6');root.mkdir(parents=True,exist_ok=True)
with (root/f'rank{rank}.jsonl').open('x') as log:
 for n in [1,6,12,32,64,127,128,129,384]:
  verify=6 if n%6==0 else 1
  q,_,weights,cache,table,req,lens,blocks=make_inputs(dict(rows=n,compressed_length=32768,rows_per_request=verify,page_size=64,candidate_blocks=2048))
  blocks.copy_(torch.arange(2048,device='cuda',dtype=torch.int32).expand(n,-1));blocks[:,-1]=4095
  flat_table=table[req.long()].contiguous();ql=[q]+[q.float().roll(i*7,-1).to(q.dtype) for i in range(1,4)]
  wl=[weights]+[weights.roll(i*3,-1).contiguous() for i in range(1,4)]
  state=CandidateState(n,q.device,parallel=parallel)
  refs=[CompactIndexer(state,ql[i],wl[i],cache,table,req) for i in range(4)]
  outputs=[torch.empty((n,512),dtype=torch.int32,device='cuda') for _ in range(4)]
  metadata=types.SimpleNamespace(global_seq_lens=None,indices=None)
  assert pool.supported(q.view(n//verify,verify,32,128),None,cache,weights,lens.view(n//verify,verify),table,blocks,8,verify,False,metadata,1,False,outputs[0])
  qa=[x.clone() for x in ql];qb=[(-x.float()).to(x.dtype) for x in ql]
  ba=blocks.clone();bb=(blocks+3)%4096;la=lens.clone();lb=lens.clone();lb[0]=0
  if n>1:lb[1]=517
  expected=[]
  for qs,bs,ls in [(qa,ba,la),(qb,bb,lb)]:
   for dst,src in zip(ql,qs):dst.copy_(src)
   blocks.copy_(bs);lens.copy_(ls);state.publish(blocks,lens,source_unique=True)
   expected.append([torch.sort(ref.forward(),dim=-1).values.clone() for ref in refs])
  def work():
   pool.publish(True,blocks,lens)
   for i in range(4):
    assert pool.consume(True,ql[i].view(n//verify,verify,32,128),cache,wl[i],table,outputs[i],blocks)
  work();torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):work()
  for step in range(20):
   side=step%2;qs,bs,ls=[(qa,ba,la),(qb,bb,lb)][side]
   for dst,src in zip(ql,qs):dst.copy_(src)
   blocks.copy_(bs);lens.copy_(ls);graph.replay()
   for i in range(4):assert torch.equal(torch.sort(outputs[i],dim=-1).values,expected[side][i]),(rank,n,step,i)
  other=blocks.clone();assert not pool.consume(True,q.view(n//verify,verify,32,128),cache,weights,table,outputs[0],other)
  pool.publish(False,blocks,lens);assert not pool.consume(True,q.view(n//verify,verify,32,128),cache,weights,table,outputs[0],blocks)
  log.write(json.dumps({'rank':rank,'rows':n,'consumer_layers':4,'changing_graph_steps':20,'exact_topk_sets':True,'foreign_candidate_buffer_rejected':True,'unpublished_rejected':True})+'\n');log.flush()
  del graph,refs,state,outputs,cache,table,flat_table
 dist.barrier()
dist.destroy_process_group()
