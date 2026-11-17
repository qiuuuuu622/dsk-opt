import json,os,pathlib,types
import torch
import torch.distributed as dist
from benchmarks.kernels.dsv41_indexer.common import check_vllm
from benchmarks.kernels.dsv41_indexer.benchmark import run_case
from vllm_dsv41_indexer.native import load_kernels
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
check_vllm();torch.backends.cuda.matmul.allow_tf32=False
native=load_kernels()
r=pathlib.Path('/root/dsk_opt_goal_20261116/results/indexer-scores');r.mkdir(parents=True,exist_ok=True)
with (r/f'rank{rank}.jsonl').open('x') as log:
 for rows,length,verify in [(1,32768,1),(6,65536,6),(12,262144,6),(32,65536,1)]:
  for stress in [False,True]:
   args=types.SimpleNamespace(stress=stress,tp_size=8,reverse_order=False,modes=[],repeats=0,inner=1)
   result=run_case(rows,length,verify,args,native)
   result.update(rank=rank,stress=stress,physical_world=dist.get_world_size(),measurement='correctness only; no performance timing; native N6 represented as per-query N1')
   log.write(json.dumps(result)+'\n');log.flush();dist.barrier()
   if rank==0: print(rows,length,verify,stress,'passed',flush=True)
dist.destroy_process_group()
