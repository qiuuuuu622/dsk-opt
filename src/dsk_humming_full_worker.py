"""Combined worker with separately opted-in full Humming finalizer."""
import os,json,pathlib
import torch.distributed as dist
from dsk_combined_worker import Worker as CombinedWorker
class Worker(CombinedWorker):
 def load_model(self,*,load_dummy_weights=False):
  super().load_model(load_dummy_weights=load_dummy_weights)
  flag=os.environ.get('DSK_HUMMING_FULL_FINALIZE','0');assert flag in ('0','1')
  if flag=='0':return
  assert os.environ.get('DSK_HUMMING_A8')=='1'
  assert os.environ.get('DSK_HUMMING_FINALIZE')=='1'
  assert os.environ.get('DSK_GC_MOE','0')=='0'
  from dsk_humming_full_finalize import install,enable_target
  from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
  install();records=[]
  for name,runner in self.model_runner.get_model().named_modules():
   if isinstance(runner,MoERunner) and runner.moe_config.num_experts==384 and runner.moe_config.hidden_dim==5120:
    enable_target(runner)
    records.append({'layer':name,'scale':runner.routed_scaling_factor,'full_finalize_max_rows':128,'before_collective':True,'preserve_bf16_rounding':True,'shared_stream':'await at finalizer; do not consume output early'})
  assert len(records)==40,('Expected40 target MoERunners',len(records))
  directory=pathlib.Path(os.environ['DSK_FULL_FINALIZE_EVIDENCE_DIR']);directory.mkdir(parents=True,exist_ok=True)
  (directory/f'rank{dist.get_rank()}.json').write_text(json.dumps({'rank':dist.get_rank(),'target_layers':records,'scope':'post-load eligibility; runtime branch evidence required'},indent=2))
