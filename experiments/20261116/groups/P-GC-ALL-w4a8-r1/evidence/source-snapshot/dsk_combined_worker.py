"""Default-off combined Humming, indexer and TP8 head-row service worker."""
import os
from dsk_indexer_worker import Worker as BaseWorker

class Worker(BaseWorker):
    def load_model(self, *, load_dummy_weights=False):
        flag=os.environ.get('DSK_HEADROW','0')
        if flag not in ('0','1'): raise ValueError('DSK_HEADROW must be 0 or 1')
        super().load_model(load_dummy_weights=load_dummy_weights)
        if flag=='0': return
        from vllm.distributed import get_tp_group
        from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
        from dsk_headrow_adapter import HeadRowAdapter
        p=self.vllm_config.parallel_config
        assert p.tensor_parallel_size==8 and p.pipeline_parallel_size==1 and p.data_parallel_size==1
        assert p.prefill_context_parallel_size==1 and p.decode_context_parallel_size==1
        layers=[m for m in self.model_runner.get_model().modules() if isinstance(m,DeepseekV4FlashMLAAttention)]
        assert len(layers)==40,len(layers)
        s=self.vllm_config.speculative_config
        rows=self.vllm_config.scheduler_config.max_num_seqs*((s.num_speculative_tokens if s else 0)+1)
        self._dsk_headrow=HeadRowAdapter(layers,get_tp_group().device_group,rows,os.environ.get('DSK_HEADROW_EVIDENCE_DIR'))
        self._dsk_headrow.install()
