"""Opt-in integration worker; compact decoder remains disabled by default."""
import json
import os
from pathlib import Path
from dsk_humming_worker import Worker as BaseWorker

class Worker(BaseWorker):
    def load_model(self, *, load_dummy_weights=False):
        enabled=os.environ.get('DSK_COMPACT_INDEXER','0')
        if enabled not in ('0','1'):raise ValueError('DSK_COMPACT_INDEXER must be 0 or 1')
        if enabled=='1':
            from vllm.distributed import get_tensor_model_parallel_rank
            from vllm_dsv41_indexer.parallel import IndexerParallel
            from dsk_indexer_adapter import install
            config=self.vllm_config
            parallel=config.parallel_config
            if parallel.pipeline_parallel_size!=1 or parallel.data_parallel_size!=1:
                raise RuntimeError('Initial compact service integration requires PP1/DP1')
            spec=config.speculative_config
            decode_width=(spec.num_speculative_tokens if spec is not None else 0)+1
            max_rows=config.scheduler_config.max_num_seqs*decode_width
            contract=IndexerParallel(tp_size=parallel.tensor_parallel_size,
                tp_rank=get_tensor_model_parallel_rank(),
                pcp_size=parallel.prefill_context_parallel_size,
                dcp_size=parallel.decode_context_parallel_size)
            evidence=os.environ.get('DSK_INDEXER_EVIDENCE_DIR')
            self._dsk_indexer_workspace=install(max_rows,contract,evidence)
            if evidence:
                tensors=[self._dsk_indexer_workspace.scores,self._dsk_indexer_workspace.requests,
                         self._dsk_indexer_workspace.state.blocks,self._dsk_indexer_workspace.state.counts,
                         self._dsk_indexer_workspace.state.plan]
                report={'enabled':True,'rank':contract.tp_rank,'max_rows':max_rows,
                    'persistent_bytes':sum(t.numel()*t.element_size() for t in tensors),
                    'supported':'FP8 decode/verify flattened N1, 32x128 index heads, page64, candidate block8, top512, PCP1 DCP1; native fallback otherwise',
                    'runtime_kernel_hits':'not established by installation; requires trace'}
                p=Path(evidence);p.mkdir(parents=True,exist_ok=True)
                (p/f'workspace-rank{contract.tp_rank}.json').write_text(json.dumps(report,indent=2))
        super().load_model(load_dummy_weights=load_dummy_weights)
