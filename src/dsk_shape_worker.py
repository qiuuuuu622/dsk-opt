"""Diagnostic-only V2 dispatch histogram; never use for formal latency runs."""
import collections
import dataclasses
import inspect
import json
import os
from pathlib import Path
from dsk_humming_worker import Worker as BaseWorker

class Worker(BaseWorker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        destination = os.environ.get('DSK_SHAPE_DIR')
        if not destination:
            raise RuntimeError('Diagnostic worker requires DSK_SHAPE_DIR')
        from vllm.v1.worker.gpu import model_runner
        if not isinstance(self.model_runner, model_runner.GPUModelRunner):
            raise RuntimeError('Expected V2 GPUModelRunner')
        original_dispatch = model_runner.dispatch_cg_and_sync_dp
        dispatch_sig = inspect.signature(original_dispatch)
        original_execute = self.model_runner.execute_model
        execute_sig = inspect.signature(original_execute)
        active = {'value': False}
        counts = collections.Counter()
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        out = path / f'rank{self.rank}.json'
        def save():
            rows = [dict(json.loads(key), steps=count) for key, count in counts.items()]
            temp = out.with_suffix('.tmp')
            temp.write_text(json.dumps({'scope':'target V2 dispatch, not per-layer kernel hit counts; diagnostic overhead excluded from performance','steps':sum(counts.values()),'shapes':rows},indent=2))
            temp.replace(out)
        def dispatch(*args, **kwargs):
            result = original_dispatch(*args, **kwargs)
            if active['value']:
                bound = dispatch_sig.bind(*args, **kwargs)
                bound.apply_defaults()
                keys = ['num_reqs','num_tokens','uniform_token_count','max_query_len','uniform_decode']
                row = {key:bound.arguments.get(key) for key in keys}
                desc = result[0]
                row['descriptor'] = {field.name:str(getattr(desc,field.name)) for field in dataclasses.fields(desc)}
                counts[json.dumps(row,sort_keys=True)] += 1
                if sum(counts.values()) % 100 == 0: save()
            return result
        def execute(*args, **kwargs):
            bound = execute_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            active['value'] = not bound.arguments.get('dummy_run',False)
            try:
                return original_execute(*args, **kwargs)
            finally:
                active['value'] = False
        model_runner.dispatch_cg_and_sync_dp = dispatch
        self.model_runner.execute_model = execute
        self._dsk_shape_save = save
        save()
