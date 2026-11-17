"""Read-only post-load model metadata audit; no forward or kernel patches."""
import dataclasses
import json
import os
from pathlib import Path
from vllm.v1.worker.gpu_worker import Worker as NativeWorker


def encode(value):
    if dataclasses.is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class Worker(NativeWorker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        output = os.environ.get('DSK_AUDIT_DIR')
        if not output:
            return
        models = [('target', self.model_runner.get_model())]
        drafter = getattr(self.model_runner, 'drafter', None)
        draft = getattr(drafter, 'model', None)
        if draft is not None and hasattr(draft, 'named_modules'):
            models.append(('draft', draft))
        rows = []
        for scope, model in models:
            for name, layer in model.named_modules():
                if hasattr(layer, 'humming_configs'):
                    rows.append({'scope': scope, 'name': name,
                                 'type': type(layer).__module__ + '.' + type(layer).__name__,
                                 'humming_configs': encode(layer.humming_configs),
                                 'input_schemas': encode(getattr(layer, 'input_schemas', None)),
                                 'weight_schemas': encode(getattr(layer, 'weight_schemas', None)),
                                 'moe_config': encode(getattr(layer, 'moe_config', None))})
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        (path / f'rank{self.rank}.json').write_text(json.dumps(rows, indent=2))
