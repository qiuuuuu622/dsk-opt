"""Opt-in Humming worker: flags default off; target-only A8 and tiled finalize."""
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
        if os.environ.get('DSK_HUMMING_FINALIZE', '0') not in ('0', '1'):
            raise ValueError('DSK_HUMMING_FINALIZE must be 0 or 1')
        if os.environ.get('DSK_HUMMING_FINALIZE', '0') == '1':
            from vllm_dsv41_moe.adapter import use_aligned_humming_finalize
            self._dsk_finalize_context = use_aligned_humming_finalize(max_tokens=128)
            self._dsk_finalize_context.__enter__()
        a8 = os.environ.get('DSK_HUMMING_A8', '0')
        if a8 not in ('0', '1'):
            raise ValueError('DSK_HUMMING_A8 must be 0 or 1')
        if a8 == '1':
            from vllm.model_executor.layers.quantization.utils import humming
            from vllm.model_executor.layers.quantization.utils.humming import moe
            from vllm.utils.humming import HummingInputSchema
            original = humming.convert_to_humming_moe_kernel_format
            original_module = moe.convert_to_humming_moe_kernel_format

            def target_a8(layer, *args, **kwargs):
                config = layer.moe_config
                if config.num_experts == 384 and config.hidden_dim == 5120:
                    kwargs['input_schema'] = HummingInputSchema.from_config(
                        {'dtype': 'float8e4m3', 'group_size': 0})
                    kwargs['allow_input_schema_fallback'] = False
                return original(layer, *args, **kwargs)

            humming.convert_to_humming_moe_kernel_format = target_a8
            moe.convert_to_humming_moe_kernel_format = target_a8
            try:
                super().load_model(load_dummy_weights=load_dummy_weights)
            finally:
                humming.convert_to_humming_moe_kernel_format = original
                moe.convert_to_humming_moe_kernel_format = original_module
        else:
            super().load_model(load_dummy_weights=load_dummy_weights)
        output = os.environ.get('DSK_AUDIT_DIR')
        if not output:
            return
        models = [('target', self.model_runner.get_model())]
        drafter = getattr(self.model_runner, 'speculator', None)
        if drafter is None:
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
