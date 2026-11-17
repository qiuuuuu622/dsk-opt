"""Opt-in TP8 head-row decode adapter; collective setup precedes graph capture.

The original per-step planner owns row-planner lifetime. Metadata is sliced
again each invocation. All ranks must see identical query rows and logical KV.
"""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from vllm_dsv41_head_exchange.workspace import FusedWorkspace


class HeadRowAdapter:
    def __init__(self, layers, group, max_rows=384, evidence=None):
        assert dist.get_world_size(group) == 8
        assert not torch.cuda.is_current_stream_capturing()
        self.rank = dist.get_rank(group)
        self.sinks = {}
        for layer in layers:
            sink = layer.attn_sink
            assert sink.dtype == torch.float32 and sink.shape == (64,)
            full = torch.empty_like(sink)
            dist.all_gather_into_tensor(full, sink[:8].contiguous(), group=group)
            self.sinks[sink.data_ptr()] = full
        assert self.sinks, 'No target FlashMLA layers found'
        self.max_rows = max_rows
        # Shared dummy input avoids retaining a full Q allocation per shape.
        dummy = torch.zeros(((max_rows + 7)//8)*8, 1, 64, 512,
                            device=layers[0].attn_sink.device, dtype=torch.bfloat16)
        self.workspaces = {chunk: FusedWorkspace(dummy[:chunk*8], layers[0].attn_sink,
                            group=group) for chunk in range(1, (max_rows+7)//8+1)}
        self.original = None
        if evidence:
            path = Path(evidence); path.mkdir(parents=True, exist_ok=True)
            unique = {}
            for ws in self.workspaces.values():
                for tensor in (ws.input,ws.q_plane.buffer,ws.o_plane.buffer,ws.row_o):
                    storage = tensor.untyped_storage()
                    unique[storage.data_ptr()] = storage.nbytes()
            (path/f'headrow-rank{self.rank}.json').write_text(json.dumps({
                'enabled':True,'layers':len(layers),'max_rows':max_rows,
                'workspace_chunks':list(self.workspaces),
                'persistent_storage_bytes':sum(unique.values()),
                'scope':'target decode; native prefill/draft/unsupported fallback',
                'kernel_hit_rate':'requires runtime trace; setup is not proof of execution'
            },indent=2))

    def shard(self, tensor, rows, chunk, invalid=0):
        if tensor is None: return None
        assert tensor.shape[0] == rows
        start = min(self.rank * chunk, rows)
        stop = min(start + chunk, rows)
        if stop-start == chunk: return tensor[start:stop]
        result = tensor.new_full((chunk, *tensor.shape[1:]), invalid)
        if stop > start: result[:stop-start].copy_(tensor[start:stop])
        return result

    def __call__(self, *args, **kw):
        if args: return self.original(*args, **kw)
        q=kw.get('q'); sink=kw.get('attn_sink'); out=kw.get('out')
        rows=q.shape[0] if q is not None else 0
        if (not 1 <= rows <= self.max_rows or q.shape[1:] != (1,64,512)
            or q.dtype != torch.bfloat16 or not q.is_contiguous()
            or sink is None or sink.data_ptr() not in self.sinks
            or out is None or out.shape != q.shape
            or not kw.get('is_fp8_kvcache') or kw.get('block_table') is not None
            or kw.get('cache_seqlens') is not None):
            return self.original(**kw)
        chunk=(rows+7)//8; ws=self.workspaces[chunk]
        # Source pointer may vary across layer and graph; transport buffers persist.
        ws._call(q, ws.q_plane, 'q', rows)
        row_args=dict(kw)
        row_args.update(q=ws.row_q, out=ws.row_o, attn_sink=self.sinks[sink.data_ptr()])
        for key,invalid in [('indices',-1),('topk_length',0),
                            ('extra_indices_in_kvcache',-1),('extra_topk_length',0)]:
            row_args[key]=self.shard(kw.get(key),rows,chunk,invalid)
        native_meta=kw['tile_scheduler_metadata']
        # Native metadata is recreated per step. Do not cache a planner on workspace.
        plans=getattr(native_meta,'_dsk_headrow_plans',None)
        if plans is None:
            plans={};native_meta._dsk_headrow_plans=plans
        if chunk not in plans: plans[chunk]=type(native_meta)()
        row_args['tile_scheduler_metadata']=plans[chunk]
        self.original(**row_args)
        ws.backward_o()
        out.copy_(ws.output[:rows])
        return out,None  # Native decode caller discards LSE.

    def install(self):
        from vllm.models.deepseek_v41.nvidia import flashmla
        assert self.original is None
        self.original=flashmla.flash_mla_with_kvcache
        flashmla.flash_mla_with_kvcache=self
