"""Experimental opt-in V4.1 FP8 compact decoder. Not yet service-validated.

Patches the registered Python custom-op function in place, preserving its
registration and ABI. All-off callers must never call install(). Prefill stays
native. One serial workspace is republished by the native candidate source
before consumers, including every captured CUDA Graph replay.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import torch

class DecodeWorkspace:
    def __init__(self, max_rows, parallel):
        from vllm_dsv41_indexer.workspace import CandidateState
        self.state = CandidateState(max_rows, torch.device('cuda', torch.cuda.current_device()), parallel=parallel)
        self.max_rows=max_rows
        self.scores=torch.empty((max_rows,16384),dtype=torch.float32,device='cuda')
        self.requests=torch.arange(max_rows,dtype=torch.int32,device='cuda')
        self.published_rows=None
        self.published_candidate_ptr=None
        self.trace_counts={'publish':0,'compact':0,'fallback':0}

    def supported(self, q, scale, cache, weights, lengths, table, candidates,
                  block_size, next_n, needs_padding, metadata, dcp, use_pcp, out):
        n=q.shape[0]
        return (0<n<=self.max_rows and next_n==1 and not needs_padding
            and scale is None and dcp==1 and not use_pcp
            and metadata.global_seq_lens is None and metadata.indices is None
            and candidates is not None and block_size==8
            and q.shape==(n,1,32,128) and q.dtype==torch.float8_e4m3fn
            and weights.shape==(n,32) and weights.dtype==torch.float32
            and lengths.shape==(n,1) and lengths.dtype==torch.int32
            and candidates.shape==(n,2048) and candidates.dtype==torch.int32
            and cache.ndim==4 and cache.shape[1:]==(64,1,132) and cache.dtype==torch.uint8
            and table.ndim==2 and table.shape[0]==n and table.dtype==torch.int32
            and out.shape==(n,512) and out.dtype==torch.int32
            and all(x.is_contiguous() for x in [q,cache,weights,lengths,table,candidates,out]))

    def publish(self, eligible, candidates, lengths):
        self.published_rows=None
        self.published_candidate_ptr=None
        if eligible:
            n=lengths.numel()
            self.state.native.publish(candidates,lengths.reshape(-1),self.state.blocks[:n],self.state.counts[:n])
            self.published_rows=n
            self.published_candidate_ptr=candidates.data_ptr()
            self.trace_counts['publish']+=1

    def consume(self, eligible, q, cache, weights, table, output, candidates):
        n=q.shape[0]
        if not eligible or self.published_rows!=n or candidates.data_ptr()!=self.published_candidate_ptr:
            self.trace_counts['fallback']+=1
            return False
        from vllm_dsv41_indexer.score import candidate_logits
        candidate_logits(q.reshape(n,32,128),weights,table,self.requests[:n],
            self.state.blocks[:n],self.state.counts[:n],cache,64,self.scores[:n],
            cache_page_size=64,candidate_block_size=8,fp32_scores=True)
        self.state.native.topk(self.scores[:n],self.state.counts[:n],self.state.blocks[:n],output,8,self.state.plan[:n+1],None)
        self.trace_counts['compact']+=1
        return True


def install(max_rows, parallel, evidence_dir=None):
    from vllm.model_executor.layers import sparse_attn_indexer as native
    fn=inspect.unwrap(native.sparse_attn_indexer)
    if getattr(fn,'_dsk_compact_installed',False):raise RuntimeError('Already installed')
    source=inspect.getsource(fn)
    # Freeze the exact function version, not merely package metadata.
    expected='958c2c91e5c425080916043436525285292939a2dfeb819de218702fa91bd957'
    digest=hashlib.sha256(source.encode()).hexdigest()
    if digest!=expected:raise RuntimeError(f'Unexpected sparse indexer source: {digest}')
    tree=ast.parse(source);definition=tree.body[0]
    definition.decorator_list=[]
    decode=next(n for n in definition.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='has_decode')
    begin=next(i for i,n in enumerate(decode.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='current_platform.is_xpu()')
    end=next(i for i,n in enumerate(decode.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='decode_metadata.global_seq_lens is not None')
    original=decode.body[begin:end]
    candidate=next(n for n in original if isinstance(n,ast.If) and ast.unparse(n.test)=='candidate_blocks is not None')
    writer=next(n for n in candidate.body if isinstance(n,ast.If) and ast.unparse(n.test)=='candidate_write')
    writer.body+=ast.parse('_dsk_decode_workspace.publish(_dsk_eligible, decode_candidates, seq_lens)').body
    prelude=ast.parse('''
_dsk_out = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
_dsk_candidates = candidate_blocks[:num_padded_tokens] if candidate_blocks is not None else None
_dsk_eligible = (not use_fp4_cache and topk_tokens == 512 and
    _dsk_decode_workspace.supported(padded_q_quant_cast, padded_q_scale, kv_cache,
        weights[:num_padded_tokens], seq_lens, decode_metadata.block_table,
        _dsk_candidates, candidate_block_size, next_n, needs_padded_path,
        decode_metadata, dcp_world_size, use_pcp, _dsk_out))
if candidate_write:
    _dsk_decode_workspace.published_rows = None
_dsk_done = (not candidate_write and _dsk_decode_workspace.consume(
    _dsk_eligible, padded_q_quant_cast, kv_cache, weights[:num_padded_tokens],
    decode_metadata.block_table, _dsk_out, _dsk_candidates))
''').body
    # If compact ran, no dense logits were allocated. DCP is excluded above.
    branch=ast.If(test=ast.UnaryOp(op=ast.Not(),operand=ast.Name(id='_dsk_done',ctx=ast.Load())),body=original,orelse=ast.parse('topk_indices = _dsk_out').body)
    decode.body[begin:end]=prelude+[branch]
    ast.fix_missing_locations(tree)
    patched=ast.unparse(tree)+'\n'
    scope=dict(fn.__globals__)
    workspace=DecodeWorkspace(max_rows,parallel)
    scope['_dsk_decode_workspace']=workspace
    exec(compile(patched,'<dsk_compact_indexer>','exec'),scope)
    fn.__globals__['_dsk_decode_workspace']=workspace
    fn.__code__=scope[fn.__name__].__code__
    fn._dsk_compact_installed=True
    if evidence_dir:
        p=Path(evidence_dir);p.mkdir(parents=True,exist_ok=True)
        (p/f'compact-function-rank{parallel.tp_rank}.py').write_text(patched)
    return workspace
