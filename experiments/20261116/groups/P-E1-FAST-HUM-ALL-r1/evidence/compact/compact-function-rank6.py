def sparse_attn_indexer(hidden_states: torch.Tensor, k_cache_prefix: LayerNameType, kv_cache: torch.Tensor, q_quant: torch.Tensor, q_scale: torch.Tensor | None, k: torch.Tensor | None, weights: torch.Tensor, quant_block_size: int, scale_fmt: str | None, topk_tokens: int, head_dim: int, max_model_len: int, total_seq_lens: int, topk_indices_buffer: torch.Tensor, skip_k_cache_insert: bool, use_pcp: bool, pcp_shard_decode_requests: bool, dense_mha_metadata_layer_name: LayerNameType, use_fp4_cache: bool=False, dcp_rank: int=0, dcp_world_size: int=1, cp_kv_cache_interleave_size: int=1, skip_topk_buffer_clear: bool=False, candidate_blocks: torch.Tensor | None=None, candidate_block_size: int=0, candidate_write: bool=False, topk_backend: str='auto') -> torch.Tensor:
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    if candidate_blocks is not None:
        assert dcp_world_size == 1, 'v4.1 two-level candidate filtering is not supported with DCP.'
        assert candidate_block_size > 0
    if not isinstance(attn_metadata, dict):
        values_spec, scales_spec = _gather_workspace_shapes(total_seq_lens, head_dim, fp8_dtype, use_fp4_cache)
        profile_specs: list[tuple[tuple[int, ...], torch.dtype]] = [values_spec, scales_spec, ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8)]
        if use_pcp and dcp_world_size > 1:
            gather_spec = _gather_workspace_shapes(total_seq_lens, head_dim, fp8_dtype, use_fp4_cache)
            profile_specs.extend(gather_spec * 2)
        current_workspace_manager().get_simultaneous(*profile_specs)
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(max_logits_elems, dtype=torch.uint8, device=hidden_states.device)
        _reserve_row_shard_exchange_transient(topk_indices_buffer, hidden_states.device)
        return sparse_attn_indexer_fake(hidden_states, k_cache_prefix, kv_cache, q_quant, q_scale, k, weights, quant_block_size, scale_fmt, topk_tokens, head_dim, max_model_len, total_seq_lens, topk_indices_buffer, skip_k_cache_insert, use_pcp, pcp_shard_decode_requests, dense_mha_metadata_layer_name, use_fp4_cache, candidate_blocks=candidate_blocks, candidate_block_size=candidate_block_size, candidate_write=candidate_write)
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
    if use_fp4_cache:
        assert q_scale is not None, 'use_fp4_cache=True requires q_scale'
    else:
        assert q_scale is None, 'q_scale must be None when use_fp4_cache=False'
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        if use_pcp and num_tokens > k.shape[0]:
            num_tokens //= get_pcp_group().world_size
        k = k[:num_tokens]
    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(k, slot_mapping, num_decode_tokens, use_pcp, pcp_shard_decode_requests=pcp_shard_decode_requests)
        assert scale_fmt is not None
        assert not use_fp4_cache, 'Unfused FP4 Insert is not supported yet'
        ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping_for_cache, quant_block_size, scale_fmt)
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            prefill_metadata = getattr(mla_metadata, 'prefill', None)
            if getattr(prefill_metadata, 'use_dense_mha', False) and getattr(mla_metadata, 'num_decode_tokens', -1) == 0 and (not torch.cuda.is_current_stream_capturing()):
                return topk_indices_buffer
    if not skip_topk_buffer_clear:
        topk_indices_buffer[:hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(total_seq_lens, head_dim, fp8_dtype, use_fp4_cache)
        pcp_chunks = [(c, c.pcp_deinterleave_idx) for c in prefill_metadata.chunks if c.pcp_deinterleave_idx is not None]
        gather_specs: list[tuple[tuple[int, int], torch.dtype]] = []
        if pcp_chunks:
            gathered_rows = max((dcp_world_size * c.local_total_seq_lens for c, _ in pcp_chunks))
            deinterleaved_rows = max((idx.shape[0] for _, idx in pcp_chunks))
            for rows in (gathered_rows, deinterleaved_rows):
                gather_specs.extend(_gather_workspace_shapes(rows, head_dim, fp8_dtype, use_fp4_cache))
        k_quant_full, k_scale_full, *gather_bufs = workspace_manager.get_simultaneous(values_spec, scales_spec, *gather_specs)
        row_shard = None if candidate_blocks is not None else PrefillRowShard.from_metadata(prefill_metadata, num_decode_tokens)
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[:chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[:chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(kv_cache, k_quant, k_scale, chunk.block_table, chunk.local_cu_seq_lens)
            row_start, row_end = (chunk.token_start, chunk.token_end)
            if row_shard is not None:
                narrowed = row_shard.narrow(chunk)
                if narrowed is None:
                    continue
                row_start, row_end, cu_seqlen_ks, cu_seqlen_ke = narrowed
            deinterleave_idx = chunk.pcp_deinterleave_idx
            if deinterleave_idx is not None:
                gathered_values, gathered_scales, out_values, out_scales = gather_bufs
                shard = k_quant[:chunk.local_total_seq_lens]
                k_quant = dcp_gather_kv_rows(shard, deinterleave_idx, gathered_values, out_values)
                k_scale = dcp_gather_kv_rows(k_scale[:chunk.local_total_seq_lens], deinterleave_idx, gathered_scales, out_scales)
            q_slice = q_quant[row_start:row_end]
            q_scale_slice = q_scale[row_start:row_end] if q_scale is not None else None
            topk_indices = topk_indices_buffer[row_start:row_end, :topk_tokens]
            if chunk.local_total_seq_lens == 0 or q_slice.shape[0] == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError('XPU fp8_mqa_logits does not support FP4 Q')
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(q_slice_cast, k_quant_cast, k_scale_cast, weights[row_start:row_end], cu_seqlen_ks, cu_seqlen_ke)
                else:
                    logits = fp8_fp4_mqa_logits((q_slice_cast, q_scale_slice), (k_quant_cast, k_scale_cast), weights[row_start:row_end], cu_seqlen_ks, cu_seqlen_ke, clean_logits=False)
                num_rows = logits.shape[0]
                if candidate_blocks is not None:
                    chunk_candidates = candidate_blocks[chunk.token_start:chunk.token_end]
                    if candidate_write:
                        _select_candidate_blocks(logits, cu_seqlen_ks, cu_seqlen_ke, chunk_candidates.shape[1], candidate_block_size, chunk_candidates)
                    else:
                        _apply_candidate_mask(logits, cu_seqlen_ks, cu_seqlen_ke, chunk_candidates, candidate_block_size)
                ops.top_k_per_row_prefill(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices, num_rows, logits.stride(0), logits.stride(1), topk_tokens)
            if deinterleave_idx is None:
                _merge_dcp_topk_global(logits, topk_indices, topk_tokens, dcp_rank, dcp_world_size, cp_kv_cache_interleave_size, row_starts=cu_seqlen_ks)
        if row_shard is not None:
            row_shard.exchange_topk(topk_indices_buffer, topk_tokens)
    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        needs_padded_path = decode_metadata.requires_padding or num_decode_tokens % decode_lens.shape[0] != 0
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = q_scale[:1].reshape(1, 1, *q_scale.shape[1:]) if q_scale is not None else None
        elif needs_padded_path:
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(q_quant[:num_decode_tokens], decode_lens, pad_value=0)
                padded_q_scale = pack_seq_triton(q_scale[:num_decode_tokens], decode_lens, pad_value=0)
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(q_quant[:num_decode_tokens], decode_lens)
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(decode_lens.shape[0], -1, *q_quant.shape[1:])
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(decode_lens.shape[0], -1, *q_scale.shape[1:])
            else:
                padded_q_scale = None
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        padded_q_quant_cast = padded_q_quant_decode_tokens.view(torch.int8) if use_fp4_cache else padded_q_quant_decode_tokens
        _dsk_out = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        _dsk_candidates = candidate_blocks[:num_padded_tokens] if candidate_blocks is not None else None
        _dsk_eligible = not use_fp4_cache and topk_tokens == 512 and _dsk_decode_workspace.supported(padded_q_quant_cast, padded_q_scale, kv_cache, weights[:num_padded_tokens], seq_lens, decode_metadata.block_table, _dsk_candidates, candidate_block_size, next_n, needs_padded_path, decode_metadata, dcp_world_size, use_pcp, _dsk_out)
        if candidate_write:
            _dsk_decode_workspace.published_rows = None
        _dsk_done = not candidate_write and _dsk_decode_workspace.consume(_dsk_eligible, padded_q_quant_cast, kv_cache, weights[:num_padded_tokens], decode_metadata.block_table, _dsk_out, _dsk_candidates)
        if not _dsk_done:
            if current_platform.is_xpu():
                if padded_q_scale is not None:
                    raise RuntimeError('XPU fp8_paged_mqa_logits does not support FP4 Q')
                seq_lens_xpu = seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
                logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(padded_q_quant_cast, kv_cache, weights[:num_padded_tokens], seq_lens_xpu, decode_metadata.block_table, decode_metadata.schedule_metadata, max_model_len)
            else:
                logits = fp8_fp4_paged_mqa_logits((padded_q_quant_cast, padded_q_scale), kv_cache, weights[:num_padded_tokens], seq_lens, decode_metadata.block_table, decode_metadata.schedule_metadata, max_model_len=max_model_len, clean_logits=False, indices=decode_metadata.indices)
            num_rows = logits.shape[0]
            if candidate_blocks is not None:
                vis = seq_lens.reshape(-1)
                row_repeat = next_n if vis.numel() != num_rows else 1
                vis = vis[:num_rows]
                decode_candidates = candidate_blocks[:num_rows]
                if candidate_write:
                    _select_candidate_blocks(logits, None, vis, decode_candidates.shape[1], candidate_block_size, decode_candidates, row_repeat)
                    _dsk_decode_workspace.publish(_dsk_eligible, decode_candidates, seq_lens)
                else:
                    _apply_candidate_mask(logits, None, vis, decode_candidates, candidate_block_size, row_repeat)
            topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
            get_indexer_topk(topk_backend)(logits, seq_lens, next_n, topk_indices, topk_tokens, attn_metadata_narrowed.max_seq_len)
        else:
            topk_indices = _dsk_out
        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(logits, topk_indices, topk_tokens, dcp_rank, dcp_world_size, cp_kv_cache_interleave_size)
        if needs_padded_path:
            topk_indices = unpack_seq_triton(topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]), decode_lens)
            topk_indices_buffer[:topk_indices.shape[0], :topk_indices.shape[-1]] = topk_indices
    return topk_indices_buffer
