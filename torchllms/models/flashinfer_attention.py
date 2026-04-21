"""Flashinfer attention-with-sinks integration for gpt-oss.

Reuses the existing ``KVArena`` dense allocation ``[n_layers, max_bsz,
max_seqlen, n_kv_heads, head_dim]`` by exposing it as a "paged" cache to
flashinfer's ``BatchAttentionWithAttentionSinkWrapper``: each live rollout
sequence occupies one logical "page" of size ``max_seqlen``, so the KV
slab is trivially compatible with the NHD paged layout
``[total_pages=max_bsz, page_size=max_seqlen, n_kv_heads, head_dim]``.

No page allocator, no block-table management — just an ``arange(B)``
index pointing each sequence at its own slab. The payoff:

- flashinfer's sink-kernel handles per-head learned sink logits and
  alternating sliding-window attention natively (gpt-oss's exact
  attention variant)
- Batched decode with diverging per-row cache lengths works out of the
  box — no more ``assert len(start_q) == 1`` from our old triton kernel
- Sliding window + sinks combined in a single op

Memory/perf trade-off: this keeps the dense KV footprint
(max_bsz * max_seqlen per layer). True paging (reclaiming unused
portions of retired rollouts' slabs) is a later optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

try:
    import flashinfer
except ImportError:
    flashinfer = None

# One 256 MB scratch per (device, dtype, head_dim, sliding_window) config.
# gpt-oss-20b uses (bf16, 64, 0) and (bf16, 64, 128) → two scratches, 512 MB.
_DEFAULT_WORKSPACE_BYTES = 256 * 1024 * 1024

# Process-level cache of (workspace, wrapper) pairs. Keyed on the
# JIT-compilation-relevant parameters so each unique kernel variant gets
# compiled once.
_WRAPPER_CACHE: Dict[tuple, Tuple[torch.Tensor, "flashinfer.BatchAttentionWithAttentionSinkWrapper"]] = {}


def _wrapper_for(
    *,
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    sliding_window: int,   # 0 or negative → full attention
    workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
) -> "flashinfer.BatchAttentionWithAttentionSinkWrapper":
    """Return (cached) flashinfer sink wrapper for the given config.

    flashinfer JIT-compiles a separate kernel for each (dtype, head_dim,
    has_sliding_window) combination. The wrapper is *stateful*: each
    ``plan()`` call writes into its internal buffers, so one wrapper per
    (config, sliding_window) is enough — all layers with matching config
    share it and plan it once per forward pass.
    """
    if flashinfer is None:
        raise RuntimeError(
            "flashinfer is not installed; cannot use flashinfer_attention"
        )
    window_left = (sliding_window - 1) if sliding_window > 0 else -1
    key = (str(device), str(dtype), int(head_dim), int(sliding_window))
    if key not in _WRAPPER_CACHE:
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchAttentionWithAttentionSinkWrapper(
            workspace,
            kv_layout="NHD",
            q_data_type=dtype,
            kv_data_type=dtype,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            window_left=window_left,
        )
        _WRAPPER_CACHE[key] = (workspace, wrapper)
    return _WRAPPER_CACHE[key][1]


@dataclass
class SinkPlanKey:
    """Cache key for a planned batch layout — avoids re-planning across
    layers of the same forward pass when all layers share the same layout.
    """
    n_qo_tokens: int
    n_pages_total: int
    sm_scale: float  # flashinfer plan caches sm_scale; include for correctness

    def __hash__(self) -> int:
        return hash((self.n_qo_tokens, self.n_pages_total, self.sm_scale))


@dataclass
class FlashinferSinkContext:
    """Pre-planned flashinfer state for one Transformer.forward.

    Built once in ``GptOSSTransformer.forward`` and passed through to each
    attention layer. The two wrappers (full + sliding) are planned with
    the same batch layout; each layer picks the right wrapper based on
    its ``sliding_window`` config.

    When ``sliding_window == 0`` model-wide (no SWA at all), ``sliding``
    is None — callers should only access ``full``.
    """

    full: "flashinfer.BatchAttentionWithAttentionSinkWrapper"
    sliding: Optional["flashinfer.BatchAttentionWithAttentionSinkWrapper"]
    sliding_window: int
    sm_scale: float
    # Per-token batch_indices / positions for append_paged_kv_cache. Same
    # layout used by all layers in this forward, so compute once.
    batch_indices: torch.Tensor
    positions: torch.Tensor
    kv_indices: torch.Tensor
    kv_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor

    def wrapper_for_layer(
        self, layer_sliding_window: int,
    ) -> "flashinfer.BatchAttentionWithAttentionSinkWrapper":
        if layer_sliding_window > 0:
            if self.sliding is None:
                raise RuntimeError(
                    f"layer has sliding_window={layer_sliding_window} but "
                    f"context has no sliding wrapper planned"
                )
            return self.sliding
        return self.full


def build_sink_context(
    *,
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    n_heads: int,
    n_kv_heads: int,
    sliding_window: int,        # model-level SWA window; 0 = none
    post_write_seqlens: torch.Tensor,   # [B] int32 — KV seqlen after this forward's write
    new_query_len: int,         # S = number of new query tokens per sequence (uniform)
    max_seqlen: int,            # KVArena's max seqlen; used as page_size
    sm_scale: float,
) -> FlashinferSinkContext:
    """Build the trivial "one page per sequence" paged layout and plan both
    full-attention and (if enabled) sliding-window sink wrappers for it.

    Precondition: ``post_write_seqlens[i]`` is the valid-token count of row
    ``i``'s cache AFTER this forward's K/V write. For a fresh prefill of
    length S with pre-write seqlens ``P[i]``, this is ``P[i] + S``.
    """
    B = post_write_seqlens.shape[0]
    # "One page per sequence, page_size = max_seqlen" trick: each sequence's
    # dense slab IS its page. kv_indices = [0, 1, ..., B-1] — each active
    # row points at its own slot in the [max_bsz, max_seqlen, ...] slab.
    kv_indices = torch.arange(B, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(B + 1, dtype=torch.int32, device=device)
    # kv_last_page_len = valid tokens in the sole page = post-write seqlen
    kv_last_page_len = post_write_seqlens.to(torch.int32)
    # qo_indptr: each sequence contributes `new_query_len` query tokens
    qo_indptr = torch.arange(
        0, (B + 1) * new_query_len, new_query_len,
        dtype=torch.int32, device=device,
    )
    # Per-new-token batch_indices/positions for append_paged_kv_cache.
    # Row `b` contributes new_query_len new tokens at absolute positions
    # [post_write_seqlens[b] - new_query_len, post_write_seqlens[b]).
    pre_write = post_write_seqlens - new_query_len
    batch_indices = torch.arange(B, device=device, dtype=torch.int32).repeat_interleave(new_query_len)
    # positions[b * S + j] = pre_write[b] + j
    offset_within = torch.arange(new_query_len, device=device, dtype=torch.int32).repeat(B)
    positions = pre_write.repeat_interleave(new_query_len) + offset_within

    # Plan the full-attention wrapper.
    full_wrapper = _wrapper_for(
        device=device, dtype=dtype, head_dim=head_dim, sliding_window=0,
    )
    full_wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=kv_last_page_len,
        num_qo_heads=n_heads,
        num_kv_heads=n_kv_heads,
        head_dim_qk=head_dim,
        page_size=max_seqlen,
        causal=True,
        sm_scale=sm_scale,
        window_left=-1,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    sliding_wrapper = None
    if sliding_window > 0:
        sliding_wrapper = _wrapper_for(
            device=device, dtype=dtype, head_dim=head_dim,
            sliding_window=sliding_window,
        )
        sliding_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=kv_last_page_len,
            num_qo_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim_qk=head_dim,
            page_size=max_seqlen,
            causal=True,
            sm_scale=sm_scale,
            window_left=sliding_window - 1,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    return FlashinferSinkContext(
        full=full_wrapper,
        sliding=sliding_wrapper,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        batch_indices=batch_indices,
        positions=positions,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
        qo_indptr=qo_indptr,
    )


def run_sink_attention(
    *,
    ctx: FlashinferSinkContext,
    q: torch.Tensor,              # [B * S, n_heads, head_dim]
    k_cache_layer: torch.Tensor,  # [max_bsz, max_seqlen, n_kv_heads, head_dim]
    v_cache_layer: torch.Tensor,  # same
    sinks: torch.Tensor,          # [n_heads] bfloat16 or float
    layer_sliding_window: int,
) -> torch.Tensor:
    """Run the sink attention kernel for one transformer layer. Returns
    [B * S, n_heads, head_dim].

    Assumes ``append_paged_kv_cache`` has already written the new K/V into
    ``k_cache_layer``/``v_cache_layer`` BEFORE this call (the caller in
    gpt-oss's Attention.forward handles that via KVArena.update_kv).
    """
    wrapper = ctx.wrapper_for_layer(layer_sliding_window)
    # The AttentionSink JIT kernel expects `sink` as a float32 tensor and
    # `sm_scale` as a double — forward positionally via *args because the
    # wrapper's `run()` only threads *args through to the JIT module
    # (kwargs like sinks= go to the non-JIT base path).
    return wrapper.run(
        q,
        (k_cache_layer, v_cache_layer),
        sinks.float(),
        float(ctx.sm_scale),
    )
