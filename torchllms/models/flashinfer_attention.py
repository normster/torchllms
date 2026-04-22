"""Flashinfer attention-with-sinks integration for gpt-oss.

Plans ``flashinfer.BatchAttentionWithAttentionSinkWrapper`` against the
shared :class:`~torchllms.models.paged_kv.PagedKVPool`. gpt-oss uses the
sink kernel because it has per-head learned sink logits plus alternating
sliding-window attention on even-indexed layers — Qwen3's prefill/decode
wrappers don't cover that, so gpt-oss keeps its own context type here.

Parallel in shape to :mod:`torchllms.models.paged_attention` — both build
a ``*Context`` once at the top of a model forward and thread it through
the per-layer attention calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

try:
    import flashinfer
except ImportError:  # pragma: no cover
    flashinfer = None  # type: ignore[assignment]

from torchllms.models.paged_kv import PagedBatchLayout, PagedKVPool, RolloutId


# One 256 MB scratch per (device, dtype, head_dim, sliding_window) config.
# gpt-oss-20b uses (bf16, 64, 0) and (bf16, 64, 128) → two scratches, 512 MB.
_DEFAULT_WORKSPACE_BYTES = 256 * 1024 * 1024

# Process-level cache of (workspace, wrapper) pairs. Keyed on the
# JIT-compilation-relevant parameters so each unique kernel variant gets
# compiled once. Two caches:
#   - ``_WRAPPER_CACHE``       : eager-mode wrappers (prefill + variable-
#                                length paths). One per (config, sliding).
#   - ``_DECODE_WRAPPER_CACHE``: cudagraph-mode wrappers with pool-bound
#                                buffers (decode only). One per
#                                (config, sliding, batch_size) — fixed
#                                batch size is part of the cudagraph
#                                contract (see flashinfer source:
#                                ``_fixed_batch_size = len(qo_indptr_buf) - 1``).
_WRAPPER_CACHE: Dict[tuple, Tuple[torch.Tensor, "flashinfer.BatchAttentionWithAttentionSinkWrapper"]] = {}
_DECODE_WRAPPER_CACHE: Dict[tuple, Tuple[torch.Tensor, "flashinfer.BatchAttentionWithAttentionSinkWrapper"]] = {}


def _require_flashinfer() -> None:
    if flashinfer is None:
        raise RuntimeError(
            "flashinfer is not installed; cannot use flashinfer_attention"
        )


def _prefill_wrapper_for(
    *,
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    sliding_window: int,   # 0 or negative → full attention
    workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
) -> "flashinfer.BatchAttentionWithAttentionSinkWrapper":
    """Return (cached) eager-mode sink wrapper for prefill / variable-
    length calls.

    Not usable under cudagraph — for decode use
    :func:`_decode_wrapper_for` instead, which binds pool buffers and
    pins batch size.
    """
    _require_flashinfer()
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


def _decode_wrapper_for(
    *,
    pool: PagedKVPool,
    dtype: torch.dtype,
    head_dim: int,
    sliding_window: int,
    batch_size: int,
    workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
) -> "flashinfer.BatchAttentionWithAttentionSinkWrapper":
    """Return (cached) cudagraph-mode sink wrapper for decode.

    Constructed with ``use_cuda_graph=True`` and pool-bound
    ``qo_indptr_buf`` / ``paged_kv_indptr_buf`` / ``paged_kv_indices_buf``
    / ``paged_kv_last_page_len_buf`` — decode ``qo_indptr`` is always
    ``arange(B+1)`` (each row contributes 1 new token), so the pool's
    ``_qo_indptr_buf[:B+1]`` is a stable address that reconstructs
    the correct content on every plan() call.

    One wrapper per ``(config, sliding_window, batch_size)``. For a
    typical gpt-oss workload with B=1 decode (single-row generate_single)
    and optionally B=max_bsz (generate_multiple), expect 2-4 wrappers
    cached across full + sliding.
    """
    _require_flashinfer()
    device = pool.device
    window_left = (sliding_window - 1) if sliding_window > 0 else -1
    key = (
        str(device), str(dtype), int(head_dim),
        int(sliding_window), int(batch_size),
    )
    if key not in _DECODE_WRAPPER_CACHE:
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchAttentionWithAttentionSinkWrapper(
            workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            qo_indptr_buf=pool._qo_indptr_buf[: batch_size + 1],
            paged_kv_indptr_buf=pool._kv_indptr_buf[: batch_size + 1],
            paged_kv_indices_buf=pool._kv_indices_buf,
            paged_kv_last_page_len_buf=pool._kv_last_page_len_buf[:batch_size],
            q_data_type=dtype,
            kv_data_type=dtype,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            window_left=window_left,
        )
        _DECODE_WRAPPER_CACHE[key] = (workspace, wrapper)
    return _DECODE_WRAPPER_CACHE[key][1]


# Module-global "active sink wrappers" slots. ``set_active_sink_wrappers``
# binds both the full-attention and sliding-window wrappers + the
# sm_scale; :func:`_sink_attn_run` reads them at execution time.
#
# Same rationale as ``paged_attention._ACTIVE_WRAPPER``: flashinfer's
# ``wrapper.run`` is plain Python that Dynamo traces into, causing graph
# breaks per layer. Wrapping it in a ``torch.library.custom_op`` with a
# tensor-only signature collapses each call to one opaque Dynamo node;
# the wrapper instances live in module globals because Dynamo cannot
# proxy arbitrary Python objects as op args. Single-threaded only.
_ACTIVE_SINK_FULL = None
_ACTIVE_SINK_SLIDING = None
_ACTIVE_SINK_SM_SCALE: float = 1.0


def set_active_sink_wrappers(
    full, sliding, sm_scale: float,
) -> None:
    """Bind the flashinfer sink wrappers (full + optional sliding) and
    the sm_scale that :func:`_sink_attn_run` dispatches against. Called
    once per forward, BEFORE the first :meth:`FlashinferSinkContext.run`.
    """
    global _ACTIVE_SINK_FULL, _ACTIVE_SINK_SLIDING, _ACTIVE_SINK_SM_SCALE
    _ACTIVE_SINK_FULL = full
    _ACTIVE_SINK_SLIDING = sliding
    _ACTIVE_SINK_SM_SCALE = float(sm_scale)


@torch.library.custom_op("torchllms::sink_attn_run", mutates_args=())
def _sink_attn_run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    is_sliding: int,
) -> torch.Tensor:
    """Custom-op wrapper around ``BatchAttentionWithAttentionSinkWrapper.run``.

    Opaque to Dynamo — the wrapper selection (full vs sliding) and the
    ``sm_scale`` are read from module globals at execution time. Tensor
    args go through the graph as normal inputs.
    """
    wrapper = _ACTIVE_SINK_SLIDING if is_sliding else _ACTIVE_SINK_FULL
    return wrapper.run(q, (k, v), sinks.float(), _ACTIVE_SINK_SM_SCALE)


@_sink_attn_run.register_fake
def _(q, k, v, sinks, is_sliding: int) -> torch.Tensor:
    return torch.empty_like(q)


@dataclass
class FlashinferSinkContext:
    """Pre-planned flashinfer state for one ``GptOSSTransformer.forward``.

    Built once in the transformer's forward and passed through to each
    attention layer. Both wrappers (full + sliding) are planned against
    the same :class:`PagedBatchLayout`; each layer picks based on its
    own ``sliding_window`` config.

    When ``sliding_window == 0`` model-wide (no SWA at all), ``sliding``
    is None — callers should only access ``full``.
    """

    full: "flashinfer.BatchAttentionWithAttentionSinkWrapper"
    sliding: Optional["flashinfer.BatchAttentionWithAttentionSinkWrapper"]
    sliding_window: int
    sm_scale: float
    layout: PagedBatchLayout
    pool: PagedKVPool

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

    def run(
        self,
        layer_id: int,
        q_flat: torch.Tensor,
        sinks: torch.Tensor,
        layer_sliding_window: int,
    ) -> torch.Tensor:
        """Run the sink attention kernel for one transformer layer.

        ``q_flat`` shape: ``[total_new_tokens, n_heads, head_dim]``.
        Returns: ``[total_new_tokens, n_heads, head_dim]``.

        Dispatches through the ``torchllms::sink_attn_run`` custom op,
        which reads the wrappers (full + sliding) from the module
        globals set by :func:`build_sink_context`. The custom-op
        boundary keeps ``wrapper.run`` opaque to Dynamo.

        ``layer_sliding_window > 0`` selects the sliding wrapper,
        ``<= 0`` selects the full wrapper. The actual window size is
        baked into each wrapper at construction time (see
        :func:`_decode_wrapper_for`).
        """
        k_cache_layer = self.pool.k_cache[layer_id]
        v_cache_layer = self.pool.v_cache[layer_id]
        is_sliding = 1 if layer_sliding_window > 0 else 0
        return _sink_attn_run(
            q_flat, k_cache_layer, v_cache_layer, sinks, is_sliding,
        )


def build_sink_context(
    *,
    pool: PagedKVPool,
    layout: PagedBatchLayout,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    sliding_window: int,
    sm_scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> FlashinferSinkContext:
    """Plan full + (optionally) sliding sink wrappers against the pool's
    post-extend block tables carried by ``layout``. ``sliding_window=0``
    skips the sliding plan entirely.

    Precondition: ``pool.extend_many(layout.rollout_ids, layout.qlens)``
    has already run — ``layout.kv_indptr`` / ``kv_indices`` /
    ``kv_last_page_len`` reflect the post-extend page lists.
    """
    _require_flashinfer()

    # Select decode vs prefill wrapper kind. Decode = every row has
    # exactly 1 new token — matches flashinfer's cudagraph contract
    # (fixed batch size, total_new_tokens == B). Anything else takes
    # the eager wrapper since prefill has variable total-new-tokens
    # across calls.
    is_decode = (
        len(layout.qlens) > 0
        and all(q == 1 for q in layout.qlens)
    )
    B = layout.batch_size

    if is_decode:
        full_wrapper = _decode_wrapper_for(
            pool=pool, dtype=dtype, head_dim=head_dim,
            sliding_window=0, batch_size=B,
        )
    else:
        full_wrapper = _prefill_wrapper_for(
            device=device, dtype=dtype, head_dim=head_dim, sliding_window=0,
        )
    full_wrapper.plan(
        qo_indptr=layout.qo_indptr,
        paged_kv_indptr=layout.kv_indptr,
        paged_kv_indices=layout.kv_indices,
        paged_kv_last_page_len=layout.kv_last_page_len,
        num_qo_heads=n_heads,
        num_kv_heads=n_kv_heads,
        head_dim_qk=head_dim,
        page_size=pool.page_size,
        causal=True,
        sm_scale=sm_scale,
        window_left=-1,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    sliding_wrapper = None
    if sliding_window > 0:
        if is_decode:
            sliding_wrapper = _decode_wrapper_for(
                pool=pool, dtype=dtype, head_dim=head_dim,
                sliding_window=sliding_window, batch_size=B,
            )
        else:
            sliding_wrapper = _prefill_wrapper_for(
                device=device, dtype=dtype, head_dim=head_dim,
                sliding_window=sliding_window,
            )
        sliding_wrapper.plan(
            qo_indptr=layout.qo_indptr,
            paged_kv_indptr=layout.kv_indptr,
            paged_kv_indices=layout.kv_indices,
            paged_kv_last_page_len=layout.kv_last_page_len,
            num_qo_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim_qk=head_dim,
            page_size=pool.page_size,
            causal=True,
            sm_scale=sm_scale,
            window_left=sliding_window - 1,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    # Bind the wrappers for the ``sink_attn_run`` custom op to find. Same
    # pattern as ``paged_attention.set_active_wrapper`` — called here
    # (outside any compile region) so per-layer ``FlashinferSinkContext.run``
    # can route through the op without coordinating per call.
    set_active_sink_wrappers(full_wrapper, sliding_wrapper, sm_scale)

    return FlashinferSinkContext(
        full=full_wrapper,
        sliding=sliding_wrapper,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        layout=layout,
        pool=pool,
    )


def build_sink_context_for_forward(
    *,
    pool: PagedKVPool,
    active_rids: Sequence[RolloutId],
    qlens: Sequence[int],
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    sliding_window: int,
    sm_scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, FlashinferSinkContext]:
    """Convenience composition of ``row_positions`` → ``extend_many`` →
    ``build_batch_layout`` → ``build_sink_context`` used by
    ``GptOSSTransformer.forward``.

    Returns ``(pre_write_seqlens, ctx)`` where ``pre_write_seqlens`` is
    the ``[B] int32`` tensor the caller uses to compute RoPE ``input_pos``.
    """
    pre_write = pool.row_positions(active_rids)
    pool.extend_many(active_rids, qlens)
    layout = pool.build_batch_layout(active_rids, qlens)
    ctx = build_sink_context(
        pool=pool,
        layout=layout,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        dtype=dtype,
        device=device,
    )
    return pre_write, ctx
