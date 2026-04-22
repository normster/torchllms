"""Flashinfer paged attention integration for the base torchllms Transformer
(Qwen3 path; gpt-oss uses ``flashinfer_attention.BatchAttentionWithAttentionSinkWrapper``
instead, kept separate during the Phase 1/2 migration).

One prefill wrapper + one decode wrapper per ``(device, dtype, head_dim,
window_left)`` process-wide. Plans are stateful — each ``.plan()`` call
overwrites the wrapper's internal buffers — so single-threaded inference
can share wrappers across layers and across ``LLM.generate()`` calls. Do
not call ``.plan()`` concurrently on the same wrapper.

Per-forward protocol (see ``PagedKVPool`` module docstring for the full
write sequence)::

    pre_write = pool.row_positions(active_rids)
    pool.extend_many(active_rids, qlens)
    layout = pool.build_batch_layout(active_rids, qlens)
    ctx = build_paged_context(pool=pool, layout=layout, ...)
    for layer_id in range(n_layers):
        pool.append_kv(layer_id, k_flat, v_flat, layout)
        out = ctx.run(layer_id, q_flat)

Prefill vs decode selection is based on per-row qlens — if every row has
``qlen == 1`` we use the decode wrapper (tuned for the batched-decode
throughput regime); anything else takes the prefill wrapper. Both handle
per-row diverging KV lengths via ``kv_indptr`` + ``kv_last_page_len``,
which replaces KVArena's ``cache_seqlens`` plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch

try:
    import flashinfer
except ImportError:  # pragma: no cover — exercised on machines w/o flashinfer
    flashinfer = None  # type: ignore[assignment]

from torchllms.models.paged_kv import PagedBatchLayout, PagedKVPool, RolloutId


# One 256 MB scratch per (device, dtype, head_dim, window_left) config.
_DEFAULT_WORKSPACE_BYTES = 256 * 1024 * 1024

# Process-level wrapper cache. Keyed on the JIT-compilation-relevant
# parameters so each unique kernel variant is compiled once.
_PREFILL_CACHE: Dict[tuple, Tuple[torch.Tensor, "flashinfer.BatchPrefillWithPagedKVCacheWrapper"]] = {}
_DECODE_CACHE: Dict[tuple, Tuple[torch.Tensor, "flashinfer.BatchDecodeWithPagedKVCacheWrapper"]] = {}


def _require_flashinfer() -> None:
    if flashinfer is None:
        raise RuntimeError(
            "flashinfer is not installed; install it or route attention "
            "through the KVArena (non-paged) path"
        )


def _prefill_wrapper_for(
    *,
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    window_left: int,
    workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
) -> "flashinfer.BatchPrefillWithPagedKVCacheWrapper":
    """Return (cached) eager-mode prefill wrapper.

    We deliberately do NOT use ``use_cuda_graph=True`` on the prefill
    wrapper. flashinfer's cudagraph-prefill contract pins the batch
    size + total new-token count at the first plan() call and rejects
    any subsequent plan that exceeds those bounds — not compatible with
    variable-length prefill across multi-turn workloads. SGLang uses
    the same pattern: eager prefill wrapper, cudagraph-mode decode
    wrapper (see `BatchDecodeWithPagedKVCacheWrapper` in
    :func:`_decode_wrapper_for`).

    For the bucketed cudagraph-on-prefill path (deferred; see
    docs/note_prefill_compile_plan.md § P1.2) we'll construct a
    separate set of per-bucket wrappers with ``use_cuda_graph=True``,
    each sized for its bucket's (B, max_total_new_tokens). For now,
    Inductor-only compile on prefill does not need buffer binding.
    """
    _require_flashinfer()
    key = (str(device), str(dtype), int(head_dim), int(window_left))
    if key not in _PREFILL_CACHE:
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD",
        )
        _PREFILL_CACHE[key] = (workspace, wrapper)
    return _PREFILL_CACHE[key][1]


def _decode_wrapper_for(
    *,
    pool: "PagedKVPool",
    dtype: torch.dtype,
    head_dim: int,
    window_left: int,
    batch_size: int,
    workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
) -> "flashinfer.BatchDecodeWithPagedKVCacheWrapper":
    """Return (cached) cudagraph-friendly decode wrapper for the given
    (device, dtype, head_dim, window_left, batch_size).

    The wrapper is constructed with ``use_cuda_graph=True`` and bound
    to the pool's pre-allocated indptr / indices / last_page_len
    buffers — a setup that keeps flashinfer's internal state at stable
    device addresses across plan() and run() calls, which is what
    torch.compile(mode="reduce-overhead") needs to capture the decode
    forward as a single cudagraph replay.

    One wrapper per distinct ``batch_size`` (flashinfer's cudagraph
    mode pins the batch size at construction; different B → different
    wrapper). For a B=1 decode-single + B=max_bsz decode-multiple
    workload, expect two wrappers to be cached.
    """
    _require_flashinfer()
    device = pool.device
    key = (str(device), str(dtype), int(head_dim), int(window_left), int(batch_size))
    if key not in _DECODE_CACHE:
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=pool._kv_indptr_buf[: batch_size + 1],
            paged_kv_indices_buffer=pool._kv_indices_buf,
            paged_kv_last_page_len_buffer=pool._kv_last_page_len_buf[:batch_size],
        )
        _DECODE_CACHE[key] = (workspace, wrapper)
    return _DECODE_CACHE[key][1]


# Module-global "active wrapper" slot. The caller (driver or eager-path
# ``PagedContext.run``) binds this before invoking the custom op below;
# the op's body reads it at execution time.
#
# Why the global indirection: flashinfer's internal
# ``register_custom_op`` is a deliberate no-op (see
# ``flashinfer/utils.py`` — they opted out of ``torch.library.custom_op``
# for overhead reasons), so ``wrapper.run`` is plain Python that Dynamo
# traces into, and breaks at the JIT module's ``Function.__call__``
# (``flashinfer/decode.py:260``). With 36 layers of decode-path
# attention, that's 36 graph breaks per forward → 36 trivial cudagraph
# partitions, many of them empty.
#
# Wrapping ``wrapper.run`` in OUR own ``torch.library.custom_op`` makes
# Dynamo treat the call as a single opaque node. The custom op's arg
# signature must be tensors only (Dynamo can't proxy arbitrary Python
# objects), so the wrapper instance goes through a module global
# instead. Single-threaded only.
_ACTIVE_WRAPPER = None


def set_active_wrapper(wrapper) -> None:
    """Bind the flashinfer wrapper that :func:`_paged_attn_run` will
    dispatch to. Call once per forward, BEFORE the first invocation of
    ``PagedContext.run``."""
    global _ACTIVE_WRAPPER
    _ACTIVE_WRAPPER = wrapper


@torch.library.custom_op("torchllms::paged_attn_run", mutates_args=())
def _paged_attn_run(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
) -> torch.Tensor:
    """Custom-op wrapper around the flashinfer wrapper's ``.run``.

    Opaque to Dynamo (no tracing into flashinfer's JIT module); the
    tensor args are normal graph inputs and the wrapper instance is
    looked up from ``_ACTIVE_WRAPPER`` at execution time. flashinfer
    allocates the output tensor internally and returns it.
    """
    return _ACTIVE_WRAPPER.run(q, (k, v))


@_paged_attn_run.register_fake
def _(q, k, v):
    return torch.empty_like(q)


@dataclass
class PagedContext:
    """Pre-planned flashinfer state for one Transformer.forward.

    Built once at the top of ``Transformer.forward``, threaded through
    each layer's ``Attention.forward``. One wrapper (prefill OR decode)
    is planned per forward; ``run`` dispatches to it.
    """

    wrapper: "flashinfer.BatchPrefillWithPagedKVCacheWrapper | flashinfer.BatchDecodeWithPagedKVCacheWrapper"
    is_decode: bool
    layout: PagedBatchLayout
    pool: PagedKVPool
    sm_scale: float

    def run(self, layer_id: int, q_flat: torch.Tensor) -> torch.Tensor:
        """Run attention for one layer.

        q_flat shape: ``[total_new_tokens, n_heads, head_dim]``.
        Returns: ``[total_new_tokens, n_heads, head_dim]``.

        Dispatches through the ``torchllms::paged_attn_run`` custom op,
        which reads the wrapper from the module-global set by
        :func:`build_paged_context`. The custom-op boundary keeps
        ``wrapper.run`` opaque to Dynamo — side-stepping the 36
        per-forward graph breaks it would otherwise cause.
        """
        k_cache_layer = self.pool.k_cache[layer_id]
        v_cache_layer = self.pool.v_cache[layer_id]
        return _paged_attn_run(q_flat, k_cache_layer, v_cache_layer)


def build_paged_context(
    *,
    pool: PagedKVPool,
    layout: PagedBatchLayout,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    sm_scale: float,
    dtype: torch.dtype,
    device: torch.device,
    window_left: int = -1,
) -> PagedContext:
    """Plan either the prefill or decode wrapper against the pool's
    current block tables, pack into a ``PagedContext``.

    Precondition: ``pool.extend_many(layout.rollout_ids, layout.qlens)``
    has run — ``kv_indptr`` / ``kv_indices`` / ``kv_last_page_len``
    reflect the post-extend page lists.

    ``window_left``: flashinfer convention for sliding-window attention.
    ``-1`` = full attention (default). Non-sink SWA would set
    ``window_left = window_size - 1``; for Qwen3 we always pass -1.
    """
    _require_flashinfer()

    is_decode = len(layout.qlens) > 0 and all(q == 1 for q in layout.qlens)

    if is_decode:
        wrapper = _decode_wrapper_for(
            pool=pool, dtype=dtype, head_dim=head_dim,
            window_left=window_left, batch_size=layout.batch_size,
        )
        # In use_cuda_graph mode, plan() copies indptr/indices/
        # last_page_len into the wrapper's bound buffers. Since our
        # layout aliases those same buffers (see
        # PagedKVPool.build_batch_layout), the copy is a no-op but the
        # plan call still updates flashinfer's internal state.
        wrapper.plan(
            indptr=layout.kv_indptr,
            indices=layout.kv_indices,
            last_page_len=layout.kv_last_page_len,
            num_qo_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
            page_size=pool.page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
            sm_scale=sm_scale,
            window_left=window_left,
        )
    else:
        wrapper = _prefill_wrapper_for(
            device=device, dtype=dtype, head_dim=head_dim,
            window_left=window_left,
        )
        wrapper.plan(
            qo_indptr=layout.qo_indptr,
            paged_kv_indptr=layout.kv_indptr,
            paged_kv_indices=layout.kv_indices,
            paged_kv_last_page_len=layout.kv_last_page_len,
            num_qo_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim_qk=head_dim,
            page_size=pool.page_size,
            causal=True,
            q_data_type=dtype,
            kv_data_type=dtype,
            sm_scale=sm_scale,
            window_left=window_left,
        )

    # Bind the wrapper for the custom op to find. Called here so every
    # caller (eager Transformer.forward + compiled-decode driver) gets
    # the binding without having to coordinate. Must happen OUTSIDE any
    # torch.compile region — which this function always is, since
    # ``wrapper.plan`` has CPU-sync ops that can't be inside a graph.
    set_active_wrapper(wrapper)

    return PagedContext(
        wrapper=wrapper,
        is_decode=is_decode,
        layout=layout,
        pool=pool,
        sm_scale=sm_scale,
    )


def build_paged_context_for_forward(
    *,
    pool: PagedKVPool,
    active_rids: Sequence[RolloutId],
    qlens: Sequence[int],
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    sm_scale: float,
    dtype: torch.dtype,
    device: torch.device,
    window_left: int = -1,
) -> Tuple[torch.Tensor, PagedContext]:
    """Convenience composition of ``row_positions`` → ``extend_many`` →
    ``build_batch_layout`` → ``build_paged_context`` used by the base
    Transformer's forward.

    Returns ``(pre_write_seqlens, ctx)`` where ``pre_write_seqlens`` is
    the ``[B] int32`` tensor the caller uses to compute RoPE ``input_pos``
    (before the write advances the rollouts' seqlens).
    """
    pre_write = pool.row_positions(active_rids)
    pool.extend_many(active_rids, qlens)
    layout = pool.build_batch_layout(active_rids, qlens)
    ctx = build_paged_context(
        pool=pool,
        layout=layout,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        sm_scale=sm_scale,
        dtype=dtype,
        device=device,
        window_left=window_left,
    )
    return pre_write, ctx
