"""torch.library wrappers for inference hot-path kernels.

Each entry is a thin wrapper around a backend kernel (flash-attn /
flashinfer) registered as a ``torch.library.custom_op`` so Dynamo can
trace through the pybind and cudagraph capture sees the correct
mutation semantics.

Entries:

- ``fa_with_kvcache`` — ``flash_attn.flash_attn_with_kvcache`` read-only
  on the KV cache (the KV scatter lives in ``update_kv_decode``).
  **Legacy**: only used by KVArena-backed attention (gpt-oss, olmo).

- ``update_kv_decode`` — KVArena decode-step scatter: write new K/V at
  each row's ``seen_tokens`` position, advance the counter, return the
  fresh ``cache_seqlens``. ``mutates_args`` tells the cudagraph safety
  pass which buffers are being written in place. **Legacy** (KVArena).

- ``update_role_ids_decode`` / ``update_attn_mask_decode`` /
  ``extend_attn_mask_decode`` — KVArena role/attn_mask scratch scatters,
  wrapped so Inductor doesn't elide them as dead code. **Legacy**
  (KVArena only; base Transformer paged path doesn't use these scratches).

- ``paged_append_kv`` — ``flashinfer.append_paged_kv_cache`` wrapper for
  the paged pool. Mutates the paged K/V slabs via the kernel's per-token
  (batch_indices, positions) scatter. Primary KV write op for Qwen3 and
  any future paged-backend model.

Legacy ops stay registered until Phase 4 migrates gpt-oss + olmo to the
paged pool. They have no runtime cost when unused.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.library import custom_op

try:
    from flash_attn import flash_attn_with_kvcache as _fa_with_kvcache
except ImportError:
    _fa_with_kvcache = None


# =====================================================================
# flash_attn_with_kvcache wrapper (read-only cache view)
# =====================================================================

if _fa_with_kvcache is not None:

    @custom_op("torchllms::fa_with_kvcache", mutates_args=())
    def fa_with_kvcache(
        q: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        cache_seqlens: Tensor,
    ) -> Tensor:
        """Decode attention with per-row cache_seqlens. Read-only on KV.

        q:             [B, qlen, n_heads, head_dim]
        k_cache/v_cache: [B, max_seqlen, n_kv_heads, head_dim] — full slab
        cache_seqlens: [B] int32 — per-row valid length
        returns:       [B, qlen, n_heads, head_dim]
        """
        return _fa_with_kvcache(
            q, k_cache, v_cache,
            cache_seqlens=cache_seqlens,
            causal=True,
        )

    @fa_with_kvcache.register_fake
    def _(q, k_cache, v_cache, cache_seqlens):
        return torch.empty_like(q)

else:
    fa_with_kvcache = None  # type: ignore[assignment]


# =====================================================================
# Decode-step KV scatter (mutates k_cache, v_cache, seen_tokens)
# =====================================================================


@custom_op(
    "torchllms::update_kv_decode",
    mutates_args={"k_cache", "v_cache", "seen_tokens"},
)
def update_kv_decode(
    k_cache: Tensor,     # [n_layers, max_bsz, max_seqlen, n_kv_heads, head_dim]
    v_cache: Tensor,     # same shape
    seen_tokens: Tensor, # [n_layers, max_bsz], int32
    layer_id: int,
    k_val: Tensor,       # [B, 1, n_kv_heads, head_dim]
    v_val: Tensor,       # same shape
) -> Tensor:
    """Write one new token's K/V into the cache at per-row positions,
    advance ``seen_tokens[layer_id]`` by 1 for each live row, and return
    the post-advance ``cache_seqlens`` (a fresh tensor, not aliasing any
    input — torch.library forbids output-aliases-input on custom ops).

    FA2 needs the post-advance value since after the write each row has
    one more valid token; FA2 reads ``k_cache[b, :cache_seqlens[b], :, :]``.

    The caller slices ``k_cache[layer_id, :B]`` and ``v_cache[layer_id, :B]``
    *outside* this op to get the full per-layer cache slabs to feed FA2.
    That slicing is a normal tensor op Dynamo traces without aliasing
    complaints.
    """
    B = k_val.shape[0]
    positions = seen_tokens[layer_id, :B]
    rows = torch.arange(B, device=k_cache.device)
    k_cache[layer_id, rows, positions] = k_val[:, 0]
    v_cache[layer_id, rows, positions] = v_val[:, 0]
    cache_seqlens = positions + 1
    seen_tokens[layer_id, :B] = cache_seqlens
    return cache_seqlens


@update_kv_decode.register_fake
def _(k_cache, v_cache, seen_tokens, layer_id, k_val, v_val):
    B = k_val.shape[0]
    return torch.empty((B,), dtype=torch.int32, device=k_cache.device)


# =====================================================================
# Decode-step role_id / attn_mask scatters (mutate aux caches)
# =====================================================================
#
# The same pattern as ``update_kv_decode``, but for the role-id and
# attention-mask scratch buffers threaded through the forward by
# ``KVArena.update_role_ids`` / ``update_attn_mask``. Without the
# ``mutates_args`` annotation, Inductor can observe that the scatter's
# result isn't consumed by the output graph (Qwen3/gpt-oss attention
# doesn't read these buffers) and elide the write as dead code, leaving
# subsequent steps reading stale state. Wrapping the scatter in a custom
# op tells Dynamo + Inductor the buffer IS mutated, preserving the
# invariant even when the result isn't used in the graph's output.
#
# Shape of role_id_cache: [max_bsz, max_seqlen], dtype=long (usually).
# Shape of attn_mask_cache: [max_bsz, max_seqlen], dtype=long.


@custom_op(
    "torchllms::update_role_ids_decode",
    mutates_args={"role_id_cache"},
)
def update_role_ids_decode(
    role_id_cache: Tensor,   # [max_bsz, max_seqlen]
    seen_tokens: Tensor,     # [n_layers, max_bsz], int32
    role_ids: Tensor,        # [B, 1]
) -> None:
    B = role_ids.shape[0]
    positions = seen_tokens[0, :B]
    rows = torch.arange(B, device=role_id_cache.device, dtype=positions.dtype)
    role_id_cache[rows, positions] = role_ids[:, 0]


@update_role_ids_decode.register_fake
def _(role_id_cache, seen_tokens, role_ids):
    return None


@custom_op(
    "torchllms::update_attn_mask_decode",
    mutates_args={"attn_mask_cache"},
)
def update_attn_mask_decode(
    attn_mask_cache: Tensor,  # [max_bsz, max_seqlen]
    seen_tokens: Tensor,      # [n_layers, max_bsz], int32
    attn_mask: Tensor,        # [B, 1]
) -> None:
    B = attn_mask.shape[0]
    positions = seen_tokens[0, :B]
    rows = torch.arange(B, device=attn_mask_cache.device, dtype=positions.dtype)
    attn_mask_cache[rows, positions] = attn_mask[:, 0]


@update_attn_mask_decode.register_fake
def _(attn_mask_cache, seen_tokens, attn_mask):
    return None


@custom_op(
    "torchllms::extend_attn_mask_decode",
    mutates_args={"attn_mask_cache"},
)
def extend_attn_mask_decode(
    attn_mask_cache: Tensor,  # [max_bsz, max_seqlen]
    seen_tokens: Tensor,      # [n_layers, max_bsz], int32
    b_live: int,
) -> None:
    """Auto-extend path (update_attn_mask(None) with is_attn_mask_cached):
    write ``1`` into each row's current seen_tokens position.
    """
    B = b_live
    if B == 0:
        return
    positions = seen_tokens[0, :B]
    rows = torch.arange(B, device=attn_mask_cache.device, dtype=positions.dtype)
    attn_mask_cache[rows, positions] = 1


@extend_attn_mask_decode.register_fake
def _(attn_mask_cache, seen_tokens, b_live):
    return None


# =====================================================================
# Paged KV append (mutates paged K/V slabs, wraps flashinfer)
# =====================================================================
#
# ``flashinfer.append_paged_kv_cache`` scatters new K/V tokens into the
# paged pool given per-token (batch_indices, positions) tensors. The
# kernel reads kv_indices / kv_indptr / kv_last_page_len to find the
# right page for each token.
#
# Wrapped as a custom op with ``mutates_args`` on the paged slabs so
# Dynamo treats this as a side-effecting call, cudagraph capture knows
# to pin the buffers, and the write isn't elided as dead code when the
# caller doesn't consume the slab in the output graph (it consumes it
# later via ``.run()`` on the flashinfer wrapper, which reads the slab
# as a plain tensor).


@custom_op(
    "torchllms::paged_append_kv",
    mutates_args={"paged_k_cache", "paged_v_cache"},
)
def paged_append_kv(
    paged_k_cache: Tensor,  # [n_layers, total_pages, page_size, n_kv, d]
    paged_v_cache: Tensor,  # same
    layer_id: int,
    append_key: Tensor,     # [total_new_tokens, n_kv_heads, head_dim]
    append_value: Tensor,   # same
    batch_indices: Tensor,  # [total_new_tokens] int32
    positions: Tensor,      # [total_new_tokens] int32
    kv_indices: Tensor,     # [total_pages] int32 (pool's _kv_indices_buf)
    kv_indptr: Tensor,      # [B+1] int32 (pool's _kv_indptr_buf[:B+1])
    kv_last_page_len: Tensor,  # [B] int32 (pool's _kv_last_page_len_buf[:B])
) -> None:
    """Scatter ``[total_new_tokens, n_kv, d]`` K/V into the paged slabs
    at (batch_indices, positions), using the per-batch page list.

    The layer slice happens INSIDE the op body so the argument seen by
    Dynamo/Inductor is the full 5D ``[n_layers, ...]`` slab (which is
    ``mark_static_address``'d by ``PagedKVPool``). If we sliced at the
    call site (``self.k_cache[layer_id]``), Inductor would treat the
    view as a non-static input and refuse cudagraph capture with
    ``skipping cudagraphs due to mutated inputs``.
    """
    import flashinfer
    flashinfer.append_paged_kv_cache(
        append_key=append_key,
        append_value=append_value,
        batch_indices=batch_indices,
        positions=positions,
        paged_kv_cache=(paged_k_cache[layer_id], paged_v_cache[layer_id]),
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
        kv_layout="NHD",
    )


@paged_append_kv.register_fake
def _(
    paged_k_cache, paged_v_cache, layer_id, append_key, append_value,
    batch_indices, positions, kv_indices, kv_indptr, kv_last_page_len,
):
    return None
