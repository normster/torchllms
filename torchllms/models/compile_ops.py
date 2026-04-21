"""torch.library wrappers for the decode hot-path ops.

Two wrappers, both thin:

- ``fa_with_kvcache`` wraps ``flash_attn.flash_attn_with_kvcache`` as a
  registered custom op so Dynamo can trace through the FA2 pybind. FA2
  upstream doesn't register itself in ``torch.library`` (FA3 does), so
  without this wrapper a ``torch.compile``'d forward graph-breaks at the
  attention call. ``mutates_args=()`` because our callers pass read-only
  cache views (the KV scatter happens separately in ``update_kv_decode``).

- ``update_kv_decode`` wraps the decode-step scatter: write new K/V into
  the cache slab at each row's current ``seen_tokens`` position, advance
  the counter, return the full per-layer cache slabs + the pre-update
  ``cache_seqlens`` for FA2. The ``mutates_args`` annotation tells Dynamo
  and the cudagraph-safety checker which buffers are being written in
  place so capture doesn't fail with spurious stale-read warnings.

The function bodies match what ``KVArena.update_kv_decode`` (formerly
``update_kv_decode_static``) and ``_flash_attention_with_kvcache`` did
before; only the op boundary is new.
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
