from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class KVBlock:
    """A contiguous slice of KV state for a run of tokens.

    CPU-resident by convention. Device transfers happen at the arena boundary
    (LinearKVCache.load_block / extract_block).

    Shapes:
        k, v:     [n_layers, length, n_kv_heads, head_dim]
        role_ids: [length]
    """

    k: torch.Tensor
    v: torch.Tensor
    role_ids: torch.Tensor

    @property
    def length(self) -> int:
        return self.k.shape[1]

    @property
    def size_bytes(self) -> int:
        return (
            self.k.element_size() * self.k.numel()
            + self.v.element_size() * self.v.numel()
            + self.role_ids.element_size() * self.role_ids.numel()
        )

    def slice(self, start: int, end: int) -> "KVBlock":
        return KVBlock(
            k=self.k[:, start:end].clone(),
            v=self.v[:, start:end].clone(),
            role_ids=self.role_ids[start:end].clone(),
        )

    @staticmethod
    def concat(blocks: list["KVBlock"]) -> "KVBlock":
        return KVBlock(
            k=torch.cat([b.k for b in blocks], dim=1),
            v=torch.cat([b.v for b in blocks], dim=1),
            role_ids=torch.cat([b.role_ids for b in blocks], dim=0),
        )


class LinearKVCache:
    """In-flight KV buffer for one generation call.

    Pre-allocated contiguous tensors for K, V, role IDs, and attention mask,
    shaped `[max_batch_size, max_seqlen, ...]`. Each transformer layer calls
    update_kv() to append new K/V at the current position; update_role_ids()
    and update_attn_mask() extend their respective buffers in lockstep.

    Prefix reuse: load_block() blits a CPU-resident KVBlock into a given row at
    a given starting position and advances seen_tokens, so subsequent writes
    append past the loaded prefix. extract_block() is the inverse: copy a row's
    current state out as a CPU KVBlock for insertion into a RadixKVCache.

    KV vectors are stored AFTER positional embeddings are applied.
    """

    def __init__(
        self, n_layers, max_bsz, max_seqlen, n_kv_heads, head_dim, device, dtype
    ):
        cache_shape = (n_layers, max_bsz, max_seqlen, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)

        self.role_id_cache = torch.zeros(
            (max_bsz, max_seqlen), device=device, dtype=torch.long
        )
        self.attn_mask_cache = torch.zeros(
            (max_bsz, max_seqlen), device=device, dtype=torch.long
        )
        self.is_attn_mask_cached = False
        self.next_start_pos = torch.zeros(max_bsz, device=device, dtype=torch.long)

        self.seen_tokens = [0] * n_layers  # seen_tokens always counts masked tokens
        self.max_seqlen = max_seqlen

    @torch.no_grad()
    def update_role_ids(self, role_ids: Optional[torch.Tensor]):
        """Should be called before KV cache update."""

        if role_ids is None:
            return None

        start_pos = self.seen_tokens[0]
        end_pos = start_pos + role_ids.shape[1]
        self.role_id_cache[:, start_pos:end_pos] = role_ids
        return self.role_id_cache[:, :end_pos]

    @torch.no_grad()
    def update_attn_mask(self, attn_mask: Optional[torch.Tensor]):
        """Should be called before KV cache update."""

        if attn_mask is None:
            # we want to support only passing in attn_mask for the first prefill step
            if not self.is_attn_mask_cached:
                return None

            # automatically extend for next decoding step
            self.attn_mask_cache[:, self.seen_tokens[0]] = 1
            return self.attn_mask_cache[:, : self.seen_tokens[0] + 1]

        self.is_attn_mask_cached = True
        start_pos = self.seen_tokens[0]
        end_pos = start_pos + attn_mask.shape[1]
        self.attn_mask_cache[:, start_pos:end_pos] = attn_mask
        return self.attn_mask_cache[:, :end_pos]

    @torch.no_grad()
    def update_kv(self, layer_id, k_val, v_val):
        # k_val, v_val: [B, S, H, D]

        start_pos = self.seen_tokens[layer_id]
        tgt_len = k_val.shape[1]

        self.k_cache[layer_id, :, start_pos : start_pos + tgt_len] = k_val
        self.v_cache[layer_id, :, start_pos : start_pos + tgt_len] = v_val

        k_full = self.k_cache[layer_id, :, : start_pos + tgt_len]
        v_full = self.v_cache[layer_id, :, : start_pos + tgt_len]

        self.seen_tokens[layer_id] += tgt_len

        return k_full, v_full

    @torch.no_grad()
    def load_block(self, block: KVBlock, row_idx: int = 0, at_pos: int = 0) -> None:
        """Blit a CPU KVBlock into this arena at (row_idx, [at_pos : at_pos+L]).

        Updates seen_tokens for all layers to at_pos+L so subsequent update_kv
        calls append past the loaded prefix. Updates next_start_pos for the
        given row so RoPE continues from the right absolute position.

        Does NOT populate the attention mask cache. Single-sequence generation
        relies on is_causal=True in the attention kernel. Batched generation
        with a loaded prefix must supply an explicit attn_mask.

        Requires the arena to be fresh (seen_tokens all zero) or at least not
        have writes past at_pos on any layer.
        """
        device = self.k_cache.device
        k_dtype = self.k_cache.dtype
        L = block.length
        end = at_pos + L
        if end > self.max_seqlen:
            raise ValueError(
                f"load_block would exceed max_seqlen: at_pos={at_pos}, L={L}, max={self.max_seqlen}"
            )
        for layer_id, seen in enumerate(self.seen_tokens):
            if seen > at_pos:
                raise RuntimeError(
                    f"load_block at_pos={at_pos} but layer {layer_id} already has seen_tokens={seen}"
                )

        self.k_cache[:, row_idx, at_pos:end] = block.k.to(device=device, dtype=k_dtype)
        self.v_cache[:, row_idx, at_pos:end] = block.v.to(device=device, dtype=k_dtype)
        self.role_id_cache[row_idx, at_pos:end] = block.role_ids.to(
            device=device, dtype=self.role_id_cache.dtype
        )
        for layer_id in range(len(self.seen_tokens)):
            self.seen_tokens[layer_id] = end
        self.next_start_pos[row_idx] = end

    @torch.no_grad()
    def extract_block(self, row_idx: int = 0, length: Optional[int] = None) -> KVBlock:
        """Copy row_idx's KV state for positions [0:length] out as a CPU KVBlock.

        If length is omitted, uses seen_tokens[0] (assumed consistent across
        layers after any full forward pass).
        """
        if length is None:
            length = self.seen_tokens[0]
        if length > self.max_seqlen:
            raise ValueError(f"extract_block length={length} > max_seqlen={self.max_seqlen}")
        for layer_id, seen in enumerate(self.seen_tokens):
            if seen < length:
                raise RuntimeError(
                    f"extract_block length={length} but layer {layer_id} only has seen_tokens={seen}"
                )

        k = self.k_cache[:, row_idx, :length].detach().to(device="cpu").clone()
        v = self.v_cache[:, row_idx, :length].detach().to(device="cpu").clone()
        role_ids = (
            self.role_id_cache[row_idx, :length].detach().to(device="cpu").clone()
        )
        return KVBlock(k=k, v=v, role_ids=role_ids)

    def evict(self, evict_mask):
        self.k_cache = self.k_cache[:, ~evict_mask]
        self.v_cache = self.v_cache[:, ~evict_mask]
        self.role_id_cache = self.role_id_cache[~evict_mask]
        self.attn_mask_cache = self.attn_mask_cache[~evict_mask]
        self.next_start_pos = self.next_start_pos[~evict_mask]

    def is_full(self):
        return any([s >= self.max_seqlen for s in self.seen_tokens])
