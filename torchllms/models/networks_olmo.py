from typing import Optional

import torch
import torch.nn as nn

from torchllms.models.cache import KVArena
from torchllms.models.networks import (
    AttentionImpl,
    FeedForward,
    ModelParams,
    RMSNorm,
    RotaryPositionalEmbeddings,
    Transformer,
    _eager_attention,
    _flash_attention,
    _sdpa_attention,
)


class OLMo2Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, layer_id: int, params: ModelParams):
        super().__init__()
        self.params = params
        self.layer_id = layer_id
        self.rope = RotaryPositionalEmbeddings(
            params.head_dim,
            params.max_seq_len,
            params.rope_theta,
            **params.rope_scaling,
        )

        self.wq = nn.Linear(
            params.dim,
            params.n_heads * params.head_dim,
            bias=params.attn_proj_bias,
        )
        self.wk = nn.Linear(
            params.dim,
            params.n_kv_heads * params.head_dim,
            bias=params.attn_proj_bias,
        )
        self.wv = nn.Linear(
            params.dim,
            params.n_kv_heads * params.head_dim,
            bias=params.attn_proj_bias,
        )
        self.wo = nn.Linear(
            params.n_heads * params.head_dim,
            params.dim,
            bias=False,
        )

        self.q_norm = RMSNorm(params.n_heads * params.head_dim, eps=params.norm_eps)
        self.k_norm = RMSNorm(params.n_kv_heads * params.head_dim, eps=params.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[KVArena] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.params.n_heads, self.params.head_dim)
        xk = xk.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)
        xv = xv.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)

        xq = self.rope(xq, input_pos=input_pos)
        xk = self.rope(xk, input_pos=input_pos)

        if cache is not None:
            xk, xv = cache.update_kv(self.layer_id, xk, xv)

        if self.params.attention_impl == AttentionImpl.FLASH:
            assert attn_mask is None
            output = _flash_attention(xq, xk, xv)
        elif self.params.attention_impl == AttentionImpl.SDPA:
            output = _sdpa_attention(xq, xk, xv, attn_mask)
        else:
            output = _eager_attention(xq, xk, xv, attn_mask)

        return self.wo(output)

    def forward_scores(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.params.n_heads, self.params.head_dim)
        xk = xk.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)
        xv = xv.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)

        xq = self.rope(xq, input_pos=input_pos)
        xk = self.rope(xk, input_pos=input_pos)

        output, scores = _eager_attention(
            xq,
            xk,
            xv,
            attn_mask,
            output_scores=True,
        )

        return self.wo(output), scores


class OLMo2TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, params: ModelParams):
        super().__init__()
        self.attention = OLMo2Attention(layer_id, params)
        self.feed_forward = FeedForward(params)
        self.attention_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.ffn_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[KVArena] = None,
    ):
        h = x + self.attention_norm(
            self.attention(
                x,
                role_ids,
                input_pos,
                cache,
                attn_mask,
            )
        )
        out = h + self.ffn_norm(self.feed_forward(h))
        return out

    def forward_scores(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        h, scores = self.attention.forward_scores(
            x,
            role_ids,
            attn_mask,
            input_pos,
        )
        h = x + self.attention_norm(h)
        out = h + self.ffn_norm(self.feed_forward(h))
        return out, scores


class OLMo2Transformer(Transformer):
    """OLMo2 Transformer.

    Not migrated to the paged KV cache in Phase 1/2 (see
    ``torchllms.models.paged_kv``) — still uses KVArena. Overrides
    ``init_cache`` + ``forward`` to route around the base ``Transformer``
    paged-cache plumbing. Migration scheduled for Phase 4 alongside gpt-oss.
    """

    def __init__(self, params: ModelParams):
        super().__init__(params)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(OLMo2TransformerBlock(layer_id, params))

    def init_cache(
        self, max_batch_size: int, device: str, max_cache_len: Optional[int] = None,
        **_ignored,
    ):
        return KVArena(
            self.params.n_layers,
            max_batch_size,
            max_cache_len or self.params.max_seq_len,
            self.params.n_kv_heads,
            self.params.head_dim,
            device,
            dtype=self.tok_embeddings.weight.dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[KVArena] = None,
        logits_to_keep: Optional[int] = None,
    ):
        """Pre-paged forward using KVArena. Mirrors the pre-Phase-1
        ``Transformer.forward`` body."""
        if input_pos is None and cache is not None:
            input_pos = torch.arange(
                input_ids.shape[1], dtype=torch.int32, device=input_ids.device,
            )[None, :] + cache.row_positions[:, None]

        h = self.tok_embeddings(input_ids)
        if self.role_embeddings is not None and role_ids is not None:
            h += self.role_embeddings(role_ids)

        active_role_ids = role_ids
        if cache is not None:
            role_ids = cache.update_role_ids(role_ids)
            attn_mask = cache.update_attn_mask(attn_mask)

        for i, layer in enumerate(self.layers):
            h = layer(h, role_ids, attn_mask, input_pos, cache)
            h = self._apply_interventions(
                h, layer_id=i, role_ids=active_role_ids,
            )

        if logits_to_keep is not None:
            h = h[:, -logits_to_keep:]
        logits = self.output(self.norm(h))
        return logits, cache
