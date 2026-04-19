"""
GPT-OSS model architecture for torchllms.

Implements the gpt-oss-20b/120b MoE transformer matching the torchllms
forward signature (input_ids, role_ids, attn_mask, input_pos, cache).

Key differences from the base torchllms Transformer:
- Sparse Mixture-of-Experts (MoE) FFN with top-k routing
- Fused QKV projection
- Sink tokens in attention
- Sliding window attention on alternating layers
- YaRN RoPE scaling for long context
- Custom SwiGLU with clamping (alpha=1.702, limit=7.0)

Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchllms.models.attention_gptoss import attention as triton_attention
from torchllms.models.attention_gptoss import attention_ref
from torchllms.models.cache import DecodingCache
from torchllms.models.networks import (
    ModelParams,
    RMSNorm,
    RoleEmbeddings,
)


class YaRNRotaryEmbeddings(nn.Module):
    """Rotary Position Embeddings with YaRN scaling (NTK-by-parts).

    See: https://arxiv.org/abs/2309.00071
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 131072,
        base: float = 150000.0,
        initial_context_length: int = 4096,
        scaling_factor: float = 32.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.rope_init()

    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        )

        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0
            d_half = self.head_dim / 2
            low = (
                d_half
                * math.log(
                    self.initial_context_length / (self.ntk_beta * 2 * math.pi)
                )
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(
                    self.initial_context_length / (self.ntk_alpha * 2 * math.pi)
                )
                / math.log(self.base)
            )

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        seq_idx = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", seq_idx, inv_freq)
        # Cache cos/sin as [max_seq_len, head_dim//2] for half-split rotation
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply rotary embeddings using half-split convention.

        The reference gpt-oss implementation splits each head into first half
        and second half (not interleaved pairs), matching the standard RoPE
        convention from the original paper.

        Args:
            x: [bsz, seqlen, n_heads, head_dim]
            input_pos: [bsz, seqlen] or None
        Returns:
            Rotated tensor with same shape as x.
        """
        seq_len = x.size(1)

        if input_pos is None:
            cos = self.cos_cached[:seq_len]  # [seqlen, head_dim//2]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[input_pos]  # [bsz, seqlen, head_dim//2]
            sin = self.sin_cached[input_pos]

        # Broadcast cos/sin to match x shape: add head dimension
        # x is [bsz, seqlen, n_heads, head_dim]
        # cos/sin need to be [..., 1, head_dim//2] for broadcasting
        if cos.dim() == 2:
            # [seqlen, head_dim//2] -> [1, seqlen, 1, head_dim//2]
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        else:
            # [bsz, seqlen, head_dim//2] -> [bsz, seqlen, 1, head_dim//2]
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)

        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        # Half-split rotation: split head_dim into first half and second half
        x1, x2 = x.chunk(2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)


def _gptoss_swiglu(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0):
    """SwiGLU activation with clamping, as used in gpt-oss."""
    x_glu = x[..., ::2].clamp(max=limit)
    x_linear = x[..., 1::2].clamp(min=-limit, max=limit)
    return x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + 1)


class GptOSSAttention(nn.Module):
    """Multi-head attention with fused QKV, GQA, sink tokens, and sliding window."""

    def __init__(self, layer_id: int, params: ModelParams):
        super().__init__()
        self.layer_id = layer_id
        self.head_dim = params.head_dim
        self.n_heads = params.n_heads
        self.n_kv_heads = params.n_kv_heads

        # Sliding window on alternating layers (even-indexed layers)
        self.sliding_window = params.gpt_oss_sliding_window if layer_id % 2 == 0 else 0

        # Sink token logits (per-head learnable bias for attention sink)
        self.sinks = nn.Parameter(
            torch.empty(params.n_heads, dtype=torch.bfloat16)
        )

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        qkv_dim = params.head_dim * (params.n_heads + 2 * params.n_kv_heads)
        self.qkv = nn.Linear(params.dim, qkv_dim, bias=True)
        self.out = nn.Linear(params.n_heads * params.head_dim, params.dim, bias=True)

        self.sm_scale = 1.0 / math.sqrt(params.head_dim)

        self.rope = YaRNRotaryEmbeddings(
            params.head_dim,
            max_seq_len=params.max_seq_len,
            base=params.rope_theta,
            initial_context_length=params.gpt_oss_initial_context_length,
            scaling_factor=params.gpt_oss_rope_scaling_factor,
            ntk_alpha=params.gpt_oss_rope_ntk_alpha,
            ntk_beta=params.gpt_oss_rope_ntk_beta,
        )

    def forward(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[DecodingCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape

        t = self.norm(x)
        qkv = self.qkv(t)

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        q = self.rope(q, input_pos=input_pos)
        k = self.rope(k, input_pos=input_pos)

        # Compute start_q for the attention kernel (position of first query token)
        if input_pos is not None:
            start_q = input_pos[:, 0:1].to(torch.long)
        elif cache is not None:
            start_q = cache.next_start_pos[:1].to(torch.long)
        else:
            start_q = torch.zeros(1, dtype=torch.long, device=x.device)

        if cache is not None:
            k, v = cache.update_kv(self.layer_id, k, v)

        # GQA: reshape q to [bsz, seqlen, n_kv_heads, q_per_kv, head_dim]
        q_per_kv = self.n_heads // self.n_kv_heads
        q = q.view(bsz, seqlen, self.n_kv_heads, q_per_kv, self.head_dim)

        # Triton kernel for prefill (seqlen >= 64), eager ref for decode
        if seqlen >= 64 and x.is_cuda:
            output = triton_attention(
                q, k, v, self.sinks, self.sm_scale,
                self.sliding_window, start_q,
            )
        else:
            output = attention_ref(
                q, k, v, self.sinks, self.sm_scale,
                self.sliding_window, start_q,
            )

        return self.out(output)


class MXFPStorage:
    """Lightweight container for MXFP4-packed expert weights.

    Stores packed uint8 blocks and uint8 scales as registered buffers on the
    parent module. Provides on-the-fly decoding to bf16 for individual experts.
    Memory footprint is ~4x smaller than bf16 for the MoE weights.
    """

    def __init__(self, blocks: torch.Tensor, scales: torch.Tensor):
        self.blocks = blocks  # [num_experts, G, B] uint8
        self.scales = scales  # [num_experts, G] uint8

    def decode(self, expert_id: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Decode a single expert's weight from MXFP4 to bf16.

        Blocks may be [num_experts, rows, groups, B] or [num_experts, G, B].
        Scales are one dim less (no B). Output is the fully decoded weight matrix.
        """
        b = self.blocks[expert_id]   # [*prefix, B]
        s = self.scales[expert_id]   # [*prefix]

        # Flatten to [total_groups, B]
        *prefix, B = b.shape
        total_groups = 1
        for d in prefix:
            total_groups *= d
        b = b.reshape(total_groups, B)
        s = s.reshape(total_groups)

        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=b.device)
        s_int = s.to(torch.int32) - 127

        idx_lo = (b & 0x0F).to(torch.long)
        idx_hi = (b >> 4).to(torch.long)

        out = torch.empty(total_groups, B * 2, dtype=dtype, device=b.device)
        out[:, 0::2] = lut[idx_lo]
        out[:, 1::2] = lut[idx_hi]
        torch.ldexp(out, s_int.unsqueeze(-1), out=out)

        # Return flattened: [total_groups * B * 2]
        return out.reshape(-1)


# FP4 lookup table for MXFP4 decoding
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


class GptOSSMoE(nn.Module):
    """Sparse Mixture-of-Experts FFN with top-k routing and SwiGLU activation.

    Supports two storage modes for expert weights:
    - bf16: Standard nn.Parameter storage (set via load_bf16_weights)
    - MXFP4: Packed uint8 buffers with on-the-fly decoding (set via load_mxfp4_weights)

    Uses group-by-expert routing: tokens are sorted by expert assignment, then
    each unique expert processes all its assigned tokens in a single matmul.
    This avoids materializing the full gathered weight tensor.
    """

    def __init__(self, layer_id: int, params: ModelParams):
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = params.gpt_oss_num_experts
        self.experts_per_token = params.gpt_oss_experts_per_token
        self.swiglu_limit = params.gpt_oss_swiglu_limit
        self.hidden_size = params.dim
        self.intermediate_size = params.gpt_oss_intermediate_size

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.gate = nn.Linear(
            params.dim, self.num_experts, bias=True, dtype=torch.bfloat16
        )

        # Biases are always bf16 (small)
        self.mlp1_bias = nn.Parameter(
            torch.empty(
                self.num_experts, self.intermediate_size * 2, dtype=torch.bfloat16
            )
        )
        self.mlp2_bias = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=torch.bfloat16)
        )

        # Expert weight storage — populated by weight loader.
        # MXFP4 mode: packed buffers decoded on-the-fly (default, ~4x smaller)
        # bf16 mode: standard parameter tensors (fallback)
        self.mlp1_mxfp: Optional[MXFPStorage] = None
        self.mlp2_mxfp: Optional[MXFPStorage] = None
        self.mlp1_weight: Optional[nn.Parameter] = None
        self.mlp2_weight: Optional[nn.Parameter] = None

    def load_mxfp4_weights(
        self,
        mlp1_blocks: torch.Tensor,
        mlp1_scales: torch.Tensor,
        mlp2_blocks: torch.Tensor,
        mlp2_scales: torch.Tensor,
    ):
        """Load expert weights in MXFP4 packed format."""
        self.register_buffer("_mlp1_blocks", mlp1_blocks)
        self.register_buffer("_mlp1_scales", mlp1_scales)
        self.register_buffer("_mlp2_blocks", mlp2_blocks)
        self.register_buffer("_mlp2_scales", mlp2_scales)
        self.mlp1_mxfp = MXFPStorage(mlp1_blocks, mlp1_scales)
        self.mlp2_mxfp = MXFPStorage(mlp2_blocks, mlp2_scales)

    def load_bf16_weights(
        self, mlp1_weight: torch.Tensor, mlp2_weight: torch.Tensor
    ):
        """Load expert weights as bf16 parameters (fallback)."""
        self.mlp1_weight = nn.Parameter(mlp1_weight, requires_grad=False)
        self.mlp2_weight = nn.Parameter(mlp2_weight, requires_grad=False)

    def _get_expert_weights(
        self, expert_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get (mlp1_weight, mlp2_weight) for a single expert, decoding MXFP4 if needed."""
        if self.mlp1_mxfp is not None:
            w1 = self.mlp1_mxfp.decode(expert_id)
            w2 = self.mlp2_mxfp.decode(expert_id)
            # Reshape to matrix form: w1=[inter*2, hidden], w2=[hidden, inter]
            w1 = w1.view(self.intermediate_size * 2, self.hidden_size)
            w2 = w2.view(self.hidden_size, self.intermediate_size)
            return w1, w2
        else:
            return self.mlp1_weight[expert_id], self.mlp2_weight[expert_id]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        num_tokens = bsz * seqlen
        t = self.norm(x)
        t_flat = t.view(num_tokens, dim)

        # Route: top-k expert selection
        gate_logits = self.gate(t_flat)  # [T, E]
        topk = torch.topk(gate_logits, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = F.softmax(topk.values, dim=1)  # [T, K]
        expert_indices = topk.indices  # [T, K]

        # Flatten assignments: each (token, expert) pair
        K = self.experts_per_token
        flat_expert_ids = expert_indices.view(-1)  # [T*K]
        flat_token_idx = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(-1, K)
            .reshape(-1)
        )  # [T*K]
        flat_weights = expert_weights.view(-1)  # [T*K]

        # Sort by expert to group tokens going to the same expert
        order = flat_expert_ids.argsort(stable=True)
        sorted_expert_ids = flat_expert_ids[order]
        sorted_token_idx = flat_token_idx[order]
        sorted_weights = flat_weights[order]
        sorted_inputs = t_flat[sorted_token_idx]  # [T*K, hidden]

        # Process each unique expert
        unique_experts, counts = torch.unique_consecutive(
            sorted_expert_ids, return_counts=True
        )

        sorted_outputs = torch.empty(
            num_tokens * K, dim, dtype=x.dtype, device=x.device
        )

        offset = 0
        for eid, cnt in zip(unique_experts.tolist(), counts.tolist()):
            inp = sorted_inputs[offset : offset + cnt]  # [cnt, hidden]
            w1, w2 = self._get_expert_weights(eid)
            b1 = self.mlp1_bias[eid]
            b2 = self.mlp2_bias[eid]

            h = inp @ w1.T + b1  # [cnt, inter*2]
            h = _gptoss_swiglu(h, limit=self.swiglu_limit)
            h = h @ w2.T + b2  # [cnt, hidden]

            sorted_outputs[offset : offset + cnt] = h
            offset += cnt

        # Scatter weighted results back to token positions
        sorted_outputs = sorted_outputs * sorted_weights.unsqueeze(-1)
        result = torch.zeros(num_tokens, dim, dtype=x.dtype, device=x.device)
        result.scatter_add_(
            0,
            sorted_token_idx.unsqueeze(-1).expand_as(sorted_outputs),
            sorted_outputs,
        )

        return x + result.view(bsz, seqlen, dim)


class GptOSSTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, params: ModelParams):
        super().__init__()
        self.layer_id = layer_id
        self.attn = GptOSSAttention(layer_id, params)
        self.mlp = GptOSSMoE(layer_id, params)

    def forward(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[DecodingCache] = None,
    ):
        x = x + self.attn(x, role_ids, input_pos, cache, attn_mask)
        x = self.mlp(x)
        return x


class GptOSSTransformer(nn.Module):
    """GPT-OSS MoE Transformer matching the torchllms Transformer interface.

    Weight names use gpt-oss conventions (embedding, block, unembedding) to
    align with the SafeTensors checkpoint format.
    """

    def __init__(self, params: ModelParams):
        super().__init__()
        self.params = params

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim, dtype=torch.bfloat16
        )

        if params.use_role_embeddings:
            self.role_embeddings = RoleEmbeddings(params)
        else:
            self.role_embeddings = None

        self.layers = nn.ModuleList(
            [GptOSSTransformerBlock(i, params) for i in range(params.n_layers)]
        )

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False, dtype=torch.bfloat16
        )

    def init_cache(
        self, max_batch_size: int, device: str, max_cache_len: Optional[int] = None
    ):
        return DecodingCache(
            self.params.n_layers,
            max_batch_size,
            max_cache_len or self.params.max_seq_len,
            self.params.n_kv_heads,
            self.params.head_dim,
            device,
            dtype=torch.bfloat16,
        )

    def get_wd_params(self):
        wd_params = []
        no_wd_params = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "bias" in n.lower() or "norm" in n.lower() or "sink" in n.lower():
                no_wd_params.append(p)
            else:
                wd_params.append(p)
        return wd_params, no_wd_params

    def forward(
        self,
        input_ids: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[DecodingCache] = None,
        logits_to_keep: Optional[int] = None,
    ):
        assert (
            cache is None or not cache.is_full()
        ), "Maximum sequence length reached, KV cache is full"

        if input_pos is None and cache is not None:
            input_pos = torch.arange(input_ids.shape[1])[None, :]
            input_pos = input_pos.to(input_ids.device) + cache.next_start_pos[:, None]

        h = self.tok_embeddings(input_ids)

        if self.role_embeddings is not None and role_ids is not None:
            h += self.role_embeddings(role_ids)

        if cache is not None:
            role_ids = cache.update_role_ids(role_ids)
            attn_mask = cache.update_attn_mask(attn_mask)
            cache.next_start_pos = input_pos[:, -1] + 1

        for layer in self.layers:
            h = layer(h, role_ids, attn_mask, input_pos, cache)

        if logits_to_keep is not None:
            h = h[:, -logits_to_keep:]

        logits = self.output(self.norm(h))

        return logits.float(), cache
