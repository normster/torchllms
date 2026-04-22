# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Set

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
except ImportError:
    print("flash_attn not found, using native PyTorch implementation")
    flash_attn_func = None
    flash_attn_with_kvcache = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from torchllms.messages import Role
from torchllms.models import utils
from torchllms.models.paged_kv import PagedKVPool


def _as_int_tuple(value) -> tuple[int, ...]:
    """Normalize a layer or role spec to a tuple of ints."""
    if isinstance(value, Role):
        return (int(value),)
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(v) for v in value)


class Intervention(nn.Module):
    """Wraps an activation-intervention module with dispatch metadata.

    Users normally don't instantiate this directly; call
    :meth:`Transformer.register_intervention` which builds the wrapper.

    The wrapped ``module``'s ``forward(hidden)`` must return a tensor
    that broadcasts to ``hidden.shape``. The driver applies
    ``hidden = hidden + mask * module(hidden)`` at each layer in
    ``layers``, gated by ``role_ids`` (or unconditionally if ``role_ids
    is None``).
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        layers: int | Sequence[int],
        role_ids=None,
    ) -> None:
        super().__init__()
        self.module = module
        ls = _as_int_tuple(layers)
        if not ls:
            raise ValueError("layers must be non-empty")
        # Tuple supports `in` natively; a set isn't worth the overhead for
        # the typical 1–4 layer configs.
        self.layers: tuple[int, ...] = ls
        if role_ids is None:
            self.role_ids: Optional[tuple[int, ...]] = None
            self.register_buffer(
                "_roles_t", torch.empty(0, dtype=torch.long), persistent=False,
            )
        else:
            rs = _as_int_tuple(role_ids)
            if not rs:
                raise ValueError("role_ids must be None or non-empty")
            self.role_ids = rs
            self.register_buffer(
                "_roles_t",
                torch.tensor(rs, dtype=torch.long),
                persistent=False,
            )

    def extra_repr(self) -> str:
        return f"layers={self.layers}, role_ids={self.role_ids}"


class AttentionImpl(Enum):
    EAGER = "eager"
    FLASH = "flash"
    SDPA = "sdpa"


class ModelParams(BaseModel):
    dim: int = 4096
    head_dim: Optional[int] = None
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float = 8 / 3  # re-defined from original impl
    tie_word_embeddings: bool = False
    norm_eps: float = 1e-5
    max_seq_len: int = 4096
    attn_proj_bias: bool = False
    rope_theta: float = 10000.0
    rope_scaling: dict = {}
    use_role_embeddings: bool = False
    role_embeddings_init: str = "zeros"
    olmo2_arch: bool = False
    gpt_oss_arch: bool = False
    # Qwen3-style per-head QK-norm: RMSNorm(head_dim) applied to xq/xk after
    # the .view() reshape into (..., n_heads, head_dim) and before RoPE. Norm
    # weights broadcast across heads. Different from OLMo2's pre-reshape
    # per-layer QK-norm (handled via networks_olmo.py).
    qk_norm_per_head: bool = False
    attention_impl: AttentionImpl = AttentionImpl.FLASH

    # gpt-oss MoE parameters
    gpt_oss_num_experts: int = 128
    gpt_oss_experts_per_token: int = 4
    gpt_oss_intermediate_size: int = 2880
    gpt_oss_swiglu_limit: float = 7.0
    gpt_oss_sliding_window: int = 128
    gpt_oss_initial_context_length: int = 4096
    gpt_oss_rope_scaling_factor: float = 32.0
    gpt_oss_rope_ntk_alpha: float = 1.0
    gpt_oss_rope_ntk_beta: float = 32.0

    model_config = {"extra": "ignore"}

    def model_post_init(self, __context):
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.attention_impl == AttentionImpl.FLASH and not flash_attn_func:
            self.attention_impl = AttentionImpl.SDPA


class RoleEmbeddings(nn.Embedding):
    def __init__(self, params: ModelParams):
        self.init = params.role_embeddings_init
        super().__init__(len(Role), params.dim)

        self.reset_parameters()

    def reset_parameters(self):
        if self.init == "zeros":
            nn.init.zeros_(self.weight)
        elif self.init.startswith("gaussian"):
            std = float(self.init.split(":")[1])
            nn.init.normal_(self.weight, std=std)
        else:
            raise ValueError(f"Unknown role embeddings init: {self.init}")


def _bias_or_norm(name):
    name = name.lower()

    if "bias" in name or "norm" in name or "ln" in name:
        return True

    return False


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        nn.init.ones_(self.weight.data)

    def forward(self, x):
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.weight


# from: https://github.com/pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        factor: int = 1,
        high_freq_factor: int = 1,
        low_freq_factor: int = 1,
        original_max_position_embeddings: int = 8192,
        rope_type: str = "default",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_type = rope_type
        self.factor = factor
        self.high_freq_factor = high_freq_factor
        self.low_freq_factor = low_freq_factor
        self.old_context_len = original_max_position_embeddings
        self.rope_init()

    # TODO: delete this once all our recipes are moved off of FSDP1 since we
    # no longer need to explicitly name our param init method reset_parameters
    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        # From: https://github.com/huggingface/transformers/blob/37ea04013b34b39c01b51aeaacd8d56f2c62a7eb/src/transformers/modeling_rope_utils.py#L310
        if self.rope_type == "llama3":
            low_freq_wavelen = self.old_context_len / self.low_freq_factor
            high_freq_wavelen = self.old_context_len / self.high_freq_factor

            wavelen = 2 * math.pi / theta
            # wavelen < high_freq_wavelen: do nothing
            # wavelen > low_freq_wavelen: divide by factor
            theta = torch.where(wavelen > low_freq_wavelen, theta / self.factor, theta)
            # otherwise: interpolate between the two, using a smooth factor
            smooth_factor = (self.old_context_len / wavelen - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * theta / self.factor + smooth_factor * theta
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(
                wavelen > low_freq_wavelen
            )
            theta = torch.where(is_medium_freq, smoothed_inv_freq, theta)
        elif self.rope_type != "default":
            raise ValueError(f"Unknown RoPE type: {self.rope_type}")

        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


def _eager_attention(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    output_scores: bool = False,
):
    bsz, qlen, n_heads, head_dim = xq.shape
    _, seqlen, n_kv_heads, _ = xk.shape
    dim = n_heads * head_dim

    # manual implementation of scaled dot product attention
    xq = torch.einsum("bshd->bhsd", xq)
    xk = torch.einsum("bshd->bhsd", xk)
    xv = torch.einsum("bshd->bhsd", xv)

    if n_heads != n_kv_heads:
        kv_repeats = n_heads // n_kv_heads
        xk = xk.repeat_interleave(kv_repeats, dim=1)
        xv = xv.repeat_interleave(kv_repeats, dim=1)

    if attn_mask is None:
        attn_mask = utils._make_causal_mask(
            (bsz, qlen),
            xq.dtype,
            device=xq.device,
            past_key_values_length=seqlen - qlen,
        )
    else:
        attn_mask = utils.to_4d_and_causal(attn_mask, qlen, xq.dtype, seqlen)

    scores = torch.einsum("bhqd,bhkd->bhqk", xq, xk) / math.sqrt(head_dim)

    scores = scores + attn_mask
    weights = F.softmax(scores, dim=-1)

    output = torch.einsum("bhqk,bhkd->bqhd", weights, xv)
    output = output.reshape(bsz, qlen, dim)

    if output_scores:
        return output, scores

    return output


def _flash_attention(xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor):
    bsz, qlen, n_heads, head_dim = xq.shape
    dim = n_heads * head_dim

    output = flash_attn_func(xq, xk, xv, causal=True)
    output = output.reshape(bsz, qlen, dim)
    return output


def _sdpa_attention(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
):
    bsz, qlen, n_heads, head_dim = xq.shape
    _, seqlen, n_kv_heads, _ = xk.shape
    dim = n_heads * head_dim

    xq = torch.einsum("bshd->bhsd", xq)
    xk = torch.einsum("bshd->bhsd", xk)
    xv = torch.einsum("bshd->bhsd", xv)

    if attn_mask is not None:
        # convert 2d to 4d and apply causal mask
        attn_mask = utils.to_4d_and_causal(attn_mask, qlen, xq.dtype, seqlen)
        is_causal = False
    else:
        if seqlen == qlen:
            is_causal = True
        elif qlen == 1:
            # Decode step: the single query is the newest token and may
            # attend to every cached key/value position.
            is_causal = False
        else:
            # Partial-cache prefill: qlen is the uncached suffix length while
            # seqlen includes the loaded prefix. SDPA's is_causal=True assumes
            # q and k have equal lengths, so build the shifted causal mask
            # explicitly.
            attn_mask = utils._make_causal_mask(
                (bsz, qlen),
                xq.dtype,
                device=xq.device,
                past_key_values_length=seqlen - qlen,
            )
            is_causal = False

    output = torch.nn.functional.scaled_dot_product_attention(
        xq,
        xk,
        xv,
        attn_mask=attn_mask,
        is_causal=is_causal,
        enable_gqa=n_kv_heads != n_heads,
    )
    output = torch.einsum("bhsd->bshd", output)
    output = output.reshape(bsz, qlen, dim)
    return output


class Attention(nn.Module):
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

        if params.qk_norm_per_head:
            # Qwen3-style: RMSNorm(head_dim) applied per-head after reshape.
            self.q_norm = RMSNorm(params.head_dim, eps=params.norm_eps)
            self.k_norm = RMSNorm(params.head_dim, eps=params.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[PagedKVPool] = None,
        attn_mask: Optional[torch.Tensor] = None,
        paged_ctx: Optional["PagedContext"] = None,  # noqa: F821
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.params.n_heads, self.params.head_dim)
        xk = xk.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)
        xv = xv.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq = self.rope(xq, input_pos=input_pos)
        xk = self.rope(xk, input_pos=input_pos)

        if cache is None:
            # Training / no-cache inference: FA/SDPA/eager over raw QKV.
            # attn_mask flows through here; FA path asserts None.
            if self.params.attention_impl == AttentionImpl.FLASH:
                assert attn_mask is None
                output = _flash_attention(xq, xk, xv)
            elif self.params.attention_impl == AttentionImpl.SDPA:
                output = _sdpa_attention(xq, xk, xv, attn_mask)
            else:
                output = _eager_attention(xq, xk, xv, attn_mask)
        else:
            # Paged path — single codepath for prefill + decode via
            # flashinfer's batched-paged wrappers. Per-row diverging KV
            # lengths are handled by ``kv_indptr`` / ``kv_last_page_len``
            # inside the planned wrapper, not by kernel-level cache_seqlens.
            # ``paged_ctx`` carries the pre-planned wrapper + the layout
            # used for both the KV scatter and the attention run.
            if paged_ctx is None:
                raise RuntimeError(
                    "Paged cache requires paged_ctx; Transformer.forward "
                    "must build it before the layer loop."
                )
            n_kv = self.params.n_kv_heads
            n_q = self.params.n_heads
            d = self.params.head_dim
            k_flat = xk.reshape(bsz * seqlen, n_kv, d)
            v_flat = xv.reshape(bsz * seqlen, n_kv, d)
            cache.append_kv(self.layer_id, k_flat, v_flat, paged_ctx.layout)
            q_flat = xq.reshape(bsz * seqlen, n_q, d)
            out_flat = paged_ctx.run(self.layer_id, q_flat)
            output = out_flat.reshape(bsz, seqlen, n_q * d)

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

        xq = xq.view(bsz, seqlen, self.params.n_heads, self.params.head_dim)
        xk = xk.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)
        xv = xv.view(bsz, seqlen, self.params.n_kv_heads, self.params.head_dim)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

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


class FeedForward(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        hidden_dim = int(params.ffn_dim_multiplier * params.dim)
        hidden_dim = params.multiple_of * (
            (hidden_dim + params.multiple_of - 1) // params.multiple_of
        )

        self.w1 = nn.Linear(params.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, params.dim, bias=False)
        self.w3 = nn.Linear(params.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TiedLinear(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = [embedding]

    def forward(self, x):
        return nn.functional.linear(x, self.embedding[0].weight)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, params: ModelParams):
        super().__init__()
        self.attention = Attention(layer_id, params)
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
        cache: Optional[PagedKVPool] = None,
        paged_ctx: Optional["PagedContext"] = None,  # noqa: F821
    ):
        h = x + self.attention(
            x=self.attention_norm(x),
            role_ids=role_ids,
            attn_mask=attn_mask,
            input_pos=input_pos,
            cache=cache,
            paged_ctx=paged_ctx,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward_scores(
        self,
        x: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        h, scores = self.attention.forward_scores(
            x=self.attention_norm(x),
            role_ids=role_ids,
            attn_mask=attn_mask,
            input_pos=input_pos,
        )
        h = x + h
        out = h + self.ffn_norm(self.feed_forward(h))
        return out, scores


class InterventionHost:
    """Mixin providing the activation-intervention API.

    Host class contract:

    1. Inherit from ``nn.Module`` in addition to ``InterventionHost``
       (MRO: ``class X(InterventionHost, nn.Module)``).
    2. In ``__init__``, set ``self.interventions = nn.ModuleList()``.
    3. Expose ``self.tok_embeddings`` — the embedding's ``weight.dtype``
       and ``weight.device`` are used to place newly-registered modules.
    4. Call ``self._apply_interventions(h, layer_id=i, role_ids=role_ids)``
       after each transformer block in the forward pass.

    Shared by :class:`Transformer` and
    :class:`torchllms.models.networks_gptoss.GptOSSTransformer` so the
    intervention machinery (role-filter mask + layer dispatch + the
    ``AddVec`` + ``Intervention`` wrappers + driver reinstall guidance)
    lives in one place.
    """

    def register_intervention(
        self,
        module: nn.Module,
        *,
        layers: int | Sequence[int],
        role_ids=None,
    ) -> None:
        """Install a delta-producing module as an activation intervention.

        At each post-block site in ``layers``, the host's forward loop
        computes ``hidden = hidden + mask * module(hidden)``. When
        ``role_ids`` is a set of role IDs, ``mask`` is 1 at positions
        whose role is in that set and 0 elsewhere. When
        ``role_ids is None``, the intervention fires unconditionally.

        Multiple interventions compose in registration order.

        The module is moved to the host's current device/dtype (tracked
        via ``self.tok_embeddings.weight``) and its parameters are marked
        ``requires_grad=False`` — these are inference-time interventions,
        not trainable params of the host model.

        Changing the installed set of interventions invalidates any
        ``torch.compile`` cache held by a driver (e.g.
        ``LLM._compiled_decode_model`` + ``_compiled_prefill_model``);
        reinstall compile after register/clear.
        """
        target_dtype = self.tok_embeddings.weight.dtype
        target_device = self.tok_embeddings.weight.device
        module.to(device=target_device, dtype=target_dtype)
        for p in module.parameters():
            p.requires_grad_(False)
        iv = Intervention(module, layers=layers, role_ids=role_ids)
        iv.to(device=target_device)
        self.interventions.append(iv)

    def clear_interventions(self) -> None:
        """Remove all registered interventions."""
        self.interventions = nn.ModuleList()

    def list_interventions(self) -> List["Intervention"]:
        return list(self.interventions)

    def intervened_roles(self) -> Optional[Set[int]]:
        """Union of role IDs targeted by any registered intervention.

        Returns:
          - ``None`` — at least one intervention matches every role (its
            ``role_ids is None``). Every position is potentially
            intervened.
          - ``set()`` — no interventions registered.
          - ``set[int]`` — union of intervened role IDs.
        """
        if len(self.interventions) == 0:
            return set()
        roles: Set[int] = set()
        for iv in self.interventions:
            if iv.role_ids is None:
                return None
            roles.update(iv.role_ids)
        return roles

    def _apply_interventions(
        self,
        hidden: torch.Tensor,
        *,
        layer_id: int,
        role_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Post-block dispatch. Runs each registered intervention whose
        ``layers`` tuple includes ``layer_id``. Role filtering is
        framework-side via a boolean mask on ``role_ids``; the user
        module only computes a delta given ``hidden``.
        """
        if len(self.interventions) == 0:
            return hidden
        for iv in self.interventions:
            if layer_id not in iv.layers:
                continue
            delta = iv.module(hidden)
            if iv.role_ids is None:
                hidden = hidden + delta
                continue
            if role_ids is None:
                # Intervention wants role filtering but the caller didn't
                # supply role_ids. Skip — safer than applying unmasked.
                continue
            roles_t = iv._roles_t.to(device=role_ids.device)
            mask = (role_ids.unsqueeze(-1) == roles_t.view(1, 1, -1)).any(dim=-1)
            mask = mask.to(hidden.dtype).unsqueeze(-1)
            hidden = hidden + mask * delta
        return hidden


class Transformer(InterventionHost, nn.Module):
    @classmethod
    def from_params(cls, params: ModelParams):
        if params.gpt_oss_arch:
            from torchllms.models.networks_gptoss import GptOSSTransformer
            return GptOSSTransformer(params)
        elif params.olmo2_arch:
            from torchllms.models.networks_olmo import OLMo2Transformer
            return OLMo2Transformer(params)
        else:
            return cls(params)

    def __init__(self, params: ModelParams):
        super().__init__()
        self.params = params
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        if params.use_role_embeddings:
            self.role_embeddings = RoleEmbeddings(params)
        else:
            self.role_embeddings = None

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        if params.tie_word_embeddings:
            self.output = lambda x: nn.functional.linear(x, self.tok_embeddings.weight)
        else:
            self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.interventions = nn.ModuleList()

    def _build_forward_context(self, cache: PagedKVPool, qlens: Sequence[int]):
        """Build the paged attention context for one forward. Shared by
        eager + the ``LLM._model_forward`` compile hoist.

        ``qlens`` is the per-row new-query-token count (list, one entry
        per active rollout). For uniform-length forwards (current
        decode + traditional ``[B, S]`` prefill), callers pass
        ``[S] * B``. For the flat-packed variable-length prefill path
        (``forward(qlens=...)``), callers pass the true per-row qlens.

        Returns ``(pre_write_seqlens, paged_ctx)``. Subclasses override
        to return a different ``*Context`` type (e.g. gpt-oss's
        sink-aware variant); the driver treats both uniformly as
        "pre-built attention handle".
        """
        from torchllms.models.paged_attention import (
            build_paged_context_for_forward,
        )
        active_rids = cache.active_rollouts()
        qlens_list = list(qlens)
        if len(qlens_list) != len(active_rids):
            raise ValueError(
                f"_build_forward_context: qlens len={len(qlens_list)} "
                f"!= active_rids len={len(active_rids)}"
            )
        head_dim = self.params.head_dim
        return build_paged_context_for_forward(
            pool=cache, active_rids=active_rids, qlens=qlens_list,
            n_heads=self.params.n_heads, n_kv_heads=self.params.n_kv_heads,
            head_dim=head_dim,
            sm_scale=head_dim ** -0.5,
            dtype=self.tok_embeddings.weight.dtype,
            device=cache.device,
            window_left=-1,
        )

    def init_cache(
        self,
        max_batch_size: int,
        device: str,
        max_cache_len: Optional[int] = None,
        *,
        page_size: int = 16,
        kv_memory_gb: Optional[float] = None,
    ) -> PagedKVPool:
        """Allocate a PagedKVPool for this model.

        Sizing rules:
        - ``kv_memory_gb`` (explicit): pool holds ``floor(kv_memory_gb * 2^30
          / bytes_per_page)`` pages. Use when you have a concrete VRAM
          budget in mind.
        - Default (``kv_memory_gb=None``): exactly the ``max_batch_size *
          max_cache_len`` token footprint — matches the KVArena memory
          envelope Phase 1 replaces. Pass ``kv_memory_gb`` explicitly if
          you want headroom for radix prefix-cache sharing (Phase 2).

        ``page_size=16`` is SGLang's default and flashinfer's documented
        sweet spot. Increase for longer contexts where plan-side overhead
        dominates.
        """
        max_seqlen = max_cache_len or self.params.max_seq_len
        dtype = self.tok_embeddings.weight.dtype
        element_size = torch.empty(0, dtype=dtype).element_size()
        bytes_per_token_kv = (
            self.params.n_layers * self.params.n_kv_heads
            * self.params.head_dim * element_size * 2  # K + V
        )
        if kv_memory_gb is None:
            total_bytes = max_batch_size * max_seqlen * bytes_per_token_kv
        else:
            total_bytes = int(kv_memory_gb * (1 << 30))
        bytes_per_page = page_size * bytes_per_token_kv
        total_pages = max(int(total_bytes // bytes_per_page), 1)
        return PagedKVPool(
            n_layers=self.params.n_layers,
            total_pages=total_pages,
            page_size=page_size,
            n_kv_heads=self.params.n_kv_heads,
            head_dim=self.params.head_dim,
            max_bsz=max_batch_size,
            device=device,
            dtype=dtype,
        )

    def get_wd_params(self):
        wd_params = [
            p
            for n, p in self.named_parameters()
            if p.requires_grad and not _bias_or_norm(n)
        ]
        no_wd_params = [
            p
            for n, p in self.named_parameters()
            if p.requires_grad and _bias_or_norm(n)
        ]
        return wd_params, no_wd_params

    def forward(
        self,
        input_ids: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[PagedKVPool] = None,
        logits_to_keep: Optional[int] = None,
        paged_ctx: Optional["PagedContext"] = None,  # noqa: F821
        qlens: Optional[Sequence[int]] = None,
    ):
        """
        Apply a single forward pass for training or inference (prefill + decoding).

        Two supported input layouts:

        1. **Uniform** (``qlens is None``) — ``input_ids`` is ``[B, S]``
           with all rows sharing the same S. Canonical layout for
           decode (S=1), single-prompt generation, and training /
           scoring. This path is cudagraph-captureable and engages the
           prefill/decode compile caches.

        2. **Flat-packed** (``qlens`` provided) — ``input_ids`` is
           ``[1, sum(qlens)]`` with the per-row segmentation given by
           ``qlens``. Enables batched prefill over rows of heterogeneous
           prompt lengths without padding or per-group temp-cache
           round-trip. ``role_ids`` must match the flat layout if
           supplied (shape ``[1, sum(qlens)]``). Logits-to-keep=1 in
           this mode returns ``[B, 1, V]`` gathered at each row's
           last-token position.

        Args:
            input_ids: ``[B, S]`` uniform, or ``[1, sum(qlens)]`` flat.
            role_ids: same shape as ``input_ids``. Threaded to registered
                interventions for role-filter mask construction without
                mutation.
            attn_mask: ``[B, S]``. Used only in the no-cache path
                (training / scoring) by SDPA / eager attention. The
                paged-cache path ignores it.
            input_pos: position IDs for RoPE. When not provided and a
                cache is present, computed from the pool's per-rollout
                pre-write seqlens — uniform layout gets per-row
                ``arange(S) + pre_write``; flat layout gets each row's
                positions concatenated.
            cache: optional :class:`PagedKVPool`. Number of active
                rollouts must match B (uniform) or len(qlens) (flat).
            logits_to_keep: keep only the last N positions of logits
                (generation uses 1; training / scoring leaves unset).
                In flat-pack mode only ``logits_to_keep=1`` is supported
                (gathers each row's last-position logit).
            qlens: per-row new-query-token counts; activates flat layout.

        Returns:
            (logits, cache) — cache mutated in place.
        """
        # Determine logical batch structure.
        if qlens is None:
            B, S = input_ids.shape
            qlens_eff = [S] * B
        else:
            qlens_eff = list(qlens)
            B = len(qlens_eff)
            total = sum(qlens_eff)
            if input_ids.shape != (1, total):
                raise RuntimeError(
                    f"flat-pack forward expects input_ids shape (1, {total}), "
                    f"got {tuple(input_ids.shape)}"
                )

        # paged_ctx is pre-built by the driver for the compile-hoist
        # paths (``LLM._model_forward``) — plan() has CPU-sync ops that
        # fragment cudagraph partitions, so it must run OUTSIDE the
        # compiled region. Eager callers (including all flat-pack
        # callers) get the build done inline here via
        # ``_build_forward_context``.
        if cache is not None and paged_ctx is None:
            active_rids = cache.active_rollouts()
            if len(active_rids) != B:
                raise RuntimeError(
                    f"cache has {len(active_rids)} live rollouts but forward "
                    f"received batch={B}"
                )
            pre_write, paged_ctx = self._build_forward_context(cache, qlens_eff)
            if input_pos is None:
                if qlens is None:
                    # Uniform: arange broadcasts over rows.
                    S = qlens_eff[0]
                    input_pos = (
                        torch.arange(
                            S, dtype=torch.int32, device=input_ids.device,
                        )[None, :]
                        + pre_write[:, None]
                    )
                else:
                    # Flat-packed: concatenate per-row positions.
                    pos_parts = []
                    pre_list = pre_write.tolist()
                    for b in range(B):
                        q = qlens_eff[b]
                        pre = pre_list[b]
                        pos_parts.append(
                            torch.arange(
                                pre, pre + q,
                                dtype=torch.int32, device=input_ids.device,
                            )
                        )
                    input_pos = torch.cat(pos_parts).unsqueeze(0)

        h = self.tok_embeddings(input_ids)

        if self.role_embeddings is not None and role_ids is not None:
            h += self.role_embeddings(role_ids)

        active_role_ids = role_ids

        for i, layer in enumerate(self.layers):
            h = layer(h, role_ids, attn_mask, input_pos, cache, paged_ctx)
            h = self._apply_interventions(
                h, layer_id=i, role_ids=active_role_ids,
            )

        if logits_to_keep is not None:
            if qlens is None:
                h = h[:, -logits_to_keep:]
            else:
                if logits_to_keep != 1:
                    raise NotImplementedError(
                        "logits_to_keep != 1 is not supported in flat-pack "
                        "mode (qlens provided)"
                    )
                # Per-row last-position gather. cumsum(qlens) - 1 gives
                # the flat index of each row's final token.
                cumsum = (
                    torch.tensor(qlens_eff, dtype=torch.int64, device=h.device)
                    .cumsum(0) - 1
                )
                h = h[0].index_select(0, cumsum).unsqueeze(1)  # [B, 1, D]

        logits = self.output(self.norm(h))

        return logits, cache

    def forward_scores(
        self,
        input_ids: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if attn_mask is not None:
            input_pos = torch.cumsum(attn_mask, dim=1) - 1
        else:
            input_pos = None

        h = self.tok_embeddings(input_ids)

        if self.role_embeddings is not None and role_ids is not None:
            h += self.role_embeddings(role_ids)

        all_scores = []
        for i, layer in enumerate(self.layers):
            h, scores = layer.forward_scores(h, role_ids, attn_mask, input_pos)
            all_scores.append(scores)

        logits = self.output(self.norm(h))

        return logits, all_scores
