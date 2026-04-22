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
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# The custom triton attention kernel in torchllms.models.attention_gptoss
# is superseded by flashinfer's BatchAttentionWithAttentionSinkWrapper
# (see torchllms.models.flashinfer_attention). attention_ref is still kept
# as a dense reference used by smoke / test scripts but is not called
# inside the production forward path.
from torchllms.models.networks import (
    InterventionHost,
    ModelParams,
    RMSNorm,
    RoleEmbeddings,
)
from torchllms.models.paged_kv import PagedKVPool


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
    """SwiGLU activation with clamping, as used in gpt-oss.

    Eager-only reference. Production MoE runs through
    ``triton_kernels.swiglu.swiglu_fn`` fused inside ``matmul_ogs``.
    """
    x_glu = x[..., ::2].clamp(max=limit)
    x_linear = x[..., 1::2].clamp(min=-limit, max=limit)
    return x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + 1)


# SM120 (RTX 5090 / desktop Blackwell) matmul_ogs constraints. SM120
# doesn't support the persistent/TMA MXFP4 path — the kernel falls back
# to StridedLayout + non-persistent tiling with block_k=128 to stay
# inside per-block shared-memory budget. Mirrors SGLang's
# ``sglang/srt/layers/quantization/mxfp4.py::_swizzle_mxfp4`` SM120
# branch. Applied once per process on first MoE weight load.
_SM120_OPT_FLAGS_APPLIED = False


def _apply_sm120_opt_flags() -> None:
    global _SM120_OPT_FLAGS_APPLIED
    if _SM120_OPT_FLAGS_APPLIED:
        return
    try:
        import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    except ImportError:
        return
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
        opt_flags.update_opt_flags_constraints({
            "is_persistent": False,
            "block_k": 128,
            "num_stages": 1,
        })
    # ``allow_in_graph(matmul_ogs)`` + ``allow_in_graph(routing)`` were
    # tried first but fail: Dynamo still fake-tensor-traces into the
    # ``routing`` Python wrapper before calling the leaf, and the
    # fake tensors hit ``.data_ptr()`` inside triton_kernels'
    # compaction logic — which is undefined on fake tensors. The
    # working shape is to wrap the entire MoE forward in a single
    # ``@torch.library.custom_op`` boundary (see
    # ``gptoss_moe_delta`` below + ``GptOSSMoE.forward``). The custom
    # op is opaque to Dynamo; its body runs after the trace and calls
    # into triton_kernels with real tensors.
    _SM120_OPT_FLAGS_APPLIED = True


# ---------------------------------------------------------------------------
# MoE forward — one Dynamo-opaque custom op per layer
# ---------------------------------------------------------------------------
#
# The triton_kernels entry points (``matmul_ogs``, ``routing``) call
# ``.data_ptr()`` during their Python prologue for shape / launch-grid
# inference. Dynamo's fake-tensor tracing route into those Python
# wrappers and crashes on the bare ``.data_ptr()`` call. The
# ``allow_in_graph`` attempt above was insufficient because the
# triggering calls happen BEFORE the leaf function — the Python
# wrapper itself fake-traces.
#
# Workaround: expose the entire MoE forward as a single
# ``torch.library.custom_op``. Dynamo sees the op as one opaque
# node; its body runs eagerly after tracing and is free to call
# triton_kernels with real tensors. ``register_fake`` gives Dynamo
# the output shape without running any real computation.
#
# Non-tensor state (the triton_kernels wrapped weights, precision
# configs) is looked up via a module-global layer registry keyed by
# ``layer_id`` (specialized as a graph constant by Dynamo, so each
# layer gets its own graph specialization). Load-time routines
# ``load_mxfp4_weights`` / ``load_bf16_weights`` register the MoE
# into the registry so the op body can find it.
_MOE_LAYER_REGISTRY: List[Optional["GptOSSMoE"]] = []


def _register_moe_layer(layer_id: int, moe: "GptOSSMoE") -> None:
    while len(_MOE_LAYER_REGISTRY) <= layer_id:
        _MOE_LAYER_REGISTRY.append(None)
    _MOE_LAYER_REGISTRY[layer_id] = moe


@torch.library.custom_op("torchllms::gptoss_moe_delta", mutates_args=())
def _gptoss_moe_delta(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    """Compute MoE output (pre-residual delta) for one layer.

    The residual add ``x + delta`` is done by the caller outside the
    op boundary so Dynamo sees the add as a standard tensor op.
    """
    moe = _MOE_LAYER_REGISTRY[layer_id]
    if moe is None:
        raise RuntimeError(
            f"gpt-oss MoE layer {layer_id} not registered; "
            "load_mxfp4_weights / load_bf16_weights must run first"
        )
    return moe._compute_delta(x)


@_gptoss_moe_delta.register_fake
def _(x: torch.Tensor, layer_id: int) -> torch.Tensor:
    """Fake-tensor shape inference for the MoE delta op.

    Output has the same shape as the input residual stream: the
    per-layer MoE maps ``[B, S, D] -> [B, S, D]``.
    """
    return torch.empty_like(x)


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
        cache: Optional[PagedKVPool] = None,
        attn_mask: Optional[torch.Tensor] = None,
        sink_ctx=None,
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

        if cache is None:
            raise RuntimeError(
                "GptOSSAttention.forward requires a PagedKVPool cache "
                "(flashinfer paged attention has no dense-KV variant)"
            )
        if sink_ctx is None:
            raise RuntimeError(
                "GptOSSAttention.forward requires sink_ctx from "
                "GptOSSTransformer.forward's build_sink_context call"
            )

        # Write new K/V into the paged pool via the planned layout.
        k_flat = k.reshape(bsz * seqlen, self.n_kv_heads, self.head_dim)
        v_flat = v.reshape(bsz * seqlen, self.n_kv_heads, self.head_dim)
        cache.append_kv(self.layer_id, k_flat, v_flat, sink_ctx.layout)

        # Run the sink attention kernel. Q is flat ``[B*S, n_heads,
        # head_dim]``; the wrapper reads the paged K/V slabs via the
        # block table baked into its plan() state.
        q_flat = q.reshape(bsz * seqlen, self.n_heads, self.head_dim)
        output_flat = sink_ctx.run(
            self.layer_id, q_flat, self.sinks, self.sliding_window,
        )
        output = output_flat.reshape(bsz, seqlen, self.n_heads * self.head_dim)
        return self.out(output)


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
        self.swiglu_alpha = 1.702  # gpt-oss constant
        self.hidden_size = params.dim
        self.intermediate_size = params.gpt_oss_intermediate_size

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.gate = nn.Linear(
            params.dim, self.num_experts, bias=True, dtype=torch.bfloat16,
        )

        # Biases stored as fp32 — matmul_ogs requires fp32 bias to
        # match its internal fp32 accumulator (mixed-precision accumulate
        # across experts). Storage cost is negligible: 32×5760×4 bytes ≈
        # 720KB for mlp1, 32×2880×4 ≈ 360KB for mlp2. Checkpoint loader
        # copies the bf16 checkpoint biases in with an implicit cast.
        self.mlp1_bias = nn.Parameter(
            torch.empty(
                self.num_experts, self.intermediate_size * 2, dtype=torch.float32,
            )
        )
        self.mlp2_bias = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=torch.float32)
        )

        # Weight storage. ``load_mxfp4_weights`` or ``load_bf16_weights``
        # populates the triton_kernels side:
        #   self._w1_triton  / self._w2_triton : triton_kernels Tensor
        #     (FP4-wrapped or bf16-wrapped); fed as ``w`` to matmul_ogs.
        #   self._w1_pcg / self._w2_pcg : PrecisionConfig (carries the
        #     MXFP4 scale tensor when quantized, empty for bf16).
        #
        # The raw checkpoint tensors are kept as registered buffers /
        # Parameters so state-dict save/load round-trips are unchanged.
        # The triton_kernels wrappers are stored as regular attributes
        # (not buffers): they are custom ``triton_kernels.tensor.Tensor``
        # objects, not ``torch.Tensor``, and nn.Module buffer machinery
        # rejects non-torch tensors.
        self.mlp1_weight: Optional[nn.Parameter] = None
        self.mlp2_weight: Optional[nn.Parameter] = None
        self._w1_triton = None
        self._w2_triton = None
        self._w1_pcg = None
        self._w2_pcg = None

    # ------------------------------------------------------------------
    # Weight loading: once at model-init, swizzle into triton_kernels layout
    # ------------------------------------------------------------------

    @staticmethod
    def _swizzle_mxfp4(blocks: torch.Tensor, scales: torch.Tensor):
        """Swizzle MXFP4 packed weights for triton_kernels.matmul_ogs.

        Checkpoint stores ``blocks`` as ``[E, R, G, B=16]`` uint8 with
        ``R = 2I`` (for mlp1) or ``H`` (for mlp2) and ``G * B * 2 = K``
        the contraction dim. This reshape+transpose step lands in the
        layout matmul_ogs expects for ``w: [E, K_packed, N]`` where
        ``K_packed = K // 2`` (two FP4 nibbles per byte, unpacked by
        the kernel via ``wrap_torch_tensor(..., dtype=FP4)``).

        Output is a triton_kernels ``Tensor`` (not a ``torch.Tensor``).
        """
        from triton_kernels.tensor import FP4, wrap_torch_tensor, convert_layout
        from triton_kernels.tensor_details.layout import StridedLayout

        E, R, G, B = blocks.shape
        K_packed = G * B  # K // 2
        # [E, R, G, B] → [E, R, K_packed] → [E, K_packed, R]
        # No .contiguous() after transpose: wrap_torch_tensor looks for
        # stride==1 to pick the axis FP4-unpacking doubles, and the
        # (pre-transpose) innermost dim is what we want doubled so the
        # logical shape becomes [E, K=H, N=2I]. A .contiguous() call
        # would move stride==1 to the trailing axis and double the
        # wrong one. SGLang's swizzle does the same transpose without
        # .contiguous().
        packed = blocks.reshape(E, R, K_packed).transpose(-2, -1)
        scales_t = scales.transpose(-2, -1)  # [E, G, R]

        w = convert_layout(
            wrap_torch_tensor(packed, dtype=FP4), StridedLayout,
        )
        s = convert_layout(wrap_torch_tensor(scales_t), StridedLayout)
        return w, s

    def _promote_biases_to_fp32(self) -> None:
        """Checkpoint biases are bf16; matmul_ogs requires fp32 bias.
        Called after ``load_state_dict(assign=True)`` has stamped the
        raw checkpoint bf16 tensors onto the bias parameters."""
        if self.mlp1_bias.dtype != torch.float32:
            self.mlp1_bias.data = self.mlp1_bias.data.to(torch.float32)
        if self.mlp2_bias.dtype != torch.float32:
            self.mlp2_bias.data = self.mlp2_bias.data.to(torch.float32)

    def load_mxfp4_weights(
        self,
        mlp1_blocks: torch.Tensor,
        mlp1_scales: torch.Tensor,
        mlp2_blocks: torch.Tensor,
        mlp2_scales: torch.Tensor,
    ):
        """Swizzle MXFP4 expert weights for triton_kernels matmul_ogs."""
        from triton_kernels.matmul_ogs import PrecisionConfig

        self.register_buffer("_mlp1_blocks", mlp1_blocks)
        self.register_buffer("_mlp1_scales", mlp1_scales)
        self.register_buffer("_mlp2_blocks", mlp2_blocks)
        self.register_buffer("_mlp2_scales", mlp2_scales)

        _apply_sm120_opt_flags()
        self._promote_biases_to_fp32()
        self._w1_triton, w1_scale = self._swizzle_mxfp4(mlp1_blocks, mlp1_scales)
        self._w2_triton, w2_scale = self._swizzle_mxfp4(mlp2_blocks, mlp2_scales)
        self._w1_pcg = PrecisionConfig(weight_scale=w1_scale)
        self._w2_pcg = PrecisionConfig(weight_scale=w2_scale)
        _register_moe_layer(self.layer_id, self)

    def load_bf16_weights(
        self, mlp1_weight: torch.Tensor, mlp2_weight: torch.Tensor,
    ):
        """bf16 expert weights (reference / fallback path).

        Wraps the bf16 parameters into triton_kernels' tensor type so
        matmul_ogs handles both quantized and non-quantized storage
        through one call site. The transpose puts the contraction dim
        last as matmul_ogs expects.
        """
        from triton_kernels.matmul_ogs import PrecisionConfig
        from triton_kernels.tensor import wrap_torch_tensor

        # [E, 2I, H] → [E, H, 2I]  (contraction on H).
        self.mlp1_weight = nn.Parameter(mlp1_weight, requires_grad=False)
        self.mlp2_weight = nn.Parameter(mlp2_weight, requires_grad=False)

        _apply_sm120_opt_flags()
        self._promote_biases_to_fp32()
        w1_t = self.mlp1_weight.transpose(-2, -1).contiguous()  # [E, H, 2I]
        w2_t = self.mlp2_weight.transpose(-2, -1).contiguous()  # [E, I, H]
        self._w1_triton = wrap_torch_tensor(w1_t, dtype=torch.bfloat16)
        self._w2_triton = wrap_torch_tensor(w2_t, dtype=torch.bfloat16)
        self._w1_pcg = PrecisionConfig()
        self._w2_pcg = PrecisionConfig()
        _register_moe_layer(self.layer_id, self)

    # ------------------------------------------------------------------
    # Forward: 1 routing + 2 matmul_ogs calls  (was ~50 Python-dispatched)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Public forward — dispatches the MoE work through a custom op
        boundary so Dynamo sees it as one opaque node, then adds the
        residual in Python.

        Keeping the residual add outside the op means Dynamo can
        reason about it as a normal tensor op (e.g., fuse it into
        Inductor's residual stream planning) — the op body only
        produces the delta.
        """
        return x + _gptoss_moe_delta(x, self.layer_id)

    def _compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """MoE delta for one layer — the actual triton_kernels work.

        Called from the ``gptoss_moe_delta`` custom op body (always in
        eager mode, after Dynamo has stopped tracing). Free to call
        into triton_kernels with real tensors.
        """
        from triton_kernels.matmul_ogs import matmul_ogs, FusedActivation, FnSpecs
        from triton_kernels.routing import routing
        from triton_kernels.swiglu import swiglu_fn

        if self._w1_triton is None:
            raise RuntimeError(
                "GptOSSMoE weights not loaded; call load_mxfp4_weights "
                "or load_bf16_weights first"
            )

        bsz, seqlen, dim = x.shape
        M = bsz * seqlen

        t_flat = self.norm(x).view(M, dim)
        gate_logits = self.gate(t_flat)  # [M, E] bf16

        # Routing: topk + softmax + sort + histogram — one kernel.
        # gpt-oss uses topk-then-softmax, which triton_kernels.routing
        # does by default (``apply_softmax`` inside topk).
        routing_data, gather_indx, scatter_indx = routing(
            gate_logits, self.experts_per_token,
        )

        # Gate-up matmul with fused clamped SwiGLU. matmul_ogs halves
        # the trailing dim via ``(fused_activation, reduction_factor=2)``;
        # the triton_kernels swiglu kernel signature matches gpt-oss's
        # ``alpha * sigmoid(alpha*g) * (l + 1)`` variant.
        act = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")),
            (self.swiglu_alpha, float(self.swiglu_limit)),
            2,
        )
        intermediate = torch.empty(
            (1, M * self.experts_per_token, self.intermediate_size),
            device=x.device, dtype=x.dtype,
        )
        matmul_ogs(
            t_flat, self._w1_triton, self.mlp1_bias,
            routing_data,
            gather_indx=gather_indx,
            precision_config=self._w1_pcg,
            fused_activation=act,
            y=intermediate,
        )

        # Down-projection with weighted scatter. ``gammas=gate_scal``
        # applies the per-(token,expert) routing weights before summing
        # into the output — equivalent to the ``sorted_outputs *
        # sorted_weights`` + ``scatter_add_`` sequence in the old loop.
        output = torch.empty(
            (1, M, dim), device=x.device, dtype=x.dtype,
        )
        matmul_ogs(
            intermediate.view(M * self.experts_per_token, self.intermediate_size),
            self._w2_triton, self.mlp2_bias,
            routing_data,
            scatter_indx=scatter_indx,
            precision_config=self._w2_pcg,
            gammas=routing_data.gate_scal,
            y=output,
        )

        return output.view(bsz, seqlen, dim)


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
        cache: Optional[PagedKVPool] = None,
        sink_ctx=None,  # FlashinferSinkContext; required when cache is not None
    ):
        x = x + self.attn(x, role_ids, input_pos, cache, attn_mask, sink_ctx=sink_ctx)
        x = self.mlp(x)
        return x


class GptOSSTransformer(InterventionHost, nn.Module):
    """GPT-OSS MoE Transformer matching the torchllms Transformer interface.

    Weight names use gpt-oss conventions (embedding, block, unembedding) to
    align with the SafeTensors checkpoint format.

    Inherits the activation-intervention API (``register_intervention`` /
    ``clear_interventions`` / ``intervened_roles`` / ``_apply_interventions``)
    from :class:`torchllms.models.networks.InterventionHost` — same API
    surface as base ``Transformer`` so the Phase-5 hook machinery works
    on both model families.
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
        # Activation-intervention registry. Populated via
        # ``register_intervention``; drained by ``clear_interventions``.
        # The layer loop below calls ``_apply_interventions`` after each
        # transformer block (InterventionHost contract).
        self.interventions = nn.ModuleList()

    def init_cache(
        self,
        max_batch_size: int,
        device: str,
        max_cache_len: Optional[int] = None,
        *,
        page_size: int = 16,
        kv_memory_gb: Optional[float] = None,
    ) -> PagedKVPool:
        """Allocate a PagedKVPool for this model. Same sizing rules as
        ``Transformer.init_cache`` — see that docstring for details."""
        max_seqlen = max_cache_len or self.params.max_seq_len
        dtype = torch.bfloat16
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

    def _build_forward_context(self, cache: PagedKVPool, qlens):
        """Build the sink context for one forward. Shared by eager + the
        compile hoist path in ``LLM._model_forward``.

        ``qlens`` is the per-row new-query-token list (see base
        :meth:`Transformer._build_forward_context`).

        Returns ``(pre_write_seqlens, sink_ctx)`` matching the
        convention used by the base ``Transformer`` hoist. ``sink_ctx``
        is a :class:`FlashinferSinkContext`; from the LLM driver's
        perspective it's the same "paged_ctx" — a pre-planned attention
        handle — just carrying the sink wrappers instead of Qwen3's
        prefill/decode wrappers.
        """
        from torchllms.models.flashinfer_attention import (
            build_sink_context_for_forward,
        )
        active_rids = cache.active_rollouts()
        qlens_list = list(qlens)
        if len(qlens_list) != len(active_rids):
            raise ValueError(
                f"_build_forward_context: qlens len={len(qlens_list)} "
                f"!= active_rids len={len(active_rids)}"
            )
        head_dim = self.params.head_dim
        return build_sink_context_for_forward(
            pool=cache, active_rids=active_rids, qlens=qlens_list,
            n_heads=self.params.n_heads, n_kv_heads=self.params.n_kv_heads,
            head_dim=head_dim,
            sliding_window=self.params.gpt_oss_sliding_window,
            sm_scale=head_dim ** -0.5,
            dtype=self.tok_embeddings.weight.dtype,
            device=cache.device,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        role_ids: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        cache: Optional[PagedKVPool] = None,
        logits_to_keep: Optional[int] = None,
        paged_ctx=None,  # FlashinferSinkContext; pre-built by LLM compile hoist
        qlens=None,
    ):
        """Parallel in shape to :meth:`torchllms.models.networks.Transformer.forward`.

        Supports both uniform ``[B, S]`` and flat-packed
        ``[1, sum(qlens)]`` layouts; see the base Transformer docstring
        for details. ``qlens`` is a per-row new-query-token list; when
        provided, ``input_ids`` and ``role_ids`` are interpreted as
        flat.
        """
        if cache is None:
            raise RuntimeError(
                "GptOSSTransformer.forward requires a PagedKVPool cache; "
                "flashinfer sink attention has no dense-KV variant."
            )

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

        if paged_ctx is None:
            active_rids = cache.active_rollouts()
            if len(active_rids) != B:
                raise RuntimeError(
                    f"cache has {len(active_rids)} live rollouts but forward "
                    f"received batch={B}"
                )
            pre_write, paged_ctx = self._build_forward_context(cache, qlens_eff)
            if input_pos is None:
                if qlens is None:
                    S = qlens_eff[0]
                    input_pos = (
                        torch.arange(
                            S, dtype=torch.int32, device=input_ids.device,
                        )[None, :]
                        + pre_write[:, None]
                    )
                else:
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

        for layer_id, layer in enumerate(self.layers):
            h = layer(h, role_ids, attn_mask, input_pos, cache, paged_ctx)
            # Post-block intervention dispatch. No-op fast path when
            # ``self.interventions`` is empty — zero overhead for runs
            # without registered interventions.
            h = self._apply_interventions(h, layer_id=layer_id, role_ids=role_ids)

        if logits_to_keep is not None:
            if qlens is None:
                h = h[:, -logits_to_keep:]
            else:
                if logits_to_keep != 1:
                    raise NotImplementedError(
                        "logits_to_keep != 1 is not supported in flat-pack "
                        "mode (qlens provided)"
                    )
                cumsum = (
                    torch.tensor(qlens_eff, dtype=torch.int64, device=h.device)
                    .cumsum(0) - 1
                )
                h = h[0].index_select(0, cumsum).unsqueeze(1)  # [B, 1, D]

        logits = self.output(self.norm(h))
        return logits.float(), cache
