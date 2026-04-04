"""
Weight loading for gpt-oss SafeTensors checkpoints with MXFP4 MoE weights.

Handles:
- SafeTensors format loading
- MXFP4 block-based quantized MoE weight decoding to bf16
- Weight name mapping from gpt-oss checkpoint names to torchllms model names
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tiktoken
import torch
import torch.nn as nn
from loguru import logger
from safetensors import safe_open

from torchllms.models.networks import ModelParams, Transformer

# MXFP4 decoding constants
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def decode_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 16384 * 512,
) -> torch.Tensor:
    """Decode MXFP4 block-encoded tensor to bf16.

    MXFP4 packs 32 FP4 values into 16 bytes (2 nibbles per byte).
    Each row of 16 bytes shares a single uint8 scale (biased by 127).

    Args:
        blocks: Packed MXFP4 blocks tensor.
        scales: Per-row uint8 scale factors.
        dtype: Output dtype.
        rows_per_chunk: Process in chunks to limit peak memory.
    Returns:
        Decoded tensor in the specified dtype.
    """
    scales = scales.to(torch.int32) - 127

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]
        torch.ldexp(sub, exp, out=sub)

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


class GptOSSCheckpoint:
    """Loader for gpt-oss SafeTensors checkpoints."""

    def __init__(self, path: str, device: torch.device):
        device_str = (
            device.type
            if device.index is None
            else f"{device.type}:{device.index}"
        )
        self.device_str = device_str

        safetensor_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".safetensors")
        ]
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {path}")

        self.tensor_name_to_file: Dict[str, str] = {}
        for sf in safetensor_files:
            with safe_open(sf, framework="pt", device=device_str) as f:
                for key in f.keys():
                    self.tensor_name_to_file[key] = sf

    def get_tensor(self, name: str) -> torch.Tensor:
        assert name in self.tensor_name_to_file, f"Tensor {name} not found"
        with safe_open(
            self.tensor_name_to_file[name], framework="pt", device=self.device_str
        ) as f:
            return f.get_tensor(name)

    def get_mxfp4_tensor(
        self, blocks_name: str, scales_name: str, dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        blocks = self.get_tensor(blocks_name)
        scales = self.get_tensor(scales_name)
        return decode_mxfp4(blocks, scales, dtype=dtype)

    def has_tensor(self, name: str) -> bool:
        return name in self.tensor_name_to_file


# Mapping from gpt-oss checkpoint param names → torchllms model param names
# gpt-oss uses: embedding, block.{n}.attn.*, block.{n}.mlp.*, norm, unembedding
# torchllms uses: tok_embeddings, layers.{n}.attn.*, layers.{n}.mlp.*, norm, output

def _build_name_map(n_layers: int) -> Dict[str, str]:
    """Build checkpoint name → model name mapping."""
    name_map = {
        "embedding.weight": "tok_embeddings.weight",
        "norm.scale": "norm.weight",
        "unembedding.weight": "output.weight",
    }

    for n in range(n_layers):
        prefix_ckpt = f"block.{n}"
        prefix_model = f"layers.{n}"

        # Attention
        name_map[f"{prefix_ckpt}.attn.norm.scale"] = f"{prefix_model}.attn.norm.weight"
        name_map[f"{prefix_ckpt}.attn.qkv.weight"] = f"{prefix_model}.attn.qkv.weight"
        name_map[f"{prefix_ckpt}.attn.out.weight"] = f"{prefix_model}.attn.out.weight"
        name_map[f"{prefix_ckpt}.attn.sinks"] = f"{prefix_model}.attn.sinks"

        # MLP / MoE
        name_map[f"{prefix_ckpt}.mlp.norm.scale"] = f"{prefix_model}.mlp.norm.weight"
        name_map[f"{prefix_ckpt}.mlp.gate.weight"] = f"{prefix_model}.mlp.gate.weight"
        name_map[f"{prefix_ckpt}.mlp.gate.bias"] = f"{prefix_model}.mlp.gate.bias"
        name_map[f"{prefix_ckpt}.mlp.mlp1_bias"] = f"{prefix_model}.mlp.mlp1_bias"
        name_map[f"{prefix_ckpt}.mlp.mlp2_bias"] = f"{prefix_model}.mlp.mlp2_bias"

        # MoE weights are MXFP4-encoded, handled separately
        # block.{n}.mlp.mlp1_weight.blocks / .scales → layers.{n}.mlp.mlp1_weight
        # block.{n}.mlp.mlp2_weight.blocks / .scales → layers.{n}.mlp.mlp2_weight

    return name_map


def get_gptoss_tokenizer():
    """Get the gpt-oss tiktoken tokenizer (o200k_harmony)."""
    o200k_base = tiktoken.get_encoding("o200k_base")
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        }
        | {f"<|reserved_{i}|>": i for i in range(200013, 201088)},
    )
    return tokenizer


def load_gptoss_config(path: str) -> ModelParams:
    """Load gpt-oss config.json and convert to ModelParams."""
    config_path = os.path.join(path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    return ModelParams(
        dim=cfg["hidden_size"],
        head_dim=cfg.get("head_dim", 64),
        n_layers=cfg["num_hidden_layers"],
        n_heads=cfg["num_attention_heads"],
        n_kv_heads=cfg["num_key_value_heads"],
        vocab_size=cfg.get("vocab_size", 201088),
        norm_eps=cfg.get("norm_eps", 1e-5),
        max_seq_len=cfg.get("max_context_length", 131072),
        rope_theta=cfg.get("rope_theta", 150000.0),
        gpt_oss_arch=True,
        attention_impl="sdpa",
        gpt_oss_num_experts=cfg.get("num_experts", 128),
        gpt_oss_experts_per_token=cfg.get("experts_per_token", 4),
        gpt_oss_intermediate_size=cfg.get("intermediate_size", 2880),
        gpt_oss_swiglu_limit=cfg.get("swiglu_limit", 7.0),
        gpt_oss_sliding_window=cfg.get("sliding_window", 128),
        gpt_oss_initial_context_length=cfg.get("initial_context_length", 4096),
        gpt_oss_rope_scaling_factor=cfg.get("rope_scaling_factor", 32.0),
        gpt_oss_rope_ntk_alpha=cfg.get("rope_ntk_alpha", 1.0),
        gpt_oss_rope_ntk_beta=cfg.get("rope_ntk_beta", 32.0),
    )


def setup_gptoss(
    checkpoint_path: str,
    device: str = "cuda",
    max_seq_len: Optional[int] = None,
    mxfp4: bool = True,
):
    """Load gpt-oss model and tokenizer for inference.

    Args:
        checkpoint_path: Path to checkpoint directory containing config.json
            and .safetensors files.
        device: Target device.
        max_seq_len: Override max sequence length (useful for limiting memory).
        mxfp4: If True (default), keep MoE expert weights in MXFP4 packed
            format and decode on-the-fly during inference (~4x memory savings).
            If False, decode all expert weights to bf16 at load time.
    Returns:
        (model, tokenizer, template_config)
        tokenizer is tiktoken-based (not HuggingFace).
        template_config is None — use openai-harmony for Harmony formatting.
    """
    params = load_gptoss_config(checkpoint_path)
    if max_seq_len is not None:
        params.max_seq_len = max_seq_len

    logger.info(
        f"Loading gpt-oss model: {params.n_layers} layers, "
        f"{params.gpt_oss_num_experts} experts, "
        f"dim={params.dim}, vocab={params.vocab_size}, "
        f"mxfp4={'on' if mxfp4 else 'off'}"
    )

    # Build model on meta device (allocates no memory)
    with torch.device("meta"):
        model = Transformer.from_params(params)

    # Load weights from SafeTensors checkpoint
    ckpt = GptOSSCheckpoint(checkpoint_path, torch.device(device))
    name_map = _build_name_map(params.n_layers)

    # --- Non-MoE weights: load directly into state_dict ---
    state_dict = {}
    for ckpt_name, model_name in name_map.items():
        if ckpt.has_tensor(ckpt_name):
            tensor = ckpt.get_tensor(ckpt_name)
            # gpt-oss stores gate weight as (hidden, experts); nn.Linear needs (experts, hidden)
            if "gate.weight" in ckpt_name and tensor.shape[0] == params.dim:
                tensor = tensor.T.contiguous()
            state_dict[model_name] = tensor

    # Load non-MoE-weight state dict (biases, norms, embeddings, attention, gate)
    result = model.load_state_dict(state_dict, strict=False, assign=True)

    # --- MoE expert weights: load as MXFP4 or bf16 ---
    for n in range(params.n_layers):
        moe_layer = model.layers[n].mlp

        mlp1_blocks_name = f"block.{n}.mlp.mlp1_weight.blocks"
        mlp1_scales_name = f"block.{n}.mlp.mlp1_weight.scales"
        mlp2_blocks_name = f"block.{n}.mlp.mlp2_weight.blocks"
        mlp2_scales_name = f"block.{n}.mlp.mlp2_weight.scales"

        has_mxfp4 = ckpt.has_tensor(mlp1_blocks_name)

        if has_mxfp4 and mxfp4:
            # Keep in packed MXFP4 format (~4x smaller, decode on-the-fly)
            moe_layer.load_mxfp4_weights(
                mlp1_blocks=ckpt.get_tensor(mlp1_blocks_name).to(device),
                mlp1_scales=ckpt.get_tensor(mlp1_scales_name).to(device),
                mlp2_blocks=ckpt.get_tensor(mlp2_blocks_name).to(device),
                mlp2_scales=ckpt.get_tensor(mlp2_scales_name).to(device),
            )
        elif has_mxfp4:
            # Decode MXFP4 → bf16 at load time (more memory, slightly faster forward)
            mlp1 = ckpt.get_mxfp4_tensor(mlp1_blocks_name, mlp1_scales_name)
            mlp2 = ckpt.get_mxfp4_tensor(mlp2_blocks_name, mlp2_scales_name)
            moe_layer.load_bf16_weights(mlp1, mlp2)
        elif ckpt.has_tensor(f"block.{n}.mlp.mlp1_weight"):
            # Raw bf16 weights in checkpoint
            mlp1 = ckpt.get_tensor(f"block.{n}.mlp.mlp1_weight")
            mlp2 = ckpt.get_tensor(f"block.{n}.mlp.mlp2_weight")
            moe_layer.load_bf16_weights(mlp1, mlp2)

        if n % 10 == 0:
            logger.info(f"  Loaded MoE weights for layers 0..{n}")
            torch.cuda.empty_cache()

    # Move remaining meta tensors to device
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    logger.info(
        f"Model loaded: {total_params/1e9:.1f}B params, "
        f"{total_buffers/1e9:.1f}B buffer elements"
    )

    tokenizer = get_gptoss_tokenizer()

    return model, tokenizer, None
