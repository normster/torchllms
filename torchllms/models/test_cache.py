"""Tests for LinearKVCache.load_block / extract_block round-trips."""

from __future__ import annotations

import torch

from torchllms.models.cache import KVBlock, LinearKVCache


N_LAYERS = 3
MAX_BSZ = 2
MAX_SEQLEN = 16
N_KV_HEADS = 2
HEAD_DIM = 4


def _make_arena() -> LinearKVCache:
    return LinearKVCache(
        n_layers=N_LAYERS,
        max_bsz=MAX_BSZ,
        max_seqlen=MAX_SEQLEN,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        device="cpu",
        dtype=torch.bfloat16,
    )


def _random_block(length: int, seed: int = 0) -> KVBlock:
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(
        N_LAYERS, length, N_KV_HEADS, HEAD_DIM, generator=g, dtype=torch.bfloat16
    )
    v = torch.randn(
        N_LAYERS, length, N_KV_HEADS, HEAD_DIM, generator=g, dtype=torch.bfloat16
    )
    role_ids = torch.arange(length, dtype=torch.long) % 4
    return KVBlock(k=k, v=v, role_ids=role_ids)


def test_load_and_extract_round_trip():
    arena = _make_arena()
    block = _random_block(length=5, seed=0)
    arena.load_block(block, row_idx=0, at_pos=0)
    out = arena.extract_block(row_idx=0, length=5)
    assert torch.equal(out.k, block.k)
    assert torch.equal(out.v, block.v)
    assert torch.equal(out.role_ids, block.role_ids)


def test_load_advances_seen_tokens_for_all_layers():
    arena = _make_arena()
    arena.load_block(_random_block(length=7), row_idx=0)
    assert arena.seen_tokens == [7] * N_LAYERS
    assert int(arena.next_start_pos[0].item()) == 7
    # load_block does NOT populate the attention mask; callers rely on
    # is_causal=True in the attention kernel (single-sequence path) or supply
    # an explicit mask (batched path).
    assert not arena.is_attn_mask_cached


def test_update_kv_appends_past_loaded_prefix():
    arena = _make_arena()
    arena.load_block(_random_block(length=4, seed=1), row_idx=0)
    # Simulate one layer forward of one new token.
    new_k = torch.ones(MAX_BSZ, 1, N_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    new_v = -new_k
    k_full, v_full = arena.update_kv(0, new_k, new_v)
    assert k_full.shape == (MAX_BSZ, 5, N_KV_HEADS, HEAD_DIM)
    assert torch.equal(k_full[0, 4], new_k[0, 0])
    assert arena.seen_tokens[0] == 5


def test_extract_rejects_if_layer_seen_less_than_requested():
    arena = _make_arena()
    arena.load_block(_random_block(length=4, seed=2), row_idx=0)
    # Extract with length greater than seen should raise.
    try:
        arena.extract_block(row_idx=0, length=6)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_load_rejects_over_seen_tokens():
    arena = _make_arena()
    arena.load_block(_random_block(length=5, seed=3), row_idx=0)
    # Second load at_pos=0 should refuse since seen_tokens is already past 0.
    try:
        arena.load_block(_random_block(length=3, seed=4), row_idx=0, at_pos=0)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"{name}: ok")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"{name}: FAIL - {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
