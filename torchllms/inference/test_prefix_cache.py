"""Tests for RadixKVCache: tree structure, partial-edge lookup, LRU eviction."""

from __future__ import annotations

import torch

from torchllms.inference.prefix_cache import RadixKVCache, _Node
from torchllms.models.cache import KVChunk


N_LAYERS = 2
N_KV_HEADS = 2
HEAD_DIM = 4


def _mk_block(length: int, fill: float = 1.0, role_id: int = 1) -> KVChunk:
    """Make a KVChunk with a recognizable fill pattern.

    The K tensor at position i contains `fill + i`, so concat/slice is
    verifiable by inspection. V = -K.
    """
    base = torch.arange(length, dtype=torch.bfloat16).view(1, length, 1, 1)
    base = base.expand(N_LAYERS, length, N_KV_HEADS, HEAD_DIM).clone()
    k = base + fill
    v = -k
    role_ids = torch.full((length,), role_id, dtype=torch.long)
    return KVChunk(k=k, v=v, role_ids=role_ids)


def _block_for_range(start: int, end: int, fill: float = 1.0, role_id: int = 1) -> KVChunk:
    """Make a block whose per-position values encode absolute token positions.

    Useful when testing splits: each block is a slice of a virtual 'prefix
    block' and its values should be preserved exactly after split/concat.
    """
    length = end - start
    positions = torch.arange(start, end, dtype=torch.bfloat16).view(1, length, 1, 1)
    positions = positions.expand(N_LAYERS, length, N_KV_HEADS, HEAD_DIM).clone()
    k = positions + fill
    v = -k
    role_ids = torch.full((length,), role_id, dtype=torch.long)
    return KVChunk(k=k, v=v, role_ids=role_ids)


# ------------------------------------------------------------------
# KVChunk basics
# ------------------------------------------------------------------


def test_kvblock_size_bytes_matches_tensors():
    b = _mk_block(10)
    expected = b.k.element_size() * b.k.numel()
    expected += b.v.element_size() * b.v.numel()
    expected += b.role_ids.element_size() * b.role_ids.numel()
    assert b.size_bytes == expected


def test_kvblock_slice_and_concat_roundtrip():
    full = _block_for_range(0, 10)
    head = full.slice(0, 4)
    tail = full.slice(4, 10)
    assert head.length == 4 and tail.length == 6
    rejoined = KVChunk.concat([head, tail])
    assert torch.equal(rejoined.k, full.k)
    assert torch.equal(rejoined.v, full.v)
    assert torch.equal(rejoined.role_ids, full.role_ids)


# ------------------------------------------------------------------
# Empty / single-insert behavior
# ------------------------------------------------------------------


def test_empty_cache_miss():
    c = RadixKVCache(max_bytes=1 << 30)
    m = c.lookup([1, 2, 3])
    assert not m.hit
    assert m.length == 0


def test_single_insert_exact_match():
    c = RadixKVCache(max_bytes=1 << 30)
    tokens = [1, 2, 3, 4]
    block = _mk_block(4)
    c.insert(tokens, block)
    m = c.lookup(tokens)
    assert m.hit and m.length == 4
    mat = m.materialize()
    assert torch.equal(mat.k, block.k)
    assert torch.equal(mat.v, block.v)


def test_lookup_longer_than_stored_falls_back_to_prefix():
    c = RadixKVCache(max_bytes=1 << 30)
    c.insert([1, 2, 3, 4], _mk_block(4))
    m = c.lookup([1, 2, 3, 4, 5, 6])
    assert m.length == 4
    assert m.materialize().length == 4


def test_insert_empty_is_noop():
    c = RadixKVCache(max_bytes=1 << 30)
    c.insert([], _mk_block(0) if False else KVChunk(
        k=torch.zeros(N_LAYERS, 0, N_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        v=torch.zeros(N_LAYERS, 0, N_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        role_ids=torch.zeros(0, dtype=torch.long),
    ))
    assert c.total_bytes == 0
    assert c.num_nodes() == 1  # just root


# ------------------------------------------------------------------
# Split on divergence
# ------------------------------------------------------------------


def test_insert_diverging_sibling_splits_edge():
    c = RadixKVCache(max_bytes=1 << 30)
    a_tokens = [10, 11, 12, 13, 14]  # 5 tokens
    b_tokens = [10, 11, 99, 98]      # shared prefix len 2
    a_block = _block_for_range(0, 5)
    b_block = _block_for_range(0, 4, role_id=2)
    c.insert(a_tokens, a_block)
    c.insert(b_tokens, b_block)

    # Both prefixes retrievable.
    ma = c.lookup(a_tokens)
    mb = c.lookup(b_tokens)
    assert ma.length == 5 and mb.length == 4

    # Materialization preserves values.
    ka = ma.materialize()
    kb = mb.materialize()
    assert torch.equal(ka.k, a_block.k)
    assert torch.equal(kb.k, b_block.k)

    # Structure: shared root -> split_node(2 tokens) -> {a_tail(3), b_tail(2)}
    root_child = c._root.children[10]
    assert root_child.edge_tokens == (10, 11)
    assert len(root_child.children) == 2


def test_insert_endpoint_on_split_boundary():
    """Inserting a shorter prefix that lands exactly on a split creates no
    extra node; the split node IS the prefix endpoint."""
    c = RadixKVCache(max_bytes=1 << 30)
    c.insert([1, 2, 3, 4, 5], _block_for_range(0, 5))
    c.insert([1, 2], _block_for_range(0, 2, role_id=2))

    m = c.lookup([1, 2])
    assert m.length == 2
    # The split created a node at depth 2 that is also the [1,2] endpoint.
    # Lookup for [1, 2] returns a full-edge match (not partial).
    assert m._last_edge_consumed == 2


# ------------------------------------------------------------------
# Partial-edge lookup (does NOT mutate the tree)
# ------------------------------------------------------------------


def test_partial_edge_match_reports_correct_length():
    c = RadixKVCache(max_bytes=1 << 30)
    c.insert([1, 2, 3, 4, 5], _block_for_range(0, 5))
    before_nodes = c.num_nodes()
    m = c.lookup([1, 2, 3, 99])  # diverges at pos 3
    assert m.length == 3
    assert c.num_nodes() == before_nodes  # tree unchanged
    mat = m.materialize()
    assert mat.length == 3
    # Values preserved: K at position i == i + 1.0
    expected = _block_for_range(0, 3)
    assert torch.equal(mat.k, expected.k)


def test_partial_edge_then_insert_splits_at_that_point():
    c = RadixKVCache(max_bytes=1 << 30)
    c.insert([1, 2, 3, 4, 5], _block_for_range(0, 5))
    _ = c.lookup([1, 2, 3, 99])  # partial, no mutation
    # Now actually insert the diverging prefix
    c.insert([1, 2, 3, 99, 100], _block_for_range(0, 5, role_id=2))
    ma = c.lookup([1, 2, 3, 4, 5])
    mb = c.lookup([1, 2, 3, 99, 100])
    assert ma.length == 5
    assert mb.length == 5


# ------------------------------------------------------------------
# Leaf extension (chain avoidance)
# ------------------------------------------------------------------


def test_extending_prefix_rewrites_same_leaf_not_new_chain():
    c = RadixKVCache(max_bytes=1 << 30)
    c.insert([1, 2, 3], _block_for_range(0, 3))
    nodes_before = c.num_nodes()
    assert nodes_before == 2  # root + one leaf

    # Turn 2: extend same prefix with more tokens.
    c.insert([1, 2, 3, 4, 5], _block_for_range(0, 5))
    nodes_after = c.num_nodes()
    assert nodes_after == 2  # still just root + one leaf, edge extended

    m = c.lookup([1, 2, 3, 4, 5])
    assert m.length == 5
    mat = m.materialize()
    # Values from the latest insert (which is the full-prefix block)
    assert torch.equal(mat.k, _block_for_range(0, 5).k)


def test_extending_then_branching_still_finds_both_prefixes():
    c = RadixKVCache(max_bytes=1 << 30)
    c.insert([1, 2, 3], _block_for_range(0, 3))
    c.insert([1, 2, 3, 4, 5], _block_for_range(0, 5))
    # Now a sibling branches at position 3
    c.insert([1, 2, 3, 99], _block_for_range(0, 4, role_id=3))

    assert c.lookup([1, 2, 3, 4, 5]).length == 5
    assert c.lookup([1, 2, 3, 99]).length == 4
    # Partial match at common prefix
    m = c.lookup([1, 2, 3])
    assert m.length == 3


# ------------------------------------------------------------------
# LRU eviction
# ------------------------------------------------------------------


def test_lru_evicts_oldest_leaf_when_over_budget():
    # Each block is L=4 tokens. n_layers=2, 2 kv_heads, head_dim=4.
    # K,V: 2 * (2*4*2*4) * 2 bytes (bf16) = 128 bytes each
    # role_ids: 4 * 8 bytes = 32 bytes
    # Total per block ~= 288 bytes.
    # Set budget to 2 blocks.
    per_block = _mk_block(4).size_bytes
    c = RadixKVCache(max_bytes=per_block * 2 + 16)

    c.insert([1, 2, 3, 4], _mk_block(4))     # oldest
    c.insert([5, 6, 7, 8], _mk_block(4, fill=10.0))
    c.insert([9, 10, 11, 12], _mk_block(4, fill=20.0))  # should evict [1..4]

    assert not c.lookup([1, 2, 3, 4]).hit
    assert c.lookup([5, 6, 7, 8]).hit
    assert c.lookup([9, 10, 11, 12]).hit


def test_lru_access_updates_recency():
    per_block = _mk_block(4).size_bytes
    c = RadixKVCache(max_bytes=per_block * 2 + 16)

    c.insert([1, 2, 3, 4], _mk_block(4))
    c.insert([5, 6, 7, 8], _mk_block(4, fill=10.0))
    # Touch [1..4] so it becomes most-recently-accessed.
    assert c.lookup([1, 2, 3, 4]).hit
    c.insert([9, 10, 11, 12], _mk_block(4, fill=20.0))  # should evict [5..8]

    assert c.lookup([1, 2, 3, 4]).hit
    assert not c.lookup([5, 6, 7, 8]).hit
    assert c.lookup([9, 10, 11, 12]).hit


def test_eviction_compacts_singleton_parent():
    """After split + sibling eviction, the remaining chain compacts into one
    edge so the tree matches its ideal shape."""
    per_block = _mk_block(3).size_bytes
    c = RadixKVCache(max_bytes=10 * per_block)

    # After these two inserts, tree has: root -> split(2) -> {tail_a(3), tail_b(2)}
    c.insert([1, 2, 3, 4, 5], _block_for_range(0, 5))
    c.insert([1, 2, 9, 10], _block_for_range(0, 4, role_id=2))
    assert c.num_nodes() == 4  # root + split + 2 leaves

    # Tighten budget so tail_b gets evicted. Recent-most should be [1,2,9,10]
    # by insertion order; touch [1..5] to make it newer, so [1,2,9,10] is
    # older and gets evicted.
    c.lookup([1, 2, 3, 4, 5])
    # Budget just below total; force one eviction
    c.max_bytes = c.total_bytes - 1
    c._evict_if_needed()

    # After eviction, split_node has only tail_a as its remaining child;
    # compaction merges split_node with tail_a. Tree should be root -> one leaf.
    assert c.num_nodes() == 2
    # The remaining leaf's path materializes to the original 5-token block.
    m = c.lookup([1, 2, 3, 4, 5])
    assert m.length == 5
    assert torch.equal(m.materialize().k, _block_for_range(0, 5).k)


# ------------------------------------------------------------------
# Multi-turn rollout flow
# ------------------------------------------------------------------


def test_multi_turn_rollout_reuses_prefix():
    c = RadixKVCache(max_bytes=1 << 30)
    # Turn 1: insert prompt + response
    c.insert(list(range(10)), _block_for_range(0, 10))
    # Turn 2 starts with prior prompt + response + tool result (new tokens 10..19)
    turn2_tokens = list(range(20))
    m = c.lookup(turn2_tokens)
    assert m.length == 10  # reused the prior turn
    prior = m.materialize()
    assert torch.equal(prior.k, _block_for_range(0, 10).k)
    # Turn 2 inserts the new full prefix (20 tokens)
    c.insert(turn2_tokens, _block_for_range(0, 20))
    m2 = c.lookup(turn2_tokens)
    assert m2.length == 20


if __name__ == "__main__":
    import sys
    # Minimal stdlib-ish runner so this works without pytest on the machine.
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
