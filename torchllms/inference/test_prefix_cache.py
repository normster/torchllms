"""Tests for the Phase 2 RadixKVCache: page-ID trie backed by PagedKVPool.

Covers tree structure (insert/lookup/partial-edge split), LRU eviction,
and the refcount handoff between live rollouts and radix nodes. Does
not test KV numeric semantics — those live in the pool / kernel tests.
"""

from __future__ import annotations

import pytest

from torchllms.inference.prefix_cache import RadixKVCache, RadixMatch, _Node
from torchllms.models.paged_kv import PagedKVPool


PAGE_SIZE = 4
TOTAL_PAGES = 32
N_LAYERS = 2
N_KV_HEADS = 2
HEAD_DIM = 8


def _mk_pool() -> PagedKVPool:
    import torch
    return PagedKVPool(
        n_layers=N_LAYERS,
        total_pages=TOTAL_PAGES,
        page_size=PAGE_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        max_bsz=4,
        device="cpu",
        dtype=torch.float32,
    )


def _tokens(n: int, start: int = 0) -> tuple:
    """Distinct token IDs so trie paths are deterministic."""
    return tuple(range(start, start + n))


def _alloc_pages(pool: PagedKVPool, n: int) -> list[int]:
    """Return ``n`` fresh page IDs with ref=1 each (simulating a
    rollout's block table)."""
    return [pool._alloc_page() for _ in range(n)]


# ------------------------------------------------------------------
# Empty-trie behavior
# ------------------------------------------------------------------


def test_empty_lookup_returns_no_hit():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    m = radix.lookup(_tokens(10))
    assert not m.hit
    assert m.length == 0
    assert m.page_ids == ()


def test_empty_insert_is_noop():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    radix.insert((), ())
    assert radix.num_nodes() == 1
    assert radix.num_pages() == 0


def test_insert_page_alignment_required():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    pages = _alloc_pages(pool, 2)  # 2 pages = 8 tokens worth
    with pytest.raises(ValueError, match="page alignment|len.tokens.*page_size"):
        radix.insert(_tokens(7), pages)  # 7 tokens ≠ 8


# ------------------------------------------------------------------
# Basic insert/lookup
# ------------------------------------------------------------------


def test_single_insert_adopts_pages():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    pages = _alloc_pages(pool, 3)
    tokens = _tokens(12)
    radix.insert(tokens, pages)
    assert radix.num_pages() == 3
    # Radix bumped each page's refcount from 1 → 2.
    assert all(pool.page_refcount(p) == 2 for p in pages)


def test_lookup_returns_matched_prefix():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    pages = _alloc_pages(pool, 4)
    radix.insert(_tokens(16), pages)
    m = radix.lookup(_tokens(16))
    assert m.hit
    assert m.length == 16
    assert m.page_ids == tuple(pages)


def test_lookup_partial_match_truncates_to_page_boundary():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    pages = _alloc_pages(pool, 4)  # 4 pages = 16 tokens
    radix.insert(_tokens(16), pages)
    # Query 11 tokens matches token-for-token but must truncate to 8
    # (2 full pages) since page_size=4 and match ends mid-page.
    m = radix.lookup(_tokens(11))
    assert m.length == 8
    assert m.page_ids == tuple(pages[:2])


def test_insert_skips_silently_on_sub_page_divergence():
    """Two sequences share < page_size tokens; neither can fully own
    the shared page. Second insert drops its suffix silently."""
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    # First sequence: 8 tokens, 2 pages.
    pages_a = _alloc_pages(pool, 2)
    tokens_a = _tokens(8, start=0)  # [0..8)
    radix.insert(tokens_a, pages_a)
    # Second sequence: shares first 2 tokens only (sub-page), diverges.
    pages_b = _alloc_pages(pool, 2)
    tokens_b = _tokens(2, start=0) + _tokens(6, start=100)
    radix.insert(tokens_b, pages_b)  # should not raise
    # Only the first insert is visible.
    m_a = radix.lookup(tokens_a)
    assert m_a.length == 8 and m_a.page_ids == tuple(pages_a)
    m_b = radix.lookup(tokens_b)
    assert m_b.length == 0  # no page-aligned match for B's variant
    # pages_b are still held by the caller (ref=1 each); they'll go
    # back to the free list once caller releases.
    assert all(pool.page_refcount(p) == 1 for p in pages_b)


def test_lookup_misses_on_first_token_divergence():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    pages = _alloc_pages(pool, 2)
    radix.insert(_tokens(8, start=0), pages)
    m = radix.lookup(_tokens(8, start=100))
    assert not m.hit


# ------------------------------------------------------------------
# Branching / split
# ------------------------------------------------------------------


def test_insert_shared_prefix_splits_edge():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    # First insert: tokens [0, 1, 2, 3, 4, 5, 6, 7] → pages A, B.
    tokens_a = _tokens(8, start=0)
    pages_a = _alloc_pages(pool, 2)
    radix.insert(tokens_a, pages_a)
    # Second insert sharing the first 4 tokens but diverging:
    # [0, 1, 2, 3, 100, 101, 102, 103] → pages C, D.
    tokens_b = _tokens(4, start=0) + _tokens(4, start=100)
    pages_b = _alloc_pages(pool, 2)
    radix.insert(tokens_b, pages_b)
    # Structure should be root → (shared 4t, 1 page) → two children.
    # Either the first tokens_a node got split at page 1, or the root
    # now has a shared-prefix node with two children.
    m_a = radix.lookup(tokens_a)
    m_b = radix.lookup(tokens_b)
    assert m_a.length == 8 and m_a.page_ids == tuple(pages_a)
    assert m_b.length == 8 and m_b.page_ids == tuple(pages_b[:0]) + (pages_a[0],) + tuple(pages_b[1:])
    # Common prefix: page A is shared.
    # (Page A owned by 2 refs: one per insert's borrow? Actually radix
    # only holds 1 ref on page A — the shared-prefix node owns it.)
    assert pool.page_refcount(pages_a[0]) == 2  # rollout + radix
    # pages_a[1] (only in the A branch) has ref 2.
    assert pool.page_refcount(pages_a[1]) == 2


def test_lookup_shared_prefix_returns_page_aligned_shared():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    pages_a = _alloc_pages(pool, 2)
    radix.insert(_tokens(8), pages_a)
    # Look up a prompt that shares the first 6 tokens — page-aligned
    # match should be 4 (one page).
    m = radix.lookup(_tokens(6))
    assert m.length == 4
    assert m.page_ids == (pages_a[0],)


# ------------------------------------------------------------------
# Leaf extension (compression)
# ------------------------------------------------------------------


def test_leaf_extension_appends_to_existing_node():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    pages_a = _alloc_pages(pool, 1)
    radix.insert(_tokens(4), pages_a)
    # Extending the same leaf with more tokens — leaf-extension rule
    # merges into one node rather than chaining.
    pages_b = _alloc_pages(pool, 2)
    radix.insert(_tokens(12), pages_a + pages_b)
    assert radix.num_nodes() == 2  # root + one leaf
    m = radix.lookup(_tokens(12))
    assert m.length == 12


# ------------------------------------------------------------------
# LRU eviction
# ------------------------------------------------------------------


def test_evict_releases_pages_to_pool():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    pages_a = _alloc_pages(pool, 1)
    pages_b = _alloc_pages(pool, 1)
    radix.insert(_tokens(4, start=0), pages_a)
    radix.insert(_tokens(4, start=100), pages_b)
    # Touch pages_a to make pages_b the LRU leaf.
    radix.lookup(_tokens(4, start=0))
    # Simulate caller dropping their refs so radix is the only holder.
    pool.release_pages(pages_a)
    pool.release_pages(pages_b)
    assert pool.page_refcount(pages_a[0]) == 1  # radix only
    assert pool.page_refcount(pages_b[0]) == 1
    n = radix.evict_oldest()
    assert n == 1
    assert pool.page_refcount(pages_b[0]) == 0
    # pages_a survives.
    assert pool.page_refcount(pages_a[0]) == 1


def test_evict_returns_zero_on_empty_trie():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    assert radix.evict_oldest() == 0


def test_clear_releases_everything():
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    pages_a = _alloc_pages(pool, 2)
    radix.insert(_tokens(8), pages_a)
    pool.release_pages(pages_a)  # caller drops refs
    radix.clear()
    assert radix.num_pages() == 0
    for p in pages_a:
        assert pool.page_refcount(p) == 0


# ------------------------------------------------------------------
# Refcount handoff on a complete rollout→retire→insert cycle
# ------------------------------------------------------------------


def test_retire_insert_release_preserves_radix_ref():
    """Full rollout lifecycle: alloc fresh pages (ref=1), retire (return
    page list without decrementing), insert into radix (radix bumps
    ref→2), release (ref→1, held only by radix)."""
    pool = _mk_pool()
    radix = RadixKVCache(pool)
    rid = pool.claim()
    pool.extend(rid, 8)  # 2 pages, fresh
    pages_before = list(pool.pages_of(rid))
    assert all(pool.page_refcount(p) == 1 for p in pages_before)

    pages, seqlen = pool.retire_pages(rid)
    assert pages == pages_before
    assert seqlen == 8

    radix.insert(_tokens(8), pages)
    # Radix bumped ref 1→2 on each page.
    assert all(pool.page_refcount(p) == 2 for p in pages)

    pool.release_pages(pages)
    # Caller released — each page now held only by radix.
    assert all(pool.page_refcount(p) == 1 for p in pages)
    # None are in the free list.
    assert not any(p in pool._free_pages for p in pages)


def test_lookup_hit_then_retire_handles_shared_pages_correctly():
    """Second rollout borrows pages from radix, extends, retires; the
    re-insert should NOT double-bump the shared pages' refcount."""
    pool = _mk_pool()
    radix = RadixKVCache(pool)

    # First rollout: creates the prefix.
    r1 = pool.claim()
    pool.extend(r1, 8)
    pages1 = list(pool.pages_of(r1))
    pages_ret1, _ = pool.retire_pages(r1)
    radix.insert(_tokens(8), pages_ret1)
    pool.release_pages(pages_ret1)
    # pages1: ref=1 each (radix only).

    # Second rollout: prefix-matches, extends with fresh pages.
    r2 = pool.claim()
    m = radix.lookup(_tokens(8))
    assert m.length == 8
    pool.borrow_pages(m.page_ids)
    pool.attach_borrowed_pages(r2, m.page_ids)
    # Shared pages: ref=2 (radix + r2). Extend with a fresh page.
    pool.extend(r2, PAGE_SIZE)
    pages2 = pool.pages_of(r2)
    fresh = pages2[2]  # new page allocated on extend
    assert pool.page_refcount(fresh) == 1

    # Retire r2 and insert extended prefix (12 tokens, 3 pages).
    tokens_ext = _tokens(8) + _tokens(4, start=8)
    pages_ret2, _ = pool.retire_pages(r2)
    radix.insert(tokens_ext, pages_ret2)
    # Shared pages (already in trie): no ref change from re-insert —
    # still at (radix + r2-borrow) = 2.
    assert pool.page_refcount(pages1[0]) == 2
    assert pool.page_refcount(pages1[1]) == 2
    # Fresh page was adopted by the new edge: ref 1 → 2.
    assert pool.page_refcount(fresh) == 2

    pool.release_pages(pages_ret2)
    # After release: all three pages drop by 1. All end up at 1,
    # held only by the radix (pages1[0,1] by the split-point node, fresh
    # by the tail node).
    assert pool.page_refcount(pages1[0]) == 1
    assert pool.page_refcount(pages1[1]) == 1
    assert pool.page_refcount(fresh) == 1
