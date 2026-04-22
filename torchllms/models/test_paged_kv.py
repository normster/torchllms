"""Tests for PagedKVPool.

These cover the allocator + bookkeeping on CPU (no flashinfer dependency)
plus a GPU-only end-to-end append_kv roundtrip. Paged-attention kernel
parity against dense attention lives in ``validate_qwen3``.
"""

from __future__ import annotations

import pytest
import torch

from torchllms.models.cache import KVChunk
from torchllms.models.paged_kv import PagedBatchLayout, PagedKVPool


N_LAYERS = 3
TOTAL_PAGES = 16
PAGE_SIZE = 4
N_KV_HEADS = 2
HEAD_DIM = 8
DTYPE = torch.float32


def _mk_pool(
    *, n_layers=N_LAYERS, total_pages=TOTAL_PAGES, page_size=PAGE_SIZE,
    max_bsz=4, device="cpu", dtype=DTYPE,
) -> PagedKVPool:
    return PagedKVPool(
        n_layers=n_layers,
        total_pages=total_pages,
        page_size=page_size,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        max_bsz=max_bsz,
        device=device,
        dtype=dtype,
    )


def _mk_chunk(length: int, *, fill_k=1.0, fill_v=-1.0, role_id=0) -> KVChunk:
    k = torch.full((N_LAYERS, length, N_KV_HEADS, HEAD_DIM), fill_k, dtype=DTYPE)
    v = torch.full((N_LAYERS, length, N_KV_HEADS, HEAD_DIM), fill_v, dtype=DTYPE)
    role_ids = torch.full((length,), role_id, dtype=torch.long)
    return KVChunk(k=k, v=v, role_ids=role_ids)


# =====================================================================
# Allocation + invariants
# =====================================================================


def test_fresh_pool_is_all_free():
    pool = _mk_pool()
    assert pool.b_live == 0
    assert pool.free_page_count == TOTAL_PAGES
    pool._check_invariants()


def test_claim_allocates_no_pages():
    pool = _mk_pool()
    rid = pool.claim()
    assert pool.b_live == 1
    assert pool.seqlen(rid) == 0
    assert pool.pages_of(rid) == []
    assert pool.free_page_count == TOTAL_PAGES
    pool._check_invariants()


def test_extend_allocates_pages_ceil_div():
    pool = _mk_pool()
    rid = pool.claim()
    pool.extend(rid, 1)
    assert len(pool.pages_of(rid)) == 1        # ceil(1 / 4) = 1
    pool.extend(rid, 3)
    assert len(pool.pages_of(rid)) == 1        # ceil(4 / 4) = 1
    pool.extend(rid, 1)
    assert len(pool.pages_of(rid)) == 2        # ceil(5 / 4) = 2
    pool.extend(rid, 8)
    assert len(pool.pages_of(rid)) == 4        # ceil(13 / 4) = 4
    assert pool.seqlen(rid) == 13
    pool._check_invariants()


def test_retire_releases_all_pages():
    pool = _mk_pool()
    rid = pool.claim()
    pool.extend(rid, 10)  # 3 pages
    assert pool.free_page_count == TOTAL_PAGES - 3
    pool.retire(rid)
    assert pool.b_live == 0
    assert pool.free_page_count == TOTAL_PAGES
    pool._check_invariants()


def test_retire_many_preserves_survivors():
    pool = _mk_pool()
    r1, r2, r3 = pool.claim(), pool.claim(), pool.claim()
    pool.extend(r1, 4)
    pool.extend(r2, 8)
    pool.extend(r3, 2)
    pool.retire_many([r1, r3])
    assert pool.b_live == 1
    assert pool.active_rollouts() == [r2]
    assert pool.seqlen(r2) == 8
    pool._check_invariants()


def test_out_of_pages_raises():
    pool = _mk_pool(total_pages=2, page_size=4)
    rid = pool.claim()
    with pytest.raises(RuntimeError, match="out of pages"):
        pool.extend(rid, 100)  # way more than 2 pages worth


def test_recycled_pages_reused():
    pool = _mk_pool(total_pages=3, page_size=4)
    r1 = pool.claim()
    pool.extend(r1, 8)  # 2 pages
    p1 = set(pool.pages_of(r1))
    pool.retire(r1)
    r2 = pool.claim()
    pool.extend(r2, 8)
    p2 = set(pool.pages_of(r2))
    # With 3 total pages and 2 pages per rollout the second rollout must
    # reuse at least one of the first rollout's freed pages.
    assert p1 & p2


def test_claim_ids_are_monotonic():
    pool = _mk_pool()
    rids = [pool.claim() for _ in range(5)]
    assert rids == sorted(rids)
    pool.retire(rids[2])
    next_rid = pool.claim()
    assert next_rid > max(rids)


# =====================================================================
# build_batch_layout
# =====================================================================


def test_layout_uniform_qlen_prefill():
    pool = _mk_pool()
    r1, r2 = pool.claim(), pool.claim()
    S = 5  # 5 tokens per row, two pages (page_size=4)
    pool.extend_many([r1, r2], [S, S])
    layout = pool.build_batch_layout([r1, r2], [S, S])
    assert isinstance(layout, PagedBatchLayout)
    assert layout.batch_size == 2
    assert layout.total_new_tokens == 2 * S
    assert layout.qo_indptr.tolist() == [0, 5, 10]
    assert layout.kv_indptr.tolist() == [0, 2, 4]  # 2 pages each
    assert layout.kv_last_page_len.tolist() == [1, 1]  # 5 - 4 = 1
    assert layout.seqlens_before_write == [0, 0]
    # batch_indices = [0,0,0,0,0, 1,1,1,1,1]; positions = [0..4, 0..4]
    assert layout.batch_indices.tolist() == [0] * S + [1] * S
    assert layout.positions.tolist() == list(range(S)) * 2


def test_layout_decode_step_after_prefill():
    pool = _mk_pool()
    r1, r2 = pool.claim(), pool.claim()
    pool.extend_many([r1, r2], [5, 3])  # diverging pre-write lengths
    # Now one decode step: qlens = [1, 1]. Pre-write seqlens are 5 and 3.
    pool.extend_many([r1, r2], [1, 1])
    layout = pool.build_batch_layout([r1, r2], [1, 1])
    assert layout.total_new_tokens == 2
    assert layout.qo_indptr.tolist() == [0, 1, 2]
    # r1: 6 tokens → 2 pages (4+2). r2: 4 tokens → 1 page full.
    assert layout.kv_indptr.tolist() == [0, 2, 3]
    assert layout.kv_last_page_len.tolist() == [2, 4]
    # batch_indices = [0, 1]; positions = [pre_write[0], pre_write[1]]
    assert layout.batch_indices.tolist() == [0, 1]
    assert layout.positions.tolist() == [5, 3]


def test_layout_ragged_qlen():
    pool = _mk_pool()
    r1, r2 = pool.claim(), pool.claim()
    pool.extend_many([r1, r2], [3, 5])
    layout = pool.build_batch_layout([r1, r2], [3, 5])
    assert layout.qo_indptr.tolist() == [0, 3, 8]
    assert layout.batch_indices.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert layout.positions.tolist() == [0, 1, 2, 0, 1, 2, 3, 4]


def test_layout_catches_missing_extend():
    pool = _mk_pool()
    rid = pool.claim()
    # Didn't call extend — build_batch_layout should detect it.
    with pytest.raises(RuntimeError, match="negative pre-write seqlen"):
        pool.build_batch_layout([rid], [5])


# =====================================================================
# row_positions
# =====================================================================


def test_row_positions_reflects_pre_extend_seqlens():
    pool = _mk_pool()
    r1, r2 = pool.claim(), pool.claim()
    pool.extend(r1, 7)
    rp = pool.row_positions([r1, r2])
    assert rp.tolist() == [7, 0]
    assert rp.dtype == torch.int32


def test_row_positions_default_uses_active_order():
    pool = _mk_pool()
    r1, r2, r3 = pool.claim(), pool.claim(), pool.claim()
    pool.extend(r1, 2)
    pool.extend(r2, 5)
    pool.extend(r3, 1)
    pool.retire(r2)
    # Survivors in claim order: [r1, r3].
    rp = pool.row_positions()
    assert rp.tolist() == [2, 1]


# =====================================================================
# load_chunk / extract_chunk (Phase 1 compat)
# =====================================================================


def test_load_chunk_then_extract_roundtrip():
    pool = _mk_pool()
    rid = pool.claim()
    chunk = _mk_chunk(length=7, fill_k=3.0, fill_v=-3.0)
    pool.load_chunk(chunk, rid)
    assert pool.seqlen(rid) == 7
    assert len(pool.pages_of(rid)) == 2  # ceil(7/4)
    out = pool.extract_chunk(rid)
    assert out.length == 7
    assert torch.allclose(out.k, chunk.k)
    assert torch.allclose(out.v, chunk.v)
    pool._check_invariants()


def test_load_chunk_rejects_nonempty_rollout():
    pool = _mk_pool()
    rid = pool.claim()
    pool.extend(rid, 1)
    chunk = _mk_chunk(length=3)
    with pytest.raises(RuntimeError, match="non-empty"):
        pool.load_chunk(chunk, rid)


def test_load_chunk_rejects_nonzero_offset():
    pool = _mk_pool()
    rid = pool.claim()
    chunk = _mk_chunk(length=3)
    with pytest.raises(NotImplementedError):
        pool.load_chunk(chunk, rid, at_pos=1)


def test_extract_chunk_partial_length():
    pool = _mk_pool()
    rid = pool.claim()
    chunk = _mk_chunk(length=7)
    pool.load_chunk(chunk, rid)
    partial = pool.extract_chunk(rid, length=3)
    assert partial.length == 3
    assert torch.allclose(partial.k, chunk.k[:, :3])


def test_load_chunk_preserves_distinct_values_across_layers():
    # Different value per (layer, position) to catch any layer-swap bug.
    pool = _mk_pool()
    rid = pool.claim()
    length = 5
    k = torch.arange(
        N_LAYERS * length * N_KV_HEADS * HEAD_DIM, dtype=DTYPE,
    ).reshape(N_LAYERS, length, N_KV_HEADS, HEAD_DIM)
    v = -k.clone()
    chunk = KVChunk(k=k, v=v, role_ids=torch.zeros(length, dtype=torch.long))
    pool.load_chunk(chunk, rid)
    out = pool.extract_chunk(rid)
    assert torch.allclose(out.k, k)
    assert torch.allclose(out.v, v)


def test_retire_returns_full_chunk():
    pool = _mk_pool()
    rid = pool.claim()
    chunk_in = _mk_chunk(length=9, fill_k=2.0, fill_v=-2.0)
    pool.load_chunk(chunk_in, rid)
    chunk_out = pool.retire(rid)
    assert chunk_out.length == 9
    assert torch.allclose(chunk_out.k, chunk_in.k)
    # All pages released.
    assert pool.free_page_count == TOTAL_PAGES
    pool._check_invariants()


# =====================================================================
# Refcount / page sharing (Phase 2)
# =====================================================================


def test_fresh_alloc_sets_refcount_to_one():
    pool = _mk_pool()
    rid = pool.claim()
    pool.extend(rid, 10)  # 3 pages allocated
    for p in pool.pages_of(rid):
        assert pool.page_refcount(p) == 1


def test_retire_returns_page_ids_without_decrement():
    pool = _mk_pool()
    rid = pool.claim()
    pool.extend(rid, 8)
    pages_before = list(pool.pages_of(rid))
    assert all(pool.page_refcount(p) == 1 for p in pages_before)

    pages, seqlen = pool.retire_pages(rid)
    assert pages == pages_before
    assert seqlen == 8
    # retire_pages does NOT decrement — caller still holds refs.
    assert all(pool.page_refcount(p) == 1 for p in pages)
    # Pages are not yet back in the free list (refs > 0).
    assert not any(p in pool._free_pages for p in pages)


def test_release_pages_frees_at_ref_zero():
    pool = _mk_pool()
    rid = pool.claim()
    pool.extend(rid, 8)
    pages, _ = pool.retire_pages(rid)
    pool.release_pages(pages)
    # All pages freed.
    assert pool.free_page_count == TOTAL_PAGES
    assert all(pool.page_refcount(p) == 0 for p in pages)


def test_borrow_pages_bumps_refcount():
    pool = _mk_pool()
    rid = pool.claim()
    pool.extend(rid, 4)
    p = pool.pages_of(rid)[0]
    assert pool.page_refcount(p) == 1
    pool.borrow_pages([p])
    assert pool.page_refcount(p) == 2


def test_release_does_not_free_while_borrowed():
    pool = _mk_pool()
    r1 = pool.claim()
    pool.extend(r1, 4)
    page = pool.pages_of(r1)[0]
    # Simulate a radix node borrowing this page.
    pool.borrow_pages([page])   # ref 1 → 2

    # Retire r1; pool-owned ref releases but radix-owned ref survives.
    pages, _ = pool.retire_pages(r1)
    pool.release_pages(pages)   # ref 2 → 1
    assert pool.page_refcount(page) == 1
    assert page not in pool._free_pages

    # Radix releases.
    pool.release_pages([page])  # ref 1 → 0
    assert pool.page_refcount(page) == 0
    assert page in pool._free_pages


def test_attach_borrowed_pages_sets_block_table():
    pool = _mk_pool()
    rid = pool.claim()
    # Simulate pages owned elsewhere (pretend-radix).
    pool._page_refcount[0] = 1
    pool._page_refcount[1] = 1
    pool._free_pages = [p for p in pool._free_pages if p not in (0, 1)]
    pool.borrow_pages([0, 1])   # now ref=2 each

    pool.attach_borrowed_pages(rid, [0, 1])
    assert pool.pages_of(rid) == [0, 1]
    assert pool.seqlen(rid) == 2 * PAGE_SIZE


def test_release_double_frees_errors():
    pool = _mk_pool()
    rid = pool.claim()
    pool.extend(rid, 1)
    page = pool.pages_of(rid)[0]
    pool.retire_pages(rid)
    pool.release_pages([page])
    with pytest.raises(RuntimeError, match="double-release"):
        pool.release_pages([page])


def test_clamp_shrinks_seqlen_and_releases_excess_pages():
    pool = _mk_pool()
    r1, r2 = pool.claim(), pool.claim()
    pool.extend(r1, 16)   # 4 pages at page_size=4
    pool.extend(r2, 8)    # 2 pages
    r1_pages_before = pool.pages_of(r1)
    # Clamp r1 to 7 tokens (2 pages), r2 stays at 8.
    pool.clamp_seqlens_per_row([r1, r2], [7, 8])
    assert pool.seqlen(r1) == 7
    assert pool.seqlen(r2) == 8
    assert len(pool.pages_of(r1)) == 2  # ceil(7/4)
    assert len(pool.pages_of(r2)) == 2
    # Released pages went back to free list.
    released = set(r1_pages_before[2:])
    assert released.issubset(set(pool._free_pages))
    pool._check_invariants()


def test_clamp_rejects_grow():
    pool = _mk_pool()
    rid = pool.claim()
    pool.extend(rid, 4)
    with pytest.raises(ValueError, match="cannot extend"):
        pool.clamp_seqlens_per_row([rid], [8])


def test_alloc_errors_when_all_held():
    pool = _mk_pool(total_pages=2, page_size=4)
    # Hold all pages via borrow (simulating radix ownership).
    pool._page_refcount = [1, 1]
    pool._free_pages = []
    rid = pool.claim()
    with pytest.raises(RuntimeError, match="out of pages"):
        pool.extend(rid, 1)


# =====================================================================
# append_kv (GPU-only — flashinfer kernel)
# =====================================================================


_CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="append_kv uses flashinfer.append_paged_kv_cache (CUDA only)",
)


@_CUDA_ONLY
def test_append_kv_scatter_roundtrip():
    """append_kv via flashinfer + extract_chunk via index read should
    round-trip the same K/V per-position."""
    pool = PagedKVPool(
        n_layers=2,
        total_pages=8,
        page_size=4,
        n_kv_heads=2,
        head_dim=64,       # flashinfer-supported
        max_bsz=1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    rid = pool.claim()
    S = 6
    pool.extend_many([rid], [S])
    layout = pool.build_batch_layout([rid], [S])

    # Distinct per-position values so we can verify the scatter routed
    # each token to the right page slot.
    k = (torch.arange(S, dtype=torch.float32, device="cuda")
         [:, None, None].expand(S, 2, 64)).to(torch.bfloat16).contiguous()
    v = -k

    for layer_id in range(pool.n_layers):
        pool.append_kv(layer_id, k, v, layout)

    # Round-trip: extract_chunk reads back the paged storage through
    # page-indexed gather and reshapes to dense [n_layers, S, ...].
    chunk = pool.extract_chunk(rid)
    assert chunk.length == S
    for layer_id in range(pool.n_layers):
        # The write was identical across layers, so every layer's slab
        # should match the source k/v.
        assert torch.allclose(
            chunk.k[layer_id].to(torch.float32),
            k.to(torch.float32).cpu(),
            atol=1e-2,
        )
        assert torch.allclose(
            chunk.v[layer_id].to(torch.float32),
            v.to(torch.float32).cpu(),
            atol=1e-2,
        )
