"""Tests for KVArena and KVChunk.

Organized around the invariants declared in cache.py docstrings. Every
test that mutates arena state calls arena._check_invariants() afterwards
as the final assertion so any invariant regression surfaces as a test
failure regardless of what the test is specifically checking.
"""

from __future__ import annotations

import random

import torch

from torchllms.models.cache import KVArena, KVChunk, RolloutId


N_LAYERS = 3
MAX_BSZ = 4
MAX_SEQLEN = 16
N_KV_HEADS = 2
HEAD_DIM = 4
DTYPE = torch.float32


def _mk_arena(
    *,
    n_layers: int = N_LAYERS,
    max_bsz: int = MAX_BSZ,
    max_seqlen: int = MAX_SEQLEN,
) -> KVArena:
    return KVArena(
        n_layers=n_layers,
        max_bsz=max_bsz,
        max_seqlen=max_seqlen,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        device="cpu",
        dtype=DTYPE,
    )


def _mk_chunk(length: int, *, fill_k: float = 1.0, fill_v: float = -1.0, role_id: int = 1) -> KVChunk:
    k = torch.full((N_LAYERS, length, N_KV_HEADS, HEAD_DIM), fill_k, dtype=DTYPE)
    v = torch.full((N_LAYERS, length, N_KV_HEADS, HEAD_DIM), fill_v, dtype=DTYPE)
    role_ids = torch.full((length,), role_id, dtype=torch.long)
    return KVChunk(k=k, v=v, role_ids=role_ids)


def _mk_unique_chunk(length: int, *, seed: int) -> KVChunk:
    """Chunk with distinctive per-(layer, pos, head, dim) values so
    compaction swaps can be detected if they misroute data."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(N_LAYERS, length, N_KV_HEADS, HEAD_DIM, generator=g, dtype=DTYPE)
    v = torch.randn(N_LAYERS, length, N_KV_HEADS, HEAD_DIM, generator=g, dtype=DTYPE)
    role_ids = (torch.arange(length, dtype=torch.long) + seed) % 5
    return KVChunk(k=k, v=v, role_ids=role_ids)


# ============================================================== #
# KVChunk — I1–I4                                                 #
# ============================================================== #


def test_chunk_rejects_wrong_ndim():
    k = torch.zeros(3, 4, 2, 4)
    v = torch.zeros(3, 4, 2)  # 3-D instead of 4-D
    role_ids = torch.zeros(4, dtype=torch.long)
    try:
        KVChunk(k=k, v=v, role_ids=role_ids)
    except ValueError:
        return
    raise AssertionError("expected ValueError on wrong ndim")


def test_chunk_rejects_shape_mismatch():
    k = torch.zeros(3, 4, 2, 4)
    v = torch.zeros(3, 5, 2, 4)
    role_ids = torch.zeros(4, dtype=torch.long)
    try:
        KVChunk(k=k, v=v, role_ids=role_ids)
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_chunk_rejects_bad_role_ids():
    k = torch.zeros(3, 4, 2, 4)
    v = torch.zeros(3, 4, 2, 4)
    role_ids = torch.zeros(5, dtype=torch.long)  # length mismatch
    try:
        KVChunk(k=k, v=v, role_ids=role_ids)
    except ValueError:
        return
    raise AssertionError("expected ValueError on role_ids length mismatch")


def test_chunk_rejects_non_cpu_tensors():
    if not torch.cuda.is_available():
        return
    k = torch.zeros(3, 4, 2, 4, device="cuda")
    v = torch.zeros(3, 4, 2, 4, device="cuda")
    role_ids = torch.zeros(4, dtype=torch.long, device="cuda")
    try:
        KVChunk(k=k, v=v, role_ids=role_ids)
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-CPU tensors")


def test_chunk_slice_bounds():
    c = _mk_chunk(length=5)
    assert c.slice(0, 5).length == 5
    assert c.slice(1, 4).length == 3
    assert c.slice(2, 2).length == 0
    for bad in [(-1, 2), (0, 6), (3, 1)]:
        try:
            c.slice(*bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError on bad slice {bad}")


def test_chunk_slice_concat_roundtrip():
    c = _mk_unique_chunk(length=8, seed=7)
    head = c.slice(0, 3)
    tail = c.slice(3, 8)
    rejoined = KVChunk.concat([head, tail])
    assert torch.equal(rejoined.k, c.k)
    assert torch.equal(rejoined.v, c.v)
    assert torch.equal(rejoined.role_ids, c.role_ids)


def test_chunk_concat_empty_raises():
    try:
        KVChunk.concat([])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty concat")


# ============================================================== #
# KVArena — construction and identity (S1–S3, S6)                 #
# ============================================================== #


def test_arena_initial_state():
    arena = _mk_arena()
    assert arena.b_live == 0
    assert arena.active_rollouts() == []
    assert arena.row_positions.shape == (0,)
    arena._check_invariants()


def test_claim_assigns_fresh_slot_and_rid():
    arena = _mk_arena()
    rid_a = arena.claim()
    rid_b = arena.claim()
    assert rid_a != rid_b
    assert arena.b_live == 2
    assert arena.resolve(rid_a) == 0
    assert arena.resolve(rid_b) == 1
    assert arena.active_rollouts() == [rid_a, rid_b]
    arena._check_invariants()


def test_claim_when_full_raises():
    arena = _mk_arena(max_bsz=2)
    arena.claim()
    arena.claim()
    try:
        arena.claim()
    except RuntimeError:
        arena._check_invariants()
        return
    raise AssertionError("expected RuntimeError on full arena")


def test_resolve_unknown_rid_raises_keyerror():
    arena = _mk_arena()
    try:
        arena.resolve(RolloutId(999))
    except KeyError:
        return
    raise AssertionError("expected KeyError on unknown rid")


def test_active_rollouts_returns_copy():
    arena = _mk_arena()
    rid = arena.claim()
    snap = arena.active_rollouts()
    snap.clear()
    assert arena.active_rollouts() == [rid]


def test_rollout_ids_monotonic():
    arena = _mk_arena(max_bsz=4)
    rids = []
    # Interleave claim and retire so slot swapping happens; RolloutIds
    # must still come out strictly increasing.
    for _ in range(3):
        rids.append(arena.claim())
    arena.retire(rids[1])
    rids.append(arena.claim())
    rids.append(arena.claim())
    assert rids == sorted(rids)
    assert len(set(rids)) == len(rids)
    arena._check_invariants()


def test_uniform_seen_tokens_tracks_state():
    arena = _mk_arena(max_bsz=3)
    # Empty arena: trivially uniform.
    assert arena.uniform_seen_tokens is True
    r0 = arena.claim()
    r1 = arena.claim()
    assert arena.uniform_seen_tokens is True
    # Load unequal-length chunks → divergent seen_tokens.
    arena.load_chunk(_mk_chunk(length=3), r0)
    arena.load_chunk(_mk_chunk(length=5), r1)
    assert arena.uniform_seen_tokens is False
    # Set per-row to equal values → back to uniform.
    arena.set_seen_tokens_per_row(
        torch.tensor([4, 4], dtype=torch.long)
    )
    assert arena.uniform_seen_tokens is True


def test_set_seen_tokens_per_row_writes_all_layers():
    arena = _mk_arena(max_bsz=2)
    r0 = arena.claim()
    r1 = arena.claim()
    arena.load_chunk(_mk_chunk(length=5), r0)
    arena.load_chunk(_mk_chunk(length=5), r1)
    arena.set_seen_tokens_per_row(torch.tensor([3, 5], dtype=torch.long))
    assert bool((arena.seen_tokens[:, 0] == 3).all().item())
    assert bool((arena.seen_tokens[:, 1] == 5).all().item())
    arena._check_invariants()


def test_set_seen_tokens_per_row_rejects_bad_shape():
    arena = _mk_arena()
    arena.claim()
    arena.claim()
    try:
        arena.set_seen_tokens_per_row(torch.tensor([3], dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_set_seen_tokens_per_row_rejects_out_of_range():
    arena = _mk_arena(max_bsz=2, max_seqlen=16)
    arena.claim()
    arena.claim()
    arena.load_chunk(_mk_chunk(length=5), arena.active_rollouts()[0])
    arena.load_chunk(_mk_chunk(length=5), arena.active_rollouts()[1])
    for bad in [torch.tensor([-1, 5]), torch.tensor([5, 17])]:
        try:
            arena.set_seen_tokens_per_row(bad.to(torch.long))
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError on {bad.tolist()}")


# ============================================================== #
# KVArena — load_chunk / extract_chunk round trips                #
# ============================================================== #


def test_load_chunk_sets_seen_tokens_all_layers():
    arena = _mk_arena()
    rid = arena.claim()
    chunk = _mk_unique_chunk(length=7, seed=0)
    arena.load_chunk(chunk, rid)
    slot = arena.resolve(rid)
    assert bool((arena.seen_tokens[:, slot] == 7).all().item())
    assert int(arena.row_positions[slot].item()) == 7
    arena._check_invariants()


def test_load_chunk_at_nonzero_at_pos_raises():
    arena = _mk_arena()
    rid = arena.claim()
    try:
        arena.load_chunk(_mk_chunk(length=2), rid, at_pos=1)
    except NotImplementedError:
        arena._check_invariants()
        return
    raise AssertionError("expected NotImplementedError on at_pos != 0")


def test_load_chunk_into_nonempty_slot_raises():
    arena = _mk_arena()
    rid = arena.claim()
    arena.load_chunk(_mk_chunk(length=3), rid)
    try:
        arena.load_chunk(_mk_chunk(length=2), rid)
    except RuntimeError:
        arena._check_invariants()
        return
    raise AssertionError("expected RuntimeError on load into non-empty slot")


def test_load_chunk_wrong_n_layers_raises():
    arena = _mk_arena()
    rid = arena.claim()
    bad_k = torch.zeros(N_LAYERS + 1, 3, N_KV_HEADS, HEAD_DIM)
    bad_v = torch.zeros(N_LAYERS + 1, 3, N_KV_HEADS, HEAD_DIM)
    bad_role = torch.zeros(3, dtype=torch.long)
    bad = KVChunk(k=bad_k, v=bad_v, role_ids=bad_role)
    try:
        arena.load_chunk(bad, rid)
    except ValueError:
        arena._check_invariants()
        return
    raise AssertionError("expected ValueError on wrong n_layers")


def test_load_chunk_wrong_head_shape_raises():
    arena = _mk_arena()
    rid = arena.claim()
    bad_k = torch.zeros(N_LAYERS, 3, N_KV_HEADS + 1, HEAD_DIM)
    bad_v = torch.zeros(N_LAYERS, 3, N_KV_HEADS + 1, HEAD_DIM)
    bad_role = torch.zeros(3, dtype=torch.long)
    bad = KVChunk(k=bad_k, v=bad_v, role_ids=bad_role)
    try:
        arena.load_chunk(bad, rid)
    except ValueError:
        arena._check_invariants()
        return
    raise AssertionError("expected ValueError on wrong head shape")


def test_extract_chunk_round_trip():
    arena = _mk_arena()
    rid = arena.claim()
    chunk = _mk_unique_chunk(length=6, seed=42)
    arena.load_chunk(chunk, rid)
    out = arena.extract_chunk(rid)
    assert out.length == chunk.length
    assert torch.equal(out.k, chunk.k)
    assert torch.equal(out.v, chunk.v)
    assert torch.equal(out.role_ids, chunk.role_ids)
    arena._check_invariants()


def test_extract_chunk_partial_length():
    arena = _mk_arena()
    rid = arena.claim()
    chunk = _mk_unique_chunk(length=8, seed=11)
    arena.load_chunk(chunk, rid)
    partial = arena.extract_chunk(rid, length=5)
    assert partial.length == 5
    assert torch.equal(partial.k, chunk.k[:, :5])
    assert torch.equal(partial.v, chunk.v[:, :5])
    assert torch.equal(partial.role_ids, chunk.role_ids[:5])


def test_extract_chunk_length_bounds():
    arena = _mk_arena()
    rid = arena.claim()
    arena.load_chunk(_mk_chunk(length=4), rid)
    for bad in [-1, 5, 100]:
        try:
            arena.extract_chunk(rid, length=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError on length={bad}")


# ============================================================== #
# KVArena — update_kv / update_role_ids / update_attn_mask        #
# ============================================================== #


def test_update_kv_lockstep_prefill():
    arena = _mk_arena()
    arena.claim()
    arena.claim()
    S = 4
    k = torch.full((2, S, N_KV_HEADS, HEAD_DIM), 3.0, dtype=DTYPE)
    v = torch.full((2, S, N_KV_HEADS, HEAD_DIM), -3.0, dtype=DTYPE)
    for layer in range(N_LAYERS):
        k_full, v_full = arena.update_kv(layer, k, v)
        assert k_full.shape == (2, S, N_KV_HEADS, HEAD_DIM)
        assert v_full.shape == (2, S, N_KV_HEADS, HEAD_DIM)
        assert bool((arena.seen_tokens[layer, :2] == S).all().item())
    arena._check_invariants()


def test_update_kv_advances_only_target_layer():
    arena = _mk_arena()
    arena.claim()
    S = 2
    k = torch.zeros((1, S, N_KV_HEADS, HEAD_DIM), dtype=DTYPE)
    v = torch.zeros((1, S, N_KV_HEADS, HEAD_DIM), dtype=DTYPE)
    arena.update_kv(0, k, v)
    assert int(arena.seen_tokens[0, 0].item()) == S
    for other in range(1, N_LAYERS):
        assert int(arena.seen_tokens[other, 0].item()) == 0
    # Not asserting _check_invariants — S7 is transiently violated mid-forward.


def test_update_kv_decode_single_token_diverging_positions():
    arena = _mk_arena()
    r0 = arena.claim()
    r1 = arena.claim()
    # Row 0 prefilled to length 3; row 1 fresh. Positions diverge.
    arena.load_chunk(_mk_unique_chunk(length=3, seed=1), r0)
    assert int(arena.row_positions[0].item()) == 3
    assert int(arena.row_positions[1].item()) == 0

    k = torch.full((2, 1, N_KV_HEADS, HEAD_DIM), 9.0, dtype=DTYPE)
    v = torch.full((2, 1, N_KV_HEADS, HEAD_DIM), -9.0, dtype=DTYPE)
    for layer in range(N_LAYERS):
        arena.update_kv(layer, k, v)
    # Row 0 seen 3+1=4; row 1 seen 0+1=1.
    assert int(arena.row_positions[0].item()) == 4
    assert int(arena.row_positions[1].item()) == 1
    # Row 0's new token landed at position 3, not 0.
    assert bool((arena.k_cache[:, 0, 3, :, :] == 9.0).all().item())
    # Row 1's new token landed at position 0.
    assert bool((arena.k_cache[:, 1, 0, :, :] == 9.0).all().item())
    arena._check_invariants()


def test_update_kv_rejects_wrong_batch():
    arena = _mk_arena()
    arena.claim()
    k = torch.zeros((2, 1, N_KV_HEADS, HEAD_DIM), dtype=DTYPE)
    try:
        arena.update_kv(0, k, k)
    except ValueError:
        return
    raise AssertionError("expected ValueError on batch mismatch")


def test_update_kv_rejects_past_max_seqlen():
    arena = _mk_arena(max_seqlen=4)
    arena.claim()
    arena.load_chunk(_mk_chunk(length=3), arena.active_rollouts()[0])
    k = torch.zeros((1, 2, N_KV_HEADS, HEAD_DIM), dtype=DTYPE)
    try:
        arena.update_kv(0, k, k)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on overflow")


def test_update_role_ids_none_returns_none():
    arena = _mk_arena()
    arena.claim()
    assert arena.update_role_ids(None) is None


def test_update_role_ids_diverging_positions():
    arena = _mk_arena()
    r0 = arena.claim()
    r1 = arena.claim()
    arena.load_chunk(_mk_unique_chunk(length=3, seed=99), r0)
    role_ids = torch.tensor([[2], [4]], dtype=torch.long)
    out = arena.update_role_ids(role_ids)
    assert out is not None
    # Row 0's new role landed at position 3; row 1's at position 0.
    assert int(arena.role_id_cache[0, 3].item()) == 2
    assert int(arena.role_id_cache[1, 0].item()) == 4


def test_update_attn_mask_auto_extend_requires_prior_write():
    arena = _mk_arena()
    arena.claim()
    # No prior explicit mask set; None should yield None.
    assert arena.update_attn_mask(None) is None
    # Set explicit mask.
    mask = torch.ones((1, 1), dtype=torch.long)
    out = arena.update_attn_mask(mask)
    assert out is not None and arena.is_attn_mask_cached
    # Advance a token via update_kv so auto-extend writes at the right slot.
    k = torch.zeros((1, 1, N_KV_HEADS, HEAD_DIM), dtype=DTYPE)
    for layer in range(N_LAYERS):
        arena.update_kv(layer, k, k)
    # Auto-extend should now populate position 1.
    out2 = arena.update_attn_mask(None)
    assert out2 is not None
    assert int(arena.attn_mask_cache[0, 1].item()) == 1


def test_is_full_reflects_max_seqlen():
    arena = _mk_arena(max_seqlen=4)
    rid = arena.claim()
    assert not arena.is_full()
    arena.load_chunk(_mk_chunk(length=4), rid)
    assert arena.is_full()


# ============================================================== #
# KVArena — retirement and compaction (S2–S3, S7)                 #
# ============================================================== #


def test_retire_returns_chunk_matching_prior_extract():
    arena = _mk_arena()
    rid = arena.claim()
    orig = _mk_unique_chunk(length=5, seed=17)
    arena.load_chunk(orig, rid)
    retired = arena.retire(rid)
    assert torch.equal(retired.k, orig.k)
    assert torch.equal(retired.v, orig.v)
    assert torch.equal(retired.role_ids, orig.role_ids)
    arena._check_invariants()


def test_retire_invalidates_rollout_id():
    arena = _mk_arena()
    rid = arena.claim()
    arena.retire(rid)
    try:
        arena.resolve(rid)
    except KeyError:
        arena._check_invariants()
        return
    raise AssertionError("expected KeyError after retire")


def test_retire_middle_slot_preserves_others_data():
    arena = _mk_arena(max_bsz=4)
    rids = [arena.claim() for _ in range(4)]
    chunks = [_mk_unique_chunk(length=3 + i, seed=100 + i) for i in range(4)]
    for rid, chunk in zip(rids, chunks):
        arena.load_chunk(chunk, rid)
    # Retire slot 1 (the second claim).
    arena.retire(rids[1])
    arena._check_invariants()
    # Surviving rollouts' extracted KV must exactly match what was loaded.
    for i in (0, 2, 3):
        out = arena.extract_chunk(rids[i])
        assert torch.equal(out.k, chunks[i].k), f"k mismatch at rid {i}"
        assert torch.equal(out.v, chunks[i].v), f"v mismatch at rid {i}"
        assert torch.equal(out.role_ids, chunks[i].role_ids), f"role mismatch at rid {i}"


def test_retire_last_slot_does_not_swap():
    arena = _mk_arena(max_bsz=3)
    rids = [arena.claim() for _ in range(3)]
    chunks = [_mk_unique_chunk(length=2 + i, seed=200 + i) for i in range(3)]
    for rid, chunk in zip(rids, chunks):
        arena.load_chunk(chunk, rid)
    arena.retire(rids[-1])
    arena._check_invariants()
    # rids[0] and rids[1] still at slots 0 and 1; no swap occurred.
    assert arena.resolve(rids[0]) == 0
    assert arena.resolve(rids[1]) == 1


def test_retire_many_returns_chunks_in_input_order():
    arena = _mk_arena(max_bsz=4)
    rids = [arena.claim() for _ in range(4)]
    chunks = [_mk_unique_chunk(length=3, seed=300 + i) for i in range(4)]
    for rid, chunk in zip(rids, chunks):
        arena.load_chunk(chunk, rid)
    # Retire out-of-slot-order (2, 0, 3) — tests both extraction order and
    # descending-slot-swap correctness.
    to_retire = [rids[2], rids[0], rids[3]]
    retired = arena.retire_many(to_retire)
    assert len(retired) == 3
    assert torch.equal(retired[0].k, chunks[2].k)
    assert torch.equal(retired[1].k, chunks[0].k)
    assert torch.equal(retired[2].k, chunks[3].k)
    # Survivor is rid[1].
    arena._check_invariants()
    assert arena.active_rollouts() == [rids[1]]
    out = arena.extract_chunk(rids[1])
    assert torch.equal(out.k, chunks[1].k)


def test_retire_many_matches_sequential():
    # Two arenas with identical state; one retires sequentially, the other
    # uses retire_many. Final surviving-slot data should be identical.
    def scenario(use_retire_many: bool) -> list[torch.Tensor]:
        arena = _mk_arena(max_bsz=5)
        rids = [arena.claim() for _ in range(5)]
        chunks = [_mk_unique_chunk(length=4, seed=400 + i) for i in range(5)]
        for rid, chunk in zip(rids, chunks):
            arena.load_chunk(chunk, rid)
        targets = [rids[1], rids[3], rids[0]]
        if use_retire_many:
            arena.retire_many(targets)
        else:
            for t in targets:
                arena.retire(t)
        arena._check_invariants()
        survivors = arena.active_rollouts()
        # Return k tensors of survivors in RolloutId order for comparison.
        ks = []
        for rid in sorted(survivors):
            ks.append(arena.extract_chunk(rid).k)
        return ks

    a = scenario(use_retire_many=False)
    b = scenario(use_retire_many=True)
    assert len(a) == len(b) == 2
    for ka, kb in zip(a, b):
        assert torch.equal(ka, kb)


def test_retire_many_duplicate_rids_raises():
    arena = _mk_arena()
    r = arena.claim()
    try:
        arena.retire_many([r, r])
    except ValueError:
        return
    raise AssertionError("expected ValueError on duplicate rids")


def test_rollout_id_resolve_stable_across_other_retires():
    arena = _mk_arena(max_bsz=5)
    rids = [arena.claim() for _ in range(5)]
    survivor = rids[3]
    survivor_chunk = _mk_unique_chunk(length=5, seed=555)
    arena.load_chunk(survivor_chunk, survivor)
    # Retire the other four in arbitrary order.
    for r in [rids[0], rids[4], rids[1], rids[2]]:
        arena.retire(r)
        arena._check_invariants()
        # Survivor's rid is still resolvable, and its data is unchanged.
        out = arena.extract_chunk(survivor)
        assert torch.equal(out.k, survivor_chunk.k), (
            f"survivor data corrupted after retiring {r}"
        )
    assert arena.b_live == 1
    assert arena.resolve(survivor) == 0


# ============================================================== #
# Property-style randomized test                                   #
# ============================================================== #


def test_random_ops_preserve_invariants():
    rng = random.Random(0)
    arena = _mk_arena(max_bsz=6, max_seqlen=32)
    loaded_by_rid: dict = {}

    for _ in range(150):
        choices = []
        if arena.b_live < arena.max_bsz:
            choices.append("claim")
        if arena.b_live > 0:
            choices.extend(["retire", "extract", "load_or_decode"])
        op = rng.choice(choices)

        if op == "claim":
            arena.claim()
        elif op == "retire":
            rid = rng.choice(arena.active_rollouts())
            chunk = arena.retire(rid)
            if rid in loaded_by_rid:
                ref = loaded_by_rid.pop(rid)
                assert torch.equal(chunk.k, ref.k), "retire returned wrong data"
        elif op == "extract":
            rid = rng.choice(arena.active_rollouts())
            arena.extract_chunk(rid)
        elif op == "load_or_decode":
            rid = rng.choice(arena.active_rollouts())
            slot = arena.resolve(rid)
            if int(arena.seen_tokens[0, slot].item()) == 0:
                length = rng.randint(1, 6)
                chunk = _mk_unique_chunk(length=length, seed=_ * 31 + slot)
                arena.load_chunk(chunk, rid)
                loaded_by_rid[rid] = chunk
            else:
                # Simulate one decode step.
                cur = int(arena.seen_tokens[0, slot].item())
                if cur + 1 > arena.max_seqlen:
                    continue
                B = arena.b_live
                k = torch.zeros((B, 1, N_KV_HEADS, HEAD_DIM), dtype=DTYPE)
                v = torch.zeros((B, 1, N_KV_HEADS, HEAD_DIM), dtype=DTYPE)
                # Only simulate the decode if no row would overflow.
                if bool((arena.seen_tokens[0, :B] + 1 <= arena.max_seqlen).all().item()):
                    for layer in range(arena.n_layers):
                        arena.update_kv(layer, k, v)
                    # update_kv writes to every live row; all prior
                    # loaded_by_rid references are now stale.
                    loaded_by_rid.clear()

        arena._check_invariants()


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
