"""Paged KV cache for torchllms inference.

Replaces KVArena's dense ``[n_layers, max_bsz, max_seqlen, ...]`` allocation
with a page-based layout that flashinfer's paged-attention kernels consume
directly. Key differences from KVArena:

- KV is stored in pages (fixed-size blocks of ``page_size`` tokens). A single
  global pool holds ``total_pages`` pages across ALL live sequences; each
  sequence's KV is identified by a list of page indices. No per-row padding.
- Memory is sized by total GPU-budget (page count × bytes-per-page), not by
  (max_bsz × max_seqlen). Sequences with wildly different lengths share the
  same pool efficiently.
- Free pages are recycled on retire(), so the pool survives across many
  generate() calls without reallocating.

Page pool layout (per layer)::

    k_cache[layer_id]:  [total_pages, page_size, n_kv_heads, head_dim]
    v_cache[layer_id]:  [total_pages, page_size, n_kv_heads, head_dim]

flashinfer's NHD layout matches this directly — (N=page_size, H=n_kv_heads,
D=head_dim).

Block table per live sequence::

    _rollout_to_pages[rid] = [page_0, page_1, ..., page_k]    # logical order
    _rollout_to_seqlen[rid] = L          # total valid tokens in this sequence

A sequence of length L occupies ``ceil(L / page_size)`` pages, with the last
page partially filled (``last_page_len = L - (num_pages - 1) * page_size``).

The forward-call protocol is::

    pre_write = pool.row_positions(active_rids)   # [B] int32, pre-extend
    pool.extend_many(active_rids, qlens)          # allocate pages
    layout = pool.build_batch_layout(active_rids, qlens)
    # ... plan flashinfer wrapper with layout ...
    for layer_id in range(n_layers):
        pool.append_kv(layer_id, k_flat, v_flat, layout)  # scatter
        out = wrapper.run(q_flat, (pool.k_cache[layer_id], pool.v_cache[layer_id]))

``append_kv`` is wrapped as a ``torch.library.custom_op`` in
``torchllms.models.compile_ops`` so Dynamo tracks the in-place KV writes
through a compiled decode forward.

Phase 1 compatibility: ``load_chunk(KVChunk, rid)`` + ``extract_chunk(rid)
-> KVChunk`` preserve the KVChunk round-trip that today's CPU-resident
``RadixKVCache`` depends on. Phase 2 replaces the radix with a GPU-resident
page-ID store and deletes both shims.

No ``role_id_cache`` / ``attn_mask_cache``: the history-shaped tensors those
buffers produced in KVArena were never consumed by the flashinfer/FA paged
attention path — base ``Transformer.forward`` passed them to layers whose
attention modules ignored them. Activation interventions see only the
*active* (current-forward's) ``role_ids`` for mask construction; see
``test_activation_hooks.test_role_filter_edits_only_selected_positions``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from torchllms.models.cache import KVChunk, RolloutId


# =====================================================================
# Flashinfer batch layout descriptor
# =====================================================================


@dataclass(frozen=True)
class PagedBatchLayout:
    """Flashinfer plan tensors for a batch of live sequences.

    Produced by :meth:`PagedKVPool.build_batch_layout`. All tensors live on
    the pool's device in int32 (flashinfer's idtype).

        ``qo_indptr[i+1] - qo_indptr[i]`` = new query tokens for seq ``i``.
        ``kv_indptr[i+1] - kv_indptr[i]`` = page count for seq ``i`` after
            the upcoming write (i.e. post-extend).
        ``kv_indices[kv_indptr[i]:kv_indptr[i+1]]`` = that seq's page list.
        ``kv_last_page_len[i]`` = valid tokens in seq ``i``'s final page
            (post-extend).
        ``batch_indices[t]`` / ``positions[t]`` = absolute (row, pos) for
            the ``t``-th new token, in the same order the caller will pack
            ``k_new`` / ``v_new`` into ``[total_new_tokens, ...]`` for
            ``append_kv``. Uniform-qlen packing is ``b_j = b, p_j =
            pre_write[b] + j``.
    """

    qo_indptr: torch.Tensor           # [B+1] int32
    kv_indptr: torch.Tensor           # [B+1] int32
    kv_indices: torch.Tensor          # [sum_of_pages] int32
    kv_last_page_len: torch.Tensor    # [B] int32
    batch_indices: torch.Tensor       # [total_new_tokens] int32
    positions: torch.Tensor           # [total_new_tokens] int32
    # Bookkeeping (not sent to flashinfer; kept for caller inspection).
    rollout_ids: List[RolloutId]
    qlens: List[int]
    seqlens_before_write: List[int]

    @property
    def batch_size(self) -> int:
        return len(self.rollout_ids)

    @property
    def total_new_tokens(self) -> int:
        return sum(self.qlens)


# =====================================================================
# PagedKVPool
# =====================================================================


class PagedKVPool:
    """Fixed-capacity page pool for multi-layer KV cache.

    All live sequences share the same pool. Each sequence has its own
    list of page indices tracked in ``_rollout_to_pages``. Pages are
    recycled to ``_free_pages`` on retire.

    Structural invariants (hold after every public method returns):
        S1. ``set(_free_pages).isdisjoint(all_claimed_pages)``.
        S2. ``set(_free_pages) | all_claimed_pages == range(total_pages)``.
        S3. For each rid: ``len(_rollout_to_pages[rid]) ==
            ceil(_rollout_to_seqlen[rid] / page_size)``.
        S4. ``RolloutId`` values are strictly monotonic; never reused.
    """

    def __init__(
        self,
        *,
        n_layers: int,
        total_pages: int,
        page_size: int,
        n_kv_heads: int,
        head_dim: int,
        max_bsz: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if n_layers <= 0 or total_pages <= 0 or page_size <= 0 or max_bsz <= 0:
            raise ValueError(
                f"invalid dims: n_layers={n_layers} total_pages={total_pages} "
                f"page_size={page_size} max_bsz={max_bsz}"
            )
        dev = torch.device(device) if not isinstance(device, torch.device) else device

        # flashinfer NHD paged layout, one slab per layer.
        kv_shape = (n_layers, total_pages, page_size, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(kv_shape, device=dev, dtype=dtype)
        self.v_cache = torch.zeros_like(self.k_cache)

        self.n_layers = n_layers
        self.total_pages = total_pages
        self.page_size = page_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_bsz = max_bsz
        self.device = dev
        self.dtype = dtype

        # Pre-allocated metadata buffers for the batch layout
        # (``kv_indptr`` / ``kv_indices`` / ``kv_last_page_len``) sized
        # to the pool's capacity. Two reasons they have to be stable
        # addresses and stable shapes:
        #
        #   1. Dynamo specializes on tensor shapes. ``kv_indices``
        #      grows by one entry each time a rollout crosses a page
        #      boundary — a variable-shape tensor would trigger an
        #      Inductor recompile every page_size'th decode step (we
        #      observed 51 distinct sizes across a 54-token decode).
        #      Fixed-shape + partial fill (flashinfer reads only up
        #      to ``kv_indptr[-1]``, trailing entries are ignored)
        #      keeps the compiled graph shape-stable.
        #
        #   2. The SGLang-style flashinfer cudagraph path requires the
        #      decode wrapper to be constructed with
        #      ``use_cuda_graph=True`` + pre-bound indptr/indices/
        #      last_page_len buffers. cudagraph captures kernel
        #      invocations by pointer; subsequent calls read whatever
        #      is at those addresses at replay time. Binding these
        #      fixed buffers to the wrapper and updating their
        #      CONTENTS (not reallocating) per forward is what lets
        #      cudagraph capture the decode sequence correctly.
        self._kv_indices_buf = torch.zeros(total_pages, dtype=torch.int32, device=dev)
        self._kv_indptr_buf = torch.zeros(max_bsz + 1, dtype=torch.int32, device=dev)
        self._kv_last_page_len_buf = torch.zeros(max_bsz, dtype=torch.int32, device=dev)
        # qo_indptr — per-row cumulative sum of new query tokens. Prefill
        # planning reads this; decode can reuse it as arange(B+1) but we
        # populate it unconditionally so both wrappers have a stable
        # pool-bound buffer address for ``use_cuda_graph=True`` mode.
        self._qo_indptr_buf = torch.zeros(max_bsz + 1, dtype=torch.int32, device=dev)

        # mark_static_address on every buffer the compiled decode
        # forward mutates in place. Without this Inductor's
        # cudagraph-safety pass refuses to capture the compiled graph
        # ("skipping cudagraphs due to mutated inputs"). The paged K/V
        # slabs are mutated by flashinfer's append_paged_kv_cache via
        # our paged_append_kv custom op; the indptr/indices/
        # last_page_len buffers are mutated by wrapper.plan() (it
        # copies the per-forward layout into its bound buffers). All
        # these tensors live for the pool's lifetime — their addresses
        # are stable, which is what cudagraph capture needs.
        torch._dynamo.mark_static_address(self.k_cache)
        torch._dynamo.mark_static_address(self.v_cache)
        torch._dynamo.mark_static_address(self._kv_indices_buf)
        torch._dynamo.mark_static_address(self._kv_indptr_buf)
        torch._dynamo.mark_static_address(self._kv_last_page_len_buf)
        torch._dynamo.mark_static_address(self._qo_indptr_buf)

        # Page allocator + refcount tracker. ``_page_refcount[p]`` is the
        # number of active holders of page p — live rollouts with p in
        # their block table *plus* radix nodes that adopted p. A page is
        # reclaimable to ``_free_pages`` iff its refcount is 0.
        #
        # Two references to keep in sync:
        # - ``_page_refcount`` (authoritative ownership count)
        # - ``_free_pages`` (fast-path for alloc; contains exactly the pages
        #   whose refcount is 0)
        self._page_refcount: List[int] = [0] * total_pages
        self._free_pages: List[int] = list(range(total_pages))  # pop from end

        # Per-rollout state. Dict iteration order is claim order (Python 3.7+
        # guarantee) so ``active_rollouts()`` returns in claim order.
        self._rollout_to_pages: Dict[RolloutId, List[int]] = {}
        self._rollout_to_seqlen: Dict[RolloutId, int] = {}
        self._next_rollout_id: int = 0

    # ---------------------- Identity / introspection -------------------

    @property
    def b_live(self) -> int:
        return len(self._rollout_to_pages)

    @property
    def free_page_count(self) -> int:
        return len(self._free_pages)

    def active_rollouts(self) -> List[RolloutId]:
        return list(self._rollout_to_pages.keys())

    def seqlen(self, rid: RolloutId) -> int:
        return self._rollout_to_seqlen[rid]

    def pages_of(self, rid: RolloutId) -> List[int]:
        return list(self._rollout_to_pages[rid])

    def row_positions(
        self, active_rids: Optional[Sequence[RolloutId]] = None,
    ) -> torch.Tensor:
        """Return ``[B] int32`` pre-extend seqlens for the given rollouts
        (in the order provided). If ``active_rids`` is None, uses the
        current ``active_rollouts()`` order.

        This is the RoPE input_pos base — ``input_pos[b, j] =
        row_positions[b] + j`` for the b-th live row's j-th new token.
        Analog of ``KVArena.row_positions`` but parameterized on rid order.
        """
        if active_rids is None:
            active_rids = self.active_rollouts()
        vals = [self._rollout_to_seqlen[rid] for rid in active_rids]
        return torch.tensor(vals, dtype=torch.int32, device=self.device)

    # ---------------------- Allocation ---------------------------------

    def claim(self) -> RolloutId:
        """Allocate a new rollout with zero pages / zero valid tokens.

        Pages are allocated lazily by :meth:`extend` / :meth:`extend_many`
        as the sequence grows.
        """
        rid = RolloutId(self._next_rollout_id)
        self._next_rollout_id += 1
        self._rollout_to_pages[rid] = []
        self._rollout_to_seqlen[rid] = 0
        return rid

    def _alloc_page(self) -> int:
        """Pop a free (ref==0) page, bump its refcount to 1, return id."""
        if not self._free_pages:
            raise RuntimeError(
                f"PagedKVPool out of pages: all {self.total_pages} pages "
                f"have refcount > 0 (held by live rollouts or radix nodes)"
            )
        p = self._free_pages.pop()
        assert self._page_refcount[p] == 0, (
            f"free-list invariant broken: page {p} has refcount "
            f"{self._page_refcount[p]}"
        )
        self._page_refcount[p] = 1
        return p

    def borrow_pages(self, page_ids: Sequence[int]) -> None:
        """Increment the refcount on each page. Used when a rollout or a
        radix node newly references an existing page. Caller is
        responsible for matching this with :meth:`release_pages`.
        """
        for p in page_ids:
            if not (0 <= p < self.total_pages):
                raise ValueError(f"bad page id {p}")
            if self._page_refcount[p] == 0:
                raise RuntimeError(
                    f"borrow_pages: page {p} has refcount 0 (not currently "
                    "held by anyone — borrow should only follow an alloc)"
                )
            self._page_refcount[p] += 1

    def release_pages(self, page_ids: Sequence[int]) -> None:
        """Decrement the refcount on each page. Pages whose refcount hits
        zero are returned to the free list for recycling.
        """
        for p in page_ids:
            if not (0 <= p < self.total_pages):
                raise ValueError(f"bad page id {p}")
            if self._page_refcount[p] <= 0:
                raise RuntimeError(
                    f"release_pages: page {p} has refcount "
                    f"{self._page_refcount[p]} (double-release)"
                )
            self._page_refcount[p] -= 1
            if self._page_refcount[p] == 0:
                self._free_pages.append(p)

    def page_refcount(self, page_id: int) -> int:
        return self._page_refcount[page_id]

    def extend(self, rid: RolloutId, n_new_tokens: int) -> None:
        """Grow this rollout by n_new_tokens. Allocates fresh pages (via
        :meth:`_alloc_page`, ref=1) as needed.

        Raises ``RuntimeError`` if the pool is out of pages.
        """
        if rid not in self._rollout_to_pages:
            raise KeyError(f"unknown rollout {rid!r}")
        if n_new_tokens < 0:
            raise ValueError(f"n_new_tokens={n_new_tokens} must be >= 0")
        if n_new_tokens == 0:
            return
        pages = self._rollout_to_pages[rid]
        cur_len = self._rollout_to_seqlen[rid]
        new_len = cur_len + n_new_tokens
        pages_needed = (new_len + self.page_size - 1) // self.page_size
        while len(pages) < pages_needed:
            pages.append(self._alloc_page())
        self._rollout_to_seqlen[rid] = new_len

    def extend_many(
        self, rids: Sequence[RolloutId], qlens: Sequence[int],
    ) -> None:
        """Extend each of ``rids`` by the corresponding ``qlens`` entry.

        Standard forward-call convenience: read pre-extend seqlens for RoPE,
        then call this, then :meth:`build_batch_layout`.
        """
        if len(rids) != len(qlens):
            raise ValueError(
                f"len(rids)={len(rids)} vs len(qlens)={len(qlens)}"
            )
        for rid, q in zip(rids, qlens):
            self.extend(rid, int(q))

    def clamp_seqlens_per_row(
        self,
        active_rids: Sequence[RolloutId],
        true_lens: Sequence[int],
    ) -> None:
        """Shorten each rollout's seqlen to ``true_lens[i]`` and release
        the now-excess trailing pages.

        Used by right-padded batched prefill: the forward writes K/V for
        ``max_prompt_len`` tokens per row, then the caller clips each
        row down to its real prompt length so subsequent decode-step
        writes land at the correct per-row position. KVArena's
        ``set_seen_tokens_per_row`` analog — but because the paged pool
        actually allocated pages for the padded positions, it must also
        release the excess.

        Each ``true_lens[i]`` must be ``<= pool.seqlen(active_rids[i])``
        — this method only ever shrinks. K/V data in released pages is
        discarded.
        """
        if len(active_rids) != len(true_lens):
            raise ValueError(
                f"len(active_rids)={len(active_rids)} vs "
                f"len(true_lens)={len(true_lens)}"
            )
        for rid, L in zip(active_rids, true_lens):
            L = int(L)
            cur_len = self._rollout_to_seqlen[rid]
            if L > cur_len:
                raise ValueError(
                    f"clamp cannot extend: rid={rid!r} L={L} > cur={cur_len}"
                )
            pages = self._rollout_to_pages[rid]
            new_n_pages = (L + self.page_size - 1) // self.page_size
            excess = pages[new_n_pages:]
            if excess:
                self.release_pages(excess)
                self._rollout_to_pages[rid] = pages[:new_n_pages]
            self._rollout_to_seqlen[rid] = L

    def attach_borrowed_pages(
        self, rid: RolloutId, page_ids: Sequence[int],
    ) -> None:
        """Prepend borrowed pages to a rollout's (empty) block table and
        set its seqlen to ``len(page_ids) * page_size``. Used on
        prefix-cache hit: the radix trie hands us a page list and we
        ``borrow_pages`` them (caller's responsibility).

        Precondition: rollout is empty (seqlen == 0).
        Post: rollout's block table == page_ids, seqlen == len * page_size.
        """
        if rid not in self._rollout_to_pages:
            raise KeyError(f"unknown rollout {rid!r}")
        if self._rollout_to_seqlen[rid] != 0:
            raise RuntimeError(
                f"attach_borrowed_pages into non-empty rollout {rid!r}: "
                f"seqlen={self._rollout_to_seqlen[rid]}"
            )
        self._rollout_to_pages[rid] = list(page_ids)
        self._rollout_to_seqlen[rid] = len(page_ids) * self.page_size

    # ---------------------- Retirement ---------------------------------

    def retire(self, rid: RolloutId) -> KVChunk:
        """Release this rollout's pages and return the full KVChunk (CPU
        copy) so the caller can hand it to the ``RadixKVCache`` prefix
        store. Pages go through refcount decrement — pages with other
        refs (e.g. already adopted by a radix node) survive the retire.

        Phase 1 behavior. Phase 2 uses :meth:`retire_pages` + the
        GPU-resident radix and drops the CPU round-trip entirely.
        """
        if rid not in self._rollout_to_pages:
            raise KeyError(f"unknown rollout {rid!r}")
        length = self._rollout_to_seqlen[rid]
        chunk = self._extract(rid, length)
        self._release_rollout(rid)
        return chunk

    def retire_many(self, rids: Sequence[RolloutId]) -> List[KVChunk]:
        if len(set(rids)) != len(rids):
            raise ValueError("retire_many: duplicate rids")
        out: List[KVChunk] = []
        for rid in rids:
            out.append(self.retire(rid))
        return out

    def retire_pages(self, rid: RolloutId) -> tuple[List[int], int]:
        """Phase-2 retirement. Returns ``(page_ids, seqlen)`` and removes
        the rollout from tracking, but does **not** decrement page
        refcounts — caller owns the returned refs and is responsible for
        eventually calling :meth:`release_pages` (typically after handing
        any page-aligned prefix to the radix for adoption).

        Flow::

            page_ids, seqlen = pool.retire_pages(rid)
            n_full = seqlen // pool.page_size
            if n_full > 0:
                radix.insert(tokens[:n_full * pool.page_size], page_ids[:n_full])
                # radix.insert bumped refs on newly-adopted pages.
            pool.release_pages(page_ids)
            # Pages only radix holds (or fully-new + newly-adopted) stay
            # alive with ref >= 1. Pages the rollout owned exclusively and
            # didn't hand to radix go ref 1→0 → back to free list.
        """
        if rid not in self._rollout_to_pages:
            raise KeyError(f"unknown rollout {rid!r}")
        pages = list(self._rollout_to_pages.pop(rid))
        seqlen = self._rollout_to_seqlen.pop(rid)
        return pages, seqlen

    def _release_rollout(self, rid: RolloutId) -> None:
        pages = self._rollout_to_pages.pop(rid)
        del self._rollout_to_seqlen[rid]
        self.release_pages(pages)

    # ---------------------- Layout building ----------------------------

    def build_batch_layout(
        self,
        active_rids: Sequence[RolloutId],
        qlens: Sequence[int],
    ) -> PagedBatchLayout:
        """Build flashinfer plan tensors + per-token scatter indices for
        the upcoming KV write. Assumes :meth:`extend_many` has already run
        (so the rollouts' post-write seqlens reflect the full write).

        Uniform-qlen fast path (all qlens equal) uses vectorized
        ``arange``/``repeat_interleave`` to build ``batch_indices`` and
        ``positions`` without a Python loop. Falls back to a Python loop
        for ragged ``qlens``.
        """
        if len(active_rids) != len(qlens):
            raise ValueError(
                f"len(active_rids)={len(active_rids)} vs len(qlens)={len(qlens)}"
            )
        B = len(active_rids)
        qlens_list = [int(q) for q in qlens]

        seqlens_before = [
            self._rollout_to_seqlen[rid] - qlens_list[i]
            for i, rid in enumerate(active_rids)
        ]
        # extend_many guarantees post >= pre; if a caller hands us a qlen
        # larger than the actual extend, this catches it.
        if any(s < 0 for s in seqlens_before):
            raise RuntimeError(
                f"build_batch_layout: negative pre-write seqlen "
                f"(extend not called or q too large): {seqlens_before}"
            )

        # qo_indptr = cumulative qlens.
        qo_indptr_list = [0]
        for q in qlens_list:
            qo_indptr_list.append(qo_indptr_list[-1] + q)

        # kv_indices / kv_indptr / kv_last_page_len from post-write seqlens.
        kv_indices_list: List[int] = []
        kv_indptr_list = [0]
        kv_last_page_len_list: List[int] = []
        for rid in active_rids:
            post_write_len = self._rollout_to_seqlen[rid]
            n_pages = (post_write_len + self.page_size - 1) // self.page_size
            pages = self._rollout_to_pages[rid]
            if len(pages) < n_pages:
                raise RuntimeError(
                    f"rollout {rid!r} has seqlen {post_write_len} but only "
                    f"{len(pages)} pages allocated"
                )
            kv_indices_list.extend(pages[:n_pages])
            kv_indptr_list.append(kv_indptr_list[-1] + n_pages)
            last = post_write_len - (n_pages - 1) * self.page_size if n_pages > 0 else 0
            kv_last_page_len_list.append(max(last, 0))

        # Per-new-token (batch_indices, positions) for append_paged_kv_cache.
        uniform = B > 0 and all(q == qlens_list[0] for q in qlens_list)
        if uniform and B > 0:
            S = qlens_list[0]
            pre = torch.tensor(seqlens_before, dtype=torch.int32, device=self.device)
            batch_indices = torch.arange(B, dtype=torch.int32, device=self.device).repeat_interleave(S)
            offsets = torch.arange(S, dtype=torch.int32, device=self.device).repeat(B)
            positions = pre.repeat_interleave(S) + offsets
        else:
            bi: List[int] = []
            pos: List[int] = []
            for b, (q, sb) in enumerate(zip(qlens_list, seqlens_before)):
                for j in range(q):
                    bi.append(b)
                    pos.append(sb + j)
            batch_indices = torch.tensor(bi, dtype=torch.int32, device=self.device)
            positions = torch.tensor(pos, dtype=torch.int32, device=self.device)

        # Write the real kv_indices into the pre-allocated
        # ``[total_pages]`` buffer; trailing entries (beyond
        # ``kv_indptr[-1]``) are ignored by both flashinfer's append
        # kernel and the planned wrapper. Similarly for kv_indptr
        # (bound to ``_kv_indptr_buf[:B+1]``) and kv_last_page_len
        # (bound to ``_kv_last_page_len_buf[:B]``). This keeps all three
        # tensors at stable device addresses across forward calls —
        # required for the cudagraph-friendly decode path and valuable
        # for Inductor shape-stability on the eager path.
        n_kv_indices = len(kv_indices_list)
        if n_kv_indices > self._kv_indices_buf.shape[0]:
            raise RuntimeError(
                f"kv_indices overflow: have {self._kv_indices_buf.shape[0]} "
                f"slots, need {n_kv_indices}"
            )
        if B > self.max_bsz:
            raise RuntimeError(
                f"batch size {B} > pool max_bsz {self.max_bsz}"
            )
        if n_kv_indices > 0:
            self._kv_indices_buf[:n_kv_indices].copy_(
                torch.tensor(kv_indices_list, dtype=torch.int32, device=self.device)
            )
        # indptr has B+1 entries; last_page_len has B. Slice the
        # pool-owned buffers and copy, then alias them in the returned
        # layout so flashinfer wrappers constructed with
        # ``use_cuda_graph=True`` see a plan call where the arg tensor
        # IS the bound internal buffer (flashinfer's self-copy inside
        # plan becomes a no-op).
        kv_indptr_view = self._kv_indptr_buf[: B + 1]
        kv_last_page_len_view = self._kv_last_page_len_buf[:B]
        qo_indptr_view = self._qo_indptr_buf[: B + 1]
        if B > 0:
            kv_indptr_view.copy_(
                torch.tensor(kv_indptr_list, dtype=torch.int32, device=self.device)
            )
            kv_last_page_len_view.copy_(
                torch.tensor(kv_last_page_len_list, dtype=torch.int32, device=self.device)
            )
        # qo_indptr written into the pool-bound buffer. Both prefill +
        # decode planning read it from this address; in the prefill
        # wrapper's use_cuda_graph mode plan() copies into its bound
        # buffer, so passing the same buffer back makes that copy a no-op.
        qo_indptr_view.copy_(
            torch.tensor(qo_indptr_list, dtype=torch.int32, device=self.device)
        )

        return PagedBatchLayout(
            qo_indptr=qo_indptr_view,
            kv_indptr=kv_indptr_view,
            kv_indices=self._kv_indices_buf,
            kv_last_page_len=kv_last_page_len_view,
            batch_indices=batch_indices,
            positions=positions,
            rollout_ids=list(active_rids),
            qlens=qlens_list,
            seqlens_before_write=seqlens_before,
        )

    # ---------------------- KV write -----------------------------------

    def append_kv(
        self,
        layer_id: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        layout: PagedBatchLayout,
    ) -> None:
        """Write ragged new K/V into this layer's paged cache.

        ``k_new`` / ``v_new`` shape: ``[total_new_tokens, n_kv_heads, head_dim]``
        where ``total_new_tokens == layout.qo_indptr[-1]``. Ordering: tokens
        packed across rollouts in ``layout.rollout_ids`` order, then within
        each rollout in logical-position order.

        Delegates to ``torchllms::paged_append_kv`` (wraps
        ``flashinfer.append_paged_kv_cache``). The custom-op boundary keeps
        the in-place mutation traceable for ``torch.compile``.
        """
        # Hot path — called once per layer, up to 36 times per decode
        # forward. Skip shape/bounds validation under torch.compile: the
        # ``layout.total_new_tokens`` property sums a Python list and
        # ``layout.qlens`` / ``layout.rollout_ids`` being Python lists
        # triggers Dynamo specialization. Eager callers still get the
        # validation.
        if not torch.compiler.is_compiling():
            if not (0 <= layer_id < self.n_layers):
                raise ValueError(
                    f"layer_id={layer_id} out of [0, {self.n_layers})"
                )
            expected = layout.total_new_tokens
            if k_new.shape[0] != expected or v_new.shape[0] != expected:
                raise ValueError(
                    f"k_new/v_new shape[0]={k_new.shape[0]}/{v_new.shape[0]} "
                    f"!= layout.total_new_tokens={expected}"
                )
            if k_new.shape[1:] != (self.n_kv_heads, self.head_dim):
                raise ValueError(
                    f"k_new shape[1:]={tuple(k_new.shape[1:])} expected "
                    f"({self.n_kv_heads}, {self.head_dim})"
                )

        from torchllms.models import compile_ops
        compile_ops.paged_append_kv(
            self.k_cache, self.v_cache, layer_id,
            k_new, v_new,
            layout.batch_indices, layout.positions,
            layout.kv_indices, layout.kv_indptr, layout.kv_last_page_len,
        )

    # ---------------------- KVChunk round-trip (Phase 1 radix compat) -

    @torch.no_grad()
    def load_chunk(
        self, chunk: KVChunk, rid: RolloutId, at_pos: int = 0,
    ) -> None:
        """Allocate pages for ``rid`` and scatter the chunk's K/V into them.

        Precondition: ``rid`` must have been claimed with zero seqlen and
        ``at_pos == 0`` (KVChunk invariant I4 — RoPE was applied at
        positions ``[0, chunk.length)`` by the writer). Post: the rollout
        has ``chunk.length`` tokens at logical positions ``[0, chunk.length)``.
        """
        if at_pos != 0:
            raise NotImplementedError(
                "load_chunk at non-zero at_pos requires RoPE rotation; "
                "not supported. See KVChunk invariant I4."
            )
        if rid not in self._rollout_to_pages:
            raise KeyError(f"unknown rollout {rid!r}")
        if self._rollout_to_seqlen[rid] != 0:
            raise RuntimeError(
                f"load_chunk into non-empty rollout {rid!r}: "
                f"seqlen={self._rollout_to_seqlen[rid]}"
            )
        if chunk.k.shape[0] != self.n_layers:
            raise ValueError(
                f"chunk n_layers={chunk.k.shape[0]} != pool {self.n_layers}"
            )
        if chunk.k.shape[2] != self.n_kv_heads or chunk.k.shape[3] != self.head_dim:
            raise ValueError(
                f"chunk head shape {tuple(chunk.k.shape[2:])} != pool "
                f"({self.n_kv_heads}, {self.head_dim})"
            )

        L = chunk.length
        if L == 0:
            return

        # Allocate pages to cover L tokens.
        self.extend(rid, L)
        pages = self._rollout_to_pages[rid]

        # H2D copy — move the full chunk to device once, then scatter by
        # page. Keeping a single contiguous transfer lets cudaMemcpyAsync
        # overlap with later setup; per-page copies would hurt both
        # bandwidth and latency.
        k_dev = chunk.k.to(device=self.device, dtype=self.dtype, non_blocking=True)
        v_dev = chunk.v.to(device=self.device, dtype=self.dtype, non_blocking=True)

        n_full_pages = L // self.page_size
        remainder = L - n_full_pages * self.page_size

        # Full-page writes.
        if n_full_pages > 0:
            full_k = k_dev[:, : n_full_pages * self.page_size].reshape(
                self.n_layers, n_full_pages, self.page_size,
                self.n_kv_heads, self.head_dim,
            )
            full_v = v_dev[:, : n_full_pages * self.page_size].reshape(
                self.n_layers, n_full_pages, self.page_size,
                self.n_kv_heads, self.head_dim,
            )
            page_ids = torch.tensor(pages[:n_full_pages], device=self.device)
            self.k_cache[:, page_ids] = full_k
            self.v_cache[:, page_ids] = full_v

        # Partial tail page.
        if remainder > 0:
            tail_page = pages[n_full_pages]
            tail_k = k_dev[:, n_full_pages * self.page_size : L]
            tail_v = v_dev[:, n_full_pages * self.page_size : L]
            self.k_cache[:, tail_page, :remainder] = tail_k
            self.v_cache[:, tail_page, :remainder] = tail_v

    @torch.no_grad()
    def extract_chunk(
        self, rid: RolloutId, length: Optional[int] = None,
    ) -> KVChunk:
        """Return a ``KVChunk`` (CPU) covering the first ``length`` tokens
        of ``rid``'s KV. ``role_ids`` is zeroed since the paged pool no
        longer stores per-token roles (see module docstring).
        """
        if rid not in self._rollout_to_pages:
            raise KeyError(f"unknown rollout {rid!r}")
        max_len = self._rollout_to_seqlen[rid]
        if length is None:
            length = max_len
        elif not (0 <= length <= max_len):
            raise ValueError(f"length={length} out of [0, {max_len}]")
        return self._extract(rid, length)

    def _extract(self, rid: RolloutId, length: int) -> KVChunk:
        if length == 0:
            k = torch.empty(
                (self.n_layers, 0, self.n_kv_heads, self.head_dim),
                dtype=self.dtype,
            )
            return KVChunk(
                k=k.clone(),
                v=k.clone(),
                role_ids=torch.zeros((0,), dtype=torch.long),
            )
        pages = self._rollout_to_pages[rid]
        n_pages = (length + self.page_size - 1) // self.page_size

        # Gather the needed pages to contiguous CPU tensors, then trim to
        # ``length``. One D2H transfer per call.
        page_ids = torch.tensor(pages[:n_pages], device=self.device)
        k_pages = self.k_cache[:, page_ids].detach().to("cpu")
        v_pages = self.v_cache[:, page_ids].detach().to("cpu")
        k_flat = k_pages.reshape(
            self.n_layers, n_pages * self.page_size,
            self.n_kv_heads, self.head_dim,
        )[:, :length].contiguous()
        v_flat = v_pages.reshape(
            self.n_layers, n_pages * self.page_size,
            self.n_kv_heads, self.head_dim,
        )[:, :length].contiguous()
        return KVChunk(
            k=k_flat,
            v=v_flat,
            role_ids=torch.zeros((length,), dtype=torch.long),
        )

    # ---------------------- Invariants (tests) -------------------------

    def _check_invariants(self) -> None:
        # S3: per-rollout page count matches seqlen.
        rollout_usage: Dict[int, int] = {}
        for rid, pages in self._rollout_to_pages.items():
            L = self._rollout_to_seqlen[rid]
            expected = (L + self.page_size - 1) // self.page_size
            assert len(pages) == expected, (
                f"S3 violation: rid {rid!r} seqlen={L} has {len(pages)} "
                f"pages (expected {expected})"
            )
            for p in pages:
                assert 0 <= p < self.total_pages, f"bad page {p}"
                rollout_usage[p] = rollout_usage.get(p, 0) + 1

        # Refcount consistency: every page in ``_free_pages`` has ref 0;
        # every other page has ref >= rollout_usage (more if radix or other
        # callers borrowed it).
        free_set = set(self._free_pages)
        for p in range(self.total_pages):
            ref = self._page_refcount[p]
            if p in free_set:
                assert ref == 0, (
                    f"free-list invariant: page {p} in free list but ref={ref}"
                )
            else:
                assert ref > 0, (
                    f"free-list invariant: page {p} not in free list but ref={ref}"
                )
            assert ref >= rollout_usage.get(p, 0), (
                f"refcount underflow: page {p} ref={ref} < "
                f"rollout uses {rollout_usage.get(p, 0)}"
            )

        # S1/S2: every page is accounted for exactly once between free
        # and not-free.
        total_pages_set = free_set | {p for p in range(self.total_pages) if self._page_refcount[p] > 0}
        assert total_pages_set == set(range(self.total_pages)), (
            f"S1/S2 violation: free={len(free_set)} held="
            f"{sum(1 for r in self._page_refcount if r > 0)} "
            f"total={self.total_pages}"
        )

        # S4: rollout IDs are strictly below the next counter.
        for rid in self._rollout_to_pages:
            assert rid < self._next_rollout_id, (
                f"S4 violation: rid {rid} >= next_rollout_id {self._next_rollout_id}"
            )
