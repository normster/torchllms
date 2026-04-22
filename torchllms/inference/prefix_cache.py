"""GPU-resident radix prefix cache for multi-turn rollouts (Phase 2).

A compressed radix trie over token sequences whose leaves reference
``PagedKVPool`` page IDs — there are no CPU-resident KV tensors. A
prefix-cache hit produces the list of page IDs the caller should
borrow into a new rollout's block table; insertion hands page
ownership from the retiring rollout to the trie.

Key invariants
--------------

I1. **Page alignment.** Every node's ``edge_tokens`` has length equal
    to ``len(page_ids) * page_size``. Tokens and pages are in
    lock-step; the trie never stores a partial page. Callers must
    truncate to page-alignment before :meth:`insert`.

I2. **Radix holds a refcount per adopted page.** :meth:`insert` calls
    ``pool.borrow_pages`` on each page the trie newly adopts.
    :meth:`evict_oldest` (and :meth:`clear`) call ``pool.release_pages``
    on each page the trie drops. Pages a retiring rollout *already*
    inserted on a prior retire stay in the trie without additional
    ref bumps.

I3. **Rollout-borrows are orthogonal to radix ownership.** A page
    can be simultaneously held by the trie (one ref) and multiple
    live rollouts that matched it on lookup (one ref each). Eviction
    only drops the trie's ref; rollout-held pages survive.

I4. **Partial tail-page tokens are not cached.** If a retiring rollout
    has ``L`` valid tokens with ``L % page_size != 0``, only the first
    ``L - (L % page_size)`` tokens are inserted. The partial page has
    no in-trie representation because storing it would require
    copy-on-write to avoid corrupting shared pages.

LRU eviction drops leaves in last-access order. Parent/child
compaction (merging a chain node into its single child) runs after
every :meth:`_remove_leaf`, identically to the pre-Phase-2 radix.

There is no internal max-bytes budget. The pool's page count is the
budget; the LLM driver is responsible for triggering eviction via
:meth:`evict_oldest` when ``pool._alloc_page`` fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

from torchllms.models.paged_kv import PagedKVPool


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class _Node:
    """One node in the radix trie.

    ``edge_tokens`` and ``page_ids`` describe the inbound edge from
    this node's parent. Both are empty for the root. ``children`` is
    keyed on the first token of each outgoing edge.
    """

    edge_tokens: Tuple[int, ...]
    page_ids: Tuple[int, ...]
    children: dict = field(default_factory=dict)
    parent: Optional["_Node"] = None
    last_access: int = 0


@dataclass(frozen=True)
class RadixMatch:
    """Result of :meth:`RadixKVCache.lookup`.

    ``length`` is the number of tokens matched (always a multiple of
    ``pool.page_size``). ``page_ids`` is the ordered list of page IDs
    spanning those tokens — the caller attaches them to a fresh
    rollout's block table and calls ``pool.borrow_pages(page_ids)``.
    """

    length: int
    page_ids: Tuple[int, ...]

    @property
    def hit(self) -> bool:
        return self.length > 0


class RadixKVCache:
    """Token-prefix radix trie over page IDs.

    Shares pages with a :class:`PagedKVPool`; adoption and eviction go
    through the pool's refcount.

    Usage (inside a generate loop)::

        match = radix.lookup(prompt_tokens)
        if match.hit:
            pool.borrow_pages(match.page_ids)
            pool.attach_borrowed_pages(rid, match.page_ids)
        # ... run prefill on prompt_tokens[match.length:] ...
        # ... decode, accumulate generated tokens ...
        pages, seqlen = pool.retire_pages(rid)
        n_full = seqlen // pool.page_size
        if n_full > 0:
            radix.insert(
                all_tokens[: n_full * pool.page_size],
                pages[:n_full],
            )
        pool.release_pages(pages)
    """

    def __init__(self, pool: PagedKVPool):
        self._pool = pool
        self._page_size = pool.page_size
        self._root = _Node(edge_tokens=(), page_ids=())
        self._clock = 0

    @property
    def page_size(self) -> int:
        return self._page_size

    # ---- Introspection ------------------------------------------------

    def num_pages(self) -> int:
        return sum(len(n.page_ids) for n in self._walk_nodes() if n is not self._root)

    def num_nodes(self) -> int:
        return sum(1 for _ in self._walk_nodes())

    def _walk_nodes(self) -> Iterator[_Node]:
        stack: List[_Node] = [self._root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    # ---- Lookup -------------------------------------------------------

    def lookup(self, tokens: Sequence[int]) -> RadixMatch:
        """Return the longest page-aligned prefix match.

        Partial-edge matches are truncated to the nearest page boundary
        so borrowed pages are never written to mid-page — avoiding
        copy-on-write corruption of shared pages.
        """
        if not tokens:
            return RadixMatch(length=0, page_ids=())
        path_pages: List[int] = []
        consumed = 0
        node = self._root
        remaining: Tuple[int, ...] = tuple(tokens)
        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                break
            shared = _common_prefix_len(child.edge_tokens, remaining)
            # Align to page boundary — pages are only exposed to a
            # borrower at page-full granularity.
            shared_aligned = shared - (shared % self._page_size)
            if shared_aligned == 0:
                break
            n_pages = shared_aligned // self._page_size
            path_pages.extend(child.page_ids[:n_pages])
            self._touch(child)
            if shared_aligned < len(child.edge_tokens):
                consumed += shared_aligned
                break
            consumed += shared_aligned
            node = child
            remaining = remaining[shared_aligned:]
        return RadixMatch(length=consumed, page_ids=tuple(path_pages))

    # ---- Insert -------------------------------------------------------

    def insert(self, tokens: Sequence[int], page_ids: Sequence[int]) -> None:
        """Insert a page-aligned prefix into the trie.

        Preconditions:
        - ``len(tokens) == len(page_ids) * page_size`` (page alignment).
        - The caller still holds a refcount on every page in
          ``page_ids`` (they're in a retiring rollout's block table).
          The trie calls ``pool.borrow_pages`` on pages it newly adopts,
          bumping their ref so the caller's subsequent ``release_pages``
          doesn't send them back to the free list.
        - Pages that are already in the trie (from an earlier insert)
          are detected and left unchanged. This happens naturally when
          a rollout is prefix-matched (borrowed pages) then extended
          with fresh pages — on insert, only the fresh ones get adopted.
        """
        tokens = tuple(tokens)
        page_ids = tuple(page_ids)
        if len(tokens) != len(page_ids) * self._page_size:
            raise ValueError(
                f"insert: len(tokens)={len(tokens)} must equal "
                f"len(page_ids)={len(page_ids)} * page_size={self._page_size}"
            )
        if len(tokens) == 0:
            return
        self._insert_walk(self._root, tokens, page_ids, consumed=0)

    def _insert_walk(
        self,
        node: _Node,
        tokens: Tuple[int, ...],
        page_ids: Tuple[int, ...],
        consumed: int,
    ) -> None:
        while consumed < len(tokens):
            remaining = tokens[consumed:]
            remaining_pages = page_ids[consumed // self._page_size :]
            first = remaining[0]
            child = node.children.get(first)
            if child is None:
                # Attach a fresh edge from ``node``.
                self._attach_new_edge(node, remaining, remaining_pages)
                return
            shared = _common_prefix_len(child.edge_tokens, remaining)
            shared_aligned = shared - (shared % self._page_size)
            if shared_aligned == 0:
                # Sub-page divergence: same first token but the two
                # sequences diverge before the next page boundary.
                # Neither sequence can fully own the shared page
                # (would need copy-on-write). We keep the existing
                # branch and silently drop the new insert's suffix —
                # the caller's rollout still owns its pages, and on
                # ``release_pages`` the non-adopted fresh pages go
                # back to the free list. Common trigger: greedy
                # decode under bf16 with a nearby argmax flip causes
                # two same-prompt runs to produce slightly different
                # completions partway through.
                return
            if shared_aligned == len(child.edge_tokens):
                # Match this whole child; descend.
                consumed += shared_aligned
                node = child
                self._touch(child)
                continue
            # Split child's edge at ``shared_aligned``; then attach the
            # remainder as a new child of the split point.
            self._split_edge(child, shared_aligned)
            consumed += shared_aligned
            node = child
            self._touch(child)
            # Loop again to attach the remainder.

    def _attach_new_edge(
        self, node: _Node, new_tokens: Tuple[int, ...],
        new_page_ids: Tuple[int, ...],
    ) -> None:
        """Attach a new child edge under ``node``. Radix adopts each of
        ``new_page_ids`` (pool.borrow_pages). If ``node`` is a childless
        non-root leaf, extend its edge in place instead of chaining a
        second node (matches pre-Phase-2 compression behavior)."""
        if node is not self._root and not node.children:
            # Leaf extension.
            node.edge_tokens = node.edge_tokens + new_tokens
            node.page_ids = node.page_ids + new_page_ids
            self._pool.borrow_pages(new_page_ids)
            self._touch(node)
            return
        new_node = _Node(
            edge_tokens=new_tokens,
            page_ids=new_page_ids,
            parent=node,
        )
        node.children[new_tokens[0]] = new_node
        self._pool.borrow_pages(new_page_ids)
        self._touch(new_node)

    def _split_edge(self, node: _Node, at: int) -> None:
        """Split ``node``'s inbound edge at offset ``at`` (page-aligned).

        Post-split, ``node`` covers ``[0:at]`` of its original edge (and
        the head pages); a new child ``tail`` covers ``[at:]`` and
        inherits ``node``'s previous children and pages. No page-ref
        changes — the pages don't move, just get redistributed between
        two trie nodes.
        """
        if at % self._page_size != 0:
            raise RuntimeError(f"split offset {at} not page-aligned")
        if not (0 < at < len(node.edge_tokens)):
            raise ValueError(
                f"invalid split offset {at} for edge of length "
                f"{len(node.edge_tokens)}"
            )
        pages_at = at // self._page_size
        tail_node = _Node(
            edge_tokens=node.edge_tokens[at:],
            page_ids=node.page_ids[pages_at:],
            children=node.children,
            parent=node,
            last_access=node.last_access,
        )
        for grand in tail_node.children.values():
            grand.parent = tail_node
        node.edge_tokens = node.edge_tokens[:at]
        node.page_ids = node.page_ids[:pages_at]
        node.children = {tail_node.edge_tokens[0]: tail_node}

    # ---- Eviction -----------------------------------------------------

    def evict_oldest(self) -> int:
        """Evict the LRU leaf. Releases radix's refcount on each of its
        pages (they may still have rollout-held refs; release just
        decrements). Returns the number of pages released. 0 if the
        trie is empty.
        """
        leaf = self._find_oldest_leaf()
        if leaf is None:
            return 0
        return self._remove_leaf(leaf)

    def _find_oldest_leaf(self) -> Optional[_Node]:
        oldest: Optional[_Node] = None
        for node in self._walk_nodes():
            if node is self._root or node.children:
                continue
            if oldest is None or node.last_access < oldest.last_access:
                oldest = node
        return oldest

    def _remove_leaf(self, leaf: _Node) -> int:
        if leaf.children:
            raise RuntimeError("cannot remove non-leaf via _remove_leaf")
        parent = leaf.parent
        assert parent is not None
        del parent.children[leaf.edge_tokens[0]]
        self._pool.release_pages(leaf.page_ids)
        n_released = len(leaf.page_ids)
        # Parent/child compaction.
        if parent is not self._root and len(parent.children) == 1:
            only_child = next(iter(parent.children.values()))
            parent.edge_tokens = parent.edge_tokens + only_child.edge_tokens
            parent.page_ids = parent.page_ids + only_child.page_ids
            parent.children = only_child.children
            for grand in parent.children.values():
                grand.parent = parent
            parent.last_access = max(parent.last_access, only_child.last_access)
        return n_released

    def clear(self) -> None:
        """Release every page held by the trie and reset."""
        all_pages: List[int] = []
        for n in self._walk_nodes():
            if n is self._root:
                continue
            all_pages.extend(n.page_ids)
        if all_pages:
            self._pool.release_pages(all_pages)
        self._root = _Node(edge_tokens=(), page_ids=())
        self._clock = 0

    # ---- Internals ----------------------------------------------------

    def _touch(self, node: _Node) -> None:
        self._clock += 1
        node.last_access = self._clock
