"""Across-call token-prefix KV cache for multi-turn rollouts.

A compressed radix trie over token sequences. Each non-root node stores a
KVChunk covering just the tokens on its inbound edge (delta storage), so a
shared system prompt across many rollouts is stored exactly once.

Materializing a matched prefix concatenates all chunks along the path from
root to the matched node. Partial-edge matches (query ends mid-edge) slice
the final chunk at read time without mutating the tree.

LRU eviction drops leaves in last-access order until total size fits the
max_bytes budget. When a leaf's removal leaves its parent with a single
remaining child, the parent and child are compacted into one node
(concatenating edge tokens and chunks) so the tree always represents the
minimum set of branching points.

Caveats for agentic rollout use:
  - Inserts are expected at turn boundaries, i.e., the full `tokens` and
    KVChunk of the prefix computed so far (prompt + generated response).
  - The cache is not touched during a single generate() call. Lookup happens
    at turn start, insert at turn end; decode interacts only with the
    KVArena in between.
  - CPU residency. KVArena handles host<->device transfer via load_chunk
    and extract_chunk / retire.

Correctness caveat — bf16 matmul shape dependence:
  A generate() call that hits a prefix (prefill qlen=1 against loaded
  K of length N-1) produces numerically slightly different logits than
  the same call without a cache (prefill qlen=N). The chunk round-trip
  is bit-exact; SDPA is bit-exact. The root cause is cuBLAS GEMM in
  bf16: nn.Linear(x[B, S1, D]) vs nn.Linear(x[B, S2, D]) give different
  values at shared positions when S1 != S2, because cuBLAS selects
  different kernels/tilings per M dimension and bf16 accumulation order
  differs across kernels. Direct test on a 2560x2560 bf16 Linear:
  max output diff ~1.0 at the first 32 rows between calls of shape
  [1,33,2560] vs [1,32,2560] on identical data.
  Compounded over 36 transformer layers, this reaches max logit diffs
  of ~0.3-10 on Qwen3-4B at the last prefill position — enough to flip
  argmax on close candidates and cascade into different greedy
  trajectories. This is NOT unique to the radix path: calling
  model.forward twice (prompt[:K] then prompt[K:]) diverges from a
  single forward on the full prompt by the same order of magnitude.
  Also not fixable without giving up batched matmul or moving attention
  layers to fp32.
  Phase 5 intervention comparisons should hold the regime fixed (both
  baseline and intervention use the same cache state) so the drift
  cancels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

from torchllms.models.cache import KVChunk


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class _Node:
    """One node in the radix trie.

    `edge_tokens` and `block` describe the inbound edge from this node's
    parent. Both are None for the root. `children` is keyed on the first
    token of each outgoing edge.
    """

    edge_tokens: Tuple[int, ...]
    block: Optional[KVChunk]
    children: dict = field(default_factory=dict)
    parent: Optional["_Node"] = None
    last_access: int = 0


@dataclass(frozen=True)
class RadixMatch:
    """Result of RadixKVCache.lookup(tokens).

    `length` is the number of query tokens matched. `_path` is the chain of
    non-root nodes walked, and `_last_edge_consumed` is how many tokens of
    the final node's edge were consumed (equal to that edge's full length
    on node-boundary matches, strictly less on partial-edge matches).
    """

    length: int
    _path: Tuple[_Node, ...]
    _last_edge_consumed: int

    @property
    def hit(self) -> bool:
        return self.length > 0

    def materialize(self) -> KVChunk:
        """Return the KVChunk covering the matched prefix.

        Walks from root to the final path node, concatenating each node's
        delta block. If the match ended mid-edge, the final block is sliced
        to `_last_edge_consumed` tokens.
        """
        if not self._path:
            raise ValueError("RadixMatch has no match to materialize")
        blocks: List[KVChunk] = [n.block for n in self._path[:-1]]
        last = self._path[-1]
        if self._last_edge_consumed < len(last.edge_tokens):
            blocks.append(last.block.slice(0, self._last_edge_consumed))
        else:
            blocks.append(last.block)
        if len(blocks) == 1:
            return blocks[0]
        return KVChunk.concat(blocks)


class RadixKVCache:
    """Compressed radix trie indexing delta KVChunks by token prefix.

    See module docstring for design notes.

    Usage:
        cache = RadixKVCache(max_bytes=16 * 1024**3)
        rid = arena.claim()
        match = cache.lookup(tokens)
        if match.hit:
            chunk = match.materialize()
            arena.load_chunk(chunk, rid)
            prefill_tokens = tokens[match.length:]
        # ... run generation, collect full_tokens = tokens + generated ...
        full_chunk = arena.retire(rid)
        cache.insert(full_tokens, full_chunk)
    """

    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._root = _Node(edge_tokens=(), block=None)
        self._total_bytes = 0
        self._clock = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def _touch(self, node: _Node) -> None:
        self._clock += 1
        node.last_access = self._clock

    def lookup(self, tokens: Sequence[int]) -> RadixMatch:
        """Longest-matching prefix of `tokens` in the trie.

        Partial-edge matches are reported without mutating the tree.
        """
        if not tokens:
            return RadixMatch(length=0, _path=(), _last_edge_consumed=0)

        path: List[_Node] = []
        consumed = 0
        node = self._root
        remaining = tuple(tokens)
        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                break
            shared = _common_prefix_len(child.edge_tokens, remaining)
            path.append(child)
            self._touch(child)
            consumed += shared
            if shared < len(child.edge_tokens):
                return RadixMatch(
                    length=consumed,
                    _path=tuple(path),
                    _last_edge_consumed=shared,
                )
            node = child
            remaining = remaining[shared:]

        if not path:
            return RadixMatch(length=0, _path=(), _last_edge_consumed=0)

        last = path[-1]
        return RadixMatch(
            length=consumed,
            _path=tuple(path),
            _last_edge_consumed=len(last.edge_tokens),
        )

    def insert(self, tokens: Sequence[int], block: KVChunk) -> None:
        """Insert a full-prefix KVChunk for the given tokens.

        The block must cover exactly `tokens` (block.length == len(tokens)).
        Internally, the radix trie is extended via one of:
          - new edge from an existing branching point / root
          - extension of a leaf's edge (compression: no spurious insertion-
            marker nodes)
          - split of an existing edge at the divergence point

        Triggers LRU eviction at the end if total_bytes exceeds max_bytes.
        """
        if len(tokens) != block.length:
            raise ValueError(
                f"tokens length ({len(tokens)}) != block length ({block.length})"
            )
        if len(tokens) == 0:
            return

        token_tuple = tuple(tokens)
        self._insert_walk(self._root, token_tuple, block, matched=0)
        self._evict_if_needed()

    def _insert_walk(
        self,
        node: _Node,
        tokens: Tuple[int, ...],
        block: KVChunk,
        matched: int,
    ) -> None:
        while matched < len(tokens):
            first = tokens[matched]
            child = node.children.get(first)
            if child is None:
                self._attach_or_extend(node, tokens[matched:], block, matched)
                return
            shared = _common_prefix_len(child.edge_tokens, tokens[matched:])
            if shared == len(child.edge_tokens):
                matched += shared
                node = child
                self._touch(child)
                if matched == len(tokens):
                    # Exact node-boundary match; nothing new to store.
                    return
                continue
            # Diverge mid-edge: split at `shared`.
            self._split_edge(child, shared)
            matched += shared
            node = child
            self._touch(child)
            if matched == len(tokens):
                # Insert ends exactly at the split point; split already stored
                # the correct prefix block on `child`.
                return
            # Attach remainder as a new sibling under `child`. Since child now
            # has exactly one child (the tail), this goes through the
            # attach-new-edge branch next iteration.
        # matched == len(tokens): fully inside an existing path; no-op

    def _attach_or_extend(
        self,
        node: _Node,
        new_tokens: Tuple[int, ...],
        block: KVChunk,
        matched: int,
    ) -> None:
        """Attach a new child edge, or if `node` is a childless non-root leaf,
        extend its existing edge and block instead of creating a chain."""
        delta = block.slice(matched, matched + len(new_tokens))
        if node is not self._root and not node.children:
            # Leaf extension: concat tokens and blocks in place.
            old_block = node.block
            new_block = KVChunk.concat([old_block, delta])
            self._total_bytes -= old_block.size_bytes
            self._total_bytes += new_block.size_bytes
            node.edge_tokens = node.edge_tokens + new_tokens
            node.block = new_block
            self._touch(node)
            return
        # Attach a new leaf edge.
        new_node = _Node(
            edge_tokens=new_tokens,
            block=delta,
            parent=node,
        )
        node.children[new_tokens[0]] = new_node
        self._total_bytes += delta.size_bytes
        self._touch(new_node)

    def _split_edge(self, node: _Node, at: int) -> None:
        """Split `node`'s inbound edge at offset `at` (0 < at < edge length).

        After the split, `node` covers tokens[0:at] of its original edge and
        has a single child holding the remaining tokens plus the original
        children.
        """
        original_tokens = node.edge_tokens
        original_block = node.block
        if not (0 < at < len(original_tokens)):
            raise ValueError(
                f"invalid split offset {at} for edge of length {len(original_tokens)}"
            )
        assert original_block is not None

        head_block = original_block.slice(0, at)
        tail_block = original_block.slice(at, len(original_tokens))

        tail_node = _Node(
            edge_tokens=original_tokens[at:],
            block=tail_block,
            children=node.children,
            parent=node,
            last_access=node.last_access,
        )
        for grand in tail_node.children.values():
            grand.parent = tail_node

        node.edge_tokens = original_tokens[:at]
        node.block = head_block
        node.children = {tail_node.edge_tokens[0]: tail_node}

        self._total_bytes -= original_block.size_bytes
        self._total_bytes += head_block.size_bytes + tail_block.size_bytes

    def _evict_if_needed(self) -> None:
        while self._total_bytes > self.max_bytes:
            leaf = self._find_oldest_leaf()
            if leaf is None:
                break
            self._remove_leaf(leaf)

    def _find_oldest_leaf(self) -> Optional[_Node]:
        oldest: Optional[_Node] = None
        for node in self._walk_nodes():
            if node is self._root:
                continue
            if node.children:
                continue
            if oldest is None or node.last_access < oldest.last_access:
                oldest = node
        return oldest

    def _walk_nodes(self) -> Iterator[_Node]:
        stack: List[_Node] = [self._root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    def _remove_leaf(self, leaf: _Node) -> None:
        if leaf.children:
            raise RuntimeError("cannot remove non-leaf node via _remove_leaf")
        parent = leaf.parent
        assert parent is not None
        assert leaf.block is not None
        del parent.children[leaf.edge_tokens[0]]
        self._total_bytes -= leaf.block.size_bytes
        # Compact: if parent is a non-root node with exactly one remaining
        # child, merge parent's inbound edge with that child's inbound edge
        # and collapse the extra node.
        if parent is not self._root and len(parent.children) == 1:
            only_child = next(iter(parent.children.values()))
            merged_tokens = parent.edge_tokens + only_child.edge_tokens
            merged_block = KVChunk.concat([parent.block, only_child.block])
            self._total_bytes -= parent.block.size_bytes + only_child.block.size_bytes
            self._total_bytes += merged_block.size_bytes
            parent.edge_tokens = merged_tokens
            parent.block = merged_block
            parent.children = only_child.children
            for grand in parent.children.values():
                grand.parent = parent
            # last_access: keep the more recently touched of the two so the
            # merged node isn't prematurely eligible for eviction.
            parent.last_access = max(parent.last_access, only_child.last_access)

    def clear(self) -> None:
        self._root = _Node(edge_tokens=(), block=None)
        self._total_bytes = 0
        self._clock = 0

    # ---- Introspection helpers (for tests / debugging) ----

    def num_nodes(self) -> int:
        return sum(1 for _ in self._walk_nodes())
