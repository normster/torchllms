"""KV-cache primitives for torchllms inference.

Three types live in this module:

- :class:`RolloutId`: opaque handle for one in-flight rollout. Stable for
  its lifetime. Issued by :meth:`KVArena.claim`, invalidated by
  :meth:`KVArena.retire`.

- :class:`KVChunk`: CPU-resident, immutable slab of (K, V, role_ids) for
  one contiguous run of tokens at positions [0, length). The only
  transport type crossing GPU<->CPU and arena<->radix boundaries.

- :class:`KVArena`: in-flight, GPU-resident, compacting KV store for one
  generate() call. External address = ``RolloutId`` (stable). Internal
  address = slot (changes on retire via swap-with-last compaction).

Contract with :class:`RadixKVCache`:

    radix.lookup(tokens) -> KVChunk -> arena.load_chunk(chunk, rid)
    arena.retire(rid)    -> KVChunk -> radix.insert(tokens, chunk)

The arena never touches the radix trie and the radix never holds device
tensors. ``KVChunk`` is the only shared vocabulary.

Forward-call protocol::

    cache.update_role_ids(role_ids)   # once, before any update_kv
    cache.update_attn_mask(attn_mask) # once, before any update_kv
    for layer in model.layers:
        layer.attention.update_kv(cache, ...)  # advances seen_tokens[l]

Position offsets for RoPE are read via :attr:`KVArena.row_positions`
before the first ``update_kv`` of a forward. Inside a forward, different
layers may transiently have different ``seen_tokens`` values; between
forwards they agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NewType, Optional, Sequence, Tuple

import torch


RolloutId = NewType("RolloutId", int)


# ===================================================================== #
# KVChunk                                                                #
# ===================================================================== #


@dataclass(frozen=True)
class KVChunk:
    """Immutable CPU slab of KV state for token positions [0, length).

    Invariants (enforced at construction):
        I1. ``k.shape == v.shape == (n_layers, length, n_kv_heads, head_dim)``.
        I2. ``role_ids.shape == (length,)``.
        I3. ``k``, ``v``, ``role_ids`` all reside on CPU.
        I4. RoPE has been applied at positions [0, length). A chunk cannot
            be loaded at a non-zero offset without rotating positions;
            callers must use ``at_pos=0``.
    """

    k: torch.Tensor
    v: torch.Tensor
    role_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.k.ndim != 4 or self.v.ndim != 4:
            raise ValueError(
                "KVChunk k/v must be 4-D (n_layers, length, n_kv, head_dim); "
                f"got k.shape={tuple(self.k.shape)} v.shape={tuple(self.v.shape)}"
            )
        if self.k.shape != self.v.shape:
            raise ValueError(
                f"KVChunk k/v shape mismatch: k={tuple(self.k.shape)} "
                f"v={tuple(self.v.shape)}"
            )
        if self.role_ids.ndim != 1 or self.role_ids.shape[0] != self.k.shape[1]:
            raise ValueError(
                "KVChunk role_ids must be 1-D of length == k.shape[1]; "
                f"got role_ids.shape={tuple(self.role_ids.shape)} "
                f"k.shape[1]={self.k.shape[1]}"
            )
        for name, t in (("k", self.k), ("v", self.v), ("role_ids", self.role_ids)):
            if t.device.type != "cpu":
                raise ValueError(
                    f"KVChunk.{name} must be on CPU; got device={t.device}"
                )

    @property
    def length(self) -> int:
        return int(self.k.shape[1])

    @property
    def size_bytes(self) -> int:
        return (
            self.k.element_size() * self.k.numel()
            + self.v.element_size() * self.v.numel()
            + self.role_ids.element_size() * self.role_ids.numel()
        )

    def slice(self, start: int, end: int) -> "KVChunk":
        if not (0 <= start <= end <= self.length):
            raise ValueError(
                f"slice out of range: start={start} end={end} length={self.length}"
            )
        return KVChunk(
            k=self.k[:, start:end].clone(),
            v=self.v[:, start:end].clone(),
            role_ids=self.role_ids[start:end].clone(),
        )

    @staticmethod
    def concat(chunks: Sequence["KVChunk"]) -> "KVChunk":
        if not chunks:
            raise ValueError("concat requires at least one chunk")
        return KVChunk(
            k=torch.cat([c.k for c in chunks], dim=1),
            v=torch.cat([c.v for c in chunks], dim=1),
            role_ids=torch.cat([c.role_ids for c in chunks], dim=0),
        )


# ===================================================================== #
# KVArena                                                                #
# ===================================================================== #


class KVArena:
    """In-flight, compacting KV store for one ``generate()`` call.

    Structural invariants (hold after every public method returns):
        S1. ``len(slot_to_rollout) == len(rollout_to_slot) == b_live``.
        S2. For every live slot ``i`` in ``[0, b_live)``::

                rollout_to_slot[slot_to_rollout[i]] == i

        S3. For every ``rid`` in ``rollout_to_slot``::

                slot_to_rollout[rollout_to_slot[rid]] == rid

        S4. ``0 <= seen_tokens[:, i] <= max_seqlen`` for ``i`` in
            ``[0, b_live)``.
        S5. Slots ``[b_live, max_bsz)`` are reserve capacity; their
            contents are undefined and must not be read.
        S6. ``RolloutId`` values are strictly monotonic per arena and
            never reused.
        S7. Between forwards, ``seen_tokens[l, i]`` is equal across all
            layers ``l`` for any given live ``i``. Inside a forward this
            may be transiently violated.

    Forward-call invariants:
        F1. ``claim`` / ``retire`` / ``retire_many`` occur only between
            forwards (never interleaved with ``update_kv``).
        F2. ``update_kv(layer_id, ...)`` writes K/V for the current
            forward's ``S`` new tokens at each live row's current
            ``seen_tokens[layer_id, row]`` position, then advances
            ``seen_tokens[layer_id, :b_live]`` by ``S``.
        F3. ``update_role_ids`` / ``update_attn_mask`` are called before
            any ``update_kv`` in a given forward.
        F4. ``update_kv`` is the only advancement point for
            ``seen_tokens``. No external caller advances it directly.
    """

    def __init__(
        self,
        n_layers: int,
        max_bsz: int,
        max_seqlen: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if n_layers <= 0 or max_bsz <= 0 or max_seqlen <= 0:
            raise ValueError(
                f"invalid dims: n_layers={n_layers} max_bsz={max_bsz} "
                f"max_seqlen={max_seqlen}"
            )
        dev = torch.device(device) if not isinstance(device, torch.device) else device

        kv_shape = (n_layers, max_bsz, max_seqlen, n_kv_heads, head_dim)
        self.k_cache = torch.zeros(kv_shape, device=dev, dtype=dtype)
        self.v_cache = torch.zeros(kv_shape, device=dev, dtype=dtype)
        self.role_id_cache = torch.zeros(
            (max_bsz, max_seqlen), device=dev, dtype=torch.long
        )
        self.attn_mask_cache = torch.zeros(
            (max_bsz, max_seqlen), device=dev, dtype=torch.long
        )
        self.is_attn_mask_cached = False

        self.seen_tokens = torch.zeros(
            (n_layers, max_bsz), device=dev, dtype=torch.long
        )

        self.slot_to_rollout: List[RolloutId] = []
        self.rollout_to_slot: Dict[RolloutId, int] = {}
        self._next_rollout_id: int = 0

        self.n_layers = n_layers
        self.max_bsz = max_bsz
        self.max_seqlen = max_seqlen
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = dev
        self.dtype = dtype

    # ---------------------------- Identity ----------------------------- #

    @property
    def b_live(self) -> int:
        return len(self.slot_to_rollout)

    @property
    def row_positions(self) -> torch.Tensor:
        """[b_live] int64. Absolute position where each live row's next
        token will be written. Derived from ``seen_tokens[0, :b_live]``;
        by S7 this equals every other layer's counter between forwards."""
        return self.seen_tokens[0, : self.b_live]

    @property
    def uniform_seen_tokens(self) -> bool:
        """True iff every live row has the same ``seen_tokens`` value (on
        layer 0 — S7 guarantees the other layers agree between forwards).
        Attention code uses this to pick between ``flash_attn_func`` (fast
        path, uniform Ks) and ``flash_attn_with_kvcache`` (per-row mask).
        """
        B = self.b_live
        if B <= 1:
            return True
        row = self.seen_tokens[0, :B]
        return bool((row == row[0]).all().item())

    @torch.no_grad()
    def set_seen_tokens_per_row(self, per_row_lengths: torch.Tensor) -> None:
        """Override ``seen_tokens`` for all live rows with the given
        per-row values. Used by ``_generate_multiple`` after a batched
        prefill on right-padded inputs: the forward advances all layers
        uniformly by the padded seqlen, but each row's real prompt ends
        earlier, so this call clips each row's seen_tokens to its real
        length before decode begins.

        Pre: per_row_lengths.shape == (b_live,), 0 <= values <= max_seqlen.
        Post: seen_tokens[:, :b_live] == per_row_lengths.unsqueeze(0).expand(n_layers, -1).
              S7 holds (layers uniform); cross-row uniformity may now be
              False — check ``uniform_seen_tokens``.
        """
        B = self.b_live
        if per_row_lengths.shape != (B,):
            raise ValueError(
                f"per_row_lengths must have shape ({B},); got "
                f"{tuple(per_row_lengths.shape)}"
            )
        vals = per_row_lengths.to(device=self.device, dtype=torch.long)
        if bool((vals < 0).any().item()) or bool(
            (vals > self.max_seqlen).any().item()
        ):
            raise ValueError(
                f"per_row_lengths out of [0, {self.max_seqlen}]: {vals.tolist()}"
            )
        self.seen_tokens[:, :B] = vals.unsqueeze(0).expand(self.n_layers, -1)

    def claim(self) -> RolloutId:
        if self.b_live >= self.max_bsz:
            raise RuntimeError(
                f"KVArena full: b_live={self.b_live} max_bsz={self.max_bsz}"
            )
        slot = self.b_live
        rid = RolloutId(self._next_rollout_id)
        self._next_rollout_id += 1
        self.slot_to_rollout.append(rid)
        self.rollout_to_slot[rid] = slot
        self.seen_tokens[:, slot] = 0
        self.attn_mask_cache[slot, :] = 0
        self.role_id_cache[slot, :] = 0
        return rid

    def resolve(self, rid: RolloutId) -> int:
        return self.rollout_to_slot[rid]

    def active_rollouts(self) -> List[RolloutId]:
        return list(self.slot_to_rollout)

    # ---------------------------- Retirement --------------------------- #

    def retire(self, rid: RolloutId) -> KVChunk:
        slot = self.rollout_to_slot[rid]
        length = int(self.seen_tokens[0, slot].item())
        chunk = self._extract(slot, length)
        self._swap_with_last(rid, slot)
        return chunk

    def retire_many(self, rids: Sequence[RolloutId]) -> List[KVChunk]:
        if len(set(rids)) != len(rids):
            raise ValueError("retire_many: duplicate rids")
        pairs = [(rid, self.rollout_to_slot[rid]) for rid in rids]
        chunks = [self._extract(s, int(self.seen_tokens[0, s].item())) for _, s in pairs]
        for rid, _ in sorted(pairs, key=lambda p: p[1], reverse=True):
            current_slot = self.rollout_to_slot[rid]
            self._swap_with_last(rid, current_slot)
        return chunks

    def _swap_with_last(self, rid: RolloutId, slot: int) -> None:
        del self.rollout_to_slot[rid]
        last = len(self.slot_to_rollout) - 1
        if slot != last:
            self.k_cache[:, slot, :] = self.k_cache[:, last, :]
            self.v_cache[:, slot, :] = self.v_cache[:, last, :]
            self.role_id_cache[slot, :] = self.role_id_cache[last, :]
            self.attn_mask_cache[slot, :] = self.attn_mask_cache[last, :]
            self.seen_tokens[:, slot] = self.seen_tokens[:, last]
            moved_rid = self.slot_to_rollout[last]
            self.slot_to_rollout[slot] = moved_rid
            self.rollout_to_slot[moved_rid] = slot
        self.slot_to_rollout.pop()

    # ---------------------------- Chunk I/O ---------------------------- #

    @torch.no_grad()
    def load_chunk(
        self, chunk: KVChunk, rid: RolloutId, at_pos: int = 0,
    ) -> None:
        if at_pos != 0:
            raise NotImplementedError(
                "load_chunk at non-zero at_pos requires RoPE rotation; "
                "not supported. See KVChunk invariant I4."
            )
        slot = self.rollout_to_slot[rid]
        pre_seen = self.seen_tokens[:, slot]
        if bool((pre_seen != 0).any().item()):
            raise RuntimeError(
                f"load_chunk into non-empty slot {slot}: "
                f"seen_tokens={pre_seen.tolist()}"
            )
        if chunk.k.shape[0] != self.n_layers:
            raise ValueError(
                f"chunk n_layers={chunk.k.shape[0]} != arena {self.n_layers}"
            )
        if chunk.k.shape[2] != self.n_kv_heads or chunk.k.shape[3] != self.head_dim:
            raise ValueError(
                f"chunk head shape {tuple(chunk.k.shape[2:])} != "
                f"arena ({self.n_kv_heads}, {self.head_dim})"
            )
        if chunk.length > self.max_seqlen:
            raise ValueError(
                f"chunk length={chunk.length} > max_seqlen={self.max_seqlen}"
            )
        L = chunk.length
        self.k_cache[:, slot, :L] = chunk.k.to(device=self.device, dtype=self.dtype)
        self.v_cache[:, slot, :L] = chunk.v.to(device=self.device, dtype=self.dtype)
        self.role_id_cache[slot, :L] = chunk.role_ids.to(
            device=self.device, dtype=self.role_id_cache.dtype
        )
        self.seen_tokens[:, slot] = L

    @torch.no_grad()
    def extract_chunk(
        self, rid: RolloutId, length: Optional[int] = None,
    ) -> KVChunk:
        slot = self.rollout_to_slot[rid]
        max_len = int(self.seen_tokens[0, slot].item())
        if length is None:
            length = max_len
        elif not (0 <= length <= max_len):
            raise ValueError(
                f"extract_chunk length={length} out of [0, {max_len}]"
            )
        return self._extract(slot, length)

    def _extract(self, slot: int, length: int) -> KVChunk:
        if length > self.max_seqlen:
            raise ValueError(
                f"extract length={length} > max_seqlen={self.max_seqlen}"
            )
        k = self.k_cache[:, slot, :length].detach().to("cpu").clone()
        v = self.v_cache[:, slot, :length].detach().to("cpu").clone()
        role_ids = self.role_id_cache[slot, :length].detach().to("cpu").clone()
        return KVChunk(k=k, v=v, role_ids=role_ids)

    # ---------------------------- Forward-time ------------------------- #

    @torch.no_grad()
    def update_kv(
        self, layer_id: int, k_val: torch.Tensor, v_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= layer_id < self.n_layers):
            raise ValueError(f"layer_id={layer_id} out of [0, {self.n_layers})")
        B = self.b_live
        if k_val.shape[0] != B or v_val.shape[0] != B:
            raise ValueError(
                f"update_kv expects batch={B}; got k={tuple(k_val.shape)} "
                f"v={tuple(v_val.shape)}"
            )
        if k_val.shape != v_val.shape:
            raise ValueError(
                f"update_kv k/v shape mismatch: k={tuple(k_val.shape)} "
                f"v={tuple(v_val.shape)}"
            )
        S = k_val.shape[1]

        positions = self.seen_tokens[layer_id, :B]
        if bool((positions + S > self.max_seqlen).any().item()):
            raise RuntimeError(
                f"update_kv would exceed max_seqlen: "
                f"positions={positions.tolist()} S={S} max={self.max_seqlen}"
            )

        pos0 = int(positions[0].item()) if B > 0 else 0
        uniform = B > 0 and bool((positions == pos0).all().item())
        if uniform:
            self.k_cache[layer_id, :B, pos0 : pos0 + S] = k_val
            self.v_cache[layer_id, :B, pos0 : pos0 + S] = v_val
        else:
            for i in range(B):
                p = int(positions[i].item())
                self.k_cache[layer_id, i, p : p + S] = k_val[i]
                self.v_cache[layer_id, i, p : p + S] = v_val[i]

        self.seen_tokens[layer_id, :B] += S

        max_end = int(self.seen_tokens[layer_id, :B].max().item()) if B > 0 else 0
        k_full = self.k_cache[layer_id, :B, :max_end]
        v_full = self.v_cache[layer_id, :B, :max_end]
        return k_full, v_full

    @torch.no_grad()
    def update_kv_decode_static(
        self, layer_id: int, k_val: torch.Tensor, v_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode-only KV update with fixed output shapes.

        This path is intended for ``torch.compile`` / CUDA graph experiments.
        It avoids Python ``.item()`` conversions and returns the full static
        per-layer cache plus tensor ``cache_seqlens``. Precondition: S == 1.
        """
        B = self.b_live
        if k_val.shape[0] != B or v_val.shape[0] != B:
            raise ValueError(
                f"update_kv_decode_static expects batch={B}; "
                f"got k={tuple(k_val.shape)} v={tuple(v_val.shape)}"
            )
        if k_val.shape != v_val.shape:
            raise ValueError(
                f"update_kv_decode_static k/v shape mismatch: "
                f"k={tuple(k_val.shape)} v={tuple(v_val.shape)}"
            )
        if k_val.shape[1] != 1:
            raise ValueError(
                f"update_kv_decode_static requires S=1; got S={k_val.shape[1]}"
            )

        positions = self.seen_tokens[layer_id, :B]
        rows = torch.arange(B, device=self.device)
        self.k_cache[layer_id, rows, positions] = k_val[:, 0]
        self.v_cache[layer_id, rows, positions] = v_val[:, 0]
        cache_seqlens = positions + 1
        self.seen_tokens[layer_id, :B] = cache_seqlens
        return (
            self.k_cache[layer_id, :B],
            self.v_cache[layer_id, :B],
            cache_seqlens,
        )

    @torch.no_grad()
    def update_role_ids(
        self, role_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if role_ids is None:
            return None
        B = self.b_live
        if role_ids.shape[0] != B:
            raise ValueError(
                f"update_role_ids expects batch={B}; got {tuple(role_ids.shape)}"
            )
        S = role_ids.shape[1]
        positions = self.seen_tokens[0, :B]
        if bool((positions + S > self.max_seqlen).any().item()):
            raise RuntimeError(
                f"update_role_ids would exceed max_seqlen: "
                f"positions={positions.tolist()} S={S}"
            )
        pos0 = int(positions[0].item()) if B > 0 else 0
        uniform = B > 0 and bool((positions == pos0).all().item())
        if uniform:
            self.role_id_cache[:B, pos0 : pos0 + S] = role_ids
        else:
            for i in range(B):
                p = int(positions[i].item())
                self.role_id_cache[i, p : p + S] = role_ids[i]
        max_end = int(positions.max().item()) + S if B > 0 else 0
        return self.role_id_cache[:B, :max_end]

    @torch.no_grad()
    def update_attn_mask(
        self, attn_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        B = self.b_live
        if attn_mask is None:
            if not self.is_attn_mask_cached or B == 0:
                return None
            positions = self.seen_tokens[0, :B]
            rows = torch.arange(B, device=self.device)
            self.attn_mask_cache[rows, positions] = 1
            max_end = int(positions.max().item()) + 1
            return self.attn_mask_cache[:B, :max_end]

        if attn_mask.shape[0] != B:
            raise ValueError(
                f"update_attn_mask expects batch={B}; got {tuple(attn_mask.shape)}"
            )
        S = attn_mask.shape[1]
        positions = self.seen_tokens[0, :B]
        if bool((positions + S > self.max_seqlen).any().item()):
            raise RuntimeError(
                f"update_attn_mask would exceed max_seqlen: "
                f"positions={positions.tolist()} S={S}"
            )
        pos0 = int(positions[0].item()) if B > 0 else 0
        uniform = B > 0 and bool((positions == pos0).all().item())
        if uniform:
            self.attn_mask_cache[:B, pos0 : pos0 + S] = attn_mask
        else:
            for i in range(B):
                p = int(positions[i].item())
                self.attn_mask_cache[i, p : p + S] = attn_mask[i]
        self.is_attn_mask_cached = True
        max_end = int(positions.max().item()) + S if B > 0 else 0
        return self.attn_mask_cache[:B, :max_end]

    def is_full(self) -> bool:
        B = self.b_live
        if B == 0:
            return False
        return bool((self.seen_tokens[0, :B] >= self.max_seqlen).any().item())

    # ---------------------------- Invariants --------------------------- #

    def _check_invariants(self) -> None:
        """Validate S1-S7. Used by tests; not a hot path. Raises
        AssertionError on violation."""
        B = self.b_live
        assert len(self.slot_to_rollout) == len(self.rollout_to_slot) == B, (
            f"S1 violation: slot_to_rollout={len(self.slot_to_rollout)} "
            f"rollout_to_slot={len(self.rollout_to_slot)} b_live={B}"
        )
        for i, rid in enumerate(self.slot_to_rollout):
            assert self.rollout_to_slot[rid] == i, (
                f"S2 violation: slot {i} holds {rid!r} but "
                f"rollout_to_slot[{rid!r}]={self.rollout_to_slot[rid]}"
            )
        for rid, slot in self.rollout_to_slot.items():
            assert 0 <= slot < B, (
                f"S2/S3 violation: slot {slot} out of [0, {B}) for rid {rid!r}"
            )
            assert self.slot_to_rollout[slot] == rid, (
                f"S3 violation: rollout_to_slot[{rid!r}]={slot} but "
                f"slot_to_rollout[{slot}]={self.slot_to_rollout[slot]!r}"
            )
        if B > 0:
            live = self.seen_tokens[:, :B]
            assert bool((live >= 0).all().item()), "S4 violation: seen_tokens < 0"
            assert bool((live <= self.max_seqlen).all().item()), (
                f"S4 violation: seen_tokens > max_seqlen={self.max_seqlen}"
            )
            # S7 — between-forward consistency across layers.
            first = live[0]
            assert bool((live == first.unsqueeze(0)).all().item()), (
                "S7 violation: seen_tokens disagree across layers"
            )
        for rid in self.rollout_to_slot:
            assert rid < self._next_rollout_id, (
                f"S6 violation: rid {rid} >= next_rollout_id {self._next_rollout_id}"
            )
