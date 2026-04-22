"""Tests for the Transformer activation-intervention API.

The new contract (see ``torchllms.models.interventions``):

- An intervention is an ``nn.Module`` whose ``forward(hidden) -> delta``
  returns a tensor broadcastable to ``hidden.shape``.
- Register via ``Transformer.register_intervention(module, layers=[...],
  role_ids=[...])``. The driver applies ``hidden += mask * delta`` at each
  layer in ``layers``, with ``mask`` 1 at positions whose role is in
  ``role_ids`` (or 1 everywhere if ``role_ids is None``).

Run without pytest:
    PYTHONPATH=torchllms python torchllms/torchllms/models/test_activation_hooks.py

Cache-using tests require CUDA (the paged KV attention path calls
flashinfer kernels that are GPU-only). No-cache tests run on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchllms.messages import Role
from torchllms.models.interventions import AddVec
from torchllms.models.networks import AttentionImpl, Intervention, ModelParams, Transformer


_CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="paged-cache path requires CUDA (flashinfer kernels)",
)


def _mk_model(n_layers: int = 2) -> Transformer:
    """Tiny Transformer for fast CPU/GPU tests. head_dim=64 because
    flashinfer's paged-attention kernels only compile for head_dim in
    {64, 128, 256} — the cache-based tests below exercise that path."""
    torch.manual_seed(0)
    params = ModelParams(
        dim=128,
        head_dim=64,
        n_layers=n_layers,
        n_heads=2,
        n_kv_heads=2,
        vocab_size=32,
        multiple_of=8,
        ffn_dim_multiplier=2.0,
        max_seq_len=32,
        attention_impl=AttentionImpl.EAGER,
    )
    model = Transformer(params)
    model.eval()
    return model


class _CallCounter(nn.Module):
    """Intervention that records calls + returns zero delta (identity)."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_shapes: list[tuple[int, ...]] = []

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.seen_shapes.append(tuple(hidden.shape))
        return torch.zeros_like(hidden)


class _ConstDelta(nn.Module):
    """Intervention that returns a constant 1-D vector as the delta. Equivalent
    to :class:`AddVec`; kept here for tests that want to assert behavior
    decoupled from the specific AddVec implementation."""

    def __init__(self, vector: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("vector", vector.detach().clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.vector.to(dtype=hidden.dtype)


# ---------------------------------------------------------------------------
# Registration / lifecycle
# ---------------------------------------------------------------------------


def test_register_and_clear_interventions():
    model = _mk_model(n_layers=2)
    assert model.list_interventions() == []
    assert model.intervened_roles() == set()

    model.register_intervention(
        _CallCounter(), layers=[0, 1], role_ids=[Role.TOOL],
    )
    assert len(model.list_interventions()) == 1
    assert model.intervened_roles() == {int(Role.TOOL)}

    model.register_intervention(
        _CallCounter(), layers=[1], role_ids=[Role.USER],
    )
    assert len(model.list_interventions()) == 2
    assert model.intervened_roles() == {int(Role.TOOL), int(Role.USER)}

    model.clear_interventions()
    assert model.list_interventions() == []
    assert model.intervened_roles() == set()


def test_intervened_roles_none_when_any_unfiltered():
    """An intervention with role_ids=None matches every role. The union
    collapses to ``None`` to signal 'all positions are intervened'."""
    model = _mk_model(n_layers=1)
    model.register_intervention(_CallCounter(), layers=[0], role_ids=[Role.TOOL])
    model.register_intervention(_CallCounter(), layers=[0], role_ids=None)
    assert model.intervened_roles() is None


def test_intervention_rejects_empty_layers():
    model = _mk_model(n_layers=1)
    with pytest.raises(ValueError):
        model.register_intervention(_CallCounter(), layers=[])


def test_intervention_rejects_empty_role_ids():
    model = _mk_model(n_layers=1)
    with pytest.raises(ValueError):
        model.register_intervention(_CallCounter(), layers=[0], role_ids=[])


# ---------------------------------------------------------------------------
# Apply semantics — via _apply_interventions direct
# ---------------------------------------------------------------------------


def test_noop_intervention_preserves_logits_and_fires_per_layer():
    model = _mk_model(n_layers=2)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    role_ids = torch.tensor(
        [[int(Role.USER), int(Role.TOOL), int(Role.TOOL)]], dtype=torch.long,
    )
    base_logits, _ = model(input_ids, role_ids=role_ids, logits_to_keep=1)

    counter = _CallCounter()
    model.register_intervention(counter, layers=list(range(model.params.n_layers)))
    hooked_logits, _ = model(input_ids, role_ids=role_ids, logits_to_keep=1)
    model.clear_interventions()

    assert len(counter.seen_shapes) == model.params.n_layers
    # Hook saw the active-token slice shape.
    for shape in counter.seen_shapes:
        assert shape == (1, 3, model.params.dim)
    assert torch.equal(base_logits, hooked_logits)


def test_role_filter_edits_only_selected_positions():
    model = _mk_model(n_layers=1)
    hidden = torch.zeros((2, 4, model.params.dim))
    role_ids = torch.tensor(
        [
            [int(Role.USER), int(Role.TOOL), int(Role.ASSISTANT), int(Role.TOOL)],
            [int(Role.TOOL), int(Role.USER), int(Role.USER), int(Role.TOOL)],
        ],
        dtype=torch.long,
    )
    direction = torch.arange(model.params.dim, dtype=hidden.dtype)
    model.register_intervention(
        _ConstDelta(direction), layers=[0], role_ids=[Role.TOOL],
    )

    out = model._apply_interventions(hidden, layer_id=0, role_ids=role_ids)

    expected = torch.zeros_like(hidden)
    expected[role_ids == int(Role.TOOL)] = direction
    assert torch.equal(out, expected)


def test_layer_filter_fires_only_at_target_layer():
    model = _mk_model(n_layers=2)
    hidden = torch.zeros((2, 3, model.params.dim))
    direction = torch.ones(model.params.dim, dtype=hidden.dtype)
    model.register_intervention(_ConstDelta(direction), layers=[1])

    layer0 = model._apply_interventions(hidden, layer_id=0, role_ids=None)
    layer1 = model._apply_interventions(hidden, layer_id=1, role_ids=None)

    assert torch.equal(layer0, hidden)
    assert torch.equal(layer1, torch.ones_like(hidden))


def test_role_filter_skipped_when_role_ids_none():
    """Role-filtered intervention without role_ids supplied in the forward
    is a no-op — we can't identify matching positions."""
    model = _mk_model(n_layers=1)
    hidden = torch.zeros((1, 3, model.params.dim))
    direction = torch.ones(model.params.dim, dtype=hidden.dtype)
    model.register_intervention(
        _ConstDelta(direction), layers=[0], role_ids=[Role.TOOL],
    )
    out = model._apply_interventions(hidden, layer_id=0, role_ids=None)
    assert torch.equal(out, hidden)


def test_addvec_behaves_as_constant_delta():
    model = _mk_model(n_layers=1)
    hidden = torch.zeros((1, 3, model.params.dim))
    direction = torch.arange(model.params.dim, dtype=hidden.dtype)

    model.register_intervention(AddVec(direction), layers=[0])
    out = model._apply_interventions(hidden, layer_id=0, role_ids=None)

    expected = direction.view(1, 1, -1).expand_as(hidden).clone()
    assert torch.equal(out, expected)


def test_addvec_with_alpha_baked_in():
    """α is baked into the stored vector at construction time."""
    model = _mk_model(n_layers=1)
    hidden = torch.zeros((1, 2, model.params.dim))
    direction = torch.ones(model.params.dim)

    alpha = 0.5
    model.register_intervention(AddVec(alpha * direction), layers=[0])
    out = model._apply_interventions(hidden, layer_id=0, role_ids=None)

    assert torch.allclose(out, 0.5 * torch.ones_like(hidden))


def test_compose_two_interventions_applied_in_order():
    """Registration order = application order. Two deltas on the same layer
    with disjoint role filters edit disjoint sets of positions; with
    overlapping filters their deltas sum."""
    model = _mk_model(n_layers=1)
    hidden = torch.zeros((1, 4, model.params.dim))
    role_ids = torch.tensor(
        [[int(Role.USER), int(Role.TOOL), int(Role.USER), int(Role.TOOL)]],
        dtype=torch.long,
    )

    dir_user = torch.full((model.params.dim,), 1.0)
    dir_tool = torch.full((model.params.dim,), 2.0)
    model.register_intervention(AddVec(dir_user), layers=[0], role_ids=[Role.USER])
    model.register_intervention(AddVec(dir_tool), layers=[0], role_ids=[Role.TOOL])

    out = model._apply_interventions(hidden, layer_id=0, role_ids=role_ids)

    expected = torch.zeros_like(hidden)
    expected[0, 0, :] = 1.0  # USER
    expected[0, 1, :] = 2.0  # TOOL
    expected[0, 2, :] = 1.0  # USER
    expected[0, 3, :] = 2.0  # TOOL
    assert torch.equal(out, expected)


# ---------------------------------------------------------------------------
# Determinism across runs — white-noise intervention
# ---------------------------------------------------------------------------


class _WhiteNoise(nn.Module):
    """Fixed-seed white-noise delta. Deterministic given the same seed."""

    def __init__(self, dim: int, seed: int, std: float = 1e-4) -> None:
        super().__init__()
        self.gen = torch.Generator(device="cpu")
        self.seed = seed
        self.std = std
        self.dim = dim
        self.gen.manual_seed(seed)

    def reset(self) -> None:
        self.gen.manual_seed(self.seed)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            hidden.shape, generator=self.gen, device="cpu", dtype=torch.float32,
        )
        return (self.std * noise).to(device=hidden.device, dtype=hidden.dtype)


def test_fp32_white_noise_intervention_is_deterministic_and_changes_logits():
    model = _mk_model(n_layers=2).float()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    noise_std = 1e-4
    target_layer = 1

    base_logits, _ = model(input_ids, logits_to_keep=1)

    wn = _WhiteNoise(model.params.dim, seed=123, std=noise_std)
    model.register_intervention(wn, layers=[target_layer])
    noisy_1, _ = model(input_ids, logits_to_keep=1)
    model.clear_interventions()

    wn2 = _WhiteNoise(model.params.dim, seed=123, std=noise_std)
    model.register_intervention(wn2, layers=[target_layer])
    noisy_2, _ = model(input_ids, logits_to_keep=1)
    model.clear_interventions()

    diff = (base_logits - noisy_1).abs()
    assert torch.equal(noisy_1, noisy_2), "same seed must produce identical output"
    assert diff.max().item() > 0.0
    assert diff.max().item() < 1e-2


# ---------------------------------------------------------------------------
# CUDA / cache integration — active-positions-only invariant
# ---------------------------------------------------------------------------


@_CUDA_ONLY
def test_partial_cache_prefill_sees_only_active_suffix():
    """After a prefix has been written to the cache, a follow-up forward on
    a suffix exposes only the active tokens to the intervention module
    (via the hidden shape)."""
    model = _mk_model(n_layers=1).to(device="cuda", dtype=torch.bfloat16)
    cache = model.init_cache(1, "cuda", max_cache_len=8)
    cache.claim()

    prefix_ids = torch.tensor([[10, 11, 12]], dtype=torch.long, device="cuda")
    prefix_roles = torch.tensor(
        [[int(Role.USER)] * 3], dtype=torch.long, device="cuda",
    )
    model(prefix_ids, role_ids=prefix_roles, cache=cache, logits_to_keep=1)

    counter = _CallCounter()
    model.register_intervention(counter, layers=[0])

    suffix_ids = torch.tensor([[13, 14]], dtype=torch.long, device="cuda")
    suffix_roles = torch.tensor(
        [[int(Role.TOOL)] * 2], dtype=torch.long, device="cuda",
    )
    model(suffix_ids, role_ids=suffix_roles, cache=cache, logits_to_keep=1)
    model.clear_interventions()

    assert counter.seen_shapes == [(1, 2, model.params.dim)]


@_CUDA_ONLY
def test_decode_sees_single_token_shape():
    model = _mk_model(n_layers=1).to(device="cuda", dtype=torch.bfloat16)
    cache = model.init_cache(1, "cuda", max_cache_len=8)
    cache.claim()

    prefix_ids = torch.tensor([[10, 11]], dtype=torch.long, device="cuda")
    prefix_roles = torch.tensor(
        [[int(Role.USER)] * 2], dtype=torch.long, device="cuda",
    )
    model(prefix_ids, role_ids=prefix_roles, cache=cache, logits_to_keep=1)

    counter = _CallCounter()
    model.register_intervention(counter, layers=[0])
    decode_ids = torch.tensor([[12]], dtype=torch.long, device="cuda")
    decode_roles = torch.tensor(
        [[int(Role.ASSISTANT)]], dtype=torch.long, device="cuda",
    )
    model(decode_ids, role_ids=decode_roles, cache=cache, logits_to_keep=1)
    model.clear_interventions()

    assert counter.seen_shapes == [(1, 1, model.params.dim)]


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
            import traceback

            traceback.print_exc()
            print(f"{name}: FAIL - {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
