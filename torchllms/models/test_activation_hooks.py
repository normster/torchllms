"""Tests for Transformer activation-intervention hooks.

These run without pytest:
    PYTHONPATH=torchllms python torchllms/torchllms/models/test_activation_hooks.py
"""

from __future__ import annotations

import torch

from torchllms.messages import Role
from torchllms.models.interventions import AdditiveVectorIntervention
from torchllms.models.networks import AttentionImpl, ModelParams, Transformer


def _mk_model(n_layers: int = 2) -> Transformer:
    torch.manual_seed(0)
    params = ModelParams(
        dim=16,
        head_dim=8,
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


def test_noop_intervention_hook_is_called_and_preserves_logits():
    model = _mk_model(n_layers=2)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    role_ids = torch.tensor(
        [[int(Role.USER), int(Role.TOOL), int(Role.TOOL)]],
        dtype=torch.long,
    )
    base_logits, _ = model(input_ids, role_ids=role_ids, logits_to_keep=1)

    calls = []

    def hook(hidden, ctx):
        calls.append(
            {
                "layer_id": ctx.layer_id,
                "alpha": ctx.alpha,
                "input_ids": ctx.input_ids.clone(),
                "role_ids": ctx.role_ids.clone(),
            }
        )
        return hidden

    model.set_activation_hooks(hook)
    hooked_logits, _ = model(input_ids, role_ids=role_ids, logits_to_keep=1)

    assert len(calls) == model.params.n_layers
    assert all(c["alpha"] is None for c in calls)
    assert torch.equal(calls[0]["input_ids"], input_ids)
    assert torch.equal(calls[0]["role_ids"], role_ids)
    assert torch.equal(base_logits, hooked_logits)


def test_zero_scale_metadata_does_not_skip_hook_dispatch():
    model = _mk_model(n_layers=1)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    role_ids = torch.tensor(
        [[int(Role.USER), int(Role.TOOL), int(Role.TOOL)]],
        dtype=torch.long,
    )
    base_logits, _ = model(input_ids, role_ids=role_ids, logits_to_keep=1)

    calls = []
    direction = torch.ones(model.params.dim)

    def hook(hidden, ctx):
        calls.append(ctx.alpha)
        scale = 1.0 if ctx.alpha is None else ctx.alpha
        mask = (ctx.role_ids == int(Role.TOOL)).unsqueeze(-1)
        return hidden + scale * mask * direction.to(hidden)

    model.set_activation_hooks(hook, alpha=0.0)
    hooked_logits, _ = model(input_ids, role_ids=role_ids, logits_to_keep=1)

    assert calls == [0.0]
    assert torch.equal(base_logits, hooked_logits)


def test_vectorized_role_mask_edits_only_selected_positions():
    model = _mk_model(n_layers=1)
    hidden = torch.zeros((2, 4, model.params.dim))
    input_ids = torch.arange(8).view(2, 4)
    role_ids = torch.tensor(
        [
            [int(Role.USER), int(Role.TOOL), int(Role.ASSISTANT), int(Role.TOOL)],
            [int(Role.TOOL), int(Role.USER), int(Role.USER), int(Role.TOOL)],
        ],
        dtype=torch.long,
    )
    input_pos = torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]], dtype=torch.long)
    direction = torch.arange(model.params.dim, dtype=hidden.dtype)

    def hook(h, ctx):
        mask = (ctx.role_ids == int(Role.TOOL)).unsqueeze(-1)
        scale = 1.0 if ctx.alpha is None else ctx.alpha
        return h + scale * mask * direction.to(h)

    model.set_activation_hooks(hook, alpha=2.0)
    out = model._apply_activation_hooks(
        hidden,
        layer_id=0,
        input_ids=input_ids,
        role_ids=role_ids,
        input_pos=input_pos,
    )

    expected = torch.zeros_like(hidden)
    expected[role_ids == int(Role.TOOL)] = 2.0 * direction
    assert torch.equal(out, expected)


def test_layer_specific_hook_edits_all_tokens_without_alpha_metadata():
    model = _mk_model(n_layers=2)
    hidden = torch.zeros((2, 3, model.params.dim))
    input_ids = torch.arange(6).view(2, 3)
    direction = torch.ones(model.params.dim)
    calls = []

    def hook(h, ctx):
        calls.append((ctx.layer_id, ctx.alpha))
        if ctx.layer_id != 1:
            return h
        return h + direction.to(h)

    model.set_activation_hooks(hook)
    layer0 = model._apply_activation_hooks(
        hidden,
        layer_id=0,
        input_ids=input_ids,
        role_ids=None,
        input_pos=None,
    )
    layer1 = model._apply_activation_hooks(
        hidden,
        layer_id=1,
        input_ids=input_ids,
        role_ids=None,
        input_pos=None,
    )

    assert calls == [(0, None), (1, None)]
    assert torch.equal(layer0, hidden)
    assert torch.equal(layer1, torch.ones_like(hidden))


def test_additive_vector_intervention_edits_selected_layer_all_tokens():
    model = _mk_model(n_layers=2)
    hidden = torch.zeros((2, 3, model.params.dim))
    input_ids = torch.arange(6).view(2, 3)
    vector = torch.arange(model.params.dim, dtype=hidden.dtype)

    hook = AdditiveVectorIntervention(vector, layers=1, scale=0.5)
    model.set_activation_hooks(hook)
    layer0 = model._apply_activation_hooks(
        hidden,
        layer_id=0,
        input_ids=input_ids,
        role_ids=None,
        input_pos=None,
    )
    layer1 = model._apply_activation_hooks(
        hidden,
        layer_id=1,
        input_ids=input_ids,
        role_ids=None,
        input_pos=None,
    )

    assert list(model.activation_hook_modules) == [hook]
    assert torch.equal(layer0, hidden)
    assert torch.equal(layer1, 0.5 * vector.view(1, 1, -1).expand_as(hidden))


def test_additive_vector_intervention_masks_by_role_and_position():
    model = _mk_model(n_layers=1)
    hidden = torch.zeros((2, 4, model.params.dim))
    input_ids = torch.arange(8).view(2, 4)
    role_ids = torch.tensor(
        [
            [int(Role.USER), int(Role.TOOL), int(Role.TOOL), int(Role.TOOL)],
            [int(Role.TOOL), int(Role.USER), int(Role.TOOL), int(Role.TOOL)],
        ],
        dtype=torch.long,
    )
    input_pos = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    vector = torch.ones(model.params.dim)

    hook = AdditiveVectorIntervention(
        vector,
        layers=0,
        role_ids=Role.TOOL,
        position_start=2,
        position_end=3,
    )
    model.set_activation_hooks(hook)
    out = model._apply_activation_hooks(
        hidden,
        layer_id=0,
        input_ids=input_ids,
        role_ids=role_ids,
        input_pos=input_pos,
    )

    expected_mask = (role_ids == int(Role.TOOL)) & (input_pos >= 2) & (input_pos <= 3)
    expected = expected_mask.to(hidden.dtype).unsqueeze(-1).expand_as(hidden)
    assert torch.equal(out, expected)


def test_fp32_white_noise_intervention_is_deterministic_and_changes_logits():
    model = _mk_model(n_layers=2).float()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    target_layer = 1
    noise_std = 1e-4

    base_logits, _ = model(input_ids, logits_to_keep=1)

    def make_hook(seed):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        calls = []

        def hook(hidden, ctx):
            if ctx.layer_id != target_layer:
                return hidden
            calls.append((ctx.layer_id, tuple(hidden.shape), ctx.alpha))
            noise = torch.randn(
                hidden.shape,
                generator=gen,
                device=hidden.device,
                dtype=torch.float32,
            )
            return (hidden.float() + noise_std * noise).to(dtype=hidden.dtype)

        return hook, calls

    hook, calls = make_hook(seed=123)
    model.set_activation_hooks(hook)
    noisy_logits_1, _ = model(input_ids, logits_to_keep=1)

    hook, calls_replay = make_hook(seed=123)
    model.set_activation_hooks(hook)
    noisy_logits_2, _ = model(input_ids, logits_to_keep=1)
    model.clear_activation_hooks()

    diff = (base_logits - noisy_logits_1).abs()
    assert calls == [(target_layer, (1, 4, model.params.dim), None)]
    assert calls_replay == calls
    assert torch.equal(noisy_logits_1, noisy_logits_2)
    assert diff.max().item() > 0.0
    assert diff.max().item() < 1e-2


def test_no_cache_context_exposes_default_positions():
    model = _mk_model(n_layers=1)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    calls = []

    def hook(hidden, ctx):
        calls.append(ctx.input_pos.clone())
        return hidden

    model.set_activation_hooks(hook, alpha=1.0)
    model(input_ids, logits_to_keep=1)

    assert len(calls) == 1
    assert torch.equal(
        calls[0],
        torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32),
    )


def test_partial_cache_prefill_context_exposes_only_active_suffix():
    model = _mk_model(n_layers=1)
    cache = model.init_cache(1, "cpu", max_cache_len=8)
    cache.claim()

    prefix_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)
    prefix_roles = torch.tensor(
        [[int(Role.USER), int(Role.USER), int(Role.USER)]],
        dtype=torch.long,
    )
    model(
        prefix_ids,
        role_ids=prefix_roles,
        cache=cache,
        logits_to_keep=1,
    )

    calls = []

    def hook(hidden, ctx):
        calls.append(
            {
                "input_ids": ctx.input_ids.clone(),
                "role_ids": ctx.role_ids.clone(),
                "input_pos": ctx.input_pos.clone(),
            }
        )
        return hidden

    model.set_activation_hooks(hook, alpha=1.0)
    suffix_ids = torch.tensor([[13, 14]], dtype=torch.long)
    suffix_roles = torch.tensor([[int(Role.TOOL), int(Role.TOOL)]], dtype=torch.long)
    model(
        suffix_ids,
        role_ids=suffix_roles,
        cache=cache,
        logits_to_keep=1,
    )

    assert len(calls) == 1
    assert torch.equal(calls[0]["input_ids"], suffix_ids)
    assert torch.equal(calls[0]["role_ids"], suffix_roles)
    assert torch.equal(calls[0]["input_pos"], torch.tensor([[3, 4]], dtype=torch.int32))


def test_decode_context_exposes_single_token_and_position():
    model = _mk_model(n_layers=1)
    cache = model.init_cache(1, "cpu", max_cache_len=8)
    cache.claim()

    prefix_ids = torch.tensor([[10, 11]], dtype=torch.long)
    prefix_roles = torch.tensor([[int(Role.USER), int(Role.USER)]], dtype=torch.long)
    model(prefix_ids, role_ids=prefix_roles, cache=cache, logits_to_keep=1)

    calls = []

    def hook(hidden, ctx):
        calls.append(
            {
                "hidden_shape": tuple(hidden.shape),
                "input_ids": ctx.input_ids.clone(),
                "role_ids": ctx.role_ids.clone(),
                "input_pos": ctx.input_pos.clone(),
            }
        )
        return hidden

    model.set_activation_hooks(hook, alpha=0.0)
    decode_ids = torch.tensor([[12]], dtype=torch.long)
    decode_roles = torch.tensor([[int(Role.ASSISTANT)]], dtype=torch.long)
    model(
        decode_ids,
        role_ids=decode_roles,
        cache=cache,
        logits_to_keep=1,
    )

    assert len(calls) == 1
    assert calls[0]["hidden_shape"] == (1, 1, model.params.dim)
    assert torch.equal(calls[0]["input_ids"], decode_ids)
    assert torch.equal(calls[0]["role_ids"], decode_roles)
    assert torch.equal(calls[0]["input_pos"], torch.tensor([[2]], dtype=torch.int32))


def test_activation_hook_must_preserve_shape():
    model = _mk_model(n_layers=1)
    hidden = torch.zeros((1, 2, model.params.dim))
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)

    def bad_hook(hidden, ctx):
        return hidden[:, :1]

    model.set_activation_hooks(bad_hook, alpha=1.0)
    try:
        model._apply_activation_hooks(
            hidden,
            layer_id=0,
            input_ids=input_ids,
            role_ids=None,
            input_pos=None,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for shape-changing hook")


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
