"""Integration tests for LLM._generate_single with prefix caching.

Uses a deterministic fake transformer (CPU, bf16) so we can verify:
  1. Cached and uncached generations produce identical output tokens.
  2. Cached generation prefills fewer tokens than uncached.
  3. The prefix cache gets populated after a call.
  4. Subsequent calls with shared prefixes reuse the cached KV.
"""

from __future__ import annotations

import torch

from torchllms.inference.llm import LLM
from torchllms.inference.prefix_cache import RadixKVCache
from torchllms.models.cache import KVArena
from torchllms.models.networks import _eager_attention, _sdpa_attention


VOCAB = 32
N_LAYERS = 2
N_KV_HEADS = 2
HEAD_DIM = 4


class _FakeParams:
    n_layers = N_LAYERS
    n_kv_heads = N_KV_HEADS
    head_dim = HEAD_DIM
    vocab_size = VOCAB
    max_seq_len = 256
    attention_impl = None


class FakeTransformer(torch.nn.Module):
    """Deterministic fake transformer.

    Semantics:
      - Embedding is identity-ish: token id at position p → K[p] = p, V[p] = -p
        (same for all layers + heads). This makes KV chunks easy to verify.
      - Forward writes deterministic K/V based on absolute position
        (row_positions[0] at forward start + offset), via cache.update_kv.
      - Forward produces logits that are argmax=(last_pos + 1) % VOCAB, so
        greedy decode produces positions 1, 2, 3... regardless of input ids.
        That lets us compare cached vs uncached sample output by "position"
        alone.
    """

    def __init__(self):
        super().__init__()
        self.params = _FakeParams()
        self.tok_embeddings = torch.nn.Embedding(VOCAB, 16, dtype=torch.bfloat16)
        self.forward_calls: list[dict] = []

    def init_cache(self, max_batch_size: int, device: str, max_cache_len=None):
        return KVArena(
            n_layers=self.params.n_layers,
            max_bsz=max_batch_size,
            max_seqlen=max_cache_len or self.params.max_seq_len,
            n_kv_heads=self.params.n_kv_heads,
            head_dim=self.params.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(
        self,
        input_ids,
        role_ids=None,
        cache=None,
        logits_to_keep=None,
        attn_mask=None,
        input_pos=None,
        use_kvcache_attn=False,
        step_type="prefill",
    ):
        # The FakeTransformer doesn't actually do attention; it writes
        # deterministic K/V and produces deterministic logits. The
        # use_kvcache_attn flag is accepted for signature parity.
        del use_kvcache_attn
        del step_type
        B, S = input_ids.shape
        self.forward_calls.append({"input_ids": input_ids.clone(), "seqlen": S})

        assert cache is not None, "FakeTransformer expects cache in tests"
        assert cache.b_live == B, (
            f"FakeTransformer: cache.b_live={cache.b_live} != input_ids.shape[0]={B}"
        )

        cache.update_role_ids(role_ids)
        cache.update_attn_mask(attn_mask)

        # Pre-update row_positions = per-row starting absolute position.
        # Supports both uniform and diverging batched forward.
        row_starts = cache.row_positions.clone()  # [B] on device
        device = input_ids.device

        # K[p] = absolute position p, V[p] = -p per row. Build a [B, S] matrix
        # of absolute positions, one row per input sequence. Under diverging
        # row_starts, each row's K values reflect its own position axis.
        arange_s = torch.arange(S, dtype=torch.bfloat16, device=device)
        abs_pos_bs = arange_s[None, :] + row_starts[:, None].to(torch.bfloat16)  # [B, S]
        k_single = abs_pos_bs[:, :, None, None].expand(B, S, N_KV_HEADS, HEAD_DIM).contiguous()
        v_single = -k_single

        for layer_id in range(self.params.n_layers):
            cache.update_kv(layer_id, k_single, v_single)

        # Logits: for each row, each output position's argmax = (abs_pos + 1) % VOCAB,
        # where abs_pos depends on that row's own pos_start.
        out_len = logits_to_keep if logits_to_keep is not None else S
        logits = torch.full(
            (B, out_len, VOCAB), fill_value=-10.0, dtype=torch.bfloat16, device=device
        )
        for b in range(B):
            end_pos_b = int(row_starts[b].item()) + S
            for i in range(out_len):
                abs_pos = end_pos_b - out_len + i
                tok = (abs_pos + 1) % VOCAB
                logits[b, i, tok] = 10.0

        return logits, cache


class _FakeTokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def _build_llm(model, prefix_cache=None, max_len=256, eos_ids=None) -> LLM:
    llm = LLM.__new__(LLM)
    llm.model = model
    llm.tokenizer = _FakeTokenizer()
    llm.template_config = None
    llm.max_len = max_len
    llm.device = "cpu"
    llm.batched = False
    # 0 is never produced by our fake (argmax=(p+1)%V, p≥0) so the default
    # exercises the max_new_tokens exit path.
    llm.eos_ids = list(eos_ids) if eos_ids is not None else [0]
    llm.eos_set = set(llm.eos_ids)
    llm.prefix_cache = prefix_cache
    return llm


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_sdpa_partial_cache_prefill_matches_eager_causal_reference():
    """Partial-cache prefill has qlen < seqlen and qlen > 1.

    That shape needs a shifted causal mask: suffix token i can attend to the
    loaded prefix and earlier suffix tokens, but not later suffix tokens.
    """
    torch.manual_seed(0)
    bsz, qlen, past, n_heads, head_dim = 2, 3, 2, 2, 4
    seqlen = past + qlen
    xq = torch.randn(bsz, qlen, n_heads, head_dim)
    xk = torch.randn(bsz, seqlen, n_heads, head_dim)
    xv = torch.randn(bsz, seqlen, n_heads, head_dim)

    eager = _eager_attention(xq, xk, xv, attn_mask=None)
    sdpa = _sdpa_attention(xq, xk, xv, attn_mask=None)
    max_diff = (eager - sdpa).abs().max().item()
    assert max_diff < 1e-5, f"max_diff={max_diff}"


def test_no_prefix_cache_matches_baseline():
    """With prefix_cache=None, behavior should be unchanged."""
    model = FakeTransformer()
    llm = _build_llm(model, prefix_cache=None)
    prompt = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    out = llm._generate_single(prompt, max_new_tokens=6)
    # Greedy logits at query pos p produce token (p+1) % VOCAB. Prompt of
    # length 5 occupies positions 0..4; first sample (pos 4) = 5, then
    # 6, 7, 8, 9, 10 from positions 5..9.
    assert out.text == "5 6 7 8 9 10", f"got {out.text!r}"
    assert out.stop_reason is None  # budget exhausted, no eos match
    # One prefill call (5 tokens) + 5 decode calls = 6 total.
    assert len(model.forward_calls) == 6
    assert model.forward_calls[0]["seqlen"] == 5


def test_single_max_new_tokens_zero_returns_empty():
    model = FakeTransformer()
    llm = _build_llm(model, prefix_cache=None)
    prompt = torch.tensor([[5, 6, 7]], dtype=torch.long)
    out = llm._generate_single(prompt, max_new_tokens=0)
    assert out.token_ids == []
    assert out.text == ""
    assert out.stop_reason is None
    assert model.forward_calls == []


def test_prefix_cache_populated_after_call():
    """First call with an empty prefix_cache should insert the prefix.

    Caveat: we cache tokens whose K/V were actually written to the arena,
    which is prompt_len + (generated_len - 1) — the final generated token
    is sampled from logits but never fed back through the model, so its
    K/V isn't computed. The next turn will re-prefill that last token
    (cheap). Worth keeping the logic simple.
    """
    model = FakeTransformer()
    cache = RadixKVCache(max_bytes=16 * 1024 ** 2)
    llm = _build_llm(model, prefix_cache=cache)
    prompt = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    out = llm._generate_single(prompt, max_new_tokens=3)
    assert out.text == "5 6 7", f"got {out.text!r}"
    # Cached: prompt (5) + generated (3) - 1 = 7 tokens.
    match = cache.lookup([5, 6, 7, 8, 9, 5, 6])
    assert match.hit and match.length == 7


def test_second_call_reuses_prefix():
    """Second call with a shared prefix should prefill only the suffix."""
    model = FakeTransformer()
    cache = RadixKVCache(max_bytes=16 * 1024 ** 2)
    llm = _build_llm(model, prefix_cache=cache)
    prompt_a = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    llm._generate_single(prompt_a, max_new_tokens=3)

    # Second call: shares 5-token prefix with cache (which has
    # [5,6,7,8,9,5,6,7]). Query is [5,6,7,8,9,100,101] — diverges at pos 5.
    prompt_b = torch.tensor([[5, 6, 7, 8, 9, 100, 101]], dtype=torch.long)
    model.forward_calls.clear()
    out_b = llm._generate_single(prompt_b, max_new_tokens=3)

    # matched=5, prefill_start=5, suffix=2 tokens. + 2 decode steps
    # (max_new_tokens=3 → 1 from prefill + 2 from decode). Total 3 calls.
    assert len(model.forward_calls) == 3, f"got {len(model.forward_calls)}"
    assert model.forward_calls[0]["seqlen"] == 2
    # Query token count = 7, abs positions 0..6. First sample from pos 6 → 7.
    # Then 8, 9.
    assert out_b.text == "7 8 9", f"got {out_b.text!r}"


def test_cached_and_uncached_produce_identical_output():
    """Critical correctness: output must be bit-identical with vs without cache."""
    prompt = torch.tensor([[5, 6, 7, 8, 9, 10, 11]], dtype=torch.long)

    # Path 1: no prefix cache, fresh model
    model_a = FakeTransformer()
    llm_a = _build_llm(model_a, prefix_cache=None)
    out_a = llm_a._generate_single(prompt, max_new_tokens=4)

    # Path 2: with prefix cache, pre-populated from a prior call
    model_b = FakeTransformer()
    cache = RadixKVCache(max_bytes=16 * 1024 ** 2)
    llm_b = _build_llm(model_b, prefix_cache=cache)
    # Seed the cache with the first 5 tokens by running generate once.
    llm_b._generate_single(prompt[:, :5], max_new_tokens=2)
    # Now generate from the full 7-token prompt.
    model_b.forward_calls.clear()
    out_b = llm_b._generate_single(prompt, max_new_tokens=4)
    assert out_a.text == out_b.text, f"text differs: {out_a.text!r} vs {out_b.text!r}"
    assert out_a.token_ids == out_b.token_ids, (
        f"token_ids differ: {out_a.token_ids!r} vs {out_b.token_ids!r}"
    )


def test_full_prefix_match_leaves_one_token_to_prefill():
    """If the prompt is entirely in the cache, we must still prefill at least
    one token so fresh logits are produced for sampling."""
    model = FakeTransformer()
    cache = RadixKVCache(max_bytes=16 * 1024 ** 2)
    llm = _build_llm(model, prefix_cache=cache)
    prompt = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    # First call inserts prompt+generated into cache.
    llm._generate_single(prompt, max_new_tokens=3)
    # Cache now has 8 tokens; the first 5 match the prompt. Second call:
    # ask it to regenerate from just [5,6,7,8,9].
    model.forward_calls.clear()
    out = llm._generate_single(prompt, max_new_tokens=3)
    # Prefill must be at least 1 token even though all 5 matched in the cache.
    assert model.forward_calls[0]["seqlen"] == 1
    # Output must match the first call's.
    assert out.text == "5 6 7", f"got {out.text!r}"


# ------------------------------------------------------------------
# Stop-token semantics
# ------------------------------------------------------------------


def test_stop_token_any_of_fires_on_first_match():
    """eos_set is any-of: the loop must stop on the first sampled stop ID.

    FakeTransformer produces tokens 5, 6, 7, ... after a 5-token prompt. With
    eos_ids=[7, 6], the *first* stop to fire is 6 (at step 2). 7 never gets
    sampled, so a multi-token-suffix interpretation would run to budget; the
    set-membership interpretation stops here.
    """
    model = FakeTransformer()
    llm = _build_llm(model, prefix_cache=None, eos_ids=[7, 6])
    prompt = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    out = llm._generate_single(prompt, max_new_tokens=10)
    assert out.stop_reason == 6, f"got {out.stop_reason!r}"
    assert out.token_ids == [5, 6], f"got {out.token_ids!r}"
    assert out.text == "5 6", f"got {out.text!r}"


def test_stop_token_on_first_sample():
    """First sampled token equal to a stop ID must return immediately with
    stop_reason set and the stop token present in token_ids."""
    model = FakeTransformer()
    llm = _build_llm(model, prefix_cache=None, eos_ids=[5])
    prompt = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    out = llm._generate_single(prompt, max_new_tokens=10)
    assert out.stop_reason == 5, f"got {out.stop_reason!r}"
    assert out.token_ids == [5], f"got {out.token_ids!r}"
    # Only the prefill forward should have happened; no decode steps.
    assert len(model.forward_calls) == 1


def test_no_stop_match_reports_stop_reason_none():
    """Budget exhaustion (no stop ID ever sampled) must set stop_reason=None."""
    model = FakeTransformer()
    # 99 is never produced by FakeTransformer (argmax=(p+1)%32 over small p).
    llm = _build_llm(model, prefix_cache=None, eos_ids=[99])
    prompt = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    out = llm._generate_single(prompt, max_new_tokens=4)
    assert out.stop_reason is None
    assert len(out.token_ids) == 4


def test_multiple_alternatives_semantics():
    """Alternatives, not a sequence: eos_ids=[A, B] stops on A or B
    independently. The original bug treated [A, B] as the ordered pair — the
    loop would only break if positions i-1, i were exactly [A, B]. Here the
    generated stream is 5, 6, 7, 8, ... so positions 1-2 are [6, 7]: under
    the buggy semantics with eos_ids=[6, 7] the break IS accidentally
    triggered. With eos_ids=[8, 6] (no subsequence match) the buggy code
    would run to budget; the fixed code stops on 6 at step 2.
    """
    model = FakeTransformer()
    llm = _build_llm(model, prefix_cache=None, eos_ids=[8, 6])
    prompt = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    out = llm._generate_single(prompt, max_new_tokens=10)
    assert out.stop_reason == 6, f"got {out.stop_reason!r}"
    assert out.token_ids == [5, 6], f"got {out.token_ids!r}"


# ------------------------------------------------------------------
# _generate_multiple (batched) — v1 contract
# ------------------------------------------------------------------


def test_multi_uniform_prompts_match_single():
    """Batched generation of B=4 identical prompts produces B identical
    GenerationResults that each match the single-row call."""
    model_s = FakeTransformer()
    llm_s = _build_llm(model_s, prefix_cache=None, eos_ids=[0])
    prompt = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    single = llm_s._generate_single(prompt, max_new_tokens=4)

    model_m = FakeTransformer()
    llm_m = _build_llm(model_m, prefix_cache=None, eos_ids=[0])
    inputs = [[5, 6, 7, 8, 9]] * 4
    multi = llm_m._generate_multiple(inputs, max_new_tokens=4)
    assert len(multi) == 4
    for r in multi:
        assert r.text == single.text, f"{r.text!r} != {single.text!r}"
        assert r.token_ids == single.token_ids
        assert r.stop_reason == single.stop_reason


def test_multi_all_rows_stop_on_eos_simultaneously():
    """FakeTransformer emits deterministic tokens (p+1)%VOCAB; same prompt
    across rows means they stop at the same step. retire_many handles the
    all-at-once retirement correctly."""
    model = FakeTransformer()
    llm = _build_llm(model, prefix_cache=None, eos_ids=[6])
    inputs = [[5, 6, 7, 8, 9]] * 3
    out = llm._generate_multiple(inputs, max_new_tokens=5)
    assert len(out) == 3
    for r in out:
        assert r.stop_reason == 6, f"got {r.stop_reason!r}"
        # Prompt length 5 → first sample at pos 4 = token 5; then pos 5 = 6.
        assert r.token_ids == [5, 6], f"got {r.token_ids!r}"


def test_multi_budget_exhausted_yields_stop_reason_none():
    model = FakeTransformer()
    llm = _build_llm(model, prefix_cache=None, eos_ids=[99])  # never hit
    inputs = [[5, 6, 7, 8, 9]] * 2
    out = llm._generate_multiple(inputs, max_new_tokens=3)
    for r in out:
        assert r.stop_reason is None
        assert len(r.token_ids) == 3


def test_multi_zero_budget_rows_skip_generation():
    model = FakeTransformer()
    llm = _build_llm(model, prefix_cache=None, max_len=5, eos_ids=[0])
    # Row 0 has zero budget because prompt_len == max_len. Row 1 still has
    # two tokens of room. The zero-budget row must not receive the first
    # sampled token from prefill.
    out = llm._generate_multiple([[1, 2, 3, 4, 5], [1, 2, 3]], max_new_tokens=None)
    assert out[0].token_ids == []
    assert out[0].text == ""
    assert out[0].stop_reason is None
    assert out[1].token_ids == [3, 4]


def test_multi_preserves_input_order():
    """Even after swap-with-last compaction shuffles internal slot order,
    the output list must be in the same order as `input_ids`."""
    model = FakeTransformer()
    llm = _build_llm(model, prefix_cache=None, eos_ids=[0])
    # All rows same prompt → same outputs, but token_ids list should be
    # present for each row at the correct index.
    inputs = [[5, 6, 7, 8, 9]] * 4
    out = llm._generate_multiple(inputs, max_new_tokens=2)
    assert len(out) == 4
    # Verify each result corresponds to its input index (not None).
    assert all(r is not None for r in out)
    # Uniform prompts produce uniform outputs; sanity check the values.
    for r in out:
        assert r.token_ids == [5, 6]


def test_multi_diverging_prompt_lengths_prefill_per_row():
    """Diverging prompt lengths should take the per-row prefill path and
    still produce one GenerationResult per input in the original order."""
    model = FakeTransformer()
    llm = _build_llm(model, prefix_cache=None, eos_ids=[0])
    inputs = [[5, 6, 7], [5, 6, 7, 8, 9]]
    out = llm._generate_multiple(inputs, max_new_tokens=2)
    assert len(out) == 2
    # First prompt is length 3; first sampled token from position 2 = 3.
    # Second prompt is length 5; first sampled token from position 4 = 5.
    # Then each row does one more decode step (max_new_tokens=2).
    assert out[0].token_ids == [3, 4], f"got {out[0].token_ids!r}"
    assert out[1].token_ids == [5, 6], f"got {out[1].token_ids!r}"


def test_multi_diverging_matches_per_row_single():
    """Each row of a diverging-length batched call should produce the same
    tokens as calling _generate_single on that row alone (using the same
    FakeTransformer)."""
    prompts = [[5, 6, 7], [5, 6, 7, 8, 9, 10], [5, 6]]

    model_m = FakeTransformer()
    llm_m = _build_llm(model_m, prefix_cache=None, eos_ids=[0])
    multi = llm_m._generate_multiple(prompts, max_new_tokens=3)

    for i, p in enumerate(prompts):
        model_s = FakeTransformer()
        llm_s = _build_llm(model_s, prefix_cache=None, eos_ids=[0])
        single = llm_s._generate_single(
            torch.tensor([p], dtype=torch.long), max_new_tokens=3,
        )
        assert multi[i].token_ids == single.token_ids, (
            f"row {i}: multi={multi[i].token_ids!r} single={single.token_ids!r}"
        )


def test_multi_empty_input_returns_empty():
    model = FakeTransformer()
    llm = _build_llm(model, prefix_cache=None)
    assert llm._generate_multiple([], max_new_tokens=2) == []


def test_multi_inserts_into_prefix_cache_per_row():
    """Each retired row should land in the radix cache as its own entry.
    FakeTransformer produces the same tokens across rows, but we use
    distinct prompts so each row inserts a distinct sequence."""
    model = FakeTransformer()
    radix = RadixKVCache(max_bytes=16 * 1024 ** 2)
    llm = _build_llm(model, prefix_cache=radix, eos_ids=[0])
    # Two rows with same-length but different prompts so they create
    # separate radix entries. Uniform length still required by v1.
    inputs = [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    out = llm._generate_multiple(inputs, max_new_tokens=3)
    assert len(out) == 2
    # Row 0: prompt [5,6,7,8,9], generated positions 5,6,7 → [5,6,7].
    # Cached: prompt + generated[:-1] = [5,6,7,8,9,5,6], length 7.
    m0 = radix.lookup([5, 6, 7, 8, 9, 5, 6])
    assert m0.hit and m0.length == 7, f"row0 lookup length={m0.length}"
    # Row 1: prompt [10,11,12,13,14]. FakeTransformer's logits depend on
    # absolute position only, not on input token content; since both rows
    # share prompt length (5), they generate the same tokens [5,6,7].
    # Cached: prompt + generated[:-1] = [10,11,12,13,14,5,6], length 7.
    m1 = radix.lookup([10, 11, 12, 13, 14, 5, 6])
    assert m1.hit and m1.length == 7, f"row1 lookup length={m1.length}"


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
