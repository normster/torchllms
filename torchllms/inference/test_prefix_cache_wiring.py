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
from torchllms.models.cache import LinearKVCache


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
        (same for all layers + heads). This makes KV blocks easy to verify.
      - Forward writes deterministic K/V based on absolute position
        (cache.seen_tokens[layer] + offset), via cache.update_kv.
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
        return LinearKVCache(
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
    ):
        B, S = input_ids.shape
        self.forward_calls.append({"input_ids": input_ids.clone(), "seqlen": S})

        assert cache is not None, "FakeTransformer expects cache in tests"

        cache.update_role_ids(role_ids)
        cache.update_attn_mask(attn_mask)

        pos_start = cache.seen_tokens[0]
        device = input_ids.device
        # K[p] = absolute position p, V[p] = -p (same across layers/heads).
        positions = torch.arange(pos_start, pos_start + S, dtype=torch.bfloat16, device=device)
        k_single = positions.view(1, S, 1, 1).expand(B, S, N_KV_HEADS, HEAD_DIM).contiguous()
        v_single = -k_single

        for layer_id in range(self.params.n_layers):
            cache.update_kv(layer_id, k_single, v_single)

        cache.next_start_pos = torch.tensor(
            [cache.seen_tokens[0]] * B, device=device, dtype=torch.long
        )

        # Logits: each position's argmax = (p + 1) % VOCAB. Greedy sampling
        # from position p thus emits token (p+1).
        end_pos = pos_start + S
        out_len = logits_to_keep if logits_to_keep is not None else S
        logits = torch.full(
            (B, out_len, VOCAB), fill_value=-10.0, dtype=torch.bfloat16, device=device
        )
        for i in range(out_len):
            abs_pos = end_pos - out_len + i  # absolute position of this query token
            tok = (abs_pos + 1) % VOCAB
            logits[:, i, tok] = 10.0

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
