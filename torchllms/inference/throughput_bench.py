"""Shared throughput benchmark across torchllms models vs SGLang reference.

Factored out of ``validate_qwen3`` so gpt-oss can use the same three
workloads and timing machinery. Prints a unified table so cross-model
comparisons are apples-to-apples.

Workloads:
    W1. short prompt, short completion         — baseline roundtrip
    W2. short prompt, long completion          — decode-heavy
    W3. multiturn long-context debug dialog    — agentic, prefix-cache engaged

The benchmark is parameterized by a ``tokenize_fn`` that renders a Chat
Completions conversation to ``(input_ids, role_ids)``. For Qwen3 this
is ``torchllms.messages.tokenization.tokenize_conversation`` with a
Qwen3 template config; for gpt-oss it is
``torchllms.messages.tokenization_harmony.tokenize_harmony_conversation``.
Everything else (llm._generate_single call, sgl.Engine.generate call,
prefill/decode split) is shared.

Model-specific concerns delegated to the caller:
    - model loading + template config / Harmony setup (validator's job)
    - SGLang engine construction and ``mem_fraction_static`` choice
    - stop-token IDs (passed via ``stop_token_ids`` argument)
    - whether the llm accepts role_ids at all (harmless if None)
    - whether to run the noop-hook trial (``run_noop_hook`` flag)
    - prefix cache for W3 (passed in; None disables)
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Workloads (Chat Completions format — tokenize_fn renders into ids/roles)
# ---------------------------------------------------------------------------


SYSTEM_SHORT = (
    "You are a helpful, concise assistant. Keep answers short and to the "
    "point unless asked for detail."
)

SYSTEM_DEBUG = (
    "You are an experienced platform engineer. The user is debugging a "
    "live production incident. Give concrete, actionable advice; prefer "
    "specific commands or config lines over generalities."
)


W1_CONV = [
    {"role": "system", "content": SYSTEM_SHORT},
    {"role": "user", "content":
        "In one sentence, what's the difference between TCP and UDP?"},
]
W1_MAX_NEW = 60


W2_CONV = [
    {"role": "system", "content": SYSTEM_SHORT},
    {"role": "user", "content":
        "Write a short story (about 300 words) about a lighthouse keeper "
        "who finds a weathered journal in a drawer. Keep the tone "
        "reflective."},
]
W2_MAX_NEW = 500


# W3: realistic multiturn debug dialog. Each user turn adds log/output
# context (~150-300 tokens); assistant replies are short. By the final
# turn the prompt is ~2000+ tokens, mimicking an agentic session where
# prefix-cache reuse matters.
W3_TURNS: list[tuple[str, Optional[str]]] = [
    ("""Intermittent 504s on our auth endpoint, ~5-10% of requests. Sample log:

[2026-04-20 14:02:17] upstream timed out (110: Operation timed out) while reading response header from upstream, client: 10.0.1.42, server: auth.example.com, request: "POST /api/v1/login HTTP/1.1", upstream: "http://auth-service.prod.svc.cluster.local:8080/api/v1/login"

Auth is a stateless Python service, 12 pods behind a K8s Service, Postgres 15 on RDS. What should I check first?""", None),
    ("""`kubectl top pods` shows two pods at ~90% CPU, the other ten at ~30%. Restart cycles fix it briefly then it climbs back in ~5 minutes. Same two pods keep getting hot.""", None),
    ("""The Service spec has sessionAffinity: ClientIP. But the auth flow is genuinely stateless — no per-pod caches or connection pools that I know of. What else could make ClientIP-sticky routing look like hot-pod saturation?""", None),
    ("""I checked ingress logs and saw that 80% of the traffic to those two pods comes from a single /24 CIDR (internal service mesh). Is that the smoking gun?""", None),
    ("""We set Service externalTrafficPolicy: Cluster and that routed traffic via kube-proxy which load-balances across pods, but now I'm seeing extra latency from the additional NAT hop. Is there a better fix?""", None),
    ("""Implemented Local policy + NodePort. Latency dropped back to normal, but now some requests fail with connection-refused from specific nodes. What's happening?""", None),
    ("""Ran `kubectl get endpoints auth-service -o wide` — only 8 of 12 pods show up. The missing 4 are all on nodes that don't have the auth-service Pod. Is Local policy incompatible with our deployment?""", None),
    ("""Okay so adding pod anti-affinity to spread pods across all nodes fixes it. Any downsides to that approach at our scale (30-node cluster, 12 replicas)?""", None),
]
W3_MAX_NEW_DEFAULT = 80


# Fabricated assistant replies — short enough to keep W3 bounded, realistic
# enough in tone. Content doesn't need to match actual model generations
# because the test measures cache/tokenize/decode throughput, not quality.
_W3_FAKE_ASSISTANT = [
    "Hot pods with intermittent timeouts usually means skewed request distribution. Check `kubectl top pods` for per-pod CPU and confirm the Service type — round-robin should even this out.",
    "That's a smoking gun: persistent hot pods after restart points to a routing layer, not a workload issue. Look at the Service config, specifically sessionAffinity, and the upstream LB hashing.",
    "Stateless doesn't rule out affinity-driven skew. ClientIP affinity concentrates all requests from one source IP onto the same pod. Inspect the source IP distribution in your ingress logs.",
    "Yes. Mesh-internal traffic hitting the Service with ClientIP affinity shows up as a small number of apparent clients with huge request volumes. Consider switching the Service to default round-robin and rely on L7 LB for session stickiness if you need it.",
    "Set externalTrafficPolicy: Local on the Service. That preserves client source IP without the kube-proxy NAT hop, while still load-balancing via the L4 LB. Your current setup trades one problem for another.",
    "Local policy requires a pod on every node serving traffic through that node's NodePort. Nodes without an auth-service pod will refuse connections. Either add pod anti-affinity to spread pods, or drain those nodes from the LB target pool.",
    "Right — Local policy's tradeoff is availability-per-node. The fix is either anti-affinity spreading or limiting the LB targets to nodes known to have pods, via the cloud-provider-specific node label.",
]


def build_w3_history_to_turn(n: int, system_content: str = SYSTEM_DEBUG) -> list[dict]:
    """Conversation seed + turns [0..n-1] with fabricated assistant replies.
    Returns a conversation ready to be tokenized as context for turn ``n``'s
    user message (which is W3_TURNS[n-1]).
    """
    msgs: list[dict] = [{"role": "system", "content": system_content}]
    for i in range(n):
        user_text, _ = W3_TURNS[i]
        msgs.append({"role": "user", "content": user_text})
        if i < n - 1:
            msgs.append({"role": "assistant", "content": _W3_FAKE_ASSISTANT[i]})
    return msgs


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class GenOutput:
    token_ids: list[int]
    text: str
    prompt_len: int


# Conversation -> (input_ids, role_ids). ``role_ids`` may be ``None`` for
# models that don't use role embeddings. Both lists have the same length
# as input_ids when non-None.
TokenizeFn = Callable[[List[Dict[str, Any]], bool], Tuple[List[int], Optional[List[int]]]]


# ---------------------------------------------------------------------------
# Generation adapters
# ---------------------------------------------------------------------------


def _gen_torchllms_once(
    llm,
    prompt_ids: list[int],
    role_ids_list: Optional[list[int]],
    max_new_tokens: int,
    tokenizer,
) -> tuple[GenOutput, float]:
    """One ``llm._generate_single`` call with CUDA sync on both sides for
    wallclock. Does not alter ``llm.prefix_cache`` — caller owns it.
    Wallclock includes prefix-cache lookup/insert, which is honest.
    """
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda:0")
    role_ids = (
        torch.tensor([role_ids_list], dtype=torch.long, device="cuda:0")
        if role_ids_list is not None
        else None
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = llm._generate_single(
        input_ids=input_ids, role_ids=role_ids,
        temperature=0.0, max_new_tokens=max_new_tokens,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    # Decode defensively: the gpt-oss tiktoken object uses .decode(list) with
    # no skip_special_tokens kwarg, while HF tokenizers do. Try both.
    try:
        text = tokenizer.decode(result.token_ids, skip_special_tokens=True)
    except TypeError:
        text = tokenizer.decode(list(result.token_ids))
    return (
        GenOutput(
            token_ids=list(result.token_ids),
            text=text,
            prompt_len=len(prompt_ids),
        ),
        wall,
    )


def _gen_sglang_once(
    engine,
    prompt_ids: list[int],
    max_new_tokens: int,
    tokenizer,
    stop_token_ids: Optional[list[int]] = None,
) -> tuple[GenOutput, float]:
    t0 = time.perf_counter()
    sampling_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    if stop_token_ids:
        sampling_params["stop_token_ids"] = list(stop_token_ids)
    out = engine.generate(
        input_ids=prompt_ids,
        sampling_params=sampling_params,
    )
    wall = time.perf_counter() - t0
    if isinstance(out, list):
        out = out[0]
    ids = list(out["output_ids"])
    try:
        text = tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        text = tokenizer.decode(list(ids))
    return (
        GenOutput(token_ids=ids, text=text, prompt_len=len(prompt_ids)),
        wall,
    )


# ---------------------------------------------------------------------------
# Prefill / decode split via two back-to-back calls per trial
# ---------------------------------------------------------------------------


def _split_prefill_decode(
    fn_once,
    max_new_tokens: int,
    trials: int = 3,
    warmup: int = 1,
) -> tuple[GenOutput, float, float]:
    """Median prefill_s and decode_s. Two calls per trial:

        t_pre  = wallclock for max_new=1       (prefill + 1-token decode)
        t_full = wallclock for max_new=N       (prefill + N-token decode)
        decode_s = median(t_full) - median(t_pre)

    Subtracting medians rather than taking median-of-differences is
    slightly biased but matches the existing validate_qwen3 harness.
    """
    for _ in range(warmup):
        fn_once(1)
        fn_once(max_new_tokens)

    prefill_times: list[float] = []
    total_times: list[float] = []
    last_out: Optional[GenOutput] = None
    for _ in range(trials):
        _, pre = fn_once(1)
        out, tot = fn_once(max_new_tokens)
        prefill_times.append(pre)
        total_times.append(tot)
        last_out = out
    p = statistics.median(prefill_times)
    d = max(statistics.median(total_times) - p, 1e-6)
    assert last_out is not None
    return last_out, p, d


# ---------------------------------------------------------------------------
# Noop-hook wrapper for optional activation-hook overhead measurement
# ---------------------------------------------------------------------------


def _make_counting_noop_hook():
    calls = 0

    def noop_intervention(hidden, ctx):
        nonlocal calls
        calls += 1
        return hidden

    def get_calls() -> int:
        return calls

    return noop_intervention, get_calls


def _split_torchllms_with_optional_noop(
    llm,
    ids: list[int],
    roles: Optional[list[int]],
    max_new: int,
    tokenizer,
    *,
    use_noop_hook: bool,
    trials: int,
    warmup: int,
):
    noop_hook, get_calls = _make_counting_noop_hook()
    llm.prefix_cache = None
    if hasattr(llm, "clear_activation_hooks"):
        llm.clear_activation_hooks()
    if use_noop_hook and hasattr(llm, "set_activation_hooks"):
        llm.set_activation_hooks(noop_hook)
    try:
        out, pre, dec = _split_prefill_decode(
            lambda n: _gen_torchllms_once(llm, ids, roles, n, tokenizer),
            max_new, trials=trials, warmup=warmup,
        )
    finally:
        if hasattr(llm, "clear_activation_hooks"):
            llm.clear_activation_hooks()
    return out, pre, dec, get_calls()


def _gen_torchllms_once_with_optional_noop(
    llm,
    ids: list[int],
    roles: Optional[list[int]],
    max_new: int,
    tokenizer,
    *,
    prefix_cache,
    use_noop_hook: bool,
):
    noop_hook, get_calls = _make_counting_noop_hook()
    llm.prefix_cache = prefix_cache
    if hasattr(llm, "clear_activation_hooks"):
        llm.clear_activation_hooks()
    if use_noop_hook and hasattr(llm, "set_activation_hooks"):
        llm.set_activation_hooks(noop_hook)
    try:
        out, wall = _gen_torchllms_once(llm, ids, roles, max_new, tokenizer)
    finally:
        if hasattr(llm, "clear_activation_hooks"):
            llm.clear_activation_hooks()
    return out, wall, get_calls()


# ---------------------------------------------------------------------------
# Single-workload bench
# ---------------------------------------------------------------------------


def _bench_single(
    llm,
    engine,
    conv: list[dict],
    max_new: int,
    tokenize_fn: TokenizeFn,
    tokenizer,
    label: str,
    *,
    stop_token_ids: Optional[list[int]] = None,
    run_noop_hook: bool = True,
    trials: int = 3,
    warmup: int = 1,
) -> None:
    ids, roles = tokenize_fn(conv, True)

    t_out, t_pre, t_dec, _ = _split_torchllms_with_optional_noop(
        llm, ids, roles, max_new, tokenizer,
        use_noop_hook=False, trials=trials, warmup=warmup,
    )
    if run_noop_hook:
        tn_out, tn_pre, tn_dec, noop_calls = _split_torchllms_with_optional_noop(
            llm, ids, roles, max_new, tokenizer,
            use_noop_hook=True, trials=trials, warmup=warmup,
        )
    else:
        tn_out, tn_pre, tn_dec, noop_calls = None, 0.0, 0.0, 0
    if engine is not None:
        s_out, s_pre, s_dec = _split_prefill_decode(
            lambda n: _gen_sglang_once(engine, ids, n, tokenizer, stop_token_ids),
            max_new, trials=trials, warmup=warmup,
        )
    else:
        s_out, s_pre, s_dec = None, 0.0, 0.0

    def tps_pair(prompt_len, out_tokens, pre_s, dec_s):
        pre_tps = prompt_len / pre_s if pre_s > 0 else 0.0
        dec_tokens = max(out_tokens - 1, 0)  # first token comes from prefill
        dec_tps = dec_tokens / dec_s if dec_s > 0 else 0.0
        return pre_tps, dec_tps, dec_tokens

    t_pre_tps, t_dec_tps, _ = tps_pair(len(ids), len(t_out.token_ids), t_pre, t_dec)

    print(
        f"\n  [{label}] prompt={len(ids)}t  torchllms_out={len(t_out.token_ids)}t"
        + (f"  noop_out={len(tn_out.token_ids)}t" if tn_out is not None else "")
        + (f"  sglang_out={len(s_out.token_ids)}t" if s_out is not None else "")
    )
    print(
        f"    {'engine':<10s}  {'prefill_tps':>11s}  "
        f"{'decode_tps':>11s}  {'total_s':>8s}"
    )
    print(
        f"    {'torchllms':<10s}  {t_pre_tps:>11.1f}  {t_dec_tps:>11.1f}  "
        f"{t_pre + t_dec:>8.2f}"
    )
    if tn_out is not None:
        tn_pre_tps, tn_dec_tps, _ = tps_pair(
            len(ids), len(tn_out.token_ids), tn_pre, tn_dec,
        )
        print(
            f"    {'torch+noop':<10s}  {tn_pre_tps:>11.1f}  "
            f"{tn_dec_tps:>11.1f}  {tn_pre + tn_dec:>8.2f}"
        )
    if s_out is not None:
        s_pre_tps, s_dec_tps, _ = tps_pair(
            len(ids), len(s_out.token_ids), s_pre, s_dec,
        )
        print(
            f"    {'sglang':<10s}  {s_pre_tps:>11.1f}  "
            f"{s_dec_tps:>11.1f}  {s_pre + s_dec:>8.2f}"
        )
        if t_pre_tps > 0 and t_dec_tps > 0:
            print(
                f"    sglang/torchllms speedup: prefill={s_pre_tps/t_pre_tps:.2f}x  "
                f"decode={s_dec_tps/t_dec_tps:.2f}x"
            )
    if tn_out is not None and t_pre > 0 and t_dec > 0:
        noop_pre_slowdown = tn_pre / t_pre
        noop_dec_slowdown = tn_dec / t_dec
        noop_total_slowdown = (tn_pre + tn_dec) / (t_pre + t_dec)
        print(
            f"    noop/torchllms time: prefill={noop_pre_slowdown:.2f}x  "
            f"decode={noop_dec_slowdown:.2f}x  total={noop_total_slowdown:.2f}x  "
            f"hook_calls={noop_calls}"
        )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


@dataclass
class WorkloadResult:
    """Per-workload summary. Times are medians across trials."""
    label: str
    prompt_len: int
    out_tokens: int
    prefill_s: float
    decode_s: float

    @property
    def prefill_tps(self) -> float:
        return self.prompt_len / self.prefill_s if self.prefill_s > 0 else 0.0

    @property
    def decode_tps(self) -> float:
        n = max(self.out_tokens - 1, 0)
        return n / self.decode_s if self.decode_s > 0 else 0.0


@dataclass
class EngineResults:
    """All measurements for one engine across W1/W2/W3."""
    engine_label: str
    w1: Optional[WorkloadResult] = None
    w2: Optional[WorkloadResult] = None
    w3_turns: list[tuple[int, float, int]] = None  # (prompt_len, wall_s, out_tokens)

    def __post_init__(self):
        if self.w3_turns is None:
            self.w3_turns = []

    @property
    def w3_total_s(self) -> float:
        return sum(w for _, w, _ in self.w3_turns)


def bench_engine(
    llm_or_engine,
    tokenize_fn: TokenizeFn,
    tokenizer,
    *,
    engine_label: str,
    kind: str,  # "torchllms" | "sglang"
    prefix_cache=None,
    stop_token_ids: Optional[list[int]] = None,
    w3_max_new: int = W3_MAX_NEW_DEFAULT,
    w3_system_content: str = SYSTEM_DEBUG,
    trials: int = 3,
    warmup: int = 1,
) -> EngineResults:
    """Run W1/W2/W3 on a single engine (either torchllms LLM or sgl.Engine)
    and return structured results. Does NOT print anything — caller can
    combine multiple EngineResults into a unified comparison table.

    Necessary for memory-constrained setups (e.g. gpt-oss on a 32 GB GPU)
    where torchllms + SGLang can't be resident simultaneously — run one,
    tear it down, run the other, then format the table externally.
    """
    assert kind in ("torchllms", "sglang")
    results = EngineResults(engine_label=engine_label)

    def _bench_workload(conv, max_new) -> WorkloadResult:
        ids, roles = tokenize_fn(conv, True)
        if kind == "torchllms":
            # Reset prefix cache between workloads so W2 doesn't get
            # free prefill on W1's prefix.
            llm_or_engine.prefix_cache = None
            out, pre, dec = _split_prefill_decode(
                lambda n: _gen_torchllms_once(llm_or_engine, ids, roles, n, tokenizer),
                max_new, trials=trials, warmup=warmup,
            )
        else:
            out, pre, dec = _split_prefill_decode(
                lambda n: _gen_sglang_once(llm_or_engine, ids, n, tokenizer, stop_token_ids),
                max_new, trials=trials, warmup=warmup,
            )
        return WorkloadResult(
            label="",  # caller fills
            prompt_len=len(ids),
            out_tokens=len(out.token_ids),
            prefill_s=pre, decode_s=dec,
        )

    w1 = _bench_workload(W1_CONV, W1_MAX_NEW); w1.label = "short/short"
    results.w1 = w1
    w2 = _bench_workload(W2_CONV, W2_MAX_NEW); w2.label = "short/long"
    results.w2 = w2

    # W3: multi-turn with prefix-cache reuse (torchllms uses prefix_cache arg,
    # SGLang uses its internal radix cache)
    if kind == "torchllms" and prefix_cache is not None:
        prefix_cache.clear()
        llm_or_engine.prefix_cache = prefix_cache
    for turn_idx in range(len(W3_TURNS)):
        conv = build_w3_history_to_turn(turn_idx + 1, w3_system_content)
        ids, roles = tokenize_fn(conv, True)
        if kind == "torchllms":
            out, wall = _gen_torchllms_once(llm_or_engine, ids, roles, w3_max_new, tokenizer)
        else:
            out, wall = _gen_sglang_once(llm_or_engine, ids, w3_max_new, tokenizer, stop_token_ids)
        results.w3_turns.append((len(ids), wall, len(out.token_ids)))
    if kind == "torchllms":
        llm_or_engine.prefix_cache = None
    return results


def print_combined_throughput(
    torchllms_res: EngineResults,
    sglang_res: Optional[EngineResults],
    *,
    model_label: str,
) -> None:
    """Format torchllms and (optional) SGLang EngineResults as a single
    side-by-side table with speedup ratios."""
    print("\n" + "=" * 72)
    print(f"THROUGHPUT — {model_label}")
    print("=" * 72)

    for w_name, t_w, s_w in [
        ("W1 short/short", torchllms_res.w1, sglang_res.w1 if sglang_res else None),
        ("W2 short/long",  torchllms_res.w2, sglang_res.w2 if sglang_res else None),
    ]:
        if t_w is None:
            continue
        print(f"\n  {w_name}  prompt={t_w.prompt_len}t  torchllms_out={t_w.out_tokens}t"
              + (f"  sglang_out={s_w.out_tokens}t" if s_w else ""))
        print(f"    {'engine':<10s}  {'prefill_tps':>11s}  {'decode_tps':>11s}  {'total_s':>8s}")
        print(f"    {'torchllms':<10s}  {t_w.prefill_tps:>11.1f}  {t_w.decode_tps:>11.1f}  "
              f"{t_w.prefill_s + t_w.decode_s:>8.2f}")
        if s_w is not None:
            print(f"    {'sglang':<10s}  {s_w.prefill_tps:>11.1f}  {s_w.decode_tps:>11.1f}  "
                  f"{s_w.prefill_s + s_w.decode_s:>8.2f}")
            if t_w.prefill_tps > 0 and t_w.decode_tps > 0:
                print(f"    sglang/torchllms speedup: "
                      f"prefill={s_w.prefill_tps/t_w.prefill_tps:.2f}x  "
                      f"decode={s_w.decode_tps/t_w.decode_tps:.2f}x")

    # W3 side-by-side turns
    print("\n  W3 multiturn (agentic, cache engaged)")
    for i in range(max(len(torchllms_res.w3_turns), len(sglang_res.w3_turns) if sglang_res else 0)):
        t_prompt, t_wall, t_out = torchllms_res.w3_turns[i]
        line = f"    turn {i}: prompt={t_prompt}t  torchllms={t_wall*1000:.0f}ms ({t_out}t)"
        if sglang_res and i < len(sglang_res.w3_turns):
            s_prompt, s_wall, s_out = sglang_res.w3_turns[i]
            line += f"  sglang={s_wall*1000:.0f}ms ({s_out}t)"
        print(line)
    if sglang_res:
        t_total = torchllms_res.w3_total_s
        s_total = sglang_res.w3_total_s
        print(f"\n    multiturn total: torchllms={t_total:.2f}s  sglang={s_total:.2f}s"
              + (f"  sglang/torchllms={t_total/s_total:.2f}x" if s_total > 0 else ""))


def run_throughput(
    llm,
    engine,
    tokenize_fn: TokenizeFn,
    tokenizer,
    *,
    prefix_cache=None,
    model_label: str = "torchllms",
    stop_token_ids: Optional[list[int]] = None,
    run_noop_hook: bool = True,
    w3_max_new: int = W3_MAX_NEW_DEFAULT,
    w3_system_content: str = SYSTEM_DEBUG,
    trials: int = 3,
    warmup: int = 1,
    compile_decode: bool = False,
) -> None:
    # Optional: enable decode-compile before benching. First call post-compile
    # triggers the Inductor warmup (cudagraph capture under reduce-overhead),
    # which is charged against "engine setup" rather than per-workload wall.
    compile_warmup_s = 0.0
    if compile_decode:
        if not hasattr(llm, "enable_decode_compile"):
            print("  [warn] --compile-decode requested but llm has no "
                  "enable_decode_compile; running eager")
        else:
            print("  Enabling decode-compile + warming up (one decode call) ...")
            llm.enable_decode_compile()
            # Warm up via a trivial short-prompt single-step decode through
            # the W1 conversation so the first timed call doesn't pay
            # compile cost. Measure that warmup separately.
            w1_ids, w1_roles = tokenize_fn(W1_CONV, True)
            llm.prefix_cache = None
            t0 = time.perf_counter()
            _gen_torchllms_once(llm, w1_ids, w1_roles, 4, tokenizer)
            compile_warmup_s = time.perf_counter() - t0
            print(f"    decode-compile warmup: {compile_warmup_s:.2f}s")
    """Run W1/W2/W3 and print a unified table.

    ``prefix_cache``: a ``RadixKVCache`` that carries across W3's
    multi-turn dialogue. For W1/W2 the cache is zeroed by
    ``_split_torchllms_with_optional_noop``. If None, W3 still runs but
    without prefix-cache benefit (degenerate but measurable).

    ``stop_token_ids``: passed to SGLang sampling_params (torchllms's own
    stop handling comes from llm.eos_ids, set at llm construction).

    ``run_noop_hook``: if False, skip the torch+noop column (useful for
    gpt-oss which may not expose activation_hooks).
    """
    print("\n" + "=" * 72)
    print(f"THROUGHPUT — {model_label}"
          + ("  [decode-compile ON]" if compile_decode else "  [eager]"))
    print("=" * 72)
    print(
        f"  Median of {trials} trials, {warmup} warm-up. Prefill measured via "
        f"max_new=1, decode inferred from full_call - prefill_call."
    )
    if compile_warmup_s > 0:
        print(f"  decode-compile warmup: {compile_warmup_s:.2f}s (one-time, "
              f"not counted in per-workload timings)")

    print("\n  -- W1: short prompt + short completion --")
    _bench_single(
        llm, engine, W1_CONV, W1_MAX_NEW, tokenize_fn, tokenizer, "short/short",
        stop_token_ids=stop_token_ids, run_noop_hook=run_noop_hook,
        trials=trials, warmup=warmup,
    )

    print("\n  -- W2: short prompt + long completion --")
    _bench_single(
        llm, engine, W2_CONV, W2_MAX_NEW, tokenize_fn, tokenizer, "short/long",
        stop_token_ids=stop_token_ids, run_noop_hook=run_noop_hook,
        trials=trials, warmup=warmup,
    )

    print("\n  -- W3: multiturn long/short (agentic, cache engaged) --")
    # W3 uses the prefix cache across turns — so fresh cache per run, but
    # not between turns.
    if prefix_cache is not None:
        prefix_cache.clear()
    # Noop-hook W3 uses its own prefix cache so the two runs don't share
    # state; only test if noop hook requested.
    if run_noop_hook and prefix_cache is not None:
        from torchllms.inference.prefix_cache import RadixKVCache
        noop_prefix_cache = RadixKVCache(max_bytes=4 * 1024**3)
    else:
        noop_prefix_cache = None

    torchllms_turn_times: list[tuple[float, int]] = []
    noop_turn_times: list[tuple[float, int]] = []
    sglang_turn_times: list[tuple[float, int]] = []
    noop_hook_calls = 0
    for turn_idx in range(len(W3_TURNS)):
        conv = build_w3_history_to_turn(turn_idx + 1, w3_system_content)
        ids, roles = tokenize_fn(conv, True)

        t_out, t_wall, _ = _gen_torchllms_once_with_optional_noop(
            llm, ids, roles, w3_max_new, tokenizer,
            prefix_cache=prefix_cache, use_noop_hook=False,
        )
        if run_noop_hook:
            tn_out, tn_wall, tn_calls = _gen_torchllms_once_with_optional_noop(
                llm, ids, roles, w3_max_new, tokenizer,
                prefix_cache=noop_prefix_cache, use_noop_hook=True,
            )
        else:
            tn_out, tn_wall, tn_calls = None, 0.0, 0
        if engine is not None:
            s_out, s_wall = _gen_sglang_once(
                engine, ids, w3_max_new, tokenizer, stop_token_ids,
            )
        else:
            s_out, s_wall = None, 0.0

        torchllms_turn_times.append((t_wall, len(t_out.token_ids)))
        if tn_out is not None:
            noop_turn_times.append((tn_wall, len(tn_out.token_ids)))
            noop_hook_calls += tn_calls
        if s_out is not None:
            sglang_turn_times.append((s_wall, len(s_out.token_ids)))

        parts = [
            f"turn {turn_idx}:",
            f"prompt={len(ids)}t",
            f"torchllms={t_wall*1000:.0f}ms ({len(t_out.token_ids)}t)",
        ]
        if tn_out is not None:
            parts.append(f"noop={tn_wall*1000:.0f}ms ({len(tn_out.token_ids)}t)")
        if s_out is not None:
            parts.append(f"sglang={s_wall*1000:.0f}ms ({len(s_out.token_ids)}t)")
        print("    " + "  ".join(parts))

    t_total = sum(w for w, _ in torchllms_turn_times)
    tn_total = sum(w for w, _ in noop_turn_times) if noop_turn_times else 0.0
    s_total = sum(w for w, _ in sglang_turn_times) if sglang_turn_times else 0.0

    line = f"\n    multiturn total: torchllms={t_total:.2f}s"
    if s_total > 0:
        line += f"  sglang={s_total:.2f}s  sglang/torchllms={t_total/s_total:.2f}x"
    print(line)
    if tn_total > 0 and t_total > 0:
        print(
            f"    noop/torchllms time={tn_total/t_total:.2f}x  "
            f"hook_calls={noop_hook_calls}"
        )
