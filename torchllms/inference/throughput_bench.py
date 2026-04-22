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


def print_matrix_table(
    results: list[tuple[str, EngineResults]],
    *,
    model_label: str,
) -> None:
    """Format the matrix of EngineResults as a single table.

    ``results`` is a list of ``(cell_label, EngineResults)`` tuples —
    typically ``[torch-eager, torch-compile, sglang-graph-on,
    sglang-graph-off]`` but any subset works.
    """
    print("\n" + "=" * 84)
    print(f"THROUGHPUT MATRIX — {model_label}")
    print("=" * 84)
    hdr = (
        f"  {'cell':<24} "
        f"{'W1 prefill tps':>15} {'W1 decode tps':>14} "
        f"{'W2 prefill tps':>15} {'W2 decode tps':>14} "
        f"{'W3 total s':>11}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, r in results:
        w1p = r.w1.prefill_tps if r.w1 else 0.0
        w1d = r.w1.decode_tps if r.w1 else 0.0
        w2p = r.w2.prefill_tps if r.w2 else 0.0
        w2d = r.w2.decode_tps if r.w2 else 0.0
        print(
            f"  {label:<24} "
            f"{w1p:>15.1f} {w1d:>14.1f} "
            f"{w2p:>15.1f} {w2d:>14.1f} "
            f"{r.w3_total_s:>11.2f}"
        )


# ---------------------------------------------------------------------------
# Matrix runner (main entry point)
# ---------------------------------------------------------------------------
#
# ``python -m torchllms.inference.throughput_bench --model {qwen3,gpt-oss}``
# runs four cells in one process:
#
#   1. torchllms eager
#   2. torchllms + decode-compile (Inductor, mode="default")
#   3. SGLang with cudagraph enabled
#   4. SGLang with cudagraph disabled
#
# Torchllms cells share one LLM instance (eager → enable_decode_compile);
# SGLang cells load + shutdown a fresh sgl.Engine each time. Memory is
# freed between the torch and sglang phases via ``del llm; gc.collect();
# torch.cuda.empty_cache()`` so gpt-oss's ~10 GB weights + SGLang's
# mem_fraction_static budget co-exist on a 32 GB GPU.


_MODELS = {
    "qwen3": {
        "hf_path": "/root/qwen3-4b",
        "ckpt": "/root/qwen3-4b/consolidated.00.pth",
        "template_config": "qwen3_chatml_nothink.yaml",
        "max_seq_len": 8192,
        "stop_ids": [151645, 151643],
        "precision": "bfloat16",
    },
    "gpt-oss": {
        "hf_path": "/root/gpt-oss-20b",
        "ckpt": "/root/gpt-oss-20b/original",
        "template_config": None,  # Harmony path
        "max_seq_len": 4096,
        "stop_ids": [200012, 200002, 199999],
        "precision": "bfloat16",
    },
}


def _make_tokenize_fn(model: str):
    cfg = _MODELS[model]
    if model == "qwen3":
        from transformers import AutoTokenizer
        from pathlib import Path
        import yaml
        from torchllms.messages.tokenization import (
            tokenize_conversation, TemplateConfig,
        )
        tpl_path = Path(__file__).resolve().parents[1] / "messages" / "configs" / cfg["template_config"]
        template = TemplateConfig(**yaml.safe_load(open(tpl_path)))
        hf_tok = AutoTokenizer.from_pretrained(cfg["hf_path"], trust_remote_code=True)

        def tokenize_fn(conv, add_gen):
            return tokenize_conversation(conv, hf_tok, template, add_generation_prompt=add_gen)
        return tokenize_fn
    else:  # gpt-oss — Harmony
        from torchllms.messages.tokenization_harmony import tokenize_harmony_conversation

        def tokenize_fn(conv, add_gen):
            return tokenize_harmony_conversation(conv, add_generation_prompt=add_gen)
        return tokenize_fn


def _run_torch_cells(model: str) -> list[tuple[str, EngineResults]]:
    """Load torchllms, bench eager + compile, tear down. Returns the two
    EngineResults."""
    from torchllms.inference.llm import LLM

    cfg = _MODELS[model]
    tokenize_fn = _make_tokenize_fn(model)

    kwargs = dict(
        ckpt_paths=[cfg["ckpt"]],
        max_len=cfg["max_seq_len"],
        device="cuda:0",
        precision=cfg["precision"],
    )
    if cfg["template_config"]:
        kwargs["template_config"] = cfg["template_config"]
    else:
        # gpt-oss uses Harmony; pass explicit eos + mxfp4 flag.
        kwargs["eos_ids"] = cfg["stop_ids"]
        kwargs["model_kwargs"] = {
            "max_seq_len": cfg["max_seq_len"], "mxfp4": True,
        }

    print("  Loading torchllms ...", flush=True)
    llm = LLM(**kwargs)
    llm._ensure_cache()
    llm.enable_prefix_cache()

    cells: list[tuple[str, EngineResults]] = []

    # Cell 1: eager.
    print("\n  [1/4] torchllms eager", flush=True)
    r = bench_engine(
        llm, tokenize_fn, llm.tokenizer,
        engine_label="torch-eager", kind="torchllms",
        prefix_cache=llm.prefix_cache, stop_token_ids=cfg["stop_ids"],
        w3_max_new=W3_MAX_NEW_DEFAULT, trials=3, warmup=1,
    )
    cells.append(("torch-eager", r))

    # Cell 2: prefill+decode-compile.
    # gpt-oss MoE dispatches through ``torchllms::gptoss_moe_delta``
    # (single Dynamo-opaque custom op per layer) so compile works for
    # both model families — see ``GptOSSMoE.forward`` and
    # ``_gptoss_moe_delta`` in ``models/networks_gptoss.py``.
    print("\n  [2/4] torchllms + prefill+decode-compile", flush=True)
    # Prefill: Inductor-only + dynamic=True — one compile covers all
    # prompt shapes (no cudagraph; see ``enable_prefill_compile``
    # docstring for why cudagraph-prefill is deferred).
    # Decode: reduce-overhead — Inductor + cudagraph capture on top.
    # With PagedKVPool's stable buffers + use_cuda_graph=True decode
    # wrapper, decode replay amortizes the per-step kernel launches.
    llm.enable_prefill_compile(mode="default")
    llm.enable_decode_compile(mode="reduce-overhead")
    # Warmup: run a representative pass across W1/W2/W3 prompt shapes so
    # Dynamo has compiled symbolic-shape kernels for the range and the
    # decode cudagraph has captured B=1 replay state across the shape
    # set. A single-shape 10-token warmup was insufficient — the W3
    # cell was landing ~2× slower than apples-to-apples because the
    # first few W3 turns paid guard-miss + compile costs on unseen
    # shapes. With a multi-shape warmup that's amortized before any
    # timed call runs.
    try:
        if llm.prefix_cache is not None:
            llm.prefix_cache.clear()
        for conv in (W1_CONV, W2_CONV):
            w_ids, w_roles = tokenize_fn(conv, True)
            w_in = torch.tensor(w_ids, device=llm.device, dtype=torch.long)
            w_r = torch.tensor(w_roles, device=llm.device, dtype=torch.long) if w_roles else None
            llm._generate_single(w_in, role_ids=w_r, temperature=0.0, max_new_tokens=16)
        # W3 shapes: warm each turn's prompt once.
        for turn_idx in range(len(W3_TURNS)):
            conv = build_w3_history_to_turn(turn_idx + 1)
            w_ids, w_roles = tokenize_fn(conv, True)
            w_in = torch.tensor(w_ids, device=llm.device, dtype=torch.long)
            w_r = torch.tensor(w_roles, device=llm.device, dtype=torch.long) if w_roles else None
            llm._generate_single(w_in, role_ids=w_r, temperature=0.0, max_new_tokens=8)
    except Exception as e:
        print(f"    [warn] warmup skipped: {e}")
    r = bench_engine(
        llm, tokenize_fn, llm.tokenizer,
        engine_label="torch-compile", kind="torchllms",
        prefix_cache=llm.prefix_cache, stop_token_ids=cfg["stop_ids"],
        w3_max_new=W3_MAX_NEW_DEFAULT, trials=3, warmup=1,
    )
    cells.append(("torch-compile", r))

    # Tear down torchllms completely before SGLang loads.
    print("\n  Tearing down torchllms ...", flush=True)
    llm.disable_decode_compile()
    llm.disable_prefill_compile()
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # Clear the flashinfer wrapper cache (stateful; would hold stale
    # ``plan`` buffers across teardown).
    try:
        from torchllms.models import flashinfer_attention as _fa
        _fa._WRAPPER_CACHE.clear()
        _fa._DECODE_WRAPPER_CACHE.clear()
    except ImportError:
        pass
    return cells


def _run_sglang_cell(model: str, *, cuda_graph: bool) -> EngineResults:
    """Load one sgl.Engine, bench, shutdown."""
    import sglang as sgl
    import gc as _gc

    cfg = _MODELS[model]
    tokenize_fn = _make_tokenize_fn(model)
    label = f"sglang-cg-{'on' if cuda_graph else 'off'}"
    print(f"\n  [{{}}/4] {label}", flush=True)

    engine = sgl.Engine(
        model_path=cfg["hf_path"],
        dtype=cfg["precision"],
        mem_fraction_static=0.80,
        context_length=cfg["max_seq_len"],
        disable_cuda_graph=(not cuda_graph),
    )

    class _ShimTok:
        def decode(self, ids, skip_special_tokens=True):
            return ""

    try:
        r = bench_engine(
            engine, tokenize_fn, _ShimTok(),
            engine_label=label, kind="sglang",
            prefix_cache=None, stop_token_ids=cfg["stop_ids"],
            w3_max_new=W3_MAX_NEW_DEFAULT, trials=3, warmup=1,
        )
    finally:
        sd = getattr(engine, "shutdown", None)
        if sd is not None:
            sd()
        del engine
        _gc.collect()
        torch.cuda.empty_cache()
    return r


def main() -> int:
    import argparse
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ap = argparse.ArgumentParser(
        description="Throughput matrix: {torch eager, torch compile, "
                    "sglang cudagraph-on, sglang cudagraph-off} × "
                    "{W1 short/short, W2 short/long, W3 multiturn}."
    )
    ap.add_argument(
        "--model", choices=list(_MODELS.keys()), required=True,
        help="Which model to benchmark. Qwen3-4B uses ChatML templates "
             "and 8192 context; gpt-oss-20b uses Harmony templates and "
             "4096 context.",
    )
    ap.add_argument(
        "--no-sglang", action="store_true",
        help="Skip the two SGLang cells (torchllms eager + compile only).",
    )
    args = ap.parse_args()

    print(f"\n==== Throughput matrix: {args.model} ====")

    cells = _run_torch_cells(args.model)

    if not args.no_sglang:
        for cg in (True, False):
            try:
                r = _run_sglang_cell(args.model, cuda_graph=cg)
                cells.append((f"sglang-cg-{'on' if cg else 'off'}", r))
            except Exception as e:
                print(f"    [warn] SGLang cg={cg} failed: {e}")

    print_matrix_table(cells, model_label=args.model)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
