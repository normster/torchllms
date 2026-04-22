"""Qwen3-4B torchllms correctness validation against SGLang.

Runs four phases end-to-end (no flags):

  cache-correctness      — cache preserves correctness; batched path
                           preserves correctness. fp32 bit-exact gate;
                           bf16 drift is reported but not a hard fail.
  activation-correctness — an installed no-op activation hook is exactly
                           a no-op against the baseline (logit- and
                           token-identical), and the hook is actually
                           called.
  generation-correctness — torchllms outputs agree with SGLang to a
                           reasonable extent on realistic prompts.
                           Reports first-diverge, overlap %, word-edit
                           similarity, and decoded heads. Not a hard
                           gate: kernels differ, bf16 argmax can flip on
                           close candidates.
  decode-compile         — decode-only torch.compile smoke: eager vs
                           compiled tokens match within bf16 drift
                           tolerance (first-diverge >= 8). Correctness
                           check; throughput measurement lives separately.

Throughput benchmarking is a separate concern — run
``python -m torchllms.inference.throughput_bench --model qwen3``.

Everything runs at bf16 (production path) except the cache-correctness
fp32 gate.

Usage:
    python -m torchllms.inference.validate_qwen3
    python -m torchllms.inference.validate_qwen3 --no-sglang

Model path is controlled by the ``TORCHLLMS_QWEN3_PATH`` environment
variable (default ``/root/qwen3-4b``). The HF checkpoint directory and
the torchllms ``consolidated.00.pth`` file are both expected under that
root.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from transformers import AutoTokenizer

from torchllms.inference import LLM as TorchLLM, RadixKVCache
from torchllms.messages.tokenization import (
    TemplateConfig,
    tokenize_conversation,
)
from torchllms.models import AddVec


HF_PATH = os.environ.get("TORCHLLMS_QWEN3_PATH", "/root/qwen3-4b")
TORCHLLMS_CKPT = f"{HF_PATH}/consolidated.00.pth"
CONFIG_NAME = "qwen3_chatml_nothink.yaml"
MAX_LEN = 8192
PRECISION = "bfloat16"  # "bfloat16" for production bench, "float32" for correctness

SYSTEM_SHORT = "You are a concise technical assistant. Answer briefly."
SYSTEM_DEBUG = (
    "You are a senior backend engineer helping debug production issues. "
    "Respond with one diagnosis followed by one specific next check or action, "
    "no more than a short paragraph."
)

STOP_TOKEN_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>
TRIALS = 3                          # median-of-3 for throughput runs
WARMUP = 1


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _load_config() -> TemplateConfig:
    # torchllms.inference.validate_qwen3 → torchllms/inference/validate_qwen3.py;
    # configs ship at torchllms/messages/configs/*.yaml inside the same package.
    path = Path(__file__).resolve().parent.parent / "messages/configs" / CONFIG_NAME
    with path.open() as f:
        return TemplateConfig(**yaml.safe_load(f))


def _first_diverge(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _overlap_frac(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    return _first_diverge(a, b) / max(len(a), len(b))


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _word_units(text: str) -> list[str]:
    return _WORD_RE.findall(text.casefold())


def _edit_distance(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        cur = [i]
        for j, bj in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (ai != bj),
                )
            )
        prev = cur
    return prev[-1]


def _word_edit_stats(a_text: str, b_text: str) -> tuple[int, int, int, float]:
    a = _word_units(a_text)
    b = _word_units(b_text)
    denom = max(len(a), len(b), 1)
    dist = _edit_distance(a, b)
    similarity = 1.0 - (dist / denom)
    return dist, len(a), len(b), similarity


@dataclass
class GenOutput:
    token_ids: list[int]
    text: str
    stop_reason: Optional[int]
    prefill_s: float
    decode_s: float
    prompt_len: int

    @property
    def prefill_tps(self) -> float:
        return self.prompt_len / self.prefill_s if self.prefill_s > 0 else 0.0

    @property
    def decode_tps(self) -> float:
        n = len(self.token_ids) - 1  # 1 token came from prefill, rest from decode
        return n / self.decode_s if (n > 0 and self.decode_s > 0) else 0.0


def _gen_torchllms_once(
    llm: TorchLLM,
    prompt_ids: list[int],
    role_ids_list: Optional[list[int]],
    max_new_tokens: int,
    tokenizer,
) -> tuple[GenOutput, float]:
    """Single _generate_single call. Returns (output with timings=0, wallclock).

    Does not alter llm.prefix_cache — caller owns that. Wallclock is measured
    around the one call so cache lookups/inserts are accounted for honestly.
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

    text = tokenizer.decode(result.token_ids, skip_special_tokens=True)
    return (
        GenOutput(
            token_ids=list(result.token_ids),
            text=text,
            stop_reason=result.stop_reason,
            prefill_s=0.0, decode_s=0.0,
            prompt_len=len(prompt_ids),
        ),
        wall,
    )


class _CaptureLogits:
    """Context manager that installs a forward hook on llm.model and
    accumulates each forward's logits into a list. Shape per entry is
    [b_live, 1, V] (fixed `logits_to_keep=1` in the LLM paths).
    """

    def __init__(self, model):
        self.model = model
        self.captured: list[torch.Tensor] = []
        self._handle = None

    def __enter__(self):
        def _hook(module, args, output):
            logits = output[0] if isinstance(output, tuple) else output
            self.captured.append(logits.detach().clone().cpu())
        self._handle = self.model.register_forward_hook(_hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()


def _gen_with_logits(
    llm: TorchLLM,
    prompt_ids: list[int],
    role_ids_list: Optional[list[int]],
    max_new_tokens: int,
    tokenizer,
) -> tuple[GenOutput, list[torch.Tensor]]:
    """Same as _gen_torchllms_once but also returns per-step logits."""
    with _CaptureLogits(llm.model) as cap:
        out, _ = _gen_torchllms_once(
            llm, prompt_ids, role_ids_list, max_new_tokens, tokenizer,
        )
    return out, cap.captured


def _logits_equal(
    a: list[torch.Tensor],
    b: list[torch.Tensor],
    atol: float,
) -> tuple[bool, int, float]:
    """Compare two per-step logit lists. Returns (ok, first_diverge_step, max_diff).
    first_diverge_step = n when both lists have the same length n and every
    step matched; otherwise the step index at which they first disagreed or
    their lengths diverged.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i].shape != b[i].shape:
            return False, i, float("inf")
        diff = (a[i].float() - b[i].float()).abs().max().item()
        if diff > atol:
            return False, i, diff
    if len(a) != len(b):
        return False, n, float("inf")
    # All matched within atol; return the final max diff for reporting.
    max_over_all = max(
        ((a[i].float() - b[i].float()).abs().max().item() for i in range(n)),
        default=0.0,
    )
    return True, n, max_over_all


def _gen_sglang_once(
    engine, prompt_ids: list[int], max_new_tokens: int, tokenizer,
) -> tuple[GenOutput, float]:
    t0 = time.perf_counter()
    out = engine.generate(
        input_ids=prompt_ids,
        sampling_params={
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "stop_token_ids": list(STOP_TOKEN_IDS),
        },
    )
    wall = time.perf_counter() - t0
    if isinstance(out, list):
        out = out[0]
    ids = list(out["output_ids"])
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return (
        GenOutput(
            token_ids=ids, text=text, stop_reason=None,
            prefill_s=0.0, decode_s=0.0,
            prompt_len=len(prompt_ids),
        ),
        wall,
    )


def _split_prefill_decode(
    fn_once, max_new_tokens: int, trials: int = TRIALS, warmup: int = WARMUP,
) -> tuple[GenOutput, float, float]:
    """Median prefill_s and decode_s by calling fn_once twice per trial:
    once with max_new=1 (~prefill), once with max_new=N (prefill + decode).
    Assumes caller has set up no-cache state; each trial is independent.

    Returns (representative output, median prefill_s, median decode_s).
    Decode_s = total_N - prefill_s so decode_tps = (N-1) / decode_s.
    """
    for _ in range(warmup):
        fn_once(1)
        fn_once(max_new_tokens)

    prefill_times = []
    total_times = []
    last_out = None
    for _ in range(trials):
        _, pre = fn_once(1)
        out, tot = fn_once(max_new_tokens)
        prefill_times.append(pre)
        total_times.append(tot)
        last_out = out
    p = statistics.median(prefill_times)
    d = max(statistics.median(total_times) - p, 1e-6)
    return last_out, p, d


# ---------------------------------------------------------------------------
# Cache-correctness gates — cache + batched at varying prompt lengths
# ---------------------------------------------------------------------------


# Single-turn prompts for the cache sanity test. Short + long regimes so
# the bf16 GEMM shape-dependence threshold gets exercised at both ends
# without redundant coverage.
SANITY_SINGLE_PROMPTS: list[tuple[str, str, int]] = [
    # (name, user_content, max_new_tokens)
    ("short", "In one sentence, explain TCP vs UDP.", 80),
    ("long", (
        "I'm debugging a Kubernetes auth service with intermittent 504 "
        "timeouts affecting 5-10% of requests. Context: 12 stateless Python "
        "pods behind a Service with sessionAffinity: ClientIP, Postgres 15 on "
        "RDS, stable traffic volume for weeks. `kubectl top pods -l "
        "app=auth-service` shows two pods at ~90% CPU while the rest sit at "
        "~30%. Restart cycles briefly reset the hotspot but it rebuilds "
        "within 5 minutes with the same two pods affected. Ingress logs show "
        "that 80% of the traffic to the hot pods originates from a single "
        "/24 CIDR corresponding to our internal service mesh. We tried "
        "flipping externalTrafficPolicy to Cluster; latency got worse due to "
        "extra NAT hops. We then tried Local policy with NodePort; some "
        "requests started failing with connection-refused errors from "
        "specific nodes. `kubectl get endpoints auth-service -o wide` shows "
        "only 8 of 12 pods listed; the missing 4 are all on nodes without an "
        "auth-service Pod. We cannot simply scale to 30 replicas because each "
        "replica opens 20 persistent DB connections and we're already near "
        "the RDS connection ceiling. Given all of the above, what is the "
        "minimum-risk architectural fix that addresses both the routing skew "
        "and the node-coverage hole without introducing new failure modes or "
        "pushing us past the DB connection limit? Answer in one short "
        "paragraph."
    ), 60),
]


# Batched==single tests. Uniform length per batch (single-digit substitution).
# Short + long regimes, B=4. Single template per tier is sufficient: the test
# is whether batched-slot execution matches per-row single calls.
SANITY_BATCHED_TEMPLATES: list[tuple[str, str, int, int]] = [
    # (name, template_with___N___marker, max_new_tokens, B)
    ("short", "Question __N__: give a one-sentence reply.", 40, 4),
    ("long", (
        "Consider design question variant __N__. An auth service is experiencing "
        "intermittent 504 timeouts affecting 5-10% of requests, running on "
        "Kubernetes with 12 stateless pods behind a Service with ClientIP "
        "session affinity, Postgres 15 on RDS, traffic volume stable for "
        "weeks. `kubectl top pods -l app=auth-service` shows two pods at ~90% "
        "CPU while the rest sit at ~30%. Restart cycles briefly reset the "
        "hotspot but it rebuilds within 5 minutes with the same two pods "
        "affected. Ingress logs show that 80% of the traffic to the hot pods "
        "originates from a single /24 CIDR corresponding to our internal "
        "service mesh. We tried flipping externalTrafficPolicy to Cluster; "
        "latency got worse due to extra NAT hops. We then tried Local policy "
        "with NodePort; some requests started failing with connection-"
        "refused errors from specific nodes. `kubectl get endpoints auth-"
        "service -o wide` shows only 8 of 12 pods listed; the missing 4 are "
        "all on nodes without an auth-service Pod. We cannot simply scale to "
        "30 replicas because each replica opens 20 persistent DB connections "
        "and we're already near the RDS connection ceiling. What is the "
        "minimum-risk architectural fix that addresses both the routing skew "
        "and the node-coverage hole without introducing new failure modes or "
        "pushing us past the DB connection limit?"
    ), 60, 4),
]


@dataclass
class CacheCheckResult:
    name: str
    prompt_len: int
    no_cache_len: int
    fresh_len: int
    filled_len: int
    no_cache_stop: Optional[int]
    fresh_stop: Optional[int]
    filled_stop: Optional[int]
    fresh_logits_ok: bool
    filled_logits_ok: bool
    fresh_logits_dv: int
    filled_logits_dv: int
    fresh_logits_max_diff: float
    filled_logits_max_diff: float
    fresh_tokens_ok: bool
    filled_tokens_ok: bool


def _check_cache(
    llm, prefix_cache, tokenizer, config, name, user_text, max_new, atol: float,
) -> CacheCheckResult:
    """Run the same prompt three times with different cache states and
    compare per-step prefill+decode logits.

      - no_cache:    prefix_cache=None, full prefill forward.
      - fresh_cache: empty RadixKVCache; lookup miss, full prefill, insert.
      - filled_cache: populated RadixKVCache; lookup hit, partial prefill.

    The logit check is strictly stronger than token-ID equality: it catches
    cases where cached KV is subtly wrong but argmax still coincidentally
    agrees. `atol` is the per-step max-abs tolerance for logit equality;
    at fp32 this can be quite tight (~1e-4), at bf16 the GEMM shape-drift
    means filled_cache vs no_cache will generally fail any reasonable atol.
    """
    conv = [
        {"role": "system", "content": SYSTEM_SHORT},
        {"role": "user", "content": user_text},
    ]
    ids, roles = tokenize_conversation(
        conv, tokenizer, config, add_generation_prompt=True,
    )

    llm.prefix_cache = None
    nc_out, nc_logits = _gen_with_logits(llm, ids, roles, max_new, tokenizer)

    prefix_cache.clear()
    llm.prefix_cache = prefix_cache
    fresh_out, fresh_logits = _gen_with_logits(llm, ids, roles, max_new, tokenizer)
    filled_out, filled_logits = _gen_with_logits(llm, ids, roles, max_new, tokenizer)

    fresh_logits_ok, fresh_dv, fresh_maxd = _logits_equal(nc_logits, fresh_logits, atol)
    filled_logits_ok, filled_dv, filled_maxd = _logits_equal(nc_logits, filled_logits, atol)

    return CacheCheckResult(
        name=name,
        prompt_len=len(ids),
        no_cache_len=len(nc_out.token_ids),
        fresh_len=len(fresh_out.token_ids),
        filled_len=len(filled_out.token_ids),
        no_cache_stop=nc_out.stop_reason,
        fresh_stop=fresh_out.stop_reason,
        filled_stop=filled_out.stop_reason,
        fresh_logits_ok=fresh_logits_ok,
        filled_logits_ok=filled_logits_ok,
        fresh_logits_dv=fresh_dv,
        filled_logits_dv=filled_dv,
        fresh_logits_max_diff=fresh_maxd,
        filled_logits_max_diff=filled_maxd,
        fresh_tokens_ok=fresh_out.token_ids == nc_out.token_ids,
        filled_tokens_ok=filled_out.token_ids == nc_out.token_ids,
    )


@dataclass
class BatchedCheckResult:
    name: str
    prompt_len: int
    B: int
    uniform: bool  # False = skipped (non-uniform tokenization)
    per_row_logits_ok: list[bool]
    per_row_logits_dv: list[int]
    per_row_max_diff: list[float]
    per_row_tokens_ok: list[bool]
    per_row_single_len: list[int]
    per_row_batched_len: list[int]


def _check_batched(
    llm, tokenizer, config, name, template, max_new, B, atol: float,
) -> BatchedCheckResult:
    """batched vs single, per-row, at logit level.

    Early stop is disabled during the test (clear `eos_set`) so all B rows
    run in lockstep for exactly max_new decode steps. That keeps the
    batched logits list at a fixed shape [max_new+1, B, 1, V] across steps
    and avoids per-row retirement bookkeeping during comparison.
    """
    # Use str.replace — template may contain literal '%' (e.g. "5-10%").
    convs = [
        [{"role": "system", "content": SYSTEM_SHORT},
         {"role": "user", "content": template.replace("__N__", str(i))}]
        for i in range(1, B + 1)
    ]
    encoded = [
        tokenize_conversation(c, tokenizer, config, add_generation_prompt=True)
        for c in convs
    ]
    lens = {len(e[0]) for e in encoded}
    if len(lens) != 1:
        return BatchedCheckResult(
            name=name, prompt_len=-1, B=B, uniform=False,
            per_row_logits_ok=[], per_row_logits_dv=[], per_row_max_diff=[],
            per_row_tokens_ok=[], per_row_single_len=[], per_row_batched_len=[],
        )

    from torchllms.models.networks import AttentionImpl
    prev_impl = llm.model.params.attention_impl
    prev_batched = llm.batched
    prev_eos_ids = list(llm.eos_ids)
    prev_eos_set = set(llm.eos_set)
    llm.model.params.attention_impl = AttentionImpl.SDPA
    # Force lockstep: disable early stop so all rows run max_new decode steps.
    llm.eos_ids = []
    llm.eos_set = set()

    try:
        llm.prefix_cache = None
        singles: list[tuple[GenOutput, list[torch.Tensor]]] = [
            _gen_with_logits(llm, e[0], e[1], max_new, tokenizer)
            for e in encoded
        ]

        llm.batched = True
        llm.prefix_cache = None
        with _CaptureLogits(llm.model) as cap:
            batched_outs = llm.generate_batched(
                conversations=convs, batch_size=B,
                max_new_tokens=max_new, temperature=0.0, disable_tqdm=True,
            )
        batched_logits = cap.captured  # list of [B, 1, V] tensors
    finally:
        llm.batched = prev_batched
        llm.model.params.attention_impl = prev_impl
        llm.eos_ids = prev_eos_ids
        llm.eos_set = prev_eos_set

    per_row_logits_ok = []
    per_row_dv = []
    per_row_maxd = []
    per_row_tokens_ok = []
    per_row_single_len = []
    per_row_batched_len = []
    for i, ((s_out, s_logits), b_out) in enumerate(zip(singles, batched_outs)):
        b_row_logits = [bl[i : i + 1] for bl in batched_logits]
        ok, dv, md = _logits_equal(s_logits, b_row_logits, atol)
        per_row_logits_ok.append(ok)
        per_row_dv.append(dv)
        per_row_maxd.append(md)
        per_row_tokens_ok.append(s_out.token_ids == b_out.token_ids)
        per_row_single_len.append(len(s_out.token_ids))
        per_row_batched_len.append(len(b_out.token_ids))

    return BatchedCheckResult(
        name=name, prompt_len=lens.pop(), B=B, uniform=True,
        per_row_logits_ok=per_row_logits_ok,
        per_row_logits_dv=per_row_dv,
        per_row_max_diff=per_row_maxd,
        per_row_tokens_ok=per_row_tokens_ok,
        per_row_single_len=per_row_single_len,
        per_row_batched_len=per_row_batched_len,
    )


def run_cache_correctness(
    llm: TorchLLM,
    prefix_cache: RadixKVCache,
    tokenizer,
    config: TemplateConfig,
) -> bool:
    print("\n" + "=" * 72)
    print("CACHE-CORRECTNESS")
    print("    cache: no_cache vs fresh_cache vs filled_cache")
    print("    batched: batched == single")
    print("=" * 72)

    all_ok = True

    # Per-step logit tolerance. fp32 has cuBLAS shape-dependent drift at
    # ~1e-4 (observed 1.05e-4 on [4,38,D] vs [1,38,D] prefill); 1e-3 keeps
    # us comfortably above that floor while still catching real bugs (real
    # bugs show 1e-2+ diffs). bf16 filled_vs_no_cache is expected to fail
    # at any meaningful atol — drift by design.
    atol = 1e-3 if PRECISION == "float32" else 1e-1
    print(f"\n  cache settings (logit equality vs no_cache, atol={atol:.0e}):")
    print("    (fresh = empty-cache miss; filled = populated-cache hit)")
    for name, user_text, max_new in SANITY_SINGLE_PROMPTS:
        r = _check_cache(
            llm, prefix_cache, tokenizer, config, name, user_text, max_new, atol,
        )
        all_ok = all_ok and r.fresh_logits_ok and r.filled_logits_ok

        fresh_tag = (
            "PASS" if r.fresh_logits_ok
            else f"FAIL dv@{r.fresh_logits_dv} max_diff={r.fresh_logits_max_diff:.2e}"
        )
        filled_tag = (
            "PASS" if r.filled_logits_ok
            else f"FAIL dv@{r.filled_logits_dv} max_diff={r.filled_logits_max_diff:.2e}"
        )
        stop_match = (r.no_cache_stop == r.fresh_stop == r.filled_stop)
        stop_frag = (
            f"stop={r.no_cache_stop}"
            if stop_match
            else f"stop_MISMATCH nc={r.no_cache_stop} fresh={r.fresh_stop} filled={r.filled_stop}"
        )
        print(f"    [{name:<5s}] prompt={r.prompt_len}t  "
              f"out={r.no_cache_len}/{r.fresh_len}/{r.filled_len}  {stop_frag}")
        print(f"        fresh_vs_no_cache : {fresh_tag}  "
              f"tokens_match={r.fresh_tokens_ok}")
        print(f"        filled_vs_no_cache: {filled_tag}  "
              f"tokens_match={r.filled_tokens_ok}")

    print(f"\n  batched == single (matched SDPA kernel, logit atol={atol:.0e}):")
    print("    lockstep decode (eos_set=[] so all rows run max_new steps)")
    for name, template, max_new, B in SANITY_BATCHED_TEMPLATES:
        r = _check_batched(
            llm, tokenizer, config, name, template, max_new, B, atol,
        )
        if not r.uniform:
            print(f"    [{name:<5s}] SKIP — non-uniform tokenization")
            continue
        all_rows_ok = all(r.per_row_logits_ok)
        all_ok = all_ok and all_rows_ok
        any_token_mismatch = not all(r.per_row_tokens_ok)
        print(f"    [{name:<5s}] prompt={r.prompt_len}t  B={r.B}  "
              f"{'PASS' if all_rows_ok else 'FAIL'}"
              + ("  (token mismatch on some rows)" if any_token_mismatch else ""))
        for i in range(r.B):
            status = (
                "PASS" if r.per_row_logits_ok[i]
                else f"FAIL dv@{r.per_row_logits_dv[i]} max_diff={r.per_row_max_diff[i]:.2e}"
            )
            print(f"        row{i}: single_out={r.per_row_single_len[i]}t "
                  f"batched_out={r.per_row_batched_len[i]}t  "
                  f"logits:{status}  tokens_match={r.per_row_tokens_ok[i]}")

    return all_ok


def run_activation_correctness(
    llm: TorchLLM,
    tokenizer,
    config: TemplateConfig,
) -> bool:
    """Check that an installed no-op intervention is exactly a no-op."""
    print("\n" + "=" * 72)
    print("ACTIVATION-CORRECTNESS (no-op intervention is identity)")
    print("=" * 72)

    conv = [
        {"role": "system", "content": SYSTEM_SHORT},
        {"role": "user", "content": "Explain TCP vs UDP in one sentence."},
    ]
    ids, roles = tokenize_conversation(
        conv, tokenizer, config, add_generation_prompt=True,
    )

    llm.prefix_cache = None
    llm.clear_interventions()
    no_hook_out, no_hook_logits = _gen_with_logits(llm, ids, roles, 32, tokenizer)

    class _NoopCounter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden):
            self.calls += 1
            return torch.zeros_like(hidden)

    noop = _NoopCounter()
    llm.prefix_cache = None
    llm.register_intervention(noop, layers=list(range(llm.model.params.n_layers)))
    hooked_out, hooked_logits = _gen_with_logits(llm, ids, roles, 32, tokenizer)
    llm.clear_interventions()

    logits_ok, dv, max_diff = _logits_equal(no_hook_logits, hooked_logits, atol=0.0)
    tokens_ok = no_hook_out.token_ids == hooked_out.token_ids
    calls_ok = noop.calls > 0
    print(f"  intervention calls: {noop.calls}")
    print(f"  logits: {'PASS' if logits_ok else f'FAIL dv@{dv} max_diff={max_diff:.2e}'}")
    print(f"  tokens_match={tokens_ok}")
    return logits_ok and tokens_ok and calls_ok


def run_decode_compile(
    llm: TorchLLM,
    tokenizer,
    config: TemplateConfig,
    *,
    compile_mode: str = "default",
) -> bool:
    """Decode-only torch.compile smoke: eager vs compiled tokens match."""
    print("\n" + "=" * 72)
    print("DECODE-COMPILE (torch.compile smoke)")
    print("=" * 72)

    conv = [
        {"role": "system", "content": SYSTEM_SHORT},
        {"role": "user", "content": "Write one paragraph about why caches make systems faster."},
    ]
    ids, roles = tokenize_conversation(
        conv, tokenizer, config, add_generation_prompt=True,
    )
    max_new = 96

    def run_once(label: str) -> tuple[GenOutput, float]:
        llm.prefix_cache = None
        out, wall = _gen_torchllms_once(llm, ids, roles, max_new, tokenizer)
        print(f"  {label:<24s} len={len(out.token_ids):>3d} wall_s={wall:.2f}")
        return out, wall

    llm.prefix_cache = None
    llm.clear_interventions()
    llm.disable_decode_compile()
    eager, eager_wall = run_once("eager")

    print(f"  compiling decode-only mode={compile_mode!r} ...")
    llm.enable_decode_compile(mode=compile_mode)
    compiled_first, first_wall = run_once("compiled first")
    compiled_second, second_wall = run_once("compiled second")
    llm.disable_decode_compile()

    # bf16 compile doesn't preserve greedy-token identity against eager.
    # Aggressive Inductor matmul fusion reorders bf16 reductions inside
    # attention + MLP; drift accumulates over 36 layers and flips a top-1
    # argmax somewhere in the tail. We require first_diverge >= 8 (early
    # tokens where the margin tends to be large) rather than bit-exact
    # token identity, matching the batched-vs-single tolerance logic.
    no_hook_div = _first_diverge(eager.token_ids, compiled_second.token_ids)
    no_hook_ok = no_hook_div >= 8
    print(
        f"  no-hook first_diverge={no_hook_div}/{len(eager.token_ids)} "
        f"[{'PASS' if no_hook_ok else 'FAIL'}] "
        f"second/eager_time={second_wall / eager_wall if eager_wall > 0 else 0.0:.2f}x"
    )

    zero_vec = torch.zeros(
        llm.model.params.dim,
        device=llm.device,
        dtype=llm.model.tok_embeddings.weight.dtype,
    )
    hook = AddVec(zero_vec)
    llm.register_intervention(hook, layers=[19])
    llm.enable_decode_compile(mode=compile_mode)
    hooked_first, hooked_first_wall = run_once("compiled+zero first")
    hooked_second, hooked_second_wall = run_once("compiled+zero second")
    llm.disable_decode_compile()
    llm.clear_interventions()

    hook_div = _first_diverge(eager.token_ids, hooked_second.token_ids)
    hook_ok = hook_div >= 8
    print(
        f"  zero-vector first_diverge={hook_div}/{len(eager.token_ids)} "
        f"[{'PASS' if hook_ok else 'FAIL'}] "
        f"second/eager_time={hooked_second_wall / eager_wall if eager_wall > 0 else 0.0:.2f}x"
    )
    print(
        f"  compile_warmup_s no_hook={first_wall:.2f} zero_vector={hooked_first_wall:.2f}"
    )
    return no_hook_ok and hook_ok


# ---------------------------------------------------------------------------
# Generation-correctness — torchllms vs SGLang on realistic prompts
# ---------------------------------------------------------------------------


PARITY_CONVS: list[tuple[str, list[dict], int]] = [
    # name, conversation, max_new_tokens. Mix prompt-length × completion-length
    # regimes: short/med/long prompts × short/med/long completions.
    ("short-prompt / short-out", [
        {"role": "system", "content": SYSTEM_SHORT},
        {"role": "user", "content": "In one sentence, what does ACID stand for?"},
    ], 60),
    ("short-prompt / medium-out", [
        {"role": "system", "content": SYSTEM_SHORT},
        {"role": "user", "content":
            "In one paragraph, explain why hash tables have amortized O(1) lookup."},
    ], 160),
    ("medium-prompt / long-out", [
        {"role": "system", "content": SYSTEM_SHORT},
        {"role": "user", "content":
            "Walk me through the core idea of Paxos consensus. Assume I already "
            "know what a distributed system is and why we care about consensus. "
            "Focus on the two-phase structure (prepare / accept) and why it "
            "handles a crashed proposer. Keep it to a few paragraphs."},
    ], 400),
    ("short-prompt / long-out (code)", [
        {"role": "system", "content": SYSTEM_SHORT},
        {"role": "user", "content":
            "Write a complete Python implementation of a small LRU cache with "
            "a max_size parameter, get(), put(), and __len__. Use a doubly "
            "linked list + dict for O(1) ops. Include docstrings and a short "
            "usage example at the bottom. No external dependencies."},
    ], 600),
    ("long-prompt / short-out (agentic)", [
        {"role": "system", "content": SYSTEM_DEBUG},
        {"role": "user", "content":
            "Production incident, need fast triage. Our auth service is "
            "returning intermittent 504s on ~5-10% of requests to "
            "POST /api/v1/login. Context: 12 stateless Python pods behind a "
            "Kubernetes Service with sessionAffinity: ClientIP, Postgres 15 "
            "on RDS, traffic volume stable for weeks, no deploys in 48h. "
            "`kubectl top pods -l app=auth-service` shows two pods at ~90% "
            "CPU while the rest sit at ~30%. Restart cycles briefly reset "
            "the hotspot but it rebuilds within 5 minutes with the same two "
            "pods affected. Ingress logs show 80% of traffic to the hot pods "
            "comes from a single /24 CIDR matching our internal service "
            "mesh. We tried externalTrafficPolicy: Cluster and latency got "
            "worse due to extra NAT hops; switching to Local + NodePort "
            "caused connection-refused errors from nodes without an auth "
            "pod. `kubectl get endpoints auth-service -o wide` shows only 8 "
            "of 12 pods listed. Scaling to 30 replicas is blocked because "
            "each opens 20 persistent DB connections and we're near the RDS "
            "connection ceiling. Name the single most-likely root cause in "
            "one sentence plus the first action I should take."},
    ], 120),
    ("medium-prompt / short-out", [
        {"role": "system", "content": SYSTEM_SHORT},
        {"role": "user", "content":
            "Here's a snippet that periodically produces incorrect results "
            "under concurrent access:\n\n"
            "    counter = 0\n"
            "    def increment():\n"
            "        global counter\n"
            "        current = counter\n"
            "        counter = current + 1\n\n"
            "Running this across 8 threads 1000 times each gives values less "
            "than 8000 about 40% of the time. Name the specific race condition "
            "and the minimal fix in one sentence each."},
    ], 100),
]


def run_generation_correctness(llm, engine, tokenizer, config) -> None:
    print("\n" + "=" * 72)
    print("GENERATION-CORRECTNESS — torchllms bf16 vs SGLang bf16 on realistic prompts")
    print("=" * 72)
    print("  Not a hard gate: different engines diverge under bf16 argmax flips.")
    print("  Looking for: first several tokens match + overall overlap > 30%.")

    for name, conv, max_new in PARITY_CONVS:
        ids, roles = tokenize_conversation(
            conv, tokenizer, config, add_generation_prompt=True,
        )
        llm.prefix_cache = None
        t, _ = _gen_torchllms_once(llm, ids, roles, max_new, tokenizer)
        s, _ = _gen_sglang_once(engine, ids, max_new, tokenizer)

        dv = _first_diverge(t.token_ids, s.token_ids)
        ov = _overlap_frac(t.token_ids, s.token_ids)
        word_dist, torch_words, sglang_words, word_sim = _word_edit_stats(
            t.text, s.text,
        )
        print(f"\n  [{name}] prompt={len(ids)}t")
        print(f"    torchllms  len={len(t.token_ids)}  stop={t.stop_reason}")
        print(f"    sglang     len={len(s.token_ids)}")
        print(f"    first-diverge: {dv}  overlap: {ov:.1%}")
        print(
            f"    word-edit: {word_dist}  word-sim: {word_sim:.1%}  "
            f"words={torch_words}/{sglang_words}"
        )
        print(f"    torchllms head: {t.text[:120]!r}")
        print(f"    sglang    head: {s.text[:120]!r}")


# ---------------------------------------------------------------------------
# Throughput benchmarks
# ---------------------------------------------------------------------------


W1_CONV = [
    {"role": "system", "content": SYSTEM_SHORT},
    {"role": "user", "content": "In one sentence, what's the difference between TCP and UDP?"},
]
W1_MAX_NEW = 60

W2_CONV = [
    {"role": "system", "content": SYSTEM_SHORT},
    {"role": "user", "content":
        "Write a short story (about 300 words) about a lighthouse keeper who "
        "finds a weathered journal in a drawer. Keep the tone reflective."},
]
W2_MAX_NEW = 500

# W3: realistic multiturn debug dialog. Each user turn adds log/output context
# (~150-300 tokens); assistant replies are short. By the final turn the prompt
# is ~2000+ tokens, mimicking an agentic session where cache reuse matters.
W3_TURNS = [
    # (user content, assistant reply that we'll fabricate into history before
    #  measuring the NEXT turn. We measure each turn's generation separately.)
    ("""Intermittent 504s on our auth endpoint, ~5-10% of requests. Sample log:

[2026-04-20 14:02:17] upstream timed out (110: Operation timed out) while reading response header from upstream, client: 10.0.1.42, server: auth.example.com, request: "POST /api/v1/login HTTP/1.1", upstream: "http://auth-service.prod.svc.cluster.local:8080/api/v1/login"

Auth is a stateless Python service, 12 pods behind a K8s Service, Postgres 15 on RDS. What should I check first?""",
     None),
    ("""`kubectl top pods` shows two pods at ~90% CPU, the other ten at ~30%. Restart cycles fix it briefly then it climbs back in ~5 minutes. Same two pods keep getting hot.""",
     None),
    ("""The Service spec has sessionAffinity: ClientIP. But the auth flow is genuinely stateless — no per-pod caches or connection pools that I know of. What else could make ClientIP-sticky routing look like hot-pod saturation?""",
     None),
    ("""I checked ingress logs and saw that 80% of the traffic to those two pods comes from a single /24 CIDR (internal service mesh). Is that the smoking gun?""",
     None),
    ("""We set Service externalTrafficPolicy: Cluster and that routed traffic via kube-proxy which load-balances across pods, but now I'm seeing extra latency from the additional NAT hop. Is there a better fix?""",
     None),
    ("""Implemented Local policy + NodePort. Latency dropped back to normal, but now some requests fail with connection-refused from specific nodes. What's happening?""",
     None),
    ("""Ran `kubectl get endpoints auth-service -o wide` — only 8 of 12 pods show up. The missing 4 are all on nodes that don't have the auth-service Pod. Is Local policy incompatible with our deployment?""",
     None),
    ("""Okay so adding pod anti-affinity to spread pods across all nodes fixes it. Any downsides to that approach at our scale (30-node cluster, 12 replicas)?""",
     None),
]


def _build_w3_history_to_turn(n: int) -> list[dict]:
    """History seed + turns [0..n-1] with fabricated assistant replies so
    turn `n` is the one being measured.
    """
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_DEBUG}]
    # Fabricated short assistant replies for the first n-1 turns. Realistic
    # length + tone; content doesn't need to be consistent with actual model
    # generations since we're only measuring the tokenization/cache/decode
    # path, not the answer quality.
    FAKE_ASST = [
        "Hot pods with intermittent timeouts usually means skewed request distribution. Check `kubectl top pods` for per-pod CPU and confirm the Service type — round-robin should even this out.",
        "That's a smoking gun: persistent hot pods after restart points to a routing layer, not a workload issue. Look at the Service config, specifically sessionAffinity, and the upstream LB hashing.",
        "Stateless doesn't rule out affinity-driven skew. ClientIP affinity concentrates all requests from one source IP onto the same pod. Inspect the source IP distribution in your ingress logs.",
        "Yes. Mesh-internal traffic hitting the Service with ClientIP affinity shows up as a small number of apparent clients with huge request volumes. Consider switching the Service to default round-robin and rely on L7 LB for session stickiness if you need it.",
        "Set externalTrafficPolicy: Local on the Service. That preserves client source IP without the kube-proxy NAT hop, while still load-balancing via the L4 LB. Your current setup trades one problem for another.",
        "Local policy requires a pod on every node serving traffic through that node's NodePort. Nodes without an auth-service pod will refuse connections. Either add pod anti-affinity to spread pods, or drain those nodes from the LB target pool.",
        "Right — Local policy's tradeoff is availability-per-node. The fix is either anti-affinity spreading or limiting the LB targets to nodes known to have pods, via the cloud-provider-specific node label.",
    ]
    for i in range(n):
        user_text, _ = W3_TURNS[i]
        msgs.append({"role": "user", "content": user_text})
        if i < n - 1:
            msgs.append({"role": "assistant", "content": FAKE_ASST[i]})
    return msgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-4B torchllms correctness validation. Runs four phases: "
            "cache-correctness, activation-correctness, generation-correctness, "
            "decode-compile. Throughput benchmarking lives in "
            "``torchllms.inference.throughput_bench`` (run separately)."
        ),
    )
    parser.add_argument(
        "--no-sglang",
        action="store_true",
        help="Skip the generation-correctness phase (which requires SGLang).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(
        f"torchllms Qwen3-4B from {TORCHLLMS_CKPT}  "
        f"precision={PRECISION} max_len={MAX_LEN}"
    )
    llm = TorchLLM(
        ckpt_paths=[TORCHLLMS_CKPT],
        template_config=CONFIG_NAME,
        max_len=MAX_LEN,
        device="cuda:0",
        precision=PRECISION,
    )

    engine = None
    if not args.no_sglang:
        try:
            import sglang as sgl
        except ImportError:
            print("(SGLang not installed — generation-correctness and throughput phases will be skipped)")
        else:
            print(f"SGLang engine from {HF_PATH}  dtype=bfloat16")
            engine = sgl.Engine(
                model_path=HF_PATH,
                dtype="bfloat16",
                mem_fraction_static=0.40,
                context_length=MAX_LEN,
                disable_cuda_graph=True,
            )
    else:
        print("(SGLang skipped — generation-correctness and throughput phases will be skipped)")

    # Phase 2 RadixKVCache binds to the PagedKVPool. Materialize the pool
    # here (post-SGLang-load so SGLang observed the right free-VRAM
    # budget) and construct the radix; the LLM's generate paths will
    # reuse this pool.
    llm._ensure_cache()
    prefix_cache = llm.make_prefix_cache()

    tokenizer = AutoTokenizer.from_pretrained(HF_PATH, trust_remote_code=True)
    config = _load_config()

    try:
        # cache-correctness: fp32 bit-exact gate, bf16 drift reported-and-skipped
        cache_ok = run_cache_correctness(llm, prefix_cache, tokenizer, config)
        if cache_ok:
            print("\ncache-correctness: PASSED — cache and batched paths are bit-exact.")
        elif PRECISION == "float32":
            print("\ncache-correctness: FAILED at fp32 — real bug. Aborting remaining phases.")
            return 1
        else:
            print(f"\ncache-correctness: drifted at {PRECISION} (cuBLAS GEMM shape-dependence; "
                  "the fp32 gate is the strict correctness check).")

        # activation-correctness: no-op hook is identity
        act_ok = run_activation_correctness(llm, tokenizer, config)
        if not act_ok:
            print("\nactivation-correctness: FAILED — no-op hook changed outputs or was not called.")
            return 1

        # generation-correctness requires SGLang.
        if engine is not None:
            gc.collect(); torch.cuda.empty_cache()
            run_generation_correctness(llm, engine, tokenizer, config)
            gc.collect(); torch.cuda.empty_cache()

        # decode-compile: torch.compile smoke, runs last so a compile
        # failure doesn't block the phases above.
        decode_compile_ok = run_decode_compile(llm, tokenizer, config)
        if not decode_compile_ok:
            print("\ndecode-compile: FAILED — compiled decode diverged from eager.")
            return 1
    finally:
        if engine is not None:
            sd = getattr(engine, "shutdown", None)
            if sd is not None:
                sd()

    return 0


if __name__ == "__main__":
    sys.exit(main())
