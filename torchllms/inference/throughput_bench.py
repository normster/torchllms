"""Throughput benchmark for torchllms inference, both Qwen3-4B and gpt-oss-20b.

Five cells × four workloads × per-turn prefill/decode decomposition.

CELLS
-----

  torch-eager           torchllms, compile off, no intervention
  torch-decode-compile  torchllms decode compile (reduce-overhead →
                        Inductor + cudagraph capture on uniform B×1
                        decode steps). Prefill stays eager because
                        all our workloads are variable-length batched
                        (``LLM._model_forward`` only compiles prefill
                        when ``qlens is None``, which is never the
                        case for mixed-length batches). This cell is
                        the torchllms baseline that actually reflects
                        batched serving — prefill compile is a no-op
                        in realistic workloads.
  torch-decode-compile-match   decode-compile + ``AddVec(α=0.01 · unit) @ L_k``
                        with ``role_ids=None`` (mask=1 every token)
  torch-decode-compile-tool    decode-compile + ``AddVec(α=0.01 · unit) @ L_k``
                        with ``role_ids=[Role.TOOL]`` (mask=1 only on
                        tool-response positions in W3)
  sglang-cg-on          sglang with cudagraph enabled — external reference

Qwen3 intervention layer = 19 (Phase-5 target). gpt-oss = 11.

WORKLOADS
---------

  W1 short/short       40-60t prompt, 60t generation, B=8 uniform
                       3 trials median-of-3
  W2 short/long        60-90t prompt, 600t generation, B=8 uniform
                       3 trials median-of-3
  W3  mixed agentic    B=8 mixed batch: 4 rows non-tool (all user turns)
                       + 4 rows tool-bearing (3 of 8 turns replaced by
                       ~2000t tool responses from ``_bench_fixtures``).
                       8 turns per row. At each turn step the batch has
                       heterogeneous per-row prefill lengths — exercises
                       the flat-pack variable-length path in the
                       driver. Per-turn prefill of incremental user or
                       tool content + asst primer only (decoded tokens
                       of prior turn sit in cache unused — no fake-asst
                       prefilled into the timing). Decode budget 80
                       tokens per turn. Role-filter intervention
                       (``torch-compile-tool``) fires at the 4 tool-
                       bearing rows' tool positions.

Run protocol: each cell runs the full suite twice (pass1 + pass2).
Pass 2 is the headline; ``cold_overhead = pass1 - pass2`` is reported
informationally (first-time compile + cudagraph capture happen during
pass 1).

OUTPUT
------

Per model: memory breakdown, correctness smoke, per-cell matrix, and
per-turn W3 breakdown. Each "*_tps" number is prefill-tokens-per-
second or decoded-tokens-per-second. W3 totals exclude the untimed
cache-maintenance steps (only the explicit prefill + decode phases are
in the numbers).

USAGE
-----

    python -m torchllms.inference.throughput_bench --model {qwen3,gpt-oss}
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from torchllms.inference._bench_fixtures import TOOL_OUTPUTS
from torchllms.inference._bench_turns import (
    build_qwen3_turn,
    build_harmony_turn,
    build_harmony_system_developer_prefix,
)
from torchllms.messages import Role


# =====================================================================
# Constants / workload fixtures
# =====================================================================


SYSTEM_SHORT = (
    "You are a helpful, concise assistant. Keep answers short and to the "
    "point unless asked for detail."
)

SYSTEM_DEBUG = (
    "You are an experienced platform engineer debugging a live production "
    "incident.  You have access to tools: kubectl, bash, pg_stat. Give "
    "concrete, actionable advice; prefer specific commands or config lines "
    "over generalities."
)


# W1: short prompt, short completion. 8 lexical variants so the batch
# has variable per-row prompt lengths (realistic server workload —
# every request is slightly different).
W1_USER_VARIANTS = [
    "In one sentence, what's the difference between TCP and UDP?",
    "Briefly, how does TCP differ from UDP?",
    "TCP vs UDP — one-sentence summary of the core difference.",
    "Give a quick one-liner contrasting TCP and UDP.",
    "Concisely, what distinguishes TCP from UDP?",
    "One sentence only: TCP compared to UDP in terms of reliability.",
    "In a single sentence, outline the main difference between TCP and UDP.",
    "What is the fundamental difference between TCP and UDP, said briefly?",
]
W1_MAX_NEW = 60


# W2: short prompt, long completion. 8 variants with minor rewording so
# the batch has variable lengths.
W2_USER_VARIANTS = [
    "Write a short story (about 300 words) about a lighthouse keeper who "
    "finds a weathered journal in a drawer. Keep the tone reflective.",
    "Compose a roughly 300-word reflective story about a lighthouse keeper "
    "who discovers an old journal in a drawer.",
    "Please write a short story of around 300 words in which a lighthouse "
    "keeper finds a worn journal tucked in a drawer. Reflective tone.",
    "I'd like a reflective short story, about 300 words, where a lighthouse "
    "keeper comes across an aged journal inside a desk drawer.",
    "Produce a ~300-word reflective short story featuring a lighthouse "
    "keeper who uncovers a weathered journal hidden in a drawer.",
    "Write a contemplative short story (roughly 300 words) about a "
    "lighthouse keeper who stumbles upon a timeworn journal in a drawer.",
    "Draft a reflective 300-word short story: a lighthouse keeper finds a "
    "faded journal tucked away in a drawer one evening.",
    "Could you write a thoughtful 300-word story about a lighthouse keeper "
    "who discovers an antique journal inside one of their drawers?",
]
W2_MAX_NEW = 600


# W3 user turns — 8 messages each row. Kept short-ish (~150-250 words
# each) so the per-turn prefill is bounded. Shared across all 8 rows
# (replicated-uniform for W3a / W3b; the row-to-row variation is
# whether selected turns are tool responses vs user messages).
W3_USER_TURNS = [
    ("Intermittent 504s on our auth endpoint, ~5-10% of requests. Sample log:\n"
     "[2026-04-22 14:02:17] upstream timed out (110: Operation timed out) "
     "while reading response header from upstream, client: 10.0.1.42, "
     'request: "POST /api/v1/login HTTP/1.1", upstream: '
     '"http://auth-service.prod.svc.cluster.local:8080/api/v1/login"\n'
     "Auth is a stateless Python service, 12 pods behind a K8s Service, "
     "Postgres 15 on RDS. What should I check first?"),
    ("kubectl top pods shows two pods at ~90% CPU, the other ten at ~30%. "
     "Restart cycles fix it briefly then it climbs back in ~5 minutes. "
     "Same two pods keep getting hot. What are the usual suspects for that "
     "pattern in a K8s Service-fronted stateless Python app?"),
    ("The Service spec has sessionAffinity: ClientIP. But the auth flow is "
     "genuinely stateless — no per-pod caches or connection pools I know "
     "of. What else could make ClientIP-sticky routing look like hot-pod "
     "saturation in this scenario?"),
    ("I checked ingress logs and saw that 80% of the traffic to those two "
     "pods comes from a single /24 CIDR (internal service mesh). The mesh "
     "talks to auth through a single virtual IP. Is that the smoking gun?"),
    ("We set Service externalTrafficPolicy: Cluster and that routed traffic "
     "via kube-proxy which load-balances across pods, but now I'm seeing "
     "extra latency from the additional NAT hop. Is there a better fix "
     "that preserves source IP without the extra hop?"),
    ("Implemented Local policy + NodePort. Latency dropped back to normal, "
     "but now some requests fail with connection-refused from specific "
     "nodes. What's the failure mode here?"),
    ("Ran kubectl get endpoints auth-service -o wide — only 8 of 12 pods "
     "show up. The missing 4 are all on nodes that don't have the "
     "auth-service Pod scheduled. Is Local policy incompatible with our "
     "deployment, or is this a scheduling issue we can work around?"),
    ("Okay so adding pod anti-affinity to spread pods across all nodes "
     "fixes it. Any downsides to that approach at our scale (30-node "
     "cluster, 12 replicas)? I want to understand the tradeoff."),
]


# W3b tool-turn placement. 8 turns per row; 3 of them are tool responses.
# Turns 2, 4, 6 are replaced by tool outputs (indexed into TOOL_OUTPUTS
# which has 3 entries: kubectl, pgstat, bash). The user turns at those
# positions are dropped; the tool content plus surrounding envelope
# markers are what the model sees instead.
W3B_TOOL_SLOTS = {1: 0, 3: 1, 5: 2}  # turn_idx -> TOOL_OUTPUTS index (0-indexed)


W3_MAX_NEW = 80


# Trim tool outputs to ~5000 chars (≈ 2000 Qwen tokens, ≈ 1500 gpt-oss
# tokens) so the W3b final-turn prompt fits 16k seq len comfortably.
_TRIMMED_TOOL_OUTPUTS = [s[:5017] for s in TOOL_OUTPUTS]


# =====================================================================
# Cell + measurement infrastructure
# =====================================================================


@dataclass
class WorkloadTiming:
    label: str
    prefill_tps: float
    decode_tps: float
    total_s: float
    prompt_tokens: int
    gen_tokens: int


@dataclass
class W3Timing:
    label: str                                    # "W3a" or "W3b"
    per_turn_prefill_s: List[float] = field(default_factory=list)
    per_turn_decode_s: List[float] = field(default_factory=list)
    per_turn_prefill_tokens: List[int] = field(default_factory=list)  # total across batch
    per_turn_decode_tokens: List[int] = field(default_factory=list)
    batch_size: int = 0

    @property
    def total_s(self) -> float:
        return sum(self.per_turn_prefill_s) + sum(self.per_turn_decode_s)

    @property
    def total_prefill_tps(self) -> float:
        t = sum(self.per_turn_prefill_s)
        return sum(self.per_turn_prefill_tokens) / t if t > 0 else 0.0

    @property
    def total_decode_tps(self) -> float:
        t = sum(self.per_turn_decode_s)
        return sum(self.per_turn_decode_tokens) / t if t > 0 else 0.0


@dataclass
class CellResult:
    label: str
    warmup_wall_s: float = 0.0     # compile/cudagraph + pass 1 overhead
    w1: Optional[WorkloadTiming] = None
    w2: Optional[WorkloadTiming] = None
    w3: Optional[W3Timing] = None


# =====================================================================
# Tokenization helpers per model family
# =====================================================================


@dataclass
class TurnTokens:
    """Per-turn incremental tokens + role_ids for a single row."""
    ids: List[int]
    roles: List[int]


def _build_qwen3_row_turns(
    tokenizer, row_idx: int, include_tool: bool,
) -> List[TurnTokens]:
    """Build 8 turns for Qwen3 W3 row ``row_idx``. Lexical variation
    via (row_idx, turn_idx) salt; tool rows have W3B_TOOL_SLOTS
    turns replaced by tool-response content."""
    turns: List[TurnTokens] = []
    for i, user_text in enumerate(W3_USER_TURNS):
        if include_tool and i in W3B_TOOL_SLOTS:
            content = _TRIMMED_TOOL_OUTPUTS[W3B_TOOL_SLOTS[i]]
            kind = "tool"
        else:
            # Small per-row rewording so content across rows has
            # slightly different token counts even at the same turn.
            content = f"[row {row_idx}] {user_text}"
            kind = "user"
        ids, roles = build_qwen3_turn(
            tokenizer,
            kind=kind,
            content=content,
            is_first_turn=(i == 0),
            system=SYSTEM_DEBUG,
        )
        turns.append(TurnTokens(ids=ids, roles=roles))
    return turns


def _build_harmony_row_turns(
    row_idx: int, include_tool: bool,
) -> List[TurnTokens]:
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    # Shared builder-state per row so diff-from-prior tokenization
    # works across turns.
    builder_ref = []
    turns: List[TurnTokens] = []
    for i, user_text in enumerate(W3_USER_TURNS):
        if include_tool and i in W3B_TOOL_SLOTS:
            content = _TRIMMED_TOOL_OUTPUTS[W3B_TOOL_SLOTS[i]]
            kind = "tool"
        else:
            content = f"[row {row_idx}] {user_text}"
            kind = "user"
        ids, roles = build_harmony_turn(
            enc, kind=kind, content=content,
            is_first_turn=(i == 0),
            system_text=SYSTEM_DEBUG,
            _builder_ref=builder_ref,
        )
        turns.append(TurnTokens(ids=ids, roles=roles))
    return turns


def _build_w1_variants_qwen3(tokenizer) -> List[TurnTokens]:
    """8 W1 variants with variable lengths."""
    out = []
    for v in W1_USER_VARIANTS:
        ids, roles = build_qwen3_turn(
            tokenizer, kind="user", content=v,
            is_first_turn=True, system=SYSTEM_SHORT,
        )
        out.append(TurnTokens(ids=ids, roles=roles))
    return out


def _build_w2_variants_qwen3(tokenizer) -> List[TurnTokens]:
    out = []
    for v in W2_USER_VARIANTS:
        ids, roles = build_qwen3_turn(
            tokenizer, kind="user", content=v,
            is_first_turn=True, system=SYSTEM_SHORT,
        )
        out.append(TurnTokens(ids=ids, roles=roles))
    return out


def _build_w1_variants_harmony() -> List[TurnTokens]:
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    out = []
    for v in W1_USER_VARIANTS:
        ids, roles = build_harmony_turn(
            enc, kind="user", content=v, is_first_turn=True,
            system_text=SYSTEM_SHORT,
        )
        out.append(TurnTokens(ids=ids, roles=roles))
    return out


def _build_w2_variants_harmony() -> List[TurnTokens]:
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    out = []
    for v in W2_USER_VARIANTS:
        ids, roles = build_harmony_turn(
            enc, kind="user", content=v, is_first_turn=True,
            system_text=SYSTEM_SHORT,
        )
        out.append(TurnTokens(ids=ids, roles=roles))
    return out


# =====================================================================
# Core per-turn batched loop
# =====================================================================


@torch.inference_mode()
def _bench_w3_mixed(
    llm,
    row_turns: List[List[TurnTokens]],   # [B rows][8 turns]
    max_new: int,
    label: str,
) -> W3Timing:
    """Run B rows in parallel through 8 turns each. Each turn:
       - timed prefill of each row's incremental turn tokens
       - timed decode of ``max_new`` tokens per row

    No fake asst content is prefilled. The model's decoded tokens sit
    in the cache between turns; the next turn's prefill opens with
    ``<|im_end|>\\n<|im_start|>user\\n...`` to syntactically close the
    prior asst envelope."""
    B = len(row_turns)
    N = len(row_turns[0])
    assert all(len(rt) == N for rt in row_turns), "ragged row_turns"

    cache = llm._ensure_cache()
    _reset_cache(llm)
    # Claim B rollouts.
    rids = [cache.claim() for _ in range(B)]

    timing = W3Timing(label=label, batch_size=B)

    device = llm.device
    for turn_idx in range(N):
        # Collect per-row prefill tokens for this turn.
        row_ids: List[List[int]] = []
        row_roles: List[List[int]] = []
        for b in range(B):
            t = row_turns[b][turn_idx]
            row_ids.append(t.ids)
            row_roles.append(t.roles)
        qlens = [len(r) for r in row_ids]
        uniform = len(set(qlens)) == 1

        # Pick the faster code path: uniform → rectangular ``[B, S]``
        # + no ``qlens`` kwarg → hits the compiled-uniform prefill path
        # (Inductor with dynamic=True). Variable-length → flat-pack +
        # ``qlens`` → eager variable-length path (driver's flat-pack
        # branch). Compile on variable-length prefill is not yet
        # supported — see ``LLM._model_forward`` gating.
        if uniform:
            S = qlens[0]
            input_ids = torch.tensor(row_ids, dtype=torch.long, device=device)
            role_ids = torch.tensor(row_roles, dtype=torch.long, device=device)
            fwd_kwargs = dict(
                input_ids=input_ids, role_ids=role_ids, cache=cache,
                logits_to_keep=1,
            )
        else:
            flat_ids = [tok for row in row_ids for tok in row]
            flat_roles = [r for row in row_roles for r in row]
            input_ids = torch.tensor([flat_ids], dtype=torch.long, device=device)
            role_ids = torch.tensor([flat_roles], dtype=torch.long, device=device)
            fwd_kwargs = dict(
                input_ids=input_ids, role_ids=role_ids, cache=cache,
                logits_to_keep=1, qlens=qlens,
            )

        # Timed prefill.
        torch.cuda.synchronize(); t0 = time.perf_counter()
        logits, _ = llm._model_forward(**fwd_kwargs)
        # Sample first decode token.
        first_tok = logits.argmax(dim=-1)  # [B, 1]
        torch.cuda.synchronize(); prefill_wall = time.perf_counter() - t0

        # Timed decode: uniform B×1, max_new-1 more steps.
        torch.cuda.synchronize(); t0 = time.perf_counter()
        cur = first_tok
        gen_tokens_per_row = [[int(cur[b, 0].item())] for b in range(B)]
        # role_ids for decode steps — assistant. This matters for the
        # role-filter intervention: asst tokens don't match Role.TOOL
        # so mask is zero; same compute path as before.
        from torchllms.messages import Role
        asst_role = int(Role.ASSISTANT)
        role_ids_dec = torch.full(
            (B, 1), asst_role, dtype=torch.long, device=device,
        )
        for _ in range(max_new - 1):
            logits, _ = llm._model_forward(
                input_ids=cur, role_ids=role_ids_dec, cache=cache,
                logits_to_keep=1,
            )
            cur = logits.argmax(dim=-1)
            for b in range(B):
                gen_tokens_per_row[b].append(int(cur[b, 0].item()))
        torch.cuda.synchronize(); decode_wall = time.perf_counter() - t0

        timing.per_turn_prefill_s.append(prefill_wall)
        timing.per_turn_decode_s.append(decode_wall)
        timing.per_turn_prefill_tokens.append(sum(qlens))
        timing.per_turn_decode_tokens.append(B * max_new)

    # Retire all rollouts (release pages).
    for rid in rids:
        try:
            pages, _ = cache.retire_pages(rid)
            cache.release_pages(pages)
        except Exception:
            pass

    return timing


@torch.inference_mode()
def _bench_w1_w2_batched(
    llm,
    variants: List[TurnTokens],
    max_new: int,
    B: int,
    trials: int = 3,
    warmup: int = 1,
    label: str = "",
) -> WorkloadTiming:
    """W1 / W2 at B=8 with 8 variable-length lexical variants. Per-row
    prompts differ in length → flat-pack prefill. Median-of-3.
    Single-pass prefill + decode, no multi-turn."""
    cache = llm._ensure_cache()
    device = llm.device
    assert len(variants) >= B, f"need {B} variants, got {len(variants)}"
    chosen = variants[:B]
    qlens = [len(v.ids) for v in chosen]
    uniform = len(set(qlens)) == 1
    total_tokens = sum(qlens)

    def _one_trial() -> Tuple[float, float]:
        _reset_cache(llm)
        rids = [cache.claim() for _ in range(B)]
        if uniform:
            input_ids = torch.tensor(
                [v.ids for v in chosen], dtype=torch.long, device=device,
            )
            role_ids = torch.tensor(
                [v.roles for v in chosen], dtype=torch.long, device=device,
            )
            fwd_kwargs = dict(
                input_ids=input_ids, role_ids=role_ids, cache=cache,
                logits_to_keep=1,
            )
        else:
            flat_ids = [tok for v in chosen for tok in v.ids]
            flat_roles = [r for v in chosen for r in v.roles]
            input_ids = torch.tensor([flat_ids], dtype=torch.long, device=device)
            role_ids = torch.tensor([flat_roles], dtype=torch.long, device=device)
            fwd_kwargs = dict(
                input_ids=input_ids, role_ids=role_ids, cache=cache,
                logits_to_keep=1, qlens=qlens,
            )

        # Prefill
        torch.cuda.synchronize(); t0 = time.perf_counter()
        logits, _ = llm._model_forward(**fwd_kwargs)
        cur = logits.argmax(dim=-1)
        torch.cuda.synchronize(); prefill_s = time.perf_counter() - t0

        # Decode (uniform B × 1, cudagraph-eligible)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        from torchllms.messages import Role
        asst_role = int(Role.ASSISTANT)
        role_ids_dec = torch.full((B, 1), asst_role, dtype=torch.long, device=device)
        for _ in range(max_new - 1):
            logits, _ = llm._model_forward(
                input_ids=cur, role_ids=role_ids_dec, cache=cache,
                logits_to_keep=1,
            )
            cur = logits.argmax(dim=-1)
        torch.cuda.synchronize(); decode_s = time.perf_counter() - t0

        for rid in rids:
            pages, _ = cache.retire_pages(rid)
            cache.release_pages(pages)
        return prefill_s, decode_s

    for _ in range(warmup):
        _one_trial()

    pref_samples, dec_samples = [], []
    for _ in range(trials):
        p, d = _one_trial()
        pref_samples.append(p)
        dec_samples.append(d)
    pref_s = statistics.median(pref_samples)
    dec_s = statistics.median(dec_samples)
    return WorkloadTiming(
        label=label,
        prefill_tps=total_tokens / pref_s if pref_s > 0 else 0.0,
        decode_tps=(B * (max_new - 1)) / dec_s if dec_s > 0 else 0.0,
        total_s=pref_s + dec_s,
        prompt_tokens=total_tokens,
        gen_tokens=B * max_new,
    )


# =====================================================================
# Cell runners (torchllms)
# =====================================================================


def _reset_cache(llm):
    """Retire all active rollouts to leave the cache empty. Releases
    their pages back to the free list (retire_pages alone does NOT
    decrement refcounts — caller owns that per
    ``PagedKVPool.retire_pages`` docstring)."""
    cache = getattr(llm, "cache", None)
    if cache is None:
        return
    for rid in list(cache.active_rollouts()):
        pages, _ = cache.retire_pages(rid)
        cache.release_pages(pages)
    # Sanity: if this is a PagedKVPool all refcounts should be 0 now
    # (no prefix-cache attached, nothing else holds pages).
    if hasattr(cache, "_page_refcount") and hasattr(cache, "total_pages"):
        held = int((cache._page_refcount > 0).sum().item()) \
            if hasattr(cache._page_refcount, "sum") else \
            sum(1 for rc in cache._page_refcount if rc > 0)
        if held > 0:
            print(f"    [warn] _reset_cache: {held}/{cache.total_pages} pages still held after retire+release",
                  flush=True)


def _run_pass(
    llm, w1_variants, w2_variants, w3_rows, B, label,
):
    """One full workload pass: W1 + W2 + W3 (mixed batch, 4 tool + 4 non-tool)."""
    result = CellResult(label=label)
    _reset_cache(llm)
    result.w1 = _bench_w1_w2_batched(llm, w1_variants, W1_MAX_NEW, B, label="W1")
    _reset_cache(llm)
    result.w2 = _bench_w1_w2_batched(llm, w2_variants, W2_MAX_NEW, B, label="W2")
    _reset_cache(llm)
    result.w3 = _bench_w3_mixed(llm, w3_rows, W3_MAX_NEW, "W3")
    _reset_cache(llm)
    return result


def _run_torch_cell(
    llm, cell_label, *, decode_compile: bool, intervention,
    w1_variants, w2_variants, w3_rows, B, alpha, layer_id,
):
    """Prep + two passes. Returns pass 2 (warm). Decode-compile only —
    prefill-compile is a no-op on our variable-length batched
    workloads (see ``LLM._model_forward`` ``qlens is None`` gate).
    Kept the enable_prefill_compile call out of the cell entirely to
    avoid the misleading impression that prefill was getting compiled.
    """
    llm.clear_interventions()
    if intervention == "match":
        from torchllms.models import AddVec
        v = _make_unit_vec(llm)
        llm.register_intervention(
            AddVec(alpha * v), layers=[layer_id], role_ids=None,
        )
    elif intervention == "tool":
        from torchllms.models import AddVec
        from torchllms.messages import Role
        v = _make_unit_vec(llm)
        llm.register_intervention(
            AddVec(alpha * v), layers=[layer_id], role_ids=[Role.TOOL],
        )

    if decode_compile:
        llm.enable_decode_compile(mode="reduce-overhead")
    else:
        llm.disable_decode_compile()
    # Always-off: prefill compile only engages on uniform qlens; all
    # our workloads are variable-length batched. See
    # docs/note_prefill_compile_plan.md § Future work: bucketed
    # cudagraph-prefill.
    llm.disable_prefill_compile()

    print(f"\n>>> Cell [{cell_label}] — pass 1 (may include compile)", flush=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    pass1 = _run_pass(llm, w1_variants, w2_variants, w3_rows, B, cell_label + "/p1")
    torch.cuda.synchronize(); pass1_wall = time.perf_counter() - t0

    print(f">>> Cell [{cell_label}] — pass 2 (warm)", flush=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    pass2 = _run_pass(llm, w1_variants, w2_variants, w3_rows, B, cell_label + "/p2")
    torch.cuda.synchronize(); pass2_wall = time.perf_counter() - t0

    print(f">>> Cell [{cell_label}] — pass 3 (warm)", flush=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    pass3 = _run_pass(llm, w1_variants, w2_variants, w3_rows, B, cell_label + "/p3")
    torch.cuda.synchronize(); pass3_wall = time.perf_counter() - t0

    # Report pass 3 as the "warmest" steady-state. cold_overhead =
    # pass1 - pass3 captures both compile-first cost and any transient
    # that shows up only in pass 1 vs steady state. The pass 2 / pass 3
    # delta tells us how noisy subsequent warm runs are.
    pass3.warmup_wall_s = pass1_wall - pass3_wall
    print(f"    pass1={pass1_wall:.2f}s  pass2={pass2_wall:.2f}s  "
          f"pass3={pass3_wall:.2f}s  "
          f"cold_overhead(p1-p3)={pass1_wall - pass3_wall:+.2f}s  "
          f"warm_jitter(p2-p3)={pass2_wall - pass3_wall:+.2f}s", flush=True)
    return pass3


def _make_unit_vec(llm, seed: int = 42) -> torch.Tensor:
    dim = llm.model.params.dim
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(dim, generator=g).to(
        device=llm.device, dtype=llm.model.tok_embeddings.weight.dtype,
    )
    return v / v.norm()


# =====================================================================
# SGLang cell
# =====================================================================


def _run_sglang_cell(model_cfg, *, w3_rows, w1_variants, w2_variants, B):
    """Load sgl.Engine, bench variable-length W1/W2 + mixed W3, shutdown.

    Prefill/decode split uses sglang's own scheduler log lines
    (``Prefill batch, #new-token: N, input throughput: T`` etc.) —
    for each prefill batch that runs during a bench call we compute
    ``time = N / T``, sum across batches to get exact prefill wall.
    Decode wall = total_wall − prefill_wall. This avoids the
    stream-iteration Python overhead that inflates the streaming
    timing approach by ~25% on batched W3.
    """
    import sglang as sgl
    import logging as _logging
    import re as _re

    # -------- sglang log capture --------
    _PREFILL_RE = _re.compile(
        r"Prefill batch.*?#new-token:\s*(\d+).*?input throughput \(token/s\):\s*([0-9.]+)"
    )
    # Each entry: (wall_time_at_log, n_new_tokens, input_throughput_tok_s)
    _captured: List[tuple] = []

    class _SglHandler(_logging.Handler):
        def emit(self, rec):
            msg = rec.getMessage()
            m = _PREFILL_RE.search(msg)
            if m:
                _captured.append(
                    (time.perf_counter(), int(m.group(1)), float(m.group(2)))
                )
    _handler = _SglHandler()
    _handler.setLevel(_logging.DEBUG)
    _root = _logging.getLogger()
    _root.addHandler(_handler)
    _root.setLevel(_logging.DEBUG)
    _logging.getLogger("sglang").setLevel(_logging.INFO)

    engine = sgl.Engine(
        model_path=model_cfg["hf_path"], dtype="bfloat16",
        mem_fraction_static=0.80,
        context_length=model_cfg["max_seq_len"],
        log_level="info",  # needed to emit Prefill/Decode batch log lines
    )
    result = CellResult(label="sglang-cg-on")
    stop_ids = list(model_cfg["stop_ids"])

    def _sgl_log_split(prompt_token_lists, max_new):
        """Run a non-streaming generate, then consult captured
        ``Prefill batch`` log lines that fell within the call window
        to compute prefill wall from sglang's own reported throughputs
        (``time_per_batch = N_tokens / reported_tps``). Decode wall =
        total_wall − prefill_wall."""
        params = {
            "max_new_tokens": max_new, "temperature": 0.0,
            "stop_token_ids": stop_ids,
        }
        window_start_idx = len(_captured)
        t0 = time.perf_counter()
        out = engine.generate(
            input_ids=prompt_token_lists, sampling_params=params,
        )
        t1 = time.perf_counter()
        total_s = t1 - t0

        prefill_s = 0.0
        total_prefill_tokens = 0
        for ts, n_tok, tps in _captured[window_start_idx:]:
            if tps > 0 and n_tok > 0:
                prefill_s += n_tok / tps
                total_prefill_tokens += n_tok
        decode_s = max(total_s - prefill_s, 1e-6)
        return prefill_s, decode_s, total_prefill_tokens

    def _bench_w1_w2_sgl(variants, max_new: int, label: str):
        """variable-length B prompts, streaming timing."""
        prompts = [v.ids for v in variants[:B]]
        total_prompt = sum(len(p) for p in prompts)
        _sgl_log_split(prompts, max_new)  # warmup
        samples_p, samples_d = [], []
        for _ in range(3):
            p, d, _ = _sgl_log_split(prompts, max_new)
            samples_p.append(p)
            samples_d.append(d)
        pref_s = statistics.median(samples_p)
        dec_s = statistics.median(samples_d)
        return WorkloadTiming(
            label=label,
            prefill_tps=total_prompt / pref_s,
            decode_tps=(B * (max_new - 1)) / dec_s,
            total_s=pref_s + dec_s,
            prompt_tokens=total_prompt,
            gen_tokens=B * max_new,
        )

    def _bench_w3_sgl(rows: List[List[TurnTokens]], label: str):
        timing = W3Timing(label=label, batch_size=B)
        B_ = len(rows)
        # Per-turn cumulative prompt. sglang's radix cache gives
        # prefix-hit pricing so the effective prefill per turn is just
        # the new turn's tokens — apples-to-apples with our torchllms
        # per-turn-incremental timing.
        for turn_idx in range(len(rows[0])):
            prompts = []
            for b in range(B_):
                cum = []
                for tt in rows[b][: turn_idx + 1]:
                    cum.extend(tt.ids)
                prompts.append(cum)
            prefill_s, decode_s, _ = _sgl_log_split(prompts, W3_MAX_NEW)
            timing.per_turn_prefill_s.append(prefill_s)
            timing.per_turn_decode_s.append(decode_s)
            inc = sum(len(rows[b][turn_idx].ids) for b in range(B_))
            timing.per_turn_prefill_tokens.append(inc)
            timing.per_turn_decode_tokens.append(B_ * W3_MAX_NEW)
        return timing

    print("\n>>> Cell [sglang-cg-on] — bench", flush=True)
    t0 = time.perf_counter()
    result.w1 = _bench_w1_w2_sgl(w1_variants, W1_MAX_NEW, "W1")
    result.w2 = _bench_w1_w2_sgl(w2_variants, W2_MAX_NEW, "W2")
    result.w3 = _bench_w3_sgl(w3_rows, "W3")
    wall = time.perf_counter() - t0
    print(f"    sglang total wall={wall:.2f}s", flush=True)

    engine.shutdown()
    return result


# =====================================================================
# Memory reporting
# =====================================================================


def _memory_snapshot(llm, label: str):
    import torch
    cache = llm.cache
    weight_bytes = sum(
        p.element_size() * p.numel()
        for p in llm.model.parameters()
    ) + sum(
        b.element_size() * b.numel()
        for b in llm.model.buffers()
    )
    kv_bytes = 0
    if cache is not None and hasattr(cache, "k_cache"):
        kv_bytes = (
            cache.k_cache.element_size() * cache.k_cache.numel()
            + cache.v_cache.element_size() * cache.v_cache.numel()
        )
    peak_bytes = torch.cuda.max_memory_allocated()
    alloc_bytes = torch.cuda.memory_allocated()
    print(f"\n  memory @ {label}:")
    print(f"    weights:    {weight_bytes / 1e9:6.2f} GB")
    print(f"    kv_pool:    {kv_bytes / 1e9:6.2f} GB")
    print(f"    allocated:  {alloc_bytes / 1e9:6.2f} GB")
    print(f"    peak:       {peak_bytes / 1e9:6.2f} GB")


# =====================================================================
# Correctness smoke
# =====================================================================


def _edit_distance_tokens(a: List[int], b: List[int]) -> int:
    """Simple Levenshtein on token sequences."""
    if len(a) < len(b):
        return _edit_distance_tokens(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            dele = cur[j] + 1
            sub = prev[j] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


@torch.inference_mode()
def _correctness_smoke(llm, w1_turn, w3_rows, alpha, layer_id):
    """Three comparisons on a short prompt:
       1. α=0 bit-exact between eager and compile.
       2. α=0.01 match-all: edit distance eager-vs-compile.
       3. α=0.01 tool-filter on a W3b tool turn: edit distance.
    """
    from torchllms.models import AddVec
    from torchllms.messages import Role

    print("\n" + "=" * 72)
    print("CORRECTNESS SMOKE (α=0 bit-exact control + α=0.01 edit-dist checks)")
    print("=" * 72)
    cache = llm._ensure_cache()
    device = llm.device

    def _short_gen(prompt_turn: TurnTokens, max_new=20) -> List[int]:
        _reset_cache(llm)
        rid = cache.claim()
        ids_t = torch.tensor([prompt_turn.ids], dtype=torch.long, device=device)
        roles_t = torch.tensor([prompt_turn.roles], dtype=torch.long, device=device)
        from torchllms.messages import Role
        asst_r = int(Role.ASSISTANT)
        role_dec = torch.full((1, 1), asst_r, dtype=torch.long, device=device)
        logits, _ = llm._model_forward(
            input_ids=ids_t, role_ids=roles_t, cache=cache, logits_to_keep=1,
        )
        cur = logits.argmax(dim=-1)
        out = [int(cur[0, 0].item())]
        for _ in range(max_new - 1):
            logits, _ = llm._model_forward(
                input_ids=cur, role_ids=role_dec, cache=cache, logits_to_keep=1,
            )
            cur = logits.argmax(dim=-1)
            out.append(int(cur[0, 0].item()))
        try:
            pages, _ = cache.retire_pages(rid)
            cache.release_pages(pages)
        except Exception:
            pass
        return out

    # 1. Control: α=0 eager vs compile — must be bit-exact.
    llm.clear_interventions()
    llm.disable_prefill_compile()
    llm.disable_decode_compile()
    eager_ctrl = _short_gen(w1_turn)
    llm.enable_decode_compile(mode="reduce-overhead")
    compile_ctrl = _short_gen(w1_turn)
    ctrl_match = eager_ctrl == compile_ctrl
    print(f"  [1] α=0: eager vs decode-compile on W1 prompt:")
    print(f"      tokens_match={ctrl_match}  (eager[:6]={eager_ctrl[:6]}  "
          f"compile[:6]={compile_ctrl[:6]})")

    # 2. Match-all α=0.01: edit distance between eager and compile.
    v = _make_unit_vec(llm)
    llm.clear_interventions()
    llm.disable_decode_compile()
    llm.register_intervention(AddVec(alpha * v), layers=[layer_id], role_ids=None)
    eager_match = _short_gen(w1_turn)
    llm.clear_interventions()
    llm.register_intervention(AddVec(alpha * v), layers=[layer_id], role_ids=None)
    llm.enable_decode_compile(mode="reduce-overhead")
    compile_match = _short_gen(w1_turn)
    ed_match = _edit_distance_tokens(eager_match, compile_match)
    print(f"  [2] α=0.01 match-all: edit_dist = {ed_match}/{len(eager_match)} "
          f"tokens  (eager[:6]={eager_match[:6]} compile[:6]={compile_match[:6]})")

    # 3. Role-filter α=0.01 on a tool-bearing prompt. Pick a tool-
    #    bearing row (rows 4-7 in w3_rows per our main()) and concat
    #    its turn 0 + turn 1 (turn 1 is a tool response per
    #    W3B_TOOL_SLOTS).
    tool_row = None
    for row in w3_rows:
        if any(any(r == int(Role.TOOL) for r in t.roles) for t in row):
            tool_row = row
            break
    assert tool_row is not None, "no tool-bearing row in w3_rows"
    cat_ids = tool_row[0].ids + tool_row[1].ids
    cat_roles = tool_row[0].roles + tool_row[1].roles
    tool_turn = TurnTokens(ids=cat_ids, roles=cat_roles)

    llm.clear_interventions()
    llm.disable_decode_compile()
    llm.register_intervention(
        AddVec(alpha * v), layers=[layer_id], role_ids=[Role.TOOL],
    )
    eager_tool = _short_gen(tool_turn)
    llm.clear_interventions()
    llm.register_intervention(
        AddVec(alpha * v), layers=[layer_id], role_ids=[Role.TOOL],
    )
    llm.enable_decode_compile(mode="reduce-overhead")
    compile_tool = _short_gen(tool_turn)
    ed_tool = _edit_distance_tokens(eager_tool, compile_tool)
    print(f"  [3] α=0.01 role_ids=[TOOL] on tool-bearing prompt: "
          f"edit_dist = {ed_tool}/{len(eager_tool)} tokens")

    llm.clear_interventions()


# =====================================================================
# Printer
# =====================================================================


def _print_matrix(model: str, cells: List[CellResult], run_seconds: float):
    print("\n" + "=" * 92)
    print(f"THROUGHPUT MATRIX — {model}")
    print("=" * 92)
    header = (
        f"  {'cell':<26} {'W1 pref':>10} {'W1 dec':>8} "
        f"{'W2 pref':>10} {'W2 dec':>8} "
        f"{'W3 pref':>10} {'W3 dec':>8} {'W3 s':>8} "
        f"{'cold_s':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in cells:
        def _fmt(w):
            return (w.prefill_tps, w.decode_tps) if w else (0.0, 0.0)
        def _fmt_w3(w):
            return (w.total_prefill_tps, w.total_decode_tps, w.total_s) if w else (0.0, 0.0, 0.0)
        w1p, w1d = _fmt(c.w1)
        w2p, w2d = _fmt(c.w2)
        w3p, w3d, w3s = _fmt_w3(c.w3)
        print(
            f"  {c.label:<26} {w1p:>10.1f} {w1d:>8.1f} "
            f"{w2p:>10.1f} {w2d:>8.1f} "
            f"{w3p:>10.1f} {w3d:>8.1f} {w3s:>7.2f}s "
            f"{c.warmup_wall_s:>7.1f}s"
        )
    print(f"\n  total run: {run_seconds:.1f}s")

    # Per-turn W3 breakdown
    print("\n" + "=" * 92)
    print("PER-TURN W3 wall-ms (prefill + decode, batched 4 tool + 4 non-tool)")
    print("=" * 92)
    turns = list(range(1, 1 + len(cells[0].w3.per_turn_prefill_s)))
    hdr = "  " + "cell".ljust(26) + "".join(f"   t{i:<2}".rjust(11) for i in turns)
    print(hdr)
    for c in cells:
        row = f"  {c.label:<26}"
        for i in range(len(c.w3.per_turn_prefill_s)):
            row += f"{c.w3.per_turn_prefill_s[i]*1000 + c.w3.per_turn_decode_s[i]*1000:10.0f}ms "
        print(row)

    print("\n  W3 batched prefill tokens per turn (sum over 8 rows):")
    print(f"    {cells[0].w3.per_turn_prefill_tokens}")


# =====================================================================
# Main
# =====================================================================


_MODELS = {
    "qwen3": {
        "hf_path": "/root/qwen3-4b",
        "ckpt": "/root/qwen3-4b/consolidated.00.pth",
        "template_config": "qwen3_chatml_nothink.yaml",
        "max_seq_len": 16384,
        "max_bsz": 8,
        "kv_memory_gb": 10.0,
        "stop_ids": [151645, 151643],
        "precision": "bfloat16",
        "intervention_layer": 19,
    },
    "gpt-oss": {
        "hf_path": "/root/gpt-oss-20b",
        "ckpt": "/root/gpt-oss-20b/original",
        "template_config": None,
        "max_seq_len": 16384,
        "max_bsz": 8,
        "kv_memory_gb": 5.0,
        "stop_ids": [200012, 200002, 199999],
        "precision": "bfloat16",
        "intervention_layer": 11,
    },
}


def _load_torchllms(model: str):
    from torchllms.inference.llm import LLM
    cfg = _MODELS[model]
    kwargs = dict(
        ckpt_paths=[cfg["ckpt"]], max_len=cfg["max_seq_len"],
        max_bsz=cfg["max_bsz"], device="cuda:0", precision=cfg["precision"],
        kv_memory_gb=cfg["kv_memory_gb"],
    )
    if cfg["template_config"]:
        kwargs["template_config"] = cfg["template_config"]
        # Override ModelParams.max_seq_len so RoPE cache is sized for
        # our 16k-token W3 workload. Qwen3's params.json carries the
        # base 4k pretrain length; we need the longer context here.
        kwargs["model_kwargs"] = {"max_seq_len": cfg["max_seq_len"]}
    else:
        kwargs["eos_ids"] = cfg["stop_ids"]
        kwargs["model_kwargs"] = {"max_seq_len": cfg["max_seq_len"], "mxfp4": True}
    return LLM(**kwargs), cfg


def _build_fixtures(model: str, llm, cfg):
    """Tokenize W1/W2 variants and W3 mixed rows per model family.

    W1/W2: 8 lexical variants → variable-length batch (realistic
    server workload).
    W3: ``max_bsz`` rows, first half non-tool, second half tool-bearing
    → mixed batch at each turn step, exercises the variable-length
    flat-pack prefill path.

    Returns (w1_variants, w2_variants, w3_rows).
    """
    B = cfg["max_bsz"]
    n_tool = B // 2
    n_notool = B - n_tool
    if model == "qwen3":
        tok = llm.tokenizer
        w1_variants = _build_w1_variants_qwen3(tok)
        w2_variants = _build_w2_variants_qwen3(tok)
        w3_rows = (
            [_build_qwen3_row_turns(tok, row_idx=r, include_tool=False)
             for r in range(n_notool)]
            + [_build_qwen3_row_turns(tok, row_idx=r, include_tool=True)
               for r in range(n_notool, B)]
        )
    else:  # gpt-oss
        w1_variants = _build_w1_variants_harmony()
        w2_variants = _build_w2_variants_harmony()
        w3_rows = (
            [_build_harmony_row_turns(row_idx=r, include_tool=False)
             for r in range(n_notool)]
            + [_build_harmony_row_turns(row_idx=r, include_tool=True)
               for r in range(n_notool, B)]
        )
    return w1_variants, w2_variants, w3_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen3", "gpt-oss"], required=True)
    ap.add_argument("--no-sglang", action="store_true",
                    help="Skip the sglang-cg-on cell.")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="Intervention scale for match/tool cells.")
    args = ap.parse_args()

    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    print(f"\n==== Loading torchllms / {args.model} ====")
    llm, cfg = _load_torchllms(args.model)
    llm._ensure_cache()
    B = cfg["max_bsz"]

    w1_variants, w2_variants, w3_rows = _build_fixtures(args.model, llm, cfg)

    # Print tokenization summary
    print(f"\n  tokenization ({args.model}):")
    print(f"    W1 variant lengths: {[len(v.ids) for v in w1_variants[:B]]}")
    print(f"    W2 variant lengths: {[len(v.ids) for v in w2_variants[:B]]}")
    print(f"    W3 row types: {['tool' if any(r == int(Role.TOOL) for t in row for r in t.roles) else 'notool' for row in w3_rows]}")
    for i, row in enumerate(w3_rows):
        print(f"    W3 row[{i}] per-turn: {[len(t.ids) for t in row]}  sum={sum(len(t.ids) for t in row)}t")

    _memory_snapshot(llm, "post-load")

    # Correctness smoke (using first W1 variant as probe prompt)
    _correctness_smoke(
        llm, w1_variants[0], w3_rows,
        alpha=args.alpha, layer_id=cfg["intervention_layer"],
    )

    cells: List[CellResult] = []

    # Cell 1: torch-eager
    c1 = _run_torch_cell(
        llm, "torch-eager", decode_compile=False, intervention=None,
        w1_variants=w1_variants, w2_variants=w2_variants, w3_rows=w3_rows,
        B=B, alpha=args.alpha, layer_id=cfg["intervention_layer"],
    )
    cells.append(c1)

    # Cell 2: torch-decode-compile
    c2 = _run_torch_cell(
        llm, "torch-decode-compile", decode_compile=True, intervention=None,
        w1_variants=w1_variants, w2_variants=w2_variants, w3_rows=w3_rows,
        B=B, alpha=args.alpha, layer_id=cfg["intervention_layer"],
    )
    cells.append(c2)

    # Cell 3: torch-decode-compile-match
    c3 = _run_torch_cell(
        llm, "torch-decode-compile-match", decode_compile=True, intervention="match",
        w1_variants=w1_variants, w2_variants=w2_variants, w3_rows=w3_rows,
        B=B, alpha=args.alpha, layer_id=cfg["intervention_layer"],
    )
    cells.append(c3)

    # Cell 4: torch-decode-compile-tool
    c4 = _run_torch_cell(
        llm, "torch-decode-compile-tool", decode_compile=True, intervention="tool",
        w1_variants=w1_variants, w2_variants=w2_variants, w3_rows=w3_rows,
        B=B, alpha=args.alpha, layer_id=cfg["intervention_layer"],
    )
    cells.append(c4)

    _memory_snapshot(llm, "post-torchllms")

    # Teardown torchllms
    print("\n==== Tearing down torchllms ====")
    llm.clear_interventions()
    llm.disable_prefill_compile()
    llm.disable_decode_compile()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    try:
        from torchllms.models import flashinfer_attention as _fa
        _fa._WRAPPER_CACHE.clear()
        _fa._DECODE_WRAPPER_CACHE.clear()
    except Exception:
        pass
    try:
        from torchllms.models import paged_attention as _pa
        _pa._PREFILL_CACHE.clear()
        _pa._DECODE_CACHE.clear()
    except Exception:
        pass

    if not args.no_sglang:
        time.sleep(3)
        c5 = _run_sglang_cell(
            cfg, w3_rows=w3_rows,
            w1_variants=w1_variants, w2_variants=w2_variants, B=B,
        )
        cells.append(c5)

    run_seconds = time.perf_counter() - start
    _print_matrix(args.model, cells, run_seconds)


if __name__ == "__main__":
    sys.exit(main() or 0)
