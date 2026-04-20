"""Chat templating and tokenization for OpenAI Chat Completions conversations.

Input format: OpenAI Chat Completions messages, with the DeepSeek
`reasoning_content` extension for assistant reasoning.

    {
        "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant",
                "reasoning_content": "...",     # optional; CoT trace
                "content": "...",                # visible output (may be None)
                "tool_calls": [                  # optional; list of calls
                    {"function": {"name": "bash", "arguments": "..."}},
                ],
            },
            {"role": "tool",
                "tool_call_id": "...",
                "content": "..."},
        ],
    }

One assistant message may carry reasoning_content, any number of tool_calls,
and visible content — all in one envelope. The renderer walks these in order
and emits role-specific string wrappers.

Rendering strategy: build the full prompt as a string with per-region
annotations, then encode in one shot. This preserves BPE merges across
content boundaries (e.g. Qwen3's `.\\n` → [624], `\\n\\n` → [271]) and matches
HF `apply_chat_template` byte-for-byte. role_ids are assigned by mapping each
token's character offset back to the region that produced it.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerBase


class Role(Enum):
    SYSTEM = 0
    USER = 1
    ASSISTANT = 2
    TOOL = 3
    REASONING = 4
    TOOL_CALL = 5
    # OTHER is retained for downstream code (training padding, legacy
    # consumers). New code should prefer NO_ROLE (-1) for template / wrapper
    # tokens so role-conditional layers can mask them cleanly.
    OTHER = 6

    def __int__(self):
        return self.value


# Sentinel for tokens that carry template / wrapper content, not semantic
# content. Role-conditional embeddings and probes should mask these out.
NO_ROLE: int = -1


# Canonical mapping from config role names (YAML keys, Chat Completions roles,
# and derived sub-role names like "reasoning" / "tool_call") to Role enum
# values. Developer messages (Harmony) collapse into SYSTEM.
ROLE_NAMES: Dict[str, Role] = {
    "system": Role.SYSTEM,
    "developer": Role.SYSTEM,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL,
    "reasoning": Role.REASONING,
    "tool_call": Role.TOOL_CALL,
}


class RoleConfig(BaseModel):
    """Per-role rendering rules, as strings.

    envelope_start/end:  wraps the whole role block (e.g.
                         "<|im_start|>assistant\\n" and "<|im_end|>\\n").
    inner_start/end:     wraps the content inside the envelope (e.g.
                         "<think>\\n" and "\\n</think>\\n\\n").
    merge_separator:     emitted between merged items when this role is
                         followed by a merge-compatible next role. Empty
                         string = no separator. Qwen3 tool_call uses "\\n"
                         here (a newline between consecutive tool calls
                         inside one assistant envelope).
    merge_with:          list of role names that, when they appear
                         consecutively, share this role's envelope. Skips
                         this role's envelope_end and the next role's
                         envelope_start; emits merge_separator instead.
    """

    envelope_start: str = ""
    envelope_end: str = ""
    inner_start: str = ""
    inner_end: str = ""
    merge_separator: str = ""
    merge_with: List[str] = Field(default_factory=list)


class TemplateConfig(BaseModel):
    """Tokenization rules for one model family / variant.

    `roles` maps role name → RoleConfig. Role names must be a subset of
    ROLE_NAMES keys.
    """

    roles: Dict[str, RoleConfig] = Field(default_factory=dict)
    bos: str = ""
    # Appended when add_generation_prompt=True. For Qwen3 thinking this is
    # "<|im_start|>assistant\\n"; for nothink it's the same plus the empty
    # "<think>\\n\\n</think>\\n\\n" pre-fill.
    generation_prompt_suffix: str = ""
    # Token IDs that signal end-of-assistant-turn. Generation stops when the
    # generated suffix matches this sequence.
    stop_token_ids: List[int] = Field(default_factory=list)
    # Drop reasoning_content from assistant turns prior to the most recent user
    # turn. Both Qwen3 (via last_query_index) and Harmony (per OpenAI's
    # subsequent-sampling guidance) recommend this; some models do not.
    strip_reasoning_from_prior_turns: bool = True
    # Strip leading/trailing whitespace from content before emission.
    strip_whitespace: bool = True
    # Qwen3-style: even when an assistant message has no reasoning_content, the
    # LAST assistant turn (typically the generation target) still receives a
    # "<think>\\n\\n</think>\\n\\n" pre-fill. When True, if the final message
    # is an assistant message lacking reasoning_content, inject an empty
    # reasoning fragment. Turn off for models that don't require this.
    inject_empty_reasoning_for_last_turn: bool = False

    def describe(self, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> str:
        """Print each field's string value, plus (optionally) its token IDs."""
        lines: List[str] = []

        def show(name: str, val: str) -> None:
            if not val:
                return
            line = f"{name}: {val!r}"
            if tokenizer is not None:
                ids = tokenizer.encode(val, add_special_tokens=False)
                line += f"  → {ids}"
            lines.append(line)

        show("bos", self.bos)
        show("generation_prompt_suffix", self.generation_prompt_suffix)
        lines.append(
            f"strip_reasoning_from_prior_turns: {self.strip_reasoning_from_prior_turns}"
        )
        lines.append(f"strip_whitespace: {self.strip_whitespace}")
        lines.append(
            f"inject_empty_reasoning_for_last_turn: "
            f"{self.inject_empty_reasoning_for_last_turn}"
        )
        for role_name, rc in self.roles.items():
            lines.append(f"[{role_name}]")
            for fname in ("envelope_start", "envelope_end", "inner_start", "inner_end"):
                v = getattr(rc, fname)
                if v:
                    line = f"  {fname}: {v!r}"
                    if tokenizer is not None:
                        ids = tokenizer.encode(v, add_special_tokens=False)
                        line += f"  → {ids}"
                    lines.append(line)
            if rc.merge_with:
                lines.append(f"  merge_with: {rc.merge_with}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------


@dataclass
class _Fragment:
    """One flat rendering unit: a role name and its content string."""

    role_name: str
    content: str


def _render_tool_call_payload(call: Dict[str, Any]) -> str:
    """Render a Chat Completions tool_call as the JSON payload that goes
    inside <tool_call>...</tool_call> (Qwen3-style) or equivalent.

    Chat Completions stores arguments as a JSON-stringified object. Qwen3's
    chat template interpolates that string verbatim after an outer
    `{"name": "...", "arguments": ...}` wrapper (note the space after each
    colon in the outer wrapper, but no spaces inside whatever the caller
    supplied as arguments). We preserve the caller's exact argument string
    to stay byte-parity with HF's template.
    """
    import json

    fn = call.get("function", call)
    name = fn["name"]
    args = fn.get("arguments", "")
    if isinstance(args, str):
        args_str = args or "{}"
    else:
        args_str = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    return f'{{"name": "{name}", "arguments": {args_str}}}'


def _split_assistant_message(
    msg: Dict[str, Any],
    *,
    inject_empty_reasoning: bool = False,
) -> List[_Fragment]:
    """Expand one Chat Completions assistant message into ordered fragments.

    Order: reasoning (if non-empty, or forced via inject_empty_reasoning),
    one fragment per tool_call, then a trailing assistant fragment carrying
    `content`. The trailing assistant fragment is skipped when both content
    is empty and at least one other fragment exists.
    """
    fragments: List[_Fragment] = []

    reasoning = msg.get("reasoning_content")
    if reasoning:
        fragments.append(_Fragment("reasoning", reasoning))
    elif inject_empty_reasoning:
        fragments.append(_Fragment("reasoning", ""))

    for call in msg.get("tool_calls") or []:
        fragments.append(_Fragment("tool_call", _render_tool_call_payload(call)))

    content = msg.get("content")
    if content or not fragments:
        fragments.append(_Fragment("assistant", content or ""))
    return fragments


def _expand_messages(
    conversation: List[Dict[str, Any]],
    *,
    inject_empty_reasoning_for_last_turn: bool,
) -> List[_Fragment]:
    """Flatten a Chat Completions conversation into a list of _Fragment."""
    out: List[_Fragment] = []
    n = len(conversation)
    for i, msg in enumerate(conversation):
        role = msg["role"]
        if role == "developer":
            role = "system"  # Harmony developer collapses into system for labeling
        if role == "assistant":
            is_last = i == n - 1
            inject = inject_empty_reasoning_for_last_turn and is_last
            out.extend(_split_assistant_message(msg, inject_empty_reasoning=inject))
        elif role in ROLE_NAMES:
            out.append(_Fragment(role, msg.get("content") or ""))
        else:
            raise ValueError(f"Unknown role in conversation: {role!r}")
    return out


def _strip_prior_reasoning(
    conversation: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop reasoning_content from assistant turns prior to the most recent
    user turn. Mirrors Qwen3's last_query_index rule and OpenAI's Harmony
    guidance on handling reasoning output in subsequent sampling."""
    last_user_idx = -1
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx < 0:
        return conversation

    stripped: List[Dict[str, Any]] = []
    for i, msg in enumerate(conversation):
        if (
            msg.get("role") == "assistant"
            and i < last_user_idx
            and msg.get("reasoning_content") is not None
        ):
            msg = {k: v for k, v in msg.items() if k != "reasoning_content"}
        stripped.append(msg)
    return stripped


def tokenize_conversation(
    conversation: Union[List[Dict[str, Any]], Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    config: TemplateConfig,
    add_generation_prompt: bool = False,
    escape_special_tokens: bool = True,
) -> Tuple[List[int], List[int]]:
    """Render a conversation to (input_ids, role_ids).

    Two rendering modes:

    - Default (escape_special_tokens=True): **per-region encoding**. Message
      content is encoded with `split_special_tokens=True` so user-, tool-,
      or any other externally-sourced content cannot forge template/control
      tokens (e.g., `<|im_start|>` embedded in a user message will NOT
      resolve to its single control token ID; it's broken into regular text
      tokens). This is the only safe configuration for any real deployment
      — sanitizing untrusted input is a baseline security requirement, not
      an optional hardening. It has one side effect: BPE merges do not
      cross content/template boundaries, so some inputs tokenize a few
      tokens differently than HF `apply_chat_template` would (e.g., a
      trailing `\\n` in envelope_start followed by a leading `\\n` in
      content stays as two `\\n` tokens instead of one `\\n\\n` token).

    - escape_special_tokens=False: full-prompt encoded in one shot with
      return_offsets_mapping for role_ids. Matches HF's
      `tokenizer.apply_chat_template` byte-for-byte. Exists **only** as a
      mechanism check to verify that a TemplateConfig mirrors the canonical
      HF template — never use this in production or in evaluations that
      consume externally-sourced content. It lets content forge template
      tokens, which is what a correct real system would sanitize away.

    Args:
        conversation: list of Chat Completions message dicts, or a single
            dict with a "messages" key.
        tokenizer: HF PreTrainedTokenizerBase.
        config: TemplateConfig loaded from YAML or constructed directly.
        add_generation_prompt: if True, append the generation-prompt suffix.
        escape_special_tokens: see above. Always True in production/eval.

    Returns:
        (input_ids, role_ids) with identical lengths. role_ids are ints:
        either a Role enum value (0..5) for content tokens, or NO_ROLE
        (-1) for template / wrapper tokens. Role-conditional embeddings and
        probes should mask NO_ROLE positions.
    """
    if isinstance(conversation, dict):
        conversation = conversation["messages"]

    if config.strip_reasoning_from_prior_turns:
        conversation = _strip_prior_reasoning(conversation)

    fragments = _expand_messages(
        conversation,
        inject_empty_reasoning_for_last_turn=config.inject_empty_reasoning_for_last_turn,
    )

    if escape_special_tokens:
        return _render_per_region(fragments, tokenizer, config, add_generation_prompt)
    return _render_full_prompt(fragments, tokenizer, config, add_generation_prompt)


def _iter_fragments(
    fragments: List[_Fragment], config: TemplateConfig
):
    """Yield (frag, RoleConfig, content_role_id, merged_from_prev, merged_to_next)."""
    for i, frag in enumerate(fragments):
        if frag.role_name not in config.roles:
            raise KeyError(
                f"No RoleConfig for role {frag.role_name!r} in this TemplateConfig "
                f"(available: {sorted(config.roles.keys())})"
            )
        rc = config.roles[frag.role_name]
        content_role_id = int(ROLE_NAMES[frag.role_name])
        prev = fragments[i - 1] if i > 0 else None
        nxt = fragments[i + 1] if i < len(fragments) - 1 else None
        merged_from_prev = (
            prev is not None
            and frag.role_name in config.roles[prev.role_name].merge_with
        )
        merged_to_next = nxt is not None and nxt.role_name in rc.merge_with
        yield frag, rc, content_role_id, merged_from_prev, merged_to_next


def _render_per_region(
    fragments: List[_Fragment],
    tokenizer: PreTrainedTokenizerBase,
    config: TemplateConfig,
    add_generation_prompt: bool,
) -> Tuple[List[int], List[int]]:
    """Encode each region (template fragment or content fragment) in
    isolation, concatenate. Content uses split_special_tokens=True so it
    cannot forge template tokens."""
    input_ids: List[int] = []
    role_ids: List[int] = []

    def enc_template(s: str) -> List[int]:
        if not s:
            return []
        return tokenizer.encode(s, add_special_tokens=False)

    def enc_content(s: str) -> List[int]:
        if not s:
            return []
        return tokenizer.encode(
            s, add_special_tokens=False, split_special_tokens=True
        )

    def emit(ids: List[int], role_id: int) -> None:
        if not ids:
            return
        input_ids.extend(ids)
        role_ids.extend([role_id] * len(ids))

    emit(enc_template(config.bos), NO_ROLE)

    for frag, rc, content_role_id, merged_from_prev, merged_to_next in _iter_fragments(
        fragments, config
    ):
        if not merged_from_prev:
            emit(enc_template(rc.envelope_start), NO_ROLE)
        emit(enc_template(rc.inner_start), NO_ROLE)

        content = frag.content
        if config.strip_whitespace:
            content = content.strip()
        emit(enc_content(content), content_role_id)

        emit(enc_template(rc.inner_end), NO_ROLE)
        if merged_to_next:
            emit(enc_template(rc.merge_separator), NO_ROLE)
        else:
            emit(enc_template(rc.envelope_end), NO_ROLE)

    if add_generation_prompt:
        emit(enc_template(config.generation_prompt_suffix), NO_ROLE)

    assert len(input_ids) == len(role_ids)
    return input_ids, role_ids


def _render_full_prompt(
    fragments: List[_Fragment],
    tokenizer: PreTrainedTokenizerBase,
    config: TemplateConfig,
    add_generation_prompt: bool,
) -> Tuple[List[int], List[int]]:
    """Build the full prompt string + per-region char spans, encode in one
    shot, map role_ids via offset_mapping. Matches HF apply_chat_template
    byte-for-byte when the TemplateConfig is correct. Does NOT sanitize
    content — special-token strings in user input resolve to their control
    token IDs. Use only for parity testing and attack simulation."""
    prompt_parts: List[str] = []
    prompt_len = 0
    content_spans: List[Tuple[int, int, int]] = []

    def emit_text(s: str) -> None:
        nonlocal prompt_len
        if s:
            prompt_parts.append(s)
            prompt_len += len(s)

    def emit_content(s: str, role_id: int) -> None:
        nonlocal prompt_len
        if not s:
            return
        start = prompt_len
        prompt_parts.append(s)
        prompt_len += len(s)
        content_spans.append((start, prompt_len, role_id))

    emit_text(config.bos)

    for frag, rc, content_role_id, merged_from_prev, merged_to_next in _iter_fragments(
        fragments, config
    ):
        if not merged_from_prev:
            emit_text(rc.envelope_start)
        emit_text(rc.inner_start)

        content = frag.content
        if config.strip_whitespace:
            content = content.strip()
        emit_content(content, content_role_id)

        emit_text(rc.inner_end)
        if merged_to_next:
            emit_text(rc.merge_separator)
        else:
            emit_text(rc.envelope_end)

    if add_generation_prompt:
        emit_text(config.generation_prompt_suffix)

    full_prompt = "".join(prompt_parts)
    encoded = tokenizer(
        full_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids: List[int] = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"]

    role_ids = [NO_ROLE] * len(input_ids)
    if content_spans:
        content_spans.sort()
        span_idx = 0
        for tok_idx, (tc_start, tc_end) in enumerate(offsets):
            if tc_start == tc_end:
                continue
            while (
                span_idx < len(content_spans)
                and content_spans[span_idx][1] <= tc_start
            ):
                span_idx += 1
            if span_idx >= len(content_spans):
                break
            s_start, s_end, s_role = content_spans[span_idx]
            if s_start <= tc_start and tc_end <= s_end:
                role_ids[tok_idx] = s_role

    assert len(input_ids) == len(role_ids)
    return input_ids, role_ids
