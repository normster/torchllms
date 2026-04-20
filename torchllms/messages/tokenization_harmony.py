"""Harmony tokenization for gpt-oss models.

Renders OpenAI Chat Completions conversations into the Harmony token format
and produces (input_ids, role_ids) compatible with the torchllms inference
and intervention pipelines. Delegates rendering to the openai-harmony SDK,
then scans the token stream to assign role IDs.

Role labels (torchllms `Role` enum):
  - Harmony system block content → Role.OTHER (model-contract metadata:
                                   identity, knowledge cutoff, reasoning
                                   effort, channel declarations, builtin
                                   tool namespaces)
  - CC system / developer content → Role.SYSTEM (rendered into Harmony
                                    DEVELOPER block as `# Instructions`
                                    and/or `# Tools`)
  - user content                 → Role.USER
  - assistant final content      → Role.ASSISTANT
  - assistant analysis channel   → Role.REASONING
  - assistant commentary calls   → Role.TOOL_CALL (tool-call argument JSON)
  - tool-role content            → Role.TOOL
  - all Harmony template bytes   → Role.OTHER (<|start|>, <|message|>, etc.)

A Harmony system block is always emitted (channel declarations matter).
Each CC `system` / `developer` message becomes its own Harmony DEVELOPER
block (one-to-one). Function-tool schemas are attached to the first such
block, or emitted as a standalone DEVELOPER block if no CC
system/developer message is present.

One Chat Completions assistant message may carry `reasoning_content`,
`tool_calls`, and `content` simultaneously. It expands into up to:
  - 1 analysis-channel Harmony message (reasoning)
  - N commentary-channel messages (one per tool_call, recipient +
    content_type set so the call JSON goes through with a `<|constrain|>`
    hint)
  - 1 final-channel message (visible response)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message as HarmonyMessage,
    ReasoningEffort,
    Role as HarmonyRole,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)

from torchllms.messages.tokenization import Role


# ---- Harmony special token IDs ------------------------------------------

HARMONY_EOT = 199999        # <|endoftext|>
HARMONY_RETURN = 200002     # <|return|>   (model-emitted end-of-turn)
HARMONY_CONSTRAIN = 200003  # <|constrain|>
HARMONY_CHANNEL = 200005    # <|channel|>
HARMONY_START = 200006      # <|start|>
HARMONY_END = 200007        # <|end|>
HARMONY_MESSAGE = 200008    # <|message|>
HARMONY_CALL = 200012       # <|call|>     (end-of-assistant-tool-call)

# Every per-message terminator we might see in a rendered or generated
# transcript. `<|end|>` is the default, `<|call|>` closes a tool call,
# `<|return|>` may appear in generations of a completed final response.
_MESSAGE_END_TOKENS = (HARMONY_END, HARMONY_CALL, HARMONY_RETURN)


# ---- Option validation sets ---------------------------------------------

ALLOWED_REASONING_EFFORTS = frozenset({"low", "medium", "high"})
ALLOWED_CHANNELS = frozenset({"analysis", "commentary", "final"})
ALLOWED_BUILTIN_TOOLS = frozenset({"browser", "python"})


# ---- Role / channel → Role enum mapping ---------------------------------

# Decoded role-name → torchllms Role. The Harmony SYSTEM block carries
# model-contract scaffolding (channels, effort, dates) — not user-facing
# instructions — so its content tokens map to Role.OTHER. The DEVELOPER
# block carries what CC calls `system`/`developer` (instructions + tool
# schemas), so those tokens map to Role.SYSTEM.
HARMONY_ROLE_NAME_MAP: Dict[str, Role] = {
    "system": Role.OTHER,
    "developer": Role.SYSTEM,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL,
}

_ASSISTANT_CHANNEL_ROLE: Dict[str, Role] = {
    "analysis": Role.REASONING,
    "commentary": Role.TOOL_CALL,
    "final": Role.ASSISTANT,
}


def _get_encoding():
    """Cached HarmonyGptOss encoding."""
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


# ---- Public API ---------------------------------------------------------

CCToolDict = Dict[str, Any]


def tokenize_harmony_conversation(
    conversation: List[Dict[str, Any]],
    tokenizer=None,  # noqa: ARG001 — kept for API parity with tokenize_conversation
    add_generation_prompt: bool = False,
    *,
    reasoning_effort: Optional[str] = "low",
    valid_channels: Sequence[str] = ("analysis", "commentary", "final"),
    current_date: Optional[str] = None,
    knowledge_cutoff: Optional[str] = None,
    system_identity: Optional[str] = None,
    tools: Optional[Sequence[CCToolDict]] = None,
    builtin_tools: Sequence[str] = (),
) -> Tuple[List[int], List[int]]:
    """Render a Chat Completions conversation as Harmony tokens + role IDs.

    Args:
        conversation: list of Chat Completions message dicts. `system` and
            `developer` messages must appear at the start (any order) and
            each becomes its own Harmony DEVELOPER block rendered as
            `# Instructions`. Assistant messages may include
            `reasoning_content`, `content`, and `tool_calls`. Tool messages
            may include `name` or `tool_call_id` (resolved against earlier
            assistant tool_calls).
        tokenizer: unused; kept so this function's signature mirrors
            `tokenize_conversation`.
        add_generation_prompt: if True, append the assistant-completion
            prefix so the result is ready to feed a generation.
        reasoning_effort: one of "low"/"medium"/"high", or None to leave the
            SDK default (MEDIUM). Populates the SYSTEM block's
            `Reasoning: ...` line.
        valid_channels: which Harmony channels to declare as valid. An
            empty tuple suppresses the `# Valid channels: ...` line.
        current_date: ISO date string for `Current date: ...`. None omits
            the line (the SDK does not inject today's date).
        knowledge_cutoff: ISO date / month for `Knowledge cutoff: ...`.
            None leaves the SDK default (2024-06).
        system_identity: override for the default
            "You are ChatGPT..." identity string. None leaves the SDK
            default.
        tools: optional list of Chat Completions tool dicts. Each entry may
            be in canonical `{"type": "function", "function": {...}}` form
            or flat `{"name", "description", "parameters"}`. Attached to
            the first DEVELOPER block (or a standalone DEVELOPER block if
            the conversation has no CC system/developer messages).
        builtin_tools: subset of {"browser", "python"}. Each flips on the
            matching Harmony built-in tool in the SYSTEM block.

    Returns:
        (input_ids, role_ids) of equal length. See module docstring for the
        Role enum each content region carries.
    """
    _validate_options(reasoning_effort, valid_channels, builtin_tools)
    enc = _get_encoding()

    prefix_count = _count_prefix_messages(conversation)
    prefix_msgs = conversation[:prefix_count]
    tool_descs = _build_tool_descriptions(tools) if tools else []

    messages: List[HarmonyMessage] = []
    messages.append(
        HarmonyMessage.from_role_and_content(
            HarmonyRole.SYSTEM,
            _build_system_content(
                reasoning_effort=reasoning_effort,
                valid_channels=valid_channels,
                current_date=current_date,
                knowledge_cutoff=knowledge_cutoff,
                system_identity=system_identity,
                builtin_tools=builtin_tools,
            ),
        )
    )
    messages.extend(_build_developer_messages(prefix_msgs, tool_descs))

    tool_name_by_id = _build_tool_name_lookup(conversation)
    for msg in conversation[prefix_count:]:
        messages.extend(_cc_to_harmony(msg, tool_name_by_id))

    convo = Conversation.from_messages(messages)
    if add_generation_prompt:
        input_ids = list(
            enc.render_conversation_for_completion(convo, HarmonyRole.ASSISTANT)
        )
    else:
        input_ids = list(enc.render_conversation(convo))

    role_ids = _assign_role_ids_from_tokens(input_ids, enc)
    return input_ids, role_ids


# ---- Option validation --------------------------------------------------


def _validate_options(
    reasoning_effort: Optional[str],
    valid_channels: Sequence[str],
    builtin_tools: Sequence[str],
) -> None:
    if reasoning_effort is not None and reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort={reasoning_effort!r}; allowed: "
            f"{sorted(ALLOWED_REASONING_EFFORTS)} or None"
        )
    bad_ch = set(valid_channels) - ALLOWED_CHANNELS
    if bad_ch:
        raise ValueError(
            f"valid_channels contains unsupported names: {sorted(bad_ch)}; "
            f"allowed: {sorted(ALLOWED_CHANNELS)}"
        )
    bad_bt = set(builtin_tools) - ALLOWED_BUILTIN_TOOLS
    if bad_bt:
        raise ValueError(
            f"builtin_tools contains unsupported names: {sorted(bad_bt)}; "
            f"allowed: {sorted(ALLOWED_BUILTIN_TOOLS)}"
        )


# ---- Prefix (system/developer) handling ---------------------------------


def _count_prefix_messages(conversation: List[Dict[str, Any]]) -> int:
    """Return N such that conversation[:N] are all system/developer.

    Raises if system/developer appears later in the conversation.
    """
    n = 0
    for msg in conversation:
        if msg.get("role") in ("system", "developer"):
            n += 1
        else:
            break
    for msg in conversation[n:]:
        if msg.get("role") in ("system", "developer"):
            raise ValueError(
                "system/developer messages must appear at the start of the "
                "conversation (collapsed into a single Harmony DEVELOPER block)"
            )
    return n


# ---- SystemContent / DeveloperContent builders --------------------------


def _build_system_content(
    *,
    reasoning_effort: Optional[str],
    valid_channels: Sequence[str],
    current_date: Optional[str],
    knowledge_cutoff: Optional[str],
    system_identity: Optional[str],
    builtin_tools: Sequence[str],
) -> SystemContent:
    sc = SystemContent.new()
    if reasoning_effort is not None:
        sc = sc.with_reasoning_effort(ReasoningEffort[reasoning_effort.upper()])
    if valid_channels is not None:
        sc = sc.with_required_channels(list(valid_channels))
    if current_date is not None:
        sc = sc.with_conversation_start_date(current_date)
    if knowledge_cutoff is not None:
        sc = sc.with_knowledge_cutoff(knowledge_cutoff)
    if system_identity is not None:
        sc = sc.with_model_identity(system_identity)
    for bt in builtin_tools:
        if bt == "browser":
            sc = sc.with_browser_tool()
        elif bt == "python":
            sc = sc.with_python_tool()
    return sc


def _build_developer_messages(
    prefix_msgs: List[Dict[str, Any]],
    tool_descs: List[ToolDescription],
) -> List[HarmonyMessage]:
    """One Harmony DEVELOPER block per CC system/developer message.

    Tool schemas attach to the first block (so they share it with the
    first instruction). If there are no CC prefix messages but tools are
    present, emit a single tools-only DEVELOPER block.
    """
    out: List[HarmonyMessage] = []
    prefix_with_content = [m for m in prefix_msgs if (m.get("content") or "")]

    if not prefix_with_content:
        if tool_descs:
            dc = DeveloperContent.new().with_function_tools(tool_descs)
            out.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, dc))
        return out

    for i, msg in enumerate(prefix_with_content):
        dc = DeveloperContent.new().with_instructions(msg["content"])
        if i == 0 and tool_descs:
            dc = dc.with_function_tools(tool_descs)
        out.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, dc))
    return out


def _build_tool_descriptions(
    tools: Sequence[CCToolDict],
) -> List[ToolDescription]:
    """Turn CC-format tool dicts into Harmony ToolDescription objects.

    Tolerates both canonical (`{"type": "function", "function": {...}}`)
    and flat (`{"name", "description", "parameters"}`) shapes.
    """
    out: List[ToolDescription] = []
    for entry in tools:
        fn = entry.get("function", entry)
        name = fn.get("name")
        if not name:
            raise ValueError("tool entry missing `name`")
        out.append(
            ToolDescription.new(
                name,
                fn.get("description", ""),
                parameters=fn.get("parameters"),
            )
        )
    return out


# ---- CC → Harmony conversion (user / assistant / tool only) -------------


def _build_tool_name_lookup(conversation: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map tool_call_id → function name across the conversation so tool-role
    messages without an explicit `name` can still recover their author."""
    out: Dict[str, str] = {}
    for msg in conversation:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            cid = call.get("id")
            if not cid:
                continue
            fn = call.get("function", call)
            out[cid] = fn.get("name", "")
    return out


def _cc_to_harmony(
    msg: Dict[str, Any],
    tool_name_by_id: Dict[str, str],
) -> List[HarmonyMessage]:
    role = msg["role"]

    if role == "user":
        return [
            HarmonyMessage.from_role_and_content(
                HarmonyRole.USER, msg.get("content") or ""
            )
        ]

    if role == "tool":
        name = msg.get("name") or tool_name_by_id.get(msg.get("tool_call_id", ""), "")
        if not name:
            raise ValueError(
                "tool-role message needs `name` (or a tool_call_id that "
                "resolves against a preceding assistant tool_call)"
            )
        author = Author.new(HarmonyRole.TOOL, name)
        return [
            HarmonyMessage.from_author_and_content(author, msg.get("content") or "")
            .with_recipient("assistant")
            .with_channel("commentary")
        ]

    if role == "assistant":
        out: List[HarmonyMessage] = []
        reasoning = msg.get("reasoning_content")
        if reasoning:
            out.append(
                HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, reasoning)
                .with_channel("analysis")
            )
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", call)
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            out.append(
                HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, args)
                .with_channel("commentary")
                .with_recipient(f"functions.{name}")
                .with_content_type("<|constrain|>json")
            )
        content = msg.get("content")
        # Emit a final-channel message when there is visible content, or
        # when the assistant has no reasoning/tool_calls at all — `""` on
        # its own is a legitimate placeholder turn.
        if content or not out:
            out.append(
                HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, content or "")
                .with_channel("final")
            )
        return out

    raise ValueError(f"Unknown role in conversation: {role!r}")


# ---- Role ID scanning ---------------------------------------------------


def _assign_role_ids_from_tokens(
    tokens: List[int], encoding
) -> List[int]:
    """Walk the token stream and tag each position with a Role int.

    Template / header tokens get Role.OTHER. Content tokens (between
    `<|message|>` and the next message terminator) get the role resolved
    from the preceding header + channel.
    """
    role_ids = [int(Role.OTHER)] * len(tokens)
    i = 0
    n = len(tokens)

    while i < n:
        if tokens[i] != HARMONY_START:
            i += 1
            continue

        j = i + 1
        header_tokens: List[int] = []
        channel_tokens: List[int] = []
        in_channel = False
        header_terminators = (HARMONY_MESSAGE,) + _MESSAGE_END_TOKENS

        while j < n and tokens[j] not in header_terminators:
            t = tokens[j]
            if t == HARMONY_CHANNEL:
                in_channel = True
            elif t == HARMONY_CONSTRAIN:
                # Stop collecting channel-name tokens; the trailing constrain
                # payload (e.g. "json") is header decoration.
                in_channel = False
            elif in_channel:
                channel_tokens.append(t)
            else:
                header_tokens.append(t)
            j += 1

        role = _resolve_role(header_tokens, channel_tokens, encoding)

        if j < n and tokens[j] == HARMONY_MESSAGE:
            j += 1
            while j < n and tokens[j] not in _MESSAGE_END_TOKENS:
                role_ids[j] = int(role)
                j += 1

        i = j + 1  # step past the terminator (or header-without-content)

    return role_ids


def _resolve_role(
    header_tokens: List[int],
    channel_tokens: List[int],
    encoding,
) -> Role:
    if not header_tokens:
        return Role.OTHER
    first = encoding.decode(header_tokens[:1]).strip()
    mapped = HARMONY_ROLE_NAME_MAP.get(first)
    if mapped is None:
        # Not a canonical role name → tool author (e.g. "functions.bash").
        return Role.TOOL
    if mapped is not Role.ASSISTANT or not channel_tokens:
        return mapped
    # Refine assistant role by channel name. `channel_tokens` may carry
    # trailing constrain bytes like "commentary json"; take the first word.
    channel_text = encoding.decode(channel_tokens).strip()
    channel = channel_text.split(maxsplit=1)[0] if channel_text else ""
    return _ASSISTANT_CHANNEL_ROLE.get(channel, Role.ASSISTANT)
