"""Harmony tokenization for gpt-oss models.

Renders OpenAI Chat Completions conversations into the Harmony token format
and produces (input_ids, role_ids) compatible with the torchllms inference
and intervention pipelines. Delegates rendering to the openai-harmony SDK,
then scans the token stream to assign role IDs.

Role labels (torchllms `Role` enum):
  - system / developer content → Role.SYSTEM
  - user content               → Role.USER
  - assistant final content    → Role.ASSISTANT
  - assistant analysis channel → Role.REASONING
  - assistant commentary calls → Role.TOOL_CALL   (tool-call argument JSON)
  - tool-role content          → Role.TOOL
  - all Harmony template bytes → Role.OTHER       (<|start|>, <|message|>,
                                                   channel/role names, etc.)

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
from typing import Any, Dict, List, Tuple

from openai_harmony import (
    Author,
    Conversation,
    HarmonyEncodingName,
    Message as HarmonyMessage,
    Role as HarmonyRole,
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


# ---- Role / channel → Role enum mapping ---------------------------------

# Decoded role-name → torchllms Role. Harmony's DEVELOPER carries what
# Chat Completions calls "system" (the user-facing instructions).
HARMONY_ROLE_NAME_MAP: Dict[str, Role] = {
    "system": Role.SYSTEM,
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


def tokenize_harmony_conversation(
    conversation: List[Dict[str, Any]],
    tokenizer=None,  # noqa: ARG001 — kept for API parity with tokenize_conversation
    add_generation_prompt: bool = False,
) -> Tuple[List[int], List[int]]:
    """Render a Chat Completions conversation as Harmony tokens + role IDs.

    Args:
        conversation: list of Chat Completions message dicts. Assistant
            messages may include `reasoning_content`, `content`, and
            `tool_calls`. Tool messages may include `name` or `tool_call_id`
            (which is resolved against earlier assistant tool_calls).
        tokenizer: unused; kept so this function's signature mirrors
            `tokenize_conversation`.
        add_generation_prompt: if True, append the assistant-completion
            prefix so the result is ready to feed a generation.

    Returns:
        (input_ids, role_ids) of equal length. See module docstring for the
        Role enum each content region carries.
    """
    enc = _get_encoding()

    tool_name_by_id = _build_tool_name_lookup(conversation)
    messages: List[HarmonyMessage] = []
    for msg in conversation:
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


# ---- CC → Harmony conversion --------------------------------------------


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

    if role in ("system", "developer"):
        return [
            HarmonyMessage.from_role_and_content(
                HarmonyRole.DEVELOPER, msg.get("content") or ""
            )
        ]

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
