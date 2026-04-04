"""
Harmony tokenization for gpt-oss models.

Uses the openai-harmony SDK for correct token rendering and produces
(input_ids, role_ids) pairs compatible with the torchllms inference pipeline.

Role IDs track the ground-truth role of each token, which is critical for
role probe training and activation steering experiments.
"""

from typing import Dict, List, Optional, Tuple

from openai_harmony import (
    Author,
    Conversation,
    HarmonyEncodingName,
    Message as HarmonyMessage,
    Role as HarmonyRole,
    load_harmony_encoding,
)

from torchllms.messages.tokenization import Role

# Harmony special token IDs
HARMONY_START = 200006    # <|start|>
HARMONY_END = 200007      # <|end|>
HARMONY_MESSAGE = 200008  # <|message|>
HARMONY_CHANNEL = 200005  # <|channel|>
HARMONY_EOT = 199999      # <|endoftext|>

# Harmony role names as they appear in the token stream
HARMONY_ROLE_MAP = {
    "system": Role.SYSTEM,
    "developer": Role.SYSTEM,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL,
}

# Probe role labels (4 classes, CoT merged into assistant)
PROBE_ROLES = ["system", "user", "assistant", "tool"]
PROBE_ROLE_TO_INT = {r: i for i, r in enumerate(PROBE_ROLES)}


def _get_encoding():
    """Get the cached HarmonyGptOss encoding."""
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def tokenize_harmony_conversation(
    conversation: List[Dict[str, str]],
    tokenizer=None,
    add_generation_prompt: bool = False,
) -> Tuple[List[int], List[int]]:
    """Tokenize a conversation using the openai-harmony SDK with role ID tracking.

    Renders the conversation in correct Harmony format:
        <|start|>role<|message|>content<|end|>

    Control tokens are assigned Role.OTHER. Content tokens get the message's role.

    Args:
        conversation: List of message dicts with 'role' and 'content' keys.
        tokenizer: Unused (kept for API compatibility). The openai-harmony
            SDK provides its own encoding.
        add_generation_prompt: If True, append assistant start tokens.
    Returns:
        (input_ids, role_ids) — parallel lists of token IDs and role indices.
    """
    enc = _get_encoding()

    harmony_role_map = {
        "system": HarmonyRole.DEVELOPER,
        "developer": HarmonyRole.DEVELOPER,
        "user": HarmonyRole.USER,
        "assistant": HarmonyRole.ASSISTANT,
        "tool": HarmonyRole.TOOL,
    }

    messages = []
    for msg in conversation:
        role = harmony_role_map[msg["role"]]
        messages.append(HarmonyMessage.from_role_and_content(role, msg["content"]))

    convo = Conversation.from_messages(messages)

    if add_generation_prompt:
        input_ids = list(enc.render_conversation_for_completion(
            convo, HarmonyRole.ASSISTANT
        ))
    else:
        input_ids = list(enc.render_conversation(convo))

    role_ids = _assign_role_ids_from_tokens(input_ids, enc)
    return input_ids, role_ids


def _assign_role_ids_from_tokens(
    tokens: List[int], encoding
) -> List[int]:
    """Infer role IDs from a Harmony token stream.

    Scans for <|start|>...<|message|>...<|end|> boundaries and assigns
    roles based on the role name tokens between <|start|> and <|message|>.
    """
    role_ids = [int(Role.OTHER)] * len(tokens)
    i = 0
    while i < len(tokens):
        if tokens[i] == HARMONY_START:
            # Find the <|message|> token
            j = i + 1
            role_name_tokens = []
            while j < len(tokens) and tokens[j] not in (HARMONY_MESSAGE, HARMONY_END):
                if tokens[j] != HARMONY_CHANNEL:
                    role_name_tokens.append(tokens[j])
                else:
                    # Skip channel name tokens too (they're part of the tag)
                    j += 1
                    while j < len(tokens) and tokens[j] not in (HARMONY_MESSAGE, HARMONY_END, HARMONY_CHANNEL):
                        j += 1
                    continue
                j += 1

            # Decode the first token(s) to get the role name
            if role_name_tokens:
                role_name = encoding.decode(role_name_tokens[:1]).strip()
                role = HARMONY_ROLE_MAP.get(role_name, Role.OTHER)
            else:
                role = Role.OTHER

            # Skip past <|message|>
            if j < len(tokens) and tokens[j] == HARMONY_MESSAGE:
                j += 1

            # Content tokens until <|end|>
            while j < len(tokens) and tokens[j] != HARMONY_END:
                role_ids[j] = int(role)
                j += 1

            i = j + 1  # skip <|end|>
        else:
            i += 1

    return role_ids


def _render_simple_message(enc, harmony_role, text, channel=None) -> List[int]:
    """Render a single Harmony message."""
    msg = HarmonyMessage.from_role_and_content(harmony_role, text)
    if channel:
        msg = msg.with_channel(channel)
    return list(enc.render(msg))


def _render_tool_message(enc, text) -> List[int]:
    """Render a tool message in Harmony format."""
    author = Author(role=HarmonyRole.TOOL, name="functions.stub")
    msg = HarmonyMessage.from_author_and_content(author, text)
    msg = msg.with_recipient(HarmonyRole.ASSISTANT).with_channel("commentary")
    return list(enc.render(msg))


def _find_content_mask(tokens: List[int]) -> List[bool]:
    """Find content tokens (between <|message|> and <|end|>) in a token list."""
    mask = [False] * len(tokens)
    in_content = False
    for i, t in enumerate(tokens):
        if t == HARMONY_MESSAGE:
            in_content = True
            continue
        if t == HARMONY_END:
            in_content = False
            continue
        if in_content:
            mask[i] = True
    return mask


def _find_last_content_mask(tokens: List[int]) -> List[bool]:
    """Find content tokens in the LAST message only (last <|message|>...<|end|> span)."""
    # Find all message/end pairs, take the last one
    spans = []
    in_content = False
    start = 0
    for i, t in enumerate(tokens):
        if t == HARMONY_MESSAGE:
            in_content = True
            start = i + 1
        elif t == HARMONY_END and in_content:
            spans.append((start, i))
            in_content = False

    mask = [False] * len(tokens)
    if spans:
        start, end = spans[-1]
        for i in range(start, end):
            mask[i] = True
    return mask


def tokenize_for_probe(
    passage: str,
    role: str,
    filler: Optional[str] = None,
    assistant_variant: str = "cot",
) -> Tuple[List[int], List[bool]]:
    """Tokenize a passage wrapped in a role's Harmony format for probe training.

    For the assistant role, two variants are used (selected by caller):
      - "cot": Content goes in analysis channel. No filler needed.
            <|start|>assistant<|channel|>analysis<|message|>{PASSAGE}<|end|>
      - "final": Filler goes in analysis channel, content in final channel.
            <|start|>assistant<|channel|>analysis<|message|>{FILLER}<|end|>
            <|start|>assistant<|channel|>final<|message|>{PASSAGE}<|end|>
        When this variant is used, all OTHER roles also get filler prepended
        (as a developer message) for positional controls.

    Per Ye et al. 2026, Appendix G.1: this prevents the probe from learning
    position as a shortcut.

    Args:
        passage: The neutral text to wrap.
        role: One of "system", "user", "assistant", "tool".
        filler: Filler text for positional controls. For non-assistant roles,
            prepended as a developer message. For assistant "final" variant,
            placed inside the analysis channel.
        assistant_variant: "cot" or "final" (only used when role=="assistant").
    Returns:
        (input_ids, content_mask) where content_mask[i] is True for tokens
        that are part of the passage content (not role tags or filler).
    """
    enc = _get_encoding()
    tokens = []

    if role == "assistant":
        if assistant_variant == "cot":
            # Content directly in analysis channel, no filler
            role_tokens = _render_simple_message(
                enc, HarmonyRole.ASSISTANT, passage, channel="analysis"
            )
            tokens.extend(role_tokens)
            content_mask = _find_content_mask(role_tokens)
        else:
            # Filler in analysis, content in final
            if filler:
                filler_tokens = _render_simple_message(
                    enc, HarmonyRole.ASSISTANT, filler, channel="analysis"
                )
                tokens.extend(filler_tokens)
            content_tokens = _render_simple_message(
                enc, HarmonyRole.ASSISTANT, passage, channel="final"
            )
            pre_len = len(tokens)
            tokens.extend(content_tokens)
            content_mask = [False] * pre_len + _find_content_mask(content_tokens)
    else:
        # Non-assistant roles: optionally prepend filler as developer message
        if filler:
            filler_tokens = _render_simple_message(
                enc, HarmonyRole.DEVELOPER, filler
            )
            tokens.extend(filler_tokens)

        pre_len = len(tokens)
        if role == "system":
            role_tokens = _render_simple_message(
                enc, HarmonyRole.DEVELOPER, passage
            )
        elif role == "user":
            role_tokens = _render_simple_message(
                enc, HarmonyRole.USER, passage
            )
        elif role == "tool":
            role_tokens = _render_tool_message(enc, passage)
        else:
            raise ValueError(f"Unknown role: {role}")

        tokens.extend(role_tokens)
        content_mask = [False] * pre_len + _find_content_mask(role_tokens)

    assert len(tokens) == len(content_mask)
    return tokens, content_mask
