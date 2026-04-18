"""
Harmony tokenization for gpt-oss models.

Uses the openai-harmony SDK for correct token rendering and produces
(input_ids, role_ids) pairs compatible with the torchllms inference pipeline.
"""

from typing import Dict, List, Tuple

from openai_harmony import (
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


