"""
Harmony tokenization for gpt-oss models.

Uses the openai-harmony library for conversation rendering and produces
(input_ids, role_ids) pairs compatible with the torchllms inference pipeline.

Role IDs track the ground-truth role of each token, which is critical for
role probe training and activation steering experiments.
"""

from typing import Dict, List, Tuple, Union

from torchllms.messages.tokenization import Role

# Harmony special token IDs
HARMONY_START = 200006  # <|start|>
HARMONY_END = 200007    # <|end|>
HARMONY_MESSAGE = 200008  # <|message|>
HARMONY_CHANNEL = 200005  # <|channel|>
HARMONY_EOT = 199999  # <|endoftext|>

# Harmony role names as they appear in the token stream
HARMONY_ROLE_MAP = {
    "system": Role.SYSTEM,
    "developer": Role.SYSTEM,  # developer is system-level
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL,
}


def tokenize_harmony_conversation(
    conversation: List[Dict[str, str]],
    tokenizer,
    add_generation_prompt: bool = False,
) -> Tuple[List[int], List[int]]:
    """Tokenize a conversation using Harmony format with role ID tracking.

    This renders the conversation in Harmony format and assigns role IDs to
    each token based on the message structure. The Harmony format is:

        <|message|>{role_tokens}<|start|>{content_tokens}<|end|>

    Control tokens (<|message|>, <|start|>, <|end|>) are assigned Role.OTHER.
    Role name tokens and content tokens are assigned the appropriate role.

    Args:
        conversation: List of message dicts with 'role' and 'content' keys.
        tokenizer: The gpt-oss tiktoken tokenizer.
        add_generation_prompt: If True, append assistant start tokens.
    Returns:
        (input_ids, role_ids) — parallel lists of token IDs and role indices.
    """
    input_ids = []
    role_ids = []

    for message in conversation:
        role_str = message["role"]
        content = message["content"]
        role = HARMONY_ROLE_MAP.get(role_str, Role.OTHER)
        role_int = int(role)

        # <|message|> — control token
        input_ids.append(HARMONY_MESSAGE)
        role_ids.append(int(Role.OTHER))

        # Role name tokens (e.g., "system", "user")
        role_tokens = tokenizer.encode(role_str, allowed_special=set())
        input_ids.extend(role_tokens)
        role_ids.extend([int(Role.OTHER)] * len(role_tokens))

        # <|start|> — control token
        input_ids.append(HARMONY_START)
        role_ids.append(int(Role.OTHER))

        # Content tokens — assigned the message's role
        content_tokens = tokenizer.encode(content, allowed_special=set())
        input_ids.extend(content_tokens)
        role_ids.extend([role_int] * len(content_tokens))

        # <|end|> — control token
        input_ids.append(HARMONY_END)
        role_ids.append(int(Role.OTHER))

    if add_generation_prompt:
        # <|message|>assistant<|start|>
        input_ids.append(HARMONY_MESSAGE)
        role_ids.append(int(Role.OTHER))

        asst_tokens = tokenizer.encode("assistant", allowed_special=set())
        input_ids.extend(asst_tokens)
        role_ids.extend([int(Role.OTHER)] * len(asst_tokens))

        input_ids.append(HARMONY_START)
        role_ids.append(int(Role.OTHER))

    assert len(input_ids) == len(role_ids)
    return input_ids, role_ids


def tokenize_harmony_conversation_openai(
    conversation: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> Tuple[List[int], List[int]]:
    """Tokenize using the official openai-harmony library.

    This produces exact Harmony formatting (including system content structure)
    but requires the openai-harmony package. Role IDs are inferred from the
    token stream by tracking <|message|>...<|start|>...<|end|> boundaries.

    Args:
        conversation: List of message dicts with 'role' and 'content' keys.
        add_generation_prompt: If True, render for assistant completion.
    Returns:
        (input_ids, role_ids) — parallel lists.
    """
    from openai_harmony import (
        Conversation,
        Message,
        Role as HarmonyRole,
        TextContent,
        load_harmony_encoding,
        HarmonyEncodingName,
    )

    harmony_role_map = {
        "system": HarmonyRole.SYSTEM,
        "developer": HarmonyRole.DEVELOPER,
        "user": HarmonyRole.USER,
        "assistant": HarmonyRole.ASSISTANT,
        "tool": HarmonyRole.TOOL,
    }

    messages = []
    for msg in conversation:
        role = harmony_role_map[msg["role"]]
        messages.append(Message.from_role_and_content(role, msg["content"]))

    convo = Conversation.from_messages(messages)
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    if add_generation_prompt:
        input_ids = encoding.render_conversation_for_completion(
            convo, HarmonyRole.ASSISTANT
        )
    else:
        input_ids = encoding.render_conversation(convo)

    # Assign role IDs by parsing the token stream structure
    role_ids = _assign_role_ids_from_tokens(input_ids, encoding)

    return list(input_ids), role_ids


def _assign_role_ids_from_tokens(
    tokens: List[int], encoding
) -> List[int]:
    """Infer role IDs from a Harmony token stream.

    Scans for <|message|>...<|start|>...<|end|> boundaries and assigns
    roles based on the role name tokens between <|message|> and <|start|>.
    """
    role_ids = [int(Role.OTHER)] * len(tokens)
    i = 0
    while i < len(tokens):
        if tokens[i] == HARMONY_MESSAGE:
            # Find the <|start|> token
            j = i + 1
            role_name_tokens = []
            while j < len(tokens) and tokens[j] != HARMONY_START:
                role_name_tokens.append(tokens[j])
                j += 1

            # Decode role name
            if role_name_tokens:
                role_name = encoding.decode(role_name_tokens).strip()
                role = HARMONY_ROLE_MAP.get(role_name, Role.OTHER)
            else:
                role = Role.OTHER

            # Skip past <|start|>
            if j < len(tokens) and tokens[j] == HARMONY_START:
                j += 1

            # Content tokens until <|end|>
            while j < len(tokens) and tokens[j] != HARMONY_END:
                role_ids[j] = int(role)
                j += 1

            i = j + 1  # skip <|end|>
        else:
            i += 1

    return role_ids
