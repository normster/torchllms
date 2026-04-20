"""Parity and role-ID tests for the Harmony tokenization path.

Mirrors `test_tokenization_qwen3.py`:
  - byte parity against openai-harmony's canonical renderer, one fixture
    per conversation shape (user-only, sys+user+asst, analysis/commentary/
    final channels, tool responses, full multi-channel turn);
  - role-ID content checks (each Chat Completions role lands on the right
    Role enum int; Harmony template bytes are Role.OTHER).

Requires the openai-harmony package (already a repo-level dependency).
"""

from __future__ import annotations

import json

from openai_harmony import (
    Author,
    Conversation,
    HarmonyEncodingName,
    Message as HarmonyMessage,
    Role as HarmonyRole,
    load_harmony_encoding,
)

from torchllms.messages.tokenization import Role
from torchllms.messages.tokenization_harmony import (
    HARMONY_CALL,
    HARMONY_CHANNEL,
    HARMONY_CONSTRAIN,
    HARMONY_END,
    HARMONY_EOT,
    HARMONY_MESSAGE,
    HARMONY_RETURN,
    HARMONY_START,
    tokenize_harmony_conversation,
)


# ---------------------------------------------------------------------------
# Fixtures: each returns (cc_messages, harmony_messages) so we can compare
# our tokenize_harmony_conversation output against the canonical rendering
# of a hand-built Harmony conversation.
# ---------------------------------------------------------------------------


def _fx_user_only():
    cc = [{"role": "user", "content": "Hi"}]
    hm = [HarmonyMessage.from_role_and_content(HarmonyRole.USER, "Hi")]
    return cc, hm


def _fx_system_user():
    cc = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]
    hm = [
        HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, "Be helpful."),
        HarmonyMessage.from_role_and_content(HarmonyRole.USER, "Hi"),
    ]
    return cc, hm


def _fx_developer_user():
    cc = [
        {"role": "developer", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]
    hm = [
        HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, "Be helpful."),
        HarmonyMessage.from_role_and_content(HarmonyRole.USER, "Hi"),
    ]
    return cc, hm


def _fx_sys_user_asst_final():
    cc = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    hm = [
        HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, "Be helpful."),
        HarmonyMessage.from_role_and_content(HarmonyRole.USER, "Hi"),
        HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, "Hello!")
            .with_channel("final"),
    ]
    return cc, hm


def _fx_asst_reasoning_then_final():
    cc = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "reasoning_content": "basic arithmetic.",
            "content": "4.",
        },
    ]
    hm = [
        HarmonyMessage.from_role_and_content(HarmonyRole.USER, "What is 2+2?"),
        HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, "basic arithmetic.")
            .with_channel("analysis"),
        HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, "4.")
            .with_channel("final"),
    ]
    return cc, hm


def _fx_tool_call_and_result():
    cc = [
        {"role": "user", "content": "List /tmp"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Use bash",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"ls /tmp"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file1\nfile2"},
        {"role": "assistant", "content": "Two files."},
    ]
    author_tool = Author.new(HarmonyRole.TOOL, "bash")
    hm = [
        HarmonyMessage.from_role_and_content(HarmonyRole.USER, "List /tmp"),
        HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, "Use bash")
            .with_channel("analysis"),
        HarmonyMessage.from_role_and_content(
            HarmonyRole.ASSISTANT, '{"command":"ls /tmp"}'
        )
            .with_channel("commentary")
            .with_recipient("functions.bash")
            .with_content_type("<|constrain|>json"),
        HarmonyMessage.from_author_and_content(author_tool, "file1\nfile2")
            .with_recipient("assistant")
            .with_channel("commentary"),
        HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, "Two files.")
            .with_channel("final"),
    ]
    return cc, hm


def _fx_asst_reasoning_only_last_turn():
    """Mid-generation trace: reasoning is the last assistant turn with no
    visible content yet. An empty final-channel turn would still trigger
    Harmony's strip-prior-analysis rule, so we emit analysis alone and the
    reasoning bytes survive the render."""
    cc = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "reasoning_content": "REASONING", "content": ""},
    ]
    hm = [
        HarmonyMessage.from_role_and_content(HarmonyRole.USER, "Q"),
        HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, "REASONING")
            .with_channel("analysis"),
    ]
    return cc, hm


def _fx_multi_turn():
    cc = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
    ]
    hm = [
        HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, "Be concise."),
        HarmonyMessage.from_role_and_content(HarmonyRole.USER, "Hi"),
        HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, "Hello")
            .with_channel("final"),
        HarmonyMessage.from_role_and_content(HarmonyRole.USER, "How are you?"),
    ]
    return cc, hm


FIXTURES = {
    "user_only": _fx_user_only,
    "system_user": _fx_system_user,
    "developer_user": _fx_developer_user,
    "sys_user_asst_final": _fx_sys_user_asst_final,
    "asst_reasoning_then_final": _fx_asst_reasoning_then_final,
    "asst_reasoning_only_last_turn": _fx_asst_reasoning_only_last_turn,
    "tool_call_and_result": _fx_tool_call_and_result,
    "multi_turn": _fx_multi_turn,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical(enc, harmony_msgs, add_generation_prompt):
    convo = Conversation.from_messages(harmony_msgs)
    if add_generation_prompt:
        return list(enc.render_conversation_for_completion(convo, HarmonyRole.ASSISTANT))
    return list(enc.render_conversation(convo))


def _assert_parity(enc, cc_msgs, harmony_msgs, add_generation_prompt):
    ours, _ = tokenize_harmony_conversation(
        cc_msgs, add_generation_prompt=add_generation_prompt
    )
    canonical = _canonical(enc, harmony_msgs, add_generation_prompt)
    if ours == canonical:
        return
    for i, (a, b) in enumerate(zip(ours, canonical)):
        if a != b:
            ctx_lo = max(0, i - 3)
            raise AssertionError(
                f"Parity mismatch at token {i}: ours={a} canonical={b}\n"
                f"  ours ctx: {ours[ctx_lo:i + 5]}\n"
                f"  canon    : {canonical[ctx_lo:i + 5]}\n"
                f"  lengths: ours={len(ours)} canonical={len(canonical)}"
            )
    raise AssertionError(
        f"Length mismatch: ours={len(ours)} canonical={len(canonical)}\n"
        f"  ours tail: {ours[-5:]}\n"
        f"  canon tail: {canonical[-5:]}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_harmony_byte_parity():
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    for name, fn in FIXTURES.items():
        cc, hm = fn()
        for add_gp in (False, True):
            try:
                _assert_parity(enc, cc, hm, add_gp)
            except AssertionError as e:
                raise AssertionError(f"fixture={name} add_gp={add_gp}: {e}") from e


def test_role_ids_basic():
    """SYSTEM / USER / ASSISTANT content carry their respective role ints;
    every Harmony special token gets Role.OTHER."""
    cc = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    ids, roles = tokenize_harmony_conversation(cc, add_generation_prompt=False)
    assert len(ids) == len(roles)

    specials = {
        HARMONY_START, HARMONY_END, HARMONY_MESSAGE, HARMONY_CHANNEL,
        HARMONY_CALL, HARMONY_RETURN, HARMONY_CONSTRAIN, HARMONY_EOT,
    }
    for i, (t, r) in enumerate(zip(ids, roles)):
        if t in specials:
            assert r == int(Role.OTHER), (
                f"special token {t} at {i} has role {r}, expected OTHER"
            )

    role_ints = set(roles)
    assert int(Role.SYSTEM) in role_ints
    assert int(Role.USER) in role_ints
    assert int(Role.ASSISTANT) in role_ints


def test_role_ids_reasoning_vs_final():
    """analysis-channel assistant tokens get Role.REASONING; final-channel
    tokens get Role.ASSISTANT.

    Note: openai-harmony drops prior-turn analysis bytes in favor of the
    subsequent final turn (the SDK's built-in "strip prior reasoning" rule),
    so we exercise reasoning labeling on a conversation whose most recent
    assistant turn is reasoning-only (e.g., a mid-generation trace).
    """
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Reasoning-only turn (mid-generation) — analysis survives.
    cc_mid = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "reasoning_content": "REASONING", "content": ""},
    ]
    ids, roles = tokenize_harmony_conversation(cc_mid, add_generation_prompt=False)
    assert _decode_role(enc, ids, roles, Role.REASONING) == "REASONING"

    # Final turn — reasoning from this turn is dropped by Harmony, but the
    # final content still lands on Role.ASSISTANT.
    cc_done = [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "reasoning_content": "hidden reasoning",
            "content": "ANSWER",
        },
    ]
    ids, roles = tokenize_harmony_conversation(cc_done, add_generation_prompt=False)
    assert _decode_role(enc, ids, roles, Role.ASSISTANT) == "ANSWER"
    # Reasoning got stripped by the SDK — no REASONING-labeled tokens.
    assert int(Role.REASONING) not in set(roles)


def test_role_ids_tool_and_tool_call():
    """Tool-role content → Role.TOOL. Assistant commentary-channel (tool_call
    arguments) → Role.TOOL_CALL."""
    cc = [
        {"role": "user", "content": "run ls"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"ls"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file1\nfile2"},
    ]
    ids, roles = tokenize_harmony_conversation(cc, add_generation_prompt=False)
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    tool_call_text = _decode_role(enc, ids, roles, Role.TOOL_CALL)
    assert tool_call_text == '{"command":"ls"}', tool_call_text

    tool_text = _decode_role(enc, ids, roles, Role.TOOL)
    assert tool_text == "file1\nfile2", tool_text


def test_role_ids_tool_name_from_message():
    """tool-role messages accept an explicit `name` field (no tool_call_id
    needed)."""
    cc = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "name": "bash", "content": "T"},
    ]
    ids, roles = tokenize_harmony_conversation(cc, add_generation_prompt=False)
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    assert _decode_role(enc, ids, roles, Role.TOOL) == "T"


def test_tool_without_name_raises():
    cc = [
        {"role": "user", "content": "q"},
        {"role": "tool", "content": "T"},  # no name, no tool_call_id
    ]
    try:
        tokenize_harmony_conversation(cc)
    except ValueError as e:
        assert "name" in str(e)
        return
    raise AssertionError("expected ValueError for nameless tool message")


def _decode_role(enc, ids, roles, target_role: Role) -> str:
    target = int(target_role)
    toks = [t for t, r in zip(ids, roles) if r == target]
    return enc.decode(toks)


# ---------------------------------------------------------------------------
# Runner so the file is executable like test_tokenization_qwen3.py.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"{name}: ok")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"{name}: FAIL - {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
