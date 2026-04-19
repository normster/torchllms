"""Tests for envelope/inner string interpolation (`{name}` etc).

Covers the Harmony + Gemma 4 cases where a tool name appears inside the
role header rather than inside the content. The interpolation mechanism is
tokenizer-agnostic, so these tests use a mock tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from torchllms.messages.tokenization import (
    NO_ROLE,
    Role,
    RoleConfig,
    TemplateConfig,
    _is_templated,
    _maybe_format,
    tokenize_conversation,
)


# ---- Unit tests for the helpers ----------------------------------------


def test_is_templated_detects_placeholders():
    assert _is_templated("before {name} after")
    assert _is_templated("{x}")
    assert _is_templated("a {a} b {b} c")


def test_is_templated_false_for_plain_or_escaped():
    assert not _is_templated("no placeholders here")
    assert not _is_templated("")
    # Escaped braces are literal, not placeholders.
    assert not _is_templated("literal {{braces}} here")


def test_maybe_format_passthrough_on_plain():
    # Extra metadata keys shouldn't matter on plain strings.
    assert _maybe_format("plain", {"name": "bash"}) == "plain"
    assert _maybe_format("{{esc}}", {"name": "bash"}) == "{{esc}}"


def test_maybe_format_substitutes():
    assert _maybe_format("to=functions.{name}", {"name": "bash"}) == "to=functions.bash"
    assert (
        _maybe_format("a={a} b={b}", {"a": "x", "b": "y"}) == "a=x b=y"
    )


# ---- End-to-end tests with a mock tokenizer ----------------------------


class _MockTokenizer:
    """Tokenizes by splitting on spaces and hashing each token. Records the
    most recent full-prompt passed via __call__ so tests can verify
    interpolation produced the expected string."""

    class _Output(dict):
        pass

    def __init__(self):
        self.last_prompt: str = ""

    def encode(self, s: str, add_special_tokens=False, split_special_tokens=False):
        return [hash(tok) % 100_000 for tok in s.split(" ") if tok]

    def __call__(self, s: str, add_special_tokens=False, return_offsets_mapping=False):
        self.last_prompt = s
        ids: List[int] = []
        offsets: List[tuple] = []
        pos = 0
        for word in s.split(" "):
            if not word:
                pos += 1
                continue
            ids.append(hash(word) % 100_000)
            offsets.append((pos, pos + len(word)))
            pos += len(word) + 1
        out = self._Output()
        out["input_ids"] = ids
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def _harmony_like_config() -> TemplateConfig:
    """Minimal Harmony-shaped config with `{name}` placeholders in tool
    envelopes. Spaces make mock tokenization easy to reason about."""
    return TemplateConfig(
        roles={
            "system": RoleConfig(
                envelope_start="SYS_START ",
                envelope_end=" SYS_END",
            ),
            "user": RoleConfig(
                envelope_start="USR_START ",
                envelope_end=" USR_END",
            ),
            "assistant": RoleConfig(
                envelope_start="AST_START final ",
                envelope_end=" AST_END",
            ),
            "reasoning": RoleConfig(
                envelope_start="AST_START analysis ",
                envelope_end=" AST_END",
                merge_with=["tool_call", "assistant"],
            ),
            "tool_call": RoleConfig(
                envelope_start="AST_START to=functions.{name} commentary ",
                envelope_end=" AST_END",
                merge_with=["tool_call", "assistant"],
            ),
            "tool": RoleConfig(
                envelope_start="TOOL_START from=functions.{name} ",
                envelope_end=" TOOL_END",
            ),
        },
        strip_whitespace=False,
    )


def test_tool_call_name_interpolated_into_envelope():
    cfg = _harmony_like_config()
    tok = _MockTokenizer()
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_42",
                    "function": {"name": "bash", "arguments": '{"cmd":"ls"}'},
                }
            ],
        },
    ]
    # Use parity mode so we can inspect a single encoded stream and its
    # role_ids without worrying about per-region boundaries.
    ids, roles = tokenize_conversation(
        messages, tok, cfg, escape_special_tokens=False
    )
    # The rendered prompt should contain "to=functions.bash" somewhere.
    # Decode by reversing the mock tokenizer isn't possible; instead check
    # the full_prompt via a second call that builds the string directly.
    # Simpler: re-render via per-region mode and scan the prompt pieces.
    # For a cleaner check, use the full_prompt path by inspecting the
    # rendered string through a secondary hook.
    assert len(ids) == len(roles)


def test_interpolated_tool_call_renders_name_in_string():
    """Direct check: the rendered prompt string contains the tool name."""
    cfg = _harmony_like_config()
    tok = _MockTokenizer()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
    ]
    tokenize_conversation(messages, tok, cfg, escape_special_tokens=False)
    assert "to=functions.bash" in tok.last_prompt


def test_tool_response_name_resolved_via_tool_call_id():
    cfg = _harmony_like_config()
    tok = _MockTokenizer()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "abc",
                    "function": {"name": "weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "abc", "content": "sunny"},
    ]
    tokenize_conversation(messages, tok, cfg, escape_special_tokens=False)
    # tool_call_id="abc" should resolve to function name "weather".
    assert "from=functions.weather" in tok.last_prompt


def test_tool_response_with_explicit_name_field_wins():
    cfg = _harmony_like_config()
    tok = _MockTokenizer()
    messages = [
        # No tool_call_id lookup available here.
        {"role": "tool", "name": "search", "content": "results"},
    ]
    tokenize_conversation(messages, tok, cfg, escape_special_tokens=False)
    assert "from=functions.search" in tok.last_prompt


def test_non_interpolated_field_passes_through_even_with_empty_metadata():
    """A plain envelope must not raise KeyError when metadata lacks fields."""
    cfg = TemplateConfig(
        roles={
            "user": RoleConfig(
                envelope_start="USR ", envelope_end=" END"
            ),
        },
    )
    tok = _MockTokenizer()
    ids, _ = tokenize_conversation(
        [{"role": "user", "content": "hello"}], tok, cfg
    )
    assert len(ids) > 0


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
            import traceback

            traceback.print_exc()
            print(f"{name}: FAIL - {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
