"""Per-turn prefill transition builders for the multi-turn W3 throughput
workload.

Each "turn" in W3 corresponds to one generation point. Between turns,
the cache carries the raw decoded tokens from the prior turn (garbage,
untrimmed). At the start of the next turn we prefill a transition that:

  1. Closes the prior assistant envelope (``<|im_end|>`` + newline).
  2. Opens the next user/tool envelope + content.
  3. Primes the assistant response (role-primer).

For turn 1 there's no prior to close — the transition is the full
system prompt + first user/tool + assistant primer.

Two families handled:

  - Qwen3 ChatML nothink — ChatML markers + ``<think>\\n\\n</think>\\n\\n``
    injected into the assistant primer (nothink-mode convention).
  - gpt-oss Harmony — Harmony envelope via openai_harmony's system/user
    primitives.

Returns ``(ids, role_ids)`` where role_ids labels content tokens with
``Role.USER`` / ``Role.TOOL`` / ``Role.SYSTEM`` and all envelope markers
with ``NO_ROLE`` (``-1``). These are the labels the intervention API's
role-filter mask compares against.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

from torchllms.messages import Role


# NO_ROLE convention matches torchllms.messages.tokenization.NO_ROLE.
_NO = -1


# ---------------------------------------------------------------------------
# Qwen3 ChatML (nothink variant)
# ---------------------------------------------------------------------------


def build_qwen3_turn(
    tokenizer,
    *,
    kind: str,           # "user" | "tool"
    content: str,
    is_first_turn: bool,
    system: str = "",
) -> Tuple[List[int], List[int]]:
    """Build per-turn incremental tokens + role_ids for Qwen3 ChatML nothink.

    Non-first turn: opens with ``<|im_end|>\\n<|im_start|>user\\n`` (closes
    the prior assistant envelope that the model's decode did NOT close
    with a real ``<|im_end|>``).

    First turn: opens with system envelope + user envelope.

    All turns end with the assistant primer ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``.
    """
    assert kind in ("user", "tool"), kind

    # Build the three segments separately so we can label roles correctly.
    # Segment 1 (pre_content): envelope markers before the content string.
    # Segment 2 (content): the actual payload.
    # Segment 3 (post_content): envelope closer + next-turn assistant primer.
    if is_first_turn:
        pre_sys = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
        if kind == "tool":
            pre_sys += "<tool_response>\n"
    else:
        pre_sys = "<|im_end|>\n<|im_start|>user\n"
        if kind == "tool":
            pre_sys += "<tool_response>\n"

    post = ""
    if kind == "tool":
        post += "\n</tool_response>"
    post += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    # Encode each segment.
    pre_ids = tokenizer.encode(pre_sys, add_special_tokens=False)
    content_ids = tokenizer.encode(content, add_special_tokens=False)
    post_ids = tokenizer.encode(post, add_special_tokens=False)

    # Role labels.
    content_role = int(Role.USER) if kind == "user" else int(Role.TOOL)
    roles: List[int] = []
    roles.extend([_NO] * len(pre_ids))
    # System content is labeled Role.SYSTEM by the production tokenizer;
    # the intervention role filter's ``intervened_roles`` set never
    # targets SYSTEM in the bench, so the distinction doesn't affect
    # measurement. Label system content as SYSTEM for fidelity if first
    # turn; otherwise content is user/tool.
    if is_first_turn and system:
        # Re-tokenize to split sys vs user segments exactly.
        sys_pre = "<|im_start|>system\n"
        sys_body = system
        sys_post_user_open = f"<|im_end|>\n<|im_start|>user\n"
        if kind == "tool":
            sys_post_user_open += "<tool_response>\n"
        sys_pre_ids = tokenizer.encode(sys_pre, add_special_tokens=False)
        sys_body_ids = tokenizer.encode(sys_body, add_special_tokens=False)
        sys_post_ids = tokenizer.encode(sys_post_user_open, add_special_tokens=False)
        expected_pre_len = len(sys_pre_ids) + len(sys_body_ids) + len(sys_post_ids)
        # Sanity: joined encoding may merge across boundaries and differ
        # by a handful of tokens; tolerate small drift by falling back
        # to NO_ROLE on mismatch.
        if expected_pre_len == len(pre_ids):
            roles = (
                [_NO] * len(sys_pre_ids)
                + [int(Role.SYSTEM)] * len(sys_body_ids)
                + [_NO] * len(sys_post_ids)
            )
        # else: keep as all NO_ROLE (already set above); won't matter
        # for the throughput bench since we don't filter on SYSTEM.
    roles.extend([content_role] * len(content_ids))
    roles.extend([_NO] * len(post_ids))

    ids = pre_ids + content_ids + post_ids
    return ids, roles


# ---------------------------------------------------------------------------
# gpt-oss Harmony
# ---------------------------------------------------------------------------
#
# Harmony's token structure for a user/tool turn followed by an
# assistant generation:
#   <|start|>user<|message|>{content}<|end|>
#   <|start|>assistant<|channel|>final<|message|>
#
# For the tool-response turn we follow Harmony's recipient convention:
#   <|start|>functions.<tool_name><|message|>{content}<|end|>
# The bench uses a fixed tool name "debug_tool" for all tool turns —
# actual tool-call matching isn't exercised here, we just need a
# well-formed envelope.
#
# First turn adds the system + developer preambles. Subsequent turns
# skip those and just emit the incremental user/tool + assistant primer.


class HarmonyTurnBuilder:
    """Stateful per-row turn builder for Harmony.

    Harmony's tokenization API renders whole Conversations; to get
    per-turn incremental tokens we diff the current-turn full rendering
    against the prior-turn rendering. Unlike Qwen3 ChatML's nothink
    template, Harmony doesn't conditionally inject think-markers into
    intermediate assistant turns, so the diff approach is stable.

    Usage::

        b = HarmonyTurnBuilder(harmony_encoding, system_text=SYSTEM_DEBUG)
        turn_1_ids, turn_1_roles = b.next_turn(kind="user", content=user_1)
        turn_2_ids, turn_2_roles = b.next_turn(kind="tool", content=tool_1)
        ...
    """

    def __init__(self, harmony_encoding, system_text: str):
        from openai_harmony import Message, Role as HRole, Conversation
        self._enc = harmony_encoding
        self._HRole = HRole
        self._Message = Message
        self._Conversation = Conversation
        # Conversation starts with a system turn only.
        self._sys_msg = Message.from_role_and_content(HRole.SYSTEM, system_text)
        self._msgs: List = [self._sys_msg]
        # Prior rendering length (tokens before THIS turn's prefill was
        # appended). Used to compute the diff.
        self._prior_len = 0
        # Also initialize: what's the pre-first-turn token length?
        # This is the rendering of just the system message (no
        # trailing user/assistant primer). It's what "cache has before
        # turn 1 prefill" would contain in a real server that sent the
        # system preamble once. For the bench we prefill it as part of
        # turn 1's "incremental" tokens, so _prior_len starts at 0.

    def next_turn(
        self, *, kind: str, content: str,
    ) -> Tuple[List[int], List[int]]:
        """Advance one turn. Returns (ids, roles) for the incremental
        tokens to prefill at this turn. Updates internal state so the
        next call computes the next diff correctly.

        kind: "user" | "tool"
        content: the user-message or tool-output text
        """
        from openai_harmony import TextContent, Author, DeveloperContent

        assert kind in ("user", "tool"), kind
        if kind == "user":
            new_msg = self._Message.from_role_and_content(
                self._HRole.USER, content,
            )
        else:
            # Tool turn in Harmony uses author=TOOL with a recipient-style name.
            new_msg = self._Message(
                author=Author.new(self._HRole.TOOL, "functions.debug_tool"),
                content=[TextContent(text=content)],
            )
        self._msgs.append(new_msg)

        # Render up through this new message + assistant primer.
        conv = self._Conversation.from_messages(self._msgs)
        full_ids = self._enc.render_conversation_for_completion(
            conv, self._HRole.ASSISTANT,
        )
        if not isinstance(full_ids, list):
            full_ids = list(full_ids)
        inc_ids = full_ids[self._prior_len:]

        # Role labeling: the content bulk is labeled; envelope tokens
        # between the marker start and the content itself, plus the
        # assistant primer at the end, get NO_ROLE.
        # Simple approximation: encode the content alone, find where
        # it starts in inc_ids, label that segment.
        content_ids = self._enc.encode(content, allowed_special=set())
        roles = [_NO] * len(inc_ids)
        # Find content_ids as a subsequence in inc_ids.
        start = _find_subsequence(inc_ids, content_ids)
        content_role = (
            int(Role.USER) if kind == "user" else int(Role.TOOL)
        )
        if start >= 0:
            for i in range(start, start + len(content_ids)):
                roles[i] = content_role
        # else: content tokens didn't match cleanly (e.g., merged across
        # envelope boundaries). Leave everything as NO_ROLE — the
        # role-filter intervention just won't fire for this turn, which
        # is acceptable approximation for bench purposes.

        # Update state for next call.
        # Append an empty-content assistant placeholder so the next
        # turn's render includes it (so our diff lines up with "turn N
        # full render" vs "turn N+1 full render").
        self._msgs.append(
            self._Message.from_role_and_content(self._HRole.ASSISTANT, "")
        )
        self._prior_len = len(self._enc.render_conversation(
            self._Conversation.from_messages(self._msgs),
        ))

        return inc_ids, roles


def _find_subsequence(haystack, needle) -> int:
    """Return the index where ``needle`` starts inside ``haystack``, or -1."""
    if not needle:
        return 0
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def build_harmony_system_developer_prefix(harmony_encoding, system: str):
    """Kept for API compatibility; returns None. The new
    :class:`HarmonyTurnBuilder` handles system-prefix tokenization
    internally (first turn's incremental includes the system + user
    envelope preamble)."""
    return None


def build_harmony_turn(
    harmony_encoding,
    *,
    kind: str,
    content: str,
    is_first_turn: bool,
    system_text: str = "",
    _builder_ref: List = None,
) -> Tuple[List[int], List[int]]:
    """Convenience wrapper matching :func:`build_qwen3_turn`'s shape.

    Maintains per-(caller, row) state via the optional
    ``_builder_ref`` list — the caller passes a list holding the
    current :class:`HarmonyTurnBuilder`; on first turn we instantiate,
    subsequent turns reuse. If ``_builder_ref`` is None we construct a
    fresh builder each call, which only works when every call is
    is_first_turn=True (e.g., W1/W2 single-turn workloads).
    """
    if is_first_turn:
        b = HarmonyTurnBuilder(harmony_encoding, system_text)
        if _builder_ref is not None:
            _builder_ref.clear()
            _builder_ref.append(b)
    else:
        if _builder_ref is None or not _builder_ref:
            raise RuntimeError(
                "build_harmony_turn: non-first turn requires a "
                "_builder_ref from a prior first-turn call"
            )
        b = _builder_ref[0]
    return b.next_turn(kind=kind, content=content)
