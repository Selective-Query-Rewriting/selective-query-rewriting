
"""tests/test_formulate.py — unit tests for src/rewriting/formulate.py"""

import pytest

from src.data.schema import ConversationTurn
from src.rewriting.formulate import (
    formulate,
    formulate_concat,
    formulate_keep,
    formulate_rewrite,
)


def make_turn(question: str, history=None, rewrite=None) -> ConversationTurn:
    return ConversationTurn(
        conversation_id="c1",
        turn_id=2,
        question=question,
        relevant_passage_ids=[],
        conversation_history=history or [],
        rewritten_question=rewrite,
    )


# ── KEEP ─────────────────────────────────────────────────────────────────────


class TestKeep:
    def test_returns_question(self):
        t = make_turn("Who directed that movie?")
        assert formulate_keep(t) == "Who directed that movie?"

    def test_strips_whitespace(self):
        t = make_turn("  spaced  ")
        assert formulate_keep(t) == "spaced"


# ── CONCAT ────────────────────────────────────────────────────────────────────


class TestConcat:
    def test_no_history_falls_back_to_keep(self):
        t = make_turn("Who directed that movie?")
        assert formulate_concat(t) == "Who directed that movie?"

    def test_prepends_history(self):
        t = make_turn(
            "Who directed that movie?",
            history=["We were discussing Inception."],
        )
        result = formulate_concat(t)
        assert result == "We were discussing Inception. Who directed that movie?"

    def test_truncates_history_to_max(self):
        t = make_turn(
            "Current question?",
            history=["H1", "H2", "H3", "H4"],
        )
        result = formulate_concat(t, max_history_turns=2)

        # Should only include H3 and H4 (the last two history entries).
        assert result == "H3 H4 Current question?"

    def test_multi_turn_history_joined_by_space(self):
        t = make_turn("C?", history=["A.", "B."])
        assert formulate_concat(t, max_history_turns=3) == "A. B. C?"

    def test_zero_history_turns_falls_back_to_keep(self):
        t = make_turn("Current question?", history=["Previous question?"])
        assert formulate_concat(t, max_history_turns=0) == "Current question?"

    def test_negative_history_turns_raise_error(self):
        t = make_turn("Current question?", history=["Previous question?"])

        with pytest.raises(ValueError, match="max_history_turns"):
            formulate_concat(t, max_history_turns=-1)

    def test_empty_history_entries_are_ignored(self):
        t = make_turn(
            "Current question?",
            history=["Previous question?", "  ", ""],
        )
        assert formulate_concat(t) == "Previous question? Current question?"

    def test_whitespace_only_history_falls_back_to_keep(self):
        t = make_turn("Current question?", history=["  ", ""])

        assert formulate_concat(t) == "Current question?"


# ── REWRITE ───────────────────────────────────────────────────────────────────


class TestRewrite:
    def test_uses_cached_rewrite_first(self):
        t = make_turn("Who?", rewrite="Gold rewrite")

        result = formulate_rewrite(t, cached_rewrite="Cached rewrite")

        assert result == "Cached rewrite"

    def test_does_not_fall_back_to_gold_rewrite(self):
        t = make_turn("Who?", rewrite="Gold rewrite")

        with pytest.raises(NotImplementedError, match="No cached LLM rewrite"):
            formulate_rewrite(t)

    def test_raises_when_no_rewrite(self):
        t = make_turn("Who?")

        with pytest.raises(NotImplementedError):
            formulate_rewrite(t)

    def test_strips_whitespace_from_cached_rewrite(self):
        t = make_turn("Who?")

        assert formulate_rewrite(t, cached_rewrite="  trimmed  ") == "trimmed"

    def test_empty_cached_rewrite_raises_error(self):
        t = make_turn("Who?")

        with pytest.raises(NotImplementedError):
            formulate_rewrite(t, cached_rewrite="   ")


# ── DISPATCH ──────────────────────────────────────────────────────────────────


class TestFormulateDispatch:
    def test_keep(self):
        t = make_turn("Q?")
        assert formulate(t, "keep") == "Q?"

    def test_concat(self):
        t = make_turn("Q?", history=["H."])
        assert formulate(t, "concat") == "H. Q?"

    def test_rewrite_with_cache(self):
        t = make_turn("Q?")
        assert formulate(t, "rewrite", cached_rewrite="R?") == "R?"

    def test_rewrite_without_cache_raises_error(self):
        t = make_turn("Q?", rewrite="Gold rewrite")

        with pytest.raises(NotImplementedError):
            formulate(t, "rewrite")

    def test_unknown_method_raises(self):
        t = make_turn("Q?")

        with pytest.raises(ValueError, match="Unknown formulation"):
            formulate(t, "unknown_method")
