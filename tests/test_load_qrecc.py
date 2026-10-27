"""tests/test_load_qrecc.py — unit tests for src/data/load_qrecc.py

These tests use synthetic QReCC-format JSON data written to temporary files.
They do NOT require the actual QReCC dataset to be downloaded.
"""
from src.data.load_qrecc import _normalize_passage_ids
import json
import tempfile
from pathlib import Path

import pytest

from src.data.load_qrecc import (
    _extract_history,
    load_turns,
    print_dataset_stats,
)


# ── Synthetic QReCC-format data ───────────────────────────────────────────────

def _make_turn(
    conv_no: int = 1,
    turn_no: int = 1,
    question: str = "What is the capital of France?",
    rewrite: str = "What is the capital of France?",
    context: list | None = None,
    answer: str = "Paris.",
    answer_url: str = "https://example.com",
    source: str = "natural_questions",
    passages: list | None = None,
) -> dict:
    """Create a single QReCC-format turn dict."""
    d = {
        "Conversation_no": conv_no,
        "Turn_no": turn_no,
        "Question": question,
        "Rewrite": rewrite,
        "Context": context or [],
        "Answer": answer,
        "Answer_URL": answer_url,
        "Conversation_source": source,
    }
    if passages is not None:
        d["Passages"] = passages
    return d


SAMPLE_CONVERSATION = [
    _make_turn(
        conv_no=1, turn_no=1,
        question="Tell me about Inception.",
        rewrite="Tell me about the movie Inception.",
        context=[],
        answer="Inception is a 2010 science fiction film.",
        passages=[101, 102],
    ),
    _make_turn(
        conv_no=1, turn_no=2,
        question="Who directed it?",
        rewrite="Who directed the movie Inception?",
        context=[
            "Tell me about Inception.",
            "Inception is a 2010 science fiction film.",
        ],
        answer="Christopher Nolan directed Inception.",
        passages=[103],
    ),
    _make_turn(
        conv_no=1, turn_no=3,
        question="What other movies did he make?",
        rewrite="What other movies did Christopher Nolan make?",
        context=[
            "Tell me about Inception.",
            "Inception is a 2010 science fiction film.",
            "Who directed it?",
            "Christopher Nolan directed Inception.",
        ],
        answer="The Dark Knight, Interstellar, Tenet, and others.",
        passages=[201, 202, 203],
    ),
]

SAMPLE_CONVERSATION_2 = [
    _make_turn(
        conv_no=2, turn_no=1,
        question="What is Python?",
        rewrite="What is the Python programming language?",
        context=[],
        answer="Python is a high-level programming language.",
        passages=[301],
    ),
]

SAMPLE_TRAINING_TURN = [
    _make_turn(
        conv_no=100, turn_no=1,
        question="Who is Einstein?",
        rewrite="Who is Albert Einstein?",
        context=[],
        answer="Albert Einstein was a theoretical physicist.",
        # No passages field — training set
    ),
]


def _write_json(data: list[dict], tmp_dir: Path, filename: str) -> Path:
    """Write a list of dicts as a JSON file and return its path."""
    path = tmp_dir / filename
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ── Tests for _extract_history ────────────────────────────────────────────────

class TestExtractHistory:
    def test_empty_context(self):
        assert _extract_history([]) == []

    def test_questions_only(self):
        ctx = ["Q1?", "A1.", "Q2?", "A2."]
        assert _extract_history(ctx, extract_questions_only=True) == ["Q1?", "Q2?"]

    def test_all_strings(self):
        ctx = ["Q1?", "A1.", "Q2?", "A2."]
        assert _extract_history(ctx, extract_questions_only=False) == ctx

    def test_single_qa_pair(self):
        ctx = ["Q1?", "A1."]
        assert _extract_history(ctx, extract_questions_only=True) == ["Q1?"]

    def test_odd_length_context(self):
        # Edge case: context with odd number of elements (shouldn't happen but be robust)
        ctx = ["Q1?", "A1.", "Q2?"]
        result = _extract_history(ctx, extract_questions_only=True)
        assert result == ["Q1?", "Q2?"]


# ── Tests for load_turns ──────────────────────────────────────────────────────

class TestLoadTurns:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_turns("/nonexistent/path.json", split="test")

    def test_load_basic(self, tmp_path):
        path = _write_json(SAMPLE_CONVERSATION, tmp_path, "test.json")
        turns = load_turns(path, split="test")
        assert len(turns) == 3

    def test_field_mapping(self, tmp_path):
        path = _write_json(SAMPLE_CONVERSATION, tmp_path, "test.json")
        turns = load_turns(path, split="test")

        t1 = turns[0]  # conv 1, turn 1 (sorted)
        assert t1.conversation_id == "1"
        assert t1.turn_id == 1
        assert t1.question == "Tell me about Inception."
        assert t1.rewritten_question == "Tell me about the movie Inception."
        assert t1.split == "test"

    def test_passage_ids_mapped(self, tmp_path):
        path = _write_json(SAMPLE_CONVERSATION, tmp_path, "test.json")
        turns = load_turns(path, split="test")

        t1 = turns[0]
        assert t1.relevant_passage_ids == ["101", "102"]

        t3 = turns[2]
        assert t3.relevant_passage_ids == ["201", "202", "203"]

    def test_context_questions_only(self, tmp_path):
        path = _write_json(SAMPLE_CONVERSATION, tmp_path, "test.json")
        turns = load_turns(path, split="test", extract_questions_only=True)

        # Turn 1: no history
        assert turns[0].conversation_history == []

        # Turn 2: one prior question
        assert turns[1].conversation_history == ["Tell me about Inception."]

        # Turn 3: two prior questions
        assert turns[2].conversation_history == [
            "Tell me about Inception.",
            "Who directed it?",
        ]

    def test_context_all_strings(self, tmp_path):
        path = _write_json(SAMPLE_CONVERSATION, tmp_path, "test.json")
        turns = load_turns(path, split="test", extract_questions_only=False)

        # Turn 3: full interleaved context
        assert len(turns[2].conversation_history) == 4
        assert turns[2].conversation_history[1] == "Inception is a 2010 science fiction film."

    def test_sorted_by_conv_and_turn(self, tmp_path):
        # Reverse the order to test sorting
        data = list(reversed(SAMPLE_CONVERSATION + SAMPLE_CONVERSATION_2))
        path = _write_json(data, tmp_path, "test.json")
        turns = load_turns(path, split="test")

        assert len(turns) == 4
        # Conversation 1 comes first (string sort: "1" < "2")
        assert turns[0].conversation_id == "1"
        assert turns[0].turn_id == 1
        assert turns[3].conversation_id == "2"
        assert turns[3].turn_id == 1

    def test_unique_id_format(self, tmp_path):
        path = _write_json(SAMPLE_CONVERSATION, tmp_path, "test.json")
        turns = load_turns(path, split="test")

        assert turns[0].unique_id == "1_1"
        assert turns[1].unique_id == "1_2"

    def test_training_no_passages(self, tmp_path):
        path = _write_json(SAMPLE_TRAINING_TURN, tmp_path, "train.json")
        turns = load_turns(path, split="train")

        assert len(turns) == 1
        assert turns[0].relevant_passage_ids == []

    def test_split_label_propagated(self, tmp_path):
        path = _write_json(SAMPLE_CONVERSATION, tmp_path, "test.json")
        turns = load_turns(path, split="test")
        assert all(t.split == "test" for t in turns)

        path2 = _write_json(SAMPLE_TRAINING_TURN, tmp_path, "train.json")
        turns2 = load_turns(path2, split="train")
        assert all(t.split == "train" for t in turns2)


# ── Tests for max_conversations (pilot mode) ─────────────────────────────────

class TestPilotMode:
    def test_limit_conversations(self, tmp_path):
        data = SAMPLE_CONVERSATION + SAMPLE_CONVERSATION_2
        path = _write_json(data, tmp_path, "test.json")

        # Only load 1 conversation
        turns = load_turns(path, split="test", max_conversations=1)
        conv_ids = {t.conversation_id for t in turns}
        assert len(conv_ids) == 1

    def test_none_loads_all(self, tmp_path):
        data = SAMPLE_CONVERSATION + SAMPLE_CONVERSATION_2
        path = _write_json(data, tmp_path, "test.json")

        turns = load_turns(path, split="test", max_conversations=None)
        conv_ids = {t.conversation_id for t in turns}
        assert len(conv_ids) == 2

    def test_limit_larger_than_available(self, tmp_path):
        data = SAMPLE_CONVERSATION  # 1 conversation
        path = _write_json(data, tmp_path, "test.json")

        turns = load_turns(path, split="test", max_conversations=100)
        assert len(turns) == 3  # all turns from the 1 conversation


# ── Tests for print_dataset_stats ─────────────────────────────────────────────

class TestPrintStats:
    def test_prints_without_error(self, tmp_path, capsys):
        path = _write_json(SAMPLE_CONVERSATION, tmp_path, "test.json")
        turns = load_turns(path, split="test")
        print_dataset_stats(turns, label="test")
        captured = capsys.readouterr()
        assert "Turns:" in captured.out
        assert "Conversations:" in captured.out

    def test_empty_turns(self, capsys):
        print_dataset_stats([], label="empty")
        captured = capsys.readouterr()
        assert "No turns loaded" in captured.out


class TestNormalizePassageIds:
    def test_extracts_markdown_url_target(self):
        refs = [
            "[https://example.com/article_p0](https://example.com/article_p0)"
        ]
        assert _normalize_passage_ids(refs) == [
            "https://example.com/article_p0"
        ]

    def test_unescapes_markdown_characters(self):
        refs = [
            r"[https://example.com/article\_p0](https://example.com/article\_p0)"
        ]
        assert _normalize_passage_ids(refs) == [
            "https://example.com/article_p0"
        ]

    def test_preserves_plain_numeric_ids_as_strings(self):
        assert _normalize_passage_ids([101, 102]) == ["101", "102"]

    def test_empty_or_missing_judgments(self):
        assert _normalize_passage_ids(None) == []
        assert _normalize_passage_ids([]) == []

    def test_ignores_empty_entries(self):
        assert _normalize_passage_ids([None, "", "  ", 101]) == ["101"]
