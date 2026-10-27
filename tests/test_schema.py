"""tests/test_schema.py — unit tests for src/data/schema.py"""
import pytest
from src.data.schema import ConversationTurn


def make_turn(**kwargs) -> ConversationTurn:
    defaults = dict(
        conversation_id="conv1",
        turn_id=1,
        question="What is the capital of France?",
        relevant_passage_ids=["p1", "p2"],
    )
    defaults.update(kwargs)
    return ConversationTurn(**defaults)


class TestConversationTurnDefaults:
    def test_basic_creation(self):
        t = make_turn()
        assert t.conversation_id == "conv1"
        assert t.turn_id == 1
        assert t.question == "What is the capital of France?"
        assert t.relevant_passage_ids == ["p1", "p2"]
        assert t.conversation_history == []
        assert t.rewritten_question is None
        assert t.split is None

    def test_unique_id(self):
        t = make_turn(conversation_id="abc", turn_id=3)
        assert t.unique_id == "abc_3"

    def test_is_first_turn_true(self):
        assert make_turn(turn_id=1).is_first_turn is True

    def test_is_first_turn_false(self):
        assert make_turn(turn_id=2).is_first_turn is False


class TestConversationTurnValidation:
    def test_turn_id_zero_raises(self):
        with pytest.raises(ValueError, match="turn_id"):
            make_turn(turn_id=0)

    def test_turn_id_negative_raises(self):
        with pytest.raises(ValueError, match="turn_id"):
            make_turn(turn_id=-1)

    def test_empty_question_raises(self):
        with pytest.raises(ValueError, match="question"):
            make_turn(question="   ")

    def test_whitespace_only_question_raises(self):
        with pytest.raises(ValueError, match="question"):
            make_turn(question="\t\n")


class TestConversationTurnOptional:
    def test_with_history(self):
        t = make_turn(conversation_history=["Prior question?"])
        assert t.conversation_history == ["Prior question?"]

    def test_with_rewrite(self):
        t = make_turn(rewritten_question="What is the capital of France?")
        assert t.rewritten_question == "What is the capital of France?"

    def test_with_split(self):
        t = make_turn(split="train")
        assert t.split == "train"

    def test_empty_passage_ids(self):
        t = make_turn(relevant_passage_ids=[])
        assert t.relevant_passage_ids == []
