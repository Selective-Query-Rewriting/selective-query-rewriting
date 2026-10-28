
from unittest.mock import Mock

import pytest

from src.data.schema import ConversationTurn
from src.rewriting.formulate import formulate_with_cache


@pytest.fixture
def turn():
    return ConversationTurn(
        conversation_id="c1",
        turn_id=2,
        question="What about its disadvantages?",
        relevant_passage_ids=[],
        conversation_history=[
            "What are the advantages of solar energy?"
        ],
    )


def test_keep_does_not_call_rewrite_cache(turn):
    cache = Mock()

    result = formulate_with_cache(
        turn, method="keep", rewrite_cache=cache
    )

    assert result == "What about its disadvantages?"
    cache.get_rewrite.assert_not_called()


def test_concat_does_not_call_rewrite_cache(turn):
    cache = Mock()

    result = formulate_with_cache(
        turn, method="concat", rewrite_cache=cache
    )

    assert result == (
        "What are the advantages of solar energy? "
        "What about its disadvantages?"
    )
    cache.get_rewrite.assert_not_called()


def test_rewrite_calls_cache_and_returns_rewrite(turn):
    cache = Mock()
    cache.get_rewrite.return_value = (
        "What are the disadvantages of solar energy?"
    )

    result = formulate_with_cache(
        turn, method="rewrite", rewrite_cache=cache
    )

    assert result == "What are the disadvantages of solar energy?"
    cache.get_rewrite.assert_called_once_with(
        turn.unique_id,
        turn.question,
        turn.conversation_history,
    )


def test_rewrite_requires_cache(turn):
    with pytest.raises(ValueError, match="rewrite_cache is required"):
        formulate_with_cache(turn, method="rewrite")
