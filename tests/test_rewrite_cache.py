import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.rewriting.cache import RewriteCache


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    instance = RewriteCache(
        cache_file=str(tmp_path / "rewrites.jsonl"),
        provider="groq",
        model="openai/gpt-oss-20b",
        max_tokens=256,
    )
    instance.client = Mock()
    return instance


def mock_response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        model="openai/gpt-oss-20b",
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        ),
    )


def test_cache_miss_calls_model_and_saves_rewrite(cache, tmp_path):
    cache.client.chat.completions.create.return_value = mock_response(
        "What are the disadvantages of solar energy?"
    )

    result = cache.get_rewrite(
        "conversation1_2",
        "What about its disadvantages?",
        ["What are the benefits of solar energy?"],
    )

    assert result == "What are the disadvantages of solar energy?"
    cache.client.chat.completions.create.assert_called_once()

    saved = (tmp_path / "rewrites.jsonl").read_text().strip()
    entry = json.loads(saved)

    assert entry["rewrite"] == result
    assert entry["turn_id"] == "conversation1_2"
    assert entry["total_tokens"] == 30


def test_cache_hit_does_not_call_model(cache):
    cache.client.chat.completions.create.return_value = mock_response(
        "What are the disadvantages of solar energy?"
    )

    first = cache.get_rewrite("c1_2", "What about its disadvantages?", [])
    second = cache.get_rewrite("c1_2", "What about its disadvantages?", [])

    assert first == second
    cache.client.chat.completions.create.assert_called_once()


def test_saved_rewrite_survives_new_cache_instance(cache, tmp_path, monkeypatch):
    cache.client.chat.completions.create.return_value = mock_response(
        "What are the disadvantages of solar energy?"
    )

    expected = cache.get_rewrite("c1_2", "What about its disadvantages?", [])

    reloaded = RewriteCache(
        cache_file=str(tmp_path / "rewrites.jsonl"),
        provider="groq",
        model="openai/gpt-oss-20b",
        max_tokens=256,
    )
    reloaded.client = Mock()

    actual = reloaded.get_rewrite("c1_2", "What about its disadvantages?", [])

    assert actual == expected
    reloaded.client.chat.completions.create.assert_not_called()


def test_empty_rewrite_is_not_cached(cache, tmp_path):
    cache.client.chat.completions.create.return_value = mock_response("   ")

    with pytest.raises(ValueError, match="empty rewrite"):
        cache.get_rewrite("c1_2", "What about its disadvantages?", [])

    assert not (tmp_path / "rewrites.jsonl").exists()
    assert cache.entries == {}
