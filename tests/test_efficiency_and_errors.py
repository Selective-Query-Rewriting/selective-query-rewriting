"""tests/test_efficiency_and_errors.py — Tests for Phase 6 efficiency aggregation and error taxonomy."""

import json
from pathlib import Path
from scripts.aggregate_efficiency import analyze_efficiency
from src.evaluation.error_analysis import (
    ErrorCategory,
    detect_lost_key_terms,
    diagnose_turn_failure,
    summarize_error_distribution,
)


def test_detect_lost_key_terms():
    # Number 2024 present in original query but lost in rewrite
    orig = "What happened in the 2024 Olympic games?"
    rewr = "What happened in the Olympic games?"
    lost = detect_lost_key_terms(orig, rewr)
    assert "2024" in lost

    # No lost terms
    orig2 = "Who directed Inception?"
    rewr2 = "Who directed the film Inception?"
    assert len(detect_lost_key_terms(orig2, rewr2)) == 0


def test_diagnose_turn_failure():
    # 1. Generation failure
    d_fail = diagnose_turn_failure("What happened?", [], None, "tied", rewrite_status="failed")
    assert d_fail["category"] == ErrorCategory.GENERATION_FAILURE.value
    assert d_fail["is_failure"] is True

    # 2. Helpful turn
    d_help = diagnose_turn_failure("Who directed it?", ["Inception"], "Who directed Inception?", "helpful")
    assert d_help["category"] == ErrorCategory.NONE.value
    assert d_help["is_failure"] is False

    # 3. Lost key term on harmful turn
    d_lost = diagnose_turn_failure("Apollo 11 astronauts?", ["Moon mission."], "Astronauts who walked on the moon?", "harmful")
    assert d_lost["category"] == ErrorCategory.LOST_KEY_TERMS.value

    # 4. Retrieval harm despite plausible wording
    d_plausible = diagnose_turn_failure("Solar cell efficiency?", ["Solar tech."], "Photovoltaic cell performance rates?", "harmful")
    assert d_plausible["category"] == ErrorCategory.RETRIEVAL_HARM_DESPITE_PLAUSIBLE_WORDING.value


def test_summarize_error_distribution():
    diagnoses = [
        {"category": ErrorCategory.NONE.value},
        {"category": ErrorCategory.GENERATION_FAILURE.value},
        {"category": ErrorCategory.LOST_KEY_TERMS.value},
        {"category": ErrorCategory.GENERATION_FAILURE.value},
    ]
    summary = summarize_error_distribution(diagnoses)
    assert summary["total_inspected"] == 4
    assert summary["categories"][ErrorCategory.GENERATION_FAILURE.value]["count"] == 2
    assert summary["categories"][ErrorCategory.GENERATION_FAILURE.value]["percentage"] == 50.0


def test_analyze_efficiency_mock(tmp_path):
    cache_path = tmp_path / "mock_cache.jsonl"
    out_dir = tmp_path / "results"

    mock_entries = [
        {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65, "latency_ms": 320.0, "retry_used": False},
        {"prompt_tokens": 60, "completion_tokens": 20, "total_tokens": 80, "latency_ms": 410.0, "retry_used": True},
        {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50, "latency_ms": 280.0, "retry_used": False},
    ]
    with cache_path.open("w", encoding="utf-8") as f:
        for entry in mock_entries:
            f.write(json.dumps(entry) + "\n")

    metrics = analyze_efficiency(cache_path=cache_path, output_dir=out_dir)

    assert metrics["total_rewrite_calls"] == 3
    assert metrics["retry_count"] == 1
    assert metrics["tokens"]["total_tokens"] == 195
    assert metrics["tokens"]["mean_total_tokens_per_call"] == 65.0
    assert metrics["latency_ms"]["median"] == 320.0
    assert (out_dir / "efficiency_summary.json").exists()
