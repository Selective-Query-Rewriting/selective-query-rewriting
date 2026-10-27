"""tests/test_oracle.py — unit tests for src/evaluation/oracle.py"""
import pytest
from src.evaluation.oracle import (
    mean_oracle_score,
    oracle_headroom_summary,
    oracle_score_per_turn,
)


SCORES = {
    "keep":    {"c1_1": 0.3, "c1_2": 0.8},
    "rewrite": {"c1_1": 0.7, "c1_2": 0.5},
}


class TestOracleScorePerTurn:
    def test_selects_best(self):
        result = oracle_score_per_turn(SCORES)
        assert result["c1_1"] == pytest.approx(0.7)
        assert result["c1_2"] == pytest.approx(0.8)

    def test_three_formulations(self):
        scores = {
            "keep":   {"c1": 0.2},
            "concat": {"c1": 0.5},
            "rewrite":{"c1": 0.4},
        }
        result = oracle_score_per_turn(scores)
        assert result["c1"] == pytest.approx(0.5)

    def test_requires_at_least_two_formulations(self):
        with pytest.raises(ValueError):
            oracle_score_per_turn({"keep": {"c1": 0.5}})

    def test_turn_only_in_one_formulation(self):
        # c1_3 only in 'keep'; should still be included with its score
        scores = {
            "keep":    {"c1_1": 0.3, "c1_3": 0.9},
            "rewrite": {"c1_1": 0.7},
        }
        result = oracle_score_per_turn(scores)
        assert "c1_3" in result
        assert result["c1_3"] == pytest.approx(0.9)


class TestMeanOracleScore:
    def test_basic(self):
        # oracle picks 0.7 and 0.8; mean = 0.75
        assert mean_oracle_score(SCORES) == pytest.approx(0.75)

    def test_empty_scores(self):
        # No turns in common → empty oracle → 0.0
        result = mean_oracle_score({"k": {}, "r": {}})
        assert result == 0.0


class TestOracleHeadroomSummary:
    def test_headroom_calculation(self):
        summary = oracle_headroom_summary(SCORES, baseline_name="rewrite")
        # rewrite scores: c1_1=0.7, c1_2=0.5 → mean=0.6
        # oracle: c1_1=0.7, c1_2=0.8 → mean=0.75
        assert summary["oracle_mean"] == pytest.approx(0.75)
        assert summary["baseline_mean"] == pytest.approx(0.6)
        assert summary["headroom"] == pytest.approx(0.15)
        assert summary["n_turns"] == 2

    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError, match="baseline"):
            oracle_headroom_summary(SCORES, baseline_name="nonexistent")
