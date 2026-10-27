"""tests/test_outcomes.py — unit tests for src/evaluation/outcomes.py"""
import pytest
from src.evaluation.outcomes import (
    Outcome,
    label_all_outcomes,
    label_outcome,
    summarise_outcomes,
)


class TestLabelOutcome:
    def test_helpful(self):
        assert label_outcome(0.4, 0.6, tie_margin=0.01) == Outcome.HELPFUL

    def test_harmful(self):
        assert label_outcome(0.6, 0.4, tie_margin=0.01) == Outcome.HARMFUL

    def test_tied_exact(self):
        assert label_outcome(0.5, 0.5, tie_margin=0.01) == Outcome.TIED

    def test_tied_within_margin(self):
        assert label_outcome(0.5, 0.509, tie_margin=0.01) == Outcome.TIED

    def test_boundary_helpful(self):
        # delta = 0.011 > 0.01 → HELPFUL
        assert label_outcome(0.5, 0.511, tie_margin=0.01) == Outcome.HELPFUL

    def test_boundary_harmful(self):
        assert label_outcome(0.511, 0.5, tie_margin=0.01) == Outcome.HARMFUL

    def test_negative_margin_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            label_outcome(0.5, 0.6, tie_margin=-0.01)

    def test_zero_margin(self):
        # Any positive delta → HELPFUL with zero margin
        assert label_outcome(0.5, 0.501, tie_margin=0.0) == Outcome.HELPFUL


class TestLabelAllOutcomes:
    def test_basic(self):
        keep = {"c1_1": 0.3, "c1_2": 0.8}
        rewrite = {"c1_1": 0.7, "c1_2": 0.5}
        outcomes = label_all_outcomes(keep, rewrite, tie_margin=0.01)
        assert len(outcomes) == 2
        assert outcomes[0].outcome == Outcome.HELPFUL  # 0.7 > 0.3
        assert outcomes[1].outcome == Outcome.HARMFUL  # 0.5 < 0.8

    def test_skips_missing_keys(self):
        keep = {"c1_1": 0.5, "c1_2": 0.5}
        rewrite = {"c1_1": 0.6}        # c1_2 missing
        outcomes = label_all_outcomes(keep, rewrite, tie_margin=0.01)
        assert len(outcomes) == 1      # only c1_1 in both

    def test_delta_stored(self):
        keep = {"c1_1": 0.4}
        rewrite = {"c1_1": 0.6}
        outcomes = label_all_outcomes(keep, rewrite, tie_margin=0.01)
        assert outcomes[0].delta == pytest.approx(0.2)


class TestSummariseOutcomes:
    def test_all_helpful(self):
        keep = {"a": 0.1, "b": 0.1}
        rewrite = {"a": 0.9, "b": 0.9}
        outcomes = label_all_outcomes(keep, rewrite, tie_margin=0.01)
        s = summarise_outcomes(outcomes)
        assert s["helpful_pct"] == 100.0
        assert s["harmful_pct"] == 0.0
        assert s["n_turns"] == 2

    def test_empty_outcomes(self):
        s = summarise_outcomes([])
        assert s["n_turns"] == 0
        assert s["helpful_pct"] == 0.0
