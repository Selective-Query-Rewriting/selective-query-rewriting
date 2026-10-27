"""tests/test_metrics.py — unit tests for src/evaluation/metrics.py"""
import pytest
from src.evaluation.metrics import (
    compute_metrics,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ── reciprocal_rank ───────────────────────────────────────────────────────────

class TestReciprocalRank:
    def test_first_hit(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_hit(self):
        assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)

    def test_third_hit(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert reciprocal_rank(["a", "b", "c"], {"z"}, k=3) == 0.0

    def test_hit_beyond_cutoff(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}, k=2) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank([], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert reciprocal_rank(["a", "b"], set()) == 0.0

    def test_multiple_relevant_returns_first(self):
        assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == pytest.approx(0.5)


# ── recall_at_k ───────────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=10) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=2) == pytest.approx(0.5)

    def test_no_recall(self):
        assert recall_at_k(["a", "b"], {"z"}) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a"], set()) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], {"a"}) == 0.0


# ── ndcg_at_k ─────────────────────────────────────────────────────────────────

class TestNdcgAtK:
    def test_perfect_ranking(self):
        # Only relevant doc is at rank 1
        assert ndcg_at_k(["a", "b", "c"], {"a"}, k=10) == pytest.approx(1.0)

    def test_worst_position(self):
        # Relevant doc at rank 2, ideal is rank 1 → nDCG < 1
        score = ndcg_at_k(["b", "a"], {"a"}, k=10)
        assert 0 < score < 1.0

    def test_no_relevant_in_results(self):
        assert ndcg_at_k(["a", "b"], {"z"}) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a"], set()) == 0.0

    def test_two_relevant_both_at_top(self):
        # Perfect: both relevant docs at top-2 positions
        assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=10) == pytest.approx(1.0)

    def test_known_value(self):
        # Single relevant doc at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
        import math
        expected = (1 / math.log2(3)) / 1.0
        assert ndcg_at_k(["x", "a"], {"a"}, k=10) == pytest.approx(expected, abs=1e-6)


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_all_keys(self):
        m = compute_metrics(["a"], {"a"}, ks=[10])
        assert set(m.keys()) == {"mrr@10", "recall@10", "ndcg@10"}

    def test_perfect_scores(self):
        m = compute_metrics(["a"], {"a"}, ks=[10])
        assert m["mrr@10"] == 1.0
        assert m["recall@10"] == 1.0
        assert m["ndcg@10"] == pytest.approx(1.0)

    def test_multiple_ks(self):
        m = compute_metrics(["a"], {"a"}, ks=[10, 100])
        assert "recall@100" in m
        assert "mrr@100" in m
