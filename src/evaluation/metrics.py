"""Retrieval metrics: MRR, Recall, nDCG.

All metric functions take:
  - retrieved_ids: ordered list of passage IDs returned by the retriever
  - relevant_ids: set of ground-truth relevant passage IDs
and return a float score in [0, 1].

The cutoff k is parameterised in all functions.

Owner: Person A

Design notes
------------
- These are pure functions with no side effects — easy to unit test.
- All functions handle the empty-relevant-ids edge case explicitly.
- nDCG assumes binary relevance (0 or 1) unless otherwise noted.
  [UNRESOLVED] Confirm whether QReCC relevance is binary or graded.
"""

from __future__ import annotations

import math
from typing import List, Set


def _check_inputs(retrieved_ids: List[str], relevant_ids: Set[str]) -> None:
    if not isinstance(retrieved_ids, list):
        raise TypeError("retrieved_ids must be a list.")
    if not isinstance(relevant_ids, set):
        raise TypeError("relevant_ids must be a set.")


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    """Mean Reciprocal Rank at cutoff k (single-query version).

    Returns 1/rank of the first relevant document in the top-k results,
    or 0 if no relevant document appears in the top k.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved passage IDs (rank 1 = index 0).
    relevant_ids:
        Set of ground-truth relevant passage IDs.
    k:
        Rank cutoff.

    Returns
    -------
    float
        Reciprocal rank in [0, 1].

    Examples
    --------
    >>> reciprocal_rank(["a", "b", "c"], {"b"}, k=10)
    0.5
    >>> reciprocal_rank(["a", "b", "c"], {"d"}, k=10)
    0.0
    >>> reciprocal_rank([], {"a"}, k=10)
    0.0
    """
    _check_inputs(retrieved_ids, relevant_ids)
    if not relevant_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    """Recall at cutoff k.

    Fraction of ground-truth relevant documents that appear in the top-k.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved passage IDs.
    relevant_ids:
        Set of ground-truth relevant passage IDs.
    k:
        Rank cutoff.

    Returns
    -------
    float
        Recall in [0, 1]. Returns 0 if relevant_ids is empty.

    Examples
    --------
    >>> recall_at_k(["a", "b", "c"], {"a", "c"}, k=2)
    0.5
    >>> recall_at_k(["a", "b", "c"], {"a", "c"}, k=10)
    1.0
    >>> recall_at_k(["a"], set(), k=10)
    0.0
    """
    _check_inputs(retrieved_ids, relevant_ids)
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain at cutoff k (binary relevance).

    Assumes binary relevance: a document is either relevant (gain=1) or not (gain=0).

    [UNRESOLVED] If QReCC provides graded relevance, this function must be
    updated to accept relevance scores rather than a set of relevant IDs.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved passage IDs (rank 1 = index 0).
    relevant_ids:
        Set of ground-truth relevant passage IDs.
    k:
        Rank cutoff.

    Returns
    -------
    float
        nDCG in [0, 1]. Returns 0 if relevant_ids is empty.

    Examples
    --------
    >>> round(ndcg_at_k(["a", "b"], {"a"}, k=10), 4)
    1.0
    >>> round(ndcg_at_k(["b", "a"], {"a"}, k=10), 4)
    0.6309
    """
    _check_inputs(retrieved_ids, relevant_ids)
    if not relevant_ids:
        return 0.0

    def dcg(ids: List[str], k: int) -> float:
        return sum(
            1.0 / math.log2(rank + 1)
            for rank, doc_id in enumerate(ids[:k], start=1)
            if doc_id in relevant_ids
        )

    actual_dcg = dcg(retrieved_ids, k)
    # Ideal DCG: place all relevant docs at the top
    ideal_ids = list(relevant_ids) + ["__pad__"] * k
    ideal_dcg = dcg(ideal_ids, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_metrics(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    ks: List[int] | None = None,
) -> dict[str, float]:
    """Compute all configured metrics for a single query.

    Parameters
    ----------
    retrieved_ids:
        Ordered list of retrieved passage IDs.
    relevant_ids:
        Set of ground-truth relevant passage IDs.
    ks:
        Cutoff values to compute. Defaults to [10, 100].

    Returns
    -------
    dict[str, float]
        Keys like 'mrr@10', 'recall@10', 'recall@100', 'ndcg@10'.

    Examples
    --------
    >>> m = compute_metrics(["a", "b"], {"a"}, ks=[10])
    >>> m['mrr@10']
    1.0
    >>> m['recall@10']
    1.0
    """
    if ks is None:
        ks = [10, 100]
    rel_set = set(relevant_ids)
    results: dict[str, float] = {}
    for k in ks:
        results[f"mrr@{k}"] = reciprocal_rank(retrieved_ids, rel_set, k=k)
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, rel_set, k=k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, rel_set, k=k)
    return results
