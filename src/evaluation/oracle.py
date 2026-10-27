"""Oracle headroom calculation.

The oracle represents the analytical upper bound on what a *perfect*
pre-retrieval gate could achieve, given the outcomes already computed.

Definition
----------
For each turn, the oracle picks the action (KEEP or REWRITE) that yields
the higher primary metric score. If CONCAT is also included, it picks the
best of all three.

This is NOT a deployable model. It shows the ceiling performance achievable
by any gate that has access to the same set of actions, measured with the
same metric and tie definition.

Interpretation
--------------
  oracle_score ≈ always_rewrite_score   → rewriting dominates; gate has low headroom
  oracle_score >> always_rewrite_score  → selective rewriting could gain a lot
  oracle_score ≈ always_keep_score      → rewriting rarely helps

Owner: Person A
"""

from __future__ import annotations

from typing import Dict, List, Optional


def oracle_score_per_turn(
    score_sets: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """For each turn, return the best score across all formulations.

    Parameters
    ----------
    score_sets:
        Mapping of formulation_name → {unique_id: score}.
        Example: {'keep': {'conv1_1': 0.4}, 'rewrite': {'conv1_1': 0.7}}
        Must contain at least two formulations.

    Returns
    -------
    dict[str, float]
        Mapping of unique_id → best score across all formulations.

    Examples
    --------
    >>> scores = {
    ...     'keep':    {'c1_1': 0.3, 'c1_2': 0.8},
    ...     'rewrite': {'c1_1': 0.7, 'c1_2': 0.5},
    ... }
    >>> oracle_score_per_turn(scores)
    {'c1_1': 0.7, 'c1_2': 0.8}
    """
    if len(score_sets) < 2:
        raise ValueError("oracle_score_per_turn requires at least two formulations.")

    # Union of all turn IDs present in any formulation
    all_ids: set[str] = set()
    for scores in score_sets.values():
        all_ids.update(scores.keys())

    oracle: Dict[str, float] = {}
    for uid in all_ids:
        best = max(
            scores[uid] for scores in score_sets.values() if uid in scores
        )
        oracle[uid] = best
    return oracle


def mean_oracle_score(score_sets: Dict[str, Dict[str, float]]) -> float:
    """Compute the macro-average oracle score across all turns.

    Parameters
    ----------
    score_sets:
        Same format as oracle_score_per_turn.

    Returns
    -------
    float
        Mean oracle score in [0, 1].

    Examples
    --------
    >>> scores = {
    ...     'keep':    {'c1_1': 0.3, 'c1_2': 0.8},
    ...     'rewrite': {'c1_1': 0.7, 'c1_2': 0.5},
    ... }
    >>> mean_oracle_score(scores)
    0.75
    """
    per_turn = oracle_score_per_turn(score_sets)
    if not per_turn:
        return 0.0
    return sum(per_turn.values()) / len(per_turn)


def oracle_headroom_summary(
    score_sets: Dict[str, Dict[str, float]],
    baseline_name: str = "always_rewrite",
) -> Dict[str, float]:
    """Summarise oracle headroom relative to a chosen baseline.

    Parameters
    ----------
    score_sets:
        Formulation → per-turn scores.
    baseline_name:
        Which formulation to use as the comparison baseline.
        Typically 'always_rewrite' or 'keep'.

    Returns
    -------
    dict with keys:
      'oracle_mean'       – mean oracle score
      'baseline_mean'     – mean score for the baseline formulation
      'headroom'          – oracle_mean - baseline_mean
      'n_turns'           – number of turns in the oracle
    """
    if baseline_name not in score_sets:
        raise ValueError(
            f"baseline '{baseline_name}' not found in score_sets. "
            f"Available: {list(score_sets.keys())}"
        )
    per_turn = oracle_score_per_turn(score_sets)
    oracle_mean = sum(per_turn.values()) / len(per_turn) if per_turn else 0.0

    baseline_scores = score_sets[baseline_name]
    common = set(per_turn.keys()) & set(baseline_scores.keys())
    baseline_mean = (
        sum(baseline_scores[uid] for uid in common) / len(common) if common else 0.0
    )

    return {
        "oracle_mean": oracle_mean,
        "baseline_mean": baseline_mean,
        "headroom": oracle_mean - baseline_mean,
        "n_turns": len(per_turn),
    }
