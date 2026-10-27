"""Help / Tie / Hurt outcome labeling.

For each query turn, compare the primary metric score under KEEP vs REWRITE
and assign an outcome label.

  HELPFUL : rewrite score > keep score + tie_margin
  HARMFUL : rewrite score < keep score - tie_margin
  TIED    : |rewrite score - keep score| <= tie_margin

The tie_margin is a research decision that must be agreed upon before
running outcome analysis. It is read from the experiment config.
[UNRESOLVED] — see configs/experiment.yaml → evaluation.tie_margin.

Owner: Person A
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, NamedTuple


class Outcome(str, Enum):
    """Rewrite outcome for a single query turn."""
    HELPFUL = "helpful"
    TIED = "tied"
    HARMFUL = "harmful"


class TurnOutcome(NamedTuple):
    """Outcome for one turn."""
    unique_id: str
    keep_score: float
    rewrite_score: float
    outcome: Outcome
    delta: float  # rewrite_score - keep_score


def label_outcome(
    keep_score: float,
    rewrite_score: float,
    tie_margin: float,
) -> Outcome:
    """Assign a help/tie/hurt label to a single turn.

    Parameters
    ----------
    keep_score:
        Primary metric score for the KEEP formulation.
    rewrite_score:
        Primary metric score for the REWRITE formulation.
    tie_margin:
        Minimum absolute difference to call a result helpful or harmful.
        Within this margin, the outcome is TIED.

    Returns
    -------
    Outcome

    Examples
    --------
    >>> label_outcome(0.5, 0.7, tie_margin=0.01)
    <Outcome.HELPFUL: 'helpful'>
    >>> label_outcome(0.5, 0.49, tie_margin=0.01)
    <Outcome.TIED: 'tied'>
    >>> label_outcome(0.5, 0.3, tie_margin=0.01)
    <Outcome.HARMFUL: 'harmful'>
    """
    if tie_margin < 0:
        raise ValueError(f"tie_margin must be non-negative, got {tie_margin}")
    delta = rewrite_score - keep_score
    if delta > tie_margin:
        return Outcome.HELPFUL
    elif delta < -tie_margin:
        return Outcome.HARMFUL
    else:
        return Outcome.TIED


def label_all_outcomes(
    keep_scores: Dict[str, float],
    rewrite_scores: Dict[str, float],
    tie_margin: float,
) -> List[TurnOutcome]:
    """Label outcomes for all turns present in both score dictionaries.

    Parameters
    ----------
    keep_scores:
        Mapping of unique_id → primary metric score under KEEP.
    rewrite_scores:
        Mapping of unique_id → primary metric score under REWRITE.
    tie_margin:
        Minimum delta to be considered helpful or harmful.

    Returns
    -------
    List[TurnOutcome]
        One entry per turn found in both dictionaries.
        Turns missing from either dictionary are silently skipped;
        callers should log such mismatches.
    """
    common_ids = set(keep_scores.keys()) & set(rewrite_scores.keys())
    outcomes = []
    for uid in sorted(common_ids):
        ks = keep_scores[uid]
        rs = rewrite_scores[uid]
        outcome = label_outcome(ks, rs, tie_margin)
        outcomes.append(TurnOutcome(uid, ks, rs, outcome, rs - ks))
    return outcomes


def summarise_outcomes(outcomes: List[TurnOutcome]) -> Dict[str, float]:
    """Compute percentage statistics over labeled outcomes.

    Returns
    -------
    dict with keys: 'helpful_pct', 'tied_pct', 'harmful_pct', 'n_turns'.
    """
    n = len(outcomes)
    if n == 0:
        return {"helpful_pct": 0.0, "tied_pct": 0.0, "harmful_pct": 0.0, "n_turns": 0}
    counts = {o: 0 for o in Outcome}
    for to in outcomes:
        counts[to.outcome] += 1
    return {
        "helpful_pct": 100 * counts[Outcome.HELPFUL] / n,
        "tied_pct": 100 * counts[Outcome.TIED] / n,
        "harmful_pct": 100 * counts[Outcome.HARMFUL] / n,
        "n_turns": n,
    }
