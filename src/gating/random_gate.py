"""Random gate: rewrite a query with probability p independently.

This baseline is matched to the learned gate's empirical rewrite rate so that
the evaluation is fair (same fraction of queries rewritten, but random
selection instead of learned selection).
"""

from __future__ import annotations

import random
from typing import Any


class RandomGate:
    """Random decision model matched to a target rewrite probability.

    Parameters
    ----------
    rewrite_rate:
        Probability in [0, 1] of selecting REWRITE over KEEP.
    seed:
        Random seed for reproducibility.
    """

    def __init__(self, rewrite_rate: float = 0.5, seed: int = 42) -> None:
        if not (0.0 <= rewrite_rate <= 1.0):
            raise ValueError(f"rewrite_rate must be between 0.0 and 1.0, got {rewrite_rate}")
        self.rewrite_rate = rewrite_rate
        self.seed = seed
        self._rng = random.Random(seed)

    def set_rewrite_rate(self, rate: float) -> None:
        """Update the target rewrite rate."""
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"rewrite_rate must be between 0.0 and 1.0, got {rate}")
        self.rewrite_rate = rate

    def predict(self, turn_or_item: Any = None) -> bool:
        """Return True for REWRITE, False for KEEP."""
        return self._rng.random() < self.rewrite_rate

    def predict_action(self, turn_or_item: Any = None) -> str:
        """Return formulation name: 'rewrite' or 'keep'."""
        return "rewrite" if self.predict(turn_or_item) else "keep"
