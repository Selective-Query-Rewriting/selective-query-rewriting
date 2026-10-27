"""Lightweight learned gate training and inference.

Status: STUB — implement after features and outcomes are available
on the training split.

Owner: Person B

Plan
----
- Candidate models: logistic regression, gradient boosting (scikit-learn).
- Features from src/gating/features.py.
- Labels from src/evaluation/outcomes.py (HELPFUL=1, else=0).
- Train/val/test split at conversation level (NOT turn level).
- Tune threshold on validation set; report on test set only.
- Do NOT touch the test set during tuning.
"""

from __future__ import annotations

# TODO (Person B): implement after features.py and outcomes are ready.
raise NotImplementedError(
    "src/gating/train_gate.py is not yet implemented."
)
