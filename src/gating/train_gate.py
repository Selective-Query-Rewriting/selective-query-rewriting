"""Lightweight learned gate training, calibration, and inference.

Features:
- Strict pre-retrieval feature extraction.
- Conversation-level splitting to prevent multi-turn data leakage.
- Threshold tuning on validation set to balance rewrite rate vs retrieval accuracy.
- Scikit-learn Logistic Regression / Gradient Boosting.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier

from src.data.schema import ConversationTurn
from src.gating.features import extract_features, feature_names


def create_conversation_split(
    items: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split records by conversation_id to avoid dialogue turn leakage.

    Parameters
    ----------
    items:
        List of turn dicts containing at least 'conversation_id'.
    train_ratio:
        Fraction of conversations in training split.
    val_ratio:
        Fraction of conversations in validation split.
    test_ratio:
        Fraction of conversations in test split.
    seed:
        Random seed for shuffling conversations.

    Returns
    -------
    dict:
        {'train': [...], 'val': [...], 'test': [...]}
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    conv_map: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        cid = str(item.get("conversation_id", item.get("turn_id", "").split("_")[0]))
        conv_map.setdefault(cid, []).append(item)

    all_convs = sorted(list(conv_map.keys()))
    rng = random.Random(seed)
    rng.shuffle(all_convs)

    n_convs = len(all_convs)
    n_train = int(n_convs * train_ratio)
    n_val = int(n_convs * val_ratio)

    train_convs = set(all_convs[:n_train])
    val_convs = set(all_convs[n_train : n_train + n_val])
    test_convs = set(all_convs[n_train + n_val :])

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for cid, turns in conv_map.items():
        if cid in train_convs:
            splits["train"].extend(turns)
        elif cid in val_convs:
            splits["val"].extend(turns)
        else:
            splits["test"].extend(turns)

    return splits


class LearnedGate:
    """Pre-retrieval binary classifier for KEEP vs REWRITE selection.

    Parameters
    ----------
    model:
        Underlying scikit-learn estimator.
    feature_keys:
        List of feature column names in fixed order.
    threshold:
        Decision threshold in [0, 1]. Predicts REWRITE if probability >= threshold.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        feature_keys: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        self.feature_keys = feature_keys or feature_names()
        self.threshold = threshold
        self.model = model or LogisticRegression(max_iter=1000, random_state=42)

    def _turn_to_vector(
        self,
        turn_or_features: Union[ConversationTurn, Dict[str, float]],
    ) -> np.ndarray:
        if isinstance(turn_or_features, ConversationTurn):
            f_dict = extract_features(turn_or_features)
        elif isinstance(turn_or_features, dict):
            # Check if this is already an extracted feature dict or a raw turn dict
            if "query_token_length" in turn_or_features:
                f_dict = turn_or_features
            else:
                f_dict = extract_features(
                    turn_or_features.get("question", ""),
                    turn_or_features.get("history", []),
                    int(turn_or_features.get("turn_number", 1)),
                )
        else:
            raise TypeError("Expected ConversationTurn or dict")

        vec = [float(f_dict.get(k, 0.0)) for k in self.feature_keys]
        return np.array([vec], dtype=np.float32)

    def predict_proba(
        self,
        turn_or_features: Union[ConversationTurn, Dict[str, float]],
    ) -> float:
        """Return the probability of choosing REWRITE (class 1)."""
        x = self._turn_to_vector(turn_or_features)
        probs = self.model.predict_proba(x)
        # Check if class 1 is present in model classes
        if hasattr(self.model, "classes_") and 1 in self.model.classes_:
            idx = list(self.model.classes_).index(1)
            return float(probs[0, idx])
        elif hasattr(self.model, "classes_") and len(self.model.classes_) == 1:
            return 1.0 if self.model.classes_[0] == 1 else 0.0
        return float(probs[0, 1])

    def predict(
        self,
        turn_or_features: Union[ConversationTurn, Dict[str, float]],
    ) -> bool:
        """Predict whether to REWRITE (True) or KEEP (False)."""
        return self.predict_proba(turn_or_features) >= self.threshold

    def predict_action(
        self,
        turn_or_features: Union[ConversationTurn, Dict[str, float]],
    ) -> str:
        """Return 'rewrite' or 'keep'."""
        return "rewrite" if self.predict(turn_or_features) else "keep"

    def save(self, path: Union[str, Path]) -> None:
        """Serialize model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_keys": self.feature_keys,
            "threshold": self.threshold,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> LearnedGate:
        """Deserialize model from disk."""
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        gate = cls(
            model=payload["model"],
            feature_keys=payload["feature_keys"],
            threshold=payload["threshold"],
        )
        return gate


def train_learned_gate(
    train_turns: List[Dict[str, Any]],
    val_turns: Optional[List[Dict[str, Any]]] = None,
    label_key: str = "label",
    model_type: str = "logistic",
    tune_threshold: bool = True,
    seed: int = 42,
) -> LearnedGate:
    """Train a LearnedGate classifier on labeled conversational turns.

    Parameters
    ----------
    train_turns:
        List of turn records with features or raw turn data, and a binary label
        (1 = REWRITE helpful, 0 = KEEP/TIED).
    val_turns:
        Optional validation records used for threshold calibration.
    label_key:
        Dict key containing binary label (1 or 0).
    model_type:
        'logistic' for LogisticRegression or 'gradient_boost' for HistGradientBoosting.
    tune_threshold:
        Whether to optimize decision threshold on validation set.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    LearnedGate
    """
    f_keys = feature_names()

    def build_matrix(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        for r in records:
            if "query_token_length" in r:
                f_dict = r
            else:
                f_dict = extract_features(
                    r.get("question", ""),
                    r.get("history", []),
                    int(r.get("turn_number", 1)),
                )
            vec = [float(f_dict.get(k, 0.0)) for k in f_keys]
            X_list.append(vec)
            y_list.append(int(r[label_key]))
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

    X_train, y_train = build_matrix(train_turns)

    # Handle single-class training edge case (e.g. tiny synthetic test splits)
    if len(np.unique(y_train)) < 2:
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        return LearnedGate(model=dummy, feature_keys=f_keys, threshold=0.5)

    if model_type == "gradient_boost":
        model = HistGradientBoostingClassifier(random_state=seed, max_iter=100)
    else:
        model = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")

    model.fit(X_train, y_train)
    gate = LearnedGate(model=model, feature_keys=f_keys, threshold=0.5)

    # Threshold tuning on validation set
    if tune_threshold and val_turns:
        X_val, y_val = build_matrix(val_turns)
        if len(np.unique(y_val)) >= 2:
            probs = model.predict_proba(X_val)[:, 1]
            best_t = 0.5
            best_f1 = -1.0
            for t in np.linspace(0.1, 0.9, 17):
                preds = (probs >= t).astype(int)
                tp = np.sum((preds == 1) & (y_val == 1))
                fp = np.sum((preds == 1) & (y_val == 0))
                fn = np.sum((preds == 0) & (y_val == 1))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)
            gate.threshold = best_t

    return gate
