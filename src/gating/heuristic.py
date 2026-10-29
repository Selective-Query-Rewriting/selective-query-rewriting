"""Heuristic gate: rule-based pre-retrieval query rewriting decision.

A transparent, zero-training baseline that decides whether to rewrite a query
based on explicit syntactic cues (pronouns, ellipsis, query length, and turn position).
"""

from __future__ import annotations

from typing import Dict, Union

from src.data.schema import ConversationTurn
from src.gating.features import extract_features


class HeuristicGate:
    """Rule-based decision model for selective query formulation.

    Parameters
    ----------
    short_query_threshold:
        Queries with fewer tokens than this on follow-up turns trigger rewrite.
    rewrite_on_pronoun:
        If True, queries with third-person/demonstrative pronouns trigger rewrite.
    rewrite_on_ellipsis:
        If True, queries starting with ellipsis markers trigger rewrite.
    never_rewrite_first_turn:
        If True, the opening question of a conversation is never rewritten.
    """

    def __init__(
        self,
        short_query_threshold: int = 4,
        rewrite_on_pronoun: bool = True,
        rewrite_on_ellipsis: bool = True,
        never_rewrite_first_turn: bool = True,
    ) -> None:
        self.short_query_threshold = short_query_threshold
        self.rewrite_on_pronoun = rewrite_on_pronoun
        self.rewrite_on_ellipsis = rewrite_on_ellipsis
        self.never_rewrite_first_turn = never_rewrite_first_turn

    def predict(
        self,
        turn_or_features: Union[ConversationTurn, Dict[str, float]],
    ) -> bool:
        """Predict whether to call REWRITE (True) or KEEP (False)."""
        if isinstance(turn_or_features, ConversationTurn):
            feats = extract_features(turn_or_features)
        elif isinstance(turn_or_features, dict):
            feats = turn_or_features
        else:
            raise TypeError("Expected ConversationTurn or feature dict.")

        # Opening turn has no conversational history to resolve
        if self.never_rewrite_first_turn and feats.get("is_first_turn", 0.0) == 1.0:
            return False

        # Pronoun indicator
        if self.rewrite_on_pronoun and feats.get("has_pronoun", 0.0) == 1.0:
            return True

        # Ellipsis indicator
        if self.rewrite_on_ellipsis and feats.get("has_ellipsis", 0.0) == 1.0:
            return True

        # Short follow-up queries tend to be fragmented or coreferent
        if feats.get("query_token_length", 0.0) < self.short_query_threshold:
            return True

        return False

    def predict_action(
        self,
        turn_or_features: Union[ConversationTurn, Dict[str, float]],
    ) -> str:
        """Return the string formulation action: 'rewrite' or 'keep'."""
        return "rewrite" if self.predict(turn_or_features) else "keep"
