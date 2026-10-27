"""Shared data schema.

This module defines the canonical data structures used across all modules.
Both teammates MUST agree on any change to these structures before merging,
because schema changes affect every downstream module.

Status: STABLE — the ConversationTurn dataclass is the agreed shared schema.
Field names that depend on QReCC's actual format are marked TODO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConversationTurn:
    """One turn in a conversational search session.

    This is the shared unit of data across all pipeline stages.

    Attributes
    ----------
    conversation_id:
        Unique identifier for the conversation. Must be the same for all
        turns in a session. Used for conversation-level train/test splitting.
    turn_id:
        Position of this turn within the conversation (1-indexed).
    question:
        The raw user question as it appears in the dataset.
    relevant_passage_ids:
        Ground-truth passage identifiers from the corpus. Used for retrieval
        evaluation. The exact ID format depends on the corpus; do not assume.
    conversation_history:
        Previous turns in the session, in chronological order (oldest first).
        May be empty for the first turn of a conversation.
    rewritten_question:
        The gold/reference rewrite provided by the dataset, if any.
        Set to None if the dataset does not provide one or the turn has
        not been loaded yet.
    split:
        Which dataset split this turn belongs to: 'train', 'dev', or 'test'.

    Notes
    -----
    - Do not store retrieved results or computed scores here.
      Those belong in downstream evaluation structures.
    - Field names match our internal schema, not necessarily the raw dataset.
      See src/data/load_qrecc.py for the mapping from raw fields.
    """

    conversation_id: str
    turn_id: int
    question: str
    relevant_passage_ids: List[str]
    conversation_history: List[str] = field(default_factory=list)
    rewritten_question: Optional[str] = None
    split: Optional[str] = None

    def __post_init__(self) -> None:
        if self.turn_id < 1:
            raise ValueError(f"turn_id must be >= 1, got {self.turn_id}")
        if not self.question.strip():
            raise ValueError("question must not be empty")

    @property
    def is_first_turn(self) -> bool:
        """True if this is the opening question of a conversation."""
        return self.turn_id == 1

    @property
    def unique_id(self) -> str:
        """Stable identifier for a single turn across the pipeline."""
        return f"{self.conversation_id}_{self.turn_id}"
