"""Pre-retrieval feature extraction for query rewriting gating.

All features are strictly pre-retrieval: they use ONLY the current question,
dialogue history, and turn position. No retrieval results, gold rewrites, or
post-retrieval information are used.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Union

from src.data.schema import ConversationTurn

# Coreference and deictic markers
PRONOUNS: Set[str] = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "their",
    "theirs", "its", "this", "that", "these", "those"
}

QUESTION_WORDS: Set[str] = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how"
}

ELLIPSIS_STARTERS: List[str] = [
    "and ", "or ", "also ", "what about", "how about", "why ", "where ", "who "
]


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric lowercase tokenization."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def compute_jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute Jaccard similarity between two token sequences."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def extract_features(
    turn_or_question: Union[ConversationTurn, str],
    conversation_history: List[str] | None = None,
    turn_id: int = 1,
) -> Dict[str, float]:
    """Extract a dictionary of numerical pre-retrieval features.

    Parameters
    ----------
    turn_or_question:
        Either a ConversationTurn instance, or a raw question string.
    conversation_history:
        List of prior questions/turns (ignored if ConversationTurn is passed).
    turn_id:
        Turn position (1-indexed; ignored if ConversationTurn is passed).

    Returns
    -------
    dict[str, float]
        Dictionary of numerical features suitable for classification.
    """
    if isinstance(turn_or_question, ConversationTurn):
        question = turn_or_question.question
        history = turn_or_question.conversation_history
        pos = turn_or_question.turn_id
    else:
        question = str(turn_or_question)
        history = conversation_history or []
        pos = max(1, int(turn_id))

    tokens = tokenize(question)
    q_len = len(tokens)
    q_lower = question.lower().strip()

    # History tokens
    history_tokens = []
    for h in history:
        history_tokens.extend(tokenize(h))

    # Pronoun counts and indicators
    pronoun_matches = [t for t in tokens if t in PRONOUNS]
    has_pronoun = 1.0 if len(pronoun_matches) > 0 else 0.0
    pronoun_count = float(len(pronoun_matches))
    pronoun_ratio = pronoun_count / q_len if q_len > 0 else 0.0

    # Ellipsis indicators (question starts with follow-up phrases or lacks a verb)
    has_ellipsis = 0.0
    for starter in ELLIPSIS_STARTERS:
        if q_lower.startswith(starter):
            has_ellipsis = 1.0
            break

    # Lexical overlap with history
    lexical_overlap = compute_jaccard_similarity(tokens, history_tokens)

    # First turn indicator
    is_first_turn = 1.0 if pos == 1 else 0.0

    # History depth
    history_turns = float(len(history))
    history_token_count = float(len(history_tokens))

    # Question word present
    has_q_word = 1.0 if any(t in QUESTION_WORDS for t in tokens) else 0.0

    return {
        "query_token_length": float(q_len),
        "query_char_length": float(len(question.strip())),
        "turn_position": float(pos),
        "is_first_turn": is_first_turn,
        "has_pronoun": has_pronoun,
        "pronoun_count": pronoun_count,
        "pronoun_ratio": pronoun_ratio,
        "has_ellipsis": has_ellipsis,
        "lexical_overlap": lexical_overlap,
        "history_turn_count": history_turns,
        "history_token_count": history_token_count,
        "has_question_word": has_q_word,
    }


def feature_names() -> List[str]:
    """Return ordered list of extracted feature keys."""
    sample = extract_features("test question", [], 1)
    return sorted(list(sample.keys()))
