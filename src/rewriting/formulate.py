
"""Query formulation: KEEP, CONCAT, and REWRITE."""

from __future__ import annotations

from typing import Optional

from src.data.schema import ConversationTurn


def formulate_keep(turn: ConversationTurn) -> str:
    """Return the original question."""
    return turn.question.strip()


def formulate_concat(
    turn: ConversationTurn,
    max_history_turns: int = 3,
) -> str:
    """Prepend recent conversation history to the current question."""
    if max_history_turns < 0:
        raise ValueError("max_history_turns must be >= 0")

    if max_history_turns == 0 or not turn.conversation_history:
        return formulate_keep(turn)

    recent = turn.conversation_history[-max_history_turns:]
    context = " ".join(
        item.strip() for item in recent if item.strip()
    )

    if not context:
        return formulate_keep(turn)

    return f"{context} {formulate_keep(turn)}"


def formulate_rewrite(
    turn: ConversationTurn,
    cached_rewrite: Optional[str] = None,
) -> str:
    """Return an LLM-generated rewrite supplied by the cache.

    Dataset reference rewrites are deliberately not used as a fallback.
    """
    if cached_rewrite is not None and cached_rewrite.strip():
        return cached_rewrite.strip()

    raise NotImplementedError(
        f"No cached LLM rewrite for turn {turn.unique_id}. "
        "Generate and cache a rewrite before running this strategy."
    )


def formulate(
    turn: ConversationTurn,
    method: str,
    max_history_turns: int = 3,
    cached_rewrite: Optional[str] = None,
) -> str:
    """Dispatch to the selected query formulation strategy."""
    if method == "keep":
        return formulate_keep(turn)

    if method == "concat":
        return formulate_concat(
            turn,
            max_history_turns=max_history_turns,
        )

    if method == "rewrite":
        return formulate_rewrite(
            turn,
            cached_rewrite=cached_rewrite,
        )

    raise ValueError(
        f"Unknown formulation method: {method!r}. "
        "Choose 'keep', 'concat', or 'rewrite'."
    )


def formulate_with_cache(
    turn: ConversationTurn,
    method: str,
    rewrite_cache=None,
    max_history_turns: int = 3,
) -> str:
    """Formulate a query, generating a rewrite through the cache if needed."""

    if method != "rewrite":
        return formulate(
            turn,
            method=method,
            max_history_turns=max_history_turns,
        )

    if rewrite_cache is None:
        raise ValueError(
            "rewrite_cache is required when method='rewrite'."
        )

    cached_rewrite = rewrite_cache.get_rewrite(
        turn.unique_id,
        turn.question,
        turn.conversation_history,
    )

    return formulate(
        turn,
        method="rewrite",
        cached_rewrite=cached_rewrite,
    )
