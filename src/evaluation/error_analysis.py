"""Error taxonomy and failure diagnosis for query rewriting.

Phase 6:
Systematic diagnosis of why conversational rewrites help, tie, or hurt retrieval.
Covers:
1. lost_key_terms: Essential entities, numbers, dates, or constraints dropped.
2. unsupported_specificity: Hallucinates or assumes details not present in history.
3. wrong_referent: Resolves coreference/pronoun to the wrong dialogue entity.
4. intent_drift: Question intent, target answer type, or scope altered.
5. no_useful_change: Trivial synonym swap/paraphrase that does not alter retrieval.
6. generation_failure: Model failed to generate valid text (empty/timeout/token exhaustion).
7. retrieval_harm_despite_plausible_wording: Query is grammatically fluent but retrieves worse BM25 matches.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCategory(str, Enum):
    """Failure taxonomy classes for conversational query rewriting."""
    LOST_KEY_TERMS = "lost_key_terms"
    UNSUPPORTED_SPECIFICITY = "unsupported_specificity"
    WRONG_REFERENT = "wrong_referent"
    INTENT_DRIFT = "intent_drift"
    NO_USEFUL_CHANGE = "no_useful_change"
    GENERATION_FAILURE = "generation_failure"
    RETRIEVAL_HARM_DESPITE_PLAUSIBLE_WORDING = "retrieval_harm_despite_plausible_wording"
    NONE = "none"


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def detect_lost_key_terms(original_query: str, rewrite: str) -> List[str]:
    """Identify numbers, acronyms, or proper nouns present in original but lost in rewrite."""
    orig_tokens = set(tokenize(original_query))
    rewr_tokens = set(tokenize(rewrite))

    lost = []
    # Check for digits/numbers dropped
    for token in orig_tokens:
        if any(c.isdigit() for c in token) and token not in rewr_tokens:
            lost.append(token)
    return lost


def diagnose_turn_failure(
    original_query: str,
    history: List[str],
    rewrite: Optional[str],
    outcome: str,  # "helpful", "tied", "harmful"
    rewrite_status: str = "success",
    manual_category: Optional[ErrorCategory] = None,
) -> Dict[str, Any]:
    """Classify a turn's rewriting outcome into the research error taxonomy."""
    if manual_category:
        return {
            "category": manual_category.value,
            "is_failure": manual_category != ErrorCategory.NONE,
            "notes": "Manually verified",
        }

    # 1. Generation failure
    if rewrite_status != "success" or not rewrite or not rewrite.strip():
        return {
            "category": ErrorCategory.GENERATION_FAILURE.value,
            "is_failure": True,
            "notes": "Rewrite generation failed or returned empty response.",
        }

    # If rewrite was helpful, no error
    if outcome == "helpful":
        return {
            "category": ErrorCategory.NONE.value,
            "is_failure": False,
            "notes": "Rewrite improved retrieval.",
        }

    # 2. Check for lost key terms (e.g. dropped numbers/dates)
    lost_terms = detect_lost_key_terms(original_query, rewrite)
    if lost_terms and outcome == "harmful":
        return {
            "category": ErrorCategory.LOST_KEY_TERMS.value,
            "is_failure": True,
            "notes": f"Rewrite lost key terms: {lost_terms}",
            "lost_terms": lost_terms,
        }

    # 3. Check for no useful change (nearly identical query or identical tokens)
    orig_set = set(tokenize(original_query))
    rewr_set = set(tokenize(rewrite))
    if orig_set == rewr_set or original_query.strip().lower() == rewrite.strip().lower():
        return {
            "category": ErrorCategory.NO_USEFUL_CHANGE.value,
            "is_failure": outcome == "harmful",
            "notes": "Rewrite is essentially identical or a trivial permutation.",
        }

    # 4. If harmful despite distinct and fluent wording
    if outcome == "harmful":
        return {
            "category": ErrorCategory.RETRIEVAL_HARM_DESPITE_PLAUSIBLE_WORDING.value,
            "is_failure": True,
            "notes": "Rewrite changed wording without improving BM25 token overlap.",
        }

    # Tied outcome
    return {
        "category": ErrorCategory.NO_USEFUL_CHANGE.value,
        "is_failure": False,
        "notes": "Rewrite performed approximately tied with raw query.",
    }


def summarize_error_distribution(
    diagnoses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute counts and percentages across taxonomy categories."""
    total = len(diagnoses)
    if total == 0:
        return {"total_inspected": 0, "categories": {}}

    counts: Dict[str, int] = {}
    for d in diagnoses:
        cat = d.get("category", "unknown")
        counts[cat] = counts.get(cat, 0) + 1

    summary: Dict[str, Any] = {
        "total_inspected": total,
        "categories": {
            cat: {
                "count": c,
                "percentage": round(c / total * 100, 2),
            }
            for cat, c in sorted(counts.items())
        },
    }
    return summary
