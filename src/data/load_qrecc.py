
"""QReCC dataset loader supporting official JSON and SCAI JSONL formats."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.data.schema import ConversationTurn

logger = logging.getLogger(__name__)

# Official QReCC fields
_F_CONV_ID = "Conversation_no"
_F_TURN_ID = "Turn_no"
_F_QUESTION = "Question"
_F_CONTEXT = "Context"

# Official and SCAI shared-task variants
_REWRITE_FIELDS = ("Rewrite", "Truth_rewrite")
_PASSAGE_FIELDS = ("Passages", "Truth_passages")


def _extract_history(
    context: List[str],
    extract_questions_only: bool = True,
) -> List[str]:
    """Extract prior questions, or all context strings, in chronological order."""
    if not context:
        return []

    if extract_questions_only:
        return [
            str(context[i])
            for i in range(0, len(context), 2)
            if context[i] is not None and str(context[i]).strip()
        ]

    return [
        str(item)
        for item in context
        if item is not None and str(item).strip()
    ]


def _first_present(item: dict, field_names: tuple[str, ...]):
    """Return the first non-None value found among alternative field names."""
    for field_name in field_names:
        if field_name in item and item[field_name] is not None:
            return item[field_name]
    return None


def _normalize_passage_id(passage: object) -> str:
    """Normalize a passage ID or Markdown link to its corpus ID.

    Examples:
        101 -> "101"
        "[https://example.com/page_p0](https://example.com/page_p0)"
            -> "https://example.com/page_p0"

    Markdown-escaped characters are unescaped to match the corpus index.
    """
    value = str(passage).strip()

    # Extract the target URL from a Markdown link, not its display label.
    match = re.search(r"\]\((https?://.*)\)$", value)
    if match:
        value = match.group(1)

    # Remove Markdown backslash escapes from punctuation.
    value = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~:&])", r"\1", value)

    return value.strip()


def _normalize_passage_ids(raw_passages) -> List[str]:
    """Normalize supplied passage references without inventing judgments.

    Missing or empty judgments remain an empty list. This function handles
    identifier formatting only; it does not check corpus availability.
    """
    if raw_passages is None:
        return []

    if not isinstance(raw_passages, list):
        raise ValueError("Passage judgments must be a list or null")

    return [
        _normalize_passage_id(passage)
        for passage in raw_passages
        if passage is not None and str(passage).strip()
    ]


def _parse_records(path: Path) -> List[dict]:
    """Read a JSON array or JSONL file into a list of records."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        return []

    # Official QReCC files contain one JSON array.
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # SCAI files contain one JSON object per line.
        data = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}"
                )
            data.append(record)

    if isinstance(data, dict):
        # Also tolerate a wrapper object containing a records array.
        for key in ("data", "records", "turns"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError(
                f"Expected a JSON array or JSONL records in {path}"
            )

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array or JSONL records in {path}")

    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Every record in {path} must be a JSON object")

    return data


def load_turns(
    json_path: str | Path,
    split: str,
    max_conversations: Optional[int] = None,
    extract_questions_only: bool = True,
) -> List[ConversationTurn]:
    """Load official QReCC JSON or SCAI JSONL into ConversationTurn objects.

    Args:
        json_path: Path to the dataset file.
        split: Dataset split label, such as 'train' or 'test'.
        max_conversations: Optional limit on distinct conversations.
        extract_questions_only: Keep only prior questions in history if True.

    Notes:
        Empty passage judgments remain empty. They do not mean that every
        passage is irrelevant. Passage ID normalization does not verify
        whether the referenced passage exists in the local corpus.
    """
    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(
            f"QReCC data file not found: {path}\n"
            "Check the dataset path in your project configuration."
        )

    if max_conversations is not None and max_conversations < 1:
        raise ValueError("max_conversations must be >= 1 or None")

    logger.info("Loading QReCC turns from %s (split=%s)", path, split)
    raw_records = _parse_records(path)
    logger.info("Raw file contains %d turn entries.", len(raw_records))

    turns: List[ConversationTurn] = []
    seen_conv_ids: set[str] = set()

    for item in raw_records:
        required_fields = (_F_CONV_ID, _F_TURN_ID, _F_QUESTION)
        missing = [field for field in required_fields if field not in item]
        if missing:
            raise KeyError(
                f"Missing required field(s) {missing} in {path}"
            )

        conv_id = str(item[_F_CONV_ID])

        if conv_id not in seen_conv_ids:
            if (
                max_conversations is not None
                and len(seen_conv_ids) >= max_conversations
            ):
                break
            seen_conv_ids.add(conv_id)

        question = item[_F_QUESTION]
        if not isinstance(question, str) or not question.strip():
            logger.warning(
                "Skipping empty question in conversation %s, turn %s",
                conv_id,
                item[_F_TURN_ID],
            )
            continue

        context_raw = item.get(_F_CONTEXT) or []
        if not isinstance(context_raw, list):
            raise ValueError(
                f"Context must be a list for conversation {conv_id}, "
                f"turn {item[_F_TURN_ID]}"
            )

        history = _extract_history(
            context_raw,
            extract_questions_only=extract_questions_only,
        )

        raw_rewrite = _first_present(item, _REWRITE_FIELDS)
        rewritten_question = (
            str(raw_rewrite).strip()
            if raw_rewrite is not None and str(raw_rewrite).strip()
            else None
        )

        raw_passages = _first_present(item, _PASSAGE_FIELDS)
        passage_ids = _normalize_passage_ids(raw_passages)

        turn = ConversationTurn(
            conversation_id=conv_id,
            turn_id=int(item[_F_TURN_ID]),
            question=question,
            relevant_passage_ids=passage_ids,
            conversation_history=history,
            rewritten_question=rewritten_question,
            split=split,
        )
        turns.append(turn)

    turns.sort(key=lambda t: (t.conversation_id, t.turn_id))

    logger.info(
        "Loaded %d turns from %d conversations (split=%s).",
        len(turns),
        len({turn.conversation_id for turn in turns}),
        split,
    )

    if split == "test":
        no_judgments = sum(
            1 for turn in turns if not turn.relevant_passage_ids
        )
        if no_judgments:
            logger.warning(
                "%d test turns have no passage-level judgments.",
                no_judgments,
            )

    return turns


def load_train_and_test(
    data_dir: str | Path,
    max_conversations: Optional[int] = None,
    extract_questions_only: bool = True,
) -> Dict[str, List[ConversationTurn]]:
    """Load train and test files, detecting the available filenames."""
    data_dir = Path(data_dir)

    train_candidates = (
        data_dir / "qrecc-training.json",
        data_dir / "train.jsonl",
        data_dir / "train.json",
    )
    test_candidates = (
        data_dir / "qrecc-test.json",
        data_dir / "test.jsonl",
        data_dir / "test.json",
    )

    train_path = next((p for p in train_candidates if p.exists()), None)
    test_path = next((p for p in test_candidates if p.exists()), None)

    if train_path is None:
        raise FileNotFoundError(
            f"No supported training file found in {data_dir}"
        )
    if test_path is None:
        raise FileNotFoundError(
            f"No supported test file found in {data_dir}"
        )

    return {
        "train": load_turns(
            train_path,
            "train",
            max_conversations=max_conversations,
            extract_questions_only=extract_questions_only,
        ),
        "test": load_turns(
            test_path,
            "test",
            max_conversations=max_conversations,
            extract_questions_only=extract_questions_only,
        ),
    }


def print_dataset_stats(
    turns: List[ConversationTurn],
    label: str = "",
) -> None:
    """Print a compact summary of loaded turns."""
    if not turns:
        print(f"[{label}] No turns loaded.")
        return

    conv_ids = {turn.conversation_id for turn in turns}
    has_passages = sum(
        1 for turn in turns if turn.relevant_passage_ids
    )
    has_rewrite = sum(
        1 for turn in turns if turn.rewritten_question
    )
    splits = {turn.split for turn in turns}

    print(f"\n{'=' * 60}")
    print(f"QReCC Dataset Summary: {label}")
    print(f"Turns: {len(turns)}")
    print(f"Conversations: {len(conv_ids)}")
    print(f"Turns with passage judgments: {has_passages}")
    print(f"Turns with reference rewrites: {has_rewrite}")
    print(f"Splits: {', '.join(sorted(str(s) for s in splits))}")
    print(f"{'=' * 60}")
