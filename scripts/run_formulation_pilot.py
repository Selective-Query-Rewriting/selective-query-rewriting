
"""Generate KEEP, CONCAT, and REWRITE formulations for a QReCC pilot."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.data.load_qrecc import load_turns
from src.rewriting.cache import RewriteCache
from src.rewriting.formulate import (
    formulate_keep,
    formulate_concat,
    formulate_with_cache,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    """Load experiment settings."""
    config_path = PROJECT_ROOT / "configs" / "experiment.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_eligible_ids(path: Path, max_conversations: int) -> set[str]:
    """Read eligible turn IDs belonging to the selected conversations."""
    eligible_ids = set()
    selected_conversations = set()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)
            conversation_id = str(row["conversation_id"])

            if (
                conversation_id not in selected_conversations
                and len(selected_conversations) >= max_conversations
            ):
                continue

            selected_conversations.add(conversation_id)
            eligible_ids.add(
                f"{conversation_id}_{int(row['turn_id'])}"
            )

    return eligible_ids


def main() -> None:
    config = load_config()

    dataset_config = config["dataset"]
    formulation_config = config["formulations"]
    rewrite_config = config["rewriting"]

    data_dir = PROJECT_ROOT / "data" / "qrecc"
    train_path = data_dir / "train.jsonl"

    eligible_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "train_eligible_turns.jsonl"
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}"
        )

    if not eligible_path.exists():
        raise FileNotFoundError(
            f"Eligible-turn file not found: {eligible_path}"
        )

    max_conversations = int(
        dataset_config["pilot_max_convs"]
    )

    # Load the selected training conversations.
    turns = load_turns(
        json_path=train_path,
        split="train",
        max_conversations=max_conversations,
        extract_questions_only=dataset_config.get(
            "extract_questions_only", True
        ),
    )

    eligible_ids = load_eligible_ids(
        eligible_path,
        max_conversations=max_conversations,
    )

    pilot_turns = [
        turn for turn in turns
        if turn.unique_id in eligible_ids
    ]

    if not pilot_turns:
        raise RuntimeError(
            "No eligible turns matched the pilot conversations."
        )

    print(f"Loaded turns: {len(turns)}")
    print(f"Eligible pilot turns: {len(pilot_turns)}")

    # Initialize the rewrite cache only when REWRITE is enabled.
    rewrite_cache = None

    if formulation_config.get("run_rewrite", True):
        rewrite_cache = RewriteCache(
            cache_file=str(
                PROJECT_ROOT / rewrite_config["cache_file"]
            ),
            provider=rewrite_config["provider"],
            model=rewrite_config["model"],
            max_tokens=int(
                rewrite_config.get("max_tokens", 256)
            ),
            temperature=float(
                rewrite_config.get("temperature", 0.0)
            ),
        )

    output_dir = (
        PROJECT_ROOT
        / config["output"]["results_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "formulation_pilot.jsonl"

    concat_max_history = int(
        formulation_config.get(
            "concat_max_history_turns", 3
        )
    )

    success_count = 0
    failure_count = 0

    # Write one record per turn, containing all enabled formulations.
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, turn in enumerate(pilot_turns, start=1):
            result = {
                "turn_id": turn.unique_id,
                "conversation_id": turn.conversation_id,
                "turn_number": turn.turn_id,
                "question": turn.question,
                "history": turn.conversation_history,
                "relevant_passage_ids": turn.relevant_passage_ids,
                "formulations": {},
            }

            # KEEP: use the original question.
            if formulation_config.get("run_keep", True):
                result["formulations"]["keep"] = formulate_keep(
                    turn
                )

            # CONCAT: prepend recent conversation history.
            if formulation_config.get("run_concat", True):
                result["formulations"]["concat"] = formulate_concat(
                    turn,
                    max_history_turns=concat_max_history,
                )

            # REWRITE: generate or reuse a cached LLM rewrite.
            if formulation_config.get("run_rewrite", True):
                try:
                    result["formulations"]["rewrite"] = (
                        formulate_with_cache(
                            turn,
                            method="rewrite",
                            rewrite_cache=rewrite_cache,
                        )
                    )
                    result["rewrite_status"] = "success"
                    success_count += 1

                except ValueError as error:
                    result["formulations"]["rewrite"] = None
                    result["rewrite_status"] = "failed"
                    result["rewrite_error"] = str(error)
                    failure_count += 1

                    print(
                        f"\nWARNING: Rewrite failed for "
                        f"{turn.unique_id}: {error}"
                    )

            # Save each turn immediately, including failed rewrites.
            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            if index % 10 == 0 or index == len(pilot_turns):
                print(
                    f"Processed {index}/{len(pilot_turns)} "
                    f"eligible turns"
                )

    print("\nPilot formulation generation complete.")
    print(f"Output: {output_path}")
    print(f"Turns processed: {len(pilot_turns)}")

    if formulation_config.get("run_rewrite", True):
        print(f"Successful rewrites: {success_count}")
        print(f"Failed rewrites: {failure_count}")

    enabled_formulations = [
        name
        for name, enabled in [
            ("KEEP", formulation_config.get("run_keep", True)),
            ("CONCAT", formulation_config.get("run_concat", True)),
            ("REWRITE", formulation_config.get("run_rewrite", True)),
        ]
        if enabled
    ]

    print(
        "Formulations: " + ", ".join(enabled_formulations)
    )


if __name__ == "__main__":
    main()
