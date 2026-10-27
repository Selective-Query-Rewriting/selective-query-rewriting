
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_FILES = {
    "train": Path("data/qrecc/train.jsonl"),
    "test": Path("data/qrecc/test.jsonl"),
}

CORPUS_DIR = Path("data/passages/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Reference normalization
# --------------------------------------------------

def normalize_reference(value):
    value = str(value).strip()

    # Extract URL target from a Markdown link.
    match = re.search(r"\]\((https?://.*)\)$", value)
    if match:
        value = match.group(1)

    # Remove Markdown escapes before punctuation.
    value = re.sub(
        r"\\([\\`*_{}\[\]()#+\-.!|>~:&])",
        r"\1",
        value,
    )

    return value.strip()


# --------------------------------------------------
# Load QReCC turns and references
# --------------------------------------------------

def load_turns(path, split):
    turns = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            passages = record.get("Truth_passages")
            if passages is None:
                passages = record.get("Passages", [])

            references = {
                normalize_reference(p)
                for p in (passages or [])
                if str(p).strip()
            }

            turns.append({
                "split": split,
                "line_number": line_number,
                "conversation_id": str(record.get("Conversation_no", "")),
                "turn_id": str(record.get("Turn_no", "")),
                "question": str(record.get("Question", "")),
                "references": sorted(references),
            })

    return turns


# --------------------------------------------------
# Scan passage IDs in local corpus
# --------------------------------------------------

def load_corpus_ids():
    corpus_ids = set()
    parquet_files = sorted(CORPUS_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in {CORPUS_DIR}"
        )

    for path in parquet_files:
        print(f"Scanning corpus IDs: {path.name}")
        parquet = pq.ParquetFile(path)

        for batch in parquet.iter_batches(
            columns=["id"],
            batch_size=50_000,
        ):
            for raw_id in batch.column("id").to_pylist():
                corpus_ids.add(normalize_reference(raw_id))

    print(f"\nUnique local corpus IDs: {len(corpus_ids):,}")
    return corpus_ids


# --------------------------------------------------
# Analyze each turn
# --------------------------------------------------

def analyze_split(turns, corpus_ids):
    stats = {
        "total_turns": len(turns),
        "turns_with_references": 0,
        "turns_without_references": 0,
        "all_references_available": 0,
        "some_references_missing": 0,
        "no_references_available": 0,
        "total_reference_entries": 0,
        "matched_reference_entries": 0,
    }

    eligible_turns = []
    report_rows = []

    for turn in turns:
        refs = turn["references"]
        available = [ref for ref in refs if ref in corpus_ids]
        missing = [ref for ref in refs if ref not in corpus_ids]

        stats["total_reference_entries"] += len(refs)
        stats["matched_reference_entries"] += len(available)

        if not refs:
            stats["turns_without_references"] += 1
            status = "no_references"
        else:
            stats["turns_with_references"] += 1

            if not missing:
                stats["all_references_available"] += 1
                status = "all_available"

                # Eligible only when at least one reference exists.
                eligible_turns.append({
                    **turn,
                    "references": refs,
                })

            elif available:
                stats["some_references_missing"] += 1
                status = "some_missing"
            else:
                stats["no_references_available"] += 1
                status = "none_available"

        report_rows.append({
            "split": turn["split"],
            "line_number": turn["line_number"],
            "conversation_id": turn["conversation_id"],
            "turn_id": turn["turn_id"],
            "question": turn["question"],
            "reference_count": len(refs),
            "matched_count": len(available),
            "missing_count": len(missing),
            "status": status,
            "missing_references": missing,
        })

    return stats, eligible_turns, report_rows


# --------------------------------------------------
# Print and save results
# --------------------------------------------------

def print_stats(split, stats):
    print(f"\n{'=' * 55}")
    print(f"{split.upper()} COVERAGE")
    print("=" * 55)

    for key, value in stats.items():
        print(f"{key}: {value:,}")

    total_refs = stats["total_reference_entries"]
    matched_refs = stats["matched_reference_entries"]

    if total_refs:
        coverage = 100 * matched_refs / total_refs
        print(f"Reference-entry coverage: {coverage:.2f}%")

    total_turns = stats["total_turns"]
    eligible = stats["all_references_available"]

    if total_turns:
        print(
            "Turns with all references available: "
            f"{eligible / total_turns * 100:.2f}%"
        )


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    # Load references from both splits.
    all_turns = []

    for split, path in DATASET_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        turns = load_turns(path, split)
        all_turns.extend(turns)
        print(f"Loaded {len(turns):,} turns from {path}")

    # Build a set of all passage IDs available locally.
    corpus_ids = load_corpus_ids()

    # Analyze train and test independently.
    for split in DATASET_FILES:
        turns = [t for t in all_turns if t["split"] == split]

        stats, eligible, report = analyze_split(turns, corpus_ids)

        print_stats(split, stats)

        save_jsonl(
            OUTPUT_DIR / f"{split}_coverage_report.jsonl",
            report,
        )

        save_jsonl(
            OUTPUT_DIR / f"{split}_eligible_turns.jsonl",
            eligible,
        )

        print(f"\nSaved report: data/processed/{split}_coverage_report.jsonl")
        print(f"Saved eligible turns: data/processed/{split}_eligible_turns.jsonl")


if __name__ == "__main__":
    main()
