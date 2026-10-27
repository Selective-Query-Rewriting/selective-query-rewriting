
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

DATASET_FILES = [
    Path("data/qrecc/train.jsonl"),
    Path("data/qrecc/test.jsonl"),
]
CORPUS_DIR = Path("data/passages/raw")



def normalize_reference(value):
    value = str(value).strip()

    # Extract the URL target from a Markdown link.
    match = re.fullmatch(r"\[.*?\]\((.*?)\)", value)
    if match:
        value = match.group(1)

    # Remove Markdown backslash escapes consistently.
    value = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~:&])", r"\1", value)

    return value.strip()




references = set()
reference_entries = 0

for path in DATASET_FILES:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            passages = record.get("Truth_passages")
            if passages is None:
                passages = record.get("Passages", [])
            passages = passages or []

            for passage in passages:
                references.add(normalize_reference(passage))
                reference_entries += 1

print("Reference entries:", reference_entries)
print("Unique references:", len(references))

matched = set()

for path in sorted(CORPUS_DIR.glob("*.parquet")):
    parquet = pq.ParquetFile(path)
    shard_matches = set()

    for batch in parquet.iter_batches(
        columns=["id"],
        batch_size=50_000,
    ):
        for raw_id in batch.column("id").to_pylist():
            normalized_id = normalize_reference(raw_id)
            if normalized_id in references:
                shard_matches.add(normalized_id)

    new_matches = shard_matches - matched
    matched.update(shard_matches)

    print(
        f"{path.name}: {len(shard_matches)} matched "
        f"({len(new_matches)} new)"
    )

print("\nTotal matched:", len(matched))
print("Total references:", len(references))

if references:
    print(f"Coverage: {100 * len(matched) / len(references):.2f}%")

print("\nUnmatched examples:")
for reference in sorted(references - matched)[:10]:
    print(reference)
