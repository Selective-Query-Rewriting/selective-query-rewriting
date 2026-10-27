
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


DATASET_FILES = [
    Path("data/qrecc/train.jsonl"),
    Path("data/qrecc/test.jsonl"),
]

CORPUS_DIR = Path("data/passages/raw")

TARGET = "http://0gomovies123.com/ai-weiwei-never-sorry/_p0"


def normalize_reference(value):
    value = str(value).strip()

    # Extract the URL from a Markdown link.
    # Example: [label](URL) -> URL
    match = re.search(r"\]\((https?://.*)\)$", value)

    if match:
        value = match.group(1)

    # Remove Markdown escaping.
    value = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~:&])", r"\1", value)

    return value.strip()


print("=" * 60)
print("STEP 1: DATASET REFERENCE")
print("=" * 60)

dataset_found = False

for path in DATASET_FILES:
    if not path.exists():
        print(f"Missing dataset file: {path}")
        continue

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            passages = record.get("Truth_passages")

            if passages is None:
                passages = record.get("Passages", [])

            for passage in passages or []:
                normalized = normalize_reference(passage)

                if normalized == TARGET:
                    dataset_found = True
                    print("Dataset file:", path)
                    print("Line:", line_number)
                    print("Original:", repr(passage))
                    print("Normalized:", repr(normalized))
                    print("Exact match:", normalized == TARGET)


print("\n" + "=" * 60)
print("STEP 2: CORPUS SEARCH")
print("=" * 60)

corpus_found = False
domain_matches = 0

for path in sorted(CORPUS_DIR.glob("*.parquet")):
    print("Scanning:", path.name)

    parquet = pq.ParquetFile(path)

    for batch in parquet.iter_batches(
        columns=["id"],
        batch_size=50_000,
    ):
        for raw_id in batch.column("id").to_pylist():
            normalized_id = normalize_reference(raw_id)

            if "0gomovies123.com" in normalized_id:
                domain_matches += 1

            if normalized_id == TARGET:
                corpus_found = True
                print("\nEXACT CORPUS MATCH")
                print("File:", path.name)
                print("Original ID:", repr(raw_id))
                print("Normalized ID:", repr(normalized_id))


print("\n" + "=" * 60)
print("STEP 3: SUMMARY")
print("=" * 60)

print("Target:", TARGET)
print("Found in dataset:", dataset_found)
print("Found in corpus:", corpus_found)
print("Corpus IDs from this domain:", domain_matches)

if dataset_found and corpus_found:
    print("Result: Exact reference exists in both dataset and corpus.")
elif dataset_found:
    print("Result: Reference is in the dataset but not in scanned shards.")
else:
    print("Result: Target reference was not found in the dataset.")
