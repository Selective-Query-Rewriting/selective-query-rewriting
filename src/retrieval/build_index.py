
"""Build a disk-backed BM25 index using SQLite FTS5.

Phase 2: Retrieval foundation.

Reads passage IDs and contents from Parquet files and stores a
searchable index in SQLite.

Duplicate policy:
- Parquet files are processed in sorted filename order.
- The first non-empty occurrence of each normalized passage ID wins.
- Later occurrences of that ID are skipped, even if their contents differ.
- This makes indexing deterministic and prevents duplicate passage IDs
  from occupying multiple retrieval positions.

By default, indexes 10,000 unique passages for a smoke test.
Use --max-passages 0 to index all unique passages.

The corpus and resulting index should not be committed to Git.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "passages" / "raw"
DEFAULT_INDEX_PATH = (
    PROJECT_ROOT / "data" / "index" / "bm25_fts5_dedup_full.sqlite"
)
BATCH_SIZE = 2_000


def normalize_passage_id(raw_id: str) -> str:
    """Extract and normalize the URL from a Markdown-formatted passage ID."""
    value = str(raw_id).strip()

    match = re.search(r"\]\((https?://[^)]*)\)\s*$", value)

    if match:
        value = match.group(1)

    value = re.sub(
        r"\\([\\`*_{}\[\]()#+\-.!|>~:&])",
        r"\1",
        value,
    )

    return value.strip()


def check_fts5(connection: sqlite3.Connection) -> None:
    """Fail early if this Python SQLite build lacks FTS5."""
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE temp.fts5_check USING fts5(content)"
        )
        connection.execute("DROP TABLE temp.fts5_check")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "SQLite FTS5 is unavailable in this Python installation."
        ) from exc


def create_schema(connection: sqlite3.Connection) -> None:
    """Create the full-text index and its metadata table."""
    connection.execute("""
        CREATE VIRTUAL TABLE passages USING fts5(
            passage_id UNINDEXED,
            contents,
            tokenize = 'porter unicode61'
        )
    """)

    connection.execute("""
        CREATE TABLE index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)


def build_index(
    corpus_dir: Path,
    index_path: Path,
    max_passages: int = 10_000,
    overwrite: bool = False,
) -> int:
    """Build the index and return the number of unique indexed passages."""
    if max_passages < 0:
        raise ValueError("max_passages must be >= 0")

    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory does not exist: {corpus_dir}"
        )

    parquet_files = sorted(corpus_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files found in: {corpus_dir}"
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Index already exists: {index_path}\n"
                "Choose another path or pass --overwrite to rebuild it."
            )
        index_path.unlink()

    connection = sqlite3.connect(str(index_path))

    indexed_count = 0
    skipped_empty_count = 0
    skipped_duplicate_count = 0
    pending_rows = []

    # Tracks IDs already encountered in the deterministic shard order.
    seen_passage_ids = set()

    try:
        check_fts5(connection)
        create_schema(connection)

        for parquet_path in parquet_files:
            print(f"\nReading: {parquet_path.name}")
            parquet_file = pq.ParquetFile(parquet_path)

            for batch in parquet_file.iter_batches(
                columns=["id", "contents"],
                batch_size=BATCH_SIZE,
            ):
                ids = batch.column("id").to_pylist()
                contents = batch.column("contents").to_pylist()

                for raw_id, raw_contents in zip(ids, contents):
                    if max_passages and indexed_count >= max_passages:
                        break

                    if raw_id is None:
                        skipped_empty_count += 1
                        continue

                    passage_id = normalize_passage_id(raw_id)
                    text = (
                        str(raw_contents).strip()
                        if raw_contents is not None
                        else ""
                    )

                    if not passage_id or not text:
                        skipped_empty_count += 1
                        continue

                    if passage_id in seen_passage_ids:
                        skipped_duplicate_count += 1
                        continue

                    # Reserve the ID only after validating its contents.
                    seen_passage_ids.add(passage_id)
                    pending_rows.append((passage_id, text))
                    indexed_count += 1

                    if len(pending_rows) >= BATCH_SIZE:
                        connection.executemany(
                            """
                            INSERT INTO passages (passage_id, contents)
                            VALUES (?, ?)
                            """,
                            pending_rows,
                        )
                        connection.commit()
                        pending_rows.clear()

                        print(f"Indexed {indexed_count:,} unique passages...")

                if max_passages and indexed_count >= max_passages:
                    break

            if max_passages and indexed_count >= max_passages:
                break

        if pending_rows:
            connection.executemany(
                """
                INSERT INTO passages (passage_id, contents)
                VALUES (?, ?)
                """,
                pending_rows,
            )
            connection.commit()

        metadata = {
            "index_type": "SQLite FTS5 BM25",
            "tokenizer": "porter unicode61",
            "corpus_dir": str(corpus_dir.resolve()),
            "indexed_passages": str(indexed_count),
            "max_passages": str(max_passages),
            "batch_size": str(BATCH_SIZE),
            "duplicate_policy": "first_valid_occurrence_by_normalized_id",
            "duplicate_ids_skipped": str(skipped_duplicate_count),
            "empty_rows_skipped": str(skipped_empty_count),
            "shard_order": ",".join(path.name for path in parquet_files),
        }

        connection.executemany(
            "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
            list(metadata.items()),
        )
        connection.commit()

    except Exception:
        connection.close()

        if index_path.exists():
            index_path.unlink()

        raise

    else:
        connection.close()

    print("\nIndex build complete.")
    print(f"Unique passages indexed: {indexed_count:,}")
    print(f"Empty/invalid rows skipped: {skipped_empty_count:,}")
    print(f"Duplicate-ID rows skipped: {skipped_duplicate_count:,}")
    print(f"Index location: {index_path}")
    print(f"Index size: {index_path.stat().st_size / (1024 ** 2):.2f} MB")

    return indexed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a disk-backed BM25 passage index."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Directory containing Parquet passage shards.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Output SQLite index path.",
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=10_000,
        help="Maximum unique passages to index; 0 means all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing index at the output path.",
    )

    args = parser.parse_args()

    build_index(
        corpus_dir=args.corpus_dir,
        index_path=args.index_path,
        max_passages=args.max_passages,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
