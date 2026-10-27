
"""Retrieve passages using a SQLite FTS5 BM25 index.

The index must first be created by src.retrieval.build_index.

Returns a list of (passage_id, score) tuples, ordered from best
to worst. Scores are negated SQLite BM25 scores so higher is better.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = (
    PROJECT_ROOT / "data" / "index" / "bm25_fts5_dedup_full.sqlite"
)


def tokenize_query(query: str) -> list[str]:
    """Tokenize a query using a simple alphanumeric baseline."""
    return re.findall(r"\w+", query.lower(), flags=re.UNICODE)


def build_match_expression(query: str) -> str:
    """Build a safe FTS5 OR query from the query tokens."""
    tokens = tokenize_query(query)

    if not tokens:
        return ""

    # Quote each token to avoid interpreting user text as FTS syntax.
    quoted_tokens = [
        '"' + token.replace('"', '""') + '"'
        for token in tokens
    ]

    return " OR ".join(quoted_tokens)


class BM25Retriever:
    """Reusable connection to a built SQLite FTS5 index."""

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
    ) -> None:
        self.index_path = Path(index_path)

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {self.index_path}\n"
                "Build it first with src.retrieval.build_index."
            )

        self.connection = sqlite3.connect(
            str(self.index_path)
        )

        # Verify that the expected index table exists.
        try:
            self.connection.execute(
                "SELECT passage_id FROM passages LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError as exc:
            self.connection.close()
            raise RuntimeError(
                "The database does not contain the expected "
                "FTS5 passages table."
            ) from exc

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return ranked (passage_id, score) pairs."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        match_expression = build_match_expression(query)

        if not match_expression:
            return []

        rows = self.connection.execute(
            """
            SELECT passage_id, bm25(passages) AS raw_score
            FROM passages
            WHERE passages MATCH ?
            ORDER BY raw_score ASC
            LIMIT ?
            """,
            (match_expression, top_k),
        ).fetchall()

        # SQLite FTS5 gives better matches lower (often more negative)
        # BM25 scores. Negate them for a higher-is-better API.
        return [
            (str(passage_id), -float(raw_score))
            for passage_id, raw_score in rows
        ]

    def metadata(self) -> dict[str, str]:
        """Read the settings recorded during index construction."""
        try:
            rows = self.connection.execute(
                "SELECT key, value FROM index_metadata"
            ).fetchall()
        except sqlite3.OperationalError:
            return {}

        return {
            str(key): str(value)
            for key, value in rows
        }

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()

    def __enter__(self) -> "BM25Retriever":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def retrieve(
    query: str,
    top_k: int = 10,
    index_path: Optional[Path] = None,
) -> list[tuple[str, float]]:
    """Convenience function for a single retrieval call."""
    path = index_path or DEFAULT_INDEX_PATH

    with BM25Retriever(path) as retriever:
        return retriever.retrieve(query, top_k=top_k)
