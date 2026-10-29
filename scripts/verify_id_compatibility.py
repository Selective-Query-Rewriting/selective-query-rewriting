"""Verify ID compatibility between ground-truth passage IDs and the BM25 index.

Phase 4 pre-flight check.
Before running full retrieval evaluation, this script inspects a sample of turns
to ensure that the passage IDs in the dataset relevance judgments match the format
of passage IDs stored in the SQLite BM25 index.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.retrieval.retrieve import BM25Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT_ROOT / "data" / "index" / "bm25_fts5_dedup_full.sqlite"
DEFAULT_PILOT = PROJECT_ROOT / "outputs" / "results" / "formulation_pilot.jsonl"


def verify_id_compatibility(
    index_path: Path,
    turns_path: Path,
    sample_size: int = 5,
    top_k: int = 10,
) -> dict:
    """Run smoke test on a sample of turns to inspect ID format and overlap."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at: {index_path}\n"
            "Ensure the index has been built or transferred before running this check."
        )

    if not turns_path.exists():
        raise FileNotFoundError(
            f"Turns file not found at: {turns_path}\n"
            "Ensure formulation pilot output exists before running this check."
        )

    logger.info("Connecting to BM25 index: %s", index_path)
    retriever = BM25Retriever(index_path=index_path)

    metadata = retriever.metadata()
    logger.info("Index metadata: %s", metadata)

    turns = []
    with turns_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                turns.append(json.loads(line))
            if len(turns) >= sample_size:
                break

    results = []
    total_hits = 0

    print("\n" + "=" * 70)
    print("PASSAGE ID COMPATIBILITY SMOKE TEST")
    print("=" * 70)

    for i, item in enumerate(turns, start=1):
        turn_id = item.get("turn_id")
        question = item.get("question")
        relevant_ids = set(item.get("relevant_passage_ids", []))

        # Test keep query
        query = item.get("formulations", {}).get("keep", question)
        retrieved = retriever.retrieve(query, top_k=top_k)
        retrieved_ids = [pid for pid, _score in retrieved]

        overlap = relevant_ids.intersection(retrieved_ids)
        total_hits += len(overlap)

        turn_report = {
            "turn_id": turn_id,
            "query": query,
            "relevant_count": len(relevant_ids),
            "relevant_sample": sorted(list(relevant_ids))[:2],
            "retrieved_sample": retrieved_ids[:2],
            "overlap_count": len(overlap),
        }
        results.append(turn_report)

        print(f"\n[{i}/{len(turns)}] Turn: {turn_id}")
        print(f"  Query:            {query!r}")
        print(f"  Relevant IDs (sample): {turn_report['relevant_sample']}")
        print(f"  Retrieved IDs (sample): {turn_report['retrieved_sample']}")
        print(f"  Hits in top-{top_k}:     {len(overlap)}")

    retriever.close()

    print("\n" + "=" * 70)
    print(f"Total evaluated turns: {len(turns)}")
    print(f"Total overlapping hits: {total_hits}")
    print("=" * 70)

    return {
        "metadata": metadata,
        "sample_size": len(turns),
        "total_hits": total_hits,
        "turns": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check BM25 passage ID compatibility.")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX, help="SQLite index path")
    parser.add_argument("--turns-path", type=Path, default=DEFAULT_PILOT, help="Path to pilot turns JSONL")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of turns to sample")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval depth for smoke check")
    args = parser.parse_args()

    verify_id_compatibility(
        index_path=args.index_path,
        turns_path=args.turns_path,
        sample_size=args.sample_size,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
