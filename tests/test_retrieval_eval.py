"""tests/test_retrieval_eval.py — Unit tests for Phase 4 retrieval evaluation pipeline."""

import json
import sqlite3
from pathlib import Path

from scripts.run_retrieval_eval import run_evaluation


def create_mock_bm25_db(db_path: Path):
    """Create a temporary SQLite FTS5 passages database for testing."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE VIRTUAL TABLE passages USING fts5(
            passage_id UNINDEXED,
            contents,
            tokenize = 'porter unicode61'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    passages = [
        ("doc_101", "Inception is a science fiction film about dreams directed by Christopher Nolan."),
        ("doc_102", "Christopher Nolan directed Inception, Interstellar, and The Dark Knight."),
        ("doc_103", "Solar energy is renewable energy obtained by harnessing radiant light from the sun."),
        ("doc_104", "The disadvantages of solar energy include intermittency and high upfront installation costs."),
    ]
    conn.executemany("INSERT INTO passages (passage_id, contents) VALUES (?, ?)", passages)
    conn.execute(
        "INSERT INTO index_metadata (key, value) VALUES ('index_type', 'SQLite FTS5 BM25')"
    )
    conn.commit()
    conn.close()


def create_mock_pilot_jsonl(file_path: Path):
    """Create a mock formulation_pilot.jsonl with 3 sample turns."""
    turns = [
        {
            "turn_id": "1_1",
            "conversation_id": "1",
            "turn_number": 1,
            "question": "Tell me about Inception movie.",
            "relevant_passage_ids": ["doc_101"],
            "formulations": {
                "keep": "Tell me about Inception movie.",
                "concat": "Tell me about Inception movie.",
                "rewrite": "Tell me about the sci-fi movie Inception.",
            },
            "rewrite_status": "success",
        },
        {
            "turn_id": "1_2",
            "conversation_id": "1",
            "turn_number": 2,
            "question": "Who directed it?",
            "relevant_passage_ids": ["doc_102"],
            "formulations": {
                "keep": "Who directed it?",
                "concat": "Tell me about Inception movie. Who directed it?",
                "rewrite": "Who directed the movie Inception?",
            },
            "rewrite_status": "success",
        },
        {
            "turn_id": "2_1",
            "conversation_id": "2",
            "turn_number": 1,
            "question": "What about solar disadvantages?",
            "relevant_passage_ids": ["doc_104"],
            "formulations": {
                "keep": "What about solar disadvantages?",
                "concat": "What about solar disadvantages?",
                "rewrite": None,
            },
            "rewrite_status": "failed",
            "rewrite_error": "Token limit exhausted",
        },
    ]
    with file_path.open("w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")


def test_run_evaluation_pipeline(tmp_path):
    index_path = tmp_path / "mock_bm25.sqlite"
    pilot_path = tmp_path / "mock_pilot.jsonl"
    output_dir = tmp_path / "results"

    create_mock_bm25_db(index_path)
    create_mock_pilot_jsonl(pilot_path)

    result = run_evaluation(
        pilot_path=pilot_path,
        index_path=index_path,
        output_dir=output_dir,
        top_k=5,
        primary_metric="ndcg@10",
        tie_margin=0.01,
        run_id="test_run",
    )

    # 1. Check generated files
    assert (output_dir / "retrieval_per_query.jsonl").exists()
    assert (output_dir / "formulation_summary.json").exists()
    assert (output_dir / "outcome_labels.jsonl").exists()
    assert (output_dir / "oracle_summary.json").exists()

    # 2. Check summary contents
    summary = result["summary"]
    assert summary["total_eligible_turns"] == 3
    assert summary["successful_rewrite_turns"] == 2
    assert summary["failed_rewrite_turns"] == 1
    assert summary["matched_subset_denominators"] == 2

    # 3. Check outcomes
    outcomes_file = output_dir / "outcome_labels.jsonl"
    lines = [json.loads(line) for line in outcomes_file.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2  # Only the 2 successful turns evaluated for outcomes
    assert {l["turn_id"] for l in lines} == {"1_1", "1_2"}

    # 4. Check oracle summary
    oracle = result["oracle"]
    assert oracle["n_evaluated"] == 2
    assert "keep_vs_rewrite_oracle" in oracle
    assert oracle["keep_vs_rewrite_oracle"]["mean_oracle_score"] >= 0.0
