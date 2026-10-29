"""tests/test_pipeline_reproducibility.py — Unit tests for Phase 7 reproducible research pipeline."""

import json
from pathlib import Path

from scripts.run_reproducible_pipeline import generate_paper_tables, run_pipeline
from tests.test_retrieval_eval import create_mock_bm25_db, create_mock_pilot_jsonl


def test_reproducible_pipeline_end_to_end(tmp_path):
    index_path = tmp_path / "mock_bm25.sqlite"
    pilot_path = tmp_path / "mock_pilot.jsonl"
    cache_path = tmp_path / "mock_cache.jsonl"
    config_path = tmp_path / "experiment.yaml"
    out_dir = tmp_path / "results"

    create_mock_bm25_db(index_path)
    create_mock_pilot_jsonl(pilot_path)

    # Mock cache entries
    mock_cache = [
        {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65, "latency_ms": 320.0, "retry_used": False},
        {"prompt_tokens": 60, "completion_tokens": 20, "total_tokens": 80, "latency_ms": 410.0, "retry_used": True},
    ]
    with cache_path.open("w", encoding="utf-8") as f:
        for c in mock_cache:
            f.write(json.dumps(c) + "\n")

    # Mock config
    config_content = """
retrieval:
  top_k: 10
evaluation:
  primary_metric: "ndcg@10"
  tie_margin: 0.01
output:
  results_dir: "outputs/results"
"""
    config_path.write_text(config_content, encoding="utf-8")

    # Run end-to-end pipeline
    summary = run_pipeline(
        config_path=config_path,
        pilot_path=pilot_path,
        index_path=index_path,
        cache_path=cache_path,
        output_dir=out_dir,
        seed=42,
    )

    # Verify all artifacts exist
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "retrieval_per_query.jsonl").exists()
    assert (out_dir / "formulation_summary.json").exists()
    assert (out_dir / "outcome_labels.jsonl").exists()
    assert (out_dir / "oracle_summary.json").exists()
    assert (out_dir / "gate_predictions.jsonl").exists()
    assert (out_dir / "gate_summary.json").exists()
    assert (out_dir / "efficiency_summary.json").exists()
    assert (out_dir / "efficiency_log.jsonl").exists()
    assert (out_dir / "final_summary.json").exists()
    assert (out_dir / "paper_tables.md").exists()

    # Check generated Markdown tables
    tables_content = (out_dir / "paper_tables.md").read_text(encoding="utf-8")
    assert "Table 1: Formulation Performance" in tables_content
    assert "Table 2: Help / Tie / Hurt Outcomes" in tables_content
    assert "Table 3: Gating Model Comparison" in tables_content
    assert "Table 4: Latency and Token Overhead" in tables_content
