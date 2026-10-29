"""Per-query retrieval evaluation, help/tie/hurt outcomes, and oracle headroom.

Phase 4:
1. Loads pilot formulations (KEEP, CONCAT, REWRITE).
2. Performs BM25 retrieval for each available formulation.
3. Computes MRR@10, Recall@10, Recall@100, and nDCG@10 per query.
4. Generates outcome labels (helpful / tied / harmful).
5. Computes oracle upper bound and headroom over baselines.
6. Writes standard research artifacts:
   - outputs/results/retrieval_per_query.jsonl
   - outputs/results/formulation_summary.json
   - outputs/results/outcome_labels.jsonl
   - outputs/results/oracle_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from src.evaluation.metrics import compute_metrics
from src.evaluation.oracle import (
    mean_oracle_score,
    oracle_headroom_summary,
    oracle_score_per_turn,
)
from src.evaluation.outcomes import Outcome, label_outcome, summarise_outcomes
from src.retrieval.retrieve import BM25Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_evaluation(
    pilot_path: Path,
    index_path: Path,
    output_dir: Path,
    top_k: int = 100,
    metric_ks: Optional[List[int]] = None,
    primary_metric: str = "ndcg@10",
    tie_margin: float = 0.01,
    run_id: str = "phase4_eval",
) -> Dict[str, Any]:
    """Execute the retrieval evaluation pipeline."""
    if metric_ks is None:
        metric_ks = [10, 100]

    if not pilot_path.exists():
        raise FileNotFoundError(f"Pilot formulations file not found: {pilot_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"BM25 index file not found: {index_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = output_dir / "retrieval_per_query.jsonl"
    summary_path = output_dir / "formulation_summary.json"
    outcomes_path = output_dir / "outcome_labels.jsonl"
    oracle_path = output_dir / "oracle_summary.json"

    logger.info("Initializing BM25 retriever from %s", index_path)
    retriever = BM25Retriever(index_path=index_path)

    turns_data = []
    with pilot_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                turns_data.append(json.loads(line))

    logger.info("Loaded %d pilot turns for retrieval evaluation.", len(turns_data))

    per_query_records: List[dict] = []
    # Dict to collect per-formulation primary metric scores: formulation -> {turn_id: score}
    scores_by_formulation: Dict[str, Dict[str, float]] = {
        "keep": {},
        "concat": {},
        "rewrite": {},
    }
    # Track all metrics for summary computation
    all_metrics_by_formulation: Dict[str, Dict[str, List[float]]] = {
        "keep": {},
        "concat": {},
        "rewrite": {},
    }

    # Evaluate each turn
    for item in turns_data:
        turn_id = item["turn_id"]
        conv_id = item.get("conversation_id", turn_id.split("_")[0])
        relevant_ids: Set[str] = set(item.get("relevant_passage_ids", []))
        formulations = item.get("formulations", {})
        rewrite_status = item.get("rewrite_status", "unknown")

        for action in ("keep", "concat", "rewrite"):
            query = formulations.get(action)
            # Only evaluate rewrite if it succeeded and text is present
            if action == "rewrite" and (rewrite_status != "success" or not query):
                continue
            if not query:
                continue

            retrieved = retriever.retrieve(query, top_k=top_k)
            retrieved_ids = [pid for pid, _score in retrieved]
            retrieved_scores = [score for _pid, score in retrieved]

            # Compute retrieval metrics (MRR, Recall, nDCG)
            metrics = compute_metrics(retrieved_ids, relevant_ids, ks=metric_ks)

            record = {
                "turn_id": turn_id,
                "conversation_id": conv_id,
                "action": action,
                "formulated_query": query,
                "relevant_ids": sorted(list(relevant_ids)),
                "retrieved_ids": retrieved_ids[:10],  # store top-10 in log
                "top_score": retrieved_scores[0] if retrieved_scores else None,
                "metrics": metrics,
                "run_id": run_id,
                "rewrite_status": rewrite_status if action == "rewrite" else None,
            }
            per_query_records.append(record)

            if primary_metric in metrics:
                scores_by_formulation[action][turn_id] = metrics[primary_metric]

            for m_name, m_val in metrics.items():
                all_metrics_by_formulation[action].setdefault(m_name, []).append(m_val)

    retriever.close()

    # Save retrieval_per_query.jsonl
    logger.info("Writing per-query results to %s", per_query_path)
    with per_query_path.open("w", encoding="utf-8") as f:
        for r in per_query_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------
    # Compute Aggregates & Summaries
    # -------------------------------------------------------------
    total_turns = len(turns_data)
    successful_rewrite_turns = len(scores_by_formulation["rewrite"])
    failed_rewrite_turns = total_turns - successful_rewrite_turns

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "total_eligible_turns": total_turns,
        "successful_rewrite_turns": successful_rewrite_turns,
        "failed_rewrite_turns": failed_rewrite_turns,
        "primary_metric": primary_metric,
        "all_turns_macro_averages": {},
        "matched_subset_macro_averages": {},
    }

    # 1. All-turns macro averages (for KEEP and CONCAT)
    for act in ("keep", "concat"):
        act_summary = {"n": len(scores_by_formulation[act])}
        for m_name, vals in all_metrics_by_formulation[act].items():
            act_summary[f"mean_{m_name}"] = sum(vals) / len(vals) if vals else 0.0
        summary["all_turns_macro_averages"][act] = act_summary

    # 2. Matched-subset comparison (turns where REWRITE succeeded)
    matched_ids = sorted(list(set(scores_by_formulation["rewrite"].keys())))
    summary["matched_subset_denominators"] = len(matched_ids)

    for act in ("keep", "concat", "rewrite"):
        matched_scores = [
            scores_by_formulation[act][tid]
            for tid in matched_ids
            if tid in scores_by_formulation[act]
        ]
        summary["matched_subset_macro_averages"][act] = {
            "n": len(matched_scores),
            f"mean_{primary_metric}": (
                sum(matched_scores) / len(matched_scores) if matched_scores else 0.0
            ),
        }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------------------
    # Outcome Labeling (Helpful / Tied / Harmful)
    # -------------------------------------------------------------
    logger.info("Labeling outcomes using %s (tie_margin=%.4f)", primary_metric, tie_margin)
    outcome_records = []
    raw_outcomes = []

    for tid in matched_ids:
        k_score = scores_by_formulation["keep"].get(tid, 0.0)
        r_score = scores_by_formulation["rewrite"].get(tid, 0.0)
        outcome = label_outcome(k_score, r_score, tie_margin=tie_margin)
        delta = r_score - k_score

        rec = {
            "turn_id": tid,
            "keep_score": k_score,
            "rewrite_score": r_score,
            "delta": delta,
            "outcome": outcome.value,
        }
        outcome_records.append(rec)
        raw_outcomes.append(outcome)

    outcome_distribution = summarise_outcomes(
        [
            type("TurnOutcome", (), {"outcome": o})()
            for o in raw_outcomes
        ]
    )

    with outcomes_path.open("w", encoding="utf-8") as f:
        for o in outcome_records:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    logger.info("Outcome distribution: %s", outcome_distribution)

    # -------------------------------------------------------------
    # Oracle Headroom Analysis
    # -------------------------------------------------------------
    logger.info("Computing oracle headroom...")
    # Oracle 1: KEEP vs REWRITE on matched subset
    kr_scores = {
        "keep": {tid: scores_by_formulation["keep"][tid] for tid in matched_ids},
        "rewrite": {tid: scores_by_formulation["rewrite"][tid] for tid in matched_ids},
    }
    kr_oracle_per_turn = oracle_score_per_turn(kr_scores)
    kr_headroom_over_rewrite = oracle_headroom_summary(kr_scores, baseline_name="rewrite")
    kr_headroom_over_keep = oracle_headroom_summary(kr_scores, baseline_name="keep")

    # Oracle 2: KEEP vs CONCAT vs REWRITE on matched subset
    kcr_scores = {
        "keep": {tid: scores_by_formulation["keep"][tid] for tid in matched_ids},
        "concat": {tid: scores_by_formulation["concat"][tid] for tid in matched_ids},
        "rewrite": {tid: scores_by_formulation["rewrite"][tid] for tid in matched_ids},
    }
    kcr_oracle_per_turn = oracle_score_per_turn(kcr_scores)
    kcr_headroom_over_rewrite = oracle_headroom_summary(kcr_scores, baseline_name="rewrite")

    oracle_results = {
        "primary_metric": primary_metric,
        "n_evaluated": len(matched_ids),
        "keep_vs_rewrite_oracle": {
            "mean_oracle_score": mean_oracle_score(kr_scores),
            "baseline_keep_mean": kr_headroom_over_keep["baseline_mean"],
            "baseline_rewrite_mean": kr_headroom_over_rewrite["baseline_mean"],
            "headroom_over_rewrite": kr_headroom_over_rewrite["headroom"],
            "headroom_over_keep": kr_headroom_over_keep["headroom"],
        },
        "keep_vs_concat_vs_rewrite_oracle": {
            "mean_oracle_score": mean_oracle_score(kcr_scores),
            "baseline_rewrite_mean": kcr_headroom_over_rewrite["baseline_mean"],
            "headroom_over_rewrite": kcr_headroom_over_rewrite["headroom"],
        },
        "outcome_distribution": outcome_distribution,
    }

    with oracle_path.open("w", encoding="utf-8") as f:
        json.dump(oracle_results, f, indent=2)

    logger.info("Phase 4 evaluation complete. Artifacts written to %s", output_dir)
    return {
        "summary": summary,
        "outcomes": outcome_distribution,
        "oracle": oracle_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 retrieval evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to experiment config")
    parser.add_argument("--pilot-path", type=Path, default=None, help="Path to pilot formulations JSONL")
    parser.add_argument("--index-path", type=Path, default=None, help="Path to SQLite BM25 index")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)

    pilot_path = (
        args.pilot_path
        or PROJECT_ROOT / config["output"]["results_dir"] / "formulation_pilot.jsonl"
    )
    index_path = (
        args.index_path
        or PROJECT_ROOT / "data" / "index" / "bm25_fts5_dedup_full.sqlite"
    )
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / config["output"]["results_dir"]
    )

    eval_config = config.get("evaluation", {})
    primary_metric = eval_config.get("primary_metric", "ndcg@10")
    tie_margin = float(eval_config.get("tie_margin", 0.01))
    top_k = int(config.get("retrieval", {}).get("top_k", 100))

    run_evaluation(
        pilot_path=pilot_path,
        index_path=index_path,
        output_dir=output_dir,
        top_k=top_k,
        primary_metric=primary_metric,
        tie_margin=tie_margin,
    )


if __name__ == "__main__":
    main()
