"""End-to-end reproducible research experiment pipeline.

Phase 7:
Freezes configuration and runs the complete evaluation sequence:
1. Retrieval evaluation (KEEP, CONCAT, REWRITE) & per-turn metrics
2. Outcome labeling (helpful / tied / harmful) & Oracle headroom
3. Pre-retrieval feature extraction and conversation-level train/val/test split
4. Gating model evaluation across all baselines (KEEP, REWRITE, CONCAT, Heuristic, Random, Learned, Oracle)
5. Efficiency & latency aggregation (tokens, calls, ms)
6. Produces publication-ready summary manifest and Markdown comparison tables:
   - outputs/results/run_manifest.json
   - outputs/results/final_summary.json
   - outputs/results/paper_tables.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from scripts.aggregate_efficiency import analyze_efficiency
from scripts.run_gate_eval import evaluate_gate_systems
from scripts.run_retrieval_eval import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


def get_git_commit() -> str:
    """Safely get current Git commit hash if in a Git repository."""
    try:
        out = subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "git-unavailable"


def generate_paper_tables(
    final_summary: Dict[str, Any],
    output_path: Path,
) -> str:
    """Format paper-ready Markdown tables from experiment results."""
    eval_summary = final_summary.get("retrieval_evaluation", {}).get("summary", {})
    oracle_summary = final_summary.get("retrieval_evaluation", {}).get("oracle", {})
    gate_summary = final_summary.get("gate_evaluation", {})
    efficiency = final_summary.get("efficiency", {})

    primary_metric = eval_summary.get("primary_metric", "ndcg@10")
    matched_n = eval_summary.get("matched_subset_denominators", 0)

    md = []
    md.append("# Experimental Results & Evaluation Tables\n")
    md.append(f"*Generated on: {final_summary.get('timestamp')}*\n")
    md.append(f"*Git commit: `{final_summary.get('git_commit')}`*\n\n")

    # Table 1: Retrieval Quality across Formulations (Matched Subset)
    md.append("## Table 1: Formulation Performance on Matched Subset\n")
    md.append(f"| Formulation | Evaluated Turns (N) | Mean {primary_metric} | Delta vs KEEP |\n")
    md.append("|---|---:|---:|---:|\n")

    matched_avgs = eval_summary.get("matched_subset_macro_averages", {})
    keep_mean = matched_avgs.get("keep", {}).get(f"mean_{primary_metric}", 0.0)

    for act in ("keep", "concat", "rewrite"):
        act_info = matched_avgs.get(act, {})
        mean_val = act_info.get(f"mean_{primary_metric}", 0.0)
        delta = mean_val - keep_mean
        delta_str = f"{delta:+.4f}" if act != "keep" else "0.0000"
        md.append(f"| **{act.upper()}** | {matched_n} | {mean_val:.4f} | {delta_str} |\n")

    # Table 2: Outcome Distribution and Oracle Headroom
    md.append("\n## Table 2: Help / Tie / Hurt Outcomes & Oracle Headroom\n")
    outcomes = oracle_summary.get("outcome_distribution", {})
    md.append("| Metric | Value |\n")
    md.append("|---|---:|\n")
    md.append(f"| Helpful turns (%) | {outcomes.get('helpful_pct', 0.0):.1f}% |\n")
    md.append(f"| Tied turns (%) | {outcomes.get('tied_pct', 0.0):.1f}% |\n")
    md.append(f"| Harmful turns (%) | {outcomes.get('harmful_pct', 0.0):.1f}% |\n")
    kr_or = oracle_summary.get("keep_vs_rewrite_oracle", {})
    md.append(f"| Oracle Score (KEEP vs REWRITE) | {kr_or.get('mean_oracle_score', 0.0):.4f} |\n")
    md.append(f"| Headroom over REWRITE | {kr_or.get('headroom_over_rewrite', 0.0):+.4f} |\n")
    md.append(f"| Headroom over KEEP | {kr_or.get('headroom_over_keep', 0.0):+.4f} |\n")

    # Table 3: Gating Methods Comparison
    md.append("\n## Table 3: Gating Model Comparison vs Controls\n")
    md.append("| Method | Rewrite Rate (%) | Total Rewrites / Evaluated |\n")
    md.append("|---|---:|---:|\n")
    for method_name, m_data in gate_summary.get("methods", {}).items():
        rate = m_data.get("rewrite_rate_pct", 0.0)
        count = m_data.get("rewrite_count", 0)
        tot = m_data.get("total_count", 0)
        md.append(f"| `{method_name}` | {rate:.1f}% | {count} / {tot} |\n")

    # Table 4: Efficiency & Computational Costs
    md.append("\n## Table 4: Latency and Token Overhead\n")
    md.append("| Measure | Value |\n")
    md.append("|---|---:|\n")
    md.append(f"| Total LLM Rewrite Calls | {efficiency.get('total_rewrite_calls', 0)} |\n")
    md.append(f"| Retries Used | {efficiency.get('retry_count', 0)} ({efficiency.get('retry_rate_pct', 0.0)}%) |\n")
    tokens = efficiency.get("tokens", {})
    md.append(f"| Total Tokens Consumed | {tokens.get('total_tokens', 0):,} |\n")
    md.append(f"| Mean Tokens per Call | {tokens.get('mean_total_tokens_per_call', 0.0):.1f} |\n")
    latency = efficiency.get("latency_ms", {})
    md.append(f"| Mean Latency (ms) | {latency.get('mean', 0.0):.1f} ms |\n")
    md.append(f"| P95 Latency (ms) | {latency.get('p95', 0.0):.1f} ms |\n")

    content = "".join(md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(content)

    return content


def run_pipeline(
    config_path: Path = DEFAULT_CONFIG,
    pilot_path: Path | None = None,
    index_path: Path | None = None,
    cache_path: Path | None = None,
    output_dir: Path | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute the end-to-end research experiment pipeline."""
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    res_dir = output_dir or PROJECT_ROOT / config.get("output", {}).get("results_dir", "outputs/results")
    res_dir.mkdir(parents=True, exist_ok=True)

    pilot_file = pilot_path or res_dir / "formulation_pilot.jsonl"
    index_file = index_path or PROJECT_ROOT / "data" / "index" / "bm25_fts5_dedup_full.sqlite"
    cache_file = cache_path or PROJECT_ROOT / ".rewrite_cache" / "rewrites.jsonl"

    run_manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config_path": str(config_path),
        "pilot_path": str(pilot_file),
        "index_path": str(index_file),
        "cache_path": str(cache_file),
        "output_dir": str(res_dir),
        "seed": seed,
    }

    with (res_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    logger.info("Recorded run manifest to %s", res_dir / "run_manifest.json")

    # Step 1: Retrieval Evaluation
    logger.info("--- Step 1: Retrieval Evaluation ---")
    retrieval_res = run_evaluation(
        pilot_path=pilot_file,
        index_path=index_file,
        output_dir=res_dir,
        top_k=int(config.get("retrieval", {}).get("top_k", 100)),
        primary_metric=config.get("evaluation", {}).get("primary_metric", "ndcg@10"),
        tie_margin=float(config.get("evaluation", {}).get("tie_margin", 0.01)),
        run_id="reproducible_pipeline_run",
    )

    # Step 2: Gate Evaluation
    logger.info("--- Step 2: Gating Systems Evaluation ---")
    pilot_turns = []
    with pilot_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pilot_turns.append(json.loads(line))

    outcomes_map = {}
    outcomes_file = res_dir / "outcome_labels.jsonl"
    if outcomes_file.exists():
        with outcomes_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    outcomes_map[r["turn_id"]] = r

    gate_res = evaluate_gate_systems(
        pilot_turns=pilot_turns,
        outcomes=outcomes_map,
        output_dir=res_dir,
        seed=seed,
    )

    # Step 3: Efficiency Aggregation
    logger.info("--- Step 3: Efficiency Aggregation ---")
    eff_res = analyze_efficiency(
        cache_path=cache_file,
        output_dir=res_dir,
        gate_predictions_path=res_dir / "gate_predictions.jsonl",
    )

    # Step 4: Final Summary & Markdown Tables
    final_summary = {
        **run_manifest,
        "retrieval_evaluation": retrieval_res,
        "gate_evaluation": gate_res,
        "efficiency": eff_res,
    }

    with (res_dir / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    generate_paper_tables(final_summary, res_dir / "paper_tables.md")
    logger.info("Generated final research summary and paper tables.")
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible experiment pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pilot-path", type=Path, default=None)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        pilot_path=args.pilot_path,
        index_path=args.index_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
