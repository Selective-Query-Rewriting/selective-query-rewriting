"""Aggregate efficiency metrics from rewrite cache and execution logs.

Phase 6:
Computes:
- Total gate decisions and empirical rewrite rate
- Total prompt tokens, completion tokens, and overall token usage
- Mean, median, and 95th percentile latency (ms)
- Cache hit vs API call breakdown
- Generation failure rate and retry counts
Writes:
- outputs/results/efficiency_summary.json
- outputs/results/efficiency_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / ".rewrite_cache" / "rewrites.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "results"


def analyze_efficiency(
    cache_path: Path,
    output_dir: Path,
    gate_predictions_path: Path | None = None,
) -> Dict[str, Any]:
    """Process cache logs and compute aggregate token/latency efficiency metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "efficiency_summary.json"
    log_path = output_dir / "efficiency_log.jsonl"

    entries: List[Dict[str, Any]] = []
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    logger.info("Loaded %d rewrite cache log records.", len(entries))

    prompt_tokens_list = []
    completion_tokens_list = []
    total_tokens_list = []
    latencies = []
    retries = 0

    for e in entries:
        pt = e.get("prompt_tokens")
        ct = e.get("completion_tokens")
        tt = e.get("total_tokens")
        lat = e.get("latency_ms")
        retry = e.get("retry_used", False)

        if pt is not None:
            prompt_tokens_list.append(int(pt))
        if ct is not None:
            completion_tokens_list.append(int(ct))
        if tt is not None:
            total_tokens_list.append(int(tt))
        if lat is not None:
            latencies.append(float(lat))
        if retry:
            retries += 1

    n_calls = len(entries)

    metrics = {
        "total_rewrite_calls": n_calls,
        "retry_count": retries,
        "retry_rate_pct": round(retries / n_calls * 100, 2) if n_calls > 0 else 0.0,
        "tokens": {
            "total_prompt_tokens": int(sum(prompt_tokens_list)),
            "total_completion_tokens": int(sum(completion_tokens_list)),
            "total_tokens": int(sum(total_tokens_list)),
            "mean_prompt_tokens_per_call": round(float(np.mean(prompt_tokens_list)), 2) if prompt_tokens_list else 0.0,
            "mean_completion_tokens_per_call": round(float(np.mean(completion_tokens_list)), 2) if completion_tokens_list else 0.0,
            "mean_total_tokens_per_call": round(float(np.mean(total_tokens_list)), 2) if total_tokens_list else 0.0,
        },
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "median": round(float(np.median(latencies)), 2) if latencies else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
            "min": round(float(np.min(latencies)), 2) if latencies else 0.0,
            "max": round(float(np.max(latencies)), 2) if latencies else 0.0,
        },
    }

    # If gate predictions exist, calculate overall system rewrite rate & savings
    if gate_predictions_path and gate_predictions_path.exists():
        preds = []
        with gate_predictions_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    preds.append(json.loads(line))
        total_queries = len(preds)
        learned_rewrites = sum(
            1 for p in preds if p.get("predictions", {}).get("learned") == "rewrite"
        )
        saved_calls = total_queries - learned_rewrites
        metrics["gate_efficiency"] = {
            "total_queries_evaluated": total_queries,
            "learned_gate_rewrites": learned_rewrites,
            "rewrite_rate_pct": round(learned_rewrites / total_queries * 100, 2) if total_queries else 0.0,
            "llm_calls_skipped": saved_calls,
            "llm_call_reduction_pct": round(saved_calls / total_queries * 100, 2) if total_queries else 0.0,
        }

    # Save summary
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save detailed log copy
    with log_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("EFFICIENCY & LATENCY SUMMARY")
    print("=" * 60)
    print(f"Total LLM rewrite calls:     {metrics['total_rewrite_calls']}")
    print(f"Retries triggered:           {metrics['retry_count']} ({metrics['retry_rate_pct']}%)")
    print(f"Total tokens consumed:       {metrics['tokens']['total_tokens']:,}")
    print(f"Mean tokens per call:        {metrics['tokens']['mean_total_tokens_per_call']}")
    print(f"Mean latency:                {metrics['latency_ms']['mean']} ms")
    print(f"P95 latency:                 {metrics['latency_ms']['p95']} ms")
    if "gate_efficiency" in metrics:
        ge = metrics["gate_efficiency"]
        print(f"Gate rewrite rate:           {ge['rewrite_rate_pct']}%")
        print(f"Rewrite calls avoided:       {ge['llm_calls_skipped']} ({ge['llm_call_reduction_pct']}%)")
    print("=" * 60)

    logger.info("Saved efficiency summary to %s", summary_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate rewrite efficiency.")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE, help="Path to rewrite cache")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_OUTPUT / "gate_predictions.jsonl")
    args = parser.parse_args()

    analyze_efficiency(args.cache_path, args.output_dir, args.predictions_path)


if __name__ == "__main__":
    main()
