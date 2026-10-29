"""Run gate evaluation and baseline comparison across all strategies.

Compares:
1. Always KEEP
2. Always REWRITE
3. CONCAT (fixed context addition)
4. Heuristic Gate (rule-based)
5. Matched-rate Random Gate (control)
6. Learned Gate (lightweight ML classifier)
7. Oracle (analytical upper bound)

Produces:
- outputs/results/gate_predictions.jsonl
- outputs/results/gate_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.gating.heuristic import HeuristicGate
from src.gating.random_gate import RandomGate
from src.gating.train_gate import LearnedGate, create_conversation_split, train_learned_gate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_gate_systems(
    pilot_turns: List[Dict[str, Any]],
    outcomes: Dict[str, Dict[str, Any]],
    output_dir: Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train learned gate and evaluate all formulation strategies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "gate_predictions.jsonl"
    summary_path = output_dir / "gate_summary.json"

    # Merge binary labels: HELPFUL (1) vs TIED/HARMFUL (0)
    labeled_turns = []
    for turn in pilot_turns:
        tid = turn["turn_id"]
        outcome_info = outcomes.get(tid, {})
        outcome_label = outcome_info.get("outcome", "tied")
        # 1 if rewrite helped, 0 if rewrite tied or harmed
        label = 1 if outcome_label == "helpful" else 0
        labeled_turns.append({**turn, "label": label, "outcome": outcome_label})

    # Conversation-level split (70% train, 15% val, 15% test)
    splits = create_conversation_split(labeled_turns, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed)
    train_set = splits["train"]
    val_set = splits["val"]
    test_set = splits["test"]

    logger.info(
        "Split dataset by conversation: Train=%d, Val=%d, Test=%d turns",
        len(train_set), len(val_set), len(test_set)
    )

    # Train learned gate
    learned_gate = train_learned_gate(
        train_turns=train_set,
        val_turns=val_set,
        model_type="logistic",
        tune_threshold=True,
        seed=seed,
    )
    logger.info("Trained LearnedGate with calibrated threshold: %.3f", learned_gate.threshold)

    # Instantiate Heuristic Gate
    heuristic_gate = HeuristicGate()

    # Evaluate on the evaluation set (or full set for benchmarking)
    eval_set = test_set if len(test_set) > 0 else labeled_turns

    # Compute empirical rewrite rate for learned gate on eval_set
    learned_decisions = [learned_gate.predict(t) for t in eval_set]
    empirical_rate = sum(learned_decisions) / len(learned_decisions) if learned_decisions else 0.5
    logger.info("Empirical Learned Gate rewrite rate: %.1f%%", empirical_rate * 100)

    # Instantiate Random Gate matched to learned gate's empirical rewrite rate
    random_gate = RandomGate(rewrite_rate=empirical_rate, seed=seed)

    # Generate predictions for each turn
    per_turn_predictions = []
    actions_by_method = {
        "always_keep": [],
        "always_rewrite": [],
        "heuristic_gate": [],
        "random_gate": [],
        "learned_gate": [],
        "oracle": [],
    }

    for t in eval_set:
        tid = t["turn_id"]
        outcome = t.get("outcome", "tied")

        h_act = heuristic_gate.predict_action(t)
        r_act = random_gate.predict_action(t)
        l_prob = learned_gate.predict_proba(t)
        l_act = "rewrite" if l_prob >= learned_gate.threshold else "keep"

        # Oracle chooses rewrite only if it is helpful
        o_act = "rewrite" if outcome == "helpful" else "keep"

        actions_by_method["always_keep"].append("keep")
        actions_by_method["always_rewrite"].append("rewrite")
        actions_by_method["heuristic_gate"].append(h_act)
        actions_by_method["random_gate"].append(r_act)
        actions_by_method["learned_gate"].append(l_act)
        actions_by_method["oracle"].append(o_act)

        rec = {
            "turn_id": tid,
            "conversation_id": t.get("conversation_id"),
            "true_outcome": outcome,
            "predictions": {
                "always_keep": "keep",
                "always_rewrite": "rewrite",
                "heuristic": h_act,
                "random": r_act,
                "learned": l_act,
                "learned_prob": round(l_prob, 4),
                "oracle": o_act,
            },
        }
        per_turn_predictions.append(rec)

    with predictions_path.open("w", encoding="utf-8") as f:
        for r in per_turn_predictions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summarize Rewrite Rates
    summary = {
        "eval_turns_count": len(eval_set),
        "split_counts": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "methods": {},
    }

    print("\n" + "=" * 70)
    print("GATE EVALUATION & BASELINE COMPARISON")
    print("=" * 70)
    print(f"{'Method':<25} | {'Rewrite Rate (%)':<18} | {'Total Calls':<12}")
    print("-" * 70)

    for method, acts in actions_by_method.items():
        n_rewrites = sum(1 for a in acts if a == "rewrite")
        rate = (n_rewrites / len(acts) * 100) if acts else 0.0
        summary["methods"][method] = {
            "rewrite_count": n_rewrites,
            "total_count": len(acts),
            "rewrite_rate_pct": round(rate, 2),
        }
        print(f"{method:<25} | {rate:>16.1f}% | {n_rewrites:>10}/{len(acts)}")

    print("=" * 70)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved gate predictions to %s", predictions_path)
    logger.info("Saved gate summary to %s", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gating models.")
    parser.add_argument(
        "--pilot-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "results" / "formulation_pilot.jsonl",
        help="Path to formulation pilot turns JSONL",
    )
    parser.add_argument(
        "--outcomes-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "results" / "outcome_labels.jsonl",
        help="Path to outcome labels JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "results",
        help="Output directory",
    )
    args = parser.parse_args()

    pilot_turns = []
    if args.pilot_path.exists():
        with args.pilot_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pilot_turns.append(json.loads(line))

    outcomes_map = {}
    if args.outcomes_path.exists():
        with args.outcomes_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    outcomes_map[row["turn_id"]] = row

    if not pilot_turns:
        logger.warning("No pilot turns found at %s. Please run Phase 3 pilot first.", args.pilot_path)
        return

    evaluate_gate_systems(pilot_turns, outcomes_map, args.output_dir)


if __name__ == "__main__":
    main()
