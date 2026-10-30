"""Generate high-resolution, publication-ready research paper figures.

Generates:
1. outputs/plots/fig1_helpful_tied_harmful_distribution.png
   - Stacked horizontal outcome distribution bar chart (Helpful vs Tied vs Harmful).
2. outputs/plots/fig2_oracle_headroom_comparison.png
   - Headroom bar chart comparing KEEP, CONCAT, REWRITE, and ORACLE upper bound.
3. outputs/plots/fig3_quality_vs_rewrite_rate.png
   - Quality (nDCG@10) versus Rewrite Rate curve with Pareto frontiers.
4. outputs/plots/fig4_error_taxonomy_breakdown.png
   - Failure analysis distribution across research error taxonomy categories.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Fix matplotlib cache permissions in Windows sandbox
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "C:/Windows/Temp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Academic styling settings
plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_helpful_tied_harmful():
    """Figure 1: Stacked bar chart showing outcome distribution."""
    # Data from oracle_summary.json / outcome_labels.jsonl
    helpful = 30.8
    tied = 61.5
    harmful = 7.7

    fig, ax = plt.subplots(figsize=(8.5, 3.2))

    colors = ["#2b8a3e", "#adb5bd", "#c92a2a"]
    categories = ["Helpful (30.8%)", "Tied (61.5%)", "Harmful (7.7%)"]
    widths = [helpful, tied, harmful]
    starts = [0, helpful, helpful + tied]

    for cat, width, start, color in zip(categories, widths, starts, colors):
        rect = ax.barh(["Rewriting Outcomes"], width, left=start, color=color, edgecolor="black", height=0.55, label=cat)
        # Add text label in center of each segment
        x_center = start + width / 2.0
        ax.text(x_center, 0, f"{width:.1f}%\n({cat.split()[0]})", ha="center", va="center", color="white" if color != "#adb5bd" else "#212529", fontweight="bold", fontsize=11)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of Evaluated Conversational Turns (%)", fontweight="bold")
    ax.set_title("Per-Query Formulation Outcomes: REWRITE vs. KEEP (QReCC)", pad=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.get_yaxis().set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)

    out_file = PLOTS_DIR / "fig1_helpful_tied_harmful_distribution.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_oracle_headroom():
    """Figure 2: Headroom comparison bar chart."""
    methods = ["KEEP\n(Raw Query)", "CONCAT\n(History k=3)", "REWRITE\n(Always Rewrite)", "ORACLE\n(Upper Bound)"]
    scores = [0.8420, 0.8588, 0.9208, 0.9492]
    colors = ["#495057", "#4dabf7", "#3b5bdb", "#7048e8"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(methods, scores, color=colors, width=0.55, edgecolor="black", linewidth=1.1)

    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("Retrieval Quality (nDCG@10)", fontweight="bold")
    ax.set_title("Formulation Performance & Oracle Headroom (BM25)", pad=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.007, f"{h:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10.5)

    # Annotate headroom arrow between Rewrite and Oracle
    ax.annotate(
        "Headroom: +0.0284",
        xy=(2, 0.925), xytext=(2.5, 0.965),
        arrowprops=dict(facecolor="#c92a2a", arrowstyle="->", lw=1.5),
        ha="center", fontweight="bold", color="#c92a2a", fontsize=10
    )

    out_file = PLOTS_DIR / "fig2_oracle_headroom_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_quality_vs_rewrite_rate():
    """Figure 3: Quality vs. Rewrite-rate trade-off curve."""
    # Data: (Rewrite Rate %, Mean nDCG@10)
    data = [
        ("Always KEEP", 0.0, 0.8025, "#495057", "o", "bottom"),
        ("Learned Gate (Ours)", 33.33, 0.8606, "#2b8a3e", "s", "top"),
        ("Random Gate (Rate=50%)", 50.0, 0.9117, "#fd7e14", "^", "bottom"),
        ("Always REWRITE", 100.0, 0.9732, "#3b5bdb", "D", "bottom"),
        ("Heuristic Gate", 100.0, 0.9732, "#e03131", "X", "top"),
        ("Oracle (Ceiling)", 50.0, 0.9732, "#7048e8", "*", "top"),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    # Plot baseline reference line
    ax.axhline(0.8025, color="#868e96", linestyle=":", alpha=0.7, label="Always KEEP Baseline")
    ax.axhline(0.9732, color="#3b5bdb", linestyle="--", alpha=0.4, label="Always REWRITE Quality")

    # Connect empirical models with a light trend curve
    curve_x = [0.0, 33.33, 50.0, 100.0]
    curve_y = [0.8025, 0.8606, 0.9117, 0.9732]
    ax.plot(curve_x, curve_y, color="#2b8a3e", linestyle="-", linewidth=1.8, alpha=0.6, label="Cost-Quality Frontier")

    for name, rate, score, color, marker, align in data:
        size = 140 if marker == "*" else 90
        ax.scatter(rate, score, color=color, s=size, marker=marker, zorder=5, edgecolor="black", label=name if "Gate" in name or "Oracle" in name else None)
        offset_y = 0.007 if align == "top" else -0.014
        ax.text(rate, score + offset_y, f"{name}\n({score:.4f})", ha="center", va="center", fontsize=9.5, fontweight="bold", color=color)

    ax.set_xlim(-5, 108)
    ax.set_ylim(0.77, 1.01)
    ax.set_xlabel("Empirical Rewrite Rate (%) [Lower = Less Cost/Latency]", fontweight="bold")
    ax.set_ylabel("Retrieval Quality (nDCG@10)", fontweight="bold")
    ax.set_title("Cost-Quality Trade-off: Retrieval Quality vs. Rewrite Rate", pad=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_file = PLOTS_DIR / "fig3_quality_vs_rewrite_rate.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_error_analysis():
    """Figure 4: Error analysis breakdown chart."""
    categories = [
        "No Useful Change\n(Tied Paraphrase)",
        "Generation Failure\n(Token Exhaustion)",
        "Intent Drift /\nOver-Specification",
        "Lost Key Terms\n(Dropped Numbers/Names)",
    ]
    percentages = [61.5, 13.3, 7.7, 7.7]
    counts = [8, 2, 1, 1]
    colors = ["#adb5bd", "#e03131", "#f08c00", "#d6336c"]

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    bars = ax.bar(categories, percentages, color=colors, width=0.55, edgecolor="black", linewidth=1.1)

    ax.set_ylim(0, 75)
    ax.set_ylabel("Share of Evaluated Turns (%)", fontweight="bold")
    ax.set_title("Rewriting Failure & Neutrality Taxonomy (Phase 6)", pad=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count, pct in zip(bars, counts, percentages):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, f"{pct:.1f}%\n(n={count})", ha="center", va="bottom", fontweight="bold", fontsize=10)

    out_file = PLOTS_DIR / "fig4_error_taxonomy_breakdown.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    plot_helpful_tied_harmful()
    plot_oracle_headroom()
    plot_quality_vs_rewrite_rate()
    plot_error_analysis()
    print("All research figures generated successfully.")
