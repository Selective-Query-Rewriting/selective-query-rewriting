"""Generate clean, publication-grade IEEE/ACM transaction-ready research plots.

Fixes & Enhancements:
- Perfect spatial separation (zero overlapping lines or text in Fig 2).
- Clean IEEE column-width proportions (single-column and double-column standard sizes).
- Outputs in both lossless SVG (vector XML) and ultra-high-resolution 600 DPI PNG.
- Professional, colorblind-safe academic palette with crisp serif typography (Times New Roman / STIX).
"""

from __future__ import annotations

import os
from pathlib import Path

# Use writable cache in Windows sandbox
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "C:/Windows/Temp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# IEEE Conference & Transactions Typography Setup
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
})

# IEEE / ColorBrewer Palette
C_HELPFUL = "#2ca25f"   # Green
C_TIED = "#999999"      # Neutral Gray
C_HARMFUL = "#de2d26"   # Muted Red
C_BLUE = "#3182bd"      # Classic IEEE Primary Blue
C_ORACLE = "#756bb1"    # Purple
C_ACCENT = "#e6550d"    # Orange


# ==========================================================================
# FIGURE 1: Helpful / Tied / Harmful Outcome Distribution
# ==========================================================================
def generate_fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))

    categories = ["Helpful (Rewriting Wins)", "Tied (|Δ| ≤ 0.01)", "Harmful (KEEP Wins)"]
    shares = [30.8, 61.5, 7.7]
    colors = [C_HELPFUL, C_TIED, C_HARMFUL]
    edge_colors = ["#006d2c", "#525252", "#a50f15"]

    left = 0.0
    y_pos = 0.4
    bar_height = 0.38

    for cat, share, col, ecol in zip(categories, shares, colors, edge_colors):
        ax.barh(y_pos, share, left=left, height=bar_height, color=col, edgecolor=ecol, linewidth=1.1, label=cat)
        
        center_x = left + share / 2.0
        text_color = "white" if col != C_TIED else "#111111"
        
        if share > 15:
            ax.text(center_x, y_pos, f"{share:.1f}%\n({cat.split()[0]})", 
                    ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)
        else:
            ax.text(center_x, y_pos, f"{share:.1f}%\n(Hurt)", 
                    ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            
        left += share

    ax.set_xlim(0, 100)
    ax.set_ylim(0.0, 0.85)
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_xlabel("Percentage of Evaluated Conversational Turns (%)", labelpad=6)
    ax.set_title("Fig. 1. Per-Query Formulation Outcomes: REWRITE vs. KEEP (QReCC Benchmark)", pad=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.grid(axis="x", linestyle="--", color="#d9d9d9")
    ax.set_axisbelow(True)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=True, fancybox=False)

    plt.savefig(PLOTS_DIR / "fig1_helpful_tied_harmful_distribution.svg")
    plt.savefig(PLOTS_DIR / "fig1_helpful_tied_harmful_distribution.png")
    plt.close()
    print("Created: Fig 1 (SVG & PNG)")


# ==========================================================================
# FIGURE 2: Formulation Performance & Oracle Headroom (Zero Overlap)
# ==========================================================================
def generate_fig2():
    # Sized for standard IEEE column width with ample vertical breathing room
    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    methods = [
        "KEEP\n(Raw Query)", 
        "CONCAT\n(History $k=3$)", 
        "REWRITE\n(Always Rewrite)", 
        "ORACLE\n(Optimal Bound)"
    ]
    ndcg_scores = [0.8420, 0.8588, 0.9208, 0.9492]
    colors = ["#636363", "#9ecae1", C_BLUE, C_ORACLE]
    edge_colors = ["#252525", "#3182bd", "#08519c", "#54278f"]

    x_positions = np.arange(len(methods))
    bars = ax.bar(x_positions, ndcg_scores, width=0.48, color=colors, edgecolor=edge_colors, linewidth=1.1, zorder=3)

    # Expanded y-limits so all annotations sit completely above the bars
    ax.set_ylim(0.75, 1.05)
    ax.set_ylabel("Retrieval Quality (nDCG@10)", labelpad=6)
    ax.set_title("Fig. 2. Formulation Performance and Oracle Action Headroom", pad=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods)

    ax.grid(axis="y", linestyle="--", color="#e0e0e0", zorder=0)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bar score labels directly above bars (clean margin of 0.006)
    for bar, score in zip(bars, ndcg_scores):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.006, f"{score:.4f}",
                ha="center", va="bottom", fontweight="bold", fontsize=10, color="#252525")

    # --- Headroom Bracket Placed Safely at y=0.995 (No collision with bar texts) ---
    x_rewr = 2
    x_orac = 3
    y_b = 0.995
    h_b = 0.012
    # Draw crisp horizontal bracket
    ax.plot([x_rewr, x_rewr, x_orac, x_orac], [y_b, y_b + h_b, y_b + h_b, y_b], color="#de2d26", lw=1.2, zorder=4)
    # Headroom text above bracket
    ax.text((x_rewr + x_orac) / 2.0, y_b + h_b + 0.006, "Headroom: +0.0284", 
            ha="center", va="bottom", color="#de2d26", fontsize=9.5, fontweight="bold")

    # Delta Rewrite vs KEEP annotation placed in lower quadrant (no overlap)
    ax.annotate(
        r"$\Delta = +0.0788\ \mathrm{vs\ KEEP}$",
        xy=(2, 0.9208), xytext=(0.8, 0.935),
        arrowprops=dict(arrowstyle="->", color="#006d2c", lw=1.1, connectionstyle="arc3,rad=0.15"),
        fontsize=9.2, color="#006d2c", fontweight="bold"
    )

    plt.savefig(PLOTS_DIR / "fig2_oracle_headroom_comparison.svg")
    plt.savefig(PLOTS_DIR / "fig2_oracle_headroom_comparison.png")
    plt.close()
    print("Created: Fig 2 (SVG & PNG)")


# ==========================================================================
# FIGURE 3: Cost-Quality Trade-Off Curve (Pareto Frontier)
# ==========================================================================
def generate_fig3():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    models = [
        {"name": "Always KEEP", "rate": 0.0, "score": 0.8025, "col": "#525252", "marker": "o", "dx": 0, "dy": -18},
        {"name": "Learned Gate (Ours)", "rate": 33.33, "score": 0.8606, "col": C_HELPFUL, "marker": "s", "dx": 28, "dy": 8},
        {"name": "Random Control", "rate": 50.0, "score": 0.9117, "col": C_ACCENT, "marker": "^", "dx": 0, "dy": -20},
        {"name": "Oracle Ceiling", "rate": 50.0, "score": 0.9732, "col": C_ORACLE, "marker": "*", "dx": -25, "dy": 12},
        {"name": "Heuristic Gate", "rate": 100.0, "score": 0.9732, "col": C_HARMFUL, "marker": "X", "dx": -65, "dy": -16},
        {"name": "Always REWRITE", "rate": 100.0, "score": 0.9732, "col": C_BLUE, "marker": "D", "dx": -65, "dy": 10},
    ]

    # Reference levels
    ax.axhline(0.8025, color="#969696", linestyle=":", lw=1.1, label="Always KEEP Lower Bound")
    ax.axhline(0.9732, color=C_BLUE, linestyle="--", lw=1.0, alpha=0.5, label="Always REWRITE Upper Baseline")

    # Trade-off curve
    xs = [0.0, 33.33, 50.0, 100.0]
    ys = [0.8025, 0.8606, 0.9117, 0.9732]
    ax.plot(xs, ys, color=C_HELPFUL, lw=2.0, linestyle="-", alpha=0.85, zorder=3, label="Cost-Quality Frontier")
    ax.fill_between(xs, ys, 0.8025, color=C_HELPFUL, alpha=0.08, zorder=2)

    for m in models:
        sz = 130 if m["marker"] in ("*", "X") else 85
        ax.scatter(m["rate"], m["score"], color=m["col"], edgecolors="#111111", linewidths=1.0, s=sz, marker=m["marker"], zorder=5)
        
        ax.annotate(
            f"{m['name']}\n({m['score']:.4f})",
            xy=(m["rate"], m["score"]),
            xytext=(m["dx"], m["dy"]),
            textcoords="offset points",
            fontweight="bold" if "Ours" in m["name"] else "normal",
            fontsize=8.8,
            color=m["col"] if "Ours" not in m["name"] else "#005a32",
            ha="center" if m["dx"] == 0 else ("right" if m["dx"] < 0 else "left"),
            arrowprops=dict(arrowstyle="->", color=m["col"], lw=0.8, alpha=0.7) if abs(m["dx"]) > 22 else None
        )

    # Callout box for efficiency
    ax.annotate(
        "66.7% Rewrite Calls Avoided\n(Only 33.3% queries rewritten)",
        xy=(33.33, 0.8606), xytext=(5, 0.93),
        arrowprops=dict(facecolor="#006d2c", edgecolor="#006d2c", arrowstyle="->", connectionstyle="arc3,rad=-0.15"),
        fontsize=9.0, fontweight="bold", color="#006d2c",
        bbox=dict(boxstyle="round,pad=0.4", fc="#e5f5e0", ec="#a1d99b", lw=0.9)
    )

    ax.set_xlim(-6, 114)
    ax.set_ylim(0.77, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlabel("Empirical Rewrite Rate (%) [Proxy for Latency & API Cost]", labelpad=6)
    ax.set_ylabel("Retrieval Quality (nDCG@10)", labelpad=6)
    ax.set_title("Fig. 3. Quality vs. Rewrite Rate: Trade-off Frontier", pad=12)

    ax.grid(True, linestyle="--", color="#e5e5e5", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fancybox=False)

    plt.savefig(PLOTS_DIR / "fig3_quality_vs_rewrite_rate.svg")
    plt.savefig(PLOTS_DIR / "fig3_quality_vs_rewrite_rate.png")
    plt.close()
    print("Created: Fig 3 (SVG & PNG)")


# ==========================================================================
# FIGURE 4: Error & Failure Taxonomy Breakdown
# ==========================================================================
def generate_fig4():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    categories = [
        "Neutral Paraphrase\n(Tied Outcomes)",
        "Token Exhaustion\n(Generation Failure)",
        "Intent Drift /\nOver-Specification",
        "Dropped Constraints\n(Lost Key Terms)",
    ]
    percentages = [61.5, 13.3, 7.7, 7.7]
    counts = [8, 2, 1, 1]
    colors = ["#878787", "#e41a1c", "#ff7f00", "#984ea3"]
    edge_colors = ["#404040", "#800026", "#b35806", "#49006a"]

    x_positions = np.arange(len(categories))
    bars = ax.bar(x_positions, percentages, width=0.48, color=colors, edgecolor=edge_colors, linewidth=1.1, zorder=3)

    ax.set_ylim(0, 75)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.yaxis.set_major_locator(MultipleLocator(15))
    ax.set_ylabel("Share of Evaluated Turns (%)", labelpad=6)
    ax.set_title("Fig. 4. Failure and Neutrality Taxonomy of Query Rewriting", pad=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories)

    ax.grid(axis="y", linestyle="--", color="#e5e5e5", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, pct, cnt in zip(bars, percentages, counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, f"{pct:.1f}%\n(n={cnt})", 
                ha="center", va="bottom", fontweight="bold", fontsize=9.5, color="#111111")

    # Explanatory bracket for harmful/failure categories
    ax.annotate(
        "Active Failures & Harmful Rewrites (28.7%)",
        xy=(2.0, 15), xytext=(2.0, 36),
        arrowprops=dict(arrowstyle="->", color="#99000d", lw=1.1),
        ha="center", va="bottom", fontweight="bold", fontsize=9.2, color="#99000d",
        bbox=dict(boxstyle="square,pad=0.3", fc="#fee5d9", ec="#fcae91", lw=0.9)
    )

    plt.savefig(PLOTS_DIR / "fig4_error_taxonomy_breakdown.svg")
    plt.savefig(PLOTS_DIR / "fig4_error_taxonomy_breakdown.png")
    plt.close()
    print("Created: Fig 4 (SVG & PNG)")


if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    print("All final IEEE-grade research plots successfully generated.")
