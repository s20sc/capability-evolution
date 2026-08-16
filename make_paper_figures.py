"""figures_out"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
EXP_DIR = Path(__file__).resolve().parent
RES = EXP_DIR / "results"
OUT = EXP_DIR.parent / "figures_out" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# IEEE-style matplotlib configuration
# -------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

FIGW_COL = 3.45       # single-column inches
FIGW_FULL = 7.16      # full textwidth inches

# Consistent palette (method colors), aligned with the revision figures:
# vermilion = ours / rotation arm, blue = agent modification, salmon/orange/
# sky/grey for the remaining baselines (Okabe-Ito, colorblind-safe).
C_OURS  = "#D55E00"   # vermilion
C_AM    = "#0072B2"   # blue
C_SECM  = "#808080"   # gray
C_SPIRL = "#E69F00"   # orange
C_SKIMO = "#56B4E9"   # sky blue
C_SUCC  = "#555555"   # dual-axis success line (neutral, not a method color)
METHOD_COLORS = {
    "ours": C_OURS, "agent_modification": C_AM, "static_ecm": C_SECM,
    "spirl": C_SPIRL, "skimo": C_SKIMO,
}
METHOD_LABELS = {
    "ours": "Ours (Capability Evolution)",
    "agent_modification": "Agent Modification",
    "static_ecm": "Static ECM",
    "spirl": "SPiRL",
    "skimo": "SkiMo",
}

TASKS = ["T1_Pick", "T2_Place", "T3_Stack",
         "T4_NutAssembly", "T5_PickPlaceMulti", "T6_TwoArmPegInHole"]
TASK_TITLES = {
    "T1_Pick": "T1: Pick",
    "T2_Place": "T2: Place",
    "T3_Stack": "T3: Stack",
    "T4_NutAssembly": "T4: NutAssembly",
    "T5_PickPlaceMulti": "T5: PickPlaceMulti",
    "T6_TwoArmPegInHole": "T6: TwoArmPegInHole",
}


def _load(name):
    with open(RES / name) as f:
        return json.load(f)


# -------------------------------------------------------------------
# Figure 1: Evolution over iterations (H1)
# -------------------------------------------------------------------
def fig_evolution_curve():
    data = _load("exp1_evolution_over_time.json")

    fig, axes = plt.subplots(2, 3, figsize=(FIGW_FULL, 3.4), sharex=True)
    axes = axes.flatten()

    for i, task in enumerate(TASKS):
        ax = axes[i]
        metrics = data[task]["metrics"]
        iters = [m["iteration"] for m in metrics]
        rewards = [m["avg_reward"] for m in metrics]
        succ = [m["success_rate"] for m in metrics]

        # Reward on primary axis (blue line)
        l1, = ax.plot(iters, rewards, color=C_OURS, linewidth=1.3,
                      marker="o", markersize=2.5, label="Reward")
        ax.set_title(TASK_TITLES[task], pad=2)
        ax.set_ylabel("Reward", color=C_OURS, fontsize=7)
        ax.tick_params(axis="y", labelcolor=C_OURS, labelsize=6)
        ax.grid(True, alpha=0.25, linewidth=0.3)

        # Success on secondary axis (red dashed)
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(True)
        l2, = ax2.plot(iters, succ, color=C_SUCC, linewidth=1.0,
                       linestyle="--", marker="s", markersize=2.5,
                       label="Success")
        ax2.set_ylabel("Success (%)", color=C_SUCC, fontsize=7)
        ax2.set_ylim(-5, 105)
        ax2.tick_params(axis="y", labelcolor=C_SUCC, labelsize=6)

        if i >= 3:
            ax.set_xlabel("Iteration", fontsize=7)

    # Shared legend (top center)
    fig.legend([l1, l2], ["Avg. reward", "Success rate"],
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_evolution_curve.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'fig_evolution_curve.pdf'}")


# -------------------------------------------------------------------
# Figure 2: T6 flagship bar chart (H2, column width)
# Ours / AM now show mean ± std over 4 seeds (42, 7, 123, 2024) with
# per-seed scatter overlay; the other three baselines remain seed=42.
# -------------------------------------------------------------------
def fig_t6_flagship():
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Single-seed data for S-ECM / SPiRL / SkiMo (seed = 42)
    data = _load("exp2_method_comparison.json")
    t6_seed42 = data["T6_TwoArmPegInHole"]

    # Multi-seed data for Ours / AM
    ms = _load("t6_multiseed/summary.json")
    ms_agg = ms["aggregate"]
    ms_per = ms["per_seed"]

    methods = ["ours", "agent_modification", "static_ecm", "spirl", "skimo"]
    short = ["Ours", "AM", "S-ECM", "SPiRL", "SkiMo"]

    peak_mean, peak_std, peak_points = [], [], []
    for m in methods:
        if m in ("ours", "agent_modification"):
            agg = ms_agg[m]["peak_success"]
            peak_mean.append(agg["mean"])
            peak_std.append(agg["std"])
            peak_points.append(list(agg["values"]))
        else:
            v = max(x["success_rate"] for x in t6_seed42[m]["metrics"])
            peak_mean.append(v)
            peak_std.append(0.0)
            peak_points.append([v])

    x = np.arange(len(methods))
    w = 0.55
    fig, ax = plt.subplots(figsize=(FIGW_COL, 2.45))
    colors = [METHOD_COLORS[m] for m in methods]

    # Bars — mean values, with error bars for multi-seed methods
    bars = ax.bar(x, peak_mean, w, color=colors, edgecolor="black",
                  linewidth=0.4, zorder=2)
    for xi, mean, std, m in zip(x, peak_mean, peak_std, methods):
        if m in ("ours", "agent_modification"):
            ax.errorbar(xi, mean, yerr=std, fmt="none",
                        ecolor="black", elinewidth=0.9, capsize=3,
                        capthick=0.9, zorder=3)

    # Per-seed scatter overlay (only multi-seed methods)
    for xi, pts, m in zip(x, peak_points, methods):
        if m in ("ours", "agent_modification"):
            jitter = np.linspace(-0.12, 0.12, len(pts))
            ax.scatter(xi + jitter, pts, s=11, color="white",
                       edgecolor="black", linewidth=0.5, zorder=4)

    # Value labels on top
    for xi, mean, std, m in zip(x, peak_mean, peak_std, methods):
        top = mean + (std if m in ("ours","agent_modification") else 0) + 3
        label = f"{mean:.1f}" + (f"\n±{std:.1f}" if std > 0 else "")
        ax.text(xi, top, label, ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel("Peak success on T6 (\\%)")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.3)

    # Annotation / legend: explain what the error bars and dots mean
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=4, label="per-seed"),
        Line2D([0], [0], color="black", lw=0.9, label="mean $\\pm$ std (n=4)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False,
              handlelength=1.4, fontsize=6.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_t6_flagship.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'fig_t6_flagship.pdf'}")


# -------------------------------------------------------------------
# Figure 3: Peak reward across all tasks and methods (H2 secondary)
# -------------------------------------------------------------------
def fig_method_peak_rewards():
    data = _load("exp2_method_comparison.json")
    methods = ["ours", "agent_modification", "static_ecm", "spirl", "skimo"]
    tasks = TASKS

    peak = np.zeros((len(tasks), len(methods)))
    for i, t in enumerate(tasks):
        for j, m in enumerate(methods):
            peak[i, j] = max(x["avg_reward"] for x in data[t][m]["metrics"])

    fig, ax = plt.subplots(figsize=(FIGW_FULL, 2.2))
    x = np.arange(len(tasks))
    w = 0.16
    for j, m in enumerate(methods):
        ax.bar(x + (j - 2) * w, peak[:, j], w,
               label=METHOD_LABELS[m], color=METHOD_COLORS[m],
               edgecolor="black", linewidth=0.3)

    ax.set_yscale("log")
    ax.set_ylabel("Peak reward (log)")
    short_labels = [TASK_TITLES[t] for t in tasks]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=15, ha="right")
    ax.legend(loc="upper left", ncol=5, frameon=False, handlelength=1.2,
              bbox_to_anchor=(0, 1.20), fontsize=7)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_method_peak_rewards.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'fig_method_peak_rewards.pdf'}")


# -------------------------------------------------------------------
# Figure 4: Runtime-governance safety (H3)
# -------------------------------------------------------------------
def fig_safety():
    data = _load("exp3_safety.json")
    tasks = TASKS
    block = [data[t]["with_governance"]["blocking_rate"] for t in tasks]
    unsafe_off = [data[t]["without_governance"]["unsafe_exec_pct"] for t in tasks]
    unsafe_on = [data[t]["with_governance"]["unsafe_exec_pct"] for t in tasks]
    overhead = [data[t]["with_governance"]["overhead_ms"] for t in tasks]

    short = ["T1", "T2", "T3", "T4", "T5", "T6"]

    fig, axes = plt.subplots(1, 3, figsize=(FIGW_FULL, 1.9))

    # Panel A: blocking rate
    ax = axes[0]
    ax.bar(short, block, color=C_OURS, edgecolor="black", linewidth=0.4)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Blocking rate (\\%)")
    ax.set_title("(a) Governance blocking", pad=3)
    for i, v in enumerate(block):
        ax.text(i, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.3)

    # Panel B: unsafe execution rate (dodged)
    ax = axes[1]
    x = np.arange(len(short))
    w = 0.35
    ax.bar(x - w/2, unsafe_off, w, color=C_AM,
           label="Governance OFF", edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, unsafe_on, w, color="#2E7D32",
           label="Governance ON",  edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel("Unsafe episodes (\\%)")
    ax.set_title("(b) Unsafe execution", pad=3)
    ax.legend(loc="upper right", frameon=False, handlelength=1.2)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.3)

    # Panel C: overhead ms
    ax = axes[2]
    ax.bar(short, overhead, color="#EF6C00", edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Overhead per action (ms)")
    ax.set_title("(c) Governance overhead", pad=3)
    ax.set_ylim(0, max(overhead) * 1.6)
    for i, v in enumerate(overhead):
        ax.text(i, v + max(overhead) * 0.05,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_safety.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'fig_safety.pdf'}")


# -------------------------------------------------------------------
# Figure 5: Ablation of learning modalities
# -------------------------------------------------------------------
def fig_ablation():
    data = _load("ablation_modalities.json")
    configs = ["rl_only", "il_only", "llm_only",
               "rl_il", "rl_llm", "il_llm", "all_three"]
    labels = ["RL", "IL", "LLM", "RL+IL", "RL+LLM", "IL+LLM", "All three"]

    final_reward, peak_reward, peak_success = [], [], []
    for c in configs:
        ms = data[c]["metrics"]
        final_reward.append(ms[-1]["avg_reward"])
        peak_reward.append(max(x["avg_reward"] for x in ms))
        peak_success.append(max(x["success_rate"] for x in ms))

    x = np.arange(len(configs))
    w = 0.4

    fig, ax1 = plt.subplots(figsize=(FIGW_COL, 2.2))
    b1 = ax1.bar(x - w/2, peak_reward, w, color=C_OURS,
                 edgecolor="black", linewidth=0.4, label="Peak reward")
    b2 = ax1.bar(x + w/2, final_reward, w, color=C_OURS, alpha=0.5,
                 edgecolor="black", linewidth=0.4, label="Final reward")
    ax1.set_ylabel("Reward", color=C_OURS, fontsize=7)
    ax1.tick_params(axis="y", labelcolor=C_OURS, labelsize=6)

    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, peak_success, color=C_AM, marker="o", markersize=4,
             linewidth=1.0, linestyle="--", label="Peak success")
    ax2.set_ylabel("Peak success (\\%)", color=C_AM, fontsize=7)
    ax2.tick_params(axis="y", labelcolor=C_SUCC, labelsize=6)
    ax2.set_ylim(-0.5, max(peak_success) * 1.5 + 1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(True, axis="y", alpha=0.25, linewidth=0.3)
    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False,
               handlelength=1.2, fontsize=6.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'fig_ablation.pdf'}")


# -------------------------------------------------------------------
# Figure 6: Identity drift (Δπ)
# -------------------------------------------------------------------
def fig_identity_drift():
    data = _load("exp2_method_comparison.json")
    methods = ["ours", "agent_modification", "static_ecm", "spirl", "skimo"]
    tasks = TASKS

    drifts = np.zeros((len(methods), len(tasks)))
    for i, m in enumerate(methods):
        for j, t in enumerate(tasks):
            last = data[t][m]["metrics"][-1]
            drifts[i, j] = last.get("policy_drift", 0.0)

    mean_drift = drifts.mean(axis=1)

    fig, ax = plt.subplots(figsize=(FIGW_COL, 1.9))
    short = ["Ours", "AM", "S-ECM", "SPiRL", "SkiMo"]
    colors = [METHOD_COLORS[m] for m in methods]
    ax.bar(short, mean_drift, color=colors, edgecolor="black", linewidth=0.4)
    for i, v in enumerate(mean_drift):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel(r"Policy drift $\Delta\pi$")
    ax.set_ylim(0, max(mean_drift.max() * 1.25, 0.1))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.4)

    fig.tight_layout()
    fig.savefig(OUT / "fig_identity_drift.pdf")
    plt.close(fig)
    print(f"  wrote {OUT/'fig_identity_drift.pdf'}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print(f"Output dir: {OUT}")
    fig_evolution_curve()
    fig_t6_flagship()
    fig_method_peak_rewards()
    fig_safety()
    fig_ablation()
    fig_identity_drift()
    print("All figures written.")


if __name__ == "__main__":
    main()
