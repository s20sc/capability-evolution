#!/usr/bin/env python3
"""Render fig_nophase_curves.pdf: budget-matched evolution curves.

Per-iteration task success rate over the matched first-8-iteration window
for three arms on T6 (TwoArmPegInHole):
  - AM+single (nophase): 4 seeds (7, 42, 123, 2024), 10 iterations recorded,
    clipped at iteration 8 to match the budget-matched comparison window.
  - Ours: 2 seeds (7, 42), first 8 of 20 iterations.
  - AM+rotate (amphase): 2 seeds (7, 42), first 8 of 20 iterations.

Thin per-seed lines (alpha 0.35) plus a thick seed-mean line per arm.
Data numbers are hard-asserted against the audited values before plotting
so that any data drift fails the script loudly instead of silently
producing a wrong figure.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Single remapping point: the artifact repo copy only needs to change BASE.
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(OUT_DIR, "fig_nophase_curves.pdf")

# Matched comparison window: first 8 evolution iterations.
WINDOW = 8

ARMS = {
    # arm key -> (relative results dir, seeds, color, legend label)
    # Ours and AM+rotate genuinely coincide near the floor, so hue alone
    # cannot separate them there: the two rotating arms differ by line
    # style, mean marker, and a hue family of its own (purple vs vermilion).
    "nophase": ("results/t6_mechanism/nophase",
                [7, 42, 123, 2024], "#0072B2", "AM+single"),
    "ours": ("results/t6_mechanism/ours",
             [7, 42, 123, 2024], "#D55E00", "Ours"),
    "amphase": ("results/t6_mechanism/amphase",
                [7, 42, 123, 2024], "#AA4499", "AM+rotate"),
}
ARM_STYLE = {
    # arm -> (linestyle, mean marker)
    "nophase": ("-", None),
    "ours": ("-", "o"),
    "amphase": ((0, (4, 2)), "^"),
}

# Audited per-seed peak success rates (%) over the first-8 window.
EXPECTED_PEAKS = {
    "nophase": {7: 100.0, 42: 93.3, 123: 100.0, 2024: 86.7},
    "ours": {7: 3.3, 42: 3.3, 123: 3.3, 2024: 3.3},
    "amphase": {7: 3.3, 42: 6.7, 123: 3.3, 2024: 43.3},
}
EXPECTED_MEAN_PEAK = {"nophase": 95.0, "ours": 3.3, "amphase": 14.2}
PEAK_TOL = 0.05


def load_series(arm_dir: str, seed: int) -> np.ndarray:
    """Load per-iteration success_rate (%) for one seed, clipped to WINDOW."""
    path = os.path.join(BASE, arm_dir, f"seed_{seed}", "mechanism.json")
    assert "__smoke" not in path, f"smoke run path leaked in: {path}"
    with open(path) as fh:
        data = json.load(fh)
    records = sorted(data["records"], key=lambda r: r["iteration"])
    iters = [r["iteration"] for r in records]
    assert iters[:WINDOW] == list(range(1, WINDOW + 1)), (
        f"unexpected iteration numbering in {path}: {iters[:WINDOW]}")
    y = np.array([r["success_rate"] for r in records[:WINDOW]], dtype=float)
    assert len(y) == WINDOW, f"{path}: only {len(y)} iterations, need {WINDOW}"
    return y


def main() -> None:
    # ---- load + hard-assert data --------------------------------------
    series = {}   # arm -> {seed: np.ndarray}
    for arm, (rel_dir, seeds, _color, _label) in ARMS.items():
        series[arm] = {}
        for seed in seeds:
            y = load_series(rel_dir, seed)
            peak = float(np.max(y))
            exp = EXPECTED_PEAKS[arm][seed]
            assert abs(peak - exp) <= PEAK_TOL, (
                f"{arm} seed {seed}: first-{WINDOW} peak {peak:.2f}% "
                f"!= expected {exp:.2f}% (tol {PEAK_TOL})")
            series[arm][seed] = y
        mean_peak = float(np.mean(
            [np.max(v) for v in series[arm].values()]))
        exp_mean = EXPECTED_MEAN_PEAK[arm]
        assert abs(mean_peak - exp_mean) <= PEAK_TOL, (
            f"{arm}: mean first-{WINDOW} peak {mean_peak:.2f}% "
            f"!= expected {exp_mean:.2f}%")
        print(f"[assert OK] {arm}: per-seed peaks "
              f"{ {s: round(float(np.max(v)), 1) for s, v in series[arm].items()} }, "
              f"mean peak {mean_peak:.1f}%")

    # ---- style --------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.155)

    x = np.arange(1, WINDOW + 1)
    for arm, (_rel_dir, seeds, color, label) in ARMS.items():
        ls, marker = ARM_STYLE[arm]
        # Thin per-seed traces (same line style as the arm's mean).
        for seed in seeds:
            ax.plot(x, series[arm][seed], color=color, linewidth=0.8,
                    alpha=0.35, zorder=2, linestyle=ls)
        # Thick seed-mean trace; markers separate the two near-zero arms
        # where their means overlap.
        mean_y = np.mean(np.vstack([series[arm][s] for s in seeds]), axis=0)
        ax.plot(x, mean_y, color=color, linewidth=2.0, alpha=1.0,
                zorder=4 if arm == "amphase" else 3, label=label,
                linestyle=ls, marker=marker, markersize=4.2,
                markerfacecolor="white" if arm == "amphase" else color,
                markeredgecolor=color, markeredgewidth=1.0)

    ax.set_xlabel("Evolution iteration (budget-matched window)")
    ax.set_ylabel("Task success rate (%)")
    ax.set_xlim(0.85, WINDOW + 0.15)
    ax.set_ylim(-4, 104)
    ax.set_xticks(x)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="0.88", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Lower-right pocket is free of curves: AM+single traces run above it
    # and the near-zero Ours/AM+rotate traces run below it.
    leg = ax.legend(loc="lower right", bbox_to_anchor=(0.985, 0.13),
                    frameon=True, framealpha=0.92, edgecolor="0.75",
                    borderpad=0.5, handlelength=1.8)
    leg.get_frame().set_linewidth(0.6)

    fig.savefig(OUT_PDF)
    print(f"[written] {OUT_PDF}")


if __name__ == "__main__":
    main()
