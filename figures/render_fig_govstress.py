#!/usr/bin/env python3
"""Render fig_govstress.pdf: governance paralysis under adversarial stress.

For each task T1..T6, three grouped bars computed from
results/governance_stress/{task}/stress_{off,on}.json:

  1. Violation reduction (%)  = (viol_off - viol_on) / viol_off * 100,
     using metrics[0].env_violations_total.
  2. Actions discarded (%)    = forbidden-zone interventions / total actions
     * 100, using governance.interventions_by_type["forbidden_zone"] and
     governance.internal_metrics.total_actions (25000 per task).
  3. Task success with governance ON (%) = metrics[0].success_rate of the
     ON condition. This is 0.0 on every task, so the third bar is drawn as
     a flat zero marker with a single annotation.

All computed values are hard-asserted against the audited numbers before
plotting so any data drift fails the script loudly.
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
OUT_PDF = os.path.join(OUT_DIR, "fig_govstress.pdf")

STRESS_DIR = os.path.join(BASE, "results",
                          "governance_stress")

TASKS = ["T1_Pick", "T2_Place", "T3_Stack", "T4_NutAssembly",
         "T5_PickPlaceMulti", "T6_TwoArmPegInHole"]
TICK_LABELS = ["T1", "T2", "T3", "T4", "T5", "T6"]

# Audited values (tolerance 0.1 percentage points).
EXPECTED_REDUCTION = [100.0, 99.2, 100.0, 98.2, 100.0, 34.9]
EXPECTED_DISCARDED = [100.0, 97.9, 99.8, 98.0, 100.0, 75.1]
EXPECTED_SUCCESS_ON = [0.0] * 6
EXPECTED_SUCCESS_OFF_T6 = 2.0
TOL = 0.1

COLOR_REDUCTION = "#808080"   # mid-grey
COLOR_DISCARDED = "#D55E00"   # vermilion (paper-wide: our machinery / its cost)
COLOR_SUCCESS = "#2b2b2b"     # dark (zero-height marker)


def load(task: str, condition: str) -> dict:
    path = os.path.join(STRESS_DIR, task, f"stress_{condition}.json")
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    # ---- compute + hard-assert data -----------------------------------
    reduction, discarded, success_on = [], [], []
    for i, task in enumerate(TASKS):
        off = load(task, "off")
        on = load(task, "on")

        viol_off = off["metrics"][0]["env_violations_total"]
        viol_on = on["metrics"][0]["env_violations_total"]
        red = (viol_off - viol_on) / viol_off * 100.0

        gov = on["governance"]
        fz = gov["interventions_by_type"].get("forbidden_zone", 0)
        total_actions = gov["internal_metrics"]["total_actions"]
        assert total_actions == 25000, (
            f"{task}: total_actions {total_actions} != 25000")
        disc = fz / total_actions * 100.0

        succ_on = float(on["metrics"][0]["success_rate"])   # already in %
        succ_off = float(off["metrics"][0]["success_rate"])  # already in %

        assert abs(red - EXPECTED_REDUCTION[i]) <= TOL, (
            f"{task}: violation reduction {red:.2f}% "
            f"!= expected {EXPECTED_REDUCTION[i]}%")
        assert abs(disc - EXPECTED_DISCARDED[i]) <= TOL, (
            f"{task}: actions discarded {disc:.2f}% "
            f"!= expected {EXPECTED_DISCARDED[i]}%")
        assert abs(succ_on - EXPECTED_SUCCESS_ON[i]) <= TOL, (
            f"{task}: success ON {succ_on:.2f}% != expected 0.0%")
        if task == "T6_TwoArmPegInHole":
            assert abs(succ_off - EXPECTED_SUCCESS_OFF_T6) <= TOL, (
                f"{task}: success OFF {succ_off:.2f}% != expected "
                f"{EXPECTED_SUCCESS_OFF_T6}%")

        reduction.append(red)
        discarded.append(disc)
        success_on.append(succ_on)
        print(f"[assert OK] {task}: viol_off={viol_off} viol_on={viol_on} "
              f"reduction={red:.1f}% discarded={disc:.1f}% "
              f"success_on={succ_on:.1f}%")

    # ---- style --------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.97, bottom=0.13)

    x = np.arange(len(TASKS))
    w = 0.26

    h_red = ax.bar(x - w, reduction, width=w, color=COLOR_REDUCTION,
                   edgecolor="white", linewidth=0.4, zorder=3,
                   label="Violation reduction")
    h_disc = ax.bar(x, discarded, width=w, color=COLOR_DISCARDED,
                    edgecolor="white", linewidth=0.4, zorder=3,
                    label="Discarded actions")
    # Third series is 0.0 everywhere: draw flat zero markers, not bars.
    h_succ, = ax.plot(x + w, success_on, linestyle="none", marker="_",
                      markersize=9, markeredgewidth=2.2, color=COLOR_SUCCESS,
                      zorder=4, label="Success (governance on)")

    # Annotate T6's partial violation reduction clearly.
    ax.annotate(f"{reduction[5]:.1f}%", xy=(x[5] - w, reduction[5]),
                xytext=(x[5] - w, reduction[5] + 3.5),
                ha="center", va="bottom", fontsize=7.5,
                color="0.25")

    # Single annotation for the all-zero success series, placed in the
    # free band above the bars. The text's right edge sits directly above
    # T6's zero marker so the leader line drops vertically, clear of the
    # T6 bars (whose right edge is at x[5] + w/2).
    ax.annotate("success under governance: 0.0% on every task",
                xy=(x[5] + w, 2.0), xytext=(x[5] + w, 109),
                ha="right", va="center", fontsize=7.2, color=COLOR_SUCCESS,
                arrowprops=dict(arrowstyle="-", color=COLOR_SUCCESS,
                                linewidth=0.7, shrinkA=5, shrinkB=2,
                                relpos=(1.0, 0.0)))

    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(TICK_LABELS)
    ax.set_xlim(-0.65, len(TASKS) - 0.20)
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="0.88", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(handles=[h_red, h_disc, h_succ], loc="upper left",
              ncol=3, frameon=False, fontsize=7,
              handlelength=1.2, handletextpad=0.5,
              columnspacing=0.9, borderaxespad=0.2)

    fig.savefig(OUT_PDF)
    print(f"[written] {OUT_PDF}")


if __name__ == "__main__":
    main()
