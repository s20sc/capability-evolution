#!/usr/bin/env python3
"""Render fig_identity_drift.pdf: agent-policy drift over T6 iterations.

Reads only the attested per-seed result files (seeds 7, 123, 2024) shipped
under results/, and asserts the audited end-of-run drift means so the
figure fails loudly if the data moves.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEEDS = ["7", "123", "2024"]

ARMS = {
    "agent_modification": ("Agent Modification", "#0072B2", "-",
        lambda sd: f"{ROOT}/results/t6_multiseed/seed_{sd}/t6_agent_modification.json"),
    "skimo": ("SkiMo", "#009E73", (0, (4, 2)),
        lambda sd: f"{ROOT}/results/multiseed/T6_TwoArmPegInHole/seed_{sd}/skimo.json"),
    "amphase": ("AM+rotate", "#5D3A9B", (0, (1, 1.2)),
        lambda sd: sorted(
            f"{ROOT}/results/t6_amphase/seed_{sd}/" + f
            for f in os.listdir(f"{ROOT}/results/t6_amphase/seed_{sd}")
            if f.endswith(".json"))[0]),
    "ours": ("Ours / SPiRL / Static ECM ($\\Delta\\pi{=}0$)", "#D55E00", "-",
        lambda sd: f"{ROOT}/results/t6_multiseed/seed_{sd}/t6_ours.json"),
}
EXPECTED_FINAL = {"agent_modification": 1.55, "skimo": 1.34, "amphase": 2.36, "ours": 0.00}


def drift_series(path):
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    mets = d["metrics"] if "metrics" in d else d
    return [m.get("policy_drift", 0.0) for m in mets]


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.17)
    for key, (label, color, ls, pathf) in ARMS.items():
        series = [drift_series(pathf(sd)) for sd in SEEDS]
        n = min(len(s) for s in series)
        arr = np.vstack([s[:n] for s in series])
        x = np.arange(1, n + 1)
        for row in arr:
            ax.plot(x, row, color=color, alpha=0.3, linewidth=0.8, linestyle=ls)
        mean = arr.mean(axis=0)
        ax.plot(x, mean, color=color, linewidth=2.0, linestyle=ls, label=label)
        final = round(float(mean[-1]), 2)
        assert abs(final - EXPECTED_FINAL[key]) < 0.02, (key, final)
    ax.set_xlabel("Evolution iteration (T6)")
    ax.set_ylabel("Agent-policy drift $\\Delta\\pi$")
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="0.75", fontsize=7.5)
    out = os.path.join(HERE, "fig_identity_drift.pdf")
    fig.savefig(out)
    print("[written]", out)


if __name__ == "__main__":
    main()
