#!/usr/bin/env python3
"""Render fig_lifecycle.pdf: deployed reward vs evolution iteration under
fault injection, one panel per promotion policy (T6 lifecycle experiment).

Data source (per arm / per seed):
  results/t6_lifecycle/<arm>/seed_<s>/t6_lifecycle.json
    - metrics[i].avg_reward          : deployed mean reward at iteration i
    - metrics[i].iteration           : 1-based iteration index
    - metrics[i].injection_flag      : True at fault-injection iterations
    - metrics[i].rollback_triggered  : True where the rollback detector fired
Aggregates:
  .../summary_lifecycle.json
    - arms/gating_off_rollback/per_seed/<seed>/n_rollbacks

The script asserts (fails loudly) that:
  1. gating_on mean dip at injection iterations is < 5% of the local
     pre-injection baseline median,
  2. gating_off mean post-injection drop is > 5%,
  3. per-seed rollback counts in the summary == {7:3, 42:3, 123:3, 2024:0}
     and match the raw rollback_triggered flags.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent / (
    "results/t6_lifecycle"
)
OUT = Path(__file__).resolve().parent / "fig_lifecycle.pdf"

SEEDS = [7, 42, 123, 2024]
ARMS = ["gating_on", "gating_off", "gating_off_rollback"]
INJECT_AT = [5, 8, 11]
EXPECTED_ROLLBACKS = {7: 3, 42: 3, 123: 3, 2024: 0}

# ---------------------------------------------------------------- load data
data = {}  # arm -> seed -> dict(iters, rewards, rollback_iters)
for arm in ARMS:
    data[arm] = {}
    for seed in SEEDS:
        p = BASE / arm / f"seed_{seed}" / "t6_lifecycle.json"
        d = json.loads(p.read_text())
        metrics = d["metrics"]
        iters = np.array([m["iteration"] for m in metrics])
        rewards = np.array([m["avg_reward"] for m in metrics])
        inj = [m["iteration"] for m in metrics if m.get("injection_flag")]
        rb = [m["iteration"] for m in metrics if m.get("rollback_triggered")]
        assert inj == INJECT_AT, (
            f"{arm}/seed_{seed}: injection_flag iterations {inj} != {INJECT_AT}"
        )
        data[arm][seed] = {"iters": iters, "rewards": rewards, "rb": rb}

summary = json.loads((BASE / "summary_lifecycle.json").read_text())

# ---------------------------------------------------------------- asserts
def seed_mean(arm):
    """Mean reward across seeds at each iteration (all runs share the grid)."""
    grids = [data[arm][s]["iters"] for s in SEEDS]
    assert all((g == grids[0]).all() for g in grids), f"{arm}: iteration grids differ"
    return grids[0], np.mean([data[arm][s]["rewards"] for s in SEEDS], axis=0)


def mean_dip_pct(arm):
    """Mean relative drop of the seed-mean reward at injection iterations,
    against the median of the 3 preceding (local baseline) iterations."""
    iters, mean = seed_mean(arm)
    dips = []
    for t in INJECT_AT:
        pre = mean[np.isin(iters, [t - 3, t - 2, t - 1])]
        baseline = float(np.median(pre))
        at_t = float(mean[iters == t][0])
        dips.append(100.0 * (baseline - at_t) / baseline)
    return float(np.mean(dips))


dip_on = mean_dip_pct("gating_on")
dip_off = mean_dip_pct("gating_off")
assert dip_on < 5.0, f"gating_on mean injection dip {dip_on:.2f}% is not < 5%"
assert dip_off > 5.0, f"gating_off mean injection dip {dip_off:.2f}% is not > 5%"

per_seed = summary["arms"]["gating_off_rollback"]["per_seed"]
summary_counts = {int(k): v["n_rollbacks"] for k, v in per_seed.items()}
assert summary_counts == EXPECTED_ROLLBACKS, (
    f"summary rollback counts {summary_counts} != {EXPECTED_ROLLBACKS}"
)
raw_counts = {s: len(data["gating_off_rollback"][s]["rb"]) for s in SEEDS}
assert raw_counts == EXPECTED_ROLLBACKS, (
    f"raw rollback_triggered counts {raw_counts} != {EXPECTED_ROLLBACKS}"
)
print(f"[asserts] gating_on mean dip {dip_on:+.2f}% (<5%)  "
      f"gating_off mean dip {dip_off:+.2f}% (>5%)  "
      f"rollbacks {summary_counts} == expected  -- all OK")

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "pdf.fonttype": 42,
})

PANELS = [
    ("gating_on", "Gated promotion", "#7fcbb5", "#009E73"),
    ("gating_off", "No gate", "#b0b0b0", "#555555"),
    ("gating_off_rollback", "No gate + rollback detector", "#f2d5a0", "#D08C00"),
]

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(5.0, 3.4))
fig.subplots_adjust(left=0.105, right=0.985, top=0.975, bottom=0.115, hspace=0.14)

ymin, ymax = 190, 372
for ax, (arm, label, c_seed, c_mean) in zip(axes, PANELS):
    # injection markers (dashed vertical guides)
    for t in INJECT_AT:
        ax.axvline(t, color="#666666", lw=0.7, ls=(0, (4, 2.5)), alpha=0.55,
                   zorder=1)
    # per-seed thin lines
    for s in SEEDS:
        d = data[arm][s]
        ax.plot(d["iters"], d["rewards"], color=c_seed, lw=0.7, alpha=0.35,
                zorder=2, solid_capstyle="round")
    # seed-mean thick line
    iters, mean = seed_mean(arm)
    ax.plot(iters, mean, color=c_mean, lw=1.6, zorder=4,
            solid_capstyle="round")
    # rollback trigger markers (small upward triangles on each seed line)
    if arm == "gating_off_rollback":
        rb_handle = None
        for s in SEEDS:
            d = data[arm][s]
            if not d["rb"]:  # seed 2024: detector never fired -- no markers
                continue
            mask = np.isin(d["iters"], d["rb"])
            (rb_handle,) = ax.plot(
                d["iters"][mask], d["rewards"][mask], linestyle="none",
                marker="^", markersize=3.4, markerfacecolor="#B77800",
                markeredgecolor="white", markeredgewidth=0.35, zorder=5)
        ax.legend([rb_handle], ["rollback"], loc="lower right", fontsize=7,
                  frameon=False, handletextpad=0.3, borderaxespad=0.2,
                  handlelength=1.0)
    # panel label
    ax.text(0.015, 0.90, label, transform=ax.transAxes, fontsize=7.5,
            va="top", ha="left", color="#222222")
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([200, 250, 300, 350])
    ax.tick_params(direction="out")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

# "inject" label once, in the top panel next to the first dashed line
axes[0].text(5.18, ymax - 8, "inject", fontsize=7, color="#555555",
             va="top", ha="left", style="italic")

axes[-1].set_xlabel("Evolution iteration")
axes[-1].set_xlim(0.6, 13.4)
axes[-1].set_xticks(range(1, 14, 2))
fig.supylabel("Deployed mean reward", fontsize=8, x=0.012)

fig.savefig(OUT)
print(f"[render] wrote {OUT}")
