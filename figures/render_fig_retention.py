#!/usr/bin/env python3
"""Render fig_retention.pdf: seed-level retention dot plot (budget-matched window).

Horizontal dot plot of per-seed retention P4/(P3+P4) for the three arms of the
matched 8-iteration window ("primary" block of retention_seed_level.json),
showing complete separation between AM+single and both rotating arms.

Layout: standalone PDF at figsize (5.0, 2.6) in, included at \\linewidth
(~5.0 in) in a 12pt single-column manuscript, so all font sizes here
are final rendered sizes (>= 8 pt).

Usage:
    python3 render_fig_retention.py
Optional: set env FIG_PREVIEW_PNG=/path/to/preview.png to also dump a 200 dpi
PNG for visual inspection (the PDF is the canonical output).
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "analysis" / "retention_seed_level.json"
OUT_PDF = HERE / "fig_retention.pdf"

# ----------------------------------------------------------------------------
# Load and validate data (fail loudly on any mismatch with the paper's numbers)
# ----------------------------------------------------------------------------
with open(DATA, "r", encoding="utf-8") as fh:
    blob = json.load(fh)

primary = blob["primary"]
assert primary["window_iterations"] == 8, "expected the matched 8-iteration window"
arms_json = primary["arms"]

ARM_ORDER = ["AM+single", "Ours", "AM+rotate"]  # top to bottom
EXPECTED_N = {"AM+single": 4, "Ours": 4, "AM+rotate": 4}

arm_seeds = {}
arm_vals = {}
for arm in ARM_ORDER:
    per_seed = arms_json[arm]["per_seed"]
    assert len(per_seed) == EXPECTED_N[arm], (
        f"{arm}: expected {EXPECTED_N[arm]} seeds, got {len(per_seed)}"
    )
    arm_seeds[arm] = [e["seed"] for e in per_seed]
    arm_vals[arm] = [e["retention"] for e in per_seed]

mean_am_single = sum(arm_vals["AM+single"]) / len(arm_vals["AM+single"])
assert abs(mean_am_single - 0.734) < 0.001, (
    f"AM+single seed mean drifted: {mean_am_single:.6f} != 0.734"
)
assert abs(mean_am_single - arms_json["AM+single"]["seed_mean"]) < 1e-12

rotating_max = max(max(arm_vals["Ours"]), max(arm_vals["AM+rotate"]))
am_single_min = min(arm_vals["AM+single"])
assert rotating_max < 0.35, f"rotating max {rotating_max:.4f} >= 0.35"
assert am_single_min > 0.51, f"AM+single min {am_single_min:.4f} <= 0.51"

for key in ("AM+single vs Ours", "AM+single vs AM+rotate"):
    test = primary["seed_level_tests"][key]
    assert test["complete_separation"] is True, f"{key}: separation broken"
    assert abs(test["p_two_sided_exact"] - 2.0 / 70.0) < 1e-12, (
        f"{key}: exact p changed: {test['p_two_sided_exact']}"
    )

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)

COLORS = {
    "AM+single": "#0072B2",  # blue (paper-wide: the single-policy comparison)
    "Ours": "#D55E00",       # vermilion (paper-wide: our system / rotation arm)
    "AM+rotate": "#AA4499",  # deep purple (learned-dispatcher rotation)
}
Y_POS = {"AM+single": 2.0, "Ours": 1.0, "AM+rotate": 0.0}
Y_LABELS = {
    "AM+single": "AM+single\n(1 policy $\\times$ 200k)",
    "Ours": "Ours\n(4 rotated, fixed\nschedule)",
    "AM+rotate": "AM+rotate\n(4 rotated, learned\ndispatcher)",
}

fig, ax = plt.subplots(figsize=(5.0, 2.6))
# Explicit margins; bbox_inches='tight' is deliberately NOT used (figure
# consistency practice: keep the physical panel size deterministic).
fig.subplots_adjust(left=0.245, right=0.985, top=0.96, bottom=0.215)

# Shaded band marking the empty gap between the two clusters of seeds.
gap_lo = rotating_max            # 0.0667 = max over both rotating arms
gap_hi = am_single_min           # 0.5191 = min over AM+single seeds
ax.axvspan(gap_lo, gap_hi, color="#000000", alpha=0.06, zorder=0)
for edge in (gap_lo, gap_hi):
    ax.axvline(edge, color="#999999", lw=0.6, ls=(0, (2, 2)), zorder=1)

# Annotation sits in the empty lower-right quadrant so it cannot collide
# with seed labels, the y-tick labels, or the band's dashed edges.
ax.annotate(
    "complete separation across seeds\n"
    "(per-arm exact $p{=}0.029$, $n{=}4$ vs $4$)",
    xy=(0.98, 0.45),
    ha="right",
    va="center",
    fontsize=8,
    color="#444444",
    zorder=3,
)

# Label plan (rebuilt after a text-overlap report on the rendered figure):
# - AM+single seeds are spread out, so each gets its own label above its dot.
# - The rotating arms' clusters are too tight for per-dot labels at this
#   scale; each cluster gets ONE combined label, placed to the right of or
#   above the cluster, always growing INTO empty plot area and never toward
#   the y-tick labels at the left edge.
LABEL_POS = {
    ("AM+single", 123): (-0.008, 0.18, "right"),
    ("AM+single", 7): (-0.006, 0.18, "right"),
    ("AM+single", 42): (0.006, 0.18, "left"),
    ("AM+single", 2024): (0.0, 0.18, "center"),
}
CLUSTER_LABELS = {
    # arm -> (x, dy, ha, va, text)
    "Ours": (0.060, 0.0, "left", "center", "seeds 7/42/123/2024"),
    "AM+rotate": (0.002, 0.18, "left", "bottom", "7/42/123"),
}
ESCAPE_LABELS = {
    ("AM+rotate", 2024): (0.0, 0.18, "center"),
}

# Mean diamonds sit slightly below the seed row so they never occlude the
# tightly clustered per-seed dots of the rotating arms.
DIAMOND_DY = -0.30

for arm in ARM_ORDER:
    y = Y_POS[arm]
    color = COLORS[arm]
    vals = arm_vals[arm]
    seeds = arm_seeds[arm]
    # Filled circles: one per seed.
    ax.scatter(
        vals,
        [y] * len(vals),
        s=30,
        facecolor=color,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    # Open diamond at the seed mean.
    mean_v = sum(vals) / len(vals)
    ax.scatter(
        [mean_v],
        [y + DIAMOND_DY],
        s=58,
        marker="D",
        facecolor="none",
        edgecolor=color,
        linewidth=1.1,
        zorder=6,
    )
    # Seed-number labels: per-dot for the spread arm, one combined label per
    # tight cluster, and an individual label only for the escape seed.
    for seed, v in zip(seeds, vals):
        if (arm, seed) in LABEL_POS:
            dx, dy, ha = LABEL_POS[(arm, seed)]
            ax.text(v + dx, y + dy, str(seed), ha=ha,
                    va="bottom" if dy > 0 else "top",
                    fontsize=8, color=color, zorder=7)
        elif (arm, seed) in ESCAPE_LABELS:
            dx, dy, ha = ESCAPE_LABELS[(arm, seed)]
            ax.text(v + dx, y + dy, str(seed), ha=ha, va="bottom",
                    fontsize=8, color=color, zorder=7)
    if arm in CLUSTER_LABELS:
        cx, cdy, cha, cva, ctext = CLUSTER_LABELS[arm]
        ax.text(cx, y + cdy, ctext, ha=cha, va=cva,
                fontsize=8, color=color, zorder=7)

ax.set_xlim(-0.02, 1.0)
ax.set_ylim(-0.75, 2.75)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([Y_POS[a] for a in ARM_ORDER])
ax.set_yticklabels([Y_LABELS[a] for a in ARM_ORDER], fontsize=8)
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", length=0)
ax.set_xlabel(
    "Retention  $P_4/(P_3{+}P_4)$  over the budget-matched window", fontsize=8.5
)
ax.xaxis.grid(True, color="#dddddd", lw=0.5, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)

fig.savefig(OUT_PDF)
preview = os.environ.get("FIG_PREVIEW_PNG")
if preview:
    fig.savefig(preview, dpi=200)
plt.close(fig)

print(f"asserts passed; wrote {OUT_PDF}")
for arm in ARM_ORDER:
    pairs = ", ".join(
        f"seed {s}: {v:.4f}" for s, v in zip(arm_seeds[arm], arm_vals[arm])
    )
    print(f"  {arm}: {pairs}  (mean {sum(arm_vals[arm]) / len(arm_vals[arm]):.4f})")
