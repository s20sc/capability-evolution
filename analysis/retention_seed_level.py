"""Seed-level re-analysis of T6 retention.

Retention is P4 / (P3 + P4): among evaluation episodes that reach the full
success geometry at some step, the fraction that still hold it when the
episode ends.

The earlier analysis pooled every evaluation episode of an arm into a single
binomial and put a Wilson interval on it.  Episodes are not independent draws:
they are clustered inside seeds, and inside iterations within a seed.  Pooling
across that structure understates the variance by a large factor, and the
symptom is that an arm's own per-seed retention values fall outside the
interval computed from its own pooled episodes.  This script recomputes
everything with the seed as the unit of analysis and reports the tests that
the design can actually support.

Windows.  The rotating arms (ours, amphase) ran 20 iterations at 4 x 50k
environment steps each; the single-policy control (nophase) ran 10 iterations
at 1 x 200k.  Per-iteration interaction is therefore 200k for all three, and
a comparison is budget-matched only when the arms are cut to the same number
of iterations.  The primary window here is the first eight iterations
(1.6M environment steps per seed), the same window the peak-success table
uses.  The full available window is reported as a secondary view because that
is what the previous version of the retention table pooled.

Outputs retention_seed_level.json and retention_seed_level.md next to itself.

Run: python3 retention_seed_level.py
"""

from __future__ import annotations

import json
import math
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent

# arm label -> (results root relative to this file, method key)
ARMS: Dict[str, Tuple[str, str]] = {
    "AM+single": ("..", "nophase"),
    "Ours": ("..", "ours"),
    "AM+rotate": ("..", "amphase"),
    "Agent Modification": ("..", "am"),
    "SkiMo": ("..", "skimo"),
}

PRIMARY_WINDOW = 8  # iterations; 8 x 200k = 1.6M env steps per seed for every arm
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 20260727


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_arm(root: str, method: str) -> Dict[int, List[Dict[str, int]]]:
    """Return {seed: [phase_histogram per iteration, in iteration order]}."""
    base = HERE / root / "results" / "t6_mechanism" / method
    out: Dict[int, List[Dict[str, int]]] = {}
    for path in sorted(base.glob("seed_*/mechanism.json")):
        if "__smoke" in path.parent.name:
            continue
        payload = json.loads(path.read_text())
        records = sorted(payload["records"], key=lambda r: r["iteration"])
        out[int(payload["seed"])] = [r["phase_histogram"] for r in records]
    if not out:
        raise FileNotFoundError(f"no mechanism.json under {base}")
    return out


def counts(hists: Sequence[Dict[str, int]], window: int | None) -> Dict[str, int]:
    use = hists if window is None else hists[:window]
    agg = {k: 0 for k in ("P0", "P1", "P2", "P3", "P4")}
    for h in use:
        for k in agg:
            agg[k] += int(h.get(k, 0))
    agg["episodes"] = sum(agg[k] for k in ("P0", "P1", "P2", "P3", "P4"))
    agg["reached"] = agg["P3"] + agg["P4"]
    agg["iterations"] = len(use)
    return agg


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval.  Kept only to reproduce the discredited numbers."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution (regularised incomplete gamma)."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    xx = x / 2.0
    if xx < a + 1.0:  # series
        term = 1.0 / a
        total = term
        n = 0
        while n < 10000:
            n += 1
            term *= xx / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    # continued fraction
    tiny = 1e-300
    b = xx + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h


def homogeneity(rows: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    """Chi-square test that per-seed retention rates are equal.

    rows: [(held, reached_but_lost)] per seed, i.e. (P4, P3).
    """
    rows = [r for r in rows if sum(r) > 0]
    k = len(rows)
    if k < 2:
        return {"applicable": False}
    n = [sum(r) for r in rows]
    total_n = sum(n)
    total_k = sum(r[0] for r in rows)
    p = total_k / total_n
    pearson = 0.0
    g = 0.0
    for (held, lost), ni in zip(rows, n):
        for obs, exp in ((held, ni * p), (lost, ni * (1 - p))):
            if exp > 0:
                pearson += (obs - exp) ** 2 / exp
            if obs > 0 and exp > 0:
                g += 2 * obs * math.log(obs / exp)
    df = k - 1
    out = {
        "applicable": True,
        "k_seeds": k,
        "df": df,
        "pooled_rate": p,
        "chi2_pearson": pearson,
        "p_pearson": chi2_sf(pearson, df),
        "g_statistic": g,
        "p_g": chi2_sf(g, df),
    }
    if k == 2:  # Yates continuity correction is defined for the 2x2 case
        (h0, l0), (h1, l1) = rows
        e = [n[0] * p, n[0] * (1 - p), n[1] * p, n[1] * (1 - p)]
        o = [h0, l0, h1, l1]
        yates = sum((max(0.0, abs(oi - ei) - 0.5)) ** 2 / ei for oi, ei in zip(o, e) if ei > 0)
        out["chi2_yates"] = yates
        out["p_yates"] = chi2_sf(yates, 1)
    return out


def mannwhitney_exact(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Exact two-sided Mann-Whitney (permutation over all group assignments)."""
    pooled = list(a) + list(b)
    n, m = len(a), len(b)

    def u_of(x: Sequence[float], y: Sequence[float]) -> float:
        u = 0.0
        for xi in x:
            for yj in y:
                if xi > yj:
                    u += 1.0
                elif xi == yj:
                    u += 0.5
        return u

    obs = u_of(a, b)
    idx = range(n + m)
    tail_lo = tail_hi = 0
    total = 0
    for pick in combinations(idx, n):
        total += 1
        xs = [pooled[i] for i in pick]
        ys = [pooled[i] for i in idx if i not in pick]
        u = u_of(xs, ys)
        if u <= obs:
            tail_lo += 1
        if u >= obs:
            tail_hi += 1
    p_two = min(1.0, 2.0 * min(tail_lo, tail_hi) / total)
    cliffs = (2.0 * obs / (n * m)) - 1.0
    return {
        "n_a": n,
        "n_b": m,
        "U": obs,
        "p_two_sided_exact": p_two,
        "min_attainable_p": 2.0 / total,
        "n_permutations": total,
        "cliffs_delta": cliffs,
        "complete_separation": bool(min(a) > max(b) or min(b) > max(a)),
    }


def cluster_bootstrap(per_seed: Sequence[Tuple[int, int]], draws: int, rng: random.Random):
    """Resample seeds with replacement; each draw re-pools that seed's episodes.

    per_seed: [(held, reached)] per seed.
    """
    seeds = [s for s in per_seed if s[1] > 0]
    if len(seeds) < 2:
        return None
    stats = []
    for _ in range(draws):
        pick = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        num = sum(h for h, _ in pick)
        den = sum(n for _, n in pick)
        if den > 0:
            stats.append(num / den)
    stats.sort()
    lo = stats[int(0.025 * (len(stats) - 1))]
    hi = stats[int(0.975 * (len(stats) - 1))]
    return {"draws": len(stats), "ci_lo": lo, "ci_hi": hi}


# --------------------------------------------------------------------------
# analysis over one window
# --------------------------------------------------------------------------

def analyse(window: int | None, raw: Dict[str, Dict[int, List[Dict[str, int]]]]):
    rng = random.Random(BOOTSTRAP_SEED)
    arms = {}
    for label, per_seed_hist in raw.items():
        rows = []
        for seed in sorted(per_seed_hist):
            c = counts(per_seed_hist[seed], window)
            ret = c["P4"] / c["reached"] if c["reached"] else None
            rows.append({
                "seed": seed,
                "iterations": c["iterations"],
                "episodes": c["episodes"],
                "P0": c["P0"], "P3": c["P3"], "P4": c["P4"],
                "reached": c["reached"],
                "retention": ret,
            })
        vals = [r["retention"] for r in rows if r["retention"] is not None]
        pooled_num = sum(r["P4"] for r in rows)
        pooled_den = sum(r["reached"] for r in rows)
        pooled = pooled_num / pooled_den if pooled_den else None
        w = wilson(pooled_num, pooled_den) if pooled_den else (None, None)
        outside = [r["seed"] for r in rows
                   if r["retention"] is not None and pooled_den
                   and not (w[0] <= r["retention"] <= w[1])]
        boot = cluster_bootstrap([(r["P4"], r["reached"]) for r in rows], BOOTSTRAP_DRAWS, rng)
        arms[label] = {
            "per_seed": rows,
            "n_seeds_with_data": len(vals),
            "seed_mean": (sum(vals) / len(vals)) if vals else None,
            "seed_min": min(vals) if vals else None,
            "seed_max": max(vals) if vals else None,
            "pooled_episode_level": pooled,
            "pooled_wilson_ci": [w[0], w[1]],
            "seeds_outside_own_pooled_wilson": outside,
            "cluster_bootstrap_over_seeds": boot,
            "homogeneity_across_seeds": homogeneity([(r["P4"], r["P3"]) for r in rows]),
            "total_episodes": sum(r["episodes"] for r in rows),
            "P0_fraction_pooled": (sum(r["P0"] for r in rows) / sum(r["episodes"] for r in rows))
                                   if sum(r["episodes"] for r in rows) else None,
            "reached_fraction_pooled": (pooled_den / sum(r["episodes"] for r in rows))
                                        if sum(r["episodes"] for r in rows) else None,
        }

    def vals(label):
        return [r["retention"] for r in arms[label]["per_seed"] if r["retention"] is not None]

    tests = {}
    for a, b in (("AM+single", "Ours"), ("AM+single", "AM+rotate"), ("AM+rotate", "Ours")):
        va, vb = vals(a), vals(b)
        if len(va) >= 1 and len(vb) >= 1:
            tests[f"{a} vs {b}"] = {
                "values_a": va, "values_b": vb,
                **mannwhitney_exact(va, vb),
            }
    return {"window_iterations": window, "arms": arms, "seed_level_tests": tests}


# --------------------------------------------------------------------------

def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main() -> None:
    raw = {label: load_arm(root, method) for label, (root, method) in ARMS.items()}

    result = {
        "generated_by": "retention_seed_level.py",
        "definition": "retention = P4 / (P3 + P4), per seed",
        "sources": {label: f"{root}/results/t6_mechanism/{method}/seed_*/mechanism.json"
                    for label, (root, method) in ARMS.items()},
        "budget_note": ("every arm consumes 200k environment steps per iteration "
                        "(AM+single 1x200k, rotating arms 4x50k), so cutting all arms to the "
                        "same iteration count matches cumulative interaction exactly"),
        "primary": analyse(PRIMARY_WINDOW, raw),
        "secondary_full_window": analyse(None, raw),
    }

    (HERE / "retention_seed_level.json").write_text(json.dumps(result, indent=2))

    lines: List[str] = []
    lines.append("# Seed-level retention on T6 (generated by retention_seed_level.py)\n")
    lines.append("retention = P4 / (P3 + P4). Unit of analysis is the seed.\n")
    for tag, key in (("PRIMARY: budget-matched, first 8 iterations (1.6M env steps/seed)", "primary"),
                     ("SECONDARY: full available window (AM+single 10 iters, rotating arms 20)",
                      "secondary_full_window")):
        blk = result[key]
        lines.append(f"\n## {tag}\n")
        lines.append("| Arm | seed | iters | episodes | P3 | P4 | reached | retention |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for label in ARMS:
            for r in blk["arms"][label]["per_seed"]:
                lines.append(f"| {label} | {r['seed']} | {r['iterations']} | {r['episodes']} | "
                             f"{r['P3']} | {r['P4']} | {r['reached']} | {fmt(r['retention'])} |")
        lines.append("")
        lines.append("| Arm | seeds | seed mean | seed range | pooled (episode-level) | "
                     "pooled Wilson (invalid) | seeds outside it | cluster bootstrap CI | "
                     "homogeneity chi2 (df, p) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for label in ARMS:
            a = blk["arms"][label]
            w = a["pooled_wilson_ci"]
            b = a["cluster_bootstrap_over_seeds"]
            h = a["homogeneity_across_seeds"]
            hs = ("n/a" if not h.get("applicable") else
                  f"{h['chi2_pearson']:.1f} (df={h['df']}, p={h['p_pearson']:.2e})")
            bs = "n/a" if not b else "[" + fmt(b["ci_lo"]) + ", " + fmt(b["ci_hi"]) + "]"
            lines.append(
                f"| {label} | {a['n_seeds_with_data']} | {fmt(a['seed_mean'])} | "
                f"{fmt(a['seed_min'])}-{fmt(a['seed_max'])} | {fmt(a['pooled_episode_level'])} | "
                f"[{fmt(w[0])}, {fmt(w[1])}] | {a['seeds_outside_own_pooled_wilson']} | "
                f"{bs} | {hs} |")
        lines.append("\n### Seed-level exact Mann-Whitney (two-sided)\n")
        lines.append("| Comparison | values A | values B | U | p (exact) | smallest p this design can return | Cliff's delta | complete separation |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, t in blk["seed_level_tests"].items():
            lines.append(
                f"| {name} | {[round(v,3) for v in t['values_a']]} | {[round(v,3) for v in t['values_b']]} | "
                f"{t['U']:.1f} | {t['p_two_sided_exact']:.4f} | {t['min_attainable_p']:.4f} | "
                f"{t['cliffs_delta']:+.2f} | {t['complete_separation']} |")
    (HERE / "retention_seed_level.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
