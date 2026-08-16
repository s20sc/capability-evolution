"""
T6 Multi-Seed Verification

Purpose
-------
The main exp2 run (seed=42) shows Ours at 86.7% peak success on
T6_TwoArmPegInHole vs Agent Modification at 3.3% — a 13x gap that
is the flagship T6 comparison. A single-seed result is not
sufficient evidence on its own; this script reruns the same
task/methods on 2–3 additional seeds so we can report mean ± std
and a Welch's t-test.

Scope (kept narrow on purpose)
------------------------------
- Task:     T6_TwoArmPegInHole only
- Methods:  "ours" + "agent_modification" only
- Default seeds: [123, 7, 2024]  (seed=42 lives in the existing
                                  exp2_method_comparison.json and
                                  is merged in at aggregation time)

Design
------
- Reuses ExperimentRunner from run_experiment.py — no edits to that file.
- Each (seed, method) result is saved under
    results/t6_multiseed/seed_<N>/t6_<method>.json
  so a crash in seed 7 / method AM does not affect seed 123 / method ours.
- Per-iteration save is inherited from ExperimentRunner, so an OOM or
  SubprocVecEnv EOFError in the middle of a seed is recoverable.
- Seeds the RNGs (torch, numpy, random, CUDA) globally at the top of
  each seed's run — not just the config field — so policy init and
  exploration noise genuinely differ across seeds.
- After every seed completes, the aggregate summary is rewritten so
  a partial run is still useful.

Usage
-----
  # Full 3-seed rerun (recommended)
  python run_t6_multiseed.py --seeds 123 7 2024 --iterations 20

  # Smoke test with 5 iterations, one seed
  python run_t6_multiseed.py --seeds 123 --iterations 5

  # Re-aggregate only (no training), e.g. after editing merge logic
  python run_t6_multiseed.py --aggregate

  # Skip merging the original seed=42 data from exp2_method_comparison.json
  python run_t6_multiseed.py --seeds 123 7 2024 --no-merge-seed42

Output
------
  results/t6_multiseed/seed_<N>/t6_ours.json
  results/t6_multiseed/seed_<N>/t6_agent_modification.json
  results/t6_multiseed/summary.json     <-- aggregated stats
  results/t6_multiseed/summary.md       <-- human-readable table
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

# Ensure unbuffered logging so nohup/tee captures progress in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)

# Imports from the experiment package (must come after sys.path insert)
from run_experiment import ExperimentRunner, load_config, log  # noqa: E402
from baselines import AgentModificationBaseline  # noqa: E402

TASK_NAME = "T6_TwoArmPegInHole"
METHODS = ["ours", "agent_modification"]  # keep narrow; other baselines are ≤6.7% anyway
SUBDIR = "t6_multiseed"


# -------------------------------------------------------------------
# Global seeding — important for real cross-seed variance
# -------------------------------------------------------------------
def seed_everything(seed: int):
    """Seed Python, NumPy, and Torch (CPU + CUDA) globally."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism comes with a throughput cost; we leave benchmark on
        # but turn off TF32 determinism knobs only if explicitly requested.
    except ImportError:
        pass


# -------------------------------------------------------------------
# Per-(seed, method) runner
# -------------------------------------------------------------------
class T6MultiSeedRunner:
    def __init__(self, seeds: List[int], iterations: int, eval_episodes: int,
                 methods: List[str], config_path: Optional[str] = None):
        self.seeds = seeds
        self.iterations = iterations
        self.eval_episodes = eval_episodes
        self.methods = methods
        self.config_path = config_path

        self.base_results_dir = os.path.join(EXP_DIR, "results", SUBDIR)
        os.makedirs(self.base_results_dir, exist_ok=True)

    # -- filesystem helpers --
    def _seed_dir(self, seed: int) -> str:
        d = os.path.join(self.base_results_dir, f"seed_{seed}")
        os.makedirs(d, exist_ok=True)
        return d

    def _result_path(self, seed: int, method: str) -> str:
        return os.path.join(self._seed_dir(seed), f"t6_{method}.json")

    def _is_complete(self, seed: int, method: str) -> bool:
        p = self._result_path(seed, method)
        if not os.path.exists(p):
            return False
        try:
            with open(p) as f:
                data = json.load(f)
            return len(data.get("iterations", [])) >= self.iterations
        except Exception:
            return False

    def _load_partial(self, seed: int, method: str) -> Optional[Dict]:
        p = self._result_path(seed, method)
        if not os.path.exists(p):
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    def _save(self, seed: int, method: str, data: Dict):
        p = self._result_path(seed, method)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, p)

    # -- one-seed driver --
    def run_seed(self, seed: int):
        log(f"========== SEED {seed} ==========")
        seed_everything(seed)

        # Build config + runner with this seed
        config = load_config(self.config_path)
        config["seed"] = seed
        # Construct a lightweight args namespace mirroring run_experiment.main
        args = argparse.Namespace(
            exp="exp2", task=TASK_NAME,
            iterations=self.iterations,
            eval_episodes=self.eval_episodes,
            seed=seed, config=self.config_path,
        )
        runner = ExperimentRunner(config, args)

        # Get the T6 task + env
        task = runner.task_suite.get_task(TASK_NAME)
        if task is None:
            raise RuntimeError(f"Task {TASK_NAME} not found in task suite")

        env = runner.task_suite.make_env(task)
        try:
            for method in self.methods:
                if self._is_complete(seed, method):
                    log(f"  [{method}] already complete for seed={seed}, skipping")
                    continue

                resume = self._load_partial(seed, method)
                if resume and len(resume.get("iterations", [])) > 0:
                    log(f"  [{method}] resuming from iter {len(resume['iterations'])}/{self.iterations}")

                def _on_iter(partial, _seed=seed, _method=method):
                    self._save(_seed, _method, partial)

                t0 = time.time()
                if method == "ours":
                    result = runner._run_evolution(
                        env, task,
                        label=f"t6_multiseed/seed_{seed}/ours",
                        on_iter_done=_on_iter,
                        resume_from=resume,
                    )
                elif method == "agent_modification":
                    am = AgentModificationBaseline(env, seed=seed)
                    result = runner._run_baseline(
                        am, self.iterations, self.eval_episodes,
                        "agent_modification",
                        on_iter_done=_on_iter,
                        resume_from=resume,
                    )
                else:
                    raise ValueError(f"unsupported method: {method}")

                # Attach seed metadata then save final
                result["_meta"] = {
                    "seed": seed, "method": method, "task": TASK_NAME,
                    "iterations": self.iterations,
                    "eval_episodes": self.eval_episodes,
                    "wall_seconds": time.time() - t0,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }
                self._save(seed, method, result)
                log(f"  [{method}] seed={seed} done in {result['_meta']['wall_seconds']:.0f}s")
        finally:
            try:
                env.close()
            except Exception:
                pass

    # -- main loop --
    def run_all(self):
        for seed in self.seeds:
            self.run_seed(seed)
            # Re-aggregate after each completed seed so partial progress is usable
            try:
                aggregate_and_write(self.base_results_dir, extra_seed42=True)
            except Exception as e:
                log(f"  [warn] aggregation failed after seed={seed}: {e}")


# -------------------------------------------------------------------
# Aggregation — mean/std/CI across seeds, Welch's t-test Ours vs AM
# -------------------------------------------------------------------
def _extract_summary_stats(result: Dict) -> Dict:
    """Pull peak success, peak reward, final reward, final variance from a run."""
    metrics_list = result.get("metrics", [])
    if not metrics_list:
        return {}
    successes = [m.get("success_rate", 0.0) for m in metrics_list]
    rewards = [m.get("reward_mean", m.get("reward", 0.0)) for m in metrics_list]
    final = metrics_list[-1]
    return {
        "peak_success": float(max(successes)),
        "peak_reward": float(max(rewards)),
        "final_success": float(final.get("success_rate", 0.0)),
        "final_reward": float(final.get("reward_mean", final.get("reward", 0.0))),
        "final_reward_var": float(final.get("reward_var", 0.0)),
        "n_iters": len(metrics_list),
    }


def _load_seed42_from_main(exp_dir: str) -> Dict[str, Dict]:
    """Read Ours / AM data for T6 from the main exp2_method_comparison.json
    so seed=42 can be merged with the new seeds without rerunning."""
    path = os.path.join(exp_dir, "results", "exp2_method_comparison.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {}
    t6 = data.get(TASK_NAME, {})
    out = {}
    for method in METHODS:
        if method in t6 and t6[method].get("metrics"):
            out[method] = t6[method]
    return out


def aggregate_and_write(base_dir: str, extra_seed42: bool = True) -> Dict:
    """Walk seed_*/t6_<method>.json, compute per-seed stats, then aggregate."""
    per_seed: Dict[str, Dict[int, Dict]] = {m: {} for m in METHODS}

    # Collect new multi-seed runs
    for entry in sorted(os.listdir(base_dir)):
        if not entry.startswith("seed_"):
            continue
        try:
            s = int(entry.split("_", 1)[1])
        except ValueError:
            continue
        for method in METHODS:
            fp = os.path.join(base_dir, entry, f"t6_{method}.json")
            if not os.path.exists(fp):
                continue
            try:
                with open(fp) as f:
                    data = json.load(f)
            except Exception:
                continue
            stats = _extract_summary_stats(data)
            if stats:
                per_seed[method][s] = stats

    # Optionally merge in seed=42 from the original exp2 run
    if extra_seed42:
        exp_dir = os.path.dirname(os.path.dirname(base_dir))  # .../Experiments_V2
        seed42 = _load_seed42_from_main(exp_dir)
        for method, data in seed42.items():
            stats = _extract_summary_stats(data)
            if stats and 42 not in per_seed[method]:
                per_seed[method][42] = stats

    summary = {
        "task": TASK_NAME,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "per_seed": {m: per_seed[m] for m in METHODS},
        "aggregate": {},
    }

    # Compute mean / std / 95% CI per method per metric
    METRICS_OF_INTEREST = [
        "peak_success", "peak_reward",
        "final_success", "final_reward", "final_reward_var",
    ]
    for method in METHODS:
        vals_by_metric = {k: [] for k in METRICS_OF_INTEREST}
        for s, stats in per_seed[method].items():
            for k in METRICS_OF_INTEREST:
                if k in stats:
                    vals_by_metric[k].append(stats[k])
        agg = {"n_seeds": len(per_seed[method])}
        for k, vals in vals_by_metric.items():
            if not vals:
                agg[k] = None
                continue
            arr = np.asarray(vals, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            # 95% CI using t-distribution when n>=2, else just ±0
            if len(arr) >= 2:
                from scipy import stats as _st
                tcrit = _st.t.ppf(0.975, df=len(arr) - 1)
                ci = float(tcrit * std / np.sqrt(len(arr)))
            else:
                ci = 0.0
            agg[k] = {"mean": mean, "std": std, "ci95": ci,
                      "n": len(arr), "values": [float(v) for v in arr]}
        summary["aggregate"][method] = agg

    # Welch's t-test on peak_success: Ours vs AM
    ours_vals = [s["peak_success"] for s in per_seed["ours"].values()]
    am_vals = [s["peak_success"] for s in per_seed["agent_modification"].values()]
    if len(ours_vals) >= 2 and len(am_vals) >= 2:
        from scipy import stats as _st
        t_stat, p_val = _st.ttest_ind(ours_vals, am_vals, equal_var=False)
        summary["ttest_peak_success_ours_vs_am"] = {
            "t_stat": float(t_stat), "p_value": float(p_val),
            "n_ours": len(ours_vals), "n_am": len(am_vals),
        }

    # Write JSON + Markdown
    out_json = os.path.join(base_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    out_md = os.path.join(base_dir, "summary.md")
    with open(out_md, "w") as f:
        f.write(_format_summary_md(summary))

    log(f"Aggregate written: {out_json}")
    log(f"Markdown summary:  {out_md}")
    return summary


def _format_summary_md(summary: Dict) -> str:
    lines = []
    lines.append(f"# T6_TwoArmPegInHole — Multi-Seed Summary\n")
    lines.append(f"Task: `{summary['task']}`  \nGenerated: {summary['generated_at']}\n")

    def fmt(stat, pct=False):
        if stat is None:
            return "—"
        mean = stat["mean"]
        std = stat["std"]
        ci = stat.get("ci95", 0.0)
        if pct:
            return f"{mean:.1f}% ± {std:.1f} (95% CI ±{ci:.1f})"
        return f"{mean:.2f} ± {std:.2f} (95% CI ±{ci:.2f})"

    agg = summary.get("aggregate", {})
    ours = agg.get("ours", {})
    am = agg.get("agent_modification", {})

    lines.append("## Headline Comparison\n")
    lines.append("| Metric | Ours (Capability Evolution) | Agent Modification |")
    lines.append("|---|---|---|")
    lines.append(f"| Seeds (n) | {ours.get('n_seeds', 0)} | {am.get('n_seeds', 0)} |")
    lines.append(f"| Peak success | {fmt(ours.get('peak_success'), pct=True)} | {fmt(am.get('peak_success'), pct=True)} |")
    lines.append(f"| Peak reward | {fmt(ours.get('peak_reward'))} | {fmt(am.get('peak_reward'))} |")
    lines.append(f"| Final success | {fmt(ours.get('final_success'), pct=True)} | {fmt(am.get('final_success'), pct=True)} |")
    lines.append(f"| Final reward | {fmt(ours.get('final_reward'))} | {fmt(am.get('final_reward'))} |")

    tt = summary.get("ttest_peak_success_ours_vs_am")
    if tt:
        lines.append("")
        lines.append(f"**Welch's t-test (peak success, Ours vs AM)**: "
                     f"t={tt['t_stat']:.3f}, p={tt['p_value']:.4f}  "
                     f"(n_ours={tt['n_ours']}, n_am={tt['n_am']})")

    lines.append("\n## Per-Seed Detail\n")
    lines.append("| Method | Seed | Peak Success | Peak Reward | Final Success | Final Reward |")
    lines.append("|---|---|---|---|---|---|")
    for method in METHODS:
        for seed, stats in sorted(summary["per_seed"].get(method, {}).items()):
            lines.append(
                f"| {method} | {seed} | {stats.get('peak_success', 0):.1f}% | "
                f"{stats.get('peak_reward', 0):.2f} | "
                f"{stats.get('final_success', 0):.1f}% | "
                f"{stats.get('final_reward', 0):.2f} |"
            )

    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed verification of T6 (Ours vs Agent Modification)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 7, 2024],
                        help="Seeds to run. Default: 123 7 2024.")
    parser.add_argument("--methods", type=str, nargs="+", default=METHODS,
                        choices=METHODS,
                        help="Methods to include. Default: ours agent_modification.")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Iterations per method per seed (default 20, matching main exp).")
    parser.add_argument("--eval-episodes", type=int, default=30,
                        help="Eval episodes per iteration (default 30).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (defaults to configs/default.yaml).")
    parser.add_argument("--aggregate", action="store_true",
                        help="Skip training, only recompute summary from existing JSONs.")
    parser.add_argument("--no-merge-seed42", action="store_true",
                        help="Do not merge seed=42 data from exp2_method_comparison.json.")
    args = parser.parse_args()

    base_dir = os.path.join(EXP_DIR, "results", SUBDIR)
    os.makedirs(base_dir, exist_ok=True)

    if args.aggregate:
        aggregate_and_write(base_dir, extra_seed42=not args.no_merge_seed42)
        return

    log(f"T6 multi-seed run — seeds={args.seeds}, methods={args.methods}, "
        f"iters={args.iterations}, eval_eps={args.eval_episodes}")
    log(f"Results dir: {base_dir}")

    runner = T6MultiSeedRunner(
        seeds=args.seeds,
        iterations=args.iterations,
        eval_episodes=args.eval_episodes,
        methods=args.methods,
        config_path=args.config,
    )
    runner.run_all()

    # Final aggregation
    summary = aggregate_and_write(base_dir, extra_seed42=not args.no_merge_seed42)

    # Print headline to stdout
    print("\n" + "=" * 60)
    print("T6 MULTI-SEED FINAL SUMMARY")
    print("=" * 60)
    agg = summary.get("aggregate", {})
    for method in METHODS:
        a = agg.get(method, {})
        ps = a.get("peak_success")
        if ps:
            print(f"  {method:22s}  peak success = "
                  f"{ps['mean']:5.1f}% ± {ps['std']:4.1f}  (n={ps['n']})")
    tt = summary.get("ttest_peak_success_ours_vs_am")
    if tt:
        print(f"\n  Welch's t-test Ours vs AM: t={tt['t_stat']:.3f}, p={tt['p_value']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
