"""
T6 AM+phase (Hierarchical-Mutable) multi-seed driver.

Runs ONLY the matched baseline AM+phase on T6_TwoArmPegInHole for seeds
{42, 7, 123, 2024}, matched to the published Ours/AM sweep:
  * 20 evolution iterations
  * 30 eval episodes / iteration
  * configs/default.yaml unchanged (50k SAC steps/iter, n_envs=12, 4 ECMs)

Ours and AM data already exist (results/t6_multiseed/ + exp2_method_comparison.json);
we do NOT rerun them. This script writes only:
  results/t6_amphase/seed_<N>/t6_amphase.json
  results/t6_amphase/summary_amphase.json / .md

Usage
-----
  python run_t6_amphase.py --seeds 42 7 123 2024 --iterations 20
  python run_t6_amphase.py --seeds 42 --iterations 3   # smoke test
  python run_t6_amphase.py --aggregate                 # rebuild summary only
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

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)

from run_experiment import load_config, log  # noqa: E402
from amphase_runner import ExperimentRunnerMutable  # noqa: E402

TASK_NAME = "T6_TwoArmPegInHole"
METHOD = "amphase"
SUBDIR = "t6_amphase"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class T6AMPhaseRunner:
    def __init__(self, seeds: List[int], iterations: int, eval_episodes: int,
                 config_path: Optional[str] = None):
        self.seeds = seeds
        self.iterations = iterations
        self.eval_episodes = eval_episodes
        self.config_path = config_path
        self.base_results_dir = os.path.join(EXP_DIR, "results", SUBDIR)
        os.makedirs(self.base_results_dir, exist_ok=True)

    def _seed_dir(self, seed: int) -> str:
        d = os.path.join(self.base_results_dir, f"seed_{seed}")
        os.makedirs(d, exist_ok=True)
        return d

    def _result_path(self, seed: int) -> str:
        return os.path.join(self._seed_dir(seed), f"t6_{METHOD}.json")

    def _is_complete(self, seed: int) -> bool:
        p = self._result_path(seed)
        if not os.path.exists(p):
            return False
        try:
            with open(p) as f:
                data = json.load(f)
            return len(data.get("iterations", [])) >= self.iterations
        except Exception:
            return False

    def _load_partial(self, seed: int) -> Optional[Dict]:
        p = self._result_path(seed)
        if not os.path.exists(p):
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None

    def _save(self, seed: int, data: Dict):
        p = self._result_path(seed)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, p)

    def run_seed(self, seed: int):
        log(f"========== AM+phase SEED {seed} ==========")
        seed_everything(seed)

        config = load_config(self.config_path)
        config["seed"] = seed
        args = argparse.Namespace(
            exp="exp2", task=TASK_NAME,
            iterations=self.iterations,
            eval_episodes=self.eval_episodes,
            seed=seed, config=self.config_path,
        )
        runner = ExperimentRunnerMutable(config, args)

        task = runner.task_suite.get_task(TASK_NAME)
        if task is None:
            raise RuntimeError(f"Task {TASK_NAME} not found in task suite")
        env = runner.task_suite.make_env(task)
        try:
            if self._is_complete(seed):
                log(f"  [amphase] already complete for seed={seed}, skipping")
                return
            resume = self._load_partial(seed)
            if resume and len(resume.get("iterations", [])) > 0:
                log(f"  [amphase] resuming from iter "
                    f"{len(resume['iterations'])}/{self.iterations}")

            def _on_iter(partial, _seed=seed):
                self._save(_seed, partial)

            t0 = time.time()
            result = runner._run_evolution_mutable(
                env, task,
                label=f"t6_amphase/seed_{seed}/amphase",
                on_iter_done=_on_iter,
                resume_from=resume,
            )
            result["_meta"] = {
                "seed": seed, "method": METHOD, "task": TASK_NAME,
                "iterations": self.iterations,
                "eval_episodes": self.eval_episodes,
                "wall_seconds": time.time() - t0,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._save(seed, result)
            log(f"  [amphase] seed={seed} done in "
                f"{result['_meta']['wall_seconds']:.0f}s")
        finally:
            try:
                env.close()
            except Exception:
                pass

    def run_all(self):
        for seed in self.seeds:
            self.run_seed(seed)
            try:
                aggregate_and_write(self.base_results_dir)
            except Exception as e:
                log(f"  [warn] aggregation failed after seed={seed}: {e}")


def _extract_summary_stats(result: Dict) -> Dict:
    metrics_list = result.get("metrics", [])
    if not metrics_list:
        return {}
    successes = [m.get("success_rate", 0.0) for m in metrics_list]
    rewards = [m.get("avg_reward", m.get("reward_mean", 0.0)) for m in metrics_list]
    drifts = [m.get("policy_drift", 0.0) for m in metrics_list]
    final = metrics_list[-1]
    return {
        "peak_success": float(max(successes)),
        "peak_reward": float(max(rewards)) if rewards else 0.0,
        "final_success": float(final.get("success_rate", 0.0)),
        "final_reward": float(final.get("avg_reward", 0.0)),
        "max_policy_drift": float(max(drifts)) if drifts else 0.0,
        "final_policy_drift": float(final.get("policy_drift", 0.0)),
        "n_iters": len(metrics_list),
    }


def aggregate_and_write(base_dir: str) -> Dict:
    per_seed: Dict[int, Dict] = {}
    for entry in sorted(os.listdir(base_dir)):
        if not entry.startswith("seed_"):
            continue
        try:
            s = int(entry.split("_", 1)[1])
        except ValueError:
            continue
        fp = os.path.join(base_dir, entry, f"t6_{METHOD}.json")
        if not os.path.exists(fp):
            continue
        try:
            with open(fp) as f:
                data = json.load(f)
        except Exception:
            continue
        stats = _extract_summary_stats(data)
        if stats:
            per_seed[s] = stats

    summary = {
        "task": TASK_NAME,
        "method": METHOD,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "per_seed": per_seed,
        "aggregate": {},
    }

    peaks = [v["peak_success"] for v in per_seed.values()]
    if peaks:
        arr = np.asarray(peaks, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        ci = 0.0
        if len(arr) >= 2:
            from scipy import stats as _st
            tcrit = _st.t.ppf(0.975, df=len(arr) - 1)
            ci = float(tcrit * std / np.sqrt(len(arr)))
        summary["aggregate"]["peak_success"] = {
            "mean": mean, "std": std, "ci95": ci, "n": len(arr),
            "values": [float(v) for v in arr],
        }
    # sanity: report drift so a reviewer can confirm the agent actually moved
    drifts = [v["max_policy_drift"] for v in per_seed.values()]
    if drifts:
        summary["aggregate"]["max_policy_drift"] = {
            "mean": float(np.mean(drifts)),
            "values": [float(d) for d in drifts],
        }

    out_json = os.path.join(base_dir, "summary_amphase.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    lines = [f"# T6 AM+phase (Hierarchical-Mutable) — Summary\n",
             f"Task: `{TASK_NAME}`  Method: `{METHOD}`  "
             f"Generated: {summary['generated_at']}\n",
             "| Seed | Peak Success | Final Success | Max Δπ | Final Δπ | n_iter |",
             "|---|---|---|---|---|---|"]
    for s, st in sorted(per_seed.items()):
        lines.append(
            f"| {s} | {st['peak_success']:.1f}% | {st['final_success']:.1f}% | "
            f"{st['max_policy_drift']:.4f} | {st['final_policy_drift']:.4f} | "
            f"{st['n_iters']} |"
        )
    agg = summary["aggregate"].get("peak_success")
    if agg:
        lines.append("")
        lines.append(f"**AM+phase peak success: {agg['mean']:.1f}% ± {agg['std']:.1f} "
                     f"(95% CI ±{agg['ci95']:.1f}, n={agg['n']})**")
    out_md = os.path.join(base_dir, "summary_amphase.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    log(f"Aggregate written: {out_json}")
    log(f"Markdown summary:  {out_md}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="T6 AM+phase (Hierarchical-Mutable) matched baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 2024])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.join(EXP_DIR, "results", SUBDIR)
    os.makedirs(base_dir, exist_ok=True)

    if args.aggregate:
        aggregate_and_write(base_dir)
        return

    log(f"T6 AM+phase run — seeds={args.seeds}, iters={args.iterations}, "
        f"eval_eps={args.eval_episodes}")
    log(f"Results dir: {base_dir}")

    runner = T6AMPhaseRunner(
        seeds=args.seeds,
        iterations=args.iterations,
        eval_episodes=args.eval_episodes,
        config_path=args.config,
    )
    runner.run_all()
    summary = aggregate_and_write(base_dir)

    print("\n" + "=" * 60)
    print("T6 AM+phase FINAL SUMMARY")
    print("=" * 60)
    agg = summary.get("aggregate", {}).get("peak_success")
    if agg:
        print(f"  AM+phase  peak success = {agg['mean']:5.1f}% ± {agg['std']:4.1f}  "
              f"(n={agg['n']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
