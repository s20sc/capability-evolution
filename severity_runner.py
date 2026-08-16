"""Detector operating characteristic on frozen incumbents (eval-only).

Characterizes the dip detector's sensitivity/specificity as a function of
injection severity, without any training. For each (seed, sigma, replicate):

  1. rebuild the T6 environment, ECM registry, and frozen agent exactly as
     the lifecycle experiment does;
  2. load the rollback arm's final module weights from its saved checkpoint
     (results/checkpoints/t6_lifecycle_gating_off_rollback_seed_<s>/);
  3. perturb every module in place with the lifecycle experiment's own
     injection rule (per-tensor Gaussian, std = sigma * std(tensor));
  4. run the standard 30-episode deployment evaluation;
  5. score the detector rule the paper reports: trigger iff the deployed
     mean reward falls more than `--trigger` (default 15%) below that run's
     baseline, where baseline = median of the final three non-injected
     deployed rewards recorded in the run's own result file.

sigma = 0.0 rows are clean re-evaluations of the unperturbed incumbent and
estimate the false-trigger rate (1 - specificity) under evaluation noise
alone. Results are labelled as a characterization of the detector on frozen
incumbents; they are not a re-run of the in-loop lifecycle experiment.

Run (full sweep):
  MUJOCO_GL=egl python severity_runner.py --seeds 7 42 123 2024 \
      --sigmas 0.0 0.05 0.1 0.15 0.2 0.35 0.5 --reps 3
Outputs: results/t6_severity/seed_<s>/sigma_<v>_rep<k>.json plus
         results/t6_severity/summary_severity.{json,md} via --aggregate.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)

from lifecycle_runner import (  # noqa: E402
    LifecycleExperimentRunner, load_config, log, seed_everything, TASK_NAME,
)
from agent.ecm import ECM, ECMRegistry, ECMDescriptor  # noqa: E402
from agent.persistent_agent import PersistentAgent  # noqa: E402

ARM = "gating_off_rollback"
RESULTS_DIR = os.path.join(EXP_DIR, "results", "t6_severity")
CKPT_TMPL = os.path.join(EXP_DIR, "results", "checkpoints",
                         "t6_lifecycle_{arm}_seed_{seed}")
LIFECYCLE_RESULT_TMPL = os.path.join(EXP_DIR, "results", "t6_lifecycle",
                                     "{arm}", "seed_{seed}",
                                     "t6_lifecycle.json")


def _baseline_from_run(seed: int) -> float:
    """Median of the final three non-injected deployed rewards of the
    rollback arm's own run: the same statistic the in-loop detector keeps."""
    path = LIFECYCLE_RESULT_TMPL.format(arm=ARM, seed=seed)
    with open(path) as fh:
        data = json.load(fh)
    clean = [m["avg_reward"] for m in data["metrics"]
             if not m.get("injection_flag")]
    if len(clean) < 3:
        raise RuntimeError(f"seed {seed}: fewer than 3 clean iterations")
    return float(np.median(clean[-3:]))


def _build(seed: int, config_path):
    """Environment + registry + frozen agent, mirroring the lifecycle run."""
    seed_everything(seed)
    config = load_config(config_path)
    config["seed"] = seed
    ns = argparse.Namespace(exp="severity", task=TASK_NAME, iterations=0,
                            eval_episodes=30, seed=seed, config=config_path)
    runner = LifecycleExperimentRunner(config, ns)
    task = runner.task_suite.get_task(TASK_NAME)
    env = runner.task_suite.make_env(task)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    registry = ECMRegistry()
    for phase in ["reach", "grasp", "lift", "place"]:
        desc = ECMDescriptor(name=f"ecm_{phase}",
                             description=f"ECM for {phase}", phase=phase,
                             input_dim=obs_dim, output_dim=act_dim)
        registry.register(ECM(desc,
                              hidden_dims=config["ecm"]["hidden_dims"],
                              rollback_threshold=config["ecm"]
                              ["rollback_threshold"]))
    agent = PersistentAgent(obs_dim, registry, seed=config["seed"])
    return runner, task, env, registry, agent


def _load_incumbent(registry: ECMRegistry, seed: int) -> None:
    ckpt = os.path.join(CKPT_TMPL.format(arm=ARM, seed=seed),
                        "ecm_weights.pt")
    states = torch.load(ckpt, weights_only=False)
    for ecm in registry.active_ecms():
        name = ecm.descriptor.name
        if name not in states:
            raise RuntimeError(f"checkpoint missing {name}")
        ecm.network.load_state_dict(states[name]["state_dict"])


def run_cell(runner, task, env, registry, agent, seed: int, sigma: float,
             rep: int, episodes: int, baseline: float, trigger: float):
    clean_snap = runner._snapshot_ecm_states(registry)
    if sigma > 0.0:
        rng = np.random.default_rng(seed * 1000 + rep)
        runner._perturb_ecm_states(registry, sigma, rng)
    metrics = runner._evaluate_agent(agent, env, episodes, task=task)
    runner._restore_ecm_states(registry, clean_snap)
    drop_pct = 100.0 * (baseline - metrics["avg_reward"]) / baseline
    return {
        "seed": seed, "sigma": sigma, "rep": rep, "episodes": episodes,
        "baseline_median_last3_clean": baseline,
        "mean_reward": metrics["avg_reward"],
        "success_rate": metrics["success_rate"],
        "reward_variance": metrics["reward_variance"],
        "drop_pct": drop_pct,
        "trigger": bool(drop_pct > trigger),
        "trigger_threshold_pct": trigger,
    }


def aggregate():
    rows = []
    for sd in sorted(os.listdir(RESULTS_DIR)):
        if not sd.startswith("seed_"):
            continue
        for fn in sorted(os.listdir(os.path.join(RESULTS_DIR, sd))):
            if fn.endswith(".json"):
                with open(os.path.join(RESULTS_DIR, sd, fn)) as fh:
                    rows.append(json.load(fh))
    by = {}
    for r in rows:
        by.setdefault((r["seed"], r["sigma"]), []).append(r)
    summary = {"experiment": "detector_severity_characterization",
               "arm": ARM, "cells": []}
    lines = ["# Detector severity characterization (frozen incumbents)",
             "", "| seed | sigma | mean drop % | trigger rate |",
             "|---|---|---|---|"]
    for (seed, sigma), rs in sorted(by.items()):
        drops = [r["drop_pct"] for r in rs]
        trig = sum(r["trigger"] for r in rs) / len(rs)
        summary["cells"].append({
            "seed": seed, "sigma": sigma, "n_reps": len(rs),
            "mean_drop_pct": float(np.mean(drops)),
            "std_drop_pct": float(np.std(drops)),
            "trigger_rate": trig,
        })
        lines.append(f"| {seed} | {sigma} | "
                     f"{np.mean(drops):.1f}±{np.std(drops):.1f} | "
                     f"{trig:.2f} |")
    with open(os.path.join(RESULTS_DIR, "summary_severity.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(RESULTS_DIR, "summary_severity.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"aggregate: {len(rows)} cells -> summary_severity.json/.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[7, 42, 123, 2024])
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.35, 0.5])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--trigger", type=float, default=15.0)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
        return
    for seed in a.seeds:
        baseline = _baseline_from_run(seed)
        runner, task, env, registry, agent = _build(seed, a.config)
        _load_incumbent(registry, seed)
        out_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)
        log(f"seed {seed}: baseline={baseline:.1f}")
        for sigma in a.sigmas:
            for rep in range(a.reps):
                out = os.path.join(out_dir,
                                   f"sigma_{sigma:g}_rep{rep}.json")
                if os.path.exists(out):
                    log(f"  skip existing {os.path.basename(out)}")
                    continue
                cell = run_cell(runner, task, env, registry, agent, seed,
                                sigma, rep, a.episodes, baseline, a.trigger)
                tmp = out + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(cell, fh, indent=1)
                os.replace(tmp, out)
                log(f"  sigma={sigma:g} rep={rep}: "
                    f"reward={cell['mean_reward']:.1f} "
                    f"drop={cell['drop_pct']:+.1f}% "
                    f"trigger={cell['trigger']}")
        env.close()


if __name__ == "__main__":
    main()
