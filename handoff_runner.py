"""State-conditioned handoff intervention on T6 (evaluation-only).

The paper's rotating configuration dispatches four whole-task modules by a
fixed clock: `run_experiment.ExperimentRunner._get_phase_schedule` divides the
episode horizon into equal windows and cycles modules through them with no
feedback. In the vocabulary of the options framework that is an architecture
whose termination function has been replaced by a timer, and the paper's
retention analysis argues, observationally, that this is what costs the
rotating arms the goal states they reach.

This driver runs the intervention that isolates it. Two arms are evaluated on
the SAME trained modules under the SAME protocol, differing in one rule. They
are paired at the level of seed and checkpoint, not episode by episode:
robosuite's placement sampler is unseeded (tests/test_seed_replay.py), so each
arm draws its own initial states.

  clock  the released schedule: control passes at every window boundary.
  hold   identical until the per-step success geometry is first satisfied;
         from that step on, whichever module currently holds control keeps it
         for the remainder of the episode.

The success geometry is the ladder's P3 condition, taken verbatim from
`mechanism_runner` so the retention numbers are comparable with the paper's
table: T_MIN <= t <= T_MAX and d < D_SUCCESS and cos > COS_ALIGN, read from the
environment's own `_compute_orientation()` after each step.

No training happens here. Modules are loaded from the checkpoints the
mechanism runs already produced, so the two arms differ in the handoff rule
and in their independent draws of initial states, nothing else.

Run:
  MUJOCO_GL=egl python handoff_runner.py --method ours --seeds 7 42 123 2024
  python handoff_runner.py --aggregate
Outputs: results/t6_handoff/<method>/seed_<s>/handoff.json and
         results/t6_handoff/summary_handoff.{json,md}
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)

from mechanism_runner import (  # noqa: E402
    COS_ALIGN, D_APPROACH, D_SUCCESS, LADDER_CONSTANTS, T_MAX, T_MIN,
    TASK_NAME, get_robosuite_env, seed_everything,
)
from run_experiment import ExperimentRunner, load_config, log  # noqa: E402
from agent.ecm import ECM, ECMRegistry, ECMDescriptor  # noqa: E402
from agent.persistent_agent import PersistentAgent  # noqa: E402

RESULTS_DIR = os.path.join(EXP_DIR, "results", "t6_handoff")
CKPT_TMPL = os.path.join(EXP_DIR, "results", "checkpoints",
                         "t6_mechanism_{method}_seed_{seed}")
ARMS = ("clock", "hold")


def paired_eval(env, schedule, hold_on_success: bool, num_episodes: int):
    """Instrumented deployment eval with an optional state-conditioned hold.

    Mirrors mechanism_runner.instrumented_eval's accounting exactly (final-step
    success, per-episode reward sum, the P0-P4 ladder) and adds the hold rule
    plus counters describing when it engaged.
    """
    rs = get_robosuite_env(env)
    successes = 0
    total_reward = 0.0
    ep_rewards: List[float] = []
    ep_max_phase: List[int] = []
    holds_engaged = 0
    hold_steps: List[int] = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        env_info: Dict = {}
        p1 = p2 = p3 = False
        held_ecm = None          # module frozen into control, if any
        hold_step: Optional[int] = None

        while not done:
            if held_ecm is not None:
                current = held_ecm
            else:
                current = schedule[-1][2]
                for start, end, ecm in schedule:
                    if start <= step < end:
                        current = ecm
                        break
            action, _ = current.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1

            t, d, cos = rs._compute_orientation()
            t, d, cos = float(t), float(d), float(cos)
            if d < D_APPROACH:
                p1 = True
                if cos > COS_ALIGN:
                    p2 = True
            geometry = (T_MIN <= t <= T_MAX) and d < D_SUCCESS and cos > COS_ALIGN
            if geometry:
                p3 = True
                # Engage the hold the first time the geometry is reached: the
                # module in control at that moment keeps it to episode end.
                if hold_on_success and held_ecm is None:
                    held_ecm = current
                    hold_step = step
                    holds_engaged += 1

        ep_success = bool(env_info.get("success", False))
        if ep_success:
            successes += 1
        phase = 4 if ep_success else (3 if p3 else (2 if p2 else (1 if p1 else 0)))
        ep_max_phase.append(phase)
        ep_rewards.append(ep_reward)
        total_reward += ep_reward
        if hold_step is not None:
            hold_steps.append(hold_step)

    hist = {f"P{p}": int(sum(1 for m in ep_max_phase if m == p))
            for p in range(5)}
    reached = hist["P3"] + hist["P4"]
    return {
        "success_rate": successes / num_episodes * 100.0,
        "avg_reward": total_reward / num_episodes,
        "reward_variance": float(np.var(ep_rewards)),
        "phase_histogram": hist,
        "episode_max_phase": ep_max_phase,
        "reached": reached,
        "retention": (hist["P4"] / reached) if reached else None,
        "holds_engaged": holds_engaged,
        "mean_hold_step": (float(np.mean(hold_steps)) if hold_steps else None),
    }


class HandoffRunner(ExperimentRunner):
    """Builds the registry/agent/env the evaluation needs, without training."""

    def build(self, seed: int, config_path):
        seed_everything(seed)
        config = load_config(config_path)
        config["seed"] = seed
        task = self.task_suite.get_task(TASK_NAME)
        env = self.task_suite.make_env(task)
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
        return task, env, registry, agent


def load_modules(registry: ECMRegistry, method: str, seed: int) -> str:
    path = os.path.join(CKPT_TMPL.format(method=method, seed=seed),
                        "ecm_weights.pt")
    if not os.path.exists(path):
        raise RuntimeError(f"checkpoint not found: {path}")
    states = torch.load(path, weights_only=False)
    for ecm in registry.active_ecms():
        name = ecm.descriptor.name
        if name not in states:
            raise RuntimeError(f"checkpoint {path} missing {name}")
        ecm.network.load_state_dict(states[name]["state_dict"])
    return path


def run_seed(method: str, seed: int, episodes: int, config_path):
    ns = argparse.Namespace(exp="handoff", task=TASK_NAME, iterations=0,
                            eval_episodes=episodes, seed=seed,
                            config=config_path)
    config = load_config(config_path)
    config["seed"] = seed
    runner = HandoffRunner(config, ns)
    task, env, registry, agent = runner.build(seed, config_path)
    ckpt = load_modules(registry, method, seed)
    horizon = env._env.horizon if hasattr(env, "_env") else 500
    schedule = runner._get_phase_schedule(registry, task, horizon)
    log(f"{method} seed {seed}: {len(schedule)} windows, checkpoint {ckpt}")

    out: Dict = {"method": method, "seed": seed, "episodes": episodes,
                 "checkpoint": os.path.relpath(ckpt, EXP_DIR),
                 "ladder_constants": LADDER_CONSTANTS,
                 "window_bounds": [[int(a), int(b)] for a, b, _ in schedule],
                 "arms": {}}
    for arm in ARMS:
        # Re-seed identically before each arm. This pairs the arms at the
        # level of (seed, checkpoint), NOT episode by episode: robosuite's
        # object placement sampler is unseeded (tests/test_seed_replay.py),
        # so each arm draws its own initial states. The comparison is between
        # two evaluations of the same modules under the same protocol,
        # differing only in the handoff rule.
        seed_everything(seed)
        res = paired_eval(env, schedule, hold_on_success=(arm == "hold"),
                          num_episodes=episodes)
        out["arms"][arm] = res
        log(f"  {arm:5s}: success={res['success_rate']:.1f}% "
            f"reached={res['reached']} "
            f"retention={res['retention']} "
            f"holds={res['holds_engaged']} "
            f"mean_hold_step={res['mean_hold_step']}")
    env.close()
    d = os.path.join(RESULTS_DIR, method, f"seed_{seed}")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "handoff.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, p)
    return out


def aggregate():
    rows = []
    for method in sorted(os.listdir(RESULTS_DIR)):
        md = os.path.join(RESULTS_DIR, method)
        if not os.path.isdir(md):
            continue
        for sd in sorted(os.listdir(md)):
            p = os.path.join(md, sd, "handoff.json")
            if os.path.exists(p):
                with open(p) as fh:
                    rows.append(json.load(fh))
    summary = {"experiment": "state_conditioned_handoff", "per_seed": [],
               "paired": {}}
    lines = ["# State-conditioned handoff intervention (T6, eval-only)", "",
             "| method | seed | arm | reached | retention | success % | holds |",
             "|---|---|---|---|---|---|---|"]
    by_method: Dict[str, List] = {}
    for r in rows:
        for arm in ARMS:
            a = r["arms"][arm]
            ret = "n/a" if a["retention"] is None else f"{a['retention']:.3f}"
            lines.append(f"| {r['method']} | {r['seed']} | {arm} | "
                         f"{a['reached']} | {ret} | "
                         f"{a['success_rate']:.1f} | {a['holds_engaged']} |")
        by_method.setdefault(r["method"], []).append(r)
        summary["per_seed"].append({
            "method": r["method"], "seed": r["seed"],
            "clock": {k: r["arms"]["clock"][k]
                      for k in ("reached", "retention", "success_rate")},
            "hold": {k: r["arms"]["hold"][k]
                     for k in ("reached", "retention", "success_rate",
                               "holds_engaged", "mean_hold_step")},
        })
    for method, rs in by_method.items():
        pairs = [(r["arms"]["clock"]["retention"], r["arms"]["hold"]["retention"])
                 for r in rs
                 if r["arms"]["clock"]["retention"] is not None
                 and r["arms"]["hold"]["retention"] is not None]
        n = len(pairs)
        gains = [h - c for c, h in pairs]
        n_pos = sum(1 for g in gains if g > 0)
        # Exact two-sided sign test; with n seeds the smallest attainable
        # two-sided p is 2 / 2**n, which we report so the floor is explicit.
        from math import comb
        k = max(n_pos, n - n_pos)
        p_two = min(1.0, 2.0 * sum(comb(n, i) for i in range(k, n + 1))
                    / (2 ** n)) if n else None
        summary["paired"][method] = {
            "n_seeds": n,
            "clock_retention": [c for c, _ in pairs],
            "hold_retention": [h for _, h in pairs],
            "gain": gains,
            "n_positive": n_pos,
            "sign_test_p_two_sided": p_two,
            "smallest_attainable_p": (2.0 / (2 ** n)) if n else None,
        }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "summary_handoff.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(RESULTS_DIR, "summary_handoff.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"aggregate: {len(rows)} runs -> summary_handoff.json/.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("ours", "amphase"), default="ours")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123, 2024])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
        return
    for seed in a.seeds:
        run_seed(a.method, seed, a.episodes, a.config)


if __name__ == "__main__":
    main()
