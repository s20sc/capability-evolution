#!/usr/bin/env python3
"""AM+nophase: the compute-matched, decomposition-free counterpart of AM+phase.

The paper's decomposition claim previously compared AM+phase against monolithic
Agent Modification. That contrast is confounded: monolithic AM uses PPO without
imitation demonstrations, while AM+phase uses SAC with demonstrations AND a phase
decomposition, so three factors move together. This arm isolates the one factor:

                     learner  demos  decomposition      planner
  AM+phase           SAC      yes    4 phase ECMs       unfrozen
  AM+nophase (here)  SAC      yes    1 whole-task ECM   unfrozen

Implementation is nophase_impl.ExperimentRunnerNoPhase, a derived copy of
amphase_runner.ExperimentRunnerMutable differing in exactly two places (the ECM
registry loop and the class name); `diff amphase_runner.py nophase_impl.py`
shows the whole change and should stay at those lines.

TOTAL per-iteration compute is matched, not per-module compute: AM+phase trains
four ECMs at `total_timesteps_per_iter` each, so the single ECM here receives
four times that. Matching per-module steps would give AM+phase a 4x advantage.

PRIMARY ENDPOINT is the phase ladder, not peak binary success: peak success on
T6 swings by a factor of five across random streams at this budget, whereas the
ladder ordering is stable across seeds. Instrumentation is reused from
mechanism_runner verbatim so this arm sits alongside the other four.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiment import ExperimentRunner, load_config, log      # noqa: E402
import mechanism_runner as M                                       # noqa: E402
from nophase_impl import ExperimentRunnerNoPhase                   # noqa: E402

METHOD = "nophase"
N_PHASE_ECMS = 4


class NoPhaseMechRunner(ExperimentRunnerNoPhase):
    """Eval instrumented exactly as mechanism_runner does for AM+phase."""

    def __init__(self, config, args, probe):
        super().__init__(config, args)
        self._mech_probe = probe
        self._mech_records: List[Dict] = []
        self._mech_prev_out: Optional[np.ndarray] = None
        self._mech_prev_planner: Optional[np.ndarray] = None

    def _evaluate_agent_via_planner(self, agent, env, num_episodes, task=None):
        def act_fn(obs, step):
            action, _sel = agent.select_action(obs, deterministic=True)
            return action

        metrics, mech = M.instrumented_eval(env, act_fn, num_episodes)

        out = M.ecm_action_probe_outputs(agent.ecm_registry, self._mech_probe)
        mech["churn"] = M.decision_churn(self._mech_prev_out, out)
        mech["probe_output_hash"] = M.matrix_hash(out)
        mech["probe_output_shape"] = list(out.shape)
        self._mech_prev_out = out

        pl = M.planner_probe_outputs(agent, self._mech_probe)
        mech["planner_churn"] = M.decision_churn(self._mech_prev_planner, pl)
        mech["planner_output_hash"] = M.matrix_hash(pl)
        mech["planner_output_shape"] = list(pl.shape)
        self._mech_prev_planner = pl

        self._mech_records.append(mech)
        return metrics


def run_seed(seed, args):
    if M.check_skip_or_restart(METHOD, seed, args.tag, args.iterations):
        return
    out_path = M.result_path(METHOD, seed, args.tag)

    M.seed_everything(seed)
    config = load_config(args.config)
    config["seed"] = seed
    per_ecm = config["learning"]["rl"]["total_timesteps_per_iter"]
    config["learning"]["rl"]["total_timesteps_per_iter"] = per_ecm * N_PHASE_ECMS
    log(f"  budget: {per_ecm} x {N_PHASE_ECMS} = "
        f"{config['learning']['rl']['total_timesteps_per_iter']} steps/iteration "
        f"for the single ECM, matching AM+phase total compute")

    ns = argparse.Namespace(
        exp="exp2", task=M.TASK_NAME, iterations=args.iterations,
        eval_episodes=args.eval_episodes, seed=seed, config=args.config,
    )

    probe_runner = ExperimentRunner(config, ns)
    task = probe_runner.task_suite.get_task(M.TASK_NAME)
    probe = M.load_or_build_probe(seed, probe_runner, task)
    M.seed_everything(seed)

    runner = NoPhaseMechRunner(config, ns, probe)
    task = runner.task_suite.get_task(M.TASK_NAME)
    env = runner.task_suite.make_env(task)
    try:
        M.get_robosuite_env(env)

        budget = config["learning"]["rl"]["total_timesteps_per_iter"]

        def _doc(records):
            doc = M.make_doc(METHOD, seed, args, records)
            # make_doc belongs to mechanism_runner and knows nothing about this
            # arm's budget scaling, so record the evidence explicitly: a reader
            # must be able to confirm from the artifact alone that this arm ran
            # at the same total steps per iteration as AM+phase.
            doc.setdefault("_meta", {}).update({
                "budget_steps_per_iter": budget,
                "budget_note": (f"single ECM at {budget} steps/iteration, matching "
                                f"AM+phase's {N_PHASE_ECMS} modules x "
                                f"{budget // N_PHASE_ECMS} steps each"),
                "n_ecms": 1,
                "decomposition": "none (single whole-task module)",
            })
            return doc

        def on_iter(partial):
            records = M._merged_records(partial, runner._mech_records)
            M.atomic_write_json(out_path, _doc(records))

        label = f"{M.SUBDIR}/{METHOD}/seed_{seed}" + (
            f"__{args.tag}" if args.tag else "")
        t0 = time.time()
        result = runner._run_evolution_mutable(
            env, task, label=label, on_iter_done=on_iter, resume_from=None)
        records = M._merged_records(result, runner._mech_records)
        M.atomic_write_json(out_path, _doc(records))
        log(f"[{METHOD}/seed_{seed}] done in {time.time()-t0:.0f}s -> {out_path}")
    finally:
        try:
            env.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="AM+nophase compute-matched arm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 2024])
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--grad-proxy", action="store_true")
    args = ap.parse_args()
    log(f"T6 {METHOD}: seeds={args.seeds} iters={args.iterations} "
        f"tag={args.tag or '(none)'}")
    for s in args.seeds:
        run_seed(s, args)


if __name__ == "__main__":
    main()
