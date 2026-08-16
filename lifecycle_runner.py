"""
T6 lifecycle fault-injection driver (versioning + gated promotion + rollback).

Purpose
-------
Isolate and evaluate the capability-LIFECYCLE mechanisms (version registry,
held-out promotion gating, rollback) that the paper describes in Sections
3.1/3.2. These mechanisms were previously introduced but not evaluated in
isolation; this driver produces that evidence on
T6_TwoArmPegInHole by injecting REGRESSED module versions at fixed iterations
and contrasting three arms:

  Arm A  gating_on           held-out gate should REJECT the regressed
                             candidate; measure injection-rejection rate and
                             an undisturbed success curve.
  Arm B  gating_off          regressed candidate deploys directly; measure
                             dip depth + dip duration.
  Arm C  gating_off_rollback no gate, but after deployment a success-rate
                             drop > --rollback-trigger points vs a SMOOTHED
                             baseline (median of the last 3 recorded
                             iterations; T6 shows natural 30-50pp swings, so
                             a single-iteration baseline mostly fires on
                             noise) restores the previously deployed version
                             (k-1); measure iterations-to-recovery. Rollbacks
                             outside injection windows are reported
                             separately as n_false_rollbacks. Inspect arm B
                             natural variance before trusting the default
                             --rollback-trigger.

3 arms x 4 seeds {42, 7, 123, 2024} x 20 iterations = 12 runs, matched to the
published T6 budget (configs/default.yaml: 50k SAC steps per ECM per
iteration, 30 eval episodes per iteration).

What exists in the codebase vs what is implemented HERE
-------------------------------------------------------
Verified against the repo (run_experiment.py, agent/ecm.py,
agent/persistent_agent.py, learning/hybrid_learner.py, learning/rl_learner.py):

* EXISTS: per-ECM training via HybridLearner.update_ecm -> RLLearner
  .finetune_ecm, which (a) syncs the ECM's current weights INTO the SAC actor
  at the start of every call (_inject_ecm_to_sac) and (b) writes trained
  weights back via ecm.evolve(). Consequence: any restore/perturbation of
  ecm.network between iterations fully propagates into the next training
  round. This is what makes external injection + rollback well-defined.
* EXISTS: a per-ECM reward-history gate (ECM.evaluate_and_gate + ECM.rollback)
  used inside ExperimentRunner._run_evolution. It is NOT a held-out,
  composition-level promotion gate: it compares consecutive per-phase reward
  scalars and its rollback restores the top of a snapshot stack (effectively
  k-1's pre-update state). It is kept ACTIVE in ALL THREE arms, exactly as in
  _run_evolution, because it is part of the base method that produced the
  published curves (2026-07-20 fix: an earlier bypass-all design flattened
  the no-injection baseline to ~0% success over 5 iterations, versus the
  published nonzero-from-iter-1 profile, making injection dips unmeasurable).
  Arms therefore differ only in the runner-level lifecycle layer stacked on
  top of the unchanged base learner. Known bookkeeping impurity: after a
  runner-level restore (gate reject / rollback), the ECM-internal
  performance_history still refers to the replaced candidate's rewards; the
  built-in gate remains a heuristic and this drift is accepted and logged.
* DOES NOT EXIST in code (implemented locally in this script):
    - an explicit version registry with version ids and full per-iteration
      capability-set snapshots (LifecycleExperimentRunner._version_registry);
    - a held-out, composition-level promotion gate (paired evaluation of
      candidate vs currently-kept version on fresh episodes; promote only if
      candidate_success >= kept_success - --gate-threshold);
    - deployment-level rollback triggered by an observed success dip.

Injection method (chosen: Gaussian parameter perturbation)
----------------------------------------------------------
Spec offered (a) truncated training (5k instead of 50k SAC steps) or (b)
Gaussian perturbation of the candidate's parameters. We fix (b) for the whole
study because:
  * it is deterministic given (seed, iteration) and independent of SAC
    internals (truncation would require mutating RLLearner.total_timesteps
    mid-run and yields a regression whose size is uncontrollable and often
    small, since the persistent replay buffer keeps helping);
  * severity is directly tunable via --inject-sigma to meet the calibration
    requirement (standalone eval drop >= 20 success points);
  * it preserves the per-iteration env/step budget exactly across arms.
Noise is relative: for every "net.*" tensor of each targeted ECM we add
N(0, (sigma * std(tensor))^2) elementwise (log_std untouched; eval is
deterministic). Default --inject-scope all perturbs all 4 phase-ECMs
(modelling a regressed capability-set release; a single perturbed ECM covers
only 2 of T6's 8 phase slots and often dips < 20 points).
CALIBRATION: the policy must be OFF THE FLOOR before a >= 20pp drop is even
arithmetically possible. Published T6 curves put success after 1-2 training
iterations at only 3-43% (exp1 seed 42: 3.3 then 13.3; t6_multiseed seed_123:
3, 13), so a 3-iteration smoke with --inject-at 2 CANNOT calibrate sigma.
Instead, before the 12-run sweep either
  (a) smoke-run with enough iterations to reach a healthy (>= 40%) baseline
      and inject late, on a seed whose published curve climbs (7 or 2024):
        python lifecycle_runner.py --arm gating_off --tag smoke --seeds 7 \
            --iterations 10 --inject-at 9
      and check the iteration-9 dip vs the pre-injection baseline; or
  (b) cheaper: restore a late-iteration checkpoint from the published
      results/checkpoints/ t6_multiseed runs, apply _perturb_ecm_states at
      several sigmas, and evaluate standalone until the drop is >= 20pp from
      a healthy baseline.
Raise --inject-sigma (default 0.5) until the injected iteration loses >= 20
success points from that healthy baseline, then keep it fixed for all runs.
NAMESPACE: calibration/smoke runs MUST use --tag (e.g. --tag smoke), which
writes to results/t6_lifecycle_smoke/ and checkpoint dirs
results/checkpoints/t6_lifecycle_smoke_*, fully separate from the production
sweep. As a second guard, lifecycle_state.pt records the run's injection/gate
parameters and iteration target; resume REFUSES (restarts fresh, loudly) if
they differ from the current CLI, so a production run can never silently
continue from a smoke run's partial state.

Honesty / budget note
---------------------
Arm A consumes 2 x --gate-episodes extra held-out episodes per iteration
(candidate + kept paired evaluation) on top of the 30 deployment-eval
episodes; arms B/C do not. This extra evaluation budget is part of the gating
mechanism itself and is recorded in _meta. If gating misses injections, report
that; never massage the verdicts.

Output schema
-------------
results/t6_lifecycle/{arm}/seed_{S}/t6_lifecycle.json:
  {
    "iterations": [1, 2, ...],
    "metrics": [
      {
        "iteration": int,               # 1-based
        "success_rate": float,          # deployment eval, 30 eps (percent)
        "avg_reward": float,
        "avg_time": float,
        "reward_variance": float,
        "policy_drift": float,          # always 0.0 (frozen agent)
        "wall_time": float,
        "candidate_version_id": str,    # e.g. "v5-inj"
        "deployed_version_id": str,     # after gate / rollback decision
        "injection_flag": bool,
        "injection_sigma": float|null,
        "gate_verdict": "promote"|"reject"|null,   # null for arms B/C
        "gate_metric": "reward"|"success",         # operative signal
        "gate_candidate_perf": float|null,         # held-out, gate-episodes
        "gate_kept_perf": float|null,              # (units of gate_metric)
        "gate_threshold": float|null,   # relative % (reward) / pp (success)
        "rollback_triggered": bool,                # arm C only
        "rollback_restored_version": str|null,
        "prev_perf": float|null,          # last iteration's metric value
        "rollback_baseline_perf": float|null  # median of last 3 (trigger)
      }, ...
    ],
    "_meta": { seed, arm, task, iterations, inject_at, resumed, ... }
  }
plus results/t6_lifecycle/summary_lifecycle.json / .md via --aggregate
(mean +/- std per arm: injection rejection rate / dip depth / dip duration /
iterations-to-recovery / false rollbacks, imitating run_t6_amphase.py's
aggregate).

Resume caveat: resume restores weights and lifecycle pointers but NOT the
numpy/torch/env RNG streams, so a resumed run is not bit-reproducible against
an uninterrupted one (same limitation as the parent runner). _meta records
"resumed" and "resumed_at_iteration" so downstream analysis can tell such
runs apart; _meta.wall_seconds covers only the final session.

Example commands
----------------
  # calibration smoke (single seed, inject late, SEPARATE --tag namespace);
  # one per arm once sigma is settled on the gating_off smoke
  python lifecycle_runner.py --arm gating_off --tag smoke --seeds 7 --iterations 10 --inject-at 9
  python lifecycle_runner.py --arm gating_on  --tag smoke --seeds 7 --iterations 10 --inject-at 9
  python lifecycle_runner.py --arm gating_off_rollback --tag smoke --seeds 7 --iterations 10 --inject-at 9

  # full sweep (run arms sequentially; each command does its 4 seeds in order)
  python lifecycle_runner.py --arm gating_on           --seeds 42 7 123 2024 --iterations 13 --inject-at 5 8 11
  python lifecycle_runner.py --arm gating_off          --seeds 42 7 123 2024 --iterations 13 --inject-at 5 8 11
  python lifecycle_runner.py --arm gating_off_rollback --seeds 42 7 123 2024 --iterations 13 --inject-at 5 8 11

  # rebuild the cross-arm summary only (add --tag smoke for a tagged namespace)
  python lifecycle_runner.py --aggregate
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
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)

from run_experiment import ExperimentRunner, load_config, log  # noqa: E402
from agent.ecm import ECM, ECMRegistry, ECMDescriptor  # noqa: E402
from agent.persistent_agent import PersistentAgent  # noqa: E402
from runtime.governance import RuntimeGovernance  # noqa: E402
from learning.hybrid_learner import HybridLearner  # noqa: E402

TASK_NAME = "T6_TwoArmPegInHole"
SUBDIR = "t6_lifecycle"
ARMS = ("gating_on", "gating_off", "gating_off_rollback")
RESULT_BASENAME = "t6_lifecycle.json"
LIFECYCLE_STATE_BASENAME = "lifecycle_state.pt"
# An injection episode counts as "recovered" once success is back within this
# many points of the pre-injection baseline (used only by --aggregate).
RECOVERY_MARGIN_PP = 5.0
# Relative margin when the lifecycle metric is the dense episode reward
# (2026-07-21 metric switch): recovered = back within 5% of baseline reward.
RECOVERY_MARGIN_FRAC = 0.05


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Runner: ExperimentRunner + local lifecycle layer
# =====================================================================
class LifecycleExperimentRunner(ExperimentRunner):
    """ExperimentRunner extended with a LOCAL version registry, a held-out
    composition-level promotion gate, fault injection, and deployment-level
    rollback. The training path (HybridLearner.update_ecm -> SAC finetune with
    persistent replay buffer) is inherited untouched, INCLUDING the built-in
    per-ECM gate (ECM.evaluate_and_gate), which stays active in all arms
    exactly as in _run_evolution (see module header, 2026-07-20 fix).
    """

    # ------------------------- state snapshot helpers -------------------
    @staticmethod
    def _snapshot_ecm_states(ecm_registry: ECMRegistry) -> Dict[str, Dict]:
        """Deep-copy every active ECM's network state_dict to CPU tensors."""
        snap = {}
        for ecm in ecm_registry.active_ecms():
            snap[ecm.descriptor.name] = {
                k: v.detach().cpu().clone()
                for k, v in ecm.network.state_dict().items()
            }
        return snap

    @staticmethod
    def _restore_ecm_states(ecm_registry: ECMRegistry, snap: Dict[str, Dict]):
        """Load a snapshot back into the live ECM networks. Because
        RLLearner.finetune_ecm re-syncs ECM -> SAC actor on every call
        (learning/rl_learner.py::_inject_ecm_to_sac), restored weights also
        become the starting point of the next training round."""
        for ecm in ecm_registry.active_ecms():
            name = ecm.descriptor.name
            if name in snap:
                ecm.network.load_state_dict(
                    {k: v.clone() for k, v in snap[name].items()}
                )

    @staticmethod
    def _perturb_ecm_states(ecm_registry: ECMRegistry, sigma: float,
                            rng: np.random.Generator,
                            target_names: Optional[List[str]] = None):
        """Regress the CANDIDATE in place: add elementwise Gaussian noise with
        std = sigma * std(tensor) to every 'net.*' parameter of the targeted
        ECMs. log_std is untouched (deterministic eval never uses it)."""
        with torch.no_grad():
            for ecm in ecm_registry.active_ecms():
                if target_names is not None and \
                        ecm.descriptor.name not in target_names:
                    continue
                for key, tensor in ecm.network.state_dict().items():
                    if not key.startswith("net."):
                        continue
                    scale = sigma * max(float(tensor.std()), 1e-3)
                    noise = rng.normal(0.0, scale, size=tuple(tensor.shape))
                    tensor.add_(torch.from_numpy(noise).to(tensor.dtype))

    # ------------------------------ gate --------------------------------
    def _held_out_perf(self, agent, env, task, episodes: int,
                       metric_key: str) -> float:
        """Held-out check: mean of `metric_key` ('avg_reward' or
        'success_rate') over `episodes` FRESH episodes (env resets with the
        standard domain perturbations), separate from the 30-episode
        deployment eval. Reuses the inherited phase-chained evaluator
        (ExperimentRunner._evaluate_agent). Default metric is the dense
        episode reward (2026-07-21 switch): T6 binary success is a late,
        heavy-tailed event that sits at 0% for long stretches on re-run
        trajectories (both smoke runs; published seeds 123/2024 similar), so
        success-based gate verdicts would mostly compare 0 with 0, while the
        reward signal is dense and stable (smoke2: plateau 288-292, injected
        iteration 217, i.e. a crisp -26% dip). The paper itself uses reward
        for exactly this criterion-strictness reason on T2-T5."""
        return self._evaluate_agent(agent, env, episodes, task=task)[
            metric_key]

    # --------------------------- main loop -------------------------------
    def _run_evolution_lifecycle(self, env, task, arm: str, seed: int,
                                 inject_at: List[int], inject_sigma: float,
                                 inject_scope: str, inject_ecm_name: str,
                                 gate_episodes: int, gate_threshold: float,
                                 rollback_trigger: float,
                                 gate_metric: str = "reward",
                                 label: str = "", on_iter_done=None,
                                 seed_dir: Optional[str] = None,
                                 resume: Optional[Dict] = None) -> Dict:
        cfg = self.config
        iterations = self.args.iterations
        eval_eps = self.args.eval_episodes
        # Operative lifecycle signal (gate + rollback + dip/recovery).
        # 'reward' (default) -> dense avg_reward; thresholds are RELATIVE
        # percentages. 'success' -> binary success_rate; thresholds are
        # absolute percentage points (the original design).
        metric_key = "avg_reward" if gate_metric == "reward" \
            else "success_rate"

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Same 4 phase-ECMs as the published Ours runs
        ecm_registry = ECMRegistry()
        for phase in ["reach", "grasp", "lift", "place"]:
            desc = ECMDescriptor(
                name=f"ecm_{phase}", description=f"ECM for {phase}",
                phase=phase, input_dim=obs_dim, output_dim=act_dim,
            )
            ecm = ECM(
                desc,
                hidden_dims=cfg["ecm"]["hidden_dims"],
                rollback_threshold=cfg["ecm"]["rollback_threshold"],
            )
            ecm_registry.register(ecm)

        agent = PersistentAgent(obs_dim, ecm_registry, seed=cfg["seed"])

        runtime = RuntimeGovernance(
            force_limit=cfg["runtime"]["force_limit"],
            max_velocity=cfg["runtime"]["max_velocity"],
            enabled=True,
        )

        learner = HybridLearner(
            rl_config=cfg["learning"]["rl"],
            il_config=cfg["learning"]["il"],
            enable_rl=True, enable_il=True, enable_llm=True,
        )

        # ---- LOCAL version registry (does not exist in the codebase) ----
        # version_id -> full capability-set snapshot (all ECM state_dicts)
        self._version_registry: Dict[str, Dict] = {}
        deployed_id = "v0"
        self._version_registry[deployed_id] = \
            self._snapshot_ecm_states(ecm_registry)
        last_deployed_id: Optional[str] = None   # deployed id one iter ago
        prev_perf: Optional[float] = None         # last recorded metric value
        perf_history: List[float] = []            # all recorded metric values

        # Parameters that define this run; persisted with the lifecycle
        # state so a later run with different CLI args refuses to resume
        # from these partials (see T6LifecycleRunner._load_resume).
        run_params = {
            "arm": arm,
            "iterations": iterations,
            "eval_episodes": eval_eps,
            "inject_at": sorted(inject_at),
            "inject_sigma": inject_sigma,
            "inject_scope": inject_scope,
            "inject_ecm_name": (inject_ecm_name
                                if inject_scope == "one" else None),
            "gate_episodes": gate_episodes if arm == "gating_on" else None,
            "gate_threshold": gate_threshold if arm == "gating_on" else None,
            "rollback_trigger": (rollback_trigger
                                 if arm == "gating_off_rollback" else None),
            "gate_metric": gate_metric,
        }

        results = {"iterations": [], "metrics": []}
        start_iter = 0

        # ---- resume (full lifecycle state + SAC via parent checkpoints) --
        if resume is not None:
            results = resume["results"]
            start_iter = len(results["iterations"])
            if start_iter >= iterations:
                log(f"  {label} already complete ({start_iter}/{iterations})")
                return results
            state = resume["state"]
            self._version_registry = state["version_registry"]
            deployed_id = state["deployed_id"]
            last_deployed_id = state["last_deployed_id"]
            prev_perf = state["prev_perf"]
            perf_history = list(state.get(
                "perf_history",
                [m[metric_key] for m in results["metrics"]]))
            # Restore SAC models (params only; replay buffer re-collects) and
            # ECM weights/version counters from the parent checkpoint system.
            self._load_checkpoint(label, ecm_registry, learner)
            # Deployed snapshot is authoritative for the live networks.
            self._restore_ecm_states(
                ecm_registry, self._version_registry[deployed_id])
            log(f"  {label} resumed at iter {start_iter}/{iterations}, "
                f"deployed={deployed_id}")

        for it in range(start_iter, iterations):
            it1 = it + 1
            t0 = time.time()
            kept_states = self._version_registry[deployed_id]

            # 1) Build the CANDIDATE: train every ECM with the full budget,
            #    keeping the base method's built-in per-ECM keep-best gate
            #    ACTIVE in all arms (2026-07-20 design fix: the published
            #    curves were produced with this gate; bypassing it flattened
            #    the baseline to ~0% in the first smoke run, killing dip
            #    measurability and comparability with the published figures).
            #    The runner-level lifecycle layer (version registry, held-out
            #    cross-module promotion gate, deployment rollback) sits ON TOP
            #    and is what the three arms vary.
            for ecm in ecm_registry.active_ecms():
                learner.update_ecm(ecm, env, runtime)
                gate_reward = self._eval_single_ecm(
                    ecm, env, n_episodes=3,
                    ecm_registry=ecm_registry, task=task,
                )
                accepted = ecm.evaluate_and_gate(gate_reward)
                if not accepted:
                    log(f"    {ecm.descriptor.name}: builtin keep-best revert "
                        f"(reward={gate_reward:.3f})")

            # 2) Fault injection: replace the candidate with a regressed one.
            injected = it1 in inject_at
            candidate_id = f"v{it1}-inj" if injected else f"v{it1}"
            if injected:
                rng = np.random.default_rng(seed * 100003 + it1)
                targets = None if inject_scope == "all" else [inject_ecm_name]
                self._perturb_ecm_states(
                    ecm_registry, inject_sigma, rng, target_names=targets)
                log(f"  {label} iter {it1}: INJECTED regressed candidate "
                    f"{candidate_id} (sigma={inject_sigma}, "
                    f"scope={inject_scope})")
            cand_states = self._snapshot_ecm_states(ecm_registry)
            self._version_registry[candidate_id] = cand_states

            # 3) Promotion decision.
            gate_verdict = None
            gate_cand = gate_kept = None
            if arm == "gating_on":
                # Held-out paired evaluation: candidate first (already live),
                # then the kept version, then act on the verdict.
                gate_cand = self._held_out_perf(
                    agent, env, task, gate_episodes, metric_key)
                self._restore_ecm_states(ecm_registry, kept_states)
                gate_kept = self._held_out_perf(
                    agent, env, task, gate_episodes, metric_key)
                if gate_metric == "reward":
                    # Relative comparison on the dense reward: promote iff
                    # candidate is within gate_threshold PERCENT of kept.
                    promote = gate_cand >= gate_kept * (
                        1.0 - gate_threshold / 100.0)
                else:
                    # Absolute percentage-point comparison on success rate.
                    promote = gate_cand >= gate_kept - gate_threshold
                if promote:
                    gate_verdict = "promote"
                    self._restore_ecm_states(ecm_registry, cand_states)
                    last_deployed_id, deployed_id = deployed_id, candidate_id
                else:
                    gate_verdict = "reject"
                    # kept weights are already live; deployed_id unchanged
                unit = "%" if gate_metric == "success" else ""
                log(f"  {label} iter {it1}: gate {gate_verdict} "
                    f"(cand={gate_cand:.1f}{unit} kept={gate_kept:.1f}{unit} "
                    f"metric={gate_metric} thr={gate_threshold})")
            else:
                # Arms B/C: unconditional promotion.
                last_deployed_id, deployed_id = deployed_id, candidate_id

            # 4) Deployment evaluation (published budget: 30 episodes).
            metrics = self._evaluate_agent(agent, env, eval_eps, task=task)
            metrics["policy_drift"] = agent.compute_policy_drift()

            # 5) Arm C rollback: dip detector on the deployment metric.
            #    Baseline is the MEDIAN of the last <=3 recorded iterations,
            #    not the single previous value: published T6 curves swing
            #    30-50pp iteration-to-iteration without any injection, so a
            #    single-iteration baseline mostly detects eval noise.
            #    metric=reward: trigger when reward drops more than
            #    rollback_trigger PERCENT below the median-3 baseline.
            #    metric=success: absolute percentage points (original).
            rollback_triggered = False
            restored_version = None
            rollback_baseline = (
                float(np.median(perf_history[-3:]))
                if perf_history else None)
            if rollback_baseline is not None:
                if gate_metric == "reward":
                    rb_floor = rollback_baseline * (
                        1.0 - rollback_trigger / 100.0)
                else:
                    rb_floor = rollback_baseline - rollback_trigger
            else:
                rb_floor = None
            if (arm == "gating_off_rollback" and rb_floor is not None
                    and last_deployed_id is not None
                    and metrics[metric_key] < rb_floor):
                self._restore_ecm_states(
                    ecm_registry, self._version_registry[last_deployed_id])
                rollback_triggered = True
                restored_version = last_deployed_id
                log(f"  {label} iter {it1}: ROLLBACK "
                    f"{deployed_id} -> {last_deployed_id} "
                    f"({metric_key} {metrics[metric_key]:.1f} < "
                    f"floor {rb_floor:.1f}, median3 "
                    f"{rollback_baseline:.1f}, trig={rollback_trigger})")
                deployed_id = last_deployed_id
                # NOTE: the dipped success_rate stays recorded for this
                # iteration; recovery shows up from the next iteration on
                # (that is the iterations-to-recovery metric).

            metrics.update({
                "iteration": it1,
                "wall_time": time.time() - t0,
                "candidate_version_id": candidate_id,
                "deployed_version_id": deployed_id,
                "injection_flag": injected,
                "injection_sigma": inject_sigma if injected else None,
                "gate_verdict": gate_verdict,
                "gate_metric": gate_metric,
                "gate_candidate_perf": gate_cand,
                "gate_kept_perf": gate_kept,
                "gate_threshold": gate_threshold if arm == "gating_on" else None,
                "rollback_triggered": rollback_triggered,
                "rollback_restored_version": restored_version,
                "prev_perf": prev_perf,
                "rollback_baseline_perf": (
                    rollback_baseline if arm == "gating_off_rollback"
                    else None),
            })
            results["iterations"].append(it1)
            results["metrics"].append(metrics)
            prev_perf = metrics[metric_key]
            perf_history.append(metrics[metric_key])

            log(f"  {label} iter {it1}/{iterations}: "
                f"success={metrics['success_rate']:.1f}% "
                f"reward={metrics['avg_reward']:.1f} "
                f"deployed={deployed_id} inj={injected} "
                f"gate={gate_verdict} rb={rollback_triggered} "
                f"({metrics['wall_time']:.0f}s)")

            # 6) Crash-safe persistence: metrics JSON, SAC/ECM checkpoint,
            #    lifecycle state (registry + pointers).
            if on_iter_done:
                on_iter_done(results)
            if label:
                self._save_checkpoint(label, ecm_registry, learner, it1)
            if seed_dir:
                torch.save({
                    "iteration": it1,
                    "version_registry": self._version_registry,
                    "deployed_id": deployed_id,
                    "last_deployed_id": last_deployed_id,
                    "prev_perf": prev_perf,
                    "perf_history": list(perf_history),
                    "run_params": run_params,
                }, os.path.join(seed_dir, LIFECYCLE_STATE_BASENAME))

        if hasattr(learner, "rl_learner") and \
                hasattr(learner.rl_learner, "close"):
            learner.rl_learner.close()
        return results


# =====================================================================
# Multi-seed driver (imitates run_t6_amphase.T6AMPhaseRunner)
# =====================================================================
def subdir_for_tag(tag: Optional[str]) -> str:
    """Results/checkpoint namespace: '' -> t6_lifecycle (production sweep),
    'smoke' -> t6_lifecycle_smoke (calibration), etc. Keeps smoke runs from
    ever sharing partials or checkpoints with the production sweep."""
    return f"{SUBDIR}_{tag}" if tag else SUBDIR


class T6LifecycleRunner:
    def __init__(self, arm: str, seeds: List[int], args):
        self.arm = arm
        self.seeds = seeds
        self.args = args
        self.subdir = subdir_for_tag(getattr(args, "tag", ""))
        self.base_results_dir = os.path.join(EXP_DIR, "results", self.subdir)
        self.arm_dir = os.path.join(self.base_results_dir, arm)
        os.makedirs(self.arm_dir, exist_ok=True)

    def _seed_dir(self, seed: int) -> str:
        d = os.path.join(self.arm_dir, f"seed_{seed}")
        os.makedirs(d, exist_ok=True)
        return d

    def _result_path(self, seed: int) -> str:
        return os.path.join(self._seed_dir(seed), RESULT_BASENAME)

    def _save(self, seed: int, data: Dict):
        p = self._result_path(seed)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, p)

    def _expected_run_params(self) -> Dict:
        """Mirror of the run_params dict persisted in lifecycle_state.pt
        (see _run_evolution_lifecycle). Keep the two in sync."""
        a = self.args
        return {
            "arm": self.arm,
            "iterations": a.iterations,
            "eval_episodes": a.eval_episodes,
            "inject_at": sorted(a.inject_at),
            "inject_sigma": a.inject_sigma,
            "inject_scope": a.inject_scope,
            "inject_ecm_name": (a.inject_ecm_name
                                if a.inject_scope == "one" else None),
            "gate_episodes": (a.gate_episodes
                              if self.arm == "gating_on" else None),
            "gate_threshold": (a.gate_threshold
                               if self.arm == "gating_on" else None),
            "rollback_trigger": (a.rollback_trigger
                                 if self.arm == "gating_off_rollback"
                                 else None),
            "gate_metric": a.gate_metric,
        }

    def _load_resume(self, seed: int) -> Optional[Dict]:
        """Return {'results':..., 'state':...} if a consistent partial run
        exists (metrics JSON + lifecycle_state.pt at the same iteration) AND
        it was produced with the SAME injection/gate/iteration parameters as
        the current CLI. On any mismatch the partials are ignored and the run
        restarts from scratch (loudly), so a production sweep can never
        silently continue from a smoke/calibration run."""
        p = self._result_path(seed)
        sp = os.path.join(self._seed_dir(seed), LIFECYCLE_STATE_BASENAME)
        if not (os.path.exists(p) and os.path.exists(sp)):
            return None
        try:
            with open(p) as f:
                results = json.load(f)
            state = torch.load(sp, weights_only=False)
        except Exception as e:
            log(f"  [resume] unreadable partials for seed={seed}: {e}; "
                f"restarting from scratch")
            return None
        n = len(results.get("iterations", []))
        if n == 0 or state.get("iteration") != n:
            log(f"  [resume] inconsistent partials for seed={seed} "
                f"(json={n}, state={state.get('iteration')}); restarting")
            return None
        expected = self._expected_run_params()
        stored = state.get("run_params")
        if stored != expected:
            log(f"  [resume] RUN-PARAMETER MISMATCH for seed={seed}: "
                f"REFUSING to resume from existing partials.")
            log(f"  [resume]   stored : {stored}")
            log(f"  [resume]   current: {expected}")
            log(f"  [resume] Restarting this run from scratch; the old "
                f"partials/checkpoints in this namespace will be "
                f"overwritten. Use --tag to keep calibration runs in a "
                f"separate namespace.")
            return None
        return {"results": results, "state": state}

    def run_seed(self, seed: int):
        a = self.args
        log(f"========== LIFECYCLE arm={self.arm} SEED {seed} ==========")
        seed_everything(seed)

        config = load_config(a.config)
        config["seed"] = seed
        ns = argparse.Namespace(
            exp="lifecycle", task=TASK_NAME,
            iterations=a.iterations, eval_episodes=a.eval_episodes,
            seed=seed, config=a.config,
        )
        runner = LifecycleExperimentRunner(config, ns)

        task = runner.task_suite.get_task(TASK_NAME)
        if task is None:
            raise RuntimeError(f"Task {TASK_NAME} not found in task suite")

        resume = self._load_resume(seed)
        if resume and len(resume["results"]["iterations"]) >= a.iterations:
            log(f"  [{self.arm}] already complete for seed={seed}, skipping")
            return

        env = runner.task_suite.make_env(task)
        try:
            def _on_iter(partial, _seed=seed):
                self._save(_seed, partial)

            t0 = time.time()
            result = runner._run_evolution_lifecycle(
                env, task, arm=self.arm, seed=seed,
                inject_at=list(a.inject_at),
                inject_sigma=a.inject_sigma,
                inject_scope=a.inject_scope,
                inject_ecm_name=a.inject_ecm_name,
                gate_episodes=a.gate_episodes,
                gate_threshold=a.gate_threshold,
                rollback_trigger=a.rollback_trigger,
                gate_metric=a.gate_metric,
                label=f"{self.subdir}/{self.arm}/seed_{seed}",
                on_iter_done=_on_iter,
                seed_dir=self._seed_dir(seed),
                resume=resume,
            )
            result["_meta"] = {
                "seed": seed, "arm": self.arm, "task": TASK_NAME,
                "iterations": a.iterations,
                "eval_episodes": a.eval_episodes,
                "inject_at": list(a.inject_at),
                "inject_sigma": a.inject_sigma,
                "inject_scope": a.inject_scope,
                "inject_ecm_name": (a.inject_ecm_name
                                    if a.inject_scope == "one" else None),
                "gate_episodes": (a.gate_episodes
                                  if self.arm == "gating_on" else None),
                "gate_threshold": (a.gate_threshold
                                   if self.arm == "gating_on" else None),
                "rollback_trigger": (a.rollback_trigger
                                     if self.arm == "gating_off_rollback"
                                     else None),
                "gate_metric": a.gate_metric,
                "extra_gate_episodes_per_iter": (
                    2 * a.gate_episodes if self.arm == "gating_on" else 0),
                "tag": getattr(a, "tag", "") or None,
                # A resumed run restores weights/pointers but NOT RNG
                # streams, and wall_seconds covers only the final session.
                "resumed": bool(resume),
                "resumed_at_iteration": (
                    len(resume["results"]["iterations"]) if resume else None),
                "wall_seconds": time.time() - t0,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            self._save(seed, result)
            log(f"  [{self.arm}] seed={seed} done in "
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


# =====================================================================
# Aggregation (--aggregate): summary_lifecycle.{json,md}
# =====================================================================
def _mean_std_ci(values: List[float]) -> Dict:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci = 0.0
    if len(arr) >= 2:
        try:
            from scipy import stats as _st
            tcrit = float(_st.t.ppf(0.975, df=len(arr) - 1))
        except ImportError:
            tcrit = 1.96
        ci = float(tcrit * std / np.sqrt(len(arr)))
    return {"mean": mean, "std": std, "ci95": ci, "n": len(arr),
            "values": [float(v) for v in arr]}


def _injection_episodes(metrics: List[Dict],
                        metric_key: str = "avg_reward") -> List[Dict]:
    """Per injected iteration, compute dip/recovery stats on `metric_key`
    within the window [t_inject, next injection or end]. Baseline = mean
    metric over the pre-injection iterations t-3..t-1 (as many as exist),
    not the single t-1 value: published T6 curves swing 30-50pp
    iteration-to-iteration, so a one-iteration baseline confounds dip
    depth/recovery with eval noise. Recovery margin: relative
    (RECOVERY_MARGIN_FRAC of baseline) for the reward metric, absolute
    (RECOVERY_MARGIN_PP points) for the success metric."""
    inject_iters = [m["iteration"] for m in metrics if m.get("injection_flag")]
    by_iter = {m["iteration"]: m for m in metrics}
    n_total = len(metrics)
    episodes = []
    for idx, t in enumerate(inject_iters):
        pre = [by_iter[k][metric_key]
               for k in range(t - 3, t) if k in by_iter]
        if not pre:
            continue  # injection at iteration 1 has no baseline
        baseline = float(np.mean(pre))
        if metric_key == "success_rate":
            floor = baseline - RECOVERY_MARGIN_PP
        else:
            floor = baseline * (1.0 - RECOVERY_MARGIN_FRAC)
        window_end = inject_iters[idx + 1] - 1 \
            if idx + 1 < len(inject_iters) else n_total
        window = [by_iter[k][metric_key]
                  for k in range(t, window_end + 1) if k in by_iter]
        if not window:
            continue
        dip_depth = baseline - min(window)
        dip_depth_pct = (100.0 * dip_depth / baseline) if baseline else None
        # duration: consecutive iterations below the margin floor, from t
        duration = 0
        for s in window:
            if s < floor:
                duration += 1
            else:
                break
        # recovery: first k >= 0 with the metric back above the floor
        recovery = None
        for k, s in enumerate(window):
            if s >= floor:
                recovery = k
                break
        episodes.append({
            "inject_iteration": t,
            "window_end": window_end,
            "metric_key": metric_key,
            "baseline_perf": baseline,
            "baseline_n_iters": len(pre),
            "dip_depth": dip_depth,
            "dip_depth_pct": dip_depth_pct,
            "dip_duration_iters": duration,
            "iterations_to_recovery": recovery,   # None => censored
            "gate_verdict": by_iter[t].get("gate_verdict"),
            "rollback_in_window": any(
                by_iter[k].get("rollback_triggered")
                for k in range(t, window_end + 1) if k in by_iter),
        })
    return episodes


def aggregate_and_write(base_dir: str) -> Dict:
    summary = {
        "task": TASK_NAME,
        "experiment": "lifecycle_fault_injection",
        "recovery_margin_pp": RECOVERY_MARGIN_PP,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "arms": {},
    }

    for arm in ARMS:
        arm_dir = os.path.join(base_dir, arm)
        if not os.path.isdir(arm_dir):
            continue
        per_seed = {}
        for entry in sorted(os.listdir(arm_dir)):
            if not entry.startswith("seed_"):
                continue
            fp = os.path.join(arm_dir, entry, RESULT_BASENAME)
            if not os.path.exists(fp):
                continue
            try:
                with open(fp) as f:
                    data = json.load(f)
                s = int(entry.split("_", 1)[1])
            except Exception:
                continue
            metrics = data.get("metrics", [])
            if not metrics:
                continue
            # Operative metric recorded per-iteration by the run itself;
            # falls back to reward for pre-switch legacy rows.
            gm = metrics[-1].get("gate_metric", "reward")
            mkey = "avg_reward" if gm == "reward" else "success_rate"
            successes = [m["success_rate"] for m in metrics]
            rewards = [m.get("avg_reward") for m in metrics]
            injected = [m for m in metrics if m.get("injection_flag")]
            clean = [m for m in metrics if not m.get("injection_flag")]
            eps = _injection_episodes(metrics, metric_key=mkey)
            seed_stats = {
                "n_iters": len(metrics),
                "gate_metric": gm,
                "peak_success": float(max(successes)),
                "final_success": float(successes[-1]),
                "peak_reward": (float(max(rewards))
                                if all(r is not None for r in rewards)
                                else None),
                "final_reward": (float(rewards[-1])
                                 if rewards[-1] is not None else None),
                "n_injections": len(injected),
                "injection_episodes": eps,
            }
            if arm == "gating_on":
                inj_rej = [1.0 for m in injected
                           if m.get("gate_verdict") == "reject"]
                clean_rej = [1.0 for m in clean
                             if m.get("gate_verdict") == "reject"]
                seed_stats["injection_rejection_rate"] = (
                    len(inj_rej) / len(injected) if injected else None)
                seed_stats["false_rejection_rate"] = (
                    len(clean_rej) / len(clean) if clean else None)
            if arm == "gating_off_rollback":
                rb_iters = [m["iteration"] for m in metrics
                            if m.get("rollback_triggered")]
                in_window = set()
                for e in eps:
                    in_window.update(
                        range(e["inject_iteration"], e["window_end"] + 1))
                seed_stats["n_rollbacks"] = len(rb_iters)
                # Honesty metric (mirrors arm A's false_rejection_rate):
                # rollbacks at iterations outside every injection window are
                # noise-triggered, not injection responses.
                seed_stats["n_false_rollbacks"] = sum(
                    1 for t in rb_iters if t not in in_window)
            per_seed[s] = seed_stats

        if not per_seed:
            continue
        agg = {}
        agg["peak_success"] = _mean_std_ci(
            [v["peak_success"] for v in per_seed.values()])
        agg["final_success"] = _mean_std_ci(
            [v["final_success"] for v in per_seed.values()])
        pk_r = [v["peak_reward"] for v in per_seed.values()
                if v.get("peak_reward") is not None]
        fin_r = [v["final_reward"] for v in per_seed.values()
                 if v.get("final_reward") is not None]
        agg["peak_reward"] = _mean_std_ci(pk_r) if pk_r else None
        agg["final_reward"] = _mean_std_ci(fin_r) if fin_r else None
        all_eps = [e for v in per_seed.values()
                   for e in v["injection_episodes"]]
        if all_eps:
            agg["dip_depth"] = _mean_std_ci(
                [e["dip_depth"] for e in all_eps])
            dpct = [e["dip_depth_pct"] for e in all_eps
                    if e.get("dip_depth_pct") is not None]
            agg["dip_depth_pct"] = _mean_std_ci(dpct) if dpct else None
            agg["dip_duration_iters"] = _mean_std_ci(
                [float(e["dip_duration_iters"]) for e in all_eps])
            recov = [float(e["iterations_to_recovery"]) for e in all_eps
                     if e["iterations_to_recovery"] is not None]
            agg["iterations_to_recovery"] = (
                _mean_std_ci(recov) if recov else None)
            agg["n_censored_recoveries"] = sum(
                1 for e in all_eps if e["iterations_to_recovery"] is None)
        if arm == "gating_on":
            rr = [v["injection_rejection_rate"] for v in per_seed.values()
                  if v.get("injection_rejection_rate") is not None]
            fr = [v["false_rejection_rate"] for v in per_seed.values()
                  if v.get("false_rejection_rate") is not None]
            agg["injection_rejection_rate"] = _mean_std_ci(rr) if rr else None
            agg["false_rejection_rate"] = _mean_std_ci(fr) if fr else None
        if arm == "gating_off_rollback":
            agg["n_rollbacks"] = _mean_std_ci(
                [float(v["n_rollbacks"]) for v in per_seed.values()])
            agg["n_false_rollbacks"] = _mean_std_ci(
                [float(v["n_false_rollbacks"]) for v in per_seed.values()])
        summary["arms"][arm] = {"per_seed": per_seed, "aggregate": agg}

    # tmp + os.replace: the run commands call this after every seed, and
    # several arm processes may be live at once, so plain open(..., "w")
    # could leave a truncated/interleaved summary.
    out_json = os.path.join(base_dir, "summary_lifecycle.json")
    tmp_json = out_json + ".tmp"
    with open(tmp_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    os.replace(tmp_json, out_json)

    # ---- markdown ----
    def _fmt(stat, pct=False, digits=1):
        if not stat:
            return "n/a"
        unit = "%" if pct else ""
        return (f"{stat['mean']:.{digits}f}{unit} ± {stat['std']:.{digits}f} "
                f"(n={stat['n']})")

    lines = ["# T6 Lifecycle Fault Injection - Summary\n",
             f"Task: `{TASK_NAME}`  Generated: {summary['generated_at']}  "
             f"Recovery margin: {RECOVERY_MARGIN_FRAC*100:.0f}% of baseline "
             f"reward (reward metric) / {RECOVERY_MARGIN_PP} pp (success "
             f"metric)\n",
             "| Arm | Peak Reward | Final Reward | Peak Success | "
             "Inj. Rejection | False Rejection | Dip Depth (% of baseline) | "
             "Dip Duration | Iters-to-Recovery | Rollbacks (false) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        a = summary["arms"].get(arm, {}).get("aggregate")
        if not a:
            continue
        rej = a.get("injection_rejection_rate")
        frej = a.get("false_rejection_rate")
        rec = a.get("iterations_to_recovery")
        cens = a.get("n_censored_recoveries", 0)
        rec_s = _fmt(rec) if rec else "n/a"
        # Annotate censoring even when rec is None (every episode censored,
        # i.e. never recovered): that worst case must stay visible.
        if cens:
            rec_s += f" [{cens} censored]"
        nrb = a.get("n_rollbacks")
        nfrb = a.get("n_false_rollbacks")
        if nrb:
            rb_s = _fmt(nrb)
            if nfrb:
                rb_s += f" / {_fmt(nfrb)} false"
        else:
            rb_s = "n/a"
        lines.append(
            f"| {arm} | {_fmt(a.get('peak_reward'))} | "
            f"{_fmt(a.get('final_reward'))} | "
            f"{_fmt(a.get('peak_success'), pct=True)} | "
            f"{_fmt({**rej, 'mean': rej['mean']*100, 'std': rej['std']*100}, pct=True) if rej else 'n/a'} | "
            f"{_fmt({**frej, 'mean': frej['mean']*100, 'std': frej['std']*100}, pct=True) if frej else 'n/a'} | "
            f"{_fmt(a.get('dip_depth_pct'), pct=True)} | "
            f"{_fmt(a.get('dip_duration_iters'))} | {rec_s} | {rb_s} |")
    out_md = os.path.join(base_dir, "summary_lifecycle.md")
    tmp_md = out_md + ".tmp"
    with open(tmp_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp_md, out_md)

    log(f"Aggregate written: {out_json}")
    log(f"Markdown summary:  {out_md}")
    return summary


# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="T6 lifecycle fault-injection experiment "
                    "(versioning + gated promotion + rollback)")
    parser.add_argument("--arm", choices=list(ARMS), default=None,
                        help="Which lifecycle arm to run "
                             "(required unless --aggregate)")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 7, 123, 2024])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--inject-at", type=int, nargs="+",
                        default=[5, 10, 15],
                        help="1-based iterations at which the candidate is "
                             "replaced by a regressed version")
    parser.add_argument("--inject-sigma", type=float, default=0.5,
                        help="Relative Gaussian noise scale: per-tensor std "
                             "= sigma * std(tensor). Calibrated 2026-07-21 "
                             "(smoke2, seed 7): sigma=0.5 dropped the "
                             "deployment reward 292 -> 217 (-26%%) at the "
                             "injected iteration, recovering next iteration.")
    parser.add_argument("--inject-scope", choices=["all", "one"],
                        default="all",
                        help="Perturb all 4 phase-ECMs (regressed release) "
                             "or a single ECM")
    parser.add_argument("--inject-ecm-name", type=str, default="ecm_reach",
                        help="Target ECM when --inject-scope one")
    parser.add_argument("--gate-episodes", type=int, default=30,
                        help="Held-out episodes per side (candidate AND kept) "
                             "for the gating_on promotion check. Default 30: "
                             "10 episodes give only 10pp granularity against "
                             "a 10pp threshold on a policy whose 30-episode "
                             "evals already swing 30-50pp, so gate verdicts "
                             "would mostly reflect eval noise. Verify verdict "
                             "stability on the calibration smoke before the "
                             "sweep.")
    parser.add_argument("--gate-metric", choices=["reward", "success"],
                        default="reward",
                        help="Operative signal for gate/rollback/dip "
                             "metrics. Default reward (dense avg_reward): "
                             "T6 binary success sits at 0%% for long "
                             "stretches on re-run trajectories, so "
                             "success-based verdicts mostly compare 0 with "
                             "0; the paper itself uses reward on T2-T5 for "
                             "the same criterion-strictness reason. Binary "
                             "success is still logged and reported.")
    parser.add_argument("--gate-threshold", type=float, default=10.0,
                        help="metric=reward: promote iff candidate_reward "
                             ">= kept_reward * (1 - thr/100), i.e. RELATIVE "
                             "percent. metric=success: promote iff "
                             "candidate >= kept - thr (absolute points). "
                             "Smoke2 reference: the injected candidate was "
                             "-26%% reward, so the default 10%% cleanly "
                             "separates injection from eval noise "
                             "(clean-iteration swing was <2%% at plateau).")
    parser.add_argument("--rollback-trigger", type=float, default=15.0,
                        help="Arm C dip detector vs the median of the last "
                             "3 iterations. metric=reward: RELATIVE percent "
                             "drop (default 15%%; smoke2 injected dip was "
                             "-26%% vs plateau, natural swing <5%%). "
                             "metric=success: absolute points. Check "
                             "n_false_rollbacks in the aggregate and raise "
                             "if rollbacks fire outside injection windows.")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional namespace suffix: results go to "
                             "results/t6_lifecycle_{tag}/ and checkpoints to "
                             "results/checkpoints/t6_lifecycle_{tag}_*. "
                             "REQUIRED for calibration/smoke runs so they "
                             "never share partials with the production "
                             "sweep.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--aggregate", action="store_true",
                        help="Only rebuild summary_lifecycle.{json,md}")
    args = parser.parse_args()

    base_dir = os.path.join(EXP_DIR, "results", subdir_for_tag(args.tag))
    os.makedirs(base_dir, exist_ok=True)

    if args.aggregate:
        aggregate_and_write(base_dir)
        return

    if args.arm is None:
        parser.error("--arm is required unless --aggregate is given")

    log(f"T6 lifecycle run: arm={args.arm}, seeds={args.seeds}, "
        f"iters={args.iterations}, inject_at={args.inject_at}, "
        f"sigma={args.inject_sigma}, scope={args.inject_scope}, "
        f"tag={args.tag or '(none)'}")
    log(f"Results dir: {os.path.join(base_dir, args.arm)}")

    runner = T6LifecycleRunner(args.arm, args.seeds, args)
    runner.run_all()
    summary = aggregate_and_write(base_dir)

    print("\n" + "=" * 60)
    print("T6 LIFECYCLE - CURRENT AGGREGATE")
    print("=" * 60)
    for arm, blob in summary.get("arms", {}).items():
        agg = blob.get("aggregate", {})
        peak = agg.get("peak_success")
        if peak:
            print(f"  {arm:22s} peak={peak['mean']:5.1f}% ± "
                  f"{peak['std']:4.1f} (n={peak['n']})")
        rej = agg.get("injection_rejection_rate")
        if rej:
            print(f"  {arm:22s} injection rejection = "
                  f"{rej['mean']*100:.0f}%")
        rec = agg.get("iterations_to_recovery")
        if rec:
            print(f"  {arm:22s} iters-to-recovery = {rec['mean']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
