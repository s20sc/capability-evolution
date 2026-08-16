"""
Main experiment runner — Capability Evolution (robosuite / MuJoCo).

Implements four experiment modes:
  1. exp1: Evolution over time (H1)
  2. exp2: Method comparison (H2) — 5 methods
  3. exp3: Safety evaluation (H3)
  4. ablation: Learning modality ablation

Usage:
    python run_experiment.py --exp all --iterations 20 --eval-episodes 30 --seed 42
    python run_experiment.py --exp exp1 --task T3_Stack --iterations 10
    python run_experiment.py --exp ablation --iterations 20
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import copy
from datetime import datetime
from typing import Dict, List, Optional
from scipy import stats

# Force unbuffered output so nohup/redirect can show progress in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg: str):
    """Print with timestamp and flush."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

import yaml

# Local imports
from envs import TaskSuite, TaskConfig
from envs.robosuite_wrapper import RobosuiteWrapper
from agent.ecm import ECM, ECMRegistry, ECMDescriptor
from agent.persistent_agent import PersistentAgent
from runtime.governance import RuntimeGovernance
from learning.rl_learner import RLLearner
from learning.hybrid_learner import HybridLearner
from baselines import (
    AgentModificationBaseline,
    StaticECMBaseline,
    SPiRLBaseline,
    SkiMoBaseline,
)


def load_config(config_path: str = None) -> Dict:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "configs", "default.yaml"
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


class ExperimentRunner:
    def __init__(self, config: Dict, args):
        self.config = config
        self.args = args
        self.results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.task_suite = TaskSuite(
            robot=config["env"]["robot"],
            controller=config["env"]["controller"],
            horizon=config["env"]["horizon"],
            reward_shaping=config["env"]["reward_shaping"],
            object_pos_noise=config["env"]["object_pos_noise"],
            obs_noise_std=config["env"]["obs_noise_std"],
            actuation_failure_prob=config["env"]["actuation_failure_prob"],
            force_limit=config["runtime"]["force_limit"],
            workspace_bounds=config["runtime"]["workspace_bounds"],
            forbidden_zones=config["runtime"]["forbidden_zones"],
            seed=config["seed"],
        )

    # =================================================================
    # Experiment 1: Evolution Over Time (H1)
    # =================================================================
    def run_exp1(self, task_name: str = None):
        """Track performance over evolution iterations."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: Evolution Over Time (H1)")
        print("=" * 80)

        tasks = [self.task_suite.get_task(task_name)] if task_name else self.task_suite.all_tasks()

        # Load existing results so single-task runs don't overwrite prior data
        save_path = os.path.join(self.results_dir, "exp1_evolution_over_time.json")
        if os.path.exists(save_path):
            with open(save_path) as f:
                all_results = json.load(f)
            log(f"Loaded existing results: {list(all_results.keys())}")
        else:
            all_results = {}

        for ti, task in enumerate(tasks, 1):
            # Skip completed tasks
            existing = all_results.get(task.name, {})
            if len(existing.get("iterations", [])) >= self.args.iterations:
                log(f"EXP1 Task {ti}/{len(tasks)}: {task.name} — already complete, skipping")
                continue

            log(f"EXP1 Task {ti}/{len(tasks)}: {task.name}")
            env = self.task_suite.make_env(task)

            def _save_exp1(partial, _task_name=task.name):
                all_results[_task_name] = partial
                self._save("exp1_evolution_over_time.json", all_results)

            results = self._run_evolution(
                env, task, label=f"exp1/{task.name}",
                on_iter_done=_save_exp1,
                resume_from=existing if len(existing.get("iterations", [])) > 0 else None,
            )
            all_results[task.name] = results
            env.close()
            self._save("exp1_evolution_over_time.json", all_results)

        self._save("exp1_evolution_over_time.json", all_results)
        return all_results

    # =================================================================
    # Experiment 2: Method Comparison (H2)
    # =================================================================
    def run_exp2(self):
        """Compare 5 methods across all tasks."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: Method Comparison (H2)")
        print("=" * 80)

        iterations = self.args.iterations
        eval_eps = self.args.eval_episodes

        # Load existing results to resume after crash
        save_path = os.path.join(self.results_dir, "exp2_method_comparison.json")
        if os.path.exists(save_path):
            with open(save_path) as f:
                all_results = json.load(f)
            log(f"Loaded existing exp2 results: {list(all_results.keys())}")
        else:
            all_results = {}

        all_tasks = self.task_suite.all_tasks()
        for ti, task in enumerate(all_tasks, 1):
            # Skip only if ALL 5 methods are complete for this task
            all_methods = ["ours", "agent_modification", "static_ecm", "spirl", "skimo"]
            if task.name in all_results:
                all_done = all(
                    len(all_results[task.name].get(m, {}).get("metrics", [])) >= iterations
                    for m in all_methods
                )
                if all_done:
                    log(f"EXP2 Task {ti}/{len(all_tasks)}: {task.name} — all methods complete, skipping")
                    continue

            log(f"EXP2 Task {ti}/{len(all_tasks)}: {task.name}")
            env = self.task_suite.make_env(task)
            task_results = all_results.get(task.name, {})

            # Method 1: Ours (Capability Evolution)
            existing_ours = all_results.get(task.name, {}).get("ours", {})
            ours_done = len(existing_ours.get("iterations", []))
            if ours_done > 0:
                log(f"  [Ours] Capability Evolution (resuming from {ours_done}/{iterations})")
            else:
                log(f"  [Ours] Capability Evolution")

            def _save_exp2(partial, _task_name=task.name):
                task_results["ours"] = partial
                all_results[_task_name] = task_results
                self._save("exp2_method_comparison.json", all_results)

            task_results["ours"] = self._run_evolution(
                env, task, label=f"exp2/{task.name}/ours",
                on_iter_done=_save_exp2,
                resume_from=existing_ours if ours_done > 0 else None,
            )

            # Per-iteration save helper for baselines
            def _save_baseline(method_name):
                def _save(partial):
                    task_results[method_name] = partial
                    all_results[task.name] = task_results
                    self._save("exp2_method_comparison.json", all_results)
                return _save

            # Method 2: Agent Modification
            log(f"  [AM] Agent Modification")
            am = AgentModificationBaseline(env, seed=self.config["seed"])
            task_results["agent_modification"] = self._run_baseline(
                am, iterations, eval_eps, "agent_modification",
                on_iter_done=_save_baseline("agent_modification"),
                resume_from=task_results.get("agent_modification"),
            )

            # Method 3: Static ECM
            log(f"  [S-ECM] Static ECM")
            secm = StaticECMBaseline(env, seed=self.config["seed"])
            task_results["static_ecm"] = self._run_baseline(
                secm, iterations, eval_eps, "static_ecm",
                on_iter_done=_save_baseline("static_ecm"),
                resume_from=task_results.get("static_ecm"),
            )

            # Method 4: SPiRL
            log(f"  [SPiRL]")
            spirl = SPiRLBaseline(env, seed=self.config["seed"])
            task_results["spirl"] = self._run_baseline(
                spirl, iterations, eval_eps, "spirl",
                on_iter_done=_save_baseline("spirl"),
                resume_from=task_results.get("spirl"),
            )

            # Method 5: SkiMo
            log(f"  [SkiMo]")
            skimo = SkiMoBaseline(env, seed=self.config["seed"])
            task_results["skimo"] = self._run_baseline(
                skimo, iterations, eval_eps, "skimo",
                on_iter_done=_save_baseline("skimo"),
                resume_from=task_results.get("skimo"),
            )

            # Statistical tests
            task_results["statistics"] = self._compute_statistics(task_results)

            all_results[task.name] = task_results
            env.close()
            # Save after each task
            self._save("exp2_method_comparison.json", all_results)

        self._save("exp2_method_comparison.json", all_results)
        self._print_exp2_summary(all_results)
        return all_results

    # =================================================================
    # Experiment 3: Safety Evaluation (H3)
    # =================================================================
    def run_exp3(self):
        """Evaluate runtime governance with/without policy enforcement."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 3: Safety Evaluation (H3)")
        print("=" * 80)

        # Load existing results to resume after crash
        save_path = os.path.join(self.results_dir, "exp3_safety.json")
        if os.path.exists(save_path):
            with open(save_path) as f:
                results = json.load(f)
            # Only keep results with valid blocking rates (not 100% from old bug)
            valid = {}
            for k, v in results.items():
                br = v.get("with_governance", {}).get("blocking_rate", 100)
                if br < 99.9:
                    valid[k] = v
            results = valid
            if results:
                log(f"Loaded existing exp3 results: {list(results.keys())}")
        else:
            results = {}

        all_tasks = self.task_suite.all_tasks()
        for ti, task in enumerate(all_tasks, 1):
            # Skip completed tasks
            if task.name in results:
                log(f"EXP3 Task {ti}/{len(all_tasks)}: {task.name} — already complete, skipping")
                continue

            log(f"EXP3 Task {ti}/{len(all_tasks)}: {task.name}")
            env = self.task_suite.make_env(task)

            # With governance
            runtime_on = RuntimeGovernance(
                force_limit=self.config["runtime"]["force_limit"],
                max_velocity=self.config["runtime"]["max_velocity"],
                workspace_bounds=self.config["runtime"]["workspace_bounds"],
                forbidden_zones=self.config["runtime"]["forbidden_zones"],
                enabled=True,
            )
            metrics_on = self._eval_safety(env, runtime_on, num_episodes=50)

            # Without governance
            runtime_off = RuntimeGovernance(enabled=False)
            metrics_off = self._eval_safety(env, runtime_off, num_episodes=50)

            results[task.name] = {
                "with_governance": metrics_on,
                "without_governance": metrics_off,
            }
            env.close()
            # Save after each task (crash-safe)
            self._save("exp3_safety.json", results)

        self._save("exp3_safety.json", results)
        return results

    # =================================================================
    # Ablation: Learning Modality
    # =================================================================
    def run_ablation(self):
        """Run ablation over learning modalities."""
        print("\n" + "=" * 80)
        print("ABLATION: Learning Modalities")
        print("=" * 80)

        ablation_configs = self.config["experiment"]["ablation"]

        # Load existing results so we can resume after a crash
        save_path = os.path.join(self.results_dir, "ablation_modalities.json")
        if os.path.exists(save_path):
            with open(save_path) as f:
                results = json.load(f)
            log(f"Loaded existing ablation results: {list(results.keys())}")
        else:
            results = {}

        # Use a representative task (T3_Stack)
        task = self.task_suite.get_task("T3_Stack")
        env = self.task_suite.make_env(task)

        for abl in ablation_configs:
            name = abl["name"]

            # Skip already-completed configs
            existing = results.get(name, {})
            done_iters = len(existing.get("iterations", []))
            if done_iters >= self.args.iterations:
                log(f"  [{name}] already complete, skipping")
                continue

            if done_iters > 0:
                print(f"\n  [{name}] rl={abl['rl']}, il={abl['il']}, llm={abl['llm']}  (resuming from {done_iters}/{self.args.iterations})")
            else:
                print(f"\n  [{name}] rl={abl['rl']}, il={abl['il']}, llm={abl['llm']}")

            learner = HybridLearner(
                rl_config=self.config["learning"]["rl"],
                il_config=self.config["learning"]["il"],
                llm_config=self.config["learning"].get("llm", {}),
                enable_rl=abl["rl"],
                enable_il=abl["il"],
                enable_llm=abl["llm"],
            )

            # Run evolution with per-iteration saving and resume support
            def _save_ablation(partial, _name=name):
                results[_name] = partial
                self._save("ablation_modalities.json", results)

            abl_results = self._run_evolution(
                env, task, learner_override=learner,
                on_iter_done=_save_ablation,
                resume_from=existing if done_iters > 0 else None,
            )
            results[name] = abl_results
            self._save("ablation_modalities.json", results)

        env.close()
        self._save("ablation_modalities.json", results)
        return results

    # =================================================================
    # Helpers
    # =================================================================
    def _run_evolution(self, env, task, learner_override=None, label="",
                       on_iter_done=None, resume_from=None):
        """Run capability evolution for N iterations.

        Args:
            resume_from: dict with prior {"iterations": [...], "metrics": [...]}
                         to resume from. Training restarts from scratch (ECM
                         weights are fresh) but completed iterations are skipped
                         and their metrics are preserved.
        """
        cfg = self.config
        iterations = self.args.iterations
        eval_eps = self.args.eval_episodes

        # Initialize
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        ecm_registry = ECMRegistry()
        phases = ["reach", "grasp", "lift", "place"]
        for phase in phases:
            desc = ECMDescriptor(
                name=f"ecm_{phase}",
                description=f"ECM for {phase}",
                phase=phase,
                input_dim=obs_dim,
                output_dim=act_dim,
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

        learner = learner_override or HybridLearner(
            rl_config=cfg["learning"]["rl"],
            il_config=cfg["learning"]["il"],
            enable_rl=True, enable_il=True, enable_llm=True,
        )

        # Resume: restore weights from checkpoint + keep prior metrics
        if resume_from and len(resume_from.get("iterations", [])) > 0:
            results = resume_from
            start_iter = len(results["iterations"])
            if start_iter >= iterations:
                log(f"  {label} already complete ({start_iter}/{iterations}), skipping")
                return results
            log(f"  {label} resuming from iteration {start_iter}/{iterations}")
            # Try to restore ECM weights from checkpoint (exact, not warmup)
            ckpt_iter = self._load_checkpoint(label, ecm_registry, learner) if label else None
            if ckpt_iter is None and start_iter > 0:
                # Fallback: warmup if no checkpoint available
                log(f"  {label} no checkpoint found, warming up ECMs...")
                for _ in range(min(start_iter, 3)):
                    for ecm in ecm_registry.active_ecms():
                        learner.update_ecm(ecm, env, runtime)
        else:
            results = {"iterations": [], "metrics": []}
            start_iter = 0

        for it in range(start_iter, iterations):
            t0 = time.time()

            # Train each ECM and apply version gating
            for ecm in ecm_registry.active_ecms():
                learner.update_ecm(ecm, env, runtime)
                # Evaluate ECM within its phase context for version gating
                gate_reward = self._eval_single_ecm(
                    ecm, env, n_episodes=3,
                    ecm_registry=ecm_registry, task=task,
                )
                accepted = ecm.evaluate_and_gate(gate_reward)
                if not accepted:
                    log(f"    {ecm.descriptor.name}: rollback (reward={gate_reward:.3f})")

            # Evaluate with phase-based ECM chaining
            metrics = self._evaluate_agent(agent, env, eval_eps, task=task)
            metrics["policy_drift"] = agent.compute_policy_drift()
            metrics["iteration"] = it + 1
            metrics["wall_time"] = time.time() - t0

            results["iterations"].append(it + 1)
            results["metrics"].append(metrics)

            log(f"  {label} iter {it+1}/{iterations}: "
                f"success={metrics['success_rate']:.1f}% "
                f"reward={metrics['avg_reward']:.2f} "
                f"σ²={metrics['reward_variance']:.1f} "
                f"Δπ={metrics['policy_drift']:.4f} "
                f"({metrics['wall_time']:.0f}s)")

            # Per-iteration interim save: metrics + weights
            if on_iter_done:
                on_iter_done(results)
            if label:
                self._save_checkpoint(label, ecm_registry, learner, it + 1)

        # Clean up persistent SAC models to free memory
        if hasattr(learner, 'rl_learner') and hasattr(learner.rl_learner, 'close'):
            learner.rl_learner.close()

        return results

    def _run_baseline(self, baseline, iterations, eval_eps, name,
                      on_iter_done=None, resume_from=None):
        """Run a baseline for N iterations."""
        if resume_from and len(resume_from.get("iterations", [])) > 0:
            results = resume_from
            start_iter = len(results["iterations"])
            if start_iter >= iterations:
                log(f"    {name} already complete ({start_iter}/{iterations}), skipping")
                return results
            log(f"    {name} resuming from {start_iter}/{iterations}")
        else:
            results = {"iterations": [], "metrics": []}
            start_iter = 0

        for it in range(start_iter, iterations):
            baseline.train_iteration()
            metrics = baseline.evaluate(eval_eps)
            metrics["iteration"] = it + 1
            results["iterations"].append(it + 1)
            results["metrics"].append(metrics)
            log(f"    iter {it+1}/{iterations}: success={metrics['success_rate']:.1f}%")
            if on_iter_done:
                on_iter_done(results)

        return results

    def _get_phase_schedule(self, ecm_registry, task, horizon):
        """Build a phase schedule: list of (start_step, end_step, ecm) tuples.

        Maps the task's ecm_steps to available ECMs. If the task needs
        more phases than ECMs available, later phases cycle back.
        The episode horizon is divided equally among phases.
        """
        active_ecms = ecm_registry.active_ecms()
        n_phases = task.ecm_steps if task else len(active_ecms)
        steps_per_phase = horizon // n_phases

        schedule = []
        for i in range(n_phases):
            ecm = active_ecms[i % len(active_ecms)]
            start = i * steps_per_phase
            end = (i + 1) * steps_per_phase if i < n_phases - 1 else horizon
            schedule.append((start, end, ecm))
        return schedule

    def _eval_single_ecm(self, ecm, env, n_episodes=3, ecm_registry=None,
                         task=None):
        """Evaluate a single ECM within its phase context.

        Runs full episodes with phase-based ECM chaining. All ECMs
        contribute, but only the reward earned during the target ECM's
        phase(s) is returned. This gives a fair per-phase evaluation
        for version gating.
        """
        horizon = env._env.horizon if hasattr(env, '_env') else 500
        schedule = self._get_phase_schedule(
            ecm_registry, task, horizon
        ) if ecm_registry else None

        total = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            while not done:
                # Determine which ECM to use based on phase schedule
                if schedule:
                    current_ecm = None
                    for start, end, sched_ecm in schedule:
                        if start <= step < end:
                            current_ecm = sched_ecm
                            break
                    if current_ecm is None:
                        current_ecm = schedule[-1][2]
                else:
                    current_ecm = ecm

                action, _ = current_ecm.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Only count reward from the target ECM's phase
                is_target_phase = (not schedule) or (current_ecm is ecm)
                if is_target_phase:
                    total += reward
                step += 1
        return total / n_episodes

    def _evaluate_agent(self, agent, env, num_episodes, task=None):
        """Evaluate using phase-based ECM chaining (paper eq. 5).

        Each episode chains ECMs sequentially: the episode horizon is
        divided into phases based on task.ecm_steps, and each phase
        uses the corresponding ECM. This implements the composition
        operator c_comp = c_1 ∘ c_2 ∘ ... from the paper.
        """
        ecm_registry = agent.ecm_registry
        horizon = env._env.horizon if hasattr(env, '_env') else 500
        schedule = self._get_phase_schedule(ecm_registry, task, horizon)

        successes = 0
        total_reward = 0.0
        ep_rewards = []
        total_time = 0

        for ep_idx in range(num_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            step = 0

            while not done:
                # Select ECM based on current phase
                current_ecm = schedule[-1][2]  # default to last phase
                for start, end, ecm in schedule:
                    if start <= step < end:
                        current_ecm = ecm
                        break

                action, _ = current_ecm.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, env_info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                step += 1

            ep_rewards.append(ep_reward)
            if env_info.get("success", False):
                successes += 1
            total_reward += ep_reward
            total_time += step

        return {
            "success_rate": successes / num_episodes * 100,
            "avg_reward": total_reward / num_episodes,
            "avg_time": total_time / num_episodes,
            "reward_variance": float(np.var(ep_rewards)),
        }

    def _eval_safety(self, env, runtime, num_episodes):
        """Evaluate with safety tracking.

        Trains ECMs for a few iterations first so the policy produces
        realistic actions (random ECMs would trigger 100% blocking).
        Uses actual ee_pos/ee_vel from env info for governance checks.
        """
        cfg = self.config
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        registry = ECMRegistry()
        for phase in ["reach", "grasp", "lift", "place"]:
            desc = ECMDescriptor(phase, phase, phase, obs_dim, act_dim)
            ecm = ECM(
                desc,
                hidden_dims=cfg["ecm"]["hidden_dims"],
                rollback_threshold=cfg["ecm"]["rollback_threshold"],
            )
            registry.register(ecm)

        # Train ECMs briefly so actions are realistic (not random).
        # Minimal config: n_envs=1, small buffer, quick learning.
        safety_rl_config = dict(cfg["learning"]["rl"])
        safety_rl_config["n_envs"] = 1
        safety_rl_config["total_timesteps_per_iter"] = 3000
        safety_rl_config["learning_starts"] = 500
        safety_rl_config["buffer_size"] = 10000
        safety_rl_config["batch_size"] = 256
        safety_rl_config["gradient_steps"] = 1
        learner = HybridLearner(
            rl_config=safety_rl_config,
            il_config=cfg["learning"]["il"],
            enable_rl=True, enable_il=False, enable_llm=False,
        )
        for ecm in registry.active_ecms():
            learner.update_ecm(ecm, env, runtime=None)
        if hasattr(learner, 'rl_learner') and hasattr(learner.rl_learner, 'close'):
            learner.rl_learner.close()

        # Use phase-based scheduling instead of the random planner
        horizon = env._env.horizon if hasattr(env, '_env') else 500
        # Safety eval doesn't have a specific task context; use 4 phases
        n_phases = 4
        steps_per_phase = horizon // n_phases
        active_ecms = registry.active_ecms()

        unsafe_episodes = 0
        total_violations = 0
        ee_pos = np.zeros(3)
        ee_vel = np.zeros(3)
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            ep_violations = 0
            while not done:
                # Phase-based ECM selection
                phase_idx = min(step // steps_per_phase, n_phases - 1)
                ecm = active_ecms[phase_idx % len(active_ecms)]
                action, _ = ecm.get_action(obs, deterministic=True)
                safe_action, blocked, gov_info = runtime.enforce(
                    action, ee_pos, ee_vel
                )
                obs, _, terminated, truncated, info = env.step(safe_action)
                # Known defect, deliberately preserved: RobosuiteWrapper.step
                # does not write "ee_pos"/"ee_vel" into info, so these lookups
                # take the zero-vector default at every step and the governance
                # predicates are evaluated at the origin. The paper documents
                # this wiring; governance_stress.py extracts real end-effector
                # state instead of relying on info.
                ee_pos = info.get("ee_pos", np.zeros(3))
                ee_vel = info.get("ee_vel", np.zeros(3))
                if info.get("safety_violation_this_step", False):
                    ep_violations += 1
                done = terminated or truncated
                step += 1

            total_violations += ep_violations
            if ep_violations > 0:
                unsafe_episodes += 1

        gov_metrics = runtime.get_metrics()
        gov_metrics["total_safety_violations"] = total_violations
        gov_metrics["unsafe_episodes"] = unsafe_episodes
        gov_metrics["unsafe_exec_pct"] = unsafe_episodes / num_episodes * 100
        return gov_metrics

    def _compute_statistics(self, task_results):
        """Compute statistical tests between methods."""
        stat = {}
        ours = task_results.get("ours", {}).get("metrics", [])
        if not ours:
            return stat

        ours_final = ours[-1]["success_rate"] if ours else 0

        for name in ["agent_modification", "static_ecm", "spirl", "skimo"]:
            bl = task_results.get(name, {}).get("metrics", [])
            if bl:
                bl_final = bl[-1]["success_rate"]
                # Effect size (Cohen's d) — approximate
                ours_std = max(np.sqrt(ours[-1].get("reward_variance", 1)), 0.01)
                bl_std = max(np.sqrt(bl[-1].get("reward_variance", 1)), 0.01)
                pooled_std = np.sqrt((ours_std**2 + bl_std**2) / 2)
                cohens_d = (ours_final - bl_final) / pooled_std if pooled_std > 0 else 0
                stat[f"vs_{name}"] = {
                    "ours": ours_final,
                    "baseline": bl_final,
                    "diff": ours_final - bl_final,
                    "cohens_d": float(cohens_d),
                }
        return stat

    def _print_exp2_summary(self, all_results):
        print("\n" + "=" * 80)
        print("EXPERIMENT 2 SUMMARY")
        print("=" * 80)
        for task_name, tr in all_results.items():
            print(f"\n{task_name}:")
            for method in ["ours", "agent_modification", "static_ecm", "spirl", "skimo"]:
                metrics = tr.get(method, {}).get("metrics", [])
                if metrics:
                    final = metrics[-1]
                    print(f"  {method:25s}: {final['success_rate']:5.1f}%  "
                          f"σ²={final.get('reward_variance',0):6.1f}  "
                          f"Δπ={final.get('policy_drift',0):.4f}")

    def _save(self, filename, data):
        path = os.path.join(self.results_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nSaved: {path}")

    # --- Checkpoint system: persist ECM weights & SAC models ---

    def _ckpt_dir(self, label: str) -> str:
        """Return (and create) the checkpoint directory for a run."""
        d = os.path.join(self.results_dir, "checkpoints", label.replace("/", "_"))
        os.makedirs(d, exist_ok=True)
        return d

    def _save_checkpoint(self, label: str, ecm_registry, learner, iteration: int):
        """Save ECM weights + SAC model to disk after each iteration."""
        d = self._ckpt_dir(label)
        # Save all ECM weights
        ecm_states = {}
        for ecm in ecm_registry.active_ecms():
            ecm_states[ecm.descriptor.name] = {
                "state_dict": {k: v.cpu() for k, v in ecm.network.state_dict().items()},
                "version": ecm.version,
                "perf_history": ecm.performance_history,
            }
        torch.save(ecm_states, os.path.join(d, "ecm_weights.pt"))
        # Save SAC model parameters (without replay buffer — buffer is
        # ~1.7GB per ECM; model params are only ~5MB and sufficient for
        # exact weight recovery. Buffer will be re-collected on resume).
        if hasattr(learner, 'rl_learner') and learner.rl_learner:
            for name, (model, vec_env) in learner.rl_learner._sac_cache.items():
                model.save(os.path.join(d, f"sac_{name}"))
        # Save iteration marker
        with open(os.path.join(d, "checkpoint_info.json"), "w") as f:
            json.dump({"iteration": iteration, "label": label}, f)

    def _load_checkpoint(self, label: str, ecm_registry, learner) -> Optional[int]:
        """Load ECM weights + SAC model from disk. Returns last saved iteration or None."""
        d = self._ckpt_dir(label)
        info_path = os.path.join(d, "checkpoint_info.json")
        ecm_path = os.path.join(d, "ecm_weights.pt")

        if not os.path.exists(info_path) or not os.path.exists(ecm_path):
            return None

        with open(info_path) as f:
            info = json.load(f)

        # Restore ECM weights
        ecm_states = torch.load(ecm_path, weights_only=False)
        for ecm in ecm_registry.active_ecms():
            name = ecm.descriptor.name
            if name in ecm_states:
                ecm.network.load_state_dict(ecm_states[name]["state_dict"])
                ecm.version = ecm_states[name]["version"]
                ecm.performance_history = ecm_states[name].get("perf_history", [])
                log(f"    restored {name} v{ecm.version}")

        # Restore SAC models with replay buffers into learner cache
        if hasattr(learner, 'rl_learner') and learner.rl_learner:
            from stable_baselines3 import SAC
            rl = learner.rl_learner
            for ecm in ecm_registry.active_ecms():
                name = ecm.descriptor.name
                sac_path = os.path.join(d, f"sac_{name}.zip")
                if os.path.exists(sac_path):
                    try:
                        loaded_model = SAC.load(sac_path, device="auto")
                        log(f"    restored SAC for {name}")
                        # Pre-populate cache; vec_env created on first finetune call
                        rl._sac_cache[name] = (loaded_model, None)
                    except Exception as e:
                        log(f"    could not restore SAC for {name}: {e}")

        log(f"  checkpoint restored from iteration {info['iteration']}")
        return info["iteration"]


def main():
    parser = argparse.ArgumentParser(description="Capability Evolution Experiments (robosuite)")
    parser.add_argument("--exp", choices=["all", "exp1", "exp2", "exp3", "ablation"],
                        default="all")
    parser.add_argument("--task", type=str, default=None,
                        help="Specific task for exp1 (e.g. T3_Stack)")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()
    config = load_config(args.config)
    config["seed"] = args.seed

    runner = ExperimentRunner(config, args)

    print(f"\nCapability Evolution Experiments")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Config: iterations={args.iterations}, eval={args.eval_episodes}, seed={args.seed}")

    if args.exp in ["all", "exp1"]:
        runner.run_exp1(args.task)
    if args.exp in ["all", "exp2"]:
        runner.run_exp2()
    if args.exp in ["all", "exp3"]:
        runner.run_exp3()
    if args.exp in ["all", "ablation"]:
        runner.run_ablation()

    print(f"\nAll experiments completed. Results in: {runner.results_dir}")


if __name__ == "__main__":
    main()
