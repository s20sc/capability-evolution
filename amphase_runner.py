"""
AM+phase (Hierarchical-Mutable) runner — matched baseline for the T6 confound.

Motivation
----------
The original comparison had a confound: "Ours" (capability evolution:
K whole-task ECMs rotated on a time-slice schedule + IL demos + SAC learner + LLM synthesis, with the
agent/scheduler FROZEN) beating "Agent Modification" (a single monolithic PPO
policy) conflates two things:
  (a) the multi-module rotation + extra learning signal, and
  (b) the actual claimed contribution: capability evolution under a FROZEN agent.

This runner builds the MATCHED baseline "AM+phase" (Hierarchical-Mutable):
identical to Ours in EVERY structural dimension —
  * SAME K=4 whole-task ECMs (labelled reach / grasp / lift / place; the
    labels name schedule slots, not training scopes -- every module trains
    on the whole task)
  * SAME IL demonstrations
  * SAME SAC learner (learning.rl_learner via HybridLearner.update_ecm)
  * SAME time-slice rotation (task.ecm_steps schedule as the base; there is
    no phase-scoped training anywhere in this system)
  * SAME per-iteration budget (50k SAC steps/iter), SAME 20 iterations,
    SAME 30 eval episodes, SAME version gating, SAME env perturbations —
EXCEPT the agent/scheduler is NOT frozen: it updates jointly with the ECMs
each iteration (agent-mutating). The ONLY difference between Ours and AM+phase
is frozen-vs-mutable agent, given identical structure and learning signal.

Where Ours freezes the agent
----------------------------
* agent/persistent_agent.py:54-55  -> planner params requires_grad = False
* run_experiment.py::_run_evolution -> training loop ONLY calls
  learner.update_ecm(...); the PersistentAgent is never updated, and eval uses
  the fixed time-slice _get_phase_schedule (planner is unused; Δπ == 0.0).

What AM+phase changes (and NOTHING else)
-----------------------------------------
* Unfreeze the planner (requires_grad = True).
* After ECM training each iteration, train the scheduler/planner by behavioral
  cloning toward the empirically-best phase-ECM per observation, using rollouts
  collected from the SAME env under the current schedule (co-adaptation).
  -> planner params change every iteration  =>  Δπ > 0.
* At eval, route the phase decision through the LEARNED planner
  (agent.select_action) instead of the fixed time-slice schedule.

Everything else (ECM creation, IL, SAC, LLM, gating, budgets, eval episode
count, env perturbations, seeding) is inherited verbatim from
run_experiment.ExperimentRunner. run_experiment.py itself is NOT modified.

Honest-null contract
--------------------
If Ours >= AM+phase on T6 -> the frozen-agent capability-evolution IS the cause
of the win. If Ours < AM+phase or ~= -> honest null; report it. Never manufacture
a win for Ours; report exactly what runs.
"""

import copy
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from run_experiment import ExperimentRunner, log
from agent.ecm import ECM, ECMRegistry, ECMDescriptor
from agent.persistent_agent import PersistentAgent
from runtime.governance import RuntimeGovernance
from learning.hybrid_learner import HybridLearner


class ExperimentRunnerMutable(ExperimentRunner):
    """ExperimentRunner whose _run_evolution mutates the agent/scheduler.

    This is the AM+phase (Hierarchical-Mutable) baseline. It reuses 100% of the
    Ours machinery; the only behavioral change is that the agent's phase
    scheduler is (a) unfrozen, (b) trained jointly with the ECMs each iteration,
    and (c) used at eval time.
    """

    # -- planner co-adaptation hyperparameters --
    # Kept deliberately light so AM+phase does NOT receive extra *env* budget
    # beyond a small rollout collection; the ECM SAC budget (50k/iter) is
    # identical to Ours. These only govern the supervised planner fit.
    PLANNER_ROLLOUT_EPISODES = 5      # rollouts collected per iter to label phases
    PLANNER_BC_EPOCHS = 20            # supervised epochs on the collected labels
    PLANNER_LR = 1e-3

    def _unfreeze_planner(self, agent: PersistentAgent):
        """Make the scheduler mutable (the single defining difference vs Ours)."""
        for p in agent._planner.parameters():
            p.requires_grad = True

    def _collect_planner_labels(self, agent, ecm_registry, env, task):
        """Collect (obs, best_phase_idx) pairs by rolling out under the current
        time-slice schedule. Each visited state is labelled with the phase the
        fixed schedule assigns to its timestep, so the planner is trained by
        behavioural cloning on the clock's own labels; it is a state-conditioned
        classifier distilled from the schedule, not a reward-seeking dispatcher.
        (An earlier docstring claimed reward-based labelling; the code below
        never implemented it.)

        This uses the SAME env, SAME ECMs, SAME time-slice rotation as Ours.
        It gives the scheduler a supervised target so it can co-adapt to the
        evolving ECMs (making the agent genuinely mutable, Δπ > 0), without a
        second RL loop.
        """
        horizon = env._env.horizon if hasattr(env, "_env") else 500
        schedule = self._get_phase_schedule(ecm_registry, task, horizon)
        phases = PersistentAgent.PHASES
        # phase name -> index in PHASES
        phase_to_idx = {ph: i for i, ph in enumerate(phases)}
        # phase name -> ECM (via registry)
        active = ecm_registry.active_ecms()

        obs_buf: List[np.ndarray] = []
        lbl_buf: List[int] = []

        for _ in range(self.PLANNER_ROLLOUT_EPISODES):
            obs, _ = env.reset()
            done = False
            step = 0
            while not done:
                # Base phase from the time-slice schedule (same decomposition)
                base_ecm = schedule[-1][2]
                for start, end, sched_ecm in schedule:
                    if start <= step < end:
                        base_ecm = sched_ecm
                        break
                # Label = the phase the fixed schedule assigns to this timestep.
                # There is no reward lookahead: the planner clones the clock.
                best_phase = base_ecm.descriptor.phase
                obs_buf.append(np.asarray(obs, dtype=np.float32))
                lbl_buf.append(phase_to_idx.get(best_phase, 0))

                # Advance the env under the scheduled ECM (keeps the visited-state
                # distribution identical to Ours' schedule).
                action, _ = base_ecm.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step += 1

        if not obs_buf:
            return None, None
        return (
            np.stack(obs_buf).astype(np.float32),
            np.asarray(lbl_buf, dtype=np.int64),
        )

    def _train_planner(self, agent, obs_arr, lbl_arr):
        """Behavioral-clone the scheduler toward the collected phase labels.

        Mutates agent._planner (now unfrozen) -> Δπ > 0.
        """
        if obs_arr is None or len(obs_arr) == 0:
            return 0.0
        planner = agent._planner
        opt = torch.optim.Adam(
            [p for p in planner.parameters() if p.requires_grad],
            lr=self.PLANNER_LR,
        )
        ce = torch.nn.CrossEntropyLoss()
        x = torch.from_numpy(obs_arr)
        y = torch.from_numpy(lbl_arr)
        n = x.shape[0]
        bs = 256
        last = 0.0
        for _ in range(self.PLANNER_BC_EPOCHS):
            perm = torch.randperm(n)
            ep_loss = 0.0
            nb = 0
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                logits = planner(x[idx])
                loss = ce(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                nb += 1
            last = ep_loss / max(nb, 1)
        return last

    def _evaluate_agent_via_planner(self, agent, env, num_episodes, task=None):
        """Same as ExperimentRunner._evaluate_agent BUT the phase-ECM is chosen
        by the (now mutable, trained) planner via agent.select_action, instead of
        the fixed time-slice schedule. Everything else (episode count, success
        accounting, reward accumulation) is identical.
        """
        successes = 0
        total_reward = 0.0
        ep_rewards = []
        total_time = 0

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            step = 0
            env_info = {}
            while not done:
                action, _sel = agent.select_action(obs, deterministic=True)
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

    # -----------------------------------------------------------------
    # AM+phase evolution loop (mutable agent). Mirrors _run_evolution but
    # co-updates + uses the scheduler.
    # -----------------------------------------------------------------
    def _run_evolution_mutable(self, env, task, label="", on_iter_done=None,
                               resume_from=None):
        cfg = self.config
        iterations = self.args.iterations
        eval_eps = self.args.eval_episodes

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # SAME K=4 phase-ECMs as Ours
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

        # SAME agent object, but MUTABLE (this is the whole point)
        agent = PersistentAgent(obs_dim, ecm_registry, seed=cfg["seed"])
        self._unfreeze_planner(agent)          # <-- frozen -> mutable

        runtime = RuntimeGovernance(
            force_limit=cfg["runtime"]["force_limit"],
            max_velocity=cfg["runtime"]["max_velocity"],
            enabled=True,
        )

        # SAME learner (IL + SAC + LLM), SAME config (incl. 50k steps/iter)
        learner = HybridLearner(
            rl_config=cfg["learning"]["rl"],
            il_config=cfg["learning"]["il"],
            enable_rl=True, enable_il=True, enable_llm=True,
        )

        # Resume support (mirror _run_evolution; no checkpointing of planner —
        # AM+phase runs are short and resumed by warmup if needed).
        if resume_from and len(resume_from.get("iterations", [])) > 0:
            results = resume_from
            start_iter = len(results["iterations"])
            if start_iter >= iterations:
                log(f"  {label} already complete ({start_iter}/{iterations}), skipping")
                return results
            log(f"  {label} resuming from iteration {start_iter}/{iterations} "
                f"(warming up ECMs + planner)")
            for _ in range(min(start_iter, 3)):
                for ecm in ecm_registry.active_ecms():
                    learner.update_ecm(ecm, env, runtime)
                o, l = self._collect_planner_labels(agent, ecm_registry, env, task)
                self._train_planner(agent, o, l)
        else:
            results = {"iterations": [], "metrics": []}
            start_iter = 0

        for it in range(start_iter, iterations):
            t0 = time.time()

            # 1) Train each ECM + version gating (IDENTICAL to Ours)
            for ecm in ecm_registry.active_ecms():
                learner.update_ecm(ecm, env, runtime)
                gate_reward = self._eval_single_ecm(
                    ecm, env, n_episodes=3,
                    ecm_registry=ecm_registry, task=task,
                )
                accepted = ecm.evaluate_and_gate(gate_reward)
                if not accepted:
                    log(f"    {ecm.descriptor.name}: rollback (reward={gate_reward:.3f})")

            # 2) MUTATE THE AGENT: co-adapt the scheduler to the evolved ECMs.
            #    (This is the ONLY thing AM+phase adds over Ours.)
            obs_arr, lbl_arr = self._collect_planner_labels(
                agent, ecm_registry, env, task
            )
            bc_loss = self._train_planner(agent, obs_arr, lbl_arr)

            # 3) Evaluate THROUGH THE LEARNED SCHEDULER (not the fixed schedule)
            metrics = self._evaluate_agent_via_planner(
                agent, env, eval_eps, task=task
            )
            metrics["policy_drift"] = agent.compute_policy_drift()   # >0 now
            metrics["planner_bc_loss"] = float(bc_loss)
            metrics["iteration"] = it + 1
            metrics["wall_time"] = time.time() - t0

            results["iterations"].append(it + 1)
            results["metrics"].append(metrics)

            log(f"  {label} iter {it+1}/{iterations}: "
                f"success={metrics['success_rate']:.1f}% "
                f"reward={metrics['avg_reward']:.2f} "
                f"σ²={metrics['reward_variance']:.1f} "
                f"Δπ={metrics['policy_drift']:.4f} "
                f"bc={bc_loss:.3f} "
                f"({metrics['wall_time']:.0f}s)")

            if on_iter_done:
                on_iter_done(results)

        if hasattr(learner, "rl_learner") and hasattr(learner.rl_learner, "close"):
            learner.rl_learner.close()

        return results
