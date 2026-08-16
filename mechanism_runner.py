"""
T6 mechanism-analysis instrumented re-run driver.

Purpose
-------
An earlier version of the paper claimed that its multi-module configuration
(later established to be a time-sliced rotation of whole-task policies, not a
phase decomposition) drives long-horizon performance on T6 (TwoArmPegInHole),
without mechanistic evidence of WHERE the monolithic / skill-prior baselines
fail. This driver re-runs four
methods with instrumentation and produces, per (method, seed, iteration):

  (a) a per-phase PROGRESS LADDER over the 30-episode deployment eval,
  (b) DECISION CHURN at the ACTION level (each method's action-producing
      machinery) on a fixed probe-state set shared across methods of the
      same seed; ours/amphase additionally record planner-logit drift as a
      separate identity check (planner_churn),
  (c) optionally (AM only, --grad-proxy) a GRADIENT-INTERFERENCE proxy
      between early-phase and late-phase minibatches.

Placement
---------
Place this script in the capability-evolution repo root, NEXT TO
run_experiment.py (same convention as run_t6_amphase.py / amphase_runner.py).

DEPLOYMENT NOTE for --method amphase: this driver imports
amphase_runner.py at run time. Both mechanism_runner.py and
amphase_runner.py must sit in the repository root (next to
run_experiment.py) before invoking --method amphase; otherwise the amphase
arm fails at import time (clear RuntimeError below).

Training loops are NOT re-implemented. Each method reuses its own published
machinery:
  ours    -> run_experiment.ExperimentRunner._run_evolution (eval overridden
             in a subclass; training code untouched)
  amphase -> amphase_runner.ExperimentRunnerMutable._run_evolution_mutable
             (eval overridden in a subclass; training code untouched)
  am      -> baselines.agent_modification.AgentModificationBaseline
             (train_iteration() reused verbatim; its evaluate() is replaced
             by the shared instrumented eval with identical accounting)
  skimo   -> baselines.skimo_baseline.SkiMoBaseline (same pattern as am)

The instrumented eval is THE SAME function (instrumented_eval) for all four
methods: same episode count, same success accounting (final-step info
"success", matching ExperimentRunner._evaluate_agent and the baselines'
evaluate()), same reward statistics. Training budgets and configs are the
published ones: configs/default.yaml untouched, AM/SkiMo constructed with
their published defaults (env, seed) exactly as run_experiment.py does.

Progress ladder (constants quoted from the robosuite 1.4.1 env source,
robosuite/environments/manipulation/two_arm_peg_in_hole.py)
-----------------------------------------------------------
The underlying env exposes _compute_orientation() -> (t, d, cos):
  t   = parallel (axial) distance of the hole-plane center along the peg axis
  d   = perpendicular distance from the hole center line to the peg axis
  cos = |cos(angle between hole normal and peg axis)|

Env ground-truth constants:
  _check_success() (line 459):  d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95
  reward() shaping (lines 241-254): reaching 1 - tanh(dist(peg, hole)),
      orientation terms 1 - tanh(d), 1 - tanh(|t|), + cos

Derived ladder thresholds (all derived from the constants above):
  D_SUCCESS = 0.06      (the success perpendicular-distance limit)
  T_MIN, T_MAX = -0.12, 0.14   (the success axial-insertion range)
  COS_ALIGN = 0.95      (the success alignment threshold)
  D_APPROACH = 2 * D_SUCCESS = 0.12
      (approach threshold: within twice the success d-limit; the env source
       has no separate approach constant, so we state this relaxation
       explicitly rather than inventing an unrelated number)

Monotone per-episode ladder (max phase reached during the episode):
  P0  none
  P1  approach:  some step with d < D_APPROACH
  P2  align:     some step with cos > COS_ALIGN and d < D_APPROACH
  P3  partial insertion: some step where the full geometric success
      condition holds transiently (T_MIN <= t <= T_MAX and d < D_SUCCESS
      and cos > COS_ALIGN) -- i.e. the peg was in the hole at some point
  P4  full success: env success signal in info at the final step (this is
      exactly the episode-success definition used by the published eval)
  Monotonicity: P3 implies P2 implies P1 (thresholds nest); P4 implies P3
  because the wrapper computes success from the same geometry at the same
  post-step state we read.

Decision churn
--------------
A fixed probe set of N_PROBE=100 observation vectors per seed, generated
ONCE from a method-independent random-action rollout (seeded by the seed
only) and persisted to results/t6_mechanism/probes/probe_seed_<S>.npz so all
methods of the same seed share the same probe file. Per iteration we compute
ACTION-LEVEL outputs of each method's action-producing machinery, so the
cross-method churn comparison is apples-to-apples (no method is measured on
an object its eval path never uses):
  ours    -> stacked per-ECM deterministic actions on the probe set,
             ecm.get_action(obs, deterministic=True) for every active ECM
             -> (n_ecms * 100, A). The ECMs are the components that
             retrain + gate every iteration and the ones the fixed
             time-slice eval schedule actually executes.
  amphase -> same stacked per-ECM deterministic actions (its ECMs also
             retrain every iteration).
  am      -> monolithic PPO deterministic actions, model.predict -> (100, A).
  skimo   -> its skill-selection output: deterministic PPO action projected
             to the skill latent exactly as the baseline itself does in
             _train_world_model (actions[:, :skill_dim], zero-padded when
             act_dim < skill_dim; for T6 act_dim=14 < skill_dim=16 so this
             equals the full action zero-padded).
For ours/amphase the planner logits agent._planner(probe) -> (100, 4) are
ADDITIONALLY recorded as planner_churn / planner_output_hash. This is NOT
the cross-method churn: for ours the eval routes actions through the fixed
time-slice schedule (planner unused, requires_grad=False), so its
planner-logit drift 0 is true by construction and is reported only as the
pi_A frozen-identity check, consistent with policy_drift. For amphase the
planner is trained, so planner_churn tracks its BC co-adaptation.
Churn at iteration i (i >= 2):
  churn_i = ||O_i - O_(i-1)||_F / (||O_(i-1)||_F + 1e-12)
churn_1 is null (no predecessor), as is any iteration where the output
shape changed (e.g. the active-ECM set changed). We also store a
sha256-prefix hash of each raw float32 output matrix so any post-hoc
auditor can verify the matrices were distinct (or identical, for the
frozen planner logits).

Gradient-interference proxy (optional, AM only, flag --grad-proxy)
------------------------------------------------------------------
After each AM train_iteration() we read the PPO rollout buffer (read-only,
no monkey-patching), split transitions into early-phase (in-episode step
index < horizon/2) vs late-phase halves, compute the plain policy-gradient
surrogate loss -(advantage * log_prob).mean() on each subset via
policy.evaluate_actions, and report the cosine between the two gradient
vectors. This is a clean read-only implementation, so it is included rather
than skipped; it is still guarded defensively (SB3 internals may differ on
the box) and on any failure logs LOUDLY and records
grad_cosine_early_late = null with the reason. It never endangers (a)/(b).
Stated approximations: (i) the surrogate is the un-clipped policy gradient,
not the full clipped PPO loss (direction-level interference is what we
measure); (ii) steps before the first episode_starts flag in each env's
buffer block belong to an episode carried over from the previous rollout,
so their in-episode index is unknown -- they are EXCLUDED from both halves
rather than mislabeled early; (iii) the buffer holds only the last rollout
of the learn() call, so the proxy samples a subset of the iteration's
training data.

Resume semantics (SIMPLE, stated here as required)
--------------------------------------------------
Per (method, seed): if mechanism.json exists and holds >= --iterations
records, the run is SKIPPED. If a PARTIAL file exists, the run RESTARTS
FROM SCRATCH (the partial file is discarded) because the churn chain and
mechanism records must stay aligned with the training state; there is no
mid-run resume. Writes are crash-safe: the JSON is rewritten atomically
(tmp + os.replace) after every iteration.

Output schema
-------------
results/t6_mechanism/<method>/seed_<S>[__<tag>]/mechanism.json:
{
  "task": "T6_TwoArmPegInHole", "method": ..., "seed": ...,
  "ladder_constants": {"D_SUCCESS":..,"T_MIN":..,"T_MAX":..,"COS_ALIGN":..,
                        "D_APPROACH":..},
  "iterations": [1, 2, ...],
  "records": [
    {"iteration": i,
     "success_rate": pct, "avg_reward": f, "avg_time": f,
     "reward_variance": f, "policy_drift": f, "wall_time": f,
     "phase_histogram": {"P0": n, "P1": n, "P2": n, "P3": n, "P4": n},
     "episode_max_phase": [int x eval_episodes],
     "episode_min_d": [...], "episode_best_cos": [...],
     "transient_success_episodes": n,
     "churn": f | null, "probe_output_hash": hex16,
     "probe_output_shape": [N, K],
     "planner_churn": f | null,                 # ours/amphase only
     "planner_output_hash": hex16,              # ours/amphase only
     "planner_output_shape": [100, 4],          # ours/amphase only
     "grad_cosine_early_late": f | null,        # am + --grad-proxy only
     "grad_proxy_note": str | absent}
  ],
  "_meta": {...}
}
results/t6_mechanism/summary_mechanism[__<tag>].json / .md
    (via --aggregate or end-of-run; tagged runs write tag-suffixed summary
     files so a smoke aggregate can never overwrite the production summary)
results/t6_mechanism/probes/probe_seed_<S>.npz     (shared probe states)

Example commands
----------------
Defaults (--seeds 42 7 123 2024 --iterations 20) match the published T6
budgets (run_t6_amphase.py / run_t6_multiseed.py); do not lower them for
production runs.
  python mechanism_runner.py --method am --seeds 42 --iterations 2 --tag smoke
  python mechanism_runner.py --method ours
  python mechanism_runner.py --method amphase
  python mechanism_runner.py --method am      --grad-proxy
  python mechanism_runner.py --method skimo
  python mechanism_runner.py --aggregate
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)

from run_experiment import ExperimentRunner, load_config, log  # noqa: E402
from baselines import AgentModificationBaseline, SkiMoBaseline  # noqa: E402

TASK_NAME = "T6_TwoArmPegInHole"
SUBDIR = "t6_mechanism"
METHODS = ("ours", "am", "skimo", "amphase")

# ---------------------------------------------------------------------------
# Ladder constants -- quoted from robosuite 1.4.1
# environments/manipulation/two_arm_peg_in_hole.py:
#   _check_success():  "return d < 0.06 and -0.12 <= t <= 0.14 and cos > 0.95"
# D_APPROACH is a stated 2x relaxation of the success d-limit (the env source
# defines no separate approach constant).
# ---------------------------------------------------------------------------
D_SUCCESS = 0.06
T_MIN = -0.12
T_MAX = 0.14
COS_ALIGN = 0.95
D_APPROACH = 2.0 * D_SUCCESS  # = 0.12

LADDER_CONSTANTS = {
    "D_SUCCESS": D_SUCCESS,
    "T_MIN": T_MIN,
    "T_MAX": T_MAX,
    "COS_ALIGN": COS_ALIGN,
    "D_APPROACH": D_APPROACH,
}

N_PROBE = 100
PROBE_EPISODES = 3  # random-action episodes used to harvest candidate states


def seed_everything(seed: int):
    """Same convention as run_t6_amphase.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Instrumentation plumbing
# ---------------------------------------------------------------------------

def get_robosuite_env(env):
    """Return the underlying robosuite env from the Gymnasium wrapper.

    RobosuiteWrapper stores it as self._env (verified in
    envs/robosuite_wrapper.py, __init__: "self._env = suite.make(...)").
    The measurement is core to this driver, so a missing hook is a HARD
    error, never a silent skip.
    """
    rs = getattr(env, "_env", None)
    if rs is None:
        raise RuntimeError(
            "Instrumentation hook missing: wrapper has no attribute '_env'. "
            "Cannot read _compute_orientation(); refusing to run without the "
            "phase-ladder measurement."
        )
    if not hasattr(rs, "_compute_orientation"):
        raise RuntimeError(
            f"Instrumentation hook missing: underlying env "
            f"{type(rs).__name__} has no _compute_orientation(). This driver "
            f"only supports {TASK_NAME} (TwoArmPegInHole). Refusing to run."
        )
    return rs


def instrumented_eval(env, act_fn: Callable[[np.ndarray, int], np.ndarray],
                      num_episodes: int) -> Tuple[Dict, Dict]:
    """Shared instrumented deployment eval, identical for ALL methods.

    Mirrors ExperimentRunner._evaluate_agent / the baselines' evaluate()
    accounting exactly: per-episode reward sum, success from the final
    step's info["success"], success_rate in percent, reward variance over
    episodes. Adds per-step reads of _compute_orientation() to build the
    per-episode progress ladder.

    act_fn(obs, step) -> action  chooses the method's action at this step.
    """
    rs = get_robosuite_env(env)

    successes = 0
    total_reward = 0.0
    ep_rewards: List[float] = []
    total_time = 0

    ep_max_phase: List[int] = []
    ep_min_d: List[float] = []
    ep_best_cos: List[float] = []
    transient_success_eps = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        env_info: Dict = {}
        min_d = float("inf")
        best_cos = -1.0
        p1 = p2 = p3 = False
        any_step_success = False

        while not done:
            action = act_fn(obs, step)
            obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1

            # Post-step geometry from the env's own helper.
            t, d, cos = rs._compute_orientation()
            t = float(t)
            d = float(d)
            cos = float(cos)
            min_d = min(min_d, d)
            best_cos = max(best_cos, cos)
            if d < D_APPROACH:
                p1 = True
                if cos > COS_ALIGN:
                    p2 = True
            if (T_MIN <= t <= T_MAX) and d < D_SUCCESS and cos > COS_ALIGN:
                p3 = True
            if env_info.get("success", False):
                any_step_success = True

        ep_success = bool(env_info.get("success", False))
        if ep_success:
            successes += 1
        if any_step_success and not ep_success:
            transient_success_eps += 1

        phase = 4 if ep_success else (3 if p3 else (2 if p2 else (1 if p1 else 0)))
        ep_max_phase.append(phase)
        ep_min_d.append(min_d if np.isfinite(min_d) else -1.0)
        ep_best_cos.append(best_cos)
        ep_rewards.append(ep_reward)
        total_reward += ep_reward
        total_time += step

    histogram = {f"P{p}": int(sum(1 for m in ep_max_phase if m == p))
                 for p in range(5)}

    metrics = {
        "success_rate": successes / num_episodes * 100,
        "avg_reward": total_reward / num_episodes,
        "avg_time": total_time / num_episodes,
        "reward_variance": float(np.var(ep_rewards)),
    }
    mech = {
        "phase_histogram": histogram,
        "episode_max_phase": ep_max_phase,
        "episode_min_d": [round(v, 5) for v in ep_min_d],
        "episode_best_cos": [round(v, 5) for v in ep_best_cos],
        "transient_success_episodes": transient_success_eps,
    }
    return metrics, mech


# ---------------------------------------------------------------------------
# Probe set (decision churn support)
# ---------------------------------------------------------------------------

def probe_path_for(seed: int) -> str:
    d = os.path.join(EXP_DIR, "results", SUBDIR, "probes")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"probe_seed_{seed}.npz")


def load_or_build_probe(seed: int, runner: ExperimentRunner, task) -> np.ndarray:
    """Load the per-seed probe matrix; generate it once if absent.

    Generation is METHOD-INDEPENDENT: a dedicated env instance is rolled
    out with uniform random actions drawn from np.random.default_rng(seed);
    observations are harvested across PROBE_EPISODES episodes and N_PROBE
    of them are sampled uniformly (rng seeded by seed+1). The matrix is
    persisted so every method of the same seed shares the same file.
    NOTE: callers must re-seed after this call (see run_method_seed) so the
    training RNG stream is identical whether the probe was generated here
    or loaded from disk.
    """
    path = probe_path_for(seed)
    if os.path.exists(path):
        data = np.load(path)
        probe = data["obs"].astype(np.float32)
        if probe.shape[0] != N_PROBE:
            raise RuntimeError(
                f"Probe file {path} holds {probe.shape[0]} states, expected "
                f"{N_PROBE}. Delete it to regenerate."
            )
        log(f"  probe: loaded {probe.shape} from {path}")
        return probe

    log(f"  probe: generating {N_PROBE} states (seed={seed}, "
        f"{PROBE_EPISODES} random-action episodes)")
    env = runner.task_suite.make_env(task)
    try:
        rng = np.random.default_rng(seed)
        act_dim = env.action_space.shape[0]
        all_obs: List[np.ndarray] = []
        obs, _ = env.reset(seed=seed)
        for _ in range(PROBE_EPISODES):
            done = False
            while not done:
                all_obs.append(np.asarray(obs, dtype=np.float32))
                action = rng.uniform(-1.0, 1.0, size=act_dim)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            obs, _ = env.reset()
    finally:
        try:
            env.close()
        except Exception:
            pass

    if len(all_obs) < N_PROBE:
        raise RuntimeError(
            f"Probe rollout produced only {len(all_obs)} states "
            f"(< {N_PROBE}); increase PROBE_EPISODES."
        )
    pick = np.random.default_rng(seed + 1).choice(
        len(all_obs), size=N_PROBE, replace=False
    )
    probe = np.stack([all_obs[i] for i in sorted(pick)]).astype(np.float32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, obs=probe,
                        meta=np.array([seed, len(all_obs)], dtype=np.int64))
    os.replace(tmp, path)
    log(f"  probe: saved {probe.shape} -> {path}")
    return probe


def matrix_hash(m: np.ndarray) -> str:
    b = np.ascontiguousarray(m.astype(np.float32)).tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def decision_churn(prev: Optional[np.ndarray],
                   cur: np.ndarray) -> Optional[float]:
    """Normalized L2 (Frobenius) drift vs the previous iteration's outputs."""
    if prev is None or prev.shape != cur.shape:
        return None
    denom = float(np.linalg.norm(prev)) + 1e-12
    return float(np.linalg.norm(cur - prev) / denom)


def planner_probe_outputs(agent, probe: np.ndarray) -> np.ndarray:
    """Planner logits for ours/amphase (agent._planner, the phase-selection
    network from agent/persistent_agent.py line 48).

    NOT the cross-method churn quantity. For ours the eval path routes
    actions through the fixed time-slice schedule + ECMs (planner unused,
    requires_grad=False), so planner-logit churn 0 is true by construction;
    it is recorded separately as the pi_A frozen-identity check
    (planner_churn), consistent with policy_drift. The cross-method churn
    comparison uses ecm_action_probe_outputs() instead.
    """
    with torch.no_grad():
        out = agent._planner(torch.as_tensor(probe, dtype=torch.float32))
    return out.cpu().numpy().astype(np.float32)


def ecm_action_probe_outputs(ecm_registry, probe: np.ndarray) -> np.ndarray:
    """Composed ACTION-LEVEL decision outputs for ours/amphase.

    For each active ECM (the components that actually change across
    iterations: retrained + gated), the deterministic action on every probe
    state via the published ecm.get_action(obs, deterministic=True)
    (agent/ecm.py), stacked over ECMs into (n_ecms * N_PROBE, act_dim).
    This is the same level of measurement as AM/SkiMo churn (deterministic
    actions of the action-producing policy), so the cross-method churn
    comparison is apples-to-apples. decision_churn() returns None on a
    shape change (e.g. the active-ECM set changed), never a fake number.
    """
    blocks = []
    for ecm in ecm_registry.active_ecms():
        acts = [np.asarray(ecm.get_action(obs, deterministic=True)[0],
                           dtype=np.float32)
                for obs in probe]
        blocks.append(np.stack(acts))
    return np.concatenate(blocks, axis=0).astype(np.float32)


def ppo_probe_outputs(model, probe: np.ndarray) -> np.ndarray:
    """Decision-maker outputs for AM: deterministic PPO actions (batched)."""
    actions, _ = model.predict(probe, deterministic=True)
    return np.asarray(actions, dtype=np.float32)


def skimo_probe_outputs(baseline: SkiMoBaseline, probe: np.ndarray) -> np.ndarray:
    """SkiMo decision-maker outputs: skill-latent projection of the
    deterministic PPO action, exactly the baseline's own convention
    (skimo_baseline.py _train_world_model lines 183-187: actions padded or
    truncated to skill_dim)."""
    actions, _ = baseline.model.predict(probe, deterministic=True)
    actions = np.asarray(actions, dtype=np.float32)
    k = baseline.skill_dim
    if actions.shape[1] < k:
        pad = np.zeros((actions.shape[0], k - actions.shape[1]),
                       dtype=np.float32)
        return np.concatenate([actions, pad], axis=1)
    return actions[:, :k]


# ---------------------------------------------------------------------------
# Gradient-interference proxy (AM only, optional)
# ---------------------------------------------------------------------------

def _subset_actor_grad(model, obs_arr, act_arr, adv_arr,
                       max_rows: int = 4096) -> np.ndarray:
    """Flattened gradient of the plain policy-gradient surrogate
    -(advantage * log_prob).mean() over one transition subset."""
    if obs_arr.shape[0] > max_rows:
        idx = np.random.default_rng(0).choice(
            obs_arr.shape[0], size=max_rows, replace=False
        )
        obs_arr, act_arr, adv_arr = obs_arr[idx], act_arr[idx], adv_arr[idx]

    policy = model.policy
    policy.set_training_mode(False)
    obs_t, _ = policy.obs_to_tensor(obs_arr)
    act_t = torch.as_tensor(act_arr, dtype=torch.float32, device=model.device)
    adv_t = torch.as_tensor(adv_arr, dtype=torch.float32,
                            device=model.device).flatten()
    _, log_prob, _ = policy.evaluate_actions(obs_t, act_t)
    loss = -(adv_t * log_prob).mean()
    params = [p for p in policy.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    flat = []
    for p, g in zip(params, grads):
        flat.append((torch.zeros_like(p) if g is None else g)
                    .detach().cpu().numpy().ravel())
    return np.concatenate(flat)


def gradient_interference_proxy(model, horizon: int) -> Tuple[Optional[float], str]:
    """Cosine between actor gradients on early-phase vs late-phase
    transitions from the PPO rollout buffer left populated by the most
    recent learn() call. Read-only; any deviation from the expected SB3
    buffer layout is logged loudly and returns (None, reason)."""
    try:
        buf = model.rollout_buffer
        n_steps = int(buf.buffer_size)
        n_envs = int(buf.n_envs)
        obs = np.asarray(buf.observations, dtype=np.float32)
        acts = np.asarray(buf.actions, dtype=np.float32)
        adv = np.asarray(buf.advantages, dtype=np.float32)
        starts = np.asarray(buf.episode_starts, dtype=np.float32)

        # Normalize to env-major flat layout (n_envs * n_steps, ...).
        # SB3 stores (n_steps, n_envs, ...) until get() flattens via
        # swap_and_flatten (env-major). Handle both states.
        def to_env_major(a):
            if a.ndim >= 2 and a.shape[0] == n_steps and a.shape[1] == n_envs:
                return a.swapaxes(0, 1).reshape(n_envs * n_steps, -1)
            if a.shape[0] == n_steps * n_envs:
                return a.reshape(n_envs * n_steps, -1)
            raise ValueError(f"unexpected buffer shape {a.shape}")

        obs_f = to_env_major(obs)
        acts_f = to_env_major(acts)
        adv_f = to_env_major(adv).ravel()
        starts_f = to_env_major(starts).ravel()

        # In-episode step index, resetting at episode starts, per env block.
        # Steps BEFORE the first episode_starts==True in each env block
        # belong to an episode carried over from the previous rollout
        # (horizon 500 vs n_steps 2048), so their true in-episode index is
        # unknown; they are excluded from both halves rather than
        # mislabeled "early". Note the buffer holds only the LAST rollout
        # of the learn() call, so the proxy samples a subset of the
        # iteration's training data (stated approximation, like the
        # un-clipped surrogate above).
        in_ep = np.zeros(n_envs * n_steps, dtype=np.int64)
        known = np.zeros(n_envs * n_steps, dtype=bool)
        for e in range(n_envs):
            c = 0
            seen_start = False
            base = e * n_steps
            for s in range(n_steps):
                if starts_f[base + s] > 0.5:
                    c = 0
                    seen_start = True
                in_ep[base + s] = c
                known[base + s] = seen_start
                c += 1

        early = known & (in_ep < (horizon // 2))
        late = known & (in_ep >= (horizon // 2))
        if early.sum() < 32 or late.sum() < 32:
            reason = (f"insufficient samples for grad proxy "
                      f"(early={int(early.sum())}, late={int(late.sum())})")
            log(f"  [grad-proxy] SKIPPED: {reason}")
            return None, reason

        g_early = _subset_actor_grad(model, obs_f[early], acts_f[early],
                                     adv_f[early])
        g_late = _subset_actor_grad(model, obs_f[late], acts_f[late],
                                    adv_f[late])
        denom = (np.linalg.norm(g_early) * np.linalg.norm(g_late))
        if denom < 1e-12:
            reason = "degenerate gradient norms"
            log(f"  [grad-proxy] SKIPPED: {reason}")
            return None, reason
        cosine = float(np.dot(g_early, g_late) / denom)
        return cosine, "ok"
    except Exception as e:  # defensive: SB3 internals may differ on the box
        reason = f"{type(e).__name__}: {e}"
        log(f"  [grad-proxy] FAILED (recorded as null, run continues): {reason}")
        return None, reason


# ---------------------------------------------------------------------------
# Result file plumbing (crash-safe, restart-if-partial)
# ---------------------------------------------------------------------------

def seed_dir(method: str, seed: int, tag: str) -> str:
    name = f"seed_{seed}" + (f"__{tag}" if tag else "")
    d = os.path.join(EXP_DIR, "results", SUBDIR, method, name)
    os.makedirs(d, exist_ok=True)
    return d


def result_path(method: str, seed: int, tag: str) -> str:
    return os.path.join(seed_dir(method, seed, tag), "mechanism.json")


def atomic_write_json(path: str, data: Dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def check_skip_or_restart(method: str, seed: int, tag: str,
                          iterations: int) -> bool:
    """Return True if this (method, seed) is complete and must be skipped.
    A partial file is discarded (restart-from-scratch semantics)."""
    p = result_path(method, seed, tag)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        n = len(data.get("records", []))
    except Exception:
        n = 0
    if n >= iterations:
        log(f"[{method}/seed_{seed}] already complete ({n}/{iterations}), skipping")
        return True
    log(f"[{method}/seed_{seed}] PARTIAL file found ({n}/{iterations}); "
        f"restarting from scratch (no mid-run resume, see header)")
    os.remove(p)
    return False


def make_doc(method: str, seed: int, args, records: List[Dict],
             extra_meta: Optional[Dict] = None) -> Dict:
    doc = {
        "task": TASK_NAME,
        "method": method,
        "seed": seed,
        "ladder_constants": LADDER_CONSTANTS,
        "iterations": [r["iteration"] for r in records],
        "records": records,
        "_meta": {
            "iterations_target": args.iterations,
            "eval_episodes": args.eval_episodes,
            "n_probe": N_PROBE,
            "tag": args.tag,
            "grad_proxy": bool(getattr(args, "grad_proxy", False)),
            "written_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    if extra_meta:
        doc["_meta"].update(extra_meta)
    return doc


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

class OursMechRunner(ExperimentRunner):
    """ExperimentRunner with ONLY the deployment eval overridden.

    Training (_run_evolution's learner.update_ecm loop, gating, budgets)
    is inherited verbatim. The override reproduces _evaluate_agent's
    phase-schedule action selection exactly (same _get_phase_schedule,
    same deterministic ECM actions) while adding instrumentation.

    Churn accounting: "churn" is the ACTION-LEVEL churn of the stacked
    per-ECM deterministic actions (the components that actually retrain
    each iteration), comparable with AM/SkiMo. The frozen planner's logit
    drift is recorded separately as "planner_churn" (pi_A identity check;
    0 by construction since the planner has requires_grad=False and the
    eval routes through the fixed time-slice schedule, but still measured).
    """

    def __init__(self, config, args, probe: np.ndarray):
        super().__init__(config, args)
        self._mech_probe = probe
        self._mech_records: List[Dict] = []
        self._mech_prev_out: Optional[np.ndarray] = None
        self._mech_prev_planner: Optional[np.ndarray] = None

    def _evaluate_agent(self, agent, env, num_episodes, task=None):
        horizon = env._env.horizon if hasattr(env, "_env") else 500
        schedule = self._get_phase_schedule(agent.ecm_registry, task, horizon)

        def act_fn(obs, step):
            current_ecm = schedule[-1][2]
            for start, end, ecm in schedule:
                if start <= step < end:
                    current_ecm = ecm
                    break
            action, _ = current_ecm.get_action(obs, deterministic=True)
            return action

        metrics, mech = instrumented_eval(env, act_fn, num_episodes)

        out = ecm_action_probe_outputs(agent.ecm_registry, self._mech_probe)
        mech["churn"] = decision_churn(self._mech_prev_out, out)
        mech["probe_output_hash"] = matrix_hash(out)
        mech["probe_output_shape"] = list(out.shape)
        self._mech_prev_out = out

        pl = planner_probe_outputs(agent, self._mech_probe)
        mech["planner_churn"] = decision_churn(self._mech_prev_planner, pl)
        mech["planner_output_hash"] = matrix_hash(pl)
        mech["planner_output_shape"] = list(pl.shape)
        self._mech_prev_planner = pl

        self._mech_records.append(mech)
        return metrics


def _make_amphase_runner_class():
    """Lazy import: amphase_runner.py must sit next to this script."""
    try:
        from amphase_runner import ExperimentRunnerMutable
    except ImportError as e:
        raise RuntimeError(
            "--method amphase requires amphase_runner.py in the repo root "
            f"next to this script (import failed: {e})"
        )

    class AmphaseMechRunner(ExperimentRunnerMutable):
        """ExperimentRunnerMutable with ONLY the planner-routed eval
        overridden. Training (_run_evolution_mutable: ECM updates, planner
        BC co-adaptation) is inherited verbatim. The override reproduces
        _evaluate_agent_via_planner's action selection exactly
        (agent.select_action, deterministic) while adding instrumentation.

        Churn accounting mirrors OursMechRunner: "churn" is the
        action-level churn of the stacked per-ECM deterministic actions
        (cross-method comparable); the trained planner's logit drift is
        recorded separately as "planner_churn"."""

        def __init__(self, config, args, probe: np.ndarray):
            super().__init__(config, args)
            self._mech_probe = probe
            self._mech_records: List[Dict] = []
            self._mech_prev_out: Optional[np.ndarray] = None
            self._mech_prev_planner: Optional[np.ndarray] = None

        def _evaluate_agent_via_planner(self, agent, env, num_episodes,
                                        task=None):
            def act_fn(obs, step):
                action, _sel = agent.select_action(obs, deterministic=True)
                return action

            metrics, mech = instrumented_eval(env, act_fn, num_episodes)

            out = ecm_action_probe_outputs(agent.ecm_registry,
                                           self._mech_probe)
            mech["churn"] = decision_churn(self._mech_prev_out, out)
            mech["probe_output_hash"] = matrix_hash(out)
            mech["probe_output_shape"] = list(out.shape)
            self._mech_prev_out = out

            pl = planner_probe_outputs(agent, self._mech_probe)
            mech["planner_churn"] = decision_churn(self._mech_prev_planner,
                                                   pl)
            mech["planner_output_hash"] = matrix_hash(pl)
            mech["planner_output_shape"] = list(pl.shape)
            self._mech_prev_planner = pl

            self._mech_records.append(mech)
            return metrics

    return AmphaseMechRunner


def _merged_records(partial: Dict, mech_records: List[Dict]) -> List[Dict]:
    """Zip the training loop's metrics list with the mech records collected
    by the overridden eval. Alignment is asserted loudly: every iteration
    produces exactly one _evaluate_agent call in both loops."""
    metrics = partial.get("metrics", [])
    if len(metrics) != len(mech_records):
        raise RuntimeError(
            f"mechanism records out of sync with training metrics "
            f"({len(mech_records)} vs {len(metrics)}); refusing to write "
            f"a misaligned file"
        )
    merged = []
    for m, r in zip(metrics, mech_records):
        rec = dict(m)
        rec.update(r)
        merged.append(rec)
    return merged


def run_evolution_method(method: str, seed: int, args):
    """ours / amphase: reuse the published evolution loops with the eval
    subclass-overridden; per-iteration atomic saves via on_iter_done."""
    if check_skip_or_restart(method, seed, args.tag, args.iterations):
        return
    out_path = result_path(method, seed, args.tag)

    seed_everything(seed)
    config = load_config(args.config)
    config["seed"] = seed
    ns = argparse.Namespace(
        exp="exp2", task=TASK_NAME, iterations=args.iterations,
        eval_episodes=args.eval_episodes, seed=seed, config=args.config,
    )

    if method == "ours":
        runner_cls = OursMechRunner
    else:
        runner_cls = _make_amphase_runner_class()

    # Probe first (method-independent), then RE-SEED so the training RNG
    # stream is identical whether the probe was generated or loaded.
    probe_runner = ExperimentRunner(config, ns)
    task = probe_runner.task_suite.get_task(TASK_NAME)
    probe = load_or_build_probe(seed, probe_runner, task)
    seed_everything(seed)

    runner = runner_cls(config, ns, probe)
    task = runner.task_suite.get_task(TASK_NAME)
    env = runner.task_suite.make_env(task)
    try:
        get_robosuite_env(env)  # hard-fail early if instrumentation missing

        def on_iter(partial):
            records = _merged_records(partial, runner._mech_records)
            atomic_write_json(out_path, make_doc(method, seed, args, records))

        label = f"{SUBDIR}/{method}/seed_{seed}" + (
            f"__{args.tag}" if args.tag else "")
        t0 = time.time()
        if method == "ours":
            result = runner._run_evolution(
                env, task, label=label, on_iter_done=on_iter, resume_from=None,
            )
        else:
            result = runner._run_evolution_mutable(
                env, task, label=label, on_iter_done=on_iter, resume_from=None,
            )

        records = _merged_records(result, runner._mech_records)
        atomic_write_json(out_path, make_doc(
            method, seed, args, records,
            extra_meta={"wall_seconds": time.time() - t0,
                        "finished": True},
        ))
        log(f"[{method}/seed_{seed}] done in {time.time() - t0:.0f}s -> {out_path}")
    finally:
        try:
            env.close()
        except Exception:
            pass


def run_baseline_method(method: str, seed: int, args):
    """am / skimo: reuse the baseline's own train_iteration() verbatim
    (published defaults, exactly as run_experiment.py constructs them:
    Baseline(env, seed=config['seed'])); the published evaluate() is
    replaced by the SAME instrumented eval used for ours/amphase."""
    if check_skip_or_restart(method, seed, args.tag, args.iterations):
        return
    out_path = result_path(method, seed, args.tag)

    seed_everything(seed)
    config = load_config(args.config)
    config["seed"] = seed
    ns = argparse.Namespace(
        exp="exp2", task=TASK_NAME, iterations=args.iterations,
        eval_episodes=args.eval_episodes, seed=seed, config=args.config,
    )
    runner = ExperimentRunner(config, ns)
    task = runner.task_suite.get_task(TASK_NAME)

    probe = load_or_build_probe(seed, runner, task)
    seed_everything(seed)

    env = runner.task_suite.make_env(task)
    baseline = None
    try:
        rs = get_robosuite_env(env)
        horizon = int(getattr(rs, "horizon", 500))

        if method == "am":
            baseline = AgentModificationBaseline(env, seed=config["seed"])
            out_fn = lambda: ppo_probe_outputs(baseline.model, probe)  # noqa: E731
        else:
            baseline = SkiMoBaseline(env, seed=config["seed"])
            out_fn = lambda: skimo_probe_outputs(baseline, probe)  # noqa: E731

        def act_fn(obs, step):
            action, _ = baseline.model.predict(obs, deterministic=True)
            return action

        records: List[Dict] = []
        prev_out: Optional[np.ndarray] = None
        t_start = time.time()

        for it in range(args.iterations):
            t0 = time.time()
            baseline.train_iteration()

            metrics, mech = instrumented_eval(env, act_fn, args.eval_episodes)

            out = out_fn()
            rec = dict(metrics)
            rec.update(mech)
            rec["churn"] = decision_churn(prev_out, out)
            rec["probe_output_hash"] = matrix_hash(out)
            rec["probe_output_shape"] = list(out.shape)
            prev_out = out

            rec["policy_drift"] = baseline.compute_policy_drift()

            if method == "am" and args.grad_proxy:
                cosine, note = gradient_interference_proxy(
                    baseline.model, horizon)
                rec["grad_cosine_early_late"] = cosine
                if note != "ok":
                    rec["grad_proxy_note"] = note

            rec["iteration"] = it + 1
            rec["wall_time"] = time.time() - t0
            records.append(rec)

            log(f"  [{method}/seed_{seed}] iter {it + 1}/{args.iterations}: "
                f"success={rec['success_rate']:.1f}% "
                f"reward={rec['avg_reward']:.2f} "
                f"phases={rec['phase_histogram']} "
                f"churn={rec['churn'] if rec['churn'] is not None else 'n/a'} "
                f"({rec['wall_time']:.0f}s)")

            atomic_write_json(out_path, make_doc(method, seed, args, records))

        atomic_write_json(out_path, make_doc(
            method, seed, args, records,
            extra_meta={"wall_seconds": time.time() - t_start,
                        "finished": True},
        ))
        log(f"[{method}/seed_{seed}] done -> {out_path}")
    finally:
        if baseline is not None and hasattr(baseline, "close"):
            try:
                baseline.close()
            except Exception:
                pass
        # SkiMoBaseline has no close(); release its vec_env explicitly.
        if baseline is not None and not hasattr(baseline, "close") \
                and hasattr(baseline, "vec_env"):
            try:
                baseline.vec_env.close()
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _band_of(idx: int, n: int) -> str:
    """Map a 0-based iteration index to one of 3 bands (early/mid/late)."""
    third = max(n // 3, 1)
    if idx < third:
        return "early"
    if idx < 2 * third:
        return "mid"
    return "late"


def aggregate_and_write(tag: str) -> Dict:
    base = os.path.join(EXP_DIR, "results", SUBDIR)
    per_method: Dict[str, Dict[int, Dict]] = {}
    for method in METHODS:
        mdir = os.path.join(base, method)
        if not os.path.isdir(mdir):
            continue
        for entry in sorted(os.listdir(mdir)):
            if not entry.startswith("seed_"):
                continue
            has_tag = "__" in entry
            if tag and not entry.endswith(f"__{tag}"):
                continue
            if not tag and has_tag:
                continue  # untagged aggregate ignores tagged (smoke) runs
            fp = os.path.join(mdir, entry, "mechanism.json")
            if not os.path.exists(fp):
                continue
            try:
                with open(fp) as f:
                    data = json.load(f)
                s = int(entry.split("__")[0].split("_", 1)[1])
            except Exception as e:
                log(f"  [aggregate] skipping unreadable {fp}: {e}")
                continue
            per_method.setdefault(method, {})[s] = data

    summary: Dict = {
        "task": TASK_NAME,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "ladder_constants": LADDER_CONSTANTS,
        "methods": {},
    }

    for method, seeds in sorted(per_method.items()):
        n_iters = max(len(d.get("records", [])) for d in seeds.values())
        per_iter_frac: List[Dict] = []
        churn_curve: List[Optional[float]] = []
        band_counts: Dict[str, Dict[str, float]] = {}
        for b in ("early", "mid", "late"):
            band_counts[b] = {f"P{p}": 0.0 for p in range(5)}
            band_counts[b]["_eps"] = 0.0
        for i in range(n_iters):
            fr_at_least = {f"P{p}": [] for p in range(1, 5)}
            fr_exact = {f"P{p}": [] for p in range(5)}
            churns = []
            for d in seeds.values():
                recs = d.get("records", [])
                if i >= len(recs):
                    continue
                r = recs[i]
                hist = r.get("phase_histogram", {})
                n_eps = sum(hist.values()) or 1
                for p in range(5):
                    fr_exact[f"P{p}"].append(hist.get(f"P{p}", 0) / n_eps)
                for p in range(1, 5):
                    reached = sum(hist.get(f"P{q}", 0) for q in range(p, 5))
                    fr_at_least[f"P{p}"].append(reached / n_eps)
                if r.get("churn") is not None:
                    churns.append(float(r["churn"]))
                b = _band_of(i, n_iters)
                for p in range(5):
                    band_counts[b][f"P{p}"] += hist.get(f"P{p}", 0)
                band_counts[b]["_eps"] += n_eps
            per_iter_frac.append({
                "iteration": i + 1,
                "frac_exact": {k: (float(np.mean(v)) if v else None)
                               for k, v in fr_exact.items()},
                "frac_at_least": {k: (float(np.mean(v)) if v else None)
                                  for k, v in fr_at_least.items()},
            })
            churn_curve.append(float(np.mean(churns)) if churns else None)

        band_frac = {}
        for b, c in band_counts.items():
            eps = c.pop("_eps")
            band_frac[b] = ({k: v / eps for k, v in c.items()}
                            if eps > 0 else None)

        grad_cos = []
        for d in seeds.values():
            vals = [r["grad_cosine_early_late"] for r in d.get("records", [])
                    if r.get("grad_cosine_early_late") is not None]
            if vals:
                grad_cos.append({"n": len(vals),
                                 "mean": float(np.mean(vals)),
                                 "min": float(np.min(vals))})

        summary["methods"][method] = {
            "seeds": sorted(seeds.keys()),
            "per_iteration": per_iter_frac,
            "churn_curve": churn_curve,
            "band_phase_fractions": band_frac,
            "grad_cosine_per_seed": grad_cos or None,
        }

    # Tagged runs get tag-suffixed summary files so a smoke aggregate can
    # never overwrite the production summary_mechanism.{json,md}.
    suffix = f"__{tag}" if tag else ""
    out_json = os.path.join(base, f"summary_mechanism{suffix}.json")
    atomic_write_json(out_json, summary)

    # Markdown companion
    lines = ["# T6 Mechanism Analysis Summary",
             "",
             f"Task: `{TASK_NAME}`  Generated: {summary['generated_at']}"
             + (f"  Tag: `{tag}`" if tag else ""),
             "",
             "Ladder: P1 approach (d < {:.2f}), P2 align (cos > {:.2f} at "
             "d < {:.2f}), P3 partial insertion (success geometry held "
             "transiently), P4 episode success.".format(
                 D_APPROACH, COS_ALIGN, D_APPROACH),
             ""]
    lines.append("## Phase-ladder fractions by iteration band "
                 "(episodes reaching exactly Pk, pooled over seeds)")
    lines.append("")
    lines.append("| Method | Band | P0 | P1 | P2 | P3 | P4 |")
    lines.append("|---|---|---|---|---|---|---|")
    for method, m in sorted(summary["methods"].items()):
        for b in ("early", "mid", "late"):
            bf = m["band_phase_fractions"].get(b)
            if bf is None:
                continue
            lines.append(
                "| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |"
                .format(method, b, *[bf[f"P{p}"] for p in range(5)]))
    lines.append("")
    lines.append("## Decision churn (mean over seeds, per iteration)")
    lines.append("")
    header = "| Method | " + " | ".join(
        f"it{i + 1}" for i in range(max(
            (len(m["churn_curve"]) for m in summary["methods"].values()),
            default=0))) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (header.count("|") - 1))
    for method, m in sorted(summary["methods"].items()):
        cells = [("-" if c is None else f"{c:.4f}") for c in m["churn_curve"]]
        pad = header.count("|") - 2 - len(cells)
        cells += ["-"] * max(pad, 0)
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    for method, m in sorted(summary["methods"].items()):
        if m["grad_cosine_per_seed"]:
            lines.append("")
            lines.append(f"## Gradient-interference proxy ({method})")
            for g in m["grad_cosine_per_seed"]:
                lines.append(f"- n={g['n']} iterations: mean cosine "
                             f"{g['mean']:.3f}, min {g['min']:.3f}")

    out_md = os.path.join(base, f"summary_mechanism{suffix}.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"Aggregate written: {out_json}")
    log(f"Markdown summary:  {out_md}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="T6 mechanism-analysis instrumented re-run driver "
                    "(phase ladder + decision churn + optional grad proxy)")
    parser.add_argument("--method", choices=list(METHODS), default=None,
                        help="Which method to run (required unless --aggregate)")
    # Defaults MUST match the published T6 budgets: run_t6_amphase.py uses
    # seeds [42, 7, 123, 2024] with --iterations 20, and run_t6_multiseed.py
    # uses 20 iterations (seeds [123, 7, 2024] + merged 42). A smaller
    # default would silently under-run the published training budget and
    # make the mechanism curves non-comparable.
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 7, 123, 2024])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--tag", type=str, default="",
                        help="Optional run tag; tagged runs live in "
                             "seed_<S>__<tag>/ and are excluded from the "
                             "untagged aggregate")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--aggregate", action="store_true",
                        help="Rebuild summary_mechanism.{json,md} and exit")
    parser.add_argument("--grad-proxy", action="store_true",
                        help="AM only: gradient-interference proxy per iter")
    args = parser.parse_args()

    os.makedirs(os.path.join(EXP_DIR, "results", SUBDIR), exist_ok=True)

    if args.aggregate:
        aggregate_and_write(args.tag)
        return

    if args.method is None:
        parser.error("--method is required unless --aggregate is given")
    if args.grad_proxy and args.method != "am":
        parser.error("--grad-proxy is only supported with --method am")

    log(f"T6 mechanism run: method={args.method} seeds={args.seeds} "
        f"iterations={args.iterations} eval_eps={args.eval_episodes} "
        f"tag={args.tag or '(none)'} grad_proxy={args.grad_proxy}")

    for seed in args.seeds:
        if args.method in ("ours", "amphase"):
            run_evolution_method(args.method, seed, args)
        else:
            run_baseline_method(args.method, seed, args)

    try:
        aggregate_and_write(args.tag)
    except Exception as e:
        log(f"[warn] aggregation failed (runs are saved): {e}")


if __name__ == "__main__":
    main()
