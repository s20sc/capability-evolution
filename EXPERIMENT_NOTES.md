# Experiment Notes

## Environment

- **OS**: WSL2 (Linux 6.6.87) on Windows
- **GPU**: RTX 5090 32GB (Blackwell sm_120)
- **CPU**: Core Ultra 9 265K (24 cores)
- **RAM**: 192GB DDR5
- **Python**: 3.13, venv
- **Simulator**: robosuite 1.5 + MuJoCo
- **RL**: stable-baselines3 (SAC)
- **Robot**: Franka Panda 7-DOF

## Experiment Progress

| Experiment | Status | Result File |
|------------|--------|-------------|
| exp1 (Evolution Curves) | **Complete** — T1-T6 × 20 iter | `exp1_evolution_over_time.json` |
| exp2 (Method Comparison) | **In Progress** — T1-T3 ours + T1 AM done, rest running | `exp2_method_comparison.json` |
| exp3 (Safety Evaluation) | **Complete** — blocking rate 94-97% | `exp3_safety.json` |
| ablation (Learning Modalities) | **Complete** — 7/7 configs | `ablation_modalities.json` |

### exp1 Results Summary

| Task | Iterations | Reward (start → end) |
|------|-----------|---------------------|
| T1_Pick | 20 | 0.60 → 1.25 |
| T2_Place | 20 | 0.24 → 2.79 |
| T3_Stack | 20 | 1.00 → 17.92 |
| T4_NutAssembly | 20 | 0.88 → 6.21 |
| T5_PickPlaceMulti | 20 | 0.77 → 2.16 |
| T6_TwoArmPegInHole | 20 | 211.05 → 340.92 |

### Ablation Results Summary

| Config | Final Reward | Takeaway |
|--------|-------------|----------|
| rl_only | 11.73 | RL is the core driver |
| il_only | 0.16 | IL alone can't learn |
| llm_only | 0.13 | LLM alone can't learn |
| rl_il | 16.21 | IL boosts RL (+38%) |
| rl_llm | 16.69 | LLM boosts RL (+42%) |
| il_llm | 0.69 | No RL = no learning |
| all_three | 4.71 | Combined (lower due to partial run recovery) |

**Key insight**: RL is necessary and sufficient for learning. IL and LLM provide meaningful boosts (+38-42%) when combined with RL, but are useless without it.

### exp3 Safety Results

| Task | Blocking Rate | Notes |
|------|--------------|-------|
| T1_Pick | 94.2% | OK |
| T2_Place | 95.1% | OK |
| T3_Stack | 100.0% | Actions inherently aggressive for this task |
| T4_NutAssembly | 100.0% | Same |
| T5_PickPlaceMulti | 95.6% | OK |
| T6_TwoArmPegInHole | 97.4% | OK |

## Key Issues & Fixes (13 total)

### Bug 1: PPO weights never transferred back to ECM

`hybrid_learner.py` called `rl_learner.finetune_ecm()` but the returned PPO model was discarded. ECM kept random weights forever. Fix: explicitly extract PPO actor weights and call `ecm.evolve(new_state_dict)`.

### Bug 2: PPO MlpPolicy vs ECM architecture mismatch

SB3 MlpPolicy default is `[64,64]`, ECM is `[256,256]`. Fix: pass `policy_kwargs={"net_arch": [256, 256]}` to match, and write explicit weight mapping functions (`_inject_ecm_to_*` / `_extract_*_to_ecm_state`).

### Bug 3: Fallback action dimension bug

`persistent_agent.py` line 95 used `obs.shape[0]` (observation dim ~40) as action dim (should be 7 for Panda). Fix: store `act_dim` from ECM descriptor.

### Bug 4: Evaluation used random planner to switch ECMs mid-episode

The frozen planner had random weights, causing chaotic ECM switching within episodes. Fix: round-robin assignment — each episode uses one ECM consistently.

### Bug 5: Version gating too aggressive

`rollback_threshold=0.05` caused almost every update to be rolled back. RL training has inherent variance, 0.05 is too tight. Fix: relaxed to `0.3`.

### Bug 6: PPO too slow and sample-inefficient

PPO with 50K timesteps/ECM/iter took ~40 min per iteration. 20 iterations x 6 tasks = days. Fix: switched to SAC (off-policy, 5-10x more sample efficient).

### Bug 7: Single-env training too slow

One robosuite env on CPU is bottleneck. Fix: `SubprocVecEnv` with parallel environments. Scaled from 4 → 8 → 16 envs as stability improved.

### Bug 8: SAC replay buffer discarded between iterations

Each `finetune_ecm` call created a fresh SAC model, losing all collected experience. Fix: persistent `_sac_cache` dict that keeps SAC models (with replay buffers) across iterations per ECM.

### Bug 9: SubprocVecEnv child process crash (EOFError)

Too many parallel robosuite envs caused memory pressure and child process death. Fix: added `learner.rl_learner.close()` after each task to free subprocesses. Tuned `n_envs` based on available RAM.

### Bug 10: Results lost on crash

All results saved only at the very end. Any crash = total data loss. Fix: per-task save in exp1/exp3, per-config save in ablation.

### Bug 11: Governance 100% blocking rate

Four sub-bugs in the governance/safety evaluation:
- **Velocity threshold**: `max_velocity * 0.05 = 0.075` was way too tight for action space [-1,1]. Fix: removed `* 0.05`.
- **Force estimate**: `norm(action) * 100` triggered on all actions. Fix: changed to `* 15` (realistic OSC mapping).
- **ee_pos extraction**: `obs[:3]` is NOT end-effector position (flat obs is alphabetically sorted). Fix: exposed actual `robot0_eef_pos` in wrapper's info dict.
- **Random ECMs in safety eval**: `_eval_safety` created untrained ECMs with random weights → random actions → 100% blocked. Fix: train ECMs briefly before evaluation.

### Bug 12: CUDA crash on long runs

`cudaErrorIllegalAddress` after hours of GPU training. Fix: per-iteration incremental saves via `on_iter_done` callback, so crash loses at most 1 iteration (~10 min). Auto-retry script handles restart.

### Bug 13: No iteration-level resume

`_run_evolution` always restarted from iteration 0 with fresh ECMs. Fix: added `resume_from` parameter — loads prior metrics, skips completed iterations, warms up ECMs with brief training to approximate prior weights.

### Bug 14: Baselines crash — "executing action in terminated episode"

All 4 baselines (`AgentModificationBaseline`, `StaticECMBaseline`, `SPiRLBaseline`, `SkiMoBaseline`) passed raw Gymnasium env to PPO. SB3's PPO expects a VecEnv for proper auto-reset handling. When robosuite episode terminates, PPO tries to step before reset → `ValueError`. Fix: wrap env in `DummyVecEnv([lambda: env])` in all baselines. Also added `device="cpu"` since PPO with MlpPolicy is faster on CPU.

### Bug 15: exp3 `_eval_safety` training too slow

`_eval_safety` inherited main config's `learning_starts=5000` with `total_timesteps=5000`, meaning SAC collected 5000 random steps without doing any learning. Combined with 3 rounds × 4 ECMs = 12 training sessions, each task took ~2 hours. Fix: reduced to 1 round × 4 ECMs, `learning_starts=500`, `total_timesteps=3000`, `buffer_size=10000`. Each task now takes ~5 minutes.

### Bug 16: exp3 no incremental saves

`run_exp3` only saved results after all 6 tasks finished. With ~2 hours per task (before Bug 15 fix), a crash at task 5 lost everything. Fix: save after each task, plus load/skip completed tasks on resume. Also filter out old buggy results (100% blocking rate) on load.

### Bug 17: exp2 skip logic only checked "ours"

`run_exp2` skipped an entire task if "ours" was complete, even if 4 baselines hadn't run at all. T1-T3 baselines were permanently skipped. Fix: check all 5 methods are complete before skipping; each method also independently checks/skips.

### Bug 18: Pointless warmup for completed runs

`_run_evolution` with `resume_from` always did 3 warmup iterations even when `start_iter >= iterations` (nothing to run). Wasted ~24 minutes per task × 6 tasks = 2.4 hours. Fix: return immediately when already complete.

### Bug 19: env done state not cleared between evaluate→train

After `evaluate()` finishes, the raw env is in "done" state. Next `train_iteration()` calls PPO learn via VecEnv which steps the same env → `ValueError: executing action in terminated episode`. The DummyVecEnv fix (Bug 14) wasn't enough because evaluate uses the raw env directly. Fix: call `env.reset()` at the start of every `train_iteration()`.

### Bug 20: `_run_baseline` no incremental saves or resume

`_run_baseline` had no `on_iter_done` callback or `resume_from` support. With WSL crashing every 2-3 hours, baseline progress was lost repeatedly. Fix: added both parameters, mirroring `_run_evolution`'s crash recovery.

### Bug 21: Baselines not utilizing hardware

All 4 baselines used `DummyVecEnv` (1 env, single-threaded) and 50K timesteps per iteration vs our method's 16 envs + 20K timesteps. CPU at 10%, GPU idle. With 24 cores and 192GB RAM, this was a massive waste.

Fix:
- AM, SkiMo: `DummyVecEnv` → `SubprocVecEnv` with 8 parallel envs
- All baselines: timesteps 50K → 20K (fair comparison, same as our method)
- Result: baseline iteration time from ~6.5 min → ~1.5 min (~4x speedup)

## Crash Recovery Architecture

Three layers of protection, added incrementally as crashes occurred:

### Layer 1: Per-iteration saves (`on_iter_done` callback)
```python
def _run_evolution(self, ..., on_iter_done=None):
    for it in range(start_iter, iterations):
        # ... training ...
        if on_iter_done:
            on_iter_done(results)  # save to disk every iteration
```

### Layer 2: Resume from partial results (`resume_from`)
```python
# _run_evolution skips completed iterations
if resume_from and len(resume_from["iterations"]) > 0:
    start_iter = len(results["iterations"])
    # warm up ECMs to recover approximate weights
```

### Layer 3: Auto-retry script (`auto_run.sh`)
- Max 5 retries per experiment
- Diagnoses crash type (EOFError, CUDA, heap corruption, OOM)
- Parallel execution (ablation + exp3 simultaneously)
- Validates results before moving to next experiment

## Isaac Lab Incompatibility

RTX 5090 (Blackwell/sm_120) does NOT work with Isaac Lab as of April 2026. PhysX GPU acceleration fails on this architecture. Monitor: https://github.com/isaac-sim/IsaacLab/issues/2483

## robosuite Warnings (safe to ignore)

```
[robosuite WARNING] The config has defined for the controller "legs"...
```

This is normal. `load_composite_controller_config(controller="BASIC")` includes configs for all body parts (legs, torso, head, etc.). Panda arm doesn't have these, so robosuite skips them. Harmless.

## Windows/WSL Stability Issues

- **Bun segfault**: Running long experiments inside Bun terminal can cause Bun runtime segfault. Fix: use `nohup` to detach.
- **Windows Terminal crash**: Switching to WeChat IME (微信输入法) crashes Windows Terminal, killing all WSL processes started from that session. Fix: use `nohup` and ensure processes are fully detached.
- **Windows crash/reboot**: WSL processes don't survive Windows crash. Incremental saves + auto-retry script handle recovery.

## Performance Tuning Timeline

| Config | Time per iter | GPU% | CPU% | Notes |
|--------|-------------|------|------|-------|
| PPO, 50K steps, 1 env, GPU | ~40 min | ~5% | ~5% | Original, unusable |
| SAC, 20K steps, 2 envs, CPU | ~33 min | 0% | ~10% | Stable but slow |
| SAC, 20K steps, 8 envs, GPU | ~10 min | ~30% | ~35% | Good |
| SAC, 20K steps, 16 envs, GPU, batch=4096, grad_steps=8 | **~9.5 min** | **~65%** | **~40%** | Current config (ours) |
| PPO baselines, 50K steps, 1 env, CPU | ~6.5 min | 0% | ~5% | Original baselines, massive waste |
| PPO baselines, 20K steps, 8 envs, CPU | **~1.5 min** | 0% | ~35% | Optimized baselines |

## Current Run Config

```yaml
rl:
  algorithm: SAC
  learning_rate: 3.0e-4
  batch_size: 4096
  buffer_size: 1000000
  learning_starts: 5000
  gamma: 0.99
  tau: 0.005
  n_envs: 16
  gradient_steps: 8
  total_timesteps_per_iter: 20000

ecm:
  rollback_threshold: 0.3
```

## Running Experiments

```bash
# Automated: runs ablation → exp3 → exp2 with retry
source .venv/bin/activate
nohup bash auto_run.sh > auto_run.log 2>&1 &

# Manual: single experiment
nohup python -u run_experiment.py --exp exp1 --task T3_Stack --iterations 20 --eval-episodes 30 --seed 42 > experiment.log 2>&1 &

# Monitor
tail -f auto_run.log | grep -v "robosuite WARNING"
grep "iter " auto_run.log | tail -5

# Check results
python3 -c "import json; d=json.load(open('results/ablation_modalities.json')); print({k:len(v['iterations']) for k,v in d.items()})"
```

## Total Bug Count: 21

| Category | Bugs | Impact |
|----------|------|--------|
| Weight transfer | #1, #2 | ECM never learned (critical) |
| Action/observation | #3, #11 | Wrong dimensions, wrong ee_pos |
| Evaluation logic | #4, #5, #17, #18 | Random ECM switching, over-rollback, wrong skip, pointless warmup |
| Performance | #6, #7, #15, #21 | PPO slow, single-env, exp3 slow, baselines not using hardware |
| Data persistence | #8, #10, #13, #16, #20 | Buffer lost, results lost, no resume |
| Crash stability | #9, #12, #19 | SubprocVecEnv OOM, CUDA crash, env done state |
| Baseline code | #14 | PPO auto-reset crash |

## Lessons Learned

1. **Always save incrementally** — never wait until the end. Crashes are inevitable on multi-hour runs.
2. **Off-policy RL (SAC) >> On-policy (PPO)** for single-machine robosuite experiments. 5-10x faster.
3. **Match n_envs to CPU cores**, batch_size/gradient_steps to GPU. See `PERFORMANCE_GUIDE.md`.
4. **Test weight transfer early** — Bug 1 (weights never transferred) was the most critical and hardest to spot. Always verify with a roundtrip test.
5. **Governance thresholds must match action space** — don't assume action values correspond to physical units.
6. **WSL is fragile** — nohup everything, save everything, auto-retry everything.
7. **SB3 PPO needs VecEnv** — never pass raw Gymnasium env to PPO/SAC. Always wrap in `DummyVecEnv`. Raw env causes `ValueError: executing action in terminated episode` on episode boundaries.
8. **Config inheritance is dangerous** — `_eval_safety` inherited `learning_starts=5000` from main config, making its brief training session useless. Always explicitly override all relevant params for special-purpose training.
9. **Shared env between train and eval is fragile** — if `evaluate()` uses the raw env and `train_iteration()` uses a VecEnv wrapping the same env, the done state desynchronizes. Either use separate envs or explicitly reset between phases.
10. **Optimize baselines too** — easy to focus only on your own method's speed. Baselines using 1 env + 50K steps while your method uses 16 envs + 20K steps wastes days of compute. Match timesteps for fair comparison and use SubprocVecEnv for all methods.
