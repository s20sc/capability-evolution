# Experiment Performance Guide

Hardware: Core Ultra 9 265K (24 cores) / RTX 5090 32GB / 192GB DDR5

This guide documents how to maximize hardware utilization for RL training experiments (robosuite + stable-baselines3). Lessons from the capability evolution experiments, applicable to future papers.

## Hardware Budget

| Resource | Total | Safe Max | Notes |
|----------|-------|----------|-------|
| CPU cores | 24 | 20 | Leave 4 for system + monitoring |
| GPU VRAM | 32GB | 28GB | Leave headroom for CUDA context |
| RAM | 192GB | 160GB | Leave for OS + other apps |
| GPU Power | 450W TDP | 450W | No throttle needed |

## Key Parameters and Tuning

### 1. Parallel Environments (`n_envs`)

**Impact**: CPU utilization. Each robosuite env runs physics in its own subprocess, pinned to one core.

| n_envs | CPU Usage | RAM Cost | Throughput |
|--------|-----------|----------|------------|
| 2 | ~10% | ~2GB | baseline |
| 8 | ~35% | ~8GB | ~3.5x |
| 16 | ~55% | ~16GB | ~6x |
| 20 | ~70% | ~20GB | ~7x (diminishing) |

```yaml
# Recommended for this machine
n_envs: 16  # sweet spot: high throughput, no OOM
```

**Pitfalls**:
- `n_envs > 20`: diminishing returns, risk of SubprocVecEnv EOFError (child OOM)
- Always use `SubprocVecEnv`, not `DummyVecEnv` (which is single-threaded)
- Each subprocess forks the full Python environment (~500MB-1GB)

### 2. Batch Size (`batch_size`)

**Impact**: GPU utilization. Larger batches = more parallel matrix ops on GPU.

| batch_size | GPU Usage | VRAM Cost | Notes |
|------------|-----------|-----------|-------|
| 256 | ~10% | ~50MB | Default SB3, wastes GPU |
| 1024 | ~30% | ~100MB | Decent |
| 4096 | ~60% | ~300MB | Good balance |
| 8192 | ~75% | ~500MB | Max practical for [256,256] MLP |

```yaml
batch_size: 4096  # recommended for RTX 5090 + small MLP policy
```

**Notes**:
- For [256,256] MLP, even batch=8192 uses <1GB VRAM — the network is tiny
- Batch size is bounded by `buffer_size` (can't sample more than what's stored)
- If using CNN/transformer policies, reduce batch size accordingly

### 3. Gradient Steps (`gradient_steps`)

**Impact**: GPU utilization. Number of gradient updates per environment step.

| gradient_steps | GPU Usage Boost | Training Quality |
|----------------|-----------------|------------------|
| 1 | baseline | Standard SAC |
| 4 | ~3x GPU time | Good for large buffers |
| 8 | ~5x GPU time | Aggressive, needs large buffer |
| 16 | ~8x GPU time | Risk of overfitting |

```yaml
gradient_steps: 8    # good with buffer_size >= 500K
```

**Rule of thumb**: `gradient_steps` should scale with `buffer_size / total_timesteps_per_iter`. Large buffer = more diverse data = can afford more gradient steps without overfitting.

### 4. Replay Buffer (`buffer_size`)

**Impact**: RAM usage. Larger buffer = more training data diversity.

| buffer_size | RAM Cost | Notes |
|-------------|----------|-------|
| 100K | ~40MB | Default, tight |
| 500K | ~200MB | Good for most tasks |
| 1M | ~400MB | Recommended with 192GB RAM |
| 5M | ~2GB | For very long experiments |

```yaml
buffer_size: 1000000  # 400MB — trivial for 192GB
```

**Memory formula**: `buffer_size × (obs_dim × 2 + act_dim + 3) × 4 bytes`
- For robosuite Panda (obs_dim≈40, act_dim=7): ~400 bytes/transition
- 1M transitions ≈ 400MB

### 5. Device Selection

```yaml
device: auto  # Use GPU
```

**When to use GPU** (`auto`):
- `batch_size >= 1024`
- `gradient_steps >= 4`
- Network has > 50K parameters

**When to use CPU** (`cpu`):
- Small MLP with `batch_size < 512` (GPU communication overhead dominates)
- CUDA keeps crashing (last resort — fix the root cause first)

**Critical**: If using GPU, implement per-iteration incremental saves. CUDA errors (`cudaErrorIllegalAddress`) can happen on long runs. With incremental saves, you lose at most 1 iteration (~5-10 min) instead of the whole experiment.

## Recommended Config for This Machine

```yaml
learning:
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
```

Expected utilization: CPU ~40-55%, GPU ~60-70%, RAM ~50GB

## Crash Recovery

Long experiments will crash. Design for it from the start.

### Per-Iteration Saves

Save results after every iteration, not just at the end:

```python
def _run_evolution(self, env, task, on_iter_done=None):
    results = {"iterations": [], "metrics": []}
    for it in range(iterations):
        # ... training and eval ...
        results["iterations"].append(it + 1)
        results["metrics"].append(metrics)
        if on_iter_done:
            on_iter_done(results)  # save to disk
    return results
```

### Resume from Partial Results

Load existing results and skip completed work:

```python
# Load existing
if os.path.exists(save_path):
    results = json.load(open(save_path))

# Skip completed
if name in results and len(results[name]["iterations"]) >= target:
    log(f"[{name}] already complete, skipping")
    continue
```

### Auto-Retry Script

```bash
run_with_retry() {
    local exp_name="$1"; shift
    local attempt=0
    while [ $attempt -lt 3 ]; do
        attempt=$((attempt + 1))
        python -u run_experiment.py "$@" 2>&1 | tee "${exp_name}_attempt${attempt}.log"
        [ ${PIPESTATUS[0]} -eq 0 ] && return 0
        sleep 30  # cool down before retry
    done
    return 1
}
```

### Common Crash Types

| Error | Cause | Fix |
|-------|-------|-----|
| `EOFError` in SubprocVecEnv | Child process OOM | Reduce `n_envs`, add `learner.close()` between tasks |
| `cudaErrorIllegalAddress` | GPU memory corruption | Per-iter saves + auto-retry; reduce `batch_size` if frequent |
| `corrupted size vs. prev_size` | Heap corruption from repeated SubprocVecEnv create/destroy | Reuse vec_env across iterations, call `close()` explicitly |
| `MemoryError` | RAM exhaustion | Reduce `buffer_size` or `n_envs` |

## Parallel Experiment Execution

When experiments have different resource profiles, run them in parallel:

```bash
# GPU-heavy (SAC training) — foreground
python -u run_experiment.py --exp ablation &

# CPU-heavy (governance eval) — background
python -u run_experiment.py --exp exp3 &

wait  # wait for both
```

**Compatible pairs** (can run simultaneously):
- RL training (GPU+CPU) + Safety eval (CPU only)
- RL training (GPU+CPU) + Baseline eval (CPU only)

**Incompatible** (will OOM):
- Two RL training runs with n_envs=16 each (32 subprocesses = too many)

## PPO vs SAC: Algorithm Choice

| Factor | PPO | SAC |
|--------|-----|-----|
| Sample efficiency | Low (on-policy, discards data) | **High (off-policy, replay buffer)** |
| Wall time for 20K steps | ~40 min | **~10 min** |
| GPU utilization | Low (large batch but few updates) | **High (continuous gradient updates)** |
| Stability | More stable | Occasionally diverges |
| Best for | Large-scale distributed training | **Single-machine, sample-limited** |

**Verdict**: For this machine and robosuite tasks, SAC is 5-10x faster than PPO. Use SAC unless you have a specific reason for PPO.

## RTX 5090 (Blackwell) Compatibility Notes

- **PyTorch**: Works with CUDA 12.8+, sm_120 architecture. Make sure torch is up to date.
- **Isaac Lab / PhysX GPU**: As of 2026-04, does NOT support sm_120. Cannot use GPU-accelerated physics. Stick with robosuite + MuJoCo (CPU physics).
- **Long CUDA runs**: May produce `cudaErrorIllegalAddress` after hours. Mitigate with per-iteration saves and auto-retry, not by switching to CPU.

## Quick Reference: Scaling Rules

```
More CPU cores idle?  → increase n_envs
GPU < 30%?            → increase batch_size and/or gradient_steps
RAM < 50%?            → increase buffer_size and/or n_envs
Experiment too slow?  → check which resource is the bottleneck first
Keeps crashing?       → add incremental saves, then increase resources
```
