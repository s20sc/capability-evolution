# Evolving Skill Modules under a Fixed Planner

Artifact for the paper "Evolving Skill Modules under a Fixed Planner:
Versioning, Rollback, and Runtime Governance for Long-Lived Robot Systems"
(under review).

The system keeps a fixed high-level decision layer and moves all learning
into versioned skill modules (ECMs) that are trained, promoted through a
held-out gate, and rolled back when a promotion regresses. Every action
passes through a runtime governance layer at the execution boundary.

## What the paper reports

The evaluation contains three negative results and two measured lifecycle
mechanisms. The negative results:

1. Single-run peak success on the flagship T6 task swings threefold across
   random seeds within one method, so single-run peaks cannot rank these
   methods at affordable seed counts.
2. The rotating multi-policy configuration (four separately trained
   whole-task policies taking turns on one task) is refuted by a
   budget-matched single-policy control; restoring the termination
   condition the clock replaced raises retention on every seed.
3. The runtime safety shield cuts flagged violations by 98-100% on the
   five single-arm tasks (34.9% on the bimanual task) while holding task
   success at zero; blocking everything is not a safety result, and no
   safety claim survives for it.

The measured lifecycle mechanisms:

1. The held-out promotion gate kept 12 of 12 injected regressions out of
   deployment, at a 22.5% clean-candidate rejection cost. The injections
   were calibrated above the gate threshold, so 12 of 12 is close to what
   that calibration guarantees; the rejection cost is the measurement.
2. The rollback dip detector caught 9 of 12 injected regressions after
   deployment, missing all three on one seed. A severity sweep locates the
   boundary: no seed triggers at or below sigma = 0.2, and sigma = 0 gives
   12 of 12 clean iterations with no trigger.

## Repository layout

Top-level modules:

| Path | Contents |
|---|---|
| `agent/` | ECM data model with version management (`ecm.py`) and the fixed-planner persistent agent |
| `envs/` | robosuite wrapper (perturbations, safety tracking) and the T1-T6 task suite |
| `learning/` | SAC (`rl_learner.py`), behavioural cloning, LLM-synthesis stub, and the hybrid learner |
| `runtime/` | runtime governance layer (velocity, workspace, and force predicates) |
| `baselines/` | Agent Modification, SkiMo, SPiRL, Static ECM |
| `configs/` | `default.yaml`, the budget and env configuration all runs share |
| `run_experiment.py` | driver for the historical evolution, method-comparison, safety, and ablation experiments |
| `run_t6_multiseed.py` | four-seed T6 sweep |
| `run_t6_amphase.py`, `amphase_runner.py` | learned-dispatcher arm (`amphase`) |
| `analyze_results.py`, `make_paper_figures.py` | aggregate analysis and figure generation |

The seven evaluation drivers, each with full usage in its module docstring,
followed by the analysis script that re-derives the retention table from their logs:

| Driver | Reproduces | Output |
|---|---|---|
| `lifecycle_runner.py` | fault-injection study of versioning, gated promotion, and rollback (3 arms x 4 seeds x 13 iterations, injections at {5, 8, 11}) | `results/t6_lifecycle/` |
| `governance_stress.py` | adversarial governance stress test on T1-T6, replay-only, with verified real end-effector state | `results/governance_stress/` |
| `mechanism_runner.py` | instrumented re-runs of ours / amphase / am / skimo on T6: per-phase progress ladder, decision churn, planner-drift audit | `results/t6_mechanism/` |
| `nophase_runner.py` + `nophase_impl.py` | budget-matched single-module control (one whole-task ECM at four times the per-module steps) | `results/t6_mechanism/nophase/` |
| `amphase_runner.py` | learned-dispatcher arm: the same four whole-task ECMs with a dispatcher cloned from the clock schedule | `results/t6_mechanism/amphase/` |
| `severity_runner.py` | detector operating characteristic: injection severity sweep on frozen incumbents, evaluation-only | `results/t6_severity/` |
| `handoff_runner.py` | state-conditioned handoff intervention: clock versus post-success hold on the fixed-schedule arm, evaluation-only | `results/t6_handoff/` |
| `analysis/retention_seed_level.py` (analysis, not a driver) | seed-level retention re-analysis; regenerates `analysis/retention_seed_level.json` and the paper's retention table | `analysis/` |

The paper's figures regenerate from the released per-seed files by the
`figures/render_fig_*.py` scripts; `analysis/method_comparison_n3.py`
regenerates the method-comparison table.

Example invocations:

```bash
# Lifecycle fault injection, one arm at a time
python lifecycle_runner.py --arm gating_on --seeds 42 7 123 2024 --iterations 13 --inject-at 5 8 11

# Governance stress (replay-only, CPU)
# The T3 and T6 rows replayed trained checkpoints that were not preserved; their
# per-episode logs are in results/governance_stress/. The other four rows ran with
# randomly initialised modules and regenerate with --policy fresh.
python governance_stress.py --tasks all --episodes 50 --policy fresh
python governance_stress.py --aggregate

# Instrumented mechanism re-runs
python mechanism_runner.py --method ours
python mechanism_runner.py --aggregate

# Budget-matched single-module control
python nophase_runner.py --seeds 42 7 123 2024 --iterations 10

# Retention table
python analysis/retention_seed_level.py
```

Results layout:

- `results/exp1_evolution_over_time.json`, `results/exp2_method_comparison.json`,
  `results/exp3_safety.json`, `results/ablation_modalities.json`: historical
  aggregate files from the original development machine (see Provenance).
- `results/t6_multiseed/`, `results/t6_amphase/`: four-seed T6 sweeps.
- `results/t6_lifecycle/`, `results/governance_stress/`, `results/t6_mechanism/`:
  evaluation experiments, per-seed JSON plus summary files.

## Environment

- Python 3.10
- robosuite 1.4.1, MuJoCo 2.3.2, stable-baselines3 2.8.0 (pinned in
  `requirements.txt`; the wrapper uses `load_controller_config`, an entry
  point removed after robosuite 1.4.1)
- GPU optional. Training runs used an RTX 5090; the replay-only and
  analysis scripts (`governance_stress.py`, `analysis/retention_seed_level.py`)
  run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import robosuite, mujoco; print(robosuite.__version__, mujoco.__version__)"
```

## Provenance

Which numbers this repository can and cannot reproduce.

1. **The historical T1-T5 rows are not reproducible from this repository.**
   The per-task rows in `results/exp1_evolution_over_time.json`,
   `results/exp2_method_comparison.json`, `results/exp3_safety.json`, and
   `results/ablation_modalities.json` were produced by an implementation
   state that predates this repository. Under the code here, T1, T2, and
   T5 construct identical default robosuite environments and share one
   success criterion, so the stored pattern (100% on T1 with 0% on T2 for
   every method) cannot come from one code state. These files are retained
   as the historical record; the paper does not build conclusions on them.
2. **EXPERIMENT_NOTES.md is preserved verbatim.** It is the original
   machine's record and it conflicts with this repository's pinned stack
   (it records robosuite 1.5 under Python 3.13; it states no MuJoCo version). It has not been edited,
   and discrepancies between it and the pinned environment are expected.
3. **The seven evaluation drivers and their logs are the attested,
   reproducible evidence base.** `lifecycle_runner.py`,
   `governance_stress.py`, `mechanism_runner.py`,
   `nophase_runner.py`/`nophase_impl.py`, `amphase_runner.py`,
   `severity_runner.py`, and `handoff_runner.py`, together with the
   per-seed result files under `results/t6_lifecycle/`,
   `results/governance_stress/`, `results/t6_mechanism/`,
   `results/t6_severity/`, and `results/t6_handoff/`, ran under the pinned
   robosuite 1.4.1 / MuJoCo 2.3.2 / stable-baselines3 2.8.0 environment and
   are reproducible from this repository, with one exception: the T3 and T6
   governance-stress rows replayed trained checkpoints (recorded as
   `multiseed_T3_Stack_seed_123_ours` and `T6_TwoArmPegInHole`) that were not
   preserved, so those two rows ship as logs and cannot be replayed with their
   original weights. `analysis/retention_seed_level.py`
   re-derives the retention table from those logs in the same environment.

One known defect is documented in place rather than silently fixed: the
historical safety evaluation loop in `run_experiment.py` reads `ee_pos`
from a dictionary the wrapper never writes it into, so its governance
predicates evaluate at the origin. The paper's withdrawal of the stored
safety table rests on this behaviour, and a comment at the site describes
it. The vec-env rebuild in `learning/rl_learner.py` previously omitted the
seed and forwarded an argument robosuite 1.4.1 rejects; both points are
corrected in this repository and the evaluation runs used the corrected path.

## Which files back which table

`results/multiseed/T6_TwoArmPegInHole/` contains per-seed baseline runs for
SkiMo, SPiRL and Static ECM. The `seed_42` files there are a later cloud
re-run and are **not** the source of the paper's published four-seed columns:
those merge the historical seed-42 row from `results/exp2_method_comparison.json`
at aggregation time. The two differ (deposited SkiMo seed-42 peak reward
369.03 against the published 348.33), and both are shipped so the difference
is inspectable. The paper's provenance section states this.

Two field names in the deposited JSON predate terminology the paper narrowed,
and are left as written rather than rewritten after the fact:

- `false_rejection_rate` in `results/t6_lifecycle/summary_lifecycle.json` is
  what the paper calls the clean-candidate rejection cost. Nothing in the run
  establishes that a rejected clean candidate would have deployed safely, so
  "false" claims a ground truth the experiment does not have. The value
  (mean 0.225 over seed values 0.4, 0.5, 0.0, 0.0) is the one the paper prints.
- `dip_depth_pct` aggregates against a mean-of-preceding-three baseline, which
  is the windowed convention. The paper's lifecycle table prints depth at the
  injection iteration instead, and its caption declares both conventions and
  where they diverge. Recomputing at-injection from the same per-episode
  records reproduces the table.

## Reproducing the evaluation-only experiments

`handoff_runner.py` and `severity_runner.py` re-evaluate trained modules and do
not train. The module weights they need are in this repository:

    results/checkpoints/t6_mechanism_ours_seed_{7,42,123,2024}/ecm_weights.pt
    results/checkpoints/t6_lifecycle_gating_off_rollback_seed_{7,42,123,2024}/ecm_weights.pt

with SHA-256 prefixes in `results/checkpoints/CHECKSUMS.json`. Only the ECM
weights are shipped (1.7 MB each). The SAC optimiser archives that sit beside
them in the original run directories are needed to resume training, not to
evaluate, and are not included.

The learned-dispatcher arm (`amphase`) has no published checkpoint: no ECM
weights survive for those runs, which is why the handoff intervention covers the
fixed-schedule arm only. Whether those runs ever wrote weights is not something
the surviving record settles. Both drivers fail loudly with the missing path if a
checkpoint is absent rather than degrading silently.

## License

Apache 2.0, see `LICENSE`.

**Test note**: `tests/test_seed_replay.py` requires the full pinned stack (robosuite 1.4.1, MuJoCo 2.3.2, torch) to run; it is not a pure-python unit test.
