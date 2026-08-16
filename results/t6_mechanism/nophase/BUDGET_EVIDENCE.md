# Budget evidence for the AM+nophase arm

The arm exists to isolate the four-way module rotation (four whole-task
policies taking turns on a time-slice schedule) from every other factor, so the
budget claim must be checkable from the artifact rather than taken on trust.

## Claim

AM+nophase trains ONE whole-task module at 200,000 environment steps per
iteration. AM+phase trains FOUR whole-task modules (rotated by the time-slice
schedule at evaluation) at 50,000 steps each, also 200,000 per iteration. Learner (SAC), imitation demonstrations, evaluation
protocol, episode counts, governance, and the built-in keep-best gate are
identical; one-module-versus-four-rotated is the only difference.

## How to verify from this repository

1. Base config: `configs/default.yaml`, `learning.rl.total_timesteps_per_iter: 50000`
   (per module per iteration).
2. AM+phase registers four modules: `amphase_runner.py`, the loop over
   `["reach", "grasp", "lift", "place"]`. 4 x 50,000 = 200,000.
3. AM+nophase registers one module: `nophase_impl.py`, the loop over
   `["wholetask"]`; `diff amphase_runner.py nophase_impl.py` shows this and the
   class rename are the only differences.
4. The multiplier is applied in `nophase_runner.py:run_seed`, which sets
   `total_timesteps_per_iter = 50000 * 4` before constructing the runner, and
   logs the result. Every run log therefore opens with the line:
   `budget: 50000 x 4 = 200000 steps/iteration for the single ECM, matching
   AM+phase total compute`
   See `np_p_<seed>.log`.

## Note on the result files

The runs recorded in `seed_<S>/mechanism.json` were launched before the driver
was amended to copy the budget into `_meta`, so those files carry
`budget_steps_per_iter: null`. The authoritative record for those runs is the
log line above, present in each `np_p_<seed>.log`. Runs launched after the
amendment carry the field directly. Nothing about the executed budget differs
between the two; only whether it was transcribed into the JSON.


> Note: the console logs cited above are session records retained by the authors (they embed machine-local paths). The per-episode evidence behind every figure in this note is in the adjacent per-seed mechanism.json files in this directory.

> Per-seed metadata state: only seed 2024's mechanism.json carries the amended `budget_steps_per_iter: 200000` / `n_ecms: 1` fields; seeds 7, 42 and 123 ran before the metadata amendment and carry null there. The budget those three consumed is established by the driver line (`nophase_runner.py:run_seed`, `50000 * 4`) and the file-length diff described above, not by their own metadata.
