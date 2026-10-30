# Fixed learning diagnostics

These entries implement the C/D diagnostics reported with the Square study. For the original parent P and the A/B continuations on the same diagnostic panels, use FEEDBACK_DIAGNOSTICS.md and run_feedback.py. They use the supplied complete Square100000 parent and the fixed generated target experience. The supplied parent is already learned. Neither diagnostic introduces an independent pretraining lineage, a new holdout, or autonomous experience collection.

C uses all three eligible generated trajectories, with 50 original-data and 50 generated-data sequence windows per update. D uses the trajectory selected by the fixed parent-failure/valid-experience intersection. Its 50 old windows use action samples from a frozen copy of the parent's learned distribution; 50 new windows use the selected trajectory. D includes the original before/after training-data fit probes. Samples are not clipped to change the learning objective. The teachers, probes, and copied policies are additional work, not free operations.

The original training files and rehearsal routines are retained in diagnostics/. The fixed evidence cases are scientific projections of the recorded parent probe and generator outcomes. selection.py applies the original intersection and eligibility rules; these files do not claim a causal diagnosis or an independently collected test set. Generated experience remains available as a companion input; its acquisition source and provenance are distinct from the learning operation here.

Use the runtime in DEPENDENCIES.md, the main INPUT_MAP_EXAMPLE.json, and a fresh output directory per operation. Every command verifies the fixed source manifest and the exact parent, training data, target pack and native-source identities. Replace MANIFEST_SHA with the actual SHA-256 of PACKAGE_IDENTITIES.json. CUDA visibility is GPU0 for training and empty for evaluation.

```sh
CUDA_VISIBLE_DEVICES=0 python3 -I -B run_diagnostic.py C train /absolute/new-C \
  --input-map INPUT_MAP.json --vendor /absolute/runtime --manifest-sha256 MANIFEST_SHA \
  --seconds 120 --max-optimizer-mutations 1000 --max-actions 0 --max-cases 0
CUDA_VISIBLE_DEVICES=0 python3 -I -B run_diagnostic.py D train /absolute/new-D \
  --input-map INPUT_MAP.json --vendor /absolute/runtime --manifest-sha256 MANIFEST_SHA \
  --seconds 175 --max-optimizer-mutations 1000 --max-actions 0 --max-cases 0
CUDA_VISIBLE_DEVICES='' python3 -I -B run_diagnostic.py C evaluate /absolute/new-C-evaluation \
  --start-reference /absolute/new-C/all_valid/status.json \
  --input-map INPUT_MAP.json --vendor /absolute/runtime --manifest-sha256 MANIFEST_SHA \
  --seconds 200 --max-optimizer-mutations 0 --max-actions 6400 --max-cases 16
CUDA_VISIBLE_DEVICES='' python3 -I -B run_diagnostic.py D evaluate /absolute/new-D-evaluation \
  --start-reference /absolute/new-D/parent_rehearsal/status.json \
  --input-map INPUT_MAP.json --vendor /absolute/runtime --manifest-sha256 MANIFEST_SHA \
  --seconds 200 --max-optimizer-mutations 0 --max-actions 6400 --max-cases 16
```

Training uses the original 1,000-update rule and original internal 110-second C / 165-second D deadlines. The command limits bound total entry execution; they do not extend those internal training deadlines. Evaluation reuses the original native low-noise action distribution and resets recurrent state each episode, across the original four development plus twelve previously exposed panel scenes. All sixteen cases must complete; a partial budget stop is not a completed diagnostic.

The status files bind a regenerated child to its actual new protocol, evidence projection and portable parent hash. They intentionally have new identities. Do not substitute a new child's hash for the archived as-run D state or rewrite the original result records. Downstream correction studies can use the supplied D101000 state directly. Recreating the complete ancestry is not required to use that declared starting input.

Each operation writes diagnostic-receipt.json and a native-operation meter. Successful exit alone is insufficient: inspect the completed learning/evaluation status, counters, teacher immutability checks, full parent/child restoration checks and the expected cases. Output limits are polled, rather than operating-system disk quotas. These operations cover the two declared diagnostics, not a complete independent replication of every study in the paper.
