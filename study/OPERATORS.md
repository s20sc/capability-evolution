# Fixed learning and evaluation operators

run_operator.py executes one declared recipe job in a new workspace. It uses the native scientific learning/evaluation bodies, complete parent state, fixed data hashes, sampling order and update schedule. Source identities and deployment adaptations are listed in OPERATOR_IDENTITIES.json. It is not an autonomous research planner.

## Inputs and operation limits

Fill INPUT_MAP_EXAMPLE.json with the locations of the two prepared learning arrays, two provider metadata datasets, five generated-experience arrays and the three distributed states. All byte identities are checked. The source/runtime tree must match the recorded requirements and source identities. Each command needs explicit mutation/action/case budgets; a budget-stop receipt describes partial work and is not a completed scientific result. The configured stage length is retained even when a short acceptance probe stops early.

From the extracted package root, use `python3 -I -B run_operator.py METHOD ACTION NEW_WORKSPACE --input-map INPUT_MAP.json --vendor VENDOR_DIRECTORY --manifest-sha256 MANIFEST_SHA --job JOB_ID --max-optimizer-mutations N --max-actions M --max-cases K`. Correction uses --arm instead of a native job ID; evaluation additionally requires --panel new, development or selected-old. Keep the three panels separate.

## Complete parent and route identities

Native learning/evaluation requires explicit reference maps. A map associates each declared reference name with its checkpoint sidecar file, or with {"input_role":"shared_S0"} for the common source only. Stage counters, fixed root hashes and continuation/node origin records are checked. A reseeded continuation is not interchangeable with the common source. Use the sidecar JSON emitted beside a generated checkpoint; its origin identifies the recipe, job and continuation seed. For multicycle runs, initialize the declared roots and follow the physical-node dependency plan before evaluating the prescribed routes. The saved plan/schedule describes this ordering and shared-node reuse; six continuation roots do not constitute six independent pretraining runs.

Use --reference-map MAP.json for routed evaluation; two routes may require two different policy checkpoints. Child evaluations require --pairing-reference pointing to the completed matching baseline panel. The acquisition development recipe and 32-scene acquisition-endpoints recipe are distinct. Defaulting every route to the common source is prohibited.

Correction training restores D101000 and uses the fixed old/target/correction windows, including the C0 continuation control. To evaluate a generated correction child, pass its completed training status or checkpoint reference with --start-reference. Without a child reference, --state-role selects the original Square parent or D. A complete original-parent panel is required for child comparisons: pass run/status.json together with its adjacent protocol.json. Their digest binding, original parent identity, evaluation settings and low-noise decoder are checked. The fixed initial-state digests and distinct panel membership are retained.

## Decoding and outcome rules

The first two studies use the original maximum-probability GMM-component decoder. Correction evaluation uses native low-noise sampling and the original action-seed schedule. The evaluator bodies preserve task success holds, clipping, workspace/actuation checks and the legacy force diagnostic. These decoding schemes and case panels must not be silently unified.

The operation meter records entries and returns separately before/after actual model, optimizer and environment calls. An entry without a return may represent interrupted work; missing receipts or incomplete meter records are not zero work. Raw measurements in results/ describe the as-run experiments. Instrumentation/deployment overhead and metadata file sizes in a new run need not reproduce those original wall times/bytes.

## Coverage

The module interfaces expose the main fixed operators and the recorded graph/recipe parameters. Full execution of every learning/evaluation job, regeneration of every starting state/selected experience and independent full-training replication are separate from the bounded entry checks. The original starting-state and generated-experience companions are separate inputs; provider data and full third-party runtime/assets must be obtained under their terms.
