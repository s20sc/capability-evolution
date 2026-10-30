# Preparing the fixed demonstration inputs

run_preparation.py reconstructs the learning arrays from the exact provider files in DEPENDENCIES.md. It does not acquire new demonstrations or evaluate a policy. The original scientific preparation bodies are retained under preparation/; deployment admission, source identities, operation limits and output receipts are handled by the new entry point.

Use the fixed runtime sources and array/HDF5 stack described in the dependency guide. Use preparation/INPUT_MAP_EXAMPLE.json for the input map: metadata_Can (the raw Can v141 HDF5), metadata_Square (the Square low-dimensional v141 HDF5), and raw_Square (the Square raw v141 HDF5). The latter file is needed for the Square state/action correspondence check. Source hashes are fixed in recipes/preparation.json. The package supplies the ordered Square split and its original demo_1 observation anchor. No protected official Square-validation trajectory values are read; the provider split masks are metadata. Can's provider validation partition was already exposed during development and is not represented as a new protected holdout.

All commands require a fresh output directory outside the source package and its fixed manifest SHA. CUDA visibility must be empty. The commands below show complete-stage limits, not a shortened probe; set a local wall/output allowance explicitly.

## Square: raw/low-dimensional checks and 160/20 packaging

```sh
CUDA_VISIBLE_DEVICES='' python3 -I -B run_preparation.py square /absolute/new-square \
  --input-map INPUT_MAP.json --vendor /absolute/runtime \
  --manifest-sha256 MANIFEST_SHA --seconds 90 --output-poll-bytes 536870912 \
  --max-constructors 0 --max-state-writes 0
```

The original method checks raw/low-dimensional actions and states, observation shapes, the native anchor and the fixed partitions. Outputs run/train.npz (160 demonstrations) and run/development.npz (20) must match the original scientific input hashes. No environment is constructed.

## Can: reconstruct native observations, then package the original split

```sh
CUDA_VISIBLE_DEVICES='' python3 -I -B run_preparation.py can-extract /absolute/new-can-extraction \
  --input-map INPUT_MAP.json --vendor /absolute/runtime \
  --manifest-sha256 MANIFEST_SHA --seconds 540 --output-poll-bytes 536870912 \
  --max-constructors 200 --max-state-writes 24007
CUDA_VISIBLE_DEVICES='' python3 -I -B run_preparation.py can-package /absolute/new-can-packages \
  --extraction-root /absolute/new-can-extraction \
  --input-map INPUT_MAP.json --vendor /absolute/runtime \
  --manifest-sha256 MANIFEST_SHA --seconds 90 --output-poll-bytes 536870912 \
  --max-constructors 0 --max-state-writes 0
```

Extraction restores each demonstration's native model XML and recorded simulator states, refreshes the observations and performs the original cache-order checks. It creates 200 environments, not 200 task rollouts: env.step is prohibited. The 24,007 explicit dataset/cache-probe state writes include the original 23,207 recorded rows plus four cache probes per demonstration. Environment construction and reset include their normal simulator initialization; they are accounted separately from control actions. The meter distinguishes explicit resets from resets nested inside XML restoration.

Packaging requires a completed extraction receipt and matching input/recipe identities. It rechecks all 200 intermediate arrays against their measurement logs and raw actions before producing run/train.npz (180 demonstrations) and run/valid.npz (20). The observation reconstruction route is not replaced with the provider's low-dimensional Can file.

## Identity and limits

| Output | Required SHA-256 |
|---|---|
| Can train | fcc10895ab27cf906a535c7524089776faf9287c68f2a2024a6b2561a74aa735 |
| Can valid | 454f389992f33b8cf67dcb2c3ae3714feb6baa8b8abe0dd8dcbb51ea2cc3dc5f |
| Square train | 2c701e737fc5690b2df18c31d410be73f1e8df85a79a22a4b83ffbf26ac8fa7c |
| Square development | 0df7e39a8aae4626fbb0070a79f7933db42eb42878e093beb96c0b60b02b42bd |

Each phase writes preparation-receipt.json and an actual-operation meter. Completed receipts and the output hashes are required; exit status alone or an incomplete output directory is insufficient. Dataset trajectory accesses, NPZ calls, environment calls and explicit state writes are counted separately. A partial-budget stop is not completed preparation. The aggregate output threshold is polled rather than an operating-system disk quota. These entries do not train models or establish an independent full-training replication.
