# Capability evolution in embodied agents: study materials

This package supplies the study's prespecified saved-result summary, fixed-length learning windows and complete learned-state restoration methods. METHOD_IDENTITIES.json identifies the unchanged scientific functions. The three study panels and learning units remain distinct. Repeated cases are not independent training replicates; six continuation roots share learned ancestry.

## Saved-result analysis

With Python 3.10 or later, run `python3 -I -B analysis/reanalyse.py /absolute/new-output --manifest-sha256 MANIFEST_SHA`. The source and saved inputs are checked before the original statistical functions run; all summary fields must match the saved result. The input contains 506 job measurements and 434 evaluation boards. The program does not retrain policies, simulate tasks or supply an independent statistical implementation. Missing work remains unknown, and returned/reserved cost fields are retained.

## Trusted inputs and restoration safety

Complete-state restoration uses `torch.load(..., weights_only=False)` because these checkpoints include optimizer, sampler and other Python object state as well as model parameters. Loading this format can execute Python object reconstruction. Restore only artifacts obtained from a trusted author release, and verify the files against a manifest obtained through a trusted release channel before loading. A matching SHA-256 proves identity with that manifest; it does not prove that code or checkpoint content is harmless, and an attacker-supplied manifest does not establish trust.

Use an isolated, least-privilege environment without unrelated credentials or sensitive files. A read-only repository or reviewer link does not sandbox local execution. Do not disable validation or use these entry points to inspect unknown checkpoints. The supported full-state restoration procedure requires exactly one visible CUDA device; saved-result analysis and CPU task evaluation have their separately documented prerequisites. These materials do not certify arbitrary checkpoints, environments or dependencies as safe.

## Learned-state restoration

Extract the starting-state companion into its own directory. It contains STATE_IDENTITIES.json and states/*.pt. With the supported tensor stack and one visible CUDA device, run `python3 -I -B learning/restore_states.py /absolute/state-companion /absolute/new-output --package-manifest-sha256 MANIFEST_SHA --states-manifest-sha256 a4c15b1599758d32d7ca534d3cfe7ed56e479253d9c8876e18979eecccbb82df`. An optional `--vendor /absolute/package-directory` selects a separately installed source tree.

The interface verifies the three distributed file identities, constructs their BC_RNN_GMM models, restores the complete sampler/optimizer/scheduler/mode/RNN and CPU/CUDA random states, exports them again, and compares the saved fields and object attributes. Additional correction-learning random generators are restored separately. It performs no policy forward pass, optimizer update or simulation. Use the distributed hashes and adapted deployment fields, rather than the original file admission hashes.

The measured stack was Linux CPython 3.11.16, torch 2.7.0+cu128, numpy 1.26.0, robomimic 0.3.0 and RTX 3090/CUDA 12.8. environment/ records wheel inputs, required source identities and the three original local-build source distributions. It does not contain wheel bodies, a full OS/driver image or the separate fixed robomimic/robosuite source companion. Provider demonstration files, prepared learning data, native task assets and terminal policies are separate inputs.

## Demonstration preparation

DATA_PREPARATION.md describes run_preparation.py: the fixed Square correspondence/packaging method and the Can native observation reconstruction followed by split-preserving packaging. These entries use provider files with fixed identities and preserve the original scientific preparation bodies.

## Learning and task evaluation

WORKFLOW.md gives the study order and parent/baseline handling; WORKFLOW_ORDER.json enumerates the recorded physical job dependencies. DEPENDENCIES.md and environment/wheel-sources.json describe the separate fixed native-source companion and recorded package retrieval locations. The three original local-build source distributions, notices and build instructions are supplied in environment/. These materials do not supply a whole-study automatic launcher or claim an independently reproduced training run.

OPERATORS.md describes the fixed acquisition, multicycle and correction learning operators and their distinct task-evaluation modes. run_operator.py accepts explicit data/state maps, recipe jobs and operation limits. recipes/ retains the recorded job graph and sampling settings; DATA_GUIDE.md maps the manuscript's tables and figures to the saved measurements in results/. Generated experience and learned starting states are supplied separately; third-party demonstrations and dependencies retain their original terms.

## Scope

These entry points expose saved-value reanalysis, state restoration, the main fixed learning operators and task evaluation. They do not automate the whole dependency graph. The fixed P/A/B/C/D diagnostic learning and evaluation entries have been checked against the original complete states and case records in the measured environment; fresh native-generation validation remains separate. They do not establish clean-machine full-training replication. Full software/data access must accompany the completed reproduction materials.

For the original parent P and fixed A/B continuations, see [FEEDBACK_DIAGNOSTICS.md](FEEDBACK_DIAGNOSTICS.md). For C/D and their original native evaluation, see [DIAGNOSTICS.md](DIAGNOSTICS.md).

For the source-demonstration extraction, bounded experience collection and original-record data reconstruction, see [EXPERIENCE.md](EXPERIENCE.md). Five target/correction arrays can be reconstructed from the recorded-transition companion; fixed source roles, generation limits and third-party dependencies remain separate.

## Licensing

Original study software is licensed under Apache-2.0 (LICENSE); original documentation, saved measurements and analysis inputs are licensed under CC BY 4.0 (LICENSE-CC-BY-4.0.txt). See LICENSE_SCOPE.md for boundaries and THIRD_PARTY_NOTICES.md for retained third-party terms. The separate experience, starting-state and recorded-transition companions carry their own notices. Licensing does not remove the restoration safety requirements above.
