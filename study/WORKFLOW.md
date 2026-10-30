# Reproduction workflow

Use three distinct scopes: recomputing the reported summaries from saved observations, repeating a prescribed learning intervention from its complete parent, and evaluating the resulting policy. A successful source installation or short operation does not establish that a complete study has been repeated.

## Saved observations

DATA_GUIDE.md maps the eight manuscript tables and three figures to results/ and the saved multicycle analysis inputs. Run analysis/reanalyse.py as described in README.md to recompute the multicycle summary. The other studies retain their case-level outcomes and original comparison records. Keep the new-scene, development and selected-old panels separate. Do not treat repeated panels or shared pretrained ancestry as independent training runs.

## Inputs and environment

Install the version-pinned Python dependencies and use the separately supplied fixed runtime-source companion. DEPENDENCIES.md lists the native sources, assets, graphics requirements and provider files. Extract the learned-state and generated-experience companions separately, then fill INPUT_MAP_EXAMPLE.json with the actual local file locations. Use the distributed state hashes, whose deployment paths differ from the original serialized inputs; the scientific state roles and stage counters remain distinct.

Can preparation reconstructs observations from the raw v141 simulator states; Square preparation uses the v141 low-dimensional provider observations. The exact URLs, ordered demonstration IDs and prepared data hashes are in the recipes. Using a newer dataset, a new split or weights alone changes the study input. DATA_PREPARATION.md gives the supplied raw-demonstration preparation entry. Upstream diagnostic and experience-generation code remain separate from the main operators; see the coverage limits below.

## Acquisition and multicycle order

WORKFLOW_ORDER.json gives a dependency order derived from each recipe's physical jobs, complete-parent references and paired baseline references. It preserves every job once. Cross-recipe origins remain explicit: acquisition-endpoints evaluates the policy produced by acquisition, rather than retraining a different policy. The multicycle order contains the recorded six continuation initializations, 66 physical learning nodes and evaluation jobs. The logical study schedules reuse some physical nodes; do not execute them again as independent observations.

For each native operation, use the command format in OPERATORS.md, with the recipe method and declared job ID. Use the job's wall_seconds and original caps to choose explicit operation limits for a complete run. Low acceptance-probe limits stop a partial operation; their output is not a complete child state or panel.

Create a reference map for each learning/evaluation job containing exactly its declared references. For the common source, use {"input_role":"shared_S0"}. For a continuation or learned node, map its logical name to the checkpoint sidecar JSON emitted by the specified predecessor job. The sidecar is beside the PT file and contains portable_origin; the nested checkpoint record in a result wrapper is not a substitute for that sidecar. Store each operation in a fresh workspace. Never replace a missing predecessor with the common source.

Initialize the multicycle seed roots before their dependent learning nodes. For a later learning stage, use the parent specified by the graph, not the best observed policy. Evaluate only after the required policy and any paired baseline board are complete. Pass --pairing-reference the predecessor's jobs/JOB_ID/result.json when a native recipe declares a paired baseline. Use the acquisition learning recipe for development panels and acquisition-endpoints for the separate 32-scene endpoints. Do not mix their baseline boards.

## Correction order

First evaluate the original Square100000 parent separately on the chosen panel using --state-role Square100000 and --panel new, development or selected-old. The new panel has 32 seeds, development has four, and selected-old has five; each uses the original single action stream. These are distinct panels, not additional training seeds.

Train C0, LP, FP, LG and FG independently from D101000 with the fixed recipe and their corresponding experience pack. This restores the complete D learning state; it does not train from the original Square parent. Each full arm uses 1,000 updates and saves its own child. Preserve FG's recorded fallback composition.

Evaluate each completed child using --start-reference pointing to its completed training status or checkpoint reference. Supply --pairing-reference pointing to the completed original-parent run/status.json for the same panel and retain its adjacent protocol.json. For D evaluation use --state-role D101000 with the same parent reference. The interface checks the original-parent identity, panel settings and initial states. Native low-noise sampling here differs from the maximum-probability component decoder in acquisition and multicycle evaluation.

## Coverage limits

This package exposes the main operators and their dependency order, not an automated whole-study launcher. The raw-demonstration preparation entry is supplied separately. C/D source and experience preparation/collection/repacking entries are supplied with their fixed roles. The fixed P/A/B/C/D diagnostic paths have been recreated against their original complete states and case records in the measured environment. Fresh native-generation validation and all required data access retain the limits stated in the corresponding guides. The provided complete parents and five generated-experience arrays permit the stated downstream interventions without treating their preparation as a newly replicated result. No clean-machine full-training reproduction is claimed.

EXPERIENCE.md supplies the source-preparation and original collection methods plus record-repacking commands. The original target and four correction learning arrays were reconstructed byte-for-byte from their recorded transitions using these entries. The saved data are distinct from fresh native generation; the native preparation/collection paths retain separate validation limits. P/A/B source and original-panel evaluation are described in FEEDBACK_DIAGNOSTICS.md. The supplied complete parents remain valid downstream inputs.
