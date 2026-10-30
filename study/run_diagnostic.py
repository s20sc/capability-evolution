"""Run one fixed C/D training diagnostic or its original native evaluation."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(package, args):
    """Resolve declared study inputs before importing numerical libraries."""
    def check_time():
        if time.monotonic() >= getattr(args, "deadline", float("inf")):
            raise TimeoutError("diagnostic input/setup time budget")

    check_time()
    manifest = package / "PACKAGE_IDENTITIES.json"
    if sha(manifest) != args.manifest_sha256:
        raise ValueError("package identity")
    for name, info in json.loads(manifest.read_text())["files"].items():
        f = package / name
        if f.is_symlink() or not f.resolve().is_relative_to(package) or f.stat().st_size != info["bytes"] or sha(f) != info["sha256"]:
            raise ValueError("package member identity: " + name)
        check_time()
    recipe_file = package / "recipes/diagnostics.json"
    p = copy.deepcopy(json.loads(recipe_file.read_text())[args.method])
    p["recipe_identity"] = sha(recipe_file)
    vendor = args.vendor.resolve(strict=True)
    p["vendor"] = str(vendor)
    for name, identity in p["official_sources"].items():
        if sha(vendor / "robomimic" / name) != identity:
            raise ValueError("native source identity: " + name)
    inputs = json.loads(args.input_map.read_text())

    def bind(role, expected):
        i = inputs[role]
        f = Path(i["path"])
        if f.is_symlink() or i["sha256"] != expected or sha(f) != expected:
            raise ValueError("fixed study input: " + role)
        check_time()
        return str(f.resolve(strict=True))

    p["parent"]["path"] = bind("Square100000", p["parent"]["sha256"])
    p["original_data"]["path"] = bind("data_Square", p["original_data"]["sha256"])
    target_sha = "2d97f6d6d15ab77de875485280330754196dbb1776be380d98970933ea6ae361"
    proposal = json.loads((package / "diagnostics/selection-specification.json").read_text())
    proposal.update(parent=p["parent"], pack={"path": bind("target_pack", target_sha), "sha256": target_sha})
    for field, file in [("parent_evidence", "parent-evidence.json"), ("acquisition_evidence", "acquisition-evidence.json")]:
        f = package / "diagnostics" / file
        proposal[field] = {"path": str(f), "sha256": sha(f)}
    root = args.workspace.resolve()
    if root == package or package in root.parents or root in package.parents:
        raise ValueError("fresh workspace outside source package required")
    if args.action == "evaluate":
        if args.start_reference is None:
            raise ValueError("completed diagnostic training reference required")
        reference = json.loads(args.start_reference.read_text())
        if reference.get("state") != "completed" or reference.get("updates_returned") != 1000:
            raise ValueError("complete diagnostic training required")
        ref = reference["checkpoint"]
        evolution = ref["evolution"]
        training_p = json.loads((args.start_reference.parent.parent / "protocol.json").read_text())
        # A reported recipe digest cannot replace checking the actual fields.
        for key, expected in p.items():
            observed = training_p.get(key)
            if key == "vendor":
                continue
            if key in {"parent", "original_data"}:
                observed = {k: v for k, v in observed.items() if k != "path"} if isinstance(observed, dict) else observed
                expected = {k: v for k, v in expected.items() if k != "path"}
            if observed != expected:
                raise ValueError("actual diagnostic training recipe differs: " + key)
        training_proposal = args.start_reference.parent.parent / "proposal.json"
        if (training_p.get("proposal_sha256") != sha(training_proposal)
                or evolution.get("proposal_sha256") != sha(training_proposal)):
            raise ValueError("actual training proposal identity")
        recorded = json.loads(training_proposal.read_text())
        for key in ["selected_demo_ids", "selected_rows", "initial_state_sha256", "scope"]:
            if recorded.get(key) != proposal[key]:
                raise ValueError("training selection differs: " + key)
        for key in ["parent_evidence", "acquisition_evidence", "pack"]:
            if recorded[key]["sha256"] != proposal[key]["sha256"]:
                raise ValueError("training evidence identity differs: " + key)
        if (ref["updates"] != 101000 or sha(ref["path"]) != ref["sha256"]
                or evolution["protocol_sha256"] != sha(args.start_reference.parent.parent / "protocol.json")
                or training_p["recipe_identity"] != p["recipe_identity"]
                or training_p["diagnostic_method"] != args.method
                or evolution["parent_sha256"] != p["parent"]["sha256"]):
            raise ValueError("diagnostic child lineage")
        metadata_sha = json.loads((package / "recipes/correction.json").read_text())["metadata_Square_sha256"]
        dataset = bind("metadata_Square", metadata_sha)
        p.update(p["evaluation_template"])
        p.update(checkpoint=ref["path"], checkpoint_sha256=ref["sha256"], selected_update=101000,
                 evolution_receipt=evolution, dataset=dataset, dataset_sha256=metadata_sha,
                 training_protocol=str(root / "training-config.json"),
                 robosuite_site_packages=str(vendor), expected_initial_states=p["fixed_initial_states"],
                 seeds=p["development_seeds"] + p["evaluation_panel_seeds"], soft_seconds=args.seconds,
                 max_actions=6400)
    p["diagnostic_method"] = args.method
    check_time()
    root.mkdir(exist_ok=False)
    args.created_workspace = str(root)
    proposal_file = root / "proposal.json"
    proposal_file.write_text(json.dumps(proposal, indent=2) + "\n")
    p["proposal_sha256"] = sha(proposal_file)
    (root / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    (root / "training-config.json").write_text(json.dumps({"config": p["config"]}, indent=2) + "\n")
    return root, p, inputs, vendor


def main():
    started = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("method", choices=["C", "D"])
    ap.add_argument("action", choices=["train", "evaluate"])
    ap.add_argument("workspace", type=Path)
    ap.add_argument("--input-map", type=Path, required=True)
    ap.add_argument("--vendor", type=Path, required=True)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--start-reference", type=Path)
    ap.add_argument("--seconds", type=int, required=True)
    ap.add_argument("--max-optimizer-mutations", type=int, required=True)
    ap.add_argument("--max-actions", type=int, required=True)
    ap.add_argument("--max-cases", type=int, required=True)
    args = ap.parse_args()
    args.deadline = started + args.seconds
    if (not __debug__ or not sys.dont_write_bytecode or args.seconds <= 0
            or min(args.max_optimizer_mutations, args.max_actions, args.max_cases) < 0):
        raise ValueError("unoptimized bytecode-disabled execution and explicit budgets required")
    expected = "0" if args.action == "train" else ""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
        raise ValueError("explicit GPU0 learning / CPU evaluation required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("deterministic CUDA workspace")
    package = Path(__file__).resolve().parent
    limits = {"models": 2 if args.method == "D" and args.action == "train" else 1,
              "optimizer_mutations": args.max_optimizer_mutations, "actions": args.max_actions,
              "environment_constructors": args.max_cases, "checkpoint_loads": 2,
              "policy_forwards": 2010 if args.action == "train" else args.max_actions,
              "gradients": args.max_optimizer_mutations}
    receipt = {"status": "running", "method": args.method, "action": args.action,
               "limits": limits, "observed": {}, "full_study_replication": False}
    contracts = None
    root = None
    try:
        root, p, inputs, vendor = prepare(package, args)
        for key in ["MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"]:
            os.environ[key] = str(root / "cache" / key.lower())
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        sys.path.insert(0, str(vendor))
        ops = package / "operators"
        load("expression", ops / "expression.py")
        contracts = load("contracts", ops / "contracts.py")
        common = load("common", ops / "common.py")
        load("baseline_core", package / "learning/complete_state.py")
        load("native_guard", ops / "native_guard.py")
        selection = load("diagnostic_selection", package / "diagnostics/selection.py")
        common.select_evidence = selection.select_evidence
        common.all_valid_ids = selection.all_valid_ids
        load("rehearsal", package / "diagnostics/rehearsal.py")
        definition = {"id": args.method + "_" + args.action, "caps": {}}
        remaining = args.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("diagnostic setup exhausted entry time budget")
        job = common.Job(root, p, definition, inputs, vendor, remaining, 268435456)
        job.observed.update(teacher_copy_attempts=0, teacher_copy_returns=0)
        common.ACTIVE = job
        receipt["observed"] = job.observed
        import torch
        if args.action == "train":
            metering = load("diagnostic_metering", package / "diagnostics/metering.py")
            metering.install(job, limits, torch)
            sys.argv = ["train_" + args.method + ".py", next(iter(p["arms"]))]
            module = load("diagnostic_training", package / "diagnostics" / ("train_" + args.method + ".py"))
        else:
            common.install_meters(job, limits, torch)
            module = load("diagnostic_evaluation", ops / "evaluate_correction.py")
        module.main()
        receipt["status"] = "completed"
    except BaseException as exc:
        if contracts is not None and isinstance(exc, contracts.LocalBudgetStop):
            receipt.update(status="budget_stop_partial", reason=str(exc))
        else:
            receipt.update(status="failed_no_retry", error=repr(exc))
            raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        if root is None and hasattr(args, "created_workspace"):
            root = Path(args.created_workspace)
        if root is not None:
            (root / "diagnostic-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
