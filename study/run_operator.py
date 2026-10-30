"""Run one declared learning or evaluation operation in a new workspace."""
from __future__ import annotations
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


def correction_baseline(path, expected, parent_identity):
    """Require the original parent's complete, identically configured panel."""
    path = Path(path)
    prior = json.loads(path.read_text())
    protocol_raw = (path.parent / "protocol.json").read_bytes()
    if hashlib.sha256(protocol_raw).hexdigest() != prior.get("protocol_sha256"):
        raise ValueError("baseline protocol identity")
    protocol = json.loads(protocol_raw)
    if (protocol.get("checkpoint_sha256") != parent_identity
            or protocol.get("selected_update") != 100000
            or protocol.get("evolution_receipt") is not None):
        raise ValueError("baseline must be the original Square parent")
    for field in ["seeds", "repeats", "horizon", "success_hold", "action_seed_offset",
                  "compiled_limits", "timestep", "workspace_xyz", "actuator_tolerance",
                  "mujoco_version", "training_protocol_sha256", "config", "metadata_Square_sha256"]:
        if protocol.get(field) != expected[field]:
            raise ValueError("baseline evaluation setting differs: " + field)
    deployment = prior.get("deployment", {})
    if deployment.get("low_noise_eval") is not True or deployment.get("rnn_reset_each_episode") is not True:
        raise ValueError("baseline must use the original low-noise/reset policy")
    pairs = [(seed, repeat) for seed in expected["seeds"] for repeat in range(expected["repeats"])]
    cases = prior["cases"]
    if (prior["status"] != "completed" or len(cases) != len(pairs)
            or [(c["seed"], c["repeat"]) for c in cases] != pairs
            or any(c["stop_reason"] == "not_started" for c in cases)):
        raise ValueError("complete same-seed/stream parent panel required")
    if any(c["initial_state_sha256"] != expected["expected_initial_states"][str(c["seed"])] for c in cases):
        raise ValueError("baseline differs from recorded initial states")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["acquisition", "acquisition-endpoints", "multicycle", "correction"])
    parser.add_argument("action", choices=["train", "evaluate", "initialize"])
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--input-map", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--job")
    parser.add_argument("--arm", choices=["C0", "LP", "FP", "LG", "FG"], default="LP")
    parser.add_argument("--start-reference", type=Path)
    parser.add_argument("--reference-map", type=Path)
    parser.add_argument("--pairing-reference", type=Path)
    parser.add_argument("--state-role", choices=["Square100000", "D101000"], default="Square100000")
    parser.add_argument("--panel", choices=["new", "development", "selected-old"])
    parser.add_argument("--seconds", type=int, default=450)
    parser.add_argument("--max-optimizer-mutations", type=int, required=True)
    parser.add_argument("--max-actions", type=int, required=True)
    parser.add_argument("--max-cases", type=int, required=True)
    args = parser.parse_args()
    if (not __debug__ or min(args.max_optimizer_mutations, args.max_actions, args.max_cases) < 0
            or args.seconds <= 0):
        raise ValueError("explicit nonnegative budgets and unoptimized Python required")
    expected_visibility = "" if args.action == "evaluate" else "0"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", expected_visibility)
    if os.environ["CUDA_VISIBLE_DEVICES"] != expected_visibility:
        raise ValueError("use CUDA device0 for learning, no visible CUDA device for CPU evaluation")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("fixed deterministic CUDA workspace required")
    package = Path(__file__).resolve().parent
    manifest = (package / "PACKAGE_IDENTITIES.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != args.manifest_sha256:
        raise ValueError("package manifest identity")
    for name, info in json.loads(manifest)["files"].items():
        path = package / name
        if path.is_symlink() or not path.resolve().is_relative_to(package):
            raise ValueError("nonlocal package member")
        raw = path.read_bytes()
        if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise ValueError("package member changed: " + name)
    root = args.workspace.resolve()
    if root == package or package in root.parents or root in package.parents:
        raise ValueError("new workspace must be outside the package")
    inputs = json.loads(args.input_map.read_text())
    roots = json.loads((package / "ROOT_STATE_IDENTITIES.json").read_text())["states"]
    for role, info in inputs.items():
        info["path"] = str(Path(info["path"]).resolve(strict=True))
        if role in roots and info["sha256"] != roots[role]["sha256"]:
            raise ValueError("published starting-state identity: " + role)
        if hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("actual declared input identity: " + role)
    recipe_bytes = (package / "recipes" / (args.method + ".json")).read_bytes()
    recipe = json.loads(recipe_bytes)
    p = copy.deepcopy(recipe)
    p["recipe_identity"] = hashlib.sha256(recipe_bytes).hexdigest()
    vendor = args.vendor.resolve(strict=True)
    p["vendor"] = str(vendor)
    if args.method != "correction":
        if args.job not in p["jobs"]:
            raise ValueError("choose a declared recipe job")
        definition = p["jobs"][args.job]
        expected = {"train": "train", "evaluate": "evaluate", "initialize": "initialize_continuation"}[args.action]
        if definition["kind"] != expected:
            raise ValueError("action differs from declared job kind")
        p["evaluation_runtime"]["robosuite_site_packages"] = str(vendor)
        p["native_source_files"] = [{"path": str(vendor / row["relative"]), "sha256": row["sha256"]} for row in p["native_source_files"]]
        default = copy.deepcopy(p["parent_reference"])
        default.pop("input_role")
        default.update(path=inputs["shared_S0"]["path"], sha256=inputs["shared_S0"]["sha256"],
                       bytes=Path(inputs["shared_S0"]["path"]).stat().st_size)
        if args.action == "train" and args.start_reference is None and args.reference_map is None:
            raise ValueError("provide the specific training parent reference")
        if args.action == "evaluate" and args.reference_map is None:
            raise ValueError("provide every declared task route explicitly")
        start = json.loads(args.start_reference.read_text()) if args.start_reference else default
        p["parent_reference"] = default
    else:
        if args.action == "initialize":
            raise ValueError("correction uses the specified learned parent")
        definition = {"id": args.action + "_" + args.arm, "kind": args.action, "caps": {}}
        p["python"] = sys.executable
        p["root"] = str(root)
        p["original_data"]["path"] = inputs["data_Square"]["path"]
        p["target_data"]["path"] = inputs["target_pack"]["path"]
        for name, expected in [("data_Square", p["original_data"]["sha256"]),
                               ("target_pack", p["target_data"]["sha256"])]:
            if inputs[name]["sha256"] != expected:
                raise ValueError("fixed correction learning input: " + name)
        p["parent"].update(path=inputs["D101000"]["path"], sha256=inputs["D101000"]["sha256"])
        if args.action == "evaluate":
            if args.panel is None:
                raise ValueError("choose the distinct new/development/selected-old evaluation panel")
            p.update(p["evaluation_template"])
            panel_seeds = {"new": recipe["new_evaluation_seeds"],
                           "development": recipe["evaluation_template"]["seeds"],
                           "selected-old": recipe["acquisition_seeds"]}
            p["seeds"] = panel_seeds[args.panel]
            p["max_actions"] = len(p["seeds"]) * p["repeats"] * p["horizon"]
            p["expected_initial_states"] = copy.deepcopy(recipe["fixed_initial_state_hashes"])
            ref = inputs[args.state_role]
            selected_update = 100000 if args.state_role == "Square100000" else 101000
            evolution = None if args.state_role == "Square100000" else p["parent"]["evolution"]
            if args.start_reference:
                ref = json.loads(args.start_reference.read_text())
                ref = ref.get("checkpoint", ref)
                selected_update = ref.get("updates", ref.get("counters", {}).get("updates"))
                evolution = ref.get("evolution", ref.get("binding", {}).get("evolution"))
                if selected_update is None or hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() != ref["sha256"]:
                    raise ValueError("declared correction endpoint identity/counter")
            p["checkpoint"] = ref["path"]
            p["checkpoint_sha256"] = ref["sha256"]
            p["selected_update"] = selected_update
            p["dataset"] = inputs["metadata_Square"]["path"]
            if inputs["metadata_Square"]["sha256"] != p["metadata_Square_sha256"]:
                raise ValueError("fixed evaluation metadata dataset")
            p["training_protocol"] = str(root / "training-config.json")
            p["robosuite_site_packages"] = str(vendor)
            p["evolution_receipt"] = evolution
            needs_baseline = args.state_role != "Square100000" or args.start_reference is not None
            if needs_baseline and args.pairing_reference is None:
                raise ValueError("a matching complete original-parent panel is required")
            if args.pairing_reference:
                cases = correction_baseline(args.pairing_reference, p, roots["Square100000"]["sha256"])
                p["expected_initial_states"].update({str(c["seed"]): c["initial_state_sha256"] for c in cases})
    root.mkdir(exist_ok=False)
    (root / "cache").mkdir()
    for key, relative in [("MPLCONFIGDIR", "mpl"), ("NUMBA_CACHE_DIR", "numba"), ("XDG_CACHE_HOME", "xdg")]:
        os.environ[key] = str(root / "cache" / relative)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    (root / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    if args.method != "correction":
        references = {definition["start_checkpoint_reference"]} if args.action == "train" else set(definition.get("routes", {}).values())
        reference_map = json.loads(args.reference_map.read_text()) if args.reference_map else {name: start for name in references}
        if set(reference_map) != references:
            raise ValueError("exact declared reference set required")
        for name in references:
            info = reference_map[name]
            if isinstance(info, str):
                info = json.loads(Path(info).read_text())
            if info.get("input_role") == "shared_S0":
                if p["expected_reference_origins"][name] != {"input_role": "shared_S0"}:
                    raise ValueError("a continuation/child reference cannot use the common root")
                info = copy.deepcopy(default)
            if info.get("counters", {}).get("updates") != p["expected_reference_updates"][name]:
                raise ValueError("reference has wrong learning-stage counter: " + name)
            expected_origin = p["expected_reference_origins"][name]
            if "input_role" in expected_origin and info["sha256"] != inputs[expected_origin["input_role"]]["sha256"]:
                raise ValueError("common source cannot be replaced by a reseeded continuation")
            if "job" in expected_origin:
                origin = info.get("portable_origin", {})
                recipe_name = expected_origin["recipe"]
                expected_identity = hashlib.sha256((package / "recipes" / (recipe_name + ".json")).read_bytes()).hexdigest()
                if (origin.get("job") != expected_origin["job"] or origin.get("recipe_identity") != expected_identity
                        or ("continuation_seed" in expected_origin and origin.get("continuation_seed") != expected_origin["continuation_seed"])):
                    raise ValueError("wrong continuation/learning-node provenance: " + name)
            if hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError("reference checkpoint identity")
            path = root / name
            if not path.resolve().is_relative_to(root):
                raise ValueError("nonlocal job reference")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(info, indent=2) + "\n")
        pairing = definition.get("pairing_reference")
        if pairing is not None:
            if args.pairing_reference is None:
                raise ValueError("a complete matching baseline board is required")
            path = root / pairing
            if not path.resolve().is_relative_to(root):
                raise ValueError("nonlocal baseline reference")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(args.pairing_reference.read_bytes())
        update = root / definition.get("U_reference", "updates/unit.json")
        update.parent.mkdir(parents=True, exist_ok=True)
        update.write_text(json.dumps(p["fixed_update"]) + "\n")
    else:
        packs = {}
        for arm, info in p["correction_packs"].items():
            if arm in inputs:
                if inputs[arm]["sha256"] != info["sha256"]:
                    raise ValueError("fixed correction pack: " + arm)
                packs[arm] = {"path": inputs[arm]["path"], "sha256": info["sha256"], "rows": info["rows"]}
        (root / "packs.json").write_text(json.dumps(packs, indent=2) + "\n")
        (root / "training-config.json").write_text(json.dumps({"config": p["config"]}, indent=2) + "\n")
    sys.path.insert(0, str(vendor))
    ops = package / "operators"
    load("expression", ops / "expression.py")
    contracts = load("contracts", ops / "contracts.py")
    load("state_core", package / "learning/complete_state.py")
    common = load("common", ops / "common.py")
    load("baseline_core", package / "learning/complete_state.py")
    load("reuse", ops / "reuse.py")
    for name in ["decoder", "native_guard", "counterfactual_target"]:
        load(name, ops / (name + ".py"))
    job = common.Job(root, p, definition, inputs, vendor, args.seconds, 128 * 1024 * 1024)
    common.ACTIVE = job
    route_count = len(set(definition.get("routes", {}).values()))
    limits = {"models": max(1, route_count), "optimizer_mutations": args.max_optimizer_mutations, "actions": args.max_actions,
              "environment_constructors": args.max_cases, "checkpoint_loads": 2,
              "policy_forwards": 2 * args.max_actions + 2 * (args.max_optimizer_mutations + 1),
              "gradients": args.max_optimizer_mutations + 1}
    receipt = {"status": "running", "method": args.method, "action": args.action,
               "limits": limits, "observed": job.observed,
               "scope": "one declared operation; budget-stop is partial work, not a completed study"}
    started = time.monotonic()
    try:
        import torch
        common.install_meters(job, limits, torch)
        if args.method != "correction":
            module = load("learner", ops / ("learner_" + args.method.split("-")[0] + ".py"))
            if args.action == "evaluate":
                load("native_eval", ops / "native_eval.py").evaluate(job)
            elif args.action == "initialize":
                module.initialize_target(job)
            else:
                module.train(job)
        else:
            sys.argv = ["train_correction.py", args.arm]
            module = load("correction_operator", ops / ("train_correction.py" if args.action == "train" else "evaluate_correction.py"))
            module.main()
        receipt["status"] = "completed"
    except contracts.LocalBudgetStop as exc:
        receipt.update(status="budget_stop_partial", reason=str(exc))
    except BaseException as exc:
        receipt.update(status="failed_no_retry", error=repr(exc))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["native_counts"] = job.counts
        (root / "operator-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
