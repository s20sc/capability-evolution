"""Reconstruct the fixed Can or Square learning inputs in a fresh workspace."""
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


def read(path):
    return json.loads(Path(path).read_text())


class LimitStop(ValueError):
    pass


class Meter:
    def __init__(self, root, seconds, output_bytes, caps):
        self.root, self.seconds, self.output_bytes = root, seconds, output_bytes
        self.caps, self.counts = caps, {}
        self.started, self.last_size_poll = time.monotonic(), 0.
        self.stream = (root / "operation-meter.jsonl").open("x")

    def check(self):
        now = time.monotonic()
        if now - self.started >= self.seconds:
            raise LimitStop("preparation wall budget")
        if now - self.last_size_poll >= .25:
            self.last_size_poll = now
            if sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file()) > self.output_bytes:
                raise LimitStop("preparation output polling threshold")

    def event(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        self.stream.write(json.dumps({"event": key, "count": self.counts[key]}) + "\n")
        self.stream.flush()

    def call(self, key, function, *args, **kwargs):
        self.check()
        self.event(key + "_attempts")
        if self.counts[key + "_attempts"] > self.caps.get(key, 0):
            raise LimitStop("preparation operation budget: " + key)
        self.event(key + "_entries")
        result = function(*args, **kwargs)
        self.event(key + "_returns")
        return result


def install_array_meters(meter, np, h5py, protected):
    original_get = h5py.Dataset.__getitem__
    original_save, original_load = np.savez_compressed, np.load

    def get(dataset, item):
        parts = dataset.name.split("/")
        if len(parts) > 2 and parts[1] == "data" and parts[2] in protected:
            meter.event("protected_trajectory_read_attempts")
            raise ValueError("protected provider-validation trajectory access")
        return meter.call("hdf5_read", original_get, dataset, item)

    h5py.Dataset.__getitem__ = get
    np.savez_compressed = lambda *a, **k: meter.call("npz_save", original_save, *a, **k)
    np.load = lambda *a, **k: meter.call("npz_load", original_load, *a, **k)


def install_native_meters(meter):
    import robosuite as suite
    from robosuite.environments.base import MujocoEnv
    original_make, original_action = suite.make, MujocoEnv.step
    MujocoEnv.step = lambda *a, **k: meter.call("action", original_action, *a, **k)

    def make(*args, **kwargs):
        env = meter.call("constructor", original_make, *args, **kwargs)
        reset, xml, action = env.reset, env.reset_from_xml_string, env.step
        context = {"in_xml": False, "bound_sim": None}

        def bind_sim():
            sim = env.sim
            if context["bound_sim"] is sim:
                return
            context["bound_sim"] = sim
            state_write, sim_reset = sim.set_state_from_flattened, sim.reset
            sim.set_state_from_flattened = lambda *a, **k: meter.call("state_write", state_write, *a, **k)
            sim.reset = lambda *a, **k: meter.call("post_reset_sim_reset", sim_reset, *a, **k)

        def counted_reset(*a, **k):
            key = "nested_reset" if context["in_xml"] else "explicit_reset"
            value = meter.call(key, reset, *a, **k)
            bind_sim()
            return value

        def counted_xml(*a, **k):
            context["in_xml"] = True
            try:
                value = meter.call("xml_restore", xml, *a, **k)
            finally:
                context["in_xml"] = False
            bind_sim()
            return value

        env.reset, env.reset_from_xml_string = counted_reset, counted_xml
        env.step = lambda *a, **k: meter.call("action", action, *a, **k)
        return env

    suite.make = make


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["square", "can-extract", "can-package"])
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--input-map", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--extraction-root", type=Path)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--output-poll-bytes", type=int, required=True)
    parser.add_argument("--max-constructors", type=int, required=True)
    parser.add_argument("--max-state-writes", type=int, required=True)
    args = parser.parse_args()
    if (not __debug__ or args.seconds <= 0 or args.output_poll_bytes <= 0
            or min(args.max_constructors, args.max_state_writes) < 0):
        raise ValueError("positive wall/output and explicit nonnegative operation limits required")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if os.environ["CUDA_VISIBLE_DEVICES"] != "":
        raise ValueError("preparation uses no visible CUDA device")
    package = Path(__file__).resolve().parent
    raw = (package / "PACKAGE_IDENTITIES.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.manifest_sha256:
        raise ValueError("preparation package identity")
    for name, info in json.loads(raw)["files"].items():
        path = package / name
        if path.is_symlink() or not path.resolve().is_relative_to(package):
            raise ValueError("nonlocal package member")
        b = path.read_bytes()
        if len(b) != info["bytes"] or hashlib.sha256(b).hexdigest() != info["sha256"]:
            raise ValueError("changed package member: " + name)
    helpers = load("preparation_helpers", package / "preparation/helpers.py")
    recipe = read(package / "recipes/preparation.json")
    inputs = read(args.input_map)
    vendor = args.vendor.resolve(strict=True)
    for name, expected in recipe["native_source_bindings"].items():
        path = vendor / name
        if path.is_symlink() or not path.resolve().is_relative_to(vendor) or helpers.sha(path) != expected:
            raise ValueError("bound native source identity: " + name)

    def input_file(role, expected):
        item = inputs[role]
        path = Path(item["path"])
        if (path.is_symlink() or item["sha256"] != expected or helpers.sha(path) != expected):
            raise ValueError("fixed provider input identity: " + role)
        return str(path.resolve(strict=True))

    root = args.workspace.resolve()
    if root == package or package in root.parents or root in package.parents:
        raise ValueError("fresh output must be outside the package")
    task = "Square" if args.phase == "square" else "Can"
    p = copy.deepcopy(recipe[task])
    p["phase"] = args.phase
    p["soft_seconds"] = args.seconds
    p["recipe_sha256"] = helpers.sha(package / "recipes/preparation.json")
    protected = []
    if task == "Square":
        sources = {name: input_file(role, p["input_sha256"][name]) for name, role in p["source_roles"].items()}
        protected = read(package / "preparation/Square_split.json")["protected_official_valid_ids"]
    else:
        p["dataset"] = input_file(p["source_role"], p["dataset_sha256"])
        p["robosuite_site_packages"] = str(vendor)
    extraction = None
    if args.phase == "can-package":
        if args.extraction_root is None:
            raise ValueError("complete extraction output required")
        extraction = args.extraction_root.resolve(strict=True)
        e, s = read(extraction / "run/protocol.json"), read(extraction / "run/status.json")
        receipt = read(extraction / "preparation-receipt.json")
        digest = helpers.sha(extraction / "run/protocol.json")
        if (s["state"] != "completed" or s["protocol_sha256"] != digest
                or receipt["status"] != "completed" or receipt["phase"] != "can-extract"
                or receipt["protocol_sha256"] != digest or e["recipe_sha256"] != p["recipe_sha256"]
                or e["dataset_sha256"] != p["dataset_sha256"] or helpers.sha(Path(e["dataset"])) != p["dataset_sha256"]):
            raise ValueError("complete same-input extraction provenance required")
        if (e.get("phase") != "can-extract"
                or any(e.get(key) != expected for key, expected in recipe["Can"].items())):
            raise ValueError("fixed extraction scientific recipe fields required")
        if ([row["demo"] for row in s["demos"]] != p["demos"]
                or len(s["demos"]) != 200
                or any(s["costs"][name] != 0 for name in ["simulation_actions", "gradient_updates", "external_model_calls"])
                or s["costs"]["state_write_attempts"] != 24007
                or s["costs"]["observation_refresh_attempts"] != 24007):
            raise ValueError("complete reconstruction counters and ordered demonstrations required")
    root.mkdir(exist_ok=False)
    (root / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    (root / "cache").mkdir()
    for name in ["MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"]:
        os.environ[name] = str(root / "cache" / name.lower())
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path.insert(0, str(vendor))
    caps = {"hdf5_read": 1444 if task == "Square" else 402,
            "npz_save": 200 if args.phase == "can-extract" else 2,
            "npz_load": 200 if args.phase == "can-package" else 0,
            "constructor": args.max_constructors, "explicit_reset": args.max_constructors,
            "nested_reset": args.max_constructors, "xml_restore": args.max_constructors,
            "post_reset_sim_reset": args.max_constructors, "state_write": args.max_state_writes,
            "action": 0}
    meter = Meter(root, args.seconds, args.output_poll_bytes, caps)
    report = {"status": "running", "phase": args.phase, "protocol_sha256": helpers.sha(root / "protocol.json"),
              "observed": meter.counts, "caps": caps,
              "scope": "fixed demonstration preparation; reconstruction of recorded states is not task execution"}
    try:
        import numpy as np
        import h5py
        if np.__version__ != "1.26.0" or h5py.__version__ != "3.16.0":
            raise ValueError("supported preparation array/HDF5 runtime")
        install_array_meters(meter, np, h5py, protected)
        if args.phase == "square":
            out = root / "run"
            out.mkdir()
            module = load("square_preparation", package / "preparation/square.py")
            counters = {"array_read_attempts": 0, "array_read_completions": 0}
            result = module.audit_and_package(Path(sources["low_dim"]), Path(sources["raw"]),
                package / "preparation/Square_split.json", package / "preparation/Square_anchor.jsonl.gz",
                out, p["audit_limits"], meter.check, counters)
            for role, expected in p["expected_packages"].items():
                if result["packages"][role]["sha256"] != expected["sha256"]:
                    raise ValueError("prepared Square package identity: " + role)
            report.update(packages=result["packages"], native_counters=counters,
                          protected_valid_array_reads=result["protected_valid_array_reads"])
        else:
            module = load("can_preparation", package / "preparation/can.py")
            if args.phase == "can-extract":
                install_native_meters(meter)
                module.extract(root, p)
            else:
                module.package(root, p, extraction, e, s)
                for filename, expected in p["expected_packages"].items():
                    if helpers.sha(root / "run" / filename) != expected:
                        raise ValueError("prepared Can package identity: " + filename)
        report["status"] = "completed"
    except LimitStop as exc:
        report.update(status="budget_stop_partial", reason=str(exc))
    except BaseException as exc:
        report.update(status="failed_no_retry", error=repr(exc))
        raise
    finally:
        meter.stream.close()
        report["elapsed_seconds"] = time.monotonic() - meter.started
        (root / "preparation-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
