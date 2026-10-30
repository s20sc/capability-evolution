"""Restore the three distributed complete states into their native BC models."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import sys
import struct
import time
from types import SimpleNamespace
from collections import OrderedDict


def strict_equal(a, b, torch, np, trail=()):
    supported = {type(None), bool, int, float, str, bytes, dict, OrderedDict, tuple, list,
                 torch.Tensor, np.ndarray, torch.torch_version.TorchVersion}
    if type(a) is not type(b) or type(a) not in supported:
        raise ValueError("uncovered or changed saved type: " + repr(trail))
    count = 0
    left, right = getattr(a, "__dict__", {}), getattr(b, "__dict__", {})
    if left or right:
        count += strict_equal(left, right, torch, np, trail + ("<attributes>",))
    if isinstance(a, torch.Tensor):
        if (a.dtype != b.dtype or a.shape != b.shape or a.stride() != b.stride()
                or a.storage_offset() != b.storage_offset() or a.requires_grad != b.requires_grad
                or not torch.equal(a.detach().cpu().reshape(-1).view(torch.uint8),
                                   b.detach().cpu().reshape(-1).view(torch.uint8))):
            raise ValueError("tensor changed: " + repr(trail))
    elif isinstance(a, np.ndarray):
        if a.dtype.hasobject or a.dtype != b.dtype or a.shape != b.shape or a.strides != b.strides or a.tobytes() != b.tobytes():
            raise ValueError("array changed: " + repr(trail))
    elif isinstance(a, dict):
        if list(a) != list(b):
            raise ValueError("dictionary keys/order changed: " + repr(trail))
        return count + sum(strict_equal(a[k], b[k], torch, np, trail + (k,)) for k in a)
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise ValueError("sequence length changed")
        return count + sum(strict_equal(x, y, torch, np, trail + (i,)) for i, (x, y) in enumerate(zip(a, b)))
    elif (struct.pack("!d", a) != struct.pack("!d", b)) if isinstance(a, float) else a != b:
        raise ValueError("saved field changed: " + repr(trail))
    return count + 1


def checked(path, identity):
    if path.is_symlink():
        raise ValueError("symlink input")
    raw = path.read_bytes()
    if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        raise ValueError("file identity: " + path.name)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--states-manifest-sha256", required=True)
    parser.add_argument("--vendor", type=Path)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    raw = (package / "PACKAGE_IDENTITIES.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.package_manifest_sha256:
        raise ValueError("package manifest identity")
    identities = json.loads(raw)["files"]
    source = package / "learning/complete_state.py"
    checked(source, identities["learning/complete_state.py"])
    states = args.states.resolve(strict=True)
    manifest = (states / "STATE_IDENTITIES.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != args.states_manifest_sha256:
        raise ValueError("state manifest identity")
    rows = json.loads(manifest)["states"]
    if [row["role"] for row in rows] != ["shared_S0", "Square100000", "D101000"]:
        raise ValueError("three distinct declared state roles required")
    for row in rows:
        if row["member"] != "states/" + row["role"] + ".pt":
            raise ValueError("canonical state member")
    output = args.output.resolve()
    for input_root in [package, states]:
        if output == input_root or input_root in output.parents or output in input_root.parents:
            raise ValueError("use a new output outside inputs")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    receipt = {"status": "running", "model_construction_attempts": 0,
               "checkpoint_load_attempts": 0, "forward_passes": 0,
               "optimizer_attempts": 0, "simulation_actions": 0, "states": []}

    def persist():
        receipt["elapsed_seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    persist()
    try:
        if args.vendor:
            sys.path.insert(0, str(args.vendor.resolve(strict=True)))
        import numpy as np
        import torch
        from robomimic.algo import algo_factory
        from robomimic.config import config_factory
        import robomimic.utils.obs_utils as ObsUtils
        a, b = OrderedDict(x=1), OrderedDict(x=1)
        a.annotation, b.annotation = {"x": 1}, {"x": 2}
        for x, y in [(a, b), (torch.tensor([1.0]), torch.tensor([2.0]))]:
            try:
                strict_equal(x, y, torch, np)
            except ValueError:
                continue
            raise ValueError("state mutation control did not reject")
        receipt["state_mutation_rejections"] = 2
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("complete saved CUDA-state restoration requires one visible CUDA device")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.06, 0)
        receipt["runtime"] = {"torch": str(torch.__version__), "numpy": np.__version__,
                              "gpu": torch.cuda.get_device_name(0)}
        spec = importlib.util.spec_from_file_location("p2_complete_state", source)
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        for row in rows:
            if time.monotonic() - started >= 100:
                raise TimeoutError("restoration batch deadline")
            path = states / row["member"]
            if not path.resolve().is_relative_to(states):
                raise ValueError("nonlocal state member")
            body = checked(path, row)
            receipt["checkpoint_load_attempts"] += 1
            persist()
            saved = torch.load(io.BytesIO(body), map_location="cpu", weights_only=False)
            binding = copy.deepcopy(saved["binding"])
            config = config_factory("bc", dic=binding["config"])
            ObsUtils.initialize_obs_utils_with_config(config)
            order = binding["config"]["observation"]["modalities"]["obs"]["low_dim"]
            shapes = binding.get("policy_shapes", {key: hi - lo for key, (lo, hi) in core.KEYS.items()})
            if set(order) != set(shapes):
                raise ValueError("native observation shape identity")
            receipt["model_construction_attempts"] += 1
            persist()
            model = algo_factory("bc", config, {key: (shapes[key],) for key in order}, 7, torch.device("cuda:0"))
            if type(model).__name__ != "BC_RNN_GMM":
                raise ValueError("native model family")
            sampler_state = saved["sampler"]
            if sampler_state.get("kind") == "pilot-Can-Square":
                packs = {key: SimpleNamespace(n=value["n"]) for key, value in sampler_state["items"].items()}
                sampler = core.Samplers(packs, core)
            else:
                sampler = core.Sampler(sampler_state["n"], 1)
            counters = core.restore(model, sampler, saved, binding)
            reconstructed = core.state(model, sampler, counters, binding)
            extra = set(saved) - set(reconstructed)
            if not extra <= {"evolution_sampler", "correction_sampler"}:
                raise ValueError("uncovered additional saved state")
            for name in extra:
                generator = np.random.default_rng()
                generator.bit_generator.state = copy.deepcopy(saved[name])
                reconstructed[name] = copy.deepcopy(generator.bit_generator.state)
            equal_leaves = core.equal(saved, reconstructed)
            strict_leaves = strict_equal(saved, reconstructed, torch, np)
            receipt["states"].append({"role": row["role"], "distributed_sha256": row["sha256"],
                                      "complete_state_equal_leaves": equal_leaves,
                                      "strict_state_and_attribute_equal_leaves": strict_leaves,
                                      "additional_generators_restored": sorted(extra)})
            persist()
            del model, saved, reconstructed, sampler
            gc.collect()
            torch.cuda.empty_cache()
        receipt["status"] = "completed"
    except BaseException as exc:
        receipt.update(status="failed_no_retry", error=repr(exc))
        raise
    finally:
        persist()
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
