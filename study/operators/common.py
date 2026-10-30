"""Portable data admission, metering and runtime for the native operators."""
from __future__ import annotations
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

from contracts import LocalBudgetStop, canonical, require

ACTIVE = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def output_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def write(path, value):
    if ACTIVE is not None:
        ACTIVE.monitor()
        if isinstance(value, dict) and {"path", "sha256", "binding", "counters", "profile"} <= set(value):
            value = copy.deepcopy(value)
            value["portable_origin"] = {"job": ACTIVE.definition["id"],
                "recipe_identity": ACTIVE.p["recipe_identity"],
                "continuation_seed": ACTIVE.definition.get("continuation_seed")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(canonical(value) + b"\n")
    temp.replace(path)


def gate():
    require(ACTIVE is not None, "use run_operator.py for a declared recipe")
    ACTIVE.monitor()
    return ACTIVE.root, ACTIVE.p


class Job:
    def __init__(self, root, recipe, definition, inputs, vendor, seconds, output_limit):
        self.root, self.p, self.definition = root, recipe, definition
        self.inputs, self.vendor = inputs, vendor
        self.out = root / "jobs" / definition["id"]
        self.out.mkdir(parents=True, exist_ok=False)
        self.started, self.seconds, self.output_limit = time.monotonic(), seconds, output_limit
        self.digest = sha(root / "protocol.json")
        self.counts = {}
        self.value = {"status": "running", "job": definition["id"], "protocol_sha256": self.digest, "counts": self.counts}
        self.observed = {"model_construction_attempts": 0, "models_constructed": 0,
                         "checkpoint_load_attempts": 0, "optimizer_step_attempts": 0,
                         "optimizer_mutation_entries": 0, "optimizer_returns": 0,
                         "policy_forward_attempts": 0, "gradient_attempts": 0,
                         "env_constructor_attempts": 0, "env_constructors": 0,
                         "explicit_reset_attempts": 0, "explicit_resets": 0,
                         "action_attempts": 0, "action_returns": 0}
        self.event_index = 0

    def event(self, key):
        self.observed[key] += 1
        self.event_index += 1
        with (self.root / "operation-meter.jsonl").open("a") as stream:
            stream.write(json.dumps({"index": self.event_index, "event": key,
                                     "count": self.observed[key]}) + "\n")

    def monitor(self):
        if time.monotonic() - self.started >= self.seconds:
            raise LocalBudgetStop("operator wall budget")
        if output_bytes(self.root) > self.output_limit:
            raise LocalBudgetStop("output polling threshold")

    def persist(self):
        self.monitor()
        self.value["elapsed_seconds"] = time.monotonic() - self.started
        write(self.out / "result.json", self.value)

    def charge(self, **charges):
        self.monitor()
        for name, amount in charges.items():
            require(type(amount) is int and amount >= 0, "nonnegative operation charge")
            candidate = self.counts.get(name, 0) + amount
            cap = self.definition.get("caps", {}).get(name)
            if cap is not None and candidate > cap:
                raise LocalBudgetStop("declared operation cap: " + name)
            self.counts[name] = candidate

    def registered_input(self, name):
        info = self.inputs[name]
        expected = self.p["input_identities"][name]["sha256"]
        require(info["sha256"] == expected, "declared study data identity " + name)
        path = Path(info["path"])
        require(not path.is_symlink() and sha(path) == expected, "actual study data identity " + name)
        return path

    def finish(self, **fields):
        self.value.update(status="completed", **fields)
        self.persist()


def runtime(job, gpu):
    sys.path.insert(0, str(job.vendor))
    import numpy as np
    import torch
    import state_core as core
    expected = job.p["tensor_runtime"]
    require(str(torch.__version__) == expected["torch"] and np.__version__ == expected["numpy"]
            and torch.version.cuda == expected["cuda"], "supported tensor runtime")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if gpu:
        sample = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
        memory, utilization = [int(x.strip()) for x in sample.split(",")]
        job.value["GPU_admission"] = {"raw": sample, "memory_mib": memory, "utilization_percent": utilization}
        job.persist()
        require(0 <= memory < 2048 and 0 <= utilization <= 5, "GPU busy; no retry")
        require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == expected["gpu"], "supported GPU")
        torch.cuda.set_per_process_memory_fraction(.06, 0)
    return core, np, torch


def load_checkpoint(job, item, torch, device, category):
    path = Path(item["path"])
    require(not path.is_symlink(), "regular declared state input")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == item["sha256"], "declared state hash")
    job.charge(**{category: 1})
    return torch.load(io.BytesIO(raw), map_location=device, weights_only=False)


def install_meters(job, limits, torch):
    """Wrap actual native calls; limits stop before an extra mutation/action."""
    import robomimic.algo as algorithms
    original_factory = algorithms.algo_factory
    original_backward = torch.autograd.backward
    original_load = torch.load

    def tick(key, limit=None):
        job.monitor()
        job.event(key)
        if limit is not None and job.observed[key] > limit:
            raise LocalBudgetStop("operator acceptance budget: " + key)

    def factory(*args, **kwargs):
        tick("model_construction_attempts", limits["models"])
        model = original_factory(*args, **kwargs)
        job.event("models_constructed")
        optimizer = model.optimizers["policy"]
        original_step = optimizer.step
        original_forward = model.nets["policy"].forward_train

        def step(*a, **k):
            tick("optimizer_step_attempts", limits["optimizer_mutations"])
            job.event("optimizer_mutation_entries")
            result = original_step(*a, **k)
            job.event("optimizer_returns")
            return result

        def forward(*a, **k):
            tick("policy_forward_attempts", limits["policy_forwards"])
            return original_forward(*a, **k)

        optimizer.step = step
        model.nets["policy"].forward_train = forward
        return model

    def backward(*args, **kwargs):
        tick("gradient_attempts", limits["gradients"])
        return original_backward(*args, **kwargs)

    algorithms.algo_factory = factory
    torch.autograd.backward = backward

    def load(*args, **kwargs):
        tick("checkpoint_load_attempts", limits["checkpoint_loads"])
        return original_load(*args, **kwargs)

    torch.load = load
    # Zero is an enforced prohibition, including unexpected training-time scenes.
    import robosuite as suite
    original_make = suite.make

    def make(*args, **kwargs):
        tick("env_constructor_attempts", limits["environment_constructors"])
        env = original_make(*args, **kwargs)
        job.event("env_constructors")
        native_step, native_reset = env.step, env.reset

        def step(action):
            tick("action_attempts", limits["actions"])
            result = native_step(action)
            job.event("action_returns")
            return result

        def reset(*a, **k):
            tick("explicit_reset_attempts", limits["environment_constructors"])
            result = native_reset(*a, **k)
            job.event("explicit_resets")
            return result

        env.step, env.reset = step, reset
        return env

    suite.make = make
