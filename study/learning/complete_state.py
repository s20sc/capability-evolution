"""Complete learned-state and fixed-length window methods."""
from __future__ import annotations
import copy
import hashlib
import random
from pathlib import Path
import numpy as np
import torch

KEYS = {"robot0_eef_pos": (0, 3), "robot0_eef_quat": (3, 7),
        "robot0_gripper_qpos": (7, 9), "object": (9, 23)}

RNN_FIELDS = ("_rnn_hidden_state", "_rnn_counter", "_rnn_horizon",
              "_rnn_is_open_loop", "_open_loop_obs")

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

class Windows:
    def __init__(self, path: Path, expected: str, episodes: int, samples: int):
        assert digest(path) == expected, "dataset hash changed"
        with np.load(path, allow_pickle=False) as z:
            self.obs = z["observations"].copy()
            self.actions = z["actions"].copy()
            self.ids = z["demo_ids"].copy()
            self.offsets = z["episode_offsets"].copy()
        assert self.obs.shape == (samples, 23) and self.actions.shape == (samples, 7)
        assert self.obs.dtype == self.actions.dtype == np.float64
        assert len(self.ids) == episodes and len(set(self.ids.tolist())) == episodes
        assert len(self.offsets) == episodes + 1
        assert self.offsets[0] == 0 and self.offsets[-1] == samples
        assert np.issubdtype(self.offsets.dtype, np.integer)
        assert np.all(np.diff(self.offsets) > 0)
        assert np.isfinite(self.obs).all() and np.isfinite(self.actions).all()
        assert np.max(np.abs(self.actions)) <= 1.000001
        self.ends = np.repeat(self.offsets[1:] - 1, np.diff(self.offsets))
        self.n = samples

    def batch(self, starts: np.ndarray, device: torch.device) -> dict:
        assert np.all((starts >= 0) & (starts < self.n))
        idx = np.minimum(starts[:, None] + np.arange(10), self.ends[starts, None])
        x = torch.as_tensor(self.obs[idx], device=device, dtype=torch.float32)
        a = torch.as_tensor(self.actions[idx], device=device, dtype=torch.float32)
        return {"obs": {k: x[..., lo:hi] for k, (lo, hi) in KEYS.items()}, "actions": a}

class Sampler:
    def __init__(self, n: int, seed: int):
        self.n = n
        self.generator = np.random.default_rng(seed)
        self.permutation = self.generator.permutation(n)
        self.cursor = 0
        self.passes = 0

    def next(self) -> np.ndarray:
        if self.cursor + 100 > self.n:
            self.permutation = self.generator.permutation(self.n)
            self.cursor = 0
            self.passes += 1
        out = self.permutation[self.cursor:self.cursor + 100].copy()
        self.cursor += 100
        return out

    def state(self) -> dict:
        return copy.deepcopy({"n": self.n, "generator": self.generator.bit_generator.state,
                              "permutation": self.permutation, "cursor": self.cursor, "passes": self.passes})

    def restore(self, state: dict) -> None:
        assert state["n"] == self.n
        assert sorted(state["permutation"].tolist()) == list(range(self.n))
        assert 0 <= state["cursor"] <= self.n
        self.generator.bit_generator.state = state["generator"]
        self.permutation = state["permutation"].copy()
        self.cursor, self.passes = state["cursor"], state["passes"]

def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}

def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])

def state(model, sampler: Sampler, counters: dict, binding: dict) -> dict:
    return copy.deepcopy({
        "schema": 1, "binding": binding, "weights": model.nets.state_dict(),
        "optimizers": {k: v.state_dict() for k, v in model.optimizers.items()},
        "schedulers": {k: None if v is None else v.state_dict() for k, v in model.lr_schedulers.items()},
        "modes": {k: v.training for k, v in model.nets.named_modules()},
        "rnn": {k: getattr(model, k) for k in RNN_FIELDS if hasattr(model, k)},
        "rng": rng_state(), "sampler": sampler.state(), "counters": counters,
        "normalization": None, "critic_target_replay": "not_applicable"})

def restore(model, sampler: Sampler, saved: dict, binding: dict) -> dict:
    assert saved["schema"] == 1 and saved["binding"] == binding, "checkpoint binding mismatch"
    model.nets.load_state_dict(saved["weights"], strict=True)
    assert set(saved["optimizers"]) == set(model.optimizers)
    assert set(saved["schedulers"]) == set(model.lr_schedulers)
    for k, v in model.optimizers.items():
        v.load_state_dict(saved["optimizers"][k])
    for k, v in model.lr_schedulers.items():
        assert (v is None) == (saved["schedulers"][k] is None)
        if v is not None:
            v.load_state_dict(saved["schedulers"][k])
    modules = dict(model.nets.named_modules())
    assert set(modules) == set(saved["modes"])
    for k, flag in saved["modes"].items():
        modules[k].training = flag
    for k in RNN_FIELDS:
        if k in saved["rnn"]:
            setattr(model, k, copy.deepcopy(saved["rnn"][k]))
        elif hasattr(model, k):
            delattr(model, k)
    sampler.restore(saved["sampler"])
    restore_rng(saved["rng"])
    return copy.deepcopy(saved["counters"])

def save(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)

def equal(a, b, trail: str = "root") -> int:
    assert type(a) is type(b), trail
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()), trail
        return 1
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and np.array_equal(a, b), trail
        return 1
    if isinstance(a, dict):
        assert a.keys() == b.keys(), trail
        return sum(equal(a[k], b[k], f"{trail}.{k}") for k in a)
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b), trail
        return sum(equal(x, y, f"{trail}.{i}") for i, (x, y) in enumerate(zip(a, b)))
    assert a == b, trail
    return 1


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)

class Samplers:
    """Source state keeps the exact native sampler representation."""

    def __init__(self, packs: dict, core, seed=29):
        self.items = {task: core.Sampler(pack.n, seed) for task, pack in packs.items()}

    def next(self, task: str, amount: int):
        item = self.items[task]
        if item.cursor + amount > item.n:
            item.permutation = item.generator.permutation(item.n)
            item.cursor = 0
            item.passes += 1
        out = item.permutation[item.cursor:item.cursor + amount].copy()
        item.cursor += amount
        return out

    def state(self):
        if set(self.items) == {"ToolHang"}:
            return self.items["ToolHang"].state()
        return {"kind": "pilot-Can-Square", "items": {key: value.state() for key, value in self.items.items()}}

    def restore(self, saved):
        if set(self.items) == {"ToolHang"}:
            self.items["ToolHang"].restore(saved)
        else:
            require(saved["kind"] == "pilot-Can-Square" and set(saved["items"]) == set(self.items), "sampler pack set")
            for key, item in self.items.items():
                item.restore(saved["items"][key])
