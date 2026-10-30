"""Train-only content audit and episode-preserving packaging; no physics or NN."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

KEYS = {"robot0_eef_pos": 3, "robot0_eef_quat": 4,
        "robot0_gripper_qpos": 2, "object": 14}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition(split: dict) -> tuple[list[str], list[str], list[str]]:
    train, dev, held = [split[k] for k in
        ("learning_train_ids", "development_ids", "protected_official_valid_ids")]
    sets = [set(x) for x in (train, dev, held)]
    if (list(map(len, (train, dev, held))) != [160, 20, 20]
            or any(len(x) != len(y) for x, y in zip((train, dev, held), sets))
            or any(sets[i] & sets[j] for i in range(3) for j in range(i))
            or set.union(*sets) != {f"demo_{i}" for i in range(200)}
            or "demo_1" not in sets[0]):
        raise ValueError("split whitelist")
    return train, dev, held


def audit_and_package(low_path: Path, raw_path: Path, split_path: Path, anchor_path: Path,
                      out: Path, protocol: dict, check, counters: dict) -> dict:
    import h5py
    import numpy as np

    train, dev, held = partition(json.loads(split_path.read_text()))
    with gzip.open(anchor_path, "rt") as stream:
        anchor = json.loads(stream.readline())
    if anchor["event"] != "initial":
        raise ValueError("anchor is not initial")
    report = {"demos": [], "packages": {}, "protected_valid_array_reads": 0,
              "scope": "within-task train/development only"}
    # Expose partial audit and accurate started/completed reads on failure as well.
    counters["content_audit"] = report
    packs = {role: {"obs": [], "actions": [], "ids": [], "offsets": [0]}
             for role in ("train", "development")}
    total_samples = 0

    def read(dataset):
        if counters["array_read_attempts"] >= protocol["max_array_reads"]:
            raise ValueError("array read cap")
        counters["array_read_attempts"] += 1
        check()
        value = dataset[:]
        counters["array_read_completions"] += 1
        return value

    with h5py.File(low_path, "r") as low, h5py.File(raw_path, "r") as raw:
        for data in (raw, low):
            if set(data["data"].keys()) != set(train + dev + held):
                raise ValueError("dataset demo IDs")
            # Mask values are metadata, not protected trajectory arrays.
            for name, expected in (("train", train + dev), ("valid", held)):
                mask = data["mask"][name]
                if (mask.shape != (len(expected),) or mask.dtype.kind != "S"
                        or mask.dtype.itemsize > 64):
                    raise ValueError("mask schema")
                ids = [x.decode() for x in mask[:]]
                if len(set(ids)) != len(ids) or set(ids) != set(expected):
                    raise ValueError("official masks differ")
        metadata = [json.loads(data["data"].attrs["env_args"]) for data in (raw, low)]
        for meta in metadata:
            if meta["env_name"] != "NutAssemblySquare" or meta["env_version"] != "1.4.1":
                raise ValueError("task identity")
        dynamic_keys = protocol["dynamic_env_keys"]
        if any(metadata[0]["env_kwargs"].get(k) != metadata[1]["env_kwargs"].get(k)
               for k in dynamic_keys):
            raise ValueError("dynamic environment metadata differs")
        report["metadata_raw"] = metadata[0]
        report["metadata_low_dim"] = metadata[1]
        report["metadata_comparison_keys"] = dynamic_keys
        report["uncompared_metadata_keys"] = sorted(
            (set(metadata[0]["env_kwargs"]) | set(metadata[1]["env_kwargs"])) - set(dynamic_keys))
        # Only these two explicit lists may reach a trajectory value read.
        for role, ids in (("train", train), ("development", dev)):
            for name in ids:
                check()
                r, q = raw["data"][name], low["data"][name]
                n = r["actions"].shape[0]
                if not 0 < n <= protocol["max_episode_samples"]:
                    raise ValueError("episode sample cap")
                total_samples += n
                if total_samples > protocol["max_total_samples"]:
                    raise ValueError("total sample cap")
                if (r["actions"].shape != q["actions"].shape or r["actions"].shape != (n, 7)
                        or r["states"].shape != q["states"].shape or r["states"].shape != (n, 45)):
                    raise ValueError("state/action shape correspondence")
                if any(g[k].dtype.kind != "f" or g[k].dtype.itemsize not in (4, 8)
                       for g in (r, q) for k in ("actions", "states")):
                    raise ValueError("unsupported source precision")
                raw_actions, actions = read(r["actions"]), read(q["actions"])
                raw_states, states = read(r["states"]), read(q["states"])
                if (not all(np.isfinite(x).all() for x in (raw_actions, actions, raw_states, states))
                        or not np.array_equal(raw_actions, actions)
                        or not np.array_equal(raw_states, states)
                        or np.abs(actions).max() > 1.000001):
                    raise ValueError("raw/low_dim values differ or invalid")
                parts = []
                dtypes = {}
                for key, width in KEYS.items():
                    array = q["obs"][key]
                    if (array.shape != (n, width) or array.dtype.kind != "f"
                            or array.dtype.itemsize not in (4, 8)):
                        raise ValueError("observation shape/dtype")
                    value = read(array)
                    if not np.isfinite(value).all():
                        raise ValueError("nonfinite observation")
                    parts.append(value)
                    dtypes[key] = str(value.dtype)
                vector = np.concatenate(parts, axis=1).astype(np.float64)
                if name == "demo_1":
                    if not np.allclose(states[0], anchor["state"], rtol=0, atol=1e-12):
                        raise ValueError("native anchor state differs")
                    expected = np.concatenate([anchor["native_observation"][k] for k in KEYS])
                    error = float(np.max(np.abs(vector[0] - expected)))
                    report["initial_native_anchor_max_abs"] = error
                    if error > protocol["anchor_tolerance"]:
                        raise ValueError("native anchor observation differs")
                pack = packs[role]
                pack["obs"].append(vector)
                pack["actions"].append(actions.astype(np.float64))
                pack["ids"].append(name)
                pack["offsets"].append(pack["offsets"][-1] + n)
                report["demos"].append({"id": name, "role": role, "samples": n,
                    "raw_actions_dtype": str(raw_actions.dtype), "low_actions_dtype": str(actions.dtype),
                    "observation_dtypes": dtypes, "state_action_equal": True})
        if "initial_native_anchor_max_abs" not in report:
            raise ValueError("anchor not checked")
        for role, pack in packs.items():
            check()
            partial = out / (role + ".npz.incomplete")
            with partial.open("xb") as stream:
                np.savez_compressed(stream, observations=np.concatenate(pack["obs"]),
                    actions=np.concatenate(pack["actions"]), demo_ids=np.asarray(pack["ids"]),
                    episode_offsets=np.asarray(pack["offsets"], dtype=np.int64))
            check()
            dest = out / (role + ".npz")
            partial.rename(dest)
            report["packages"][role] = {"path": dest.name, "sha256": sha(dest),
                "bytes": dest.stat().st_size, "episodes": len(pack["ids"]),
                "samples": pack["offsets"][-1], "stored_dtype": "float64",
                "note": "lossless widening of any float32 observations adds no precision"}
    check()
    return report
