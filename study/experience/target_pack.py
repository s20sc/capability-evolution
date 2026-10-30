"""Reconstruct the fixed target-learning array from its actual stored transitions."""
import json
from pathlib import Path
import sys

from .record import EpisodeRecorder
from .utility import sha, write


def run(workspace, generation, native_vendor):
    root = Path(workspace).resolve()
    source = Path(generation).resolve(strict=True)
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("fresh repacking workspace outside record input")
    profile = json.loads((Path(__file__).parent / "recipes.json").read_text())["profiles"]["target"]
    result = json.loads((source / "result.json").read_text())
    if (result["status"] != "completed" or [c["seed"] for c in result["cases"]] != profile["seeds"]
            or result["source_demo_ids"] != profile["learning_ids"][:10]):
        raise ValueError("fixed complete target acquisition")
    accepted = []
    for case in result["cases"]:
        path = source / ("episode-" + str(case["seed"]) + ".json")
        if sha(path) != case["episode_sha256"]:
            raise ValueError("original recorded episode identity")
        original = json.loads(path.read_text())
        if original["unfinished_transition"] is not None:
            raise ValueError("unfinished recorded action")
        replay = EpisodeRecorder()
        previous = None
        for row in original["transitions"]:
            if previous is not None and (previous["next_obs"] != row["obs"] or previous["next_state"] != row["state"]):
                raise ValueError("recorded transition continuity")
            replay.begin(row["obs"], row["state"], row["action"], row["phase"])
            if row["return_observed"] is not True:
                raise ValueError("unreturned recorded action")
            replay.mark_returned()
            replay.finish(row["next_obs"], row["next_state"], row["reward"], row["success"], row["guard_stop"])
            previous = row
        if replay.snapshot() != original:
            raise ValueError("record schema or values differ from actual recorder")
        eligible = replay.eligible(case["stop_reason"], profile["native_task"]["success_hold"])
        if eligible is not case["training_eligible"] or len(replay.transitions) != case["returned_transitions"]:
            raise ValueError("recorded eligibility/count differs")
        if eligible:
            accepted.append((case["seed"], replay.transitions))
    sys.path.insert(0, str(Path(native_vendor).resolve(strict=True)))
    import numpy as np
    if np.__version__ != "1.26.0":
        raise ValueError("fixed array stack")
    observations, actions, ids, offsets = [], [], [], [0]
    for seed, transitions in accepted:
        ids.append("generated_" + str(seed))
        observations.extend(r["obs"] for r in transitions)
        actions.extend(r["action"] for r in transitions)
        offsets.append(len(actions))
    arrays = {"observations": np.asarray(observations, dtype=np.float64).reshape(-1, 23),
              "actions": np.asarray(actions, dtype=np.float64).reshape(-1, 7),
              "demo_ids": np.asarray(ids, dtype="U40"),
              "episode_offsets": np.asarray(offsets, dtype=np.int64)}
    if (len(ids) != 3 or len(actions) != 480 or not np.isfinite(arrays["observations"]).all()
            or not np.isfinite(arrays["actions"]).all() or np.max(np.abs(arrays["actions"])) > 1):
        raise ValueError("fixed target package membership/values")
    root.mkdir(parents=True, exist_ok=False)
    path = root / "accepted-trajectories.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise ValueError("saved array roundtrip")
    identity = sha(path)
    if identity != "2d97f6d6d15ab77de875485280330754196dbb1776be380d98970933ea6ae361":
        raise ValueError("original target input bytes differ")
    report = {"status": "completed", "pack_sha256": identity, "pack_bytes": path.stat().st_size,
              "episodes": len(ids), "rows": len(actions), "npz_saves": 1, "npz_loads": 1,
              "scope": "repacking original recorded transitions; no generation, learned policy or task evaluation"}
    write(root / "repack-result.json", report)
    return report
