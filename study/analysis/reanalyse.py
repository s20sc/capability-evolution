"""Recalculate the saved multicycle study with its original summary methods."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def checked_file(root, name, identities):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("nonlocal package member")
    raw = path.read_bytes()
    info = identities[name]
    if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
        raise ValueError("package member changed: " + name)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = (root / "PACKAGE_IDENTITIES.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.manifest_sha256:
        raise ValueError("package manifest identity")
    identities = json.loads(raw)["files"]
    source = root / "analysis/study_statistics.py"
    checked_file(root, "analysis/study_statistics.py", identities)
    payload = json.loads(checked_file(root, "analysis/inputs.json", identities))
    if payload["schema"] != "p2-public-multicycle-inputs-v1" or payload["report"]["status"] != "completed":
        raise ValueError("completed saved study required")
    jobs, boards = payload["jobs"], payload["boards"]
    if set(payload["measurements"]) != set(jobs) or set(boards) != {name for name, row in jobs.items() if row["kind"] == "evaluate"}:
        raise ValueError("complete job measurements and evaluation set required")
    for name, board in boards.items():
        job = jobs[name]
        if (board["status"] != "completed" or board["task"] != job["task"]
                or [case["seed"] for case in board["cases"]] != job["seeds"]):
            raise ValueError("task/panel identity: " + name)
    spec = importlib.util.spec_from_file_location("p2_study_statistics", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.summarize(payload["design"], payload["report"], boards,
                               payload["training"], {}, payload["measurements"])
    if summary != payload["expected_summary"]:
        raise ValueError("full summary differs from saved result")
    output = args.output.resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("use a new output outside the package")
    output.mkdir(exist_ok=False)
    text = (json.dumps(summary, allow_nan=False, indent=2) + "\n").encode()
    (output / "summary.json").write_bytes(text)
    receipt = {"status": "completed", "all_saved_summary_fields_equal": True,
               "jobs": len(jobs), "boards": len(boards),
               "summary_sha256": hashlib.sha256(text).hexdigest(),
               "scope": "original saved-case summary implementation; no training, simulation or independent statistical implementation"}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
