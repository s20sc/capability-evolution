"""Resolve fixed scientific profiles into a fresh ordinary workspace."""
import copy
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, identity):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or digest(path) != identity["sha256"]:
        raise ValueError("input/source identity: " + str(path))
    if "bytes" in identity and path.stat().st_size != identity["bytes"]:
        raise ValueError("input/source size: " + str(path))
    return path.resolve()


def relative(root, name):
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError("source member path")
    return path


def configure(profile, workspace, inputs, native_vendor, mimicgen_source):
    """Prepare paths/copies only; numerical work is performed by run()."""
    source = Path(__file__).resolve().parent
    recipe = json.loads((source / "recipes.json").read_text())
    p = copy.deepcopy(recipe["profiles"][profile])
    native = Path(native_vendor).resolve(strict=True)
    generator = Path(mimicgen_source).resolve(strict=True)
    root = Path(workspace).resolve()
    if root == source or source in root.parents or root in source.parents:
        raise ValueError("workspace must be separate from scientific source")
    for name, declaration in p.pop("input_roles").items():
        item = inputs[declaration["role"]]
        path = checked(item["path"], declaration)
        if item["sha256"] != declaration["sha256"]:
            raise ValueError("declared scientific data identity")
        p[name] = {"path": str(path), "sha256": declaration["sha256"], "bytes": path.stat().st_size}
    external = []
    for row in p["external_sources"]:
        path = checked(relative(native, row["relative"]), row)
        external.append({"path": str(path), "sha256": row["sha256"]})
    p["external_sources"] = external
    supplied = json.loads((generator / "SOURCE_IDENTITIES.json").read_text())
    if supplied["commit"] != p["source_commit"]:
        raise ValueError("fixed MimicGen commit")
    if supplied["files"] != recipe["mimicgen_files"]:
        raise ValueError("actual fixed MimicGen source fields")
    support = json.loads((source / "SUPPORT_IDENTITIES.json").read_text())["files"]
    copies = []
    for directory, entries in [(generator, supplied["files"]), (source / "vendor-support", support)]:
        for name, identity in entries.items():
            path = checked(relative(directory, name), identity)
            copies.append((name, path))
    names = [name for name, path in copies]
    if len(names) != len(set(names)):
        raise ValueError("overlapping vendor source members")
    root.mkdir(parents=True, exist_ok=False)
    for name, path in copies:
        target = relative(root / "vendor", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name in ["compat_env_robosuite.py", "native_guard.py"]:
        shutil.copyfile(source / name, root / name)
    p.update(root=str(root), robomimic_vendor=str(native), robosuite_site=str(native))
    p["files"] = [{"path": str(f.relative_to(root)), "sha256": digest(f)}
                  for f in sorted(root.rglob("*")) if f.is_file()]
    (root / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    return p


def configure_correction_packaging(workspace, generation, parent_inputs, native_vendor):
    """Bind the fixed original parent logs and generated records for packing."""
    source = Path(__file__).resolve().parent
    p = copy.deepcopy(json.loads((source / "recipes.json").read_text())["correction_packaging"])
    generated = Path(generation).resolve(strict=True)
    result = json.loads((generated / "result.json").read_text())
    if result["status"] != "completed" or [c["seed"] for c in result["cases"]] != p["acquisition_seeds"]:
        raise ValueError("complete ordered acquisition required")
    episodes = []
    for case in result["cases"]:
        episodes.append(checked(generated / ("episode-" + str(case["seed"]) + ".json"), {"sha256": case["episode_sha256"]}))
    for seed, identity in p["parent_logs"].items():
        identity["path"] = str(checked(parent_inputs[seed]["path"], identity))
    root = Path(workspace).resolve()
    for inputs in [source, generated]:
        if root == inputs or inputs in root.parents or root in inputs.parents:
            raise ValueError("fresh packaging workspace outside inputs")
    root.mkdir(parents=True, exist_ok=False)
    (root / "acquisition").mkdir()
    shutil.copyfile(generated / "result.json", root / "acquisition/result.json")
    for path in episodes:
        shutil.copyfile(path, root / "acquisition" / path.name)
    p["acquisition_result_sha256"] = digest(root / "acquisition/result.json")
    p["vendor"] = str(Path(native_vendor).resolve(strict=True))
    (root / "protocol.json").write_text(json.dumps(p, indent=2) + "\n")
    return p
