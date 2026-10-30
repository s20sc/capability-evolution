"""Run fixed experience preparation, collection or original-record repacking."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prepare-source", "generate-target", "generate-correction", "repack-target", "repack-correction"])
    ap.add_argument("workspace", type=Path)
    ap.add_argument("--vendor", type=Path, required=True)
    ap.add_argument("--mimicgen-source", type=Path)
    ap.add_argument("--input-map", type=Path)
    ap.add_argument("--records", type=Path)
    ap.add_argument("--parent-log-map", type=Path)
    ap.add_argument("--manifest-sha256", required=True)
    ap.add_argument("--seconds", type=int, required=True)
    args = ap.parse_args()
    if not __debug__ or not sys.dont_write_bytecode or args.seconds <= 0:
        raise ValueError("unoptimized bytecode-disabled execution and positive wall allowance")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("these operations use CPU and no visible CUDA device")
    package = Path(__file__).resolve().parent
    manifest = package / "PACKAGE_IDENTITIES.json"
    if digest(manifest) != args.manifest_sha256:
        raise ValueError("fixed source manifest")
    for name, identity in json.loads(manifest.read_text())["files"].items():
        f = package / name
        if (f.is_symlink() or not f.resolve().is_relative_to(package)
                or f.stat().st_size != identity["bytes"] or digest(f) != identity["sha256"]):
            raise ValueError("source member changed: " + name)
    root = args.workspace.resolve()
    if root == package or package in root.parents or root in package.parents or root.exists():
        raise ValueError("fresh workspace outside source package")
    for name in ["MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"]:
        os.environ[name] = str(root / "cache" / name.lower())
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    spec = importlib.util.spec_from_file_location("experience", package / "experience/__init__.py",
                                                submodule_search_locations=[str(package / "experience")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["experience"] = module
    spec.loader.exec_module(module)
    from experience import context, generation, source_preparation, correction_packs, target_pack
    if args.phase.startswith("repack"):
        if args.records is None:
            raise ValueError("original record directory required")
        if args.phase == "repack-target":
            result = target_pack.run(root, args.records, args.vendor)
        else:
            if args.parent_log_map is None:
                raise ValueError("fixed parent-log map required")
            p = context.configure_correction_packaging(root, args.records,
                    json.loads(args.parent_log_map.read_text()), args.vendor)
            correction_packs.run(root, p)
            result = {"status": "completed", "packs": json.loads((root / "packs.json").read_text()),
                      "scope": "original generated/parent records repacked; no new collection"}
    else:
        if args.input_map is None or args.mimicgen_source is None:
            raise ValueError("data-role map and original MimicGen source companion required")
        profile = {"prepare-source": "source-preparation", "generate-target": "target", "generate-correction": "correction"}[args.phase]
        p = context.configure(profile, root, json.loads(args.input_map.read_text()), args.vendor, args.mimicgen_source)
        remaining = args.seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("setup exhausted declared allowance")
        implementation = source_preparation if args.phase == "prepare-source" else generation
        meter = implementation.Meter(root, p["caps"], min(p["wall_seconds"], remaining))
        result = implementation.run(root, p, meter)
    if time.monotonic() - started > args.seconds:
        raise TimeoutError("declared entry allowance exceeded; operation not accepted")
    result["entry_elapsed_seconds"] = time.monotonic() - started
    from experience.utility import write
    write(root / "result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
