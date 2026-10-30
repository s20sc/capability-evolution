"""One-shot source-trajectory controller qualification; no learned model."""
from __future__ import annotations
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

class BudgetStop(RuntimeError):
    pass

class CaseStop(RuntimeError):
    pass

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temp.replace(path)

def enforce_case_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise BudgetStop('original 20s case deadline')

def next_success_hold(previous: int, success: bool, guard_stop: bool, deadline: float) -> int:
    enforce_case_deadline(deadline)
    return previous + 1 if success and (not guard_stop) else 0

def make_interface(wrapper: object, factory):
    return factory(wrapper.env)

class Meter:

    def __init__(self, root: Path, caps: dict, wall: float):
        (self.root, self.caps) = (root, caps)
        self.counts = {key: 0 for key in caps}
        self.started = time.monotonic()
        self.deadline = self.started + wall

    def monitor(self) -> None:
        if time.monotonic() >= self.deadline:
            raise BudgetStop('whole worker deadline')
        size = sum((p.stat().st_size for p in self.root.rglob('*') if p.is_file()))
        if size > 128 * 1024 * 1024:
            raise BudgetStop('128MiB output polling threshold')

    def charge(self, key: str) -> None:
        self.monitor()
        if key not in self.caps or self.counts[key] >= self.caps[key]:
            raise BudgetStop('operation cap: ' + key)
        self.counts[key] += 1
        write_json(self.root / 'meter.json', {'counts': self.counts, 'elapsed_seconds': time.monotonic() - self.started})
        with (self.root / 'operations.jsonl').open('a') as f:
            f.write(json.dumps({'attempt': key, 'counts': self.counts}) + '\n')
            f.flush()
            os.fsync(f.fileno())

def source_groups(learning_ids: list[str], internal_dev: list[str]) -> list[str]:
    selected = learning_ids[:10]
    if len(selected) != 10 or len(set(selected)) != 10 or set(selected) & set(internal_dev):
        raise ValueError('fixed first-ten source membership')
    return selected

def check_files(base: Path, rows: list[dict]) -> None:
    for row in rows:
        path = Path(row['path'])
        if not path.is_absolute():
            if '..' in path.parts:
                raise ValueError('source traversal')
            path = base / path
        if digest(path) != row['sha256']:
            raise ValueError('source identity: ' + str(path))

def run(root: Path, protocol: dict, meter: Meter) -> dict:
    if sys.flags.optimize:
        raise RuntimeError('assertions must remain enabled')
    if root.resolve() != Path(protocol['root']).resolve():
        raise ValueError('declared output directory')
    sys.path.insert(0, str(root))
    check_files(root, protocol['files'])
    check_files(root, protocol['external_sources'])
    for name in ['lowdim', 'raw']:
        row = protocol[name]
        if Path(row['path']).stat().st_size != row['bytes'] or digest(Path(row['path'])) != row['sha256']:
            raise ValueError('dataset identity ' + name)
    sys.path[:0] = [str(root / 'vendor'), protocol['robomimic_vendor'], protocol['robosuite_site']]
    import h5py
    import numpy as np
    import mujoco
    import robosuite
    import robomimic
    import mimicgen
    from compat_env_robosuite import EnvRobosuite
    import robomimic.utils.obs_utils as ObsUtils
    from mimicgen.env_interfaces.robosuite import MG_Square
    from mimicgen.datagen.data_generator import DataGenerator
    from mimicgen.configs.task_spec import MG_TaskSpec
    from mimicgen.scripts.prepare_src_dataset import extract_datagen_info_from_trajectory
    from native_guard import inspect_actuation, guard_decision
    if robosuite.__version__ != '1.4.1' or mujoco.__version__ != '2.3.2':
        raise RuntimeError('original simulator versions required')
    if Path(importlib.import_module(EnvRobosuite.__module__).__file__).resolve() != root / 'compat_env_robosuite.py':
        raise RuntimeError('isolated wrapper origin')
    if Path(mimicgen.__file__).resolve() != root / 'vendor/mimicgen/__init__.py':
        raise RuntimeError('MimicGen origin')
    if Path(robomimic.__file__).resolve() != Path(protocol['robomimic_vendor']) / 'robomimic/__init__.py':
        raise RuntimeError('robomimic origin')
    with h5py.File(protocol['lowdim']['path'], 'r') as f:
        meta = json.loads(f['data'].attrs['env_args'])
    if meta['env_name'] != 'NutAssemblySquare' or meta['env_version'] != '1.4.1':
        raise RuntimeError('native task metadata')
    kwargs = dict(meta['env_kwargs'])
    kwargs.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    ObsUtils.initialize_obs_utils_with_obs_specs({'obs': {'low_dim': [], 'rgb': []}})

    def create_env():
        meter.charge('environment_constructors')
        env = EnvRobosuite(env_name=meta['env_name'], render=False, render_offscreen=False, use_image_obs=False, **kwargs)
        actual = Path(importlib.import_module(type(env.env).__module__).__file__)
        if digest(actual) != protocol['native_task']['environment_source_sha256']:
            env.env.close()
            raise RuntimeError('actual native task source')
        (original_reset, original_reset_to) = (env.reset, env.reset_to)

        def counted_reset(*args, **kw):
            meter.charge('explicit_resets')
            return original_reset(*args, **kw)

        def counted_reset_to(state):
            meter.charge('reset_to_attempts')
            if 'model' in state:
                meter.charge('xml_reload_attempts')
            return original_reset_to(state)
        (env.reset, env.reset_to) = (counted_reset, counted_reset_to)
        return env
    template = json.loads((root / 'vendor/mimicgen/exps/templates/robosuite/square.json').read_text())
    spec = MG_TaskSpec.from_json(json_dict=template['task']['task_spec'])
    selected = source_groups(protocol['learning_ids'], protocol['internal_dev_ids'])
    prepared = root / 'source10.hdf5'
    env = create_env()
    try:
        interface = make_interface(env, MG_Square)
        with h5py.File(protocol['raw']['path'], 'r') as src, h5py.File(prepared, 'x') as dst:
            dg = dst.create_group('data')
            dg.attrs['env_args'] = json.dumps(meta)
            lengths = [src['data'][name]['actions'].shape[0] for name in selected]
            if sum(lengths) + 10 > protocol['caps']['reset_to_attempts']:
                raise BudgetStop('source lengths exceed fixed preparation allowance')
            for name in selected:
                meter.monitor()
                group = src['data'][name]
                (states, actions) = (group['states'][()], group['actions'][()])
                if not np.isfinite(states).all() or not np.isfinite(actions).all():
                    raise ValueError('nonfinite source trajectory')
                initial = {'states': states[0], 'model': group.attrs['model_file']}
                info = extract_datagen_info_from_trajectory(env, interface, initial, states, actions)
                dest = dg.create_group(name)
                dest.create_dataset('actions', data=actions)
                dest.attrs['num_samples'] = len(actions)
                out = dest.create_group('datagen_info')
                for (key, value) in info.items():
                    if isinstance(value, dict):
                        sub = out.create_group(key)
                        for (child, array) in value.items():
                            sub.create_dataset(child, data=array)
                    else:
                        out.create_dataset(key, data=value)
    finally:
        env.env.close()
    return {'status': 'completed', 'path': str(prepared), 'sha256': digest(prepared), 'source_demo_ids': selected, 'counts': meter.counts}
