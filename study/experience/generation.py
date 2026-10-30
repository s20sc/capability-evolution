"""Bounded generation and storage of actual training transitions; no learning."""
from __future__ import annotations
from contextlib import closing
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from .record import EpisodeRecorder, terminal_hold_actions

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
        if key != 'action_returns':
            self.monitor()
        if key not in self.caps or self.counts[key] >= self.caps[key]:
            raise BudgetStop('operation cap: ' + key)
        self.counts[key] += 1
        write_json(self.root / 'meter.json', {'counts': self.counts, 'elapsed_seconds': time.monotonic() - self.started})
        with (self.root / 'operations.jsonl').open('a') as f:
            f.write(json.dumps({'attempt': key, 'counts': self.counts}) + '\n')
            f.flush()
            os.fsync(f.fileno())
        if key == 'action_returns':
            self.monitor()

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
    for name in ['lowdim', 'prepared_source']:
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
    prepared = Path(protocol['prepared_source']['path'])
    generator = DataGenerator(spec, str(prepared), demo_keys=selected)
    cases = []
    accepted_episodes = []
    execution = protocol['native_task']
    for seed in protocol['seeds']:
        case_deadline = time.monotonic() + execution['case_seconds']
        write_json(root / 'active-case.json', {'active': True, 'seed': seed, 'deadline_monotonic': case_deadline})
        random.seed(seed)
        np.random.seed(seed)
        env = create_env()
        with closing(env.env) as native:
            (original_reset, original_step) = (env.reset, env.step)
            case = {'seed': seed, 'steps': 0, 'success': False, 'max_hold': 0, 'stop_reason': 'not_started', 'generator_returned': False}
            cases.append(case)
            hold = 0
            recorder = EpisodeRecorder()
            phase = 'generation'
            current_observation = None
            interface = make_interface(env, MG_Square)
            if native.robots[0].controller.use_delta is not True or native.robots[0].controller.use_ori is not True:
                raise RuntimeError('relative full-pose controller required')
            journal = (root / ('case-' + str(seed) + '.jsonl')).open('x')
            limits = None

            def measure():
                state = np.asarray(native.sim.get_state().flatten())
                obs = native._get_observations()
                force = np.asarray(native.robots[0].ee_force, dtype=float)
                finite = bool(np.isfinite(state).all() and np.isfinite(force).all() and all((np.isfinite(np.asarray(v)).all() for v in obs.values())))
                if not finite:
                    raise ValueError('nonfinite native state, observation or force')
                workspace = bool(all((lo <= v <= hi for (v, (lo, hi)) in zip(obs['robot0_eef_pos'], execution['workspace_xyz']))))
                data = native.sim.data._data
                actuation = inspect_actuation(data.ctrl.tolist(), data.actuator_force.tolist(), limits, execution['actuator_tolerance'])
                decision = guard_decision(finite, workspace, float(np.linalg.norm(force)), actuation)
                return (state, obs, decision)

            def obs_vector(obs):
                return np.concatenate([obs[k] for k in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']]).astype(np.float64).tolist()

            def reset_for_seed(*args, **kw):
                nonlocal limits, current_observation
                np.random.seed(seed)
                result = original_reset(*args, **kw)
                model = native.sim.model._model
                limits = {key: getattr(model, key).tolist() for key in ['actuator_ctrllimited', 'actuator_ctrlrange', 'actuator_forcelimited', 'actuator_forcerange']}
                limits['actuator_names'] = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
                if limits != execution['compiled_limits'] or model.opt.timestep != execution['timestep']:
                    raise ValueError('compiled actuator identity')
                (state, _, decision) = measure()
                case['initial_state_sha256'] = hashlib.sha256(state.astype(np.float64).tobytes()).hexdigest()
                current_observation = obs_vector(result)
                write_json(root / ('initial-' + str(seed) + '.json'), {'state': state.tolist(), 'observation': current_observation, 'model_xml': native.sim.model.get_xml()})
                if native._check_success():
                    raise RuntimeError('initial success; no resampling')
                if decision['stop']:
                    raise CaseStop('initial_guard')
                return result

            def checked_step(action):
                nonlocal hold, current_observation
                meter.monitor()
                enforce_case_deadline(case_deadline)
                if case['steps'] >= execution['horizon']:
                    raise CaseStop('horizon')
                action = np.asarray(action)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError('finite seven-dimensional action')
                (low, high) = native.action_spec
                action = np.clip(action, low, high)
                pre_state = np.asarray(native.sim.get_state().flatten()).tolist()
                attempt = recorder.begin(current_observation, pre_state, action.tolist(), phase)
                meter.charge('action_attempts')
                journal.write(json.dumps({'event': 'attempt', **attempt}, allow_nan=False) + '\n')
                journal.flush()
                result = original_step(action)
                recorder.mark_returned()
                meter.charge('action_returns')
                journal.write(json.dumps({'event': 'returned_unvalidated', 'step': case['steps'] + 1}) + '\n')
                journal.flush()
                case['steps'] += 1
                enforce_case_deadline(case_deadline)
                if not math.isfinite(float(result[1])):
                    raise ValueError('nonfinite native reward')
                (post_state, _, decision) = measure()
                success = bool(native._check_success())
                current_observation = obs_vector(result[0])
                recorder.finish(current_observation, post_state.tolist(), float(result[1]), success, decision['stop'])
                hold = next_success_hold(hold, success, decision['stop'], case_deadline)
                case['max_hold'] = max(case['max_hold'], hold)
                journal.write(json.dumps({'event': 'result', 'step': case['steps'], 'native_success': success, 'hold': hold, 'guard': decision, 'transition': recorder.transitions[-1]}, allow_nan=False) + '\n')
                journal.flush()
                if decision['stop']:
                    raise CaseStop('guard')
                if hold >= execution['success_hold']:
                    enforce_case_deadline(case_deadline)
                    case['success'] = True
                    raise CaseStop('success_hold')
                return result
            (env.reset, env.step) = (reset_for_seed, checked_step)
            try:
                info = generator.generate(env, interface, select_src_per_subtask=False, transform_first_robot_pose=False, interpolate_from_last_target_pose=True, render=False)
                case.update(stop_reason='source_trajectory_complete', generator_returned=True, generator_anytime_success=bool(info['success']))
                if not recorder.transitions:
                    raise RuntimeError('empty generated episode')
                phase = 'terminal_hold'
                gripper = recorder.transitions[-1]['action'][-1]
                for hold_action in terminal_hold_actions(interface, gripper, protocol['terminal_extension']['max_steps']):
                    checked_step(hold_action)
                case['stop_reason'] = 'terminal_extension_complete'
            except CaseStop as exc:
                case['stop_reason'] = str(exc)
            finally:
                journal.close()
                try:
                    case['returned_transitions'] = len(recorder.transitions)
                    case['terminal_hold_steps'] = sum((r['phase'] == 'terminal_hold' for r in recorder.transitions))
                    case['training_eligible'] = recorder.eligible(case['stop_reason'], execution['success_hold'])
                    episode_file = root / ('episode-' + str(seed) + '.json')
                    write_json(episode_file, recorder.snapshot())
                    case['episode_sha256'] = digest(episode_file)
                    if case['training_eligible'] != case['success']:
                        raise RuntimeError('acceptance and actual success disagree')
                    if case['training_eligible']:
                        accepted_episodes.append((seed, recorder.transitions))
                    write_json(root / 'partial-cases.json', cases)
                finally:
                    write_json(root / 'active-case.json', {'active': False})
    (observations, actions, ids, offsets) = ([], [], [], [0])
    for (seed, transitions) in accepted_episodes:
        ids.append('generated_' + str(seed))
        observations.extend((r['obs'] for r in transitions))
        actions.extend((r['action'] for r in transitions))
        offsets.append(len(actions))
    pack = root / 'accepted-trajectories.npz'
    arrays = {'observations': np.asarray(observations, dtype=np.float64).reshape(-1, 23), 'actions': np.asarray(actions, dtype=np.float64).reshape(-1, 7), 'demo_ids': np.asarray(ids, dtype='U40'), 'episode_offsets': np.asarray(offsets, dtype=np.int64)}
    np.savez(pack, **arrays)
    with np.load(pack, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any((not np.array_equal(saved[k], v) for (k, v) in arrays.items())):
            raise RuntimeError('saved learning pack differs from actual transitions')
    return {'status': 'completed', 'cases': cases, 'source_demo_ids': selected, 'accepted_episodes': len(ids), 'accepted_rows': len(actions), 'pack_sha256': digest(pack), 'pack_bytes': pack.stat().st_size, 'pack_observation_order': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'], 'scope': 'development generated data only; not learning or final evaluation', 'counts': meter.counts, 'prepared_source_sha256': digest(prepared)}
