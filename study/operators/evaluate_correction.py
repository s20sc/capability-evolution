"""Native low-noise stochastic Square evaluation."""
from __future__ import annotations
import argparse,gzip,hashlib,importlib,json,math,os,random,sys,time
from pathlib import Path
from common import gate,sha
from native_guard import inspect_actuation,guard_decision
from types import SimpleNamespace

def safe(x):
    if isinstance(x, float) and not math.isfinite(x):
        return {"nonfinite": repr(x)}
    if isinstance(x, dict):
        return {k: safe(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [safe(v) for v in x]
    return x

def main() -> None:
    (root, p) = gate()
    args = SimpleNamespace(protocol=root / 'protocol.json', sha256=sha(root / 'protocol.json'))
    out = root / 'run'
    out.mkdir(exist_ok=False)
    (out / 'protocol.json').write_bytes((root / 'protocol.json').read_bytes())
    costs = {'constructors': 0, 'explicit_resets': 0, 'model_construction_attempts': 0, 'model_constructions': 0, 'inference_attempts': 0, 'inference_returned': 0, 'action_attempts': 0, 'returned_steps': 0, 'gradient_updates': 0, 'external_model_calls': 0, 'state_writes': 0, 'xml_restores': 0, 'clipped_components': 0}
    result = {'status': 'running', 'protocol_sha256': args.sha256, 'costs': costs, 'cases': [], 'planned_cases': len(p['seeds']) * p['repeats']}
    started = time.monotonic()

    def save() -> None:
        result['wall_seconds'] = time.monotonic() - started
        temp = out / 'status.tmp'
        temp.write_text(json.dumps(safe(result), indent=2, allow_nan=False) + '\n')
        temp.replace(out / 'status.json')

    def budget() -> None:
        if time.monotonic() - started > p['soft_seconds']:
            raise TimeoutError('wall cap')
        if sum((x.stat().st_size for x in root.rglob('*') if x.is_file())) > p['output_bytes']:
            raise RuntimeError('output cap')
    save()
    try:
        sys.path.insert(0, p['vendor'])
        sys.path.append(p['robosuite_site_packages'])
        import numpy as np
        import torch
        import h5py
        import robosuite as suite
        import mujoco
        assert mujoco.__version__ == p['mujoco_version']
        from robomimic.algo import algo_factory
        from robomimic.config import config_factory
        import robomimic.utils.obs_utils as ObsUtils
        assert suite.__version__ == '1.4.1'
        assert Path(suite.__file__).resolve() == Path(p['robosuite_site_packages']) / 'robosuite/__init__.py'
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(1)
        cfg_dict = json.loads(Path(p['training_protocol']).read_text())['config']
        checkpoint = torch.load(p['checkpoint'], map_location='cpu', weights_only=False)
        assert checkpoint['schema'] == 1 and checkpoint['normalization'] is None
        assert checkpoint['binding']['protocol_sha256'] == p['training_protocol_sha256']
        assert checkpoint['binding']['config'] == cfg_dict
        assert checkpoint['counters']['updates'] == p['selected_update']
        if p['evolution_receipt'] is not None:
            assert checkpoint['binding'].get('evolution') == p['evolution_receipt']
            assert 'evolution_sampler' in checkpoint
        cfg = config_factory('bc', dic=cfg_dict)
        ObsUtils.initialize_obs_utils_with_config(cfg)
        shapes = {'robot0_eef_pos': (3,), 'robot0_eef_quat': (4,), 'robot0_gripper_qpos': (2,), 'object': (14,)}
        costs['model_construction_attempts'] += 1
        save()
        model = algo_factory('bc', cfg, shapes, 7, torch.device('cpu'))
        costs['model_constructions'] += 1
        save()
        model.nets.load_state_dict(checkpoint['weights'], strict=True)
        model.set_eval()
        assert model.nets['policy'].low_noise_eval is True
        result['deployment'] = {'rnn_reset_each_episode': True, 'low_noise_eval': True, 'device': 'cpu', 'clipping': 'native action_spec'}
        with h5py.File(p['dataset'], 'r') as f:
            meta = json.loads(f['data'].attrs['env_args'])
        assert meta['env_name'] == 'NutAssemblySquare' and meta['env_version'] == '1.4.1'
        result['env_metadata'] = meta
        for seed in p['seeds']:
            for repeat in range(p['repeats']):
                budget()
                case = {'seed': seed, 'repeat': repeat, 'steps': 0, 'max_success_hold': 0, 'ever_native_success': False, 'grasp_steps': 0, 'success': False, 'stop_reason': 'not_started', 'clipped_components': 0}
                result['cases'].append(case)
                action_seed = p['action_seed_offset'] + seed * 2 + repeat
                case['action_seed'] = action_seed
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(action_seed)
                costs['constructors'] += 1
                save()
                env = suite.make(meta['env_name'], **meta['env_kwargs'])
                try:
                    np.random.seed(seed)
                    costs['explicit_resets'] += 1
                    save()
                    obs = env.reset()
                    model.reset()
                    native_model = env.sim.model._model
                    signature = {k: getattr(native_model, k).tolist() for k in ['actuator_ctrllimited', 'actuator_ctrlrange', 'actuator_forcelimited', 'actuator_forcerange']}
                    signature['actuator_names'] = [mujoco.mj_id2name(native_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(native_model.nu)]
                    assert signature == p['compiled_limits'], 'native compiled limits mismatch'
                    assert native_model.opt.timestep == p['timestep']
                    case['compiled_limits'] = signature
                    case['legacy_20N_steps'] = 0
                    assert env.single_object_mode == 2 and env.nut_id == 0
                    obj = env.nuts[env.nut_id]
                    component_names = list(env.nut_id_to_sensors[env.nut_id])
                    case['object_components'] = component_names
                    (lower, upper) = env.action_spec
                    assert lower.shape == upper.shape == (7,)
                    hold = 0
                    with gzip.open(out / f'seed{seed}_repeat{repeat}.jsonl.gz', 'wt') as log:

                        def emit(row):
                            log.write(json.dumps(safe(row), allow_nan=False) + '\n')
                            log.flush()

                        def measure(reward_value=None):
                            state = env.sim.get_state().flatten()
                            eef = np.asarray(obs['robot0_eef_pos'])
                            force = np.asarray(env.robots[0].ee_force)
                            mapped = {k: np.asarray(obs['object-state' if k == 'object' else k]) for k in shapes}
                            for (k, shape) in shapes.items():
                                assert mapped[k].shape == shape
                            parts = [np.asarray(obs[name]) for name in component_names]
                            assert [part.shape for part in parts] == [(3,), (4,), (3,), (4,)]
                            assert np.array_equal(mapped['object'], np.concatenate(parts), equal_nan=True)
                            finite = bool(all((np.isfinite(v).all() for v in [state, force, *mapped.values()])) and (reward_value is None or math.isfinite(float(reward_value))))
                            force_norm = float(np.linalg.norm(force))
                            workspace = bool(all((lo <= x <= hi for (x, (lo, hi)) in zip(eef, p['workspace_xyz']))))
                            data = env.sim.data._data
                            actuation = inspect_actuation(data.ctrl.tolist(), data.actuator_force.tolist(), signature, p['actuator_tolerance'])
                            decision = guard_decision(finite, workspace, force_norm, actuation)
                            return (mapped, {'state': state.tolist(), 'native_observation': {k: v.tolist() for (k, v) in mapped.items()}, 'force_vector_N': force.tolist(), 'force_norm_N': force_norm, 'finite': finite, 'workspace_ok': workspace, 'guard': decision['stop'], 'guard_decision': decision, 'actuation': actuation})
                        (mapped, initial) = measure()
                        initial_success = bool(env._check_success())
                        case['initial_native_success'] = initial_success
                        emit({'event': 'initial', **initial, 'native_success': initial_success})
                        case['initial_state_sha256'] = hashlib.sha256(np.asarray(initial['state'], dtype=np.float64).tobytes()).hexdigest()
                        if str(seed) in p['expected_initial_states'] and case['initial_state_sha256'] != p['expected_initial_states'][str(seed)]:
                            raise ValueError('paired initial state mismatch')
                        if initial_success:
                            case['stop_reason'] = 'initial_already_success'
                            save()
                            continue
                        if initial['guard']:
                            case['stop_reason'] = 'initial_guard'
                            save()
                            continue
                        for step in range(1, p['horizon'] + 1):
                            budget()
                            costs['inference_attempts'] += 1
                            save()
                            tensor_obs = {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for (k, v) in mapped.items()}
                            with torch.no_grad():
                                raw = model.get_action(tensor_obs).squeeze(0).numpy()
                            costs['inference_returned'] += 1
                            assert raw.shape == (7,)
                            if not np.isfinite(raw).all():
                                emit({'event': 'inference_nonfinite', 'step': step, 'raw_action': raw.tolist()})
                                case['stop_reason'] = 'inference_nonfinite'
                                break
                            action = np.clip(raw, lower, upper)
                            clipped = int(np.count_nonzero(raw != action))
                            costs['clipped_components'] += clipped
                            case['clipped_components'] += clipped
                            assert costs['action_attempts'] < p['max_actions']
                            costs['action_attempts'] += 1
                            emit({'event': 'attempt', 'step': step, 'raw_action': raw.tolist(), 'action': action.tolist(), 'clipped_components': clipped})
                            save()
                            (obs, reward, done, info) = env.step(action)
                            costs['returned_steps'] += 1
                            emit({'event': 'step_returned', 'step': step, 'reward': float(reward), 'done': bool(done)})
                            save()
                            (mapped, measured) = measure(reward)
                            case['legacy_20N_steps'] += int(measured['guard_decision']['legacy_20N_exceeded'])
                            native = bool(env._check_success())
                            grasp = bool(env._check_grasp(env.robots[0].gripper, obj.contact_geoms))
                            hold = hold + 1 if native and (not measured['guard']) else 0
                            case.update(steps=step, max_success_hold=max(case['max_success_hold'], hold), ever_native_success=case['ever_native_success'] or native)
                            case['grasp_steps'] += int(grasp)
                            emit({'event': 'result', 'step': step, **measured, 'native_success': native, 'native_grasp': grasp, 'success_hold': hold})
                            save()
                            if measured['guard']:
                                case['stop_reason'] = 'guard'
                                break
                            if hold >= p['success_hold']:
                                case.update(stop_reason='success_hold', success=True)
                                break
                        else:
                            case['stop_reason'] = 'horizon'
                finally:
                    env.close()
                save()
        result['status'] = 'completed'
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        save()
