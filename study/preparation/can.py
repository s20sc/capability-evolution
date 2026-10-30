"""Original native reconstruction and split-preserving packaging bodies."""
import gzip,json,sys,time
from pathlib import Path
from types import SimpleNamespace
from preparation_helpers import sha,safe

def extract(root: Path, p: dict) -> None:
    args = SimpleNamespace(protocol_sha256=sha(root / "protocol.json"))
    out = root / "run"
    out.mkdir(exist_ok=False)
    (out / "protocol.json").write_bytes((root / "protocol.json").read_bytes())
    start = time.monotonic()
    costs = dict(constructors=0, explicit_resets=0, xml_restores=0, sim_resets=0,
                 state_write_attempts=0, observation_refresh_attempts=0,
                 simulation_actions=0, gradient_updates=0, external_model_calls=0)
    result = dict(state='running', protocol_sha256=args.protocol_sha256, costs=costs, demos=[])

    def save() -> None:
        result['wall_seconds'] = time.monotonic() - start
        tmp = out / 'status.tmp'
        tmp.write_text(json.dumps(safe(result), allow_nan=False, indent=2) + '\n')
        tmp.replace(out / 'status.json')

    save()
    try:
        sys.path.append(p['robosuite_site_packages'])
        import numpy as np
        import h5py
        import robosuite as suite
        assert suite.__version__ == '1.4.1'
        assert Path(suite.__file__).resolve() == Path(p['robosuite_site_packages']) / 'robosuite/__init__.py'
        result['robosuite_origin'] = suite.__file__
        keys = p['observation_keys']
        with h5py.File(p['dataset'], 'r') as f:
            meta = json.loads(f['data'].attrs['env_args'])
            assert meta['env_version'] == '1.4.1' and meta['env_name'] == 'PickPlaceCan'
            train = [x.decode() for x in f['mask/train'][:]]
            valid = [x.decode() for x in f['mask/valid'][:]]
            assert len(set(train)) == len(train) == 180 and len(set(valid)) == len(valid) == 20
            assert set(train).isdisjoint(valid) and set(train) | set(valid) == set(f['data'].keys())
            result['split'] = dict(train_ids=train, valid_ids=valid, disjoint=True)
            assert set(p['demos']) == set(f['data'].keys()) and len(p['demos']) == 200, 'full dataset IDs mismatch'
            assert sum(f['data'][name]['actions'].shape[0] for name in p['demos']) == p['expected_samples'], 'sample count mismatch'
            for name in p['demos']:
                if time.monotonic() - start > p['soft_seconds']:
                    raise TimeoutError('wall cap before construction')
                g = f['data'][name]
                states, actions = g['states'][:], g['actions'][:]
                n = p['demo_lengths'][name]
                assert states.shape == (n, 71) and actions.shape == (n, 7), 'frozen demo shape mismatch'
                assert np.isfinite(states).all() and np.isfinite(actions).all()
                assert np.abs(actions).max() <= 1, 'source action bound mismatch'
                result['active_demo'] = name
                result['last_operation'] = 'constructor_attempt'
                costs['constructors'] += 1
                save()
                np.random.seed(p['construction_seed'])
                env = suite.make(meta['env_name'], **meta['env_kwargs'])
                try:
                    costs['explicit_resets'] += 1
                    result['last_operation'] = 'explicit_reset_attempt'
                    save()
                    env.reset()
                    xml = env.edit_model_xml(g.attrs['model_file'])
                    costs['xml_restores'] += 1
                    result['last_operation'] = 'xml_restore_attempt'
                    save()
                    env.reset_from_xml_string(xml)
                    costs['sim_resets'] += 1
                    env.sim.reset()
                    row = dict(demo=name, split='train' if name in train else 'valid', samples=0)
                    result['demos'].append(row)
                    with gzip.open(out / (name + '.jsonl.gz'), 'wt') as log:
                        def extract(index: int, phase: str):
                            if time.monotonic() - start > p['soft_seconds']:
                                raise TimeoutError('wall cap')
                            assert costs['state_write_attempts'] < p['max_state_writes']
                            costs['state_write_attempts'] += 1
                            result['last_operation'] = dict(event='state_write_attempt', demo=name, index=index, phase=phase)
                            log.write(json.dumps(result['last_operation']) + '\n')
                            log.flush()
                            save()
                            env.sim.set_state_from_flattened(states[index])
                            env.sim.forward()
                            costs['observation_refresh_attempts'] += 1
                            obs = env._get_observations(force_update=True)
                            state = env.sim.get_state().flatten()
                            features = {key: np.asarray(obs[key]).copy() for key in keys}
                            measurement = dict(event='raw_measurement', index=index, phase=phase,
                                state=state.tolist(), reference_state=states[index].tolist(),
                                shapes={k: list(v.shape) for k, v in features.items()},
                                observations={k: v.tolist() for k, v in features.items()})
                            log.write(json.dumps(safe(measurement), allow_nan=False) + '\n')
                            log.flush()
                            result['last_operation'] = dict(event='measurement_validation', demo=name, index=index, phase=phase)
                            save()
                            assert [v.shape for v in features.values()] == [(3,), (4,), (2,), (14,)], 'feedback shape mismatch'
                            assert all(np.isfinite(x).all() for x in [state, *features.values()]), 'nonfinite state/feedback'
                            assert state.shape == states[index].shape, 'state shape mismatch'
                            assert np.allclose(state, states[index], rtol=0, atol=1e-12), 'state restoration mismatch'
                            vector = np.concatenate(list(features.values()))
                            event = dict(event='observation', index=index, phase=phase, state=state.tolist(),
                                         observations={k: v.tolist() for k, v in features.items()})
                            log.write(json.dumps(event, allow_nan=False) + '\n')
                            log.flush()
                            return vector, features

                        vectors = []
                        for index in range(len(states)):
                            vector, features = extract(index, 'dataset')
                            vectors.append(vector)
                            row['samples'] += 1
                            save()
                        errors = []
                        for index in [0, len(states) - 1, 0, len(states) - 1]:
                            vector, _ = extract(index, 'cache_probe')
                            errors.append(float(np.max(np.abs(vector - vectors[index]))))
                        row['cache_probe_errors'] = errors
                        assert max(errors) <= p['cache_tolerance'], 'cache-order dependence exceeds tolerance'
                    vectors = np.stack(vectors)
                    row.update(feature_dim=int(vectors.shape[1]), observation_keys=keys)
                    np.savez_compressed(out / (name + '.npz'), observations=vectors, actions=actions)
                    save()
                finally:
                    env.close()
            assert sum(d['samples'] for d in result['demos']) == p['expected_samples'], 'output sample count mismatch'
            assert costs['state_write_attempts'] == p['max_state_writes'], 'output write count mismatch'
            result['state'] = 'completed'
    except Exception as exc:
        result.update(state='failed', error=repr(exc))
        raise
    finally:
        save()


def package(root: Path, p: dict, source: Path, e: dict, s: dict) -> None:
    args = SimpleNamespace(protocol_sha256=sha(root / "protocol.json"))
    out = root / "run"
    out.mkdir(exist_ok=False)
    (out / "protocol.json").write_bytes((root / "protocol.json").read_bytes())
    start = time.monotonic()
    report = dict(state='running', protocol_sha256=args.protocol_sha256, checked_demos=[],
                  simulation_actions=0, gradient_updates=0, external_model_calls=0)

    def save() -> None:
        report['wall_seconds'] = time.monotonic() - start
        tmp = out / 'status.tmp'
        tmp.write_text(json.dumps(safe(report), allow_nan=False, indent=2) + '\n')
        tmp.replace(out / 'status.json')

    save()
    try:
        import numpy as np
        import h5py
        packs = {name: dict(observations=[], actions=[], ids=[], offsets=[0]) for name in ['train', 'valid']}
        hashes = {}
        with h5py.File(e['dataset'], 'r') as raw:
            train = [x.decode() for x in raw['mask/train'][:]]
            valid = [x.decode() for x in raw['mask/valid'][:]]
            assert s['split']['train_ids'] == train and s['split']['valid_ids'] == valid
            assert len(set(train)) == 180 and len(set(valid)) == 20 and set(train).isdisjoint(valid)
            assert set(train) | set(valid) == set(e['demos'])
            for d in s['demos']:
                if time.monotonic() - start > p['soft_seconds']:
                    raise TimeoutError('package wall cap')
                name, n = d['demo'], e['demo_lengths'][d['demo']]
                report['active_demo'] = name
                save()
                npz, gz = source / 'run' / (name + '.npz'), source / 'run' / (name + '.jsonl.gz')
                hashes[name] = dict(npz_sha256=sha(npz), gzip_sha256=sha(gz))
                with np.load(npz, allow_pickle=False) as a:
                    obs, actions = a['observations'].copy(), a['actions'].copy()
                assert obs.shape == (n, 23) and actions.shape == (n, 7)
                assert str(obs.dtype) == str(actions.dtype) == 'float64', 'source dtype mismatch'
                assert np.isfinite(obs).all() and np.isfinite(actions).all() and np.abs(actions).max() <= 1
                states = raw['data'][name]['states'][:]
                assert np.array_equal(actions, raw['data'][name]['actions'][:]), 'action label drift'
                cache_errors = []
                with gzip.open(gz, 'rt') as log:
                    for j in range(n + 4):
                        attempt, measurement, value = [json.loads(next(log)) for _ in range(3)]
                        assert [x['event'] for x in [attempt, measurement, value]] == ['state_write_attempt', 'raw_measurement', 'observation']
                        index = j if j < n else [0, n - 1, 0, n - 1][j - n]
                        phase = 'dataset' if j < n else 'cache_probe'
                        assert all(x['index'] == index and x['phase'] == phase for x in [attempt, measurement, value])
                        state = np.asarray(value['state'])
                        assert np.allclose(state, states[index], rtol=0, atol=1e-12)
                        assert value['state'] == measurement['state'] and value['observations'] == measurement['observations']
                        vector = np.concatenate([value['observations'][k] for k in e['observation_keys']])
                        if j < n:
                            assert np.array_equal(vector, obs[index]), 'packed feedback mismatch'
                        else:
                            cache_errors.append(float(np.max(np.abs(vector - obs[index]))))
                            assert np.allclose(vector, obs[index], rtol=0, atol=e['cache_tolerance']), 'cache probe mismatch'
                    assert next(log, None) is None, 'unexpected extra events'
                split = 'train' if name in train else 'valid'
                assert d['split'] == split and d['samples'] == n and d['feature_dim'] == 23
                assert len(d['cache_probe_errors']) == 4 and max(d['cache_probe_errors']) <= e['cache_tolerance']
                assert cache_errors == d['cache_probe_errors'], 'cache error receipt mismatch'
                q = packs[split]
                q['observations'].append(obs)
                q['actions'].append(actions)
                q['ids'].append(name)
                q['offsets'].append(q['offsets'][-1] + n)
                report['checked_demos'].append(dict(demo=name, split=split, samples=n))
                save()
        report['splits'] = {}
        for name, q in packs.items():
            dest = out / (name + '.npz')
            np.savez_compressed(dest, observations=np.concatenate(q['observations']),
                actions=np.concatenate(q['actions']), demo_ids=np.asarray(q['ids']),
                episode_offsets=np.asarray(q['offsets'], dtype=np.int64))
            report['splits'][name] = dict(demos=len(q['ids']), samples=q['offsets'][-1],
                                        sha256=sha(dest), dtype=str(q['observations'][0].dtype))
        assert report['splits']['train']['demos'] == 180 and report['splits']['valid']['demos'] == 20
        assert sum(x['samples'] for x in report['splits'].values()) == 23207
        (out / 'source_manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
        report['state'] = 'completed'
    except Exception as exc:
        report.update(state='failed', error=repr(exc))
        raise
    finally:
        save()
