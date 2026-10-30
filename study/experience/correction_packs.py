import sys
from pathlib import Path
from .utility import read, write, sha, correction_source
from .log_inputs import read_episode, SHAPES

def run(root, p) -> None:
    for item in p['parent_logs'].values():
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError('parent trajectory changed after configuration')
    if sha(root / 'acquisition/result.json') != p['acquisition_result_sha256']:
        raise ValueError('generated result changed after configuration')
    sys.path.insert(0, p['vendor'])
    import numpy as np
    acquisition = read(root / 'acquisition/result.json')
    if acquisition['status'] != 'completed':
        raise ValueError('acquisition engineering failure')
    cases = {c['seed']: c for c in acquisition['cases']}
    if set(cases) != set(p['acquisition_seeds']):
        raise ValueError('acquisition membership')
    (parent, generated) = ({}, {})
    for seed in p['acquisition_seeds']:
        episode = read_episode(Path(p['parent_logs'][str(seed)]['path']))
        parent[seed] = ([[x for key in SHAPES for x in obs[key]] for obs in episode['inputs']], episode['actions'])
        case = cases[seed]
        if case['initial_state_sha256'] != p['parent_logs'][str(seed)]['initial_state_sha256']:
            raise ValueError('generation/parent initial state mismatch')
        record = read(root / 'acquisition' / ('episode-' + str(seed) + '.json'))
        if sha(root / 'acquisition' / ('episode-' + str(seed) + '.json')) != case['episode_sha256']:
            raise ValueError('generated episode identity')
        if case['training_eligible']:
            rows = record['transitions']
            if record['unfinished_transition'] is not None or len(rows) < 5:
                raise ValueError('unfinished accepted generation')
            if not all((x['success'] and (not x['guard_stop']) for x in rows[-5:])):
                raise ValueError('accepted hold5')
            if not all((x['return_observed'] for x in rows)):
                raise ValueError('unreturned action')
            if any((x['next_obs'] != y['obs'] or x['next_state'] != y['state'] for (x, y) in zip(rows, rows[1:]))):
                raise ValueError('generated continuity')
            generated[seed] = ([x['obs'] for x in rows], [x['action'] for x in rows])
    out = root / 'packs'
    out.mkdir()
    manifest = {}
    for (arm, specification) in p['arms'].items():
        if arm == 'C0':
            continue
        (observations, actions, ids, offsets, sources) = ([], [], [], [0], [])
        for seed in specification['seeds']:
            source = correction_source(seed, specification['source'], seed in generated)
            (obs, act) = (generated if source == 'generated' else parent)[seed]
            observations.extend(obs)
            actions.extend(act)
            ids.append(str(seed))
            offsets.append(len(actions))
            sources.append({'seed': seed, 'source': source, 'fallback': specification['source'] == 'G' and source == 'parent'})
        path = out / (arm + '.npz')
        arrays = {'observations': np.asarray(observations, dtype=np.float64), 'actions': np.asarray(actions, dtype=np.float64), 'demo_ids': np.asarray(ids, dtype='U40'), 'episode_offsets': np.asarray(offsets, dtype=np.int64)}
        if arrays['observations'].shape != (len(actions), 23) or arrays['actions'].shape != (len(actions), 7):
            raise ValueError('pack shapes')
        if not np.isfinite(arrays['observations']).all() or not np.isfinite(arrays['actions']).all() or np.max(np.abs(arrays['actions'])) > 1:
            raise ValueError('pack values')
        np.savez(path, **arrays)
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(arrays) or any((not np.array_equal(loaded[k], v) for (k, v) in arrays.items())):
                raise ValueError('pack roundtrip')
        manifest[arm] = {'path': str(path), 'sha256': sha(path), 'rows': len(actions), 'episodes': 3, 'sources': sources}
    write(root / 'packs.json', manifest)
