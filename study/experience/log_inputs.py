"""Strict stdlib reader for successful, previously exposed policy episodes."""
import gzip
import json
import math
from pathlib import Path

SHAPES = {'robot0_eef_pos': 3, 'robot0_eef_quat': 4,
          'robot0_gripper_qpos': 2, 'object': 14}


def vector(value: list, size: int) -> None:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError('vector shape')
    if not all(type(v) in (float, int) and math.isfinite(v) for v in value):
        raise ValueError('finite numeric vector')


def observation(row: dict) -> dict:
    obs = row['native_observation']
    if set(obs) != set(SHAPES):
        raise ValueError('observation fields')
    for key, size in SHAPES.items():
        vector(obs[key], size)
    if row['guard'] is not False:
        raise ValueError('guarded trajectory')
    return obs


def parse_rows(rows: list[dict], max_steps: int = 400) -> dict:
    if not rows or rows[0].get('event') != 'initial':
        raise ValueError('missing initial observation')
    if rows[0].get('native_success') is not False:
        raise ValueError('initial success')
    previous = observation(rows[0])
    if (len(rows) - 1) % 3:
        raise ValueError('unfinished transition')
    count = (len(rows) - 1) // 3
    if not 1 <= count <= max_steps:
        raise ValueError('step budget')
    inputs, actions = [], []
    hold = 0
    for index in range(count):
        attempt, returned, result = rows[1 + index * 3:4 + index * 3]
        for row, event in zip((attempt, returned, result), ('attempt', 'step_returned', 'result')):
            if row.get('event') != event or row.get('step') != index + 1:
                raise ValueError('event ordering or step mismatch')
        vector(attempt['action'], 7)
        vector(attempt['raw_action'], 7)
        if any(abs(v) > 1 for v in attempt['action']):
            raise ValueError('action bounds')
        if attempt['action'] != [max(-1., min(1., x)) for x in attempt['raw_action']]:
            raise ValueError('clipping mismatch')
        next_obs = observation(result)
        if type(result['native_success']) is not bool:
            raise ValueError('success flag type')
        hold = hold + 1 if result['native_success'] else 0
        if result['success_hold'] != hold or (hold >= 5 and index != count - 1):
            raise ValueError('hold counter or post-success actions')
        inputs.append(previous)
        actions.append(attempt['action'])
        previous = next_obs
    if hold != 5:
        raise ValueError('not a completed hold5 success')
    return {'inputs': inputs, 'actions': actions, 'steps': count}


def read_episode(path: Path) -> dict:
    with gzip.open(path, 'rb') as stream:
        content = stream.read(16 * 1024 * 1024 + 1)
    if len(content) > 16 * 1024 * 1024:
        raise ValueError('decompressed input limit')
    return parse_rows([json.loads(line) for line in content.splitlines()])


def top_tail_mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError('drift values')
    k = math.ceil(len(values) / 5)
    return sum(sorted(values, reverse=True)[:k]) / k
