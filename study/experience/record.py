"""Record actual returned transitions; never synthesize terminal observations."""
from __future__ import annotations

import math


def terminal_hold_actions(interface, gripper: float, max_steps: int):
    """Hold the single measured terminal pose, not repeated motion deltas."""
    if type(max_steps) is not int or not 0 < max_steps <= 5:
        raise ValueError('bounded terminal extension')
    finite_vector([gripper], 1)
    if abs(gripper) > 1:
        raise ValueError('gripper bounds')
    target_pose = interface.get_robot_eef_pose().copy()
    for _ in range(max_steps):
        yield [*interface.target_pose_to_action(target_pose), gripper]


def finite_vector(values: list, size: int | None = None) -> list[float]:
    if not isinstance(values, list) or not values or (size is not None and len(values) != size):
        raise ValueError('vector shape')
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
        raise ValueError('finite numeric vector')
    return [float(v) for v in values]


class EpisodeRecorder:
    """Keep attempts separate from transitions that actually returned."""
    def __init__(self) -> None:
        self.transitions: list[dict] = []
        self.pending: dict | None = None

    def begin(self, obs: list, state: list, action: list, phase: str) -> dict:
        if self.pending is not None or phase not in {'generation', 'terminal_hold'}:
            raise ValueError('pending transition or invalid phase')
        row = {'obs': finite_vector(obs, 23), 'state': finite_vector(state),
               'action': finite_vector(action, 7), 'phase': phase, 'return_observed': False}
        if any(abs(v) > 1 for v in row['action']):
            raise ValueError('action bounds')
        if self.transitions:
            last = self.transitions[-1]
            if row['obs'] != last['next_obs'] or row['state'] != last['next_state']:
                raise ValueError('discontinuous transition')
        self.pending = row
        return row

    def mark_returned(self) -> None:
        if self.pending is None or self.pending['return_observed']:
            raise ValueError('return without pending call or duplicate return')
        self.pending['return_observed'] = True

    def finish(self, next_obs: list, next_state: list, reward: float,
               success: bool, guard_stop: bool) -> None:
        if self.pending is None or not self.pending['return_observed']:
            raise ValueError('validated transition without observed return')
        if type(reward) not in (float, int) or not math.isfinite(reward):
            raise ValueError('nonfinite reward')
        if type(success) is not bool or type(guard_stop) is not bool:
            raise ValueError('boolean outcome required')
        post_obs = finite_vector(next_obs, 23)
        post_state = finite_vector(next_state, len(self.pending['state']))
        self.transitions.append({**self.pending, 'next_obs': post_obs,
                                 'next_state': post_state, 'reward': float(reward),
                                 'success': success, 'guard_stop': guard_stop})
        self.pending = None

    def eligible(self, stop_reason: str, required_hold: int = 5) -> bool:
        if type(required_hold) is not int or required_hold <= 0:
            raise ValueError('positive hold count')
        return (self.pending is None and stop_reason == 'success_hold'
                and len(self.transitions) >= required_hold
                and not any(r['guard_stop'] for r in self.transitions)
                and all(r['success'] for r in self.transitions[-required_hold:]))

    def snapshot(self) -> dict:
        return {'transitions': self.transitions, 'unfinished_transition': self.pending}
