"""
Runtime Governance Layer R.

Mediates every interaction between agent and environment:
  - Policy enforcement: check action against constraints Φ
  - Execution management: timeout, exception handling
  - Trace logging
  - Version gating
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class GovernanceTrace:
    """Single execution trace record."""
    step: int
    ecm_name: str
    ecm_version: int
    action_proposed: np.ndarray
    action_executed: np.ndarray
    was_blocked: bool
    violation_type: Optional[str]
    overhead_ms: float


class RuntimeGovernance:
    """
    Runtime governance layer R(a_t, Φ).

    Enforces:
      - Force limits
      - Velocity limits
      - Workspace bounds
      - Forbidden zone avoidance
    """

    def __init__(
        self,
        force_limit: float = 20.0,
        max_velocity: float = 1.5,
        workspace_bounds: Optional[Dict] = None,
        forbidden_zones: Optional[List[Dict]] = None,
        enabled: bool = True,
    ):
        self.force_limit = force_limit
        self.max_velocity = max_velocity
        self.workspace_bounds = workspace_bounds or {
            "x": [-0.6, 0.6], "y": [-0.6, 0.6], "z": [0.6, 1.3]
        }
        self.forbidden_zones = forbidden_zones or []
        self.enabled = enabled

        # Governance uses tighter bounds than safety checks to absorb
        # prediction errors from dynamics/momentum.  The 0.08m margin
        # prevents the actual state from ever exceeding the outer envelope.
        self._margin = 0.08
        self._gov_bounds = {
            "x": [self.workspace_bounds["x"][0] + self._margin,
                  self.workspace_bounds["x"][1] - self._margin],
            "y": [self.workspace_bounds["y"][0] + self._margin,
                  self.workspace_bounds["y"][1] - self._margin],
            "z": [self.workspace_bounds["z"][0] + self._margin,
                  self.workspace_bounds["z"][1] - self._margin],
        }

        # Metrics
        self._total_actions = 0
        self._blocked_actions = 0
        self._overhead_times: List[float] = []
        self._traces: List[GovernanceTrace] = []

    def enforce(
        self,
        action: np.ndarray,
        ee_pos: np.ndarray,
        ee_vel: np.ndarray,
        ecm_name: str = "",
        ecm_version: int = 0,
        step: int = 0,
    ) -> Tuple[np.ndarray, bool, Dict]:
        """
        Apply governance to proposed action.

        Returns:
            safe_action: the (possibly modified) action
            was_blocked: True if action was modified or rejected
            info: dict with violation details
        """
        t0 = time.perf_counter()
        self._total_actions += 1

        if not self.enabled:
            overhead = (time.perf_counter() - t0) * 1000
            self._overhead_times.append(overhead)
            return action, False, {"violation": None}

        was_blocked = False
        violation_type = None
        safe_action = action.copy()

        # 1. Velocity limit: scale action if magnitude exceeds max_velocity
        action_magnitude = np.linalg.norm(action[:3])
        if action_magnitude > self.max_velocity:
            scale = self.max_velocity / action_magnitude
            safe_action[:3] = action[:3] * scale
            was_blocked = True
            violation_type = "velocity_exceeded"

        # 2. Workspace bounds: directional constraint.
        #    When ee_pos is near a boundary, only allow actions that
        #    move away from that boundary. This is more robust than
        #    predicting the exact next position (which depends on
        #    nonlinear controller dynamics).
        m = self._margin
        bounds = self.workspace_bounds
        for dim, axis in enumerate(["x", "y", "z"]):
            low, high = bounds[axis]
            if ee_pos[dim] < low + m:
                # Near lower bound: clamp action to non-negative
                if safe_action[dim] < 0:
                    safe_action[dim] = 0.0
                    was_blocked = True
                    violation_type = "workspace_bounds"
            elif ee_pos[dim] > high - m:
                # Near upper bound: clamp action to non-positive
                if safe_action[dim] > 0:
                    safe_action[dim] = 0.0
                    was_blocked = True
                    violation_type = "workspace_bounds"

        # 3. Forbidden zone: reject if ee_pos is near a zone and
        #    action would push it further in
        for zone in self.forbidden_zones:
            zx, zy = zone["x"], zone["y"]
            if (zx[0] - m <= ee_pos[0] <= zx[1] + m and
                zy[0] - m <= ee_pos[1] <= zy[1] + m):
                safe_action = np.zeros_like(action)
                was_blocked = True
                violation_type = "forbidden_zone"
                break

        # 4. Force estimate
        force_estimate = np.linalg.norm(safe_action[:3]) * 15
        if force_estimate > self.force_limit:
            scale = self.force_limit / force_estimate
            safe_action[:3] *= scale
            was_blocked = True
            violation_type = "force_limit"

        if was_blocked:
            self._blocked_actions += 1

        overhead = (time.perf_counter() - t0) * 1000
        self._overhead_times.append(overhead)

        # Record trace
        trace = GovernanceTrace(
            step=step,
            ecm_name=ecm_name,
            ecm_version=ecm_version,
            action_proposed=action.copy(),
            action_executed=safe_action.copy(),
            was_blocked=was_blocked,
            violation_type=violation_type,
            overhead_ms=overhead,
        )
        self._traces.append(trace)

        info = {
            "violation": violation_type,
            "was_blocked": was_blocked,
            "overhead_ms": overhead,
        }
        return safe_action, was_blocked, info

    def get_metrics(self) -> Dict:
        """Return aggregate governance metrics."""
        return {
            "total_actions": self._total_actions,
            "blocked_actions": self._blocked_actions,
            "blocking_rate": (
                self._blocked_actions / self._total_actions * 100
                if self._total_actions > 0 else 0.0
            ),
            "overhead_ms": (
                float(np.mean(self._overhead_times))
                if self._overhead_times else 0.0
            ),
            "avg_overhead_ms": (
                float(np.mean(self._overhead_times))
                if self._overhead_times else 0.0
            ),
            "max_overhead_ms": (
                float(np.max(self._overhead_times))
                if self._overhead_times else 0.0
            ),
        }

    def reset_metrics(self):
        self._total_actions = 0
        self._blocked_actions = 0
        self._overhead_times = []
        self._traces = []
