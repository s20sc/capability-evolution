"""
Persistent Agent A = (π_A, M_A^id, M_A^ep).

The agent's decision policy π_A and identity memory M_A^id are FROZEN.
Only episodic memory M_A^ep grows over time.

The agent decomposes tasks into ECM invocation sequences.
"""

import copy
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
from .ecm import ECMRegistry, ECM


class PersistentAgent:
    """
    Persistent Agent with frozen identity.

    Identity invariant: (π_A, M_A^id) never changes.
    The agent selects which ECM to invoke based on the current
    observation and task phase.
    """

    # Manipulation phases — the agent cycles through these
    PHASES = ["reach", "grasp", "lift", "place"]

    def __init__(
        self,
        obs_dim: int,
        ecm_registry: ECMRegistry,
        hidden_dim: int = 128,
        identity_dim: int = 64,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        # Infer action dim from first ECM
        active = ecm_registry.active_ecms()
        self.act_dim = active[0].descriptor.output_dim if active else obs_dim
        self.ecm_registry = ecm_registry
        self.hidden_dim = hidden_dim
        self.identity_dim = identity_dim

        torch.manual_seed(seed)

        # π_A: a simple phase-selection network (FROZEN)
        self._planner = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(self.PHASES)),
        )
        # Freeze planner
        for p in self._planner.parameters():
            p.requires_grad = False

        # M_A^id: identity memory (FROZEN)
        self._identity_memory = torch.randn(identity_dim)
        self._identity_memory.requires_grad = False

        # Snapshot for drift measurement
        self._initial_planner_params = self._get_planner_vector()
        self._initial_identity = self._identity_memory.clone()

        # M_A^ep: episodic memory (grows over time)
        self._episodic_memory: List[Dict] = []
        self._max_episodic = 1000

        # Current phase tracker
        self._current_phase_idx = 0

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Agent decision loop:
        1. Determine current phase from planner
        2. Dispatch to the appropriate ECM
        3. Return the ECM's action
        """
        # Phase selection via planner
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs)
            logits = self._planner(obs_t)
            phase_idx = logits.argmax().item()

        phase = self.PHASES[phase_idx % len(self.PHASES)]
        ecm = self.ecm_registry.get_by_phase(phase)

        if ecm is None:
            # Fallback: use first available ECM
            active = self.ecm_registry.active_ecms()
            ecm = active[0] if active else None

        if ecm is None:
            return np.zeros(self.act_dim), {"phase": phase, "ecm": None}

        action, log_prob = ecm.get_action(obs, deterministic)

        info = {
            "phase": phase,
            "ecm": ecm.descriptor.name,
            "ecm_version": ecm.version,
            "log_prob": log_prob,
        }
        return action, info

    def add_episodic_memory(self, record: Dict):
        """Append to episodic memory (M_A^ep grows, identity untouched)."""
        self._episodic_memory.append(record)
        if len(self._episodic_memory) > self._max_episodic:
            self._episodic_memory.pop(0)

    def compute_policy_drift(self) -> float:
        """
        Δπ: normalised L2 drift between initial and current planner params.
        Should always be 0.0 for our method (frozen by design).
        """
        current = self._get_planner_vector()
        diff = current - self._initial_planner_params
        norm = np.linalg.norm(self._initial_planner_params)
        if norm < 1e-10:
            return 0.0
        return float(np.linalg.norm(diff) / norm)

    def _get_planner_vector(self) -> np.ndarray:
        return np.concatenate([
            p.detach().cpu().numpy().ravel()
            for p in self._planner.parameters()
        ])
