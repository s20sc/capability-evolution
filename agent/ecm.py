"""
Embodied Capability Module (ECM) — robosuite-compatible version.

Each ECM c^(k) = (θ^(k), I_c, O_c, φ_c):
  - θ^(k): neural network weights (learnable)
  - I_c, O_c: input/output interface
  - φ_c: capability descriptor

Lifecycle: creation → deployment → evolution → deprecation
With version gating and rollback support.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ECMDescriptor:
    """Capability descriptor φ_c."""
    name: str
    description: str
    phase: str              # manipulation phase: reach, grasp, lift, place, etc.
    input_dim: int = 0      # set dynamically from env
    output_dim: int = 0     # set dynamically from env


class ECMNetwork(nn.Module):
    """
    Neural network parameterisation θ^(k) for an ECM.
    Two hidden layers, Gaussian policy output.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = (256, 256),
        activation: str = "relu",
    ):
        super().__init__()

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}[activation]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), act_fn()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # actions in [-1, 1]

        self.net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(-5, 2))
        return mean, std

    def get_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[float]]:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            mean, std = self.forward(obs_t)
            if deterministic:
                action = mean
                log_prob = None
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1).item()
        return action.squeeze(0).numpy(), log_prob

    def get_log_prob(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(-1)

    def param_vector(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        return np.concatenate([
            p.detach().cpu().numpy().ravel() for p in self.parameters()
        ])


class ECM:
    """
    Embodied Capability Module with lifecycle management.
    Supports: creation, deployment, evolution, version gating, rollback, deprecation.
    """

    def __init__(
        self,
        descriptor: ECMDescriptor,
        hidden_dims: List[int] = (256, 256),
        activation: str = "relu",
        version_history_size: int = 5,
        rollback_threshold: float = 0.05,
    ):
        self.descriptor = descriptor
        self.version_history_size = version_history_size
        self.rollback_threshold = rollback_threshold

        self.network = ECMNetwork(
            descriptor.input_dim, descriptor.output_dim,
            hidden_dims, activation,
        )
        self.version = 0
        self.status = "created"

        self._version_history: List[Dict] = []
        self._save_version()
        self.performance_history: List[float] = []

    def deploy(self):
        self.status = "deployed"

    def deprecate(self):
        self.status = "deprecated"

    def get_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[float]]:
        if self.status == "deprecated":
            raise RuntimeError(f"ECM '{self.descriptor.name}' is deprecated")
        return self.network.get_action(obs, deterministic)

    def evolve(self, new_state_dict: dict):
        """Update to a new version k+1. Saves k for rollback."""
        self._save_version()
        self.network.load_state_dict(new_state_dict)
        self.version += 1

    def evaluate_and_gate(self, performance: float) -> bool:
        """
        Version gating: accept update only if performance doesn't degrade
        beyond rollback_threshold. Returns True if accepted.
        """
        self.performance_history.append(performance)
        if len(self.performance_history) >= 2:
            prev = self.performance_history[-2]
            if performance < prev - self.rollback_threshold:
                self.rollback()
                return False
        return True

    def rollback(self):
        """Roll back to the previous version."""
        if len(self._version_history) >= 2:
            self._version_history.pop()
            prev = self._version_history[-1]
            self.network.load_state_dict(prev["state_dict"])
            self.version = prev["version"]
            if self.performance_history:
                self.performance_history.pop()

    def _save_version(self):
        snapshot = {
            "version": self.version,
            "state_dict": copy.deepcopy(self.network.state_dict()),
        }
        self._version_history.append(snapshot)
        if len(self._version_history) > self.version_history_size:
            self._version_history.pop(0)

    def param_vector(self) -> np.ndarray:
        return self.network.param_vector()


class ECMRegistry:
    """
    Registry managing the capability set C(t).
    Handles creation, lookup, lifecycle of all ECMs.
    """

    def __init__(self):
        self._ecms: Dict[str, ECM] = {}

    def register(self, ecm: ECM):
        self._ecms[ecm.descriptor.name] = ecm
        ecm.deploy()

    def get(self, name: str) -> Optional[ECM]:
        return self._ecms.get(name)

    def get_by_phase(self, phase: str) -> Optional[ECM]:
        for ecm in self._ecms.values():
            if ecm.descriptor.phase == phase and ecm.status != "deprecated":
                return ecm
        return None

    def active_ecms(self) -> List[ECM]:
        return [e for e in self._ecms.values() if e.status != "deprecated"]

    def snapshot(self) -> Dict[str, dict]:
        return {
            name: copy.deepcopy(ecm.network.state_dict())
            for name, ecm in self._ecms.items()
        }

    def total_param_vector(self) -> np.ndarray:
        vecs = [ecm.param_vector() for ecm in self._ecms.values()]
        return np.concatenate(vecs) if vecs else np.array([])
