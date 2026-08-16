"""
Gymnasium-compatible wrapper around robosuite environments.

Maps robosuite's dict-based observation/action to flat vectors
compatible with our ECM networks and stable-baselines3.

Also injects stochastic perturbations described in the paper:
  - object position noise (±5cm)
  - observation noise (σ=0.01m)
  - actuation failure (5% per step)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any

try:
    import robosuite as suite
    from robosuite.wrappers import GymWrapper
    HAS_ROBOSUITE = True
except ImportError:
    HAS_ROBOSUITE = False


class RobosuiteWrapper(gym.Env):
    """
    Wraps a robosuite environment into a Gymnasium env with:
      - Flat observation vector
      - Continuous action space
      - Stochastic perturbations
      - Safety violation tracking
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env_name: str = "PickPlace",
        robot: str = "Panda",
        controller: str = "OSC_POSE",
        horizon: int = 500,
        reward_shaping: bool = True,
        object_pos_noise: float = 0.05,
        obs_noise_std: float = 0.01,
        actuation_failure_prob: float = 0.05,
        force_limit: float = 20.0,
        workspace_bounds: Optional[Dict] = None,
        forbidden_zones: Optional[list] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        if not HAS_ROBOSUITE:
            raise ImportError(
                "robosuite is required. Install with: pip install robosuite"
            )

        self.env_name = env_name
        self.object_pos_noise = object_pos_noise
        self.obs_noise_std = obs_noise_std
        self.actuation_failure_prob = actuation_failure_prob
        self.force_limit = force_limit
        self.workspace_bounds = workspace_bounds or {
            "x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.0, 0.6]
        }
        self.forbidden_zones = forbidden_zones or []
        # Record the constructor seed so vec-env factories can rebuild
        # subprocess envs with derived per-worker seeds (learning/rl_learner).
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Determine robots arg
        robots = [robot]
        if "TwoArm" in env_name:
            robots = [robot, robot]

        # Create robosuite env
        controller_config = suite.load_controller_config(
            default_controller=controller
        )

        self._env = suite.make(
            env_name,
            robots=robots,
            controller_configs=controller_config,
            has_renderer=(render_mode == "human"),
            has_offscreen_renderer=(render_mode == "rgb_array"),
            use_camera_obs=False,
            horizon=horizon,
            reward_shaping=reward_shaping,
            **kwargs,
        )

        # Determine obs/action dims from a reset
        obs = self._env.reset()
        self._obs_keys = sorted([
            k for k in obs.keys()
            if isinstance(obs[k], np.ndarray) and k != "image"
        ])
        flat_obs = self._flatten_obs(obs)
        self._obs_dim = flat_obs.shape[0]

        action_dim = self._env.action_spec[0].shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(action_dim,), dtype=np.float32,
        )

        # Safety tracking
        self.safety_violations = 0
        self._step_count = 0
        self._episode_forces = []

    def _flatten_obs(self, obs_dict: Dict) -> np.ndarray:
        """Flatten robosuite dict obs into a single vector."""
        parts = []
        for k in self._obs_keys:
            v = obs_dict[k]
            if v.ndim == 0:
                v = v.reshape(1)
            parts.append(v.astype(np.float32))
        return np.concatenate(parts)

    def _add_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add Gaussian observation noise (paper: σ=0.01m)."""
        if self.obs_noise_std > 0:
            noise = self._rng.normal(0, self.obs_noise_std, size=obs.shape)
            obs = obs + noise.astype(np.float32)
        return obs

    def _check_safety(self, obs_dict: Dict) -> bool:
        """Check if current state violates safety constraints."""
        violation = False

        # Extract end-effector position
        ee_pos = obs_dict.get("robot0_eef_pos", np.zeros(3))

        # Workspace bounds
        bounds = self.workspace_bounds
        if (ee_pos[0] < bounds["x"][0] or ee_pos[0] > bounds["x"][1] or
            ee_pos[1] < bounds["y"][0] or ee_pos[1] > bounds["y"][1] or
            ee_pos[2] < bounds["z"][0] or ee_pos[2] > bounds["z"][1]):
            violation = True

        # Forbidden zones
        for zone in self.forbidden_zones:
            if (zone["x"][0] <= ee_pos[0] <= zone["x"][1] and
                zone["y"][0] <= ee_pos[1] <= zone["y"][1]):
                violation = True

        # Force limit (from robosuite contact forces if available)
        if hasattr(self._env, 'robots') and len(self._env.robots) > 0:
            try:
                ee_force = np.linalg.norm(
                    self._env.robots[0].ee_force
                )
                self._episode_forces.append(ee_force)
                if ee_force > self.force_limit:
                    violation = True
            except (AttributeError, IndexError):
                pass

        if violation:
            self.safety_violations += 1
        return violation

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        obs_dict = self._env.reset()
        self.safety_violations = 0
        self._step_count = 0
        self._episode_forces = []

        flat_obs = self._flatten_obs(obs_dict)
        flat_obs = self._add_obs_noise(flat_obs)

        info = {
            "safety_violations": 0,
            "success": False,
        }
        return flat_obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Actuation failure: with probability p, replace action with zeros
        if self._rng.random() < self.actuation_failure_prob:
            action = np.zeros_like(action)

        # Step robosuite
        obs_dict, reward, done, info = self._env.step(action)

        # Safety check
        safety_violation = self._check_safety(obs_dict)

        # Flatten obs with noise
        flat_obs = self._flatten_obs(obs_dict)
        flat_obs = self._add_obs_noise(flat_obs)

        # Determine success from robosuite info
        success = self._env._check_success() if hasattr(self._env, '_check_success') else False

        terminated = done and success
        truncated = done and not success

        info.update({
            "safety_violations": self.safety_violations,
            "safety_violation_this_step": safety_violation,
            "success": success,
            "step": self._step_count,
        })

        return flat_obs, float(reward), terminated, truncated, info

    def close(self):
        self._env.close()

    def render(self):
        return self._env.render()
