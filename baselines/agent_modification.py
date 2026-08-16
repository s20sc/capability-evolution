"""
Baseline: Agent Modification (AM).

Each iteration, the agent's own policy network is updated via RL.
Capabilities (ECMs) remain fixed.
This represents the "modify the agent directly" paradigm.
"""

import numpy as np
import torch
import copy
from typing import Dict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym


def _make_env_fn(env_config: dict):
    """Factory for SubprocVecEnv — each process creates its own env."""
    def _init():
        from envs.robosuite_wrapper import RobosuiteWrapper
        return RobosuiteWrapper(**env_config)
    return _init


class AgentModificationBaseline:
    """
    Baseline that modifies the agent's planning policy directly.

    Unlike our method, the "agent" here is a monolithic PPO policy
    that gets retrained each iteration. No ECM separation.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        total_timesteps_per_iter: int = 20000,
        n_envs: int = 8,
        seed: int = 42,
    ):
        self.env = env
        self.total_timesteps = total_timesteps_per_iter
        self.n_envs = n_envs

        # Use SubprocVecEnv for parallel training
        if hasattr(env, 'env_name') and n_envs > 1:
            env_config = {
                "env_name": env.env_name,
                "object_pos_noise": env.object_pos_noise,
                "obs_noise_std": env.obs_noise_std,
                "actuation_failure_prob": env.actuation_failure_prob,
                "force_limit": env.force_limit,
                "horizon": env._env.horizon,
                "task_name": getattr(env, 'task_name', None),
            }
            self.vec_env = SubprocVecEnv([_make_env_fn(env_config) for _ in range(n_envs)])
        else:
            self.vec_env = DummyVecEnv([lambda: env])

        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=learning_rate,
            seed=seed,
            device="cpu",
            verbose=0,
        )

        # Save initial params for drift measurement
        self._initial_params = self._get_param_vector()

    def train_iteration(self) -> Dict:
        """Train one evolution iteration."""
        self.model.learn(
            total_timesteps=self.total_timesteps,
            reset_num_timesteps=False,
        )
        return {"method": "agent_modification", "timesteps": self.total_timesteps}

    def evaluate(self, num_episodes: int = 30) -> Dict:
        """Evaluate current policy."""
        successes = 0
        total_reward = 0.0
        episode_rewards = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward

            episode_rewards.append(ep_reward)
            if info.get("success", False):
                successes += 1
            total_reward += ep_reward

        success_rate = successes / num_episodes * 100
        variance = np.var(episode_rewards)

        return {
            "success_rate": success_rate,
            "avg_reward": total_reward / num_episodes,
            "reward_variance": variance,
            "policy_drift": self.compute_policy_drift(),
        }

    def compute_policy_drift(self) -> float:
        """Δπ: normalised L2 drift from initial params."""
        current = self._get_param_vector()
        diff = current - self._initial_params
        norm = np.linalg.norm(self._initial_params)
        if norm < 1e-10:
            return 0.0
        return float(np.linalg.norm(diff) / norm)

    def _get_param_vector(self) -> np.ndarray:
        params = self.model.policy.parameters()
        return np.concatenate([
            p.detach().cpu().numpy().ravel() for p in params
        ])

    def close(self):
        if hasattr(self, 'vec_env'):
            self.vec_env.close()
