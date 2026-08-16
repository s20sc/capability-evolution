"""
Baseline: Static ECM (S-ECM).

Capabilities are frozen after initialisation.
The agent observes experience but ECMs are never updated.
Tests whether evolution is necessary.
"""

import numpy as np
from typing import Dict
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class StaticECMBaseline:
    """
    Baseline with modular ECMs that are never updated.
    Initialised with random weights and frozen.
    """

    def __init__(
        self,
        env: gym.Env,
        seed: int = 42,
    ):
        self.env = env
        # Create a PPO model but never train it — acts as frozen policy
        vec_env = DummyVecEnv([lambda: env])
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            seed=seed,
            device="cpu",
            verbose=0,
        )
        self._initial_params = self._get_param_vector()

    def train_iteration(self) -> Dict:
        """No training — capabilities are frozen."""
        return {"method": "static_ecm", "timesteps": 0}

    def evaluate(self, num_episodes: int = 30) -> Dict:
        """Evaluate frozen policy."""
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

        return {
            "success_rate": successes / num_episodes * 100,
            "avg_reward": total_reward / num_episodes,
            "reward_variance": np.var(episode_rewards),
            "policy_drift": 0.0,  # always zero — frozen
        }

    def compute_policy_drift(self) -> float:
        return 0.0

    def _get_param_vector(self) -> np.ndarray:
        return np.concatenate([
            p.detach().cpu().numpy().ravel()
            for p in self.model.policy.parameters()
        ])
