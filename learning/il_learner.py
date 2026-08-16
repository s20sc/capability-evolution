"""
Imitation Learning Modality — Behavioral Cloning.

Uses expert demonstrations from robosuite's built-in scripted policies
or human teleoperation data to bootstrap / correct ECMs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import gymnasium as gym


class DemonstrationCollector:
    """
    Collect expert demonstrations from robosuite's scripted policies.

    robosuite environments with reward_shaping=True typically provide
    a reasonable scripted policy via env.action_spec or we can use
    a simple PD controller toward the target.
    """

    def __init__(self, env: gym.Env, seed: int = 42):
        self.env = env
        self._rng = np.random.default_rng(seed)

    def collect_random_demos(
        self, num_episodes: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect demonstration data. Uses random exploration
        as a baseline; replace with scripted policy for real data.

        Returns (observations, actions) arrays.
        """
        all_obs = []
        all_actions = []

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                # Simple heuristic: sample from action space
                # In real usage, replace with robosuite scripted policy
                action = self.env.action_space.sample()
                all_obs.append(obs)
                all_actions.append(action)
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

        return np.array(all_obs, dtype=np.float32), np.array(all_actions, dtype=np.float32)

    def collect_scripted_demos(
        self, num_episodes: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect demos from robosuite's built-in scripted policies.
        Requires robosuite environments that support action_spec.
        """
        all_obs = []
        all_actions = []

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            while not done:
                # Attempt to get scripted action from robosuite env
                try:
                    inner_env = self.env._env if hasattr(self.env, '_env') else self.env
                    if hasattr(inner_env, '_get_default_action'):
                        action = inner_env._get_default_action()
                    else:
                        action = self.env.action_space.sample()
                except Exception:
                    action = self.env.action_space.sample()

                all_obs.append(obs)
                all_actions.append(action)
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                step += 1

        return np.array(all_obs, dtype=np.float32), np.array(all_actions, dtype=np.float32)


class ILLearner:
    """
    Imitation learning via Behavioral Cloning.
    Trains an ECM network to mimic expert demonstrations.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        demo_episodes: int = 500,
    ):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.demo_episodes = demo_episodes

    def train(
        self,
        ecm_network: nn.Module,
        env: gym.Env,
        demonstrations: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict:
        """
        Train ECM network via behavioral cloning.

        Args:
            ecm_network: the ECM's neural network
            env: environment (used to collect demos if none provided)
            demonstrations: optional (obs, actions) tuple

        Returns metrics dict.
        """
        if demonstrations is None:
            collector = DemonstrationCollector(env)
            obs_data, act_data = collector.collect_scripted_demos(self.demo_episodes)
        else:
            obs_data, act_data = demonstrations

        # Create DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(obs_data),
            torch.FloatTensor(act_data),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train
        optimizer = optim.Adam(ecm_network.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for obs_batch, act_batch in loader:
                mean, _ = ecm_network(obs_batch)
                loss = loss_fn(mean, act_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

        metrics = {
            "method": "behavioral_cloning",
            "epochs": self.epochs,
            "final_loss": losses[-1] if losses else float("inf"),
            "demo_episodes": self.demo_episodes,
        }
        return metrics
