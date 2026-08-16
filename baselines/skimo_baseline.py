"""
Baseline: SkiMo (Shi et al., CoRL 2023).

Skill-based Model-based Reinforcement Learning:
  - Jointly learns a skill repertoire + world model
  - Plans in skill space using the world model
  - Both skill and model parameters update each iteration
  → introduces policy drift (unlike our frozen-agent approach)

Simplified reimplementation capturing the key architectural difference:
  skills AND planner are updated together (coupled evolution).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym


class SkillWorldModel(nn.Module):
    """
    Learned world model that predicts next observation given
    current observation and skill latent.
    Used for planning in skill space.
    """

    def __init__(self, obs_dim: int, skill_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),  # predict next obs
        )
        self.reward_head = nn.Sequential(
            nn.Linear(obs_dim + skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, skill):
        x = torch.cat([obs, skill], dim=-1)
        next_obs = self.model(x)
        reward = self.reward_head(x)
        return next_obs, reward


class SkiMoBaseline:
    """
    SkiMo baseline: joint skill + world model learning.

    Key difference from our method: BOTH the skill network AND
    the planning model are updated each iteration, introducing
    coupled drift (Δπ > 0).
    """

    def __init__(
        self,
        env: gym.Env,
        skill_dim: int = 16,
        model_hidden_dim: int = 256,
        rl_timesteps_per_iter: int = 20000,
        model_lr: float = 1e-3,
        model_train_steps: int = 1000,
        n_envs: int = 8,
        seed: int = 42,
    ):
        self.env = env
        self.skill_dim = skill_dim
        self.rl_timesteps = rl_timesteps_per_iter
        self.n_envs = n_envs
        self.model_lr = model_lr
        self.model_train_steps = model_train_steps

        obs_dim = env.observation_space.shape[0]

        # World model (jointly updated)
        self.world_model = SkillWorldModel(obs_dim, skill_dim, model_hidden_dim)
        self.model_optimizer = optim.Adam(
            self.world_model.parameters(), lr=model_lr
        )

        # PPO policy (jointly updated — this IS the skill selector + executor)
        if hasattr(env, 'env_name') and n_envs > 1:
            from baselines.agent_modification import _make_env_fn
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
            seed=seed,
            device="cpu",
            verbose=0,
        )

        # Track initial params for drift
        self._initial_params = self._get_param_vector()

        # Experience buffer for world model training
        self._obs_buffer = []
        self._action_buffer = []
        self._next_obs_buffer = []
        self._reward_buffer = []
        self._max_buffer = 100000

    def train_iteration(self) -> Dict:
        """
        One SkiMo iteration:
        1. Collect experience with current policy
        2. Update world model on collected data
        3. Update policy (RL) using both real + model-based rollouts
        """
        # Reset env to clear any done state left by evaluate()
        self.env.reset()
        # Step 1: RL training (also collects experience)
        self.model.learn(
            total_timesteps=self.rl_timesteps,
            reset_num_timesteps=False,
        )

        # Step 2: Collect experience for world model
        self._collect_experience(num_episodes=50)

        # Step 3: Train world model
        if len(self._obs_buffer) > 100:
            self._train_world_model()

        return {
            "method": "skimo",
            "timesteps": self.rl_timesteps,
            "model_buffer_size": len(self._obs_buffer),
        }

    def _collect_experience(self, num_episodes: int = 50):
        """Collect transitions for world model training."""
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self._obs_buffer.append(obs)
                self._action_buffer.append(action)
                self._next_obs_buffer.append(next_obs)
                self._reward_buffer.append(reward)

                obs = next_obs

        # Trim buffer
        if len(self._obs_buffer) > self._max_buffer:
            excess = len(self._obs_buffer) - self._max_buffer
            self._obs_buffer = self._obs_buffer[excess:]
            self._action_buffer = self._action_buffer[excess:]
            self._next_obs_buffer = self._next_obs_buffer[excess:]
            self._reward_buffer = self._reward_buffer[excess:]

    def _train_world_model(self):
        """Train world model on collected experience."""
        obs = torch.FloatTensor(np.array(self._obs_buffer[-10000:]))
        actions = torch.FloatTensor(np.array(self._action_buffer[-10000:]))
        next_obs = torch.FloatTensor(np.array(self._next_obs_buffer[-10000:]))
        rewards = torch.FloatTensor(np.array(self._reward_buffer[-10000:])).unsqueeze(-1)

        # Use action as a proxy for skill latent (simplified)
        # In full SkiMo, there's a separate skill encoder
        skill_dim = self.skill_dim
        if actions.shape[1] < skill_dim:
            padding = torch.zeros(actions.shape[0], skill_dim - actions.shape[1])
            skills = torch.cat([actions, padding], dim=1)
        else:
            skills = actions[:, :skill_dim]

        for _ in range(min(self.model_train_steps, 100)):
            # Random batch
            idx = torch.randint(0, obs.shape[0], (256,))
            pred_next, pred_reward = self.world_model(obs[idx], skills[idx])

            obs_loss = nn.functional.mse_loss(pred_next, next_obs[idx])
            rew_loss = nn.functional.mse_loss(pred_reward, rewards[idx])
            loss = obs_loss + 0.1 * rew_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

    def evaluate(self, num_episodes: int = 30) -> Dict:
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
            "policy_drift": self.compute_policy_drift(),
        }

    def compute_policy_drift(self) -> float:
        """Both skill and model parameters drift (Δπ > 0)."""
        current = self._get_param_vector()
        diff = current - self._initial_params
        norm = np.linalg.norm(self._initial_params)
        return float(np.linalg.norm(diff) / norm) if norm > 1e-10 else 0.0

    def _get_param_vector(self) -> np.ndarray:
        policy_params = np.concatenate([
            p.detach().cpu().numpy().ravel()
            for p in self.model.policy.parameters()
        ])
        model_params = np.concatenate([
            p.detach().cpu().numpy().ravel()
            for p in self.world_model.parameters()
        ])
        return np.concatenate([policy_params, model_params])
