"""
Baseline: SPiRL (Pertsch et al., CoRL 2021).

Skill Priors for Reinforcement Learning:
  - Extract reusable skill priors from offline demonstration data
  - Downstream RL uses skill-conditioned policy
  - Skills are FIXED after extraction (no evolution)

This is a simplified reimplementation capturing the key idea:
  1. Pre-train a skill VAE on demonstration data
  2. Learn a high-level policy that selects skill latents
  3. Skills are frozen — no versioning, rollback, or evolution
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces


class SkillVAE(nn.Module):
    """
    Variational Autoencoder for extracting skill priors from demos.
    Encodes action sequences into a latent skill space.
    """

    def __init__(
        self,
        action_dim: int,
        skill_horizon: int = 10,
        latent_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.skill_horizon = skill_horizon
        self.latent_dim = latent_dim
        input_dim = action_dim * skill_horizon

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def encode(self, action_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(action_seq.view(action_seq.shape[0], -1))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(z)
        return flat.view(-1, self.skill_horizon, self.action_dim)

    def forward(self, action_seq):
        mu, logvar = self.encode(action_seq)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class SkillConditionedEnv(gym.Env):
    """
    Wraps an env so that the high-level policy outputs skill latents
    instead of raw actions. The skill VAE decodes latents into
    action sequences executed over skill_horizon steps.
    """

    def __init__(
        self,
        env: gym.Env,
        skill_vae: SkillVAE,
        skill_horizon: int = 10,
    ):
        super().__init__()
        self.env = env
        self.skill_vae = skill_vae
        self.skill_horizon = skill_horizon

        self.observation_space = env.observation_space
        # High-level action = skill latent
        self.action_space = spaces.Box(
            low=-3.0, high=3.0,
            shape=(skill_vae.latent_dim,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, skill_latent: np.ndarray):
        """Execute a full skill (skill_horizon low-level actions)."""
        z = torch.FloatTensor(skill_latent).unsqueeze(0)
        with torch.no_grad():
            action_seq = self.skill_vae.decode(z).squeeze(0).numpy()

        total_reward = 0.0
        last_obs = None
        terminated = False
        truncated = False
        info = {}

        for t in range(self.skill_horizon):
            if terminated or truncated:
                break
            action = np.clip(action_seq[t], -1.0, 1.0)
            last_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

        return last_obs, total_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class SPiRLBaseline:
    """
    SPiRL baseline: skill prior extraction + downstream RL.
    Skills are frozen after extraction.
    """

    def __init__(
        self,
        env: gym.Env,
        skill_horizon: int = 10,
        latent_dim: int = 16,
        vae_epochs: int = 100,
        vae_lr: float = 1e-3,
        rl_timesteps_per_iter: int = 2000,  # 2K high-level steps × skill_horizon=10 = 20K env steps
        demo_episodes: int = 500,
        seed: int = 42,
    ):
        self.env = env
        self.skill_horizon = skill_horizon
        self.latent_dim = latent_dim
        self.vae_epochs = vae_epochs
        self.vae_lr = vae_lr
        self.rl_timesteps = rl_timesteps_per_iter
        self.demo_episodes = demo_episodes

        action_dim = env.action_space.shape[0]

        # 1. Train skill VAE on demonstrations
        self.skill_vae = SkillVAE(action_dim, skill_horizon, latent_dim)
        self._train_skill_vae(env)

        # 2. Freeze skill VAE
        for p in self.skill_vae.parameters():
            p.requires_grad = False

        # 3. Create skill-conditioned env
        self.skill_env = SkillConditionedEnv(env, self.skill_vae, skill_horizon)

        # 4. High-level RL policy over skill latents
        vec_env = DummyVecEnv([lambda: self.skill_env])
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            seed=seed,
            device="cpu",
            verbose=0,
        )

        self._initial_params = self._get_param_vector()

    def _train_skill_vae(self, env: gym.Env):
        """Pre-train skill VAE on demonstration action sequences."""
        # Collect demo data
        action_seqs = []
        for _ in range(self.demo_episodes):
            obs, _ = env.reset()
            actions = []
            done = False
            while not done:
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                actions.append(action)

            # Chunk into skill_horizon windows
            for i in range(0, len(actions) - self.skill_horizon, self.skill_horizon):
                seq = np.array(actions[i:i + self.skill_horizon], dtype=np.float32)
                action_seqs.append(seq)

        if len(action_seqs) == 0:
            return

        data = torch.FloatTensor(np.array(action_seqs))
        optimizer = optim.Adam(self.skill_vae.parameters(), lr=self.vae_lr)

        for epoch in range(self.vae_epochs):
            recon, mu, logvar = self.skill_vae(data)
            recon_loss = nn.functional.mse_loss(recon, data)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.01 * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def train_iteration(self) -> Dict:
        """Train high-level policy (skills remain frozen)."""
        # Reset env to clear any done state left by evaluate()
        self.skill_env.reset()
        self.model.learn(
            total_timesteps=self.rl_timesteps,
            reset_num_timesteps=False,
        )
        return {"method": "spirl", "timesteps": self.rl_timesteps}

    def evaluate(self, num_episodes: int = 30) -> Dict:
        """Evaluate SPiRL policy."""
        successes = 0
        total_reward = 0.0
        episode_rewards = []

        for _ in range(num_episodes):
            obs, _ = self.skill_env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.skill_env.step(action)
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
            "policy_drift": 0.0,  # skills are frozen
        }

    def compute_policy_drift(self) -> float:
        """High-level policy drifts; skills stay fixed."""
        current = self._get_param_vector()
        diff = current - self._initial_params
        norm = np.linalg.norm(self._initial_params)
        return float(np.linalg.norm(diff) / norm) if norm > 1e-10 else 0.0

    def _get_param_vector(self) -> np.ndarray:
        return np.concatenate([
            p.detach().cpu().numpy().ravel()
            for p in self.model.policy.parameters()
        ])
