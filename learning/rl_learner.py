"""
RL Learning Modality — SAC via stable-baselines3 with SubprocVecEnv.

Trains ECM networks on robosuite environments using off-policy RL.
SAC is far more sample-efficient than PPO for manipulation tasks.
SubprocVecEnv runs multiple envs in parallel for faster data collection.
"""

import numpy as np
import torch
from typing import Dict, Optional, List, Tuple

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym
import copy


class ECMPolicyWrapper(gym.Env):
    """
    Wraps a robosuite env so that a single ECM acts as the policy.
    Used to train one ECM at a time with SB3.
    """

    def __init__(self, env: gym.Env, runtime=None):
        super().__init__()
        self.env = env
        self.runtime = runtime
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.runtime is not None:
            ee_pos = np.zeros(3)
            ee_vel = np.zeros(3)
            safe_action, blocked, gov_info = self.runtime.enforce(
                action, ee_pos, ee_vel
            )
            return self.env.step(safe_action)
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


def _make_env_fn(env_config: dict, worker_idx: int = 0):
    """Factory function for SubprocVecEnv — each process creates its own env."""
    def _init():
        from envs.robosuite_wrapper import RobosuiteWrapper
        cfg = dict(env_config)
        # Each worker gets a distinct derived seed (base seed + worker index),
        # which controls the wrapper's own streams (observation noise,
        # actuation failures). The underlying robosuite env receives no seed,
        # so its internal placement stream remains uncontrolled.
        if cfg.get("seed") is not None:
            cfg["seed"] = cfg["seed"] + worker_idx
        env = RobosuiteWrapper(**cfg)
        return ECMPolicyWrapper(env)
    return _init


def _inject_ecm_to_sac(ecm, model):
    """Copy ECM network weights into SAC's actor (mean part)."""
    actor = model.policy.actor
    ecm_sd = ecm.network.state_dict()

    with torch.no_grad():
        # ECM net: [Linear(0), ReLU(1), Linear(2), ReLU(3), Linear(4), Tanh(5)]
        # SAC actor.latent_pi: [Linear(0), ReLU(1), Linear(2), ReLU(3)]
        # SAC actor.mu: Linear(256, act_dim)
        latent = actor.latent_pi
        latent[0].weight.copy_(ecm_sd["net.0.weight"])
        latent[0].bias.copy_(ecm_sd["net.0.bias"])
        latent[2].weight.copy_(ecm_sd["net.2.weight"])
        latent[2].bias.copy_(ecm_sd["net.2.bias"])
        # Output mean -> mu
        actor.mu.weight.copy_(ecm_sd["net.4.weight"])
        actor.mu.bias.copy_(ecm_sd["net.4.bias"])


def _extract_sac_to_ecm_state(model):
    """Extract SAC actor weights as an ECM-compatible state_dict."""
    actor = model.policy.actor
    latent = actor.latent_pi

    state_dict = {
        "net.0.weight": latent[0].weight.detach().clone(),
        "net.0.bias": latent[0].bias.detach().clone(),
        "net.2.weight": latent[2].weight.detach().clone(),
        "net.2.bias": latent[2].bias.detach().clone(),
        "net.4.weight": actor.mu.weight.detach().clone(),
        "net.4.bias": actor.mu.bias.detach().clone(),
    }
    return state_dict


class RLLearner:
    """
    RL-based ECM training using SAC + parallel environments.

    One ECM is trained at a time; the persistent agent and other ECMs
    are held fixed. Weights are transferred between ECM and SAC actor.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        gamma: float = 0.99,
        tau: float = 0.005,
        total_timesteps_per_iter: int = 20000,
        n_envs: int = 8,
        gradient_steps: int = 1,
        verbose: int = 0,
        # Legacy PPO params (ignored, kept for config compat)
        n_steps: int = 2048,
        n_epochs: int = 10,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        algorithm: str = "SAC",
    ):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.tau = tau
        self.total_timesteps = total_timesteps_per_iter
        self.n_envs = n_envs
        self.gradient_steps = gradient_steps
        self.verbose = verbose
        # Persistent SAC models per ECM (keeps replay buffer across iterations)
        self._sac_cache: Dict[str, Tuple] = {}  # ecm_name -> (model, vec_env)

    def _make_vec_env(self, env, n_envs=None):
        """Create a vectorized environment for parallel training."""
        n = n_envs or self.n_envs

        # If the env is a wrapper, get its config for subprocess creation
        base_env = env
        if isinstance(env, ECMPolicyWrapper):
            base_env = env.env

        # For SubprocVecEnv, we need to create fresh envs in each process
        # Extract config from the existing env to recreate it
        if hasattr(base_env, 'env_name'):
            # "task_name" is not forwarded: robosuite 1.4.1 env constructors
            # reject unknown keyword arguments. The seed is threaded through
            # so subprocess envs are seed-controlled; each worker derives a
            # distinct seed in _make_env_fn.
            env_config = {
                "env_name": base_env.env_name,
                "object_pos_noise": base_env.object_pos_noise,
                "obs_noise_std": base_env.obs_noise_std,
                "actuation_failure_prob": base_env.actuation_failure_prob,
                "force_limit": base_env.force_limit,
                "horizon": base_env._env.horizon,
                "seed": getattr(base_env, "seed", None),
            }
            # Use SubprocVecEnv for true parallelism
            if n > 1:
                return SubprocVecEnv([
                    _make_env_fn(env_config, i) for i in range(n)
                ])

        # Fallback: single env wrapped in DummyVecEnv
        return DummyVecEnv([lambda: ECMPolicyWrapper(env)])

    def _make_sac(self, vec_env, ecm_hidden_dims=(256, 256)):
        """Create a SAC model with architecture matching the ECM."""
        return SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=self.lr,
            batch_size=self.batch_size,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            gamma=self.gamma,
            tau=self.tau,
            gradient_steps=self.gradient_steps,
            verbose=self.verbose,
            device="auto",
            policy_kwargs={"net_arch": list(ecm_hidden_dims)},
        )

    def train(self, env: gym.Env, runtime=None) -> Dict:
        """Train a fresh SAC agent. Returns (model, metrics)."""
        vec_env = self._make_vec_env(env)
        model = self._make_sac(vec_env)
        model.learn(total_timesteps=self.total_timesteps)
        vec_env.close()
        return model, {"method": "rl_sac", "timesteps": self.total_timesteps}

    def finetune_ecm(self, ecm, env: gym.Env, runtime=None) -> Dict:
        """
        Fine-tune an ECM using SAC with persistent replay buffer.
        On first call for an ECM, creates SAC model + vec_env.
        On subsequent calls, reuses the same model (keeps replay buffer).
        """
        ecm_name = ecm.descriptor.name

        if ecm_name not in self._sac_cache:
            # First time: create model and vec_env
            hidden_dims = [
                m.out_features for m in ecm.network.net
                if hasattr(m, 'out_features')
            ][:-1]
            vec_env = self._make_vec_env(env)
            model = self._make_sac(vec_env, ecm_hidden_dims=hidden_dims or (256, 256))
            self._sac_cache[ecm_name] = (model, vec_env)

        model, vec_env = self._sac_cache[ecm_name]

        # If restored from checkpoint without vec_env, create one now
        if vec_env is None:
            vec_env = self._make_vec_env(env)
            # n_envs may have changed since checkpoint was saved;
            # reload model with new env instead of set_env
            if model.n_envs != vec_env.num_envs:
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    model.save(tmp.name)
                    model = SAC.load(tmp.name, env=vec_env, device="auto")
                    os.unlink(tmp.name)
            else:
                model.set_env(vec_env)
            self._sac_cache[ecm_name] = (model, vec_env)

        # Sync current ECM weights into SAC actor
        _inject_ecm_to_sac(ecm, model)

        # Continue training (replay buffer accumulates across iterations)
        model.learn(total_timesteps=self.total_timesteps, reset_num_timesteps=False)

        # Extract trained weights back to ECM (only mean, keep ECM's log_std)
        new_state = _extract_sac_to_ecm_state(model)
        new_state["log_std"] = ecm.network.state_dict()["log_std"]
        ecm.evolve(new_state)

        return model, {"method": "rl_sac_finetune", "timesteps": self.total_timesteps}

    def close(self):
        """Clean up persistent environments."""
        for model, vec_env in self._sac_cache.values():
            if vec_env is not None:
                vec_env.close()
        self._sac_cache.clear()
