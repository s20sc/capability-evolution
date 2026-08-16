"""
Hybrid Learner — orchestrates the three learning modalities.

Controls which modalities are active (for ablation experiments)
and manages the Update function from Eq. 4 in the paper.
"""

import copy
import numpy as np
from typing import Dict, Optional, Tuple
import gymnasium as gym

from .rl_learner import RLLearner
from .il_learner import ILLearner
from .llm_synthesizer import LLMSynthesizer
import inspect


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Keep only kwargs that cls.__init__ accepts."""
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in valid}


class HybridLearner:
    """
    Implements the UPDATE function: C(t+1) = Update(C(t), D_t).

    Orchestrates RL, IL, and LLM synthesis modalities.
    Supports ablation by enabling/disabling individual modalities.
    """

    def __init__(
        self,
        rl_config: Optional[Dict] = None,
        il_config: Optional[Dict] = None,
        llm_config: Optional[Dict] = None,
        enable_rl: bool = True,
        enable_il: bool = True,
        enable_llm: bool = True,
    ):
        self.enable_rl = enable_rl
        self.enable_il = enable_il
        self.enable_llm = enable_llm

        if enable_rl:
            self.rl_learner = RLLearner(**_filter_kwargs(RLLearner, rl_config or {}))
        else:
            self.rl_learner = None

        if enable_il:
            self.il_learner = ILLearner(**_filter_kwargs(ILLearner, il_config or {}))
        else:
            self.il_learner = None

        if enable_llm:
            self.llm_synthesizer = LLMSynthesizer(**_filter_kwargs(LLMSynthesizer, llm_config or {}))
        else:
            self.llm_synthesizer = None

    def update_ecm(
        self,
        ecm,
        env: gym.Env,
        runtime=None,
        demonstrations=None,
        failure_traces=None,
    ) -> Dict:
        """
        Execute one evolution step for a single ECM.

        Order:
        1. IL bootstrap (if demos available and IL enabled)
        2. RL fine-tuning (main gradient-based improvement)
        3. LLM synthesis (structural changes based on failures)

        The ECM's version gating decides whether to keep each update.
        """
        metrics = {"modalities_applied": []}

        # Save pre-update state for potential rollback
        pre_state = copy.deepcopy(ecm.network.state_dict())

        # --- Phase 1: Imitation Learning ---
        if self.enable_il and self.il_learner and demonstrations is not None:
            il_metrics = self.il_learner.train(
                ecm.network, env, demonstrations
            )
            metrics["il"] = il_metrics
            metrics["modalities_applied"].append("il")

        # --- Phase 2: RL fine-tuning ---
        if self.enable_rl and self.rl_learner:
            model, rl_metrics = self.rl_learner.finetune_ecm(
                ecm, env, runtime
            )
            metrics["rl"] = rl_metrics
            metrics["modalities_applied"].append("rl")

        # --- Phase 3: LLM Synthesis ---
        if self.enable_llm and self.llm_synthesizer:
            if failure_traces:
                for trace in failure_traces:
                    self.llm_synthesizer.record_failure(trace)

            if self.llm_synthesizer.should_synthesize():
                llm_metrics = self.llm_synthesizer.synthesize(
                    ecm, env_info={"env": str(env)}
                )
                metrics["llm"] = llm_metrics
                metrics["modalities_applied"].append("llm")

        return metrics

    def get_ablation_config(self, ablation_name: str) -> Dict:
        """
        Return modality flags for a named ablation configuration.
        Matches configs/default.yaml ablation section.
        """
        configs = {
            "rl_only":    {"enable_rl": True,  "enable_il": False, "enable_llm": False},
            "il_only":    {"enable_rl": False, "enable_il": True,  "enable_llm": False},
            "llm_only":   {"enable_rl": False, "enable_il": False, "enable_llm": True},
            "rl_il":      {"enable_rl": True,  "enable_il": True,  "enable_llm": False},
            "rl_llm":     {"enable_rl": True,  "enable_il": False, "enable_llm": True},
            "il_llm":     {"enable_rl": False, "enable_il": True,  "enable_llm": True},
            "all_three":  {"enable_rl": True,  "enable_il": True,  "enable_llm": True},
        }
        if ablation_name not in configs:
            raise ValueError(
                f"Unknown ablation: {ablation_name}. "
                f"Available: {list(configs.keys())}"
            )
        return configs[ablation_name]
