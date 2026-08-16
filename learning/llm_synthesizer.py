"""
LLM-based Code Synthesis Modality.

Analyses failure traces and proposes code-level modifications to ECMs.
Uses an LLM to suggest structural changes (control flow, parameters)
that gradient-based methods cannot discover.
"""

import json
import numpy as np
from typing import Dict, List, Optional


class LLMSynthesizer:
    """
    LLM-based ECM code synthesis.

    Workflow:
    1. Collect failure traces from recent executions
    2. Summarise failures into a structured prompt
    3. Query LLM for proposed code/parameter modifications
    4. Apply modifications and test via version gating

    NOTE: In the current implementation, this is a STUB that applies
    heuristic modifications. For full functionality, set up an API key
    and enable the LLM backend.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_failures_before_synthesis: int = 10,
        enabled: bool = True,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_failures = max_failures_before_synthesis
        self.enabled = enabled
        self.api_key = api_key
        self._failure_buffer: List[Dict] = []

    def record_failure(self, trace: Dict):
        """Record a failure trace for later analysis."""
        self._failure_buffer.append(trace)

    def should_synthesize(self) -> bool:
        """Check if enough failures have accumulated."""
        return (
            self.enabled
            and len(self._failure_buffer) >= self.max_failures
        )

    def synthesize(self, ecm, env_info: Dict) -> Dict:
        """
        Propose modifications to an ECM based on failure analysis.

        In the full implementation, this calls the LLM API.
        Current stub applies heuristic modifications:
          - Reduce action magnitude if force violations detected
          - Increase exploration if stuck in local minima
          - Adjust gripper timing based on grasp failures
        """
        if not self.enabled:
            return {"method": "llm_synthesis", "applied": False, "reason": "disabled"}

        # Analyse failure patterns
        failure_types = self._analyse_failures()

        modifications = []

        if failure_types.get("force_violations", 0) > 3:
            # Heuristic: scale down action outputs
            self._scale_action_outputs(ecm, scale=0.8)
            modifications.append("scaled_down_actions_for_force")

        if failure_types.get("timeout", 0) > 5:
            # Heuristic: increase exploration by raising log_std
            self._increase_exploration(ecm, delta=0.1)
            modifications.append("increased_exploration")

        if failure_types.get("grasp_failure", 0) > 3:
            # Heuristic: adjust gripper bias
            self._adjust_gripper_bias(ecm, bias=0.2)
            modifications.append("adjusted_gripper_bias")

        # Clear buffer
        self._failure_buffer = []

        return {
            "method": "llm_synthesis",
            "applied": len(modifications) > 0,
            "modifications": modifications,
            "failure_analysis": failure_types,
        }

    def _analyse_failures(self) -> Dict:
        """Categorise failures from the buffer."""
        types = {
            "force_violations": 0,
            "timeout": 0,
            "grasp_failure": 0,
            "workspace_violation": 0,
            "other": 0,
        }
        for trace in self._failure_buffer:
            reason = trace.get("failure_reason", "other")
            if "force" in reason:
                types["force_violations"] += 1
            elif "timeout" in reason or "truncated" in reason:
                types["timeout"] += 1
            elif "grasp" in reason:
                types["grasp_failure"] += 1
            elif "workspace" in reason or "bounds" in reason:
                types["workspace_violation"] += 1
            else:
                types["other"] += 1
        return types

    def _scale_action_outputs(self, ecm, scale: float = 0.8):
        """Scale down the output layer weights to reduce action magnitude."""
        import torch
        with torch.no_grad():
            # Find the last linear layer
            for module in reversed(list(ecm.network.net.modules())):
                if isinstance(module, torch.nn.Linear):
                    module.weight *= scale
                    if module.bias is not None:
                        module.bias *= scale
                    break

    def _increase_exploration(self, ecm, delta: float = 0.1):
        """Increase log_std to encourage more exploration."""
        import torch
        with torch.no_grad():
            ecm.network.log_std += delta

    def _adjust_gripper_bias(self, ecm, bias: float = 0.2):
        """Add positive bias to gripper action dimension."""
        import torch
        with torch.no_grad():
            for module in reversed(list(ecm.network.net.modules())):
                if isinstance(module, torch.nn.Linear):
                    if module.bias is not None and module.bias.shape[0] >= 4:
                        module.bias[-1] += bias  # gripper is last dim
                    break
