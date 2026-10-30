"""Finite Counter-BC objective; task-specific target sampling is an adaptation."""
from __future__ import annotations

import hashlib
import math
import random

import torch


def counterfactual_cross_entropy(log_density: torch.Tensor) -> torch.Tensor:
    """Eq. 10: gradients include the normalized policy-dependent weights.

    The last axis enumerates targets. All other axes contain valid positions,
    with no padding. Values are log densities and may be positive.
    """
    if log_density.ndim < 1 or log_density.numel() == 0:
        raise ValueError("nonempty target axis required")
    if not torch.is_floating_point(log_density) or not bool(torch.isfinite(log_density).all()):
        raise ValueError("finite floating log densities required")
    weights = torch.softmax(log_density, dim=-1)
    return -(weights * log_density).sum(dim=-1).mean()


def sample_arm_targets(action: list[float], radius: float, count: int,
                       seed: int, row_key: str) -> list[list[float]]:
    """Anchor plus uniform-ball arm offsets, conditioned on command bounds.

    This proposed adaptation fixes the seventh (gripper) command. Rejection,
    rather than clipping, keeps the six-dimensional target ball intact. This
    checks command coordinates, not physical safety or task feasibility.
    """
    if len(action) != 7 or not all(math.isfinite(x) and -1 <= x <= 1 for x in action):
        raise ValueError("seven finite native command coordinates required")
    if not math.isfinite(radius) or radius < 0 or type(count) is not int or not 1 <= count <= 8:
        raise ValueError("nonnegative radius and bounded target count required")
    anchor = [float(x) for x in action]
    if radius == 0 or count == 1:
        return [anchor]
    digest = hashlib.sha256(f"{seed}:{row_key}".encode()).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    targets = [anchor]
    seen = {tuple(anchor)}
    for _ in range(128 * (count - 1)):
        direction = [rng.gauss(0, 1) for _ in range(6)]
        norm = math.sqrt(sum(x * x for x in direction))
        if norm == 0:
            continue
        length = radius * rng.random() ** (1 / 6)
        candidate = [anchor[i] + length * direction[i] / norm for i in range(6)] + [anchor[6]]
        if any(x < -1 or x > 1 for x in candidate[:6]) or tuple(candidate) in seen:
            continue
        targets.append(candidate)
        seen.add(tuple(candidate))
        if len(targets) == count:
            return targets
    raise ValueError("bounded target sampler exhausted; no fallback or clipping")
