"""Control-step diagnostics; these checks do not certify physical safety."""
from __future__ import annotations

import math


def inspect_actuation(ctrl: list, forces: list, limits: dict, tolerance: float) -> dict:
    keys = {"actuator_names", "actuator_ctrllimited", "actuator_ctrlrange",
            "actuator_forcelimited", "actuator_forcerange"}
    if set(limits) != keys or len(ctrl) != 9 or len(forces) != 9:
        raise ValueError("actuator schema")
    if any(len(limits[key]) != 9 for key in keys):
        raise ValueError("actuator field length")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("comparison tolerance")
    reasons = []
    for i in range(9):
        if not math.isfinite(ctrl[i]) or not math.isfinite(forces[i]):
            reasons.append(f"nonfinite_actuator_{i}")
        for value, flags, bounds, label in [
            (ctrl[i], limits["actuator_ctrllimited"], limits["actuator_ctrlrange"], "ctrl"),
            (forces[i], limits["actuator_forcelimited"], limits["actuator_forcerange"], "force")]:
            if flags[i] not in (0, 1) or len(bounds[i]) != 2:
                raise ValueError("actuator limit flag/range")
            lo, hi = bounds[i]
            if not all(math.isfinite(x) for x in (lo, hi)) or lo > hi:
                raise ValueError("actuator range values")
            if flags[i] and not lo - tolerance <= value <= hi + tolerance:
                reasons.append(f"{label}_range_{i}")
    return {"ok": not reasons, "reasons": reasons, "ctrl": list(ctrl), "actuator_force": list(forces)}


def guard_decision(finite: bool, workspace: bool, force_norm: float, actuation: dict) -> dict:
    if not isinstance(finite, bool) or not isinstance(workspace, bool):
        raise ValueError("guard labels")
    numeric = finite and math.isfinite(force_norm)
    return {"stop": not numeric or not workspace or not actuation["ok"],
            "legacy_20N_exceeded": math.isfinite(force_norm) and force_norm > 20.0,
            "legacy_guard_would_stop": not numeric or not workspace or force_norm > 20.0}
