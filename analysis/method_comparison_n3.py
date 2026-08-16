#!/usr/bin/env python3
"""Recompute the T6 five-method comparison from the attested cloud seeds.

Every number here comes from per-seed result files produced on the attested
cloud stack (robosuite 1.4.1 / MuJoCo 2.3.2): seeds 7, 123 and 2024 for all
five methods. No stored development-era aggregate is read.

Outputs analysis/method_comparison_n3.json and prints the table values.
"""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEEDS = ["7", "123", "2024"]
METHODS = ["ours", "agent_modification", "static_ecm", "spirl", "skimo"]


def load(method: str, seed: str) -> dict:
    if method in ("ours", "agent_modification"):
        p = ROOT / f"results/t6_multiseed/seed_{seed}/t6_{method}.json"
    else:
        p = ROOT / f"results/multiseed/T6_TwoArmPegInHole/seed_{seed}/{method}.json"
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    out = {"seeds": SEEDS, "methods": {}}
    for m in METHODS:
        peak_succ, peak_rew, drift = [], [], []
        for sd in SEEDS:
            mets = load(m, sd)["metrics"]
            s = max(x["success_rate"] for x in mets)
            peak_succ.append(s * 100 if s <= 1.0 else s)
            peak_rew.append(max(x["avg_reward"] for x in mets))
            drift.append(mets[-1].get("policy_drift", 0.0))
        out["methods"][m] = {
            "peak_success_per_seed": peak_succ,
            "peak_success_mean": st.mean(peak_succ),
            "peak_success_std": st.stdev(peak_succ),
            "peak_reward_per_seed": peak_rew,
            "peak_reward_mean": st.mean(peak_rew),
            "peak_reward_std": st.stdev(peak_rew),
            "final_drift_per_seed": drift,
            "final_drift_mean": st.mean(drift),
        }
    # Guards: the frozen-planner arms must show zero drift; the mutating arms must not.
    assert out["methods"]["ours"]["final_drift_mean"] == 0.0
    assert out["methods"]["static_ecm"]["final_drift_mean"] == 0.0
    assert out["methods"]["spirl"]["final_drift_mean"] == 0.0
    assert out["methods"]["agent_modification"]["final_drift_mean"] > 1.0
    assert out["methods"]["skimo"]["final_drift_mean"] > 1.0
    with open(HERE / "method_comparison_n3.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    for m in METHODS:
        d = out["methods"][m]
        print(f"{m:20s} succ {d['peak_success_mean']:5.1f}±{d['peak_success_std']:4.1f}  "
              f"per-seed {[round(x,1) for x in d['peak_success_per_seed']]}  "
              f"reward {d['peak_reward_mean']:6.1f}±{d['peak_reward_std']:5.1f}  drift {d['final_drift_mean']:.2f}")


if __name__ == "__main__":
    main()
