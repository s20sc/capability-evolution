# T6 Lifecycle Fault Injection - Summary

Task: `T6_TwoArmPegInHole`  Generated: 2026-07-24T02:38:04  Recovery margin: 5% of baseline reward (reward metric) / 5.0 pp (success metric)

| Arm | Peak Reward | Final Reward | Peak Success | Inj. Rejection | False Rejection | Dip Depth (% of baseline) | Dip Duration | Iters-to-Recovery | Rollbacks (false) |
|---|---|---|---|---|---|---|---|---|---|
| gating_on | 309.1 ± 35.9 (n=4) | 296.4 ± 41.3 (n=4) | 8.3% ± 8.8 (n=4) | 100.0% ± 0.0 (n=4) | 22.5% ± 26.3 (n=4) | 0.7% ± 3.9 (n=12) | 0.0 ± 0.0 (n=12) | 0.0 ± 0.0 (n=12) | n/a |
| gating_off | 323.5 ± 8.2 (n=4) | 287.4 ± 27.9 (n=4) | 10.0% ± 4.7 (n=4) | n/a | n/a | 14.5% ± 8.1 (n=12) | 1.1 ± 0.7 (n=12) | 0.9 ± 0.3 (n=11) [1 censored] | n/a |
| gating_off_rollback | 312.1 ± 26.5 (n=4) | 292.3 ± 44.8 (n=4) | 6.7% ± 9.0 (n=4) | n/a | n/a | 17.8% ± 6.8 (n=12) | 0.9 ± 0.3 (n=12) | 0.9 ± 0.3 (n=12) | 2.2 ± 1.5 (n=4) / 0.0 ± 0.0 (n=4) false |
