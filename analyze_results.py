"""
Generate analysis plots for the capability evolution experiments.
Produces 4 figures + summary tables matching the paper's structure.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

RESULTS_DIR = 'results'
FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)

METHOD_COLORS = {
    'ours': '#1f77b4',
    'agent_modification': '#d62728',
    'static_ecm': '#7f7f7f',
    'spirl': '#bcbd22',
    'skimo': '#17becf',
}
METHOD_LABELS = {
    'ours': 'Ours (Capability Evolution)',
    'agent_modification': 'Agent Modification',
    'static_ecm': 'Static ECM',
    'spirl': 'SPiRL',
    'skimo': 'SkiMo',
}
TASKS = ['T1_Pick', 'T2_Place', 'T3_Stack', 'T4_NutAssembly',
         'T5_PickPlaceMulti', 'T6_TwoArmPegInHole']


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    return json.load(open(path)) if os.path.exists(path) else None


# ============================================================
# Figure 1: EXP1 evolution curves (H1)
# Shows reward vs iteration for each task
# ============================================================
def fig_exp1():
    d = load('exp1_evolution_over_time.json')
    if not d:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, task in zip(axes, TASKS):
        if task not in d:
            continue
        metrics = d[task]['metrics']
        iters = [m['iteration'] for m in metrics]
        rewards = [m['avg_reward'] for m in metrics]
        successes = [m['success_rate'] for m in metrics]
        ax2 = ax.twinx()
        ax.plot(iters, rewards, 'b-o', markersize=4, label='Avg Reward')
        ax2.plot(iters, successes, 'r--s', markersize=4, label='Success %')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Avg Reward', color='b')
        ax2.set_ylabel('Success Rate (%)', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(-5, 105)
        ax.set_title(task)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Exp1 (H1): Capability Evolution Over 20 Iterations',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/exp1_evolution.png')
    plt.close()
    print(f'Saved: {FIG_DIR}/exp1_evolution.png')


# ============================================================
# Figure 2: EXP2 method comparison (H2)
# Reward trajectories per task, one subplot per task
# ============================================================
def fig_exp2_curves():
    d = load('exp2_method_comparison.json')
    if not d:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, task in zip(axes, TASKS):
        if task not in d:
            continue
        for method in METHOD_COLORS:
            metrics = d[task].get(method, {}).get('metrics', [])
            if not metrics:
                continue
            iters = [m['iteration'] for m in metrics]
            rewards = [m['avg_reward'] for m in metrics]
            ax.plot(iters, rewards, marker='o', markersize=3,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Avg Reward')
        ax.set_title(task)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Exp2 (H2): Reward Trajectories by Method',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/exp2_reward_curves.png')
    plt.close()
    print(f'Saved: {FIG_DIR}/exp2_reward_curves.png')


def fig_exp2_success_bars():
    """Final-iteration & peak success rate comparison."""
    d = load('exp2_method_comparison.json')
    if not d:
        return
    methods = list(METHOD_COLORS.keys())
    # Peak success per (task, method)
    peak = np.zeros((len(TASKS), len(methods)))
    final = np.zeros((len(TASKS), len(methods)))
    for i, task in enumerate(TASKS):
        for j, method in enumerate(methods):
            metrics = d.get(task, {}).get(method, {}).get('metrics', [])
            if metrics:
                successes = [m['success_rate'] for m in metrics]
                peak[i, j] = max(successes)
                final[i, j] = successes[-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(TASKS))
    width = 0.15

    for metric_idx, (matrix, label) in enumerate([(peak, 'Peak Success'),
                                                    (final, 'Final Success')]):
        ax = axes[metric_idx]
        for j, method in enumerate(methods):
            offset = (j - 2) * width
            ax.bar(x + offset, matrix[:, j], width,
                   label=METHOD_LABELS[method],
                   color=METHOD_COLORS[method])
        ax.set_xticks(x)
        ax.set_xticklabels(TASKS, rotation=20, ha='right')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'{label} Rate by Task')
        ax.grid(True, alpha=0.3, axis='y')
        if metric_idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/exp2_success_comparison.png')
    plt.close()
    print(f'Saved: {FIG_DIR}/exp2_success_comparison.png')


def fig_exp2_reward_bars():
    """Peak reward comparison."""
    d = load('exp2_method_comparison.json')
    if not d:
        return
    methods = list(METHOD_COLORS.keys())
    peak = np.zeros((len(TASKS), len(methods)))
    for i, task in enumerate(TASKS):
        for j, method in enumerate(methods):
            metrics = d.get(task, {}).get(method, {}).get('metrics', [])
            if metrics:
                peak[i, j] = max(m['avg_reward'] for m in metrics)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(TASKS))
    width = 0.15
    for j, method in enumerate(methods):
        offset = (j - 2) * width
        ax.bar(x + offset, peak[:, j], width,
               label=METHOD_LABELS[method],
               color=METHOD_COLORS[method])
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, rotation=20, ha='right')
    ax.set_ylabel('Peak Avg Reward')
    ax.set_title('Exp2: Peak Reward by Task and Method')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_yscale('symlog', linthresh=1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/exp2_reward_comparison.png')
    plt.close()
    print(f'Saved: {FIG_DIR}/exp2_reward_comparison.png')


# ============================================================
# Figure 3: EXP3 safety (H3)
# ============================================================
def fig_exp3():
    d = load('exp3_safety.json')
    if not d:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    tasks = list(d.keys())
    blocking = [d[t]['with_governance']['blocking_rate'] for t in tasks]
    unsafe_on = [d[t]['with_governance'].get('unsafe_exec_pct', 0) for t in tasks]
    unsafe_off = [d[t]['without_governance'].get('unsafe_exec_pct', 0) for t in tasks]
    overhead = [d[t]['with_governance'].get('overhead_ms', 0) for t in tasks]

    x = np.arange(len(tasks))
    axes[0].bar(x, blocking, color='#1f77b4')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks, rotation=20, ha='right')
    axes[0].set_ylabel('Blocking Rate (%)')
    axes[0].set_title('Governance Blocking Rate')
    axes[0].axhline(100, color='red', linestyle='--', alpha=0.5, label='Target: 100%')
    axes[0].legend()
    axes[0].set_ylim(0, 110)

    width = 0.35
    axes[1].bar(x - width/2, unsafe_off, width, label='Without governance', color='#d62728')
    axes[1].bar(x + width/2, unsafe_on, width, label='With governance', color='#2ca02c')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, rotation=20, ha='right')
    axes[1].set_ylabel('Unsafe Episodes (%)')
    axes[1].set_title('Unsafe Execution Rate')
    axes[1].legend()

    axes[2].bar(x, overhead, color='#ff7f0e')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tasks, rotation=20, ha='right')
    axes[2].set_ylabel('Overhead per action (ms)')
    axes[2].set_title('Governance Overhead')
    axes[2].axhline(2.3, color='gray', linestyle='--', alpha=0.5,
                    label='Paper claim: 2.3ms')
    axes[2].legend()

    fig.suptitle('Exp3 (H3): Runtime Governance Safety Evaluation',
                 fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/exp3_safety.png')
    plt.close()
    print(f'Saved: {FIG_DIR}/exp3_safety.png')


# ============================================================
# Figure 4: Ablation (learning modalities)
# ============================================================
def fig_ablation():
    d = load('ablation_modalities.json')
    if not d:
        return
    names = ['rl_only', 'il_only', 'llm_only', 'rl_il', 'rl_llm', 'il_llm', 'all_three']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward curves
    for name in names:
        v = d.get(name)
        if v:
            m = v['metrics']
            iters = [x['iteration'] for x in m]
            rewards = [x['avg_reward'] for x in m]
            axes[0].plot(iters, rewards, marker='o', markersize=3, label=name)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Avg Reward')
    axes[0].set_title('Ablation: Reward by Modality Combination')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Final reward + max success bars
    final_rewards = []
    peak_success = []
    for name in names:
        v = d.get(name)
        if v:
            m = v['metrics']
            final_rewards.append(m[-1]['avg_reward'])
            peak_success.append(max(x['success_rate'] for x in m))
        else:
            final_rewards.append(0)
            peak_success.append(0)

    x = np.arange(len(names))
    width = 0.35
    ax = axes[1]
    ax2 = ax.twinx()
    ax.bar(x - width/2, final_rewards, width, label='Final Reward', color='#1f77b4')
    ax2.bar(x + width/2, peak_success, width, label='Peak Success (%)', color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Final Reward', color='#1f77b4')
    ax2.set_ylabel('Peak Success (%)', color='#d62728')
    ax.set_title('Ablation: Final Reward vs Peak Success')

    fig.suptitle('Ablation: Learning Modalities (RL / IL / LLM)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/ablation.png')
    plt.close()
    print(f'Saved: {FIG_DIR}/ablation.png')


# ============================================================
# Summary tables
# ============================================================
def print_summary_tables():
    print('\n' + '=' * 70)
    print('SUMMARY TABLES')
    print('=' * 70)

    # exp1
    d = load('exp1_evolution_over_time.json')
    if d:
        print('\n--- EXP1 (H1): Evolution Over 20 Iterations ---')
        print(f'{"Task":25s} {"first_reward":>12s} {"final_reward":>12s} '
              f'{"peak_success":>12s} {"final_success":>13s}')
        for t in TASKS:
            if t not in d:
                continue
            m = d[t]['metrics']
            first_r = m[0]['avg_reward']
            final_r = m[-1]['avg_reward']
            peak_s = max(x['success_rate'] for x in m)
            final_s = m[-1]['success_rate']
            print(f'{t:25s} {first_r:>12.2f} {final_r:>12.2f} '
                  f'{peak_s:>11.1f}% {final_s:>12.1f}%')

    # exp2
    d = load('exp2_method_comparison.json')
    if d:
        print('\n--- EXP2 (H2): Method Comparison (peak reward, peak success) ---')
        methods = list(METHOD_COLORS.keys())
        print(f'{"Task":25s}', end='')
        for m in methods:
            print(f' {m[:12]:>14s}', end='')
        print()
        for t in TASKS:
            if t not in d:
                continue
            print(f'{t:25s}', end='')
            for method in methods:
                metrics = d[t].get(method, {}).get('metrics', [])
                if metrics:
                    peak_r = max(x['avg_reward'] for x in metrics)
                    peak_s = max(x['success_rate'] for x in metrics)
                    print(f' r={peak_r:5.1f} s={peak_s:4.1f}%', end='')
                else:
                    print(f' {"—":>14s}', end='')
            print()

    # exp3
    d = load('exp3_safety.json')
    if d:
        print('\n--- EXP3 (H3): Safety Evaluation ---')
        print(f'{"Task":25s} {"block%":>8s} {"unsafe_off%":>12s} '
              f'{"unsafe_on%":>11s} {"overhead_ms":>12s}')
        for t, v in d.items():
            on = v['with_governance']
            off = v['without_governance']
            print(f'{t:25s} {on["blocking_rate"]:>7.1f}% '
                  f'{off.get("unsafe_exec_pct",0):>11.1f}% '
                  f'{on.get("unsafe_exec_pct",0):>10.1f}% '
                  f'{on.get("overhead_ms",0):>11.4f}')

    # ablation
    d = load('ablation_modalities.json')
    if d:
        print('\n--- Ablation: Learning Modalities ---')
        print(f'{"Config":15s} {"final_reward":>12s} {"peak_success":>12s}')
        names = ['rl_only', 'il_only', 'llm_only', 'rl_il', 'rl_llm', 'il_llm', 'all_three']
        for name in names:
            v = d.get(name)
            if v:
                m = v['metrics']
                print(f'{name:15s} {m[-1]["avg_reward"]:>12.2f} '
                      f'{max(x["success_rate"] for x in m):>11.1f}%')


if __name__ == '__main__':
    fig_exp1()
    fig_exp2_curves()
    fig_exp2_success_bars()
    fig_exp2_reward_bars()
    fig_exp3()
    fig_ablation()
    print_summary_tables()
