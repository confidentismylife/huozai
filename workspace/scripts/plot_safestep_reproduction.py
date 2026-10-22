"""Export standalone figures; all titles explicitly identify synthetic FER."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/safestep_reproduction_20261014'
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
plan = json.loads((OUT / 'frozen_manifest.json').read_text(encoding='utf-8'))
summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
labels = plan['labels']
coords = [tuple(c) for c in plan['coordinates']]


def trace_path(trace):
    if not trace: return [], []
    points = [trace[0]['from_state']] + [r['to_state'] for r in trace]
    return [coords[s][0] for s in points], [coords[s][1] for s in points]


fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 1]})
for i, case in enumerate(['case1_early', 'case2_late']):
    ax = axes[i, 0]
    for label, (x, y) in zip(labels, coords):
        fill = '#e5f5e0' if label.startswith('Exit') else '#fdd0a2' if label == 'Room_A' else '#eef1f4'
        ax.add_patch(Rectangle((x-.45, y-.45), .9, .9, facecolor=fill, edgecolor='#59636d', linewidth=.5))
        if label in ['Room_A', 'Room_B', 'Exit_A', 'Exit_B']:
            ax.text(x, y + (.7 if y > 0 else -.8), label.replace('_', ' '), ha='center', fontsize=8)
        elif label in ['R1', 'R14', 'R18', 'R20']:
            ax.text(x, y, label, ha='center', va='center', fontsize=6)
    dqn_trace = json.loads((OUT / 'runs' / f'{case}_seed41' / 'RoomB_evaluation_trace.json').read_text(encoding='utf-8'))
    da = json.loads((OUT / f'{case}_paper_stationary_FED_threshold_Dijkstra_trace.json').read_text(encoding='utf-8'))
    shortest = json.loads((OUT / f'{case}_unblocked_shortest_Dijkstra_trace.json').read_text(encoding='utf-8'))
    for trace, color, name, style in [(dqn_trace, '#1764ab', 'DQN final (seed 41)', '-'),
        (shortest['trace'], '#c9453c', 'Unblocked shortest', '--')]:
        xs, ys = trace_path(trace)
        ax.plot(xs, ys, style, color=color, marker='o', markersize=3, linewidth=1.6, label=name)
    if da['outcome']['no_path']:
        ax.text(.5, .9, 'Paper DA: no path', transform=ax.transAxes, ha='center', fontsize=9,
            bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': '#c9453c'})
    ax.set_xlim(-1, 20)
    ax.set_ylim(-2.1, 2.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{'Early' if i == 0 else 'Late'} start: {plan['case_times'][case]} s", fontsize=11)
    if i == 0: ax.legend(loc='upper center', bbox_to_anchor=(.5, -.05), fontsize=8, ncol=2)
    ax = axes[i, 1]
    for trace, name, color in [(dqn_trace, 'DQN final (seed 41)', '#1764ab'),
        (shortest['trace'], 'Unblocked shortest', '#c9453c')]:
        dt = 2 / .3
        elapsed = [0] + [r['step'] * dt for r in trace]
        dose = [0] + [r['cumulative_personal_FED'] for r in trace]
        ax.plot(elapsed, dose, label=name, color=color, linewidth=1.8)
    ax.axhline(.3, color='#777777', linestyle=':', label='Paper reward boundary 0.3')
    ax.set_xlabel('Elapsed evacuation time (s)')
    ax.set_ylabel('Personal cumulative FED')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle('SafeStep method reimplementation: controlled synthetic FER, not original FDS', fontsize=12)
fig.tight_layout()
fig.savefig(FIG / 'routes_and_personal_FED.png', dpi=180)
fig.savefig(FIG / 'routes_and_personal_FED.svg')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
for ax, case in zip(axes, ['case1_early', 'case2_late']):
    for seed in plan['seeds']:
        with (OUT / 'runs' / f'{case}_seed{seed}' / 'training_trace.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        values = np.array([float(r['total_reward']) for r in rows])
        smooth = np.convolve(values, np.ones(100)/100, mode='valid')
        ax.plot(np.arange(100, len(values)+1), smooth, label=f'Seed {seed}', linewidth=1.3)
    ax.set_title(case.replace('_', ' '))
    ax.set_xlabel('Training episode')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
axes[0].set_ylabel('Training return, trailing 100-episode mean')
fig.suptitle('Fixed 2,000-episode budgets; synthetic FER; final policies retained', fontsize=11)
fig.tight_layout()
fig.savefig(FIG / 'training_returns.png', dpi=180)
fig.savefig(FIG / 'training_returns.svg')
plt.close(fig)
print(json.dumps({'figures': ['routes_and_personal_FED.png', 'training_returns.png'], 'synthetic_only': True}))
