"""
Generate figures and analysis from experiment_results_formatted.csv
for the extended paper: dual-parameter learning, noisy targets, baseline search.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report', 'experiment_results_formatted.csv')
OUT = os.path.dirname(os.path.abspath(__file__))

# Parse CSV manually by section boundaries
with open(CSV, 'r') as f:
    raw = list(csv.reader(f))

# Find section boundaries
sections = defaultdict(list)
current_sec = None
for r in raw:
    if not r or not r[0]:
        continue
    t = r[0]
    if t.startswith('## SECTION 1'):
        current_sec = 1; continue
    elif t.startswith('## SECTION 2'):
        current_sec = 2; continue
    elif t.startswith('## SECTION 3'):
        current_sec = 3; continue
    elif t.startswith('## SECTION 4'):
        current_sec = 4; continue
    if t.startswith('#') or 'task' in t.lower():
        continue
    if current_sec:
        sections[current_sec].append(r)

# ── Section 1 ──
sec1_epochs = defaultdict(list)      # (task,method) -> rows
sec1_iterations = defaultdict(list)
sec1_evaluations = defaultdict(list)

for r in sections[1]:
    task, method, row_type = r[0], r[1], r[2]
    if row_type == 'epoch':
        sec1_epochs[(task, method)].append(r)
    elif row_type == 'iteration':
        sec1_iterations[(task, method)].append(r)
    elif row_type == 'evaluation':
        sec1_evaluations[(task, method)].append(r)

# ── Section 2 ──
sec2 = sections[2]  # list of milestone rows

# ── Section 3 ──
sec3 = sections[3]  # already filtered by the header check above

# ── Section 4 ──
sec4 = sections[4]  # list of summary rows

print('Section 1: {} epoch groups, {} iter groups, {} eval groups'.format(
    len(sec1_epochs), len(sec1_iterations), len(sec1_evaluations)))
print('Section 2: {} milestones'.format(len(sec2)))
print('Section 3: {} inference rows'.format(len(sec3)))
print('Section 4: {} summary rows'.format(len(sec4)))

# ═══════════════════════════════════════════════════════
# FIGURE 1: NN+FD Convergence — 3 tasks, 3 columns
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 3, figsize=(22, 16))
fig.suptitle('NN+FD Training Convergence: Three Tasks  (E* = 450, nu* = 0.35)', fontsize=15, fontweight='bold')

task_info = [
    ('DualParam_External_Noisy', 'DualParam + Ext Obs + Noisy', '#ef4444'),
    ('DualParam_Clean',          'DualParam + Full Obs (Clean)', '#2563eb'),
    ('SingleE_Clean',             'Single-E + Full Obs (Clean)', '#059669'),
]

for row_idx, (task, label, color) in enumerate(task_info):
    rows = sec1_epochs.get((task, 'NN+FD'), [])
    if not rows: continue
    ep = np.array([int(r[3]) for r in rows])
    loss = np.array([float(r[3]) for r in rows])
    E_pred = np.array([float(r[4]) for r in rows])
    nu_pred = np.array([float(r[5]) for r in rows])
    E_err = np.array([float(r[6]) for r in rows])
    nu_err = np.array([float(r[7]) for r in rows])

    # (1) Loss
    ax = axes[row_idx, 0]
    ax.semilogy(ep, loss, 'o-', color=color, ms=2, lw=1)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
    ax.set_title('{} — Loss'.format(label[:50]), fontsize=10)
    ax.grid(True, alpha=0.3)

    # (2) E_pred + |E-E*|
    ax = axes[row_idx, 1]
    ax.axhline(y=450, color='gray', ls='--', lw=1, alpha=0.5)
    ax.plot(ep, E_pred, 'o-', color=color, ms=2, lw=1, label='E_pred')
    ax.plot(ep, E_err, 's-', color='#f59e0b', ms=1.5, lw=0.7, alpha=0.6, label='|E-E*|')
    ax.set_xlabel('Epoch'); ax.set_ylabel('E / |E-E*|')
    ax.set_title('{} — E Prediction'.format(label[:50]), fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (3) nu_pred + |nu-nu*|
    ax = axes[row_idx, 2]
    ax.axhline(y=0.35, color='gray', ls='--', lw=1, alpha=0.5)
    ax.plot(ep, nu_pred, 'o-', color=color, ms=2, lw=1, label='nu_pred')
    ax.plot(ep, nu_err, 's-', color='#f59e0b', ms=1.5, lw=0.7, alpha=0.6, label='|nu-nu*|')
    ax.set_xlabel('Epoch'); ax.set_ylabel('nu / |nu-nu*|')
    ax.set_title('{} — nu Prediction'.format(label[:50]), fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_dual_param_convergence.png'), dpi=200, bbox_inches='tight', facecolor='white')
print('Saved fig_dual_param_convergence.png')


# ═══════════════════════════════════════════════════════
# FIGURE 2: NN+FD vs Baseline Search comparison
# ═══════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))
fig2.suptitle('NN+FD vs Derivative-Free Grid Search Baseline', fontsize=14, fontweight='bold')

compare = [
    ('DualParam_External_Noisy', 'Dual+Ext+Noisy', '#ef4444', 'D'),
    ('DualParam_Clean',          'Dual+FullObs',   '#2563eb', 's'),
    ('SingleE_Clean',             'SingleE',        '#059669', 'o'),
]

# (a) E error comparison
ax = axes2[0, 0]
for task, label, color, marker in compare:
    # NN+FD
    rows = sec1_epochs.get((task, 'NN+FD'), [])
    if rows:
        ep = np.array([int(r[3]) for r in rows])
        err = np.array([float(r[6]) for r in rows])
        ax.plot(ep, err, '-', color=color, lw=1.5, label='NN+FD {}'.format(label))

    # Baseline Search — best-so-far from iterations
    iters = sec1_iterations.get((task, 'Baseline_Search'), [])
    if iters:
        it_err = np.array([abs(float(r[4])-450) for r in iters])
        ax.plot(range(len(iters)), it_err, '--', color=color, lw=1.2, marker=marker, ms=6,
                markerfacecolor='white', label='Baseline {}'.format(label))

ax.set_xlabel('Step (epoch or iteration)'); ax.set_ylabel('|E - E*|')
ax.set_title('(a) E Prediction Error')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# (b) nu error comparison
ax = axes2[0, 1]
for task, label, color, marker in compare:
    rows = sec1_epochs.get((task, 'NN+FD'), [])
    if rows:
        ep = np.array([int(r[3]) for r in rows])
        nu_err = np.array([float(r[7]) for r in rows])
        ax.plot(ep, nu_err, '-', color=color, lw=1.5, label='NN+FD {}'.format(label))

    iters = sec1_iterations.get((task, 'Baseline_Search'), [])
    if iters:
        it_nu_err = np.array([abs(float(r[5])-0.35) for r in iters])
        ax.plot(range(len(iters)), it_nu_err, '--', color=color, lw=1.2, marker=marker, ms=6,
                markerfacecolor='white', label='Baseline {}'.format(label))

ax.set_xlabel('Step (epoch or iteration)'); ax.set_ylabel('|nu - nu*|')
ax.set_title('(b) nu Prediction Error')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# (c) Loss comparison across NN+FD tasks
ax = axes2[1, 0]
for task, label, color, marker in compare:
    rows = sec1_epochs.get((task, 'NN+FD'), [])
    if rows:
        ep = np.array([int(r[3]) for r in rows])
        loss = np.array([float(r[3]) for r in rows])
        ax.semilogy(ep, loss, '-', color=color, lw=1.5, label=label)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
ax.set_title('(c) NN+FD Training Loss by Task')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (d) Baseline Search: E-value vs Loss (q=4 contour)
ax = axes2[1, 1]
for task, label, color, marker in compare:
    evals = sec1_evaluations.get((task, 'Baseline_Search'), [])
    if evals:
        Es = np.array([float(r[4]) for r in evals])
        Ls = np.array([float(r[3]) for r in evals])
        valid = (Ls < 1e10) & (Es > 0) & (Es < 900)
        order = np.argsort(Es[valid])
        ax.plot(Es[valid][order], Ls[valid][order], '-', color=color, lw=1, alpha=0.8, label=label)
        # highlight minimum
        best_idx = np.argmin(Ls[valid])
        ax.plot(Es[valid][best_idx], Ls[valid][best_idx], marker, color=color, ms=10,
                markerfacecolor='white')

ax.axvline(x=450, color='gray', ls='--', lw=1.5, alpha=0.5, label='E*=450')
ax.set_xlabel('E'); ax.set_ylabel('Loss')
ax.set_title('(d) Baseline Grid Search: E vs Loss Landscape')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
fig2.savefig(os.path.join(OUT, 'fig_nnfd_vs_baseline.png'), dpi=200, bbox_inches='tight', facecolor='white')
print('Saved fig_nnfd_vs_baseline.png')


# ═══════════════════════════════════════════════════════
# FIGURE 3: Forward simulation MSE bar chart
# ═══════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(16, 6))
fig3.suptitle('Forward Simulation MSE: Predicted vs Target Rollout', fontsize=14, fontweight='bold')

# Normal display (not split)
labels = []
mse_h_vals = []
mse_F_vals = []
colors_bar = []
for r in sec3:
    task, method = r[0], r[1]
    E_pred = float(r[2])
    mse_h = float(r[5])
    mse_F = float(r[9])
    short = task.replace('DualParam_External_Noisy','DP+Noisy').replace('DualParam_Clean','DP+Clean').replace('SingleE_Clean','SE+Clean')
    labels.append('{}\n{}'.format(short, method))
    mse_h_vals.append(mse_h)
    mse_F_vals.append(mse_F)
    colors_bar.append('#ef4444' if 'NN+FD' in method else '#2563eb')

x = np.arange(len(labels))
w = 0.35
bars1 = ax3.bar(x - w/2, mse_h_vals, w, label='mse(h)', color='#ef4444', alpha=0.85, edgecolor='white')
bars2 = ax3.bar(x + w/2, mse_F_vals, w, label='mse(F_mean)', color='#f59e0b', alpha=0.85, edgecolor='white')
ax3.set_yscale('log')
ax3.set_ylabel('MSE (log scale)')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=8)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars1, mse_h_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, val*2.5, '{:.1e}'.format(val),
             ha='center', va='bottom', fontsize=7, rotation=90)
for bar, val in zip(bars2, mse_F_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, val*2.5, '{:.1e}'.format(val),
             ha='center', va='bottom', fontsize=7, rotation=90)

plt.tight_layout()
fig3.savefig(os.path.join(OUT, 'fig_forward_mse.png'), dpi=200, bbox_inches='tight', facecolor='white')
print('Saved fig_forward_mse.png')


# ═══════════════════════════════════════════════════════
# Print all data for LaTeX tables
# ═══════════════════════════════════════════════════════

print()
print('='*80)
print('LATEX DATA')
print('='*80)

print()
print('--- SECTION 1: Per-task NN+FD summary ---')
for task, label, color in task_info:
    rows = sec1_epochs.get((task, 'NN+FD'), [])
    if rows:
        r0, rf = rows[0], rows[-1]
        print('{}: {} epochs, E {:.1f}->{:.1f} (|err| {:.1f}->{:.3f}), nu {:.3f}->{:.3f} (|err| {:.3f}->{:.4f}), loss {:.2e}->{:.2e}'.format(
            label, len(rows),
            float(r0[4]), float(rf[4]), float(r0[6]), float(rf[6]),
            float(r0[5]), float(rf[5]), float(r0[7]), float(rf[7]),
            float(r0[3]), float(rf[3])))

print()
print('--- SECTION 2: Convergence Milestones ---')
for r in sec2:
    reached = r[4]
    val = r[5] if len(r) > 5 else ''
    print('{} | {} | param={} | thresh={} | epoch={} | val={}'.format(
        r[0], r[1], r[2], r[3], reached, val))

print()
print('--- SECTION 3: Forward Simulation MSE ---')
for r in sec3:
    print('{} | {} | E_pred={} | {}={} | {}={} | {}={}'.format(
        r[0], r[1], r[2], r[4], r[5], r[8] if len(r)>8 else '?', r[9] if len(r)>9 else '?',
        r[10] if len(r)>10 else '?', r[11] if len(r)>11 else '?'))

print()
print('--- SECTION 4: Final Summary ---')
for r in sec4:
    print('{} | {} | epochs={} | loss {:.2e}->{:.2e} | E {:.1f}->{:.1f} (err {:.1f}->{:.3f}) | nu {:.3f}->{:.3f} (err {:.3f}->{:.4f}) | conv_E={}'.format(
        r[0], r[1], r[3],
        float(r[4]), float(r[5]),
        float(r[7]), float(r[8]),
        float(r[10]), float(r[11]),
        float(r[9]), float(r[13]),
        float(r[12]), float(r[15]) if len(r)>15 else 0,
        r[16] if len(r)>16 else '?'))

print()
print('Done!')
