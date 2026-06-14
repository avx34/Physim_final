"""Generate experiments_results.csv with all experiment data."""
import numpy as np
import os

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Inversive')

experiments = {
    ('Baseline_3D', 'FD'): os.path.join(base_dir, 'data', 'training_log.npz'),
    ('Camera_2D',  'FD'): os.path.join(base_dir, 'data_camera', 'training_log.npz'),
    ('Baseline_3D', 'AD'): os.path.join(base_dir, 'data_ad', 'training_log.npz'),
}

pred_trajs = {
    ('Baseline_3D', 'FD'): os.path.join(base_dir, 'data', 'predicted_trajectory.npz'),
    ('Camera_2D',  'FD'): os.path.join(base_dir, 'data_camera', 'predicted_trajectory.npz'),
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments_results.csv')

with open(out_path, 'w', newline='') as f:
    # Helper to write CSV rows
    def W(row):
        f.write(','.join(str(x) for x in row) + '\n')

    W(['# ================================================================================'])
    W(['# Inverse Physics Learning from Camera Projections - Experiment Results'])
    W(['# Target: E*=400, nu=0.4, Neo-Hookean block-drop, MPM 8192 particles, 64 steps'])
    W(['# ================================================================================'])
    W([])

    # ==== SECTION 1: PER-EPOCH TRAINING DATA ====
    W(['## SECTION 1: Per-Epoch Training Data'])
    W(['task', 'method', 'epoch', 'loss', 'E_pred', 'E_abs_error',
       'dL_dE_est', 'epoch_time_s', 'E_true', 'nu_true', 'N_particles', 'N_steps'])

    for (task, method), path in experiments.items():
        d = np.load(path)
        E_true = float(d['E_true'])
        nu_true = float(d['nu_true'])
        n_part = 512 if method == 'AD' else 8192
        n_steps = 8 if method == 'AD' else 64

        for i in range(len(d['epoch'])):
            ep = int(d['epoch'][i])
            loss = float(d['loss'][i])
            E_pred = float(d['E_pred'][i])
            E_err = float(d['E_abs_error'][i])
            if i > 0:
                dL = float(abs(d['loss'][i] - d['loss'][i-1]))
            elif len(d['loss']) > 1:
                dL = float(abs(d['loss'][1] - d['loss'][0]))
            else:
                dL = 0.0
            t = float(d['epoch_time'][i])
            W([task, method, ep,
               '{:.6e}'.format(loss),
               '{:.6f}'.format(E_pred),
               '{:.6f}'.format(E_err),
               '{:.6e}'.format(dL),
               '{:.3f}'.format(t),
               '{:.1f}'.format(E_true),
               '{:.3f}'.format(nu_true),
               n_part, n_steps])

    W([])

    # ==== SECTION 2: CONVERGENCE MILESTONES ====
    W(['## SECTION 2: Convergence Milestones (epochs to reach |E-E*| threshold)'])
    W(['task', 'method', 'threshold', 'epoch_reached', 'E_pred_at_threshold', 'loss_at_threshold'])

    thresholds = [20, 15, 10, 5, 2, 1, 0.5, 0.1]
    for (task, method), path in experiments.items():
        d = np.load(path)
        for thresh in thresholds:
            idx = np.where(d['E_abs_error'] < thresh)[0]
            if len(idx) > 0:
                i = idx[0]
                W([task, method, thresh, int(d['epoch'][i]),
                   '{:.6f}'.format(float(d['E_pred'][i])),
                   '{:.6e}'.format(float(d['loss'][i]))])
            else:
                W([task, method, thresh, 'NOT_REACHED', '', ''])

    W([])

    # ==== SECTION 3: FORWARD SIMULATION MSE ====
    W(['## SECTION 3: Forward Simulation MSE (predicted rollout vs target)'])
    W(['task', 'method', 'E_pred_infer', 'mse_channel_1', 'mse_val_1',
       'mse_channel_2', 'mse_val_2', 'mse_channel_3', 'mse_val_3'])

    for (task, method), path in pred_trajs.items():
        if os.path.exists(path):
            d = np.load(path)
            E_pred = float(d['E_pred'])
            if 'mse_h' in d:
                W([task, method, '{:.6f}'.format(E_pred),
                   'mse_h', '{:.4e}'.format(float(d['mse_h'])),
                   'mse_S(3x3_cov)', '{:.4e}'.format(float(d['mse_s'])),
                   'mse_F_mean', '{:.4e}'.format(float(d['mse_F']))])
            elif 'mse_proj_mean' in d:
                W([task, method, '{:.6f}'.format(E_pred),
                   'mse_proj_mean(2D)', '{:.4e}'.format(float(d['mse_proj_mean'])),
                   'mse_proj_cov(2x2)', '{:.4e}'.format(float(d['mse_proj_cov'])),
                   'mse_depth', '{:.4e}'.format(float(d['mse_depth']))])

    W(['Baseline_3D', 'AD', 'N/A', 'NaN gradients', '', '', '', '', ''])
    W([])

    # ==== SECTION 4: FINAL SUMMARY ====
    W(['## SECTION 4: Final Summary'])
    W(['task', 'method', 'total_epochs', 'initial_loss', 'final_loss',
       'loss_reduction_x', 'initial_E_pred', 'final_E_pred',
       'initial_abs_err', 'final_abs_err', 'best_abs_err', 'best_epoch',
       'converged_E_err_lt_1', 'avg_time_per_epoch_s', 'total_time_s',
       'inference_E_pred', 'inference_MSE_main_channel'])

    for (task, method), path in experiments.items():
        d = np.load(path)
        best_idx = np.argmin(d['E_abs_error'])

        infer_E = ''
        infer_MSE = ''
        if (task, method) in pred_trajs and os.path.exists(pred_trajs[(task, method)]):
            pd = np.load(pred_trajs[(task, method)])
            infer_E = '{:.6f}'.format(float(pd['E_pred']))
            if 'mse_h' in pd:
                infer_MSE = 'mse_h={:.4e}'.format(float(pd['mse_h']))
            elif 'mse_proj_mean' in pd:
                infer_MSE = 'mse_proj_mean={:.4e}'.format(float(pd['mse_proj_mean']))

        W([task, method,
           len(d['epoch']),
           '{:.6e}'.format(float(d['loss'][0])),
           '{:.6e}'.format(float(d['loss'][-1])),
           '{:.1f}'.format(float(d['loss'][0] / max(d['loss'][-1], 1e-30))),
           '{:.3f}'.format(float(d['E_pred'][0])),
           '{:.6f}'.format(float(d['E_pred'][-1])),
           '{:.4f}'.format(float(d['E_abs_error'][0])),
           '{:.6f}'.format(float(d['E_abs_error'][-1])),
           '{:.6f}'.format(float(d['E_abs_error'][best_idx])),
           int(d['epoch'][best_idx]),
           'YES' if d['E_abs_error'][-1] < 1.0 else 'NO',
           '{:.3f}'.format(float(d['epoch_time'][1:].mean())),
           '{:.1f}'.format(float(d['epoch_time'].sum())),
           infer_E, infer_MSE])

print('CSV written to:', out_path)
print('Rows:', sum(1 for _ in open(out_path)))
