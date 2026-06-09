"""Scan forward loss over candidate Young's modulus values.

This diagnostic bypasses NN training. It sets E directly, runs the differentiable
MPM forward model, and compares the resulting observables with target data.
If the minimum is not near E_true, the issue is in the observation/model setup,
not in the neural optimizer.
"""
import argparse
import os
import numpy as np

import sim_config as scfg


parser = argparse.ArgumentParser()
parser.add_argument("--values", type=str,
                    default="300,325,350,375,400,425,450,475,500",
                    help="comma-separated E values to scan")
parser.add_argument("--random_init", action="store_true",
                    help="use the original stochastic particle initialization")
args = parser.parse_args()

scfg.cfg.deterministic_init = not args.random_init
scfg.cfg.init_taichi()

import mpm_sim as sim
import observables as obs

cfg = scfg.cfg
DATA_DIR = scfg.DATA_DIR


def load_target_data():
    path = os.path.join(DATA_DIR, "target_trajectory.npz")
    data = np.load(path)
    h_np = data["h"].astype(np.float32)[:cfg.n_steps]
    s_np = data["s"].astype(np.float32)[:cfg.n_steps]
    f_np = data["F_mean"].astype(np.float32)[:cfg.n_steps]
    obs.target_h.from_numpy(h_np)
    obs.target_s.from_numpy(s_np)
    obs.target_F_mean.from_numpy(f_np)
    return h_np, s_np, f_np, float(data["E_true"])


def run_obs_kernels(step):
    obs.zero_mean_acc(step)
    obs.accum_mean(step)
    obs.copy_mean_to_h(step)
    obs.zero_cov_acc(step)
    obs.accum_cov(step)
    obs.zero_F_acc(step)
    obs.accum_F(step)


def run_forward(E_value):
    sim.init_particles()
    sim.E_pred[None] = float(E_value)
    sim.compute_lame_params()

    pred_h = np.zeros(cfg.n_steps, dtype=np.float32)
    pred_s = np.zeros((cfg.n_steps, 3, 3), dtype=np.float32)
    pred_f = np.zeros((cfg.n_steps, 3, 3), dtype=np.float32)

    for step in range(cfg.n_steps):
        for _ in range(cfg.substeps_per_step):
            sim.differentiable_substep()
        run_obs_kernels(step)
        pred_h[step] = obs.pred_h[step]
        pred_s[step] = obs.pred_s[step].to_numpy()
        pred_f[step] = obs.pred_F_mean[step].to_numpy()

    return pred_h, pred_s, pred_f


def compute_losses(pred_h, pred_s, pred_f, target_h, target_s, target_f):
    h_mse = float(np.mean((pred_h - target_h) ** 2))
    s_mse = float(np.mean((pred_s - target_s) ** 2))
    f_mse = float(np.mean((pred_f - target_f) ** 2))
    weighted = h_mse
    weighted += 10.0 * np.sum((pred_s - target_s) ** 2) / cfg.n_steps
    weighted += 5.0 * np.sum((pred_f - target_f) ** 2) / cfg.n_steps
    return h_mse, s_mse, f_mse, float(weighted)


def main():
    target_h, target_s, target_f, e_true = load_target_data()
    values = [float(v.strip()) for v in args.values.split(",") if v.strip()]
    rows = []

    print(f"Scanning {len(values)} E values; target E_true={e_true:.3f}")
    print("E,h_mse,s_mse,F_mse,weighted_loss")
    for E_value in values:
        pred_h, pred_s, pred_f = run_forward(E_value)
        row = (E_value, *compute_losses(pred_h, pred_s, pred_f,
                                        target_h, target_s, target_f))
        rows.append(row)
        print(",".join(f"{x:.8g}" for x in row))

    out = np.array(rows, dtype=np.float32)
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "E_loss_scan.npy")
    np.save(out_path, out)
    best = out[np.argmin(out[:, 4])]
    print(f"Saved {out_path}")
    print(f"Best scanned E={best[0]:.3f}, weighted_loss={best[4]:.8g}")


if __name__ == "__main__":
    main()
