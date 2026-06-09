"""Compare finite-difference and Taichi AD gradients for E.

This script diagnoses whether the training gradient matches the real forward
loss landscape. It is especially useful when truncated BPTT (`seg_len`) changes
the apparent optimum.
"""
import argparse
import os
import numpy as np
import taichi as ti

import sim_config as scfg


parser = argparse.ArgumentParser()
parser.add_argument("--E", type=float, default=425.0)
parser.add_argument("--eps", type=float, default=1.0)
parser.add_argument("--seg_len", type=int, default=2)
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
    obs.target_h.from_numpy(data["h"].astype(np.float32)[:cfg.n_steps])
    obs.target_s.from_numpy(data["s"].astype(np.float32)[:cfg.n_steps])
    obs.target_F_mean.from_numpy(data["F_mean"].astype(np.float32)[:cfg.n_steps])
    return float(data["E_true"])


def run_obs_kernels(step):
    obs.zero_mean_acc(step)
    obs.accum_mean(step)
    obs.copy_mean_to_h(step)
    obs.zero_cov_acc(step)
    obs.accum_cov(step)
    obs.zero_F_acc(step)
    obs.accum_F(step)


def zero_all_grads():
    for field in [sim.x, sim.v, sim.C, sim.F, sim.grid_v, sim.grid_m,
                  sim.mu_tmp, sim.lambda_tmp, sim.E_pred,
                  obs.pred_h, obs.pred_s, obs.pred_F_mean,
                  obs.mean_tmp, obs.loss]:
        sim.zero_grad(field)


def forward_loss(E_value):
    sim.init_particles()
    sim.E_pred[None] = float(E_value)
    sim.compute_lame_params()
    obs.loss[None] = 0.0
    for step in range(cfg.n_steps):
        for _ in range(cfg.substeps_per_step):
            sim.differentiable_substep()
        run_obs_kernels(step)
        obs.compute_step_loss(step)
    return float(obs.loss[None])


def finite_difference_grad(E_value, eps):
    loss_plus = forward_loss(E_value + eps)
    loss_minus = forward_loss(E_value - eps)
    return loss_plus, loss_minus, (loss_plus - loss_minus) / (2.0 * eps)


def ad_grad(E_value, seg_len):
    sim.init_particles()
    zero_all_grads()
    total_loss = 0.0
    total_grad = 0.0
    num_segments = (cfg.n_steps + seg_len - 1) // seg_len

    for seg in range(num_segments):
        seg_start = seg * seg_len
        seg_end = min(seg_start + seg_len, cfg.n_steps)
        obs.loss[None] = 0.0
        zero_all_grads()
        sim.E_pred[None] = float(E_value)

        with ti.ad.Tape(loss=obs.loss):
            sim.compute_lame_params()
            for step in range(seg_start, seg_end):
                for _ in range(cfg.substeps_per_step):
                    sim.differentiable_substep()
                run_obs_kernels(step)
                obs.compute_step_loss(step)

        total_loss += float(obs.loss[None])
        total_grad += float(sim.E_pred.grad[None])

    return total_loss, total_grad


def main():
    e_true = load_target_data()
    loss = forward_loss(args.E)
    loss_plus, loss_minus, fd_grad = finite_difference_grad(args.E, args.eps)
    trunc_loss, trunc_grad = ad_grad(args.E, args.seg_len)

    print(f"E_true={e_true:.6g}")
    print(f"E={args.E:.6g}, eps={args.eps:.6g}, seg_len={args.seg_len}")
    print(f"forward_loss(E)={loss:.10g}")
    print(f"forward_loss(E+eps)={loss_plus:.10g}")
    print(f"forward_loss(E-eps)={loss_minus:.10g}")
    print(f"finite_difference_dL_dE={fd_grad:.10g}")
    print(f"truncated_ad_loss={trunc_loss:.10g}")
    print(f"truncated_ad_dL_dE={trunc_grad:.10g}")
    if fd_grad * trunc_grad < 0:
        print("[DIAG] Gradient signs disagree.")
    else:
        print("[DIAG] Gradient signs agree.")


if __name__ == "__main__":
    main()
