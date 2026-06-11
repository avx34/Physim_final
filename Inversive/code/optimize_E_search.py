"""Derivative-free inverse baseline for E, nu, or both parameters.

The one-dimensional E case keeps the original golden-section search. Two
parameter recovery uses a coarse-to-fine grid because it is simple, stable, and
easy to compare against NN+FD on this small problem.
"""
import argparse
import os
import time
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--learn_param", choices=("E", "nu", "both"), default="E",
                    help="material parameter(s) to recover")
parser.add_argument("--E_lo", type=float, default=50.0)
parser.add_argument("--E_hi", type=float, default=800.0)
parser.add_argument("--nu_lo", type=float, default=0.05)
parser.add_argument("--nu_hi", type=float, default=0.49)
parser.add_argument("--iters", type=int, default=24,
                    help="golden-section iterations for 1D E search")
parser.add_argument("--grid_size", type=int, default=7,
                    help="samples per axis for grid-based searches")
parser.add_argument("--levels", type=int, default=3,
                    help="coarse-to-fine levels for grid-based searches")
args = parser.parse_args()

import sim_config as scfg

scfg.cfg.init_taichi()

import mpm_sim as sim
import observables as obs

cfg = scfg.cfg
DATA_DIR = scfg.DATA_DIR
target_data_npz = None


def load_target_data():
    global target_data_npz
    path = os.path.join(DATA_DIR, "target_trajectory.npz")
    data = np.load(path)
    target_data_npz = data
    return (
        data["h"].astype(np.float32)[:cfg.n_steps],
        data["s"].astype(np.float32)[:cfg.n_steps],
        data["F_mean"].astype(np.float32)[:cfg.n_steps],
        float(data["E_true"]),
        float(data["nu_true"]),
    )


def run_obs_kernels(step):
    obs.zero_mean_acc(step)
    obs.accum_mean(step)
    obs.copy_mean_to_h(step)
    obs.zero_cov_acc(step)
    obs.accum_cov(step)
    obs.zero_F_acc(step)
    obs.accum_F(step)


def forward_observables(E_value, nu_value):
    sim.init_from_target_data(target_data_npz)
    sim.E_pred[None] = float(E_value)
    sim.nu_pred[None] = float(nu_value)
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


def weighted_loss(pred_h, pred_s, pred_f, target_h, target_s, target_f):
    h_loss = np.mean((pred_h - target_h) ** 2)
    s_loss = np.sum((pred_s - target_s) ** 2) / cfg.n_steps
    f_loss = np.sum((pred_f - target_f) ** 2) / cfg.n_steps
    total = h_loss + 10.0 * s_loss + 5.0 * f_loss
    if not np.isfinite(total):
        return 1e30, 1e30, 1e30, 1e30
    return float(total), float(h_loss), float(s_loss), float(f_loss)


def evaluate(E_value, nu_value, target_h, target_s, target_f):
    pred_h, pred_s, pred_f = forward_observables(E_value, nu_value)
    return weighted_loss(pred_h, pred_s, pred_f, target_h, target_s, target_f)


def save_result(best_E, best_nu, eval_rows, iter_rows,
                target_h, target_s, target_f, E_true, nu_true):
    pred_h, pred_s, pred_f = forward_observables(best_E, best_nu)
    best_total, best_h, best_s, best_f = weighted_loss(
        pred_h, pred_s, pred_f, target_h, target_s, target_f)

    eval_out = np.array(eval_rows, dtype=np.float32)
    iter_out = np.array(iter_rows, dtype=np.float32)
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "baseline_search_result.npz")
    np.savez(out_path,
             evaluations=eval_out,
             iterations=iter_out,
             learn_param=args.learn_param,
             best_E=np.float32(best_E),
             best_nu=np.float32(best_nu),
             best_loss=np.float32(best_total),
             best_h_loss=np.float32(best_h),
             best_s_loss=np.float32(best_s),
             best_F_loss=np.float32(best_f),
             E_true=np.float32(E_true),
             nu_true=np.float32(nu_true),
             pred_h=pred_h,
             pred_s=pred_s,
             pred_F_mean=pred_f,
             target_h=target_h,
             target_s=target_s,
             target_F_mean=target_f)

    if args.learn_param == "E":
        legacy_path = os.path.join(DATA_DIR, "E_search_result.npz")
        np.savez(legacy_path,
                 evaluations=eval_out,
                 iterations=iter_out,
                 best_E=np.float32(best_E),
                 best_loss=np.float32(best_total),
                 best_h_loss=np.float32(best_h),
                 best_s_loss=np.float32(best_s),
                 best_F_loss=np.float32(best_f),
                 E_true=np.float32(E_true),
                 nu_true=np.float32(nu_true),
                 pred_h=pred_h,
                 pred_s=pred_s,
                 pred_F_mean=pred_f,
                 target_h=target_h,
                 target_s=target_s,
                 target_F_mean=target_f)
        np.save(os.path.join(DATA_DIR, "E_search_log.npy"),
                np.column_stack([eval_out[:, 0], eval_out[:, 2]]))

    print("-" * 72)
    print(f"  best E:      {best_E:.8g}  true E:  {E_true:.8g}")
    print(f"  best nu:     {best_nu:.8g}  true nu: {nu_true:.8g}")
    print(f"  abs errors:  E={abs(best_E - E_true):.8g}, "
          f"nu={abs(best_nu - nu_true):.8g}")
    print(f"  final loss:  {best_total:.8g}")
    print(f"  saved:       {out_path}")
    print("=" * 72 + "\n")


def golden_section_E(target_h, target_s, target_f, E_true, nu_true):
    lo, hi = args.E_lo, args.E_hi
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    c = hi - (hi - lo) * inv_phi
    d = lo + (hi - lo) * inv_phi
    fc, hc, sc, Fc = evaluate(c, nu_true, target_h, target_s, target_f)
    fd, hd, sd, Fd = evaluate(d, nu_true, target_h, target_s, target_f)
    eval_rows = [(c, nu_true, fc, hc, sc, Fc), (d, nu_true, fd, hd, sd, Fd)]
    iter_rows = []
    best_E, best_loss = (c, fc) if fc < fd else (d, fd)

    print("\n" + "=" * 72)
    print("  DERIVATIVE-FREE BASELINE: 1D E SEARCH")
    print("=" * 72)
    print(f"  target: E={E_true:.6g}, nu fixed at {nu_true:.6g}")
    print(f"  bracket: [{lo:.6g}, {hi:.6g}], iters={args.iters}")
    print("-" * 72)
    print(f"{'it':>3} {'lo':>10} {'hi':>10} {'best_E':>10} "
          f"{'best_loss':>12} {'width':>10} {'time':>8}")
    print("-" * 72)

    for it in range(args.iters):
        start = time.perf_counter()
        if fc < fd:
            hi = d
            d, fd = c, fc
            hd, sd, Fd = hc, sc, Fc
            c = hi - (hi - lo) * inv_phi
            fc, hc, sc, Fc = evaluate(c, nu_true, target_h, target_s, target_f)
            eval_rows.append((c, nu_true, fc, hc, sc, Fc))
        else:
            lo = c
            c, fc = d, fd
            hc, sc, Fc = hd, sd, Fd
            d = lo + (hi - lo) * inv_phi
            fd, hd, sd, Fd = evaluate(d, nu_true, target_h, target_s, target_f)
            eval_rows.append((d, nu_true, fd, hd, sd, Fd))

        best_E, best_loss = min([(best_E, best_loss), (c, fc), (d, fd)],
                                key=lambda row: row[1])
        elapsed = time.perf_counter() - start
        width = hi - lo
        iter_rows.append((it, lo, hi, args.nu_lo, args.nu_hi,
                          best_E, nu_true, best_loss, width, 0.0, elapsed))
        print(f"{it:3d} {lo:10.4f} {hi:10.4f} {best_E:10.4f} "
              f"{best_loss:12.4e} {width:10.4f} {elapsed:8.2f}")

    save_result(best_E, nu_true, eval_rows, iter_rows,
                target_h, target_s, target_f, E_true, nu_true)


def grid_search(target_h, target_s, target_f, E_true, nu_true):
    if args.learn_param == "nu":
        E_lo = E_hi = E_true
        nu_lo, nu_hi = args.nu_lo, args.nu_hi
    else:
        E_lo, E_hi = args.E_lo, args.E_hi
        nu_lo, nu_hi = args.nu_lo, args.nu_hi

    best_E = E_true
    best_nu = nu_true
    best_loss = np.inf
    eval_rows = []
    iter_rows = []

    print("\n" + "=" * 72)
    print("  DERIVATIVE-FREE BASELINE: GRID SEARCH")
    print("=" * 72)
    print(f"  learn_param = {args.learn_param}")
    print(f"  target: E={E_true:.6g}, nu={nu_true:.6g}")
    print(f"  grid_size = {args.grid_size}, levels = {args.levels}")
    print("-" * 72)
    print(f"{'level':>5} {'best_E':>10} {'best_nu':>9} {'best_loss':>12} "
          f"{'E_width':>10} {'nu_width':>10} {'time':>8}")
    print("-" * 72)

    for level in range(args.levels):
        start = time.perf_counter()
        E_values = np.array([E_true], dtype=np.float32) if E_lo == E_hi else (
            np.linspace(E_lo, E_hi, args.grid_size, dtype=np.float32))
        nu_values = np.linspace(nu_lo, nu_hi, args.grid_size, dtype=np.float32)

        for E_value in E_values:
            for nu_value in nu_values:
                total, h_loss, s_loss, f_loss = evaluate(
                    float(E_value), float(nu_value),
                    target_h, target_s, target_f)
                eval_rows.append((E_value, nu_value, total,
                                  h_loss, s_loss, f_loss))
                if total < best_loss:
                    best_E = float(E_value)
                    best_nu = float(nu_value)
                    best_loss = float(total)

        elapsed = time.perf_counter() - start
        E_width = E_hi - E_lo
        nu_width = nu_hi - nu_lo
        iter_rows.append((level, E_lo, E_hi, nu_lo, nu_hi,
                          best_E, best_nu, best_loss,
                          E_width, nu_width, elapsed))
        print(f"{level:5d} {best_E:10.4f} {best_nu:9.5f} "
              f"{best_loss:12.4e} {E_width:10.4f} {nu_width:10.5f} "
              f"{elapsed:8.2f}")

        if level < args.levels - 1:
            E_step = E_width / max(args.grid_size - 1, 1)
            nu_step = nu_width / max(args.grid_size - 1, 1)
            if args.learn_param != "nu":
                E_lo = max(args.E_lo, best_E - E_step)
                E_hi = min(args.E_hi, best_E + E_step)
            nu_lo = max(args.nu_lo, best_nu - nu_step)
            nu_hi = min(args.nu_hi, best_nu + nu_step)

    save_result(best_E, best_nu, eval_rows, iter_rows,
                target_h, target_s, target_f, E_true, nu_true)


def main():
    target_h, target_s, target_f, E_true, nu_true = load_target_data()
    if args.learn_param == "E":
        golden_section_E(target_h, target_s, target_f, E_true, nu_true)
    else:
        grid_search(target_h, target_s, target_f, E_true, nu_true)


if __name__ == "__main__":
    main()
