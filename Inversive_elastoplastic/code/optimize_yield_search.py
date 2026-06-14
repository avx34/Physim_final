"""Estimate plasticity yield limit (yield_min) with derivative-free 1D search.

This avoids Taichi reverse-mode AD and finite-difference gradients entirely. 
It uses golden-section search as a robust global optimization baseline.
"""
import argparse
import os
import time
import numpy as np

import sim_config as scfg

parser = argparse.ArgumentParser()
parser.add_argument("--lo", type=float, default=0.8,
                    help="Lower bound of golden section search for yield_min")
parser.add_argument("--hi", type=float, default=1.0,
                    help="Upper bound of golden section search for yield_min")
parser.add_argument("--iters", type=int, default=24,
                    help="Number of search iterations")
args = parser.parse_args()

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
        float(data["yield_min_true"]),
    )


def run_obs_kernels(step):
    obs.zero_mean_acc(step)
    obs.accum_mean(step)
    obs.copy_mean_to_h(step)
    obs.zero_cov_acc(step)
    obs.accum_cov(step)
    obs.zero_F_acc(step)
    obs.accum_F(step)


def forward_observables(yield_value):
    sim.init_from_target_data(target_data_npz)
    
    sim.yield_min_pred[None] = float(yield_value)
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
    return float(total), float(h_loss), float(s_loss), float(f_loss)


def evaluate(yield_value, target_h, target_s, target_f):
    pred_h, pred_s, pred_f = forward_observables(yield_value)
    return weighted_loss(pred_h, pred_s, pred_f, target_h, target_s, target_f)


def print_header(yield_min_true):
    print("\n" + "=" * 72)
    print("  DERIVATIVE-FREE INVERSE BASELINE (PLASTICITY)")
    print("=" * 72)
    print(f"  objective: recover plasticity yield limit (yield_min)")
    print(f"  target yield_min:  {yield_min_true:.6f}")
    print(f"  method:     golden-section search on forward simulation loss")
    print(f"  bracket:    [{args.lo:.6f}, {args.hi:.6f}]")
    print(f"  iters:      {args.iters}")
    print("-" * 72)
    print(f"{'it':>3} {'lo':>10} {'hi':>10} {'best_yield':>12} "
          f"{'best_loss':>12} {'width':>10} {'time':>8}")
    print("-" * 72)


def main():
    target_h, target_s, target_f, yield_min_true = load_target_data()
    lo, hi = args.lo, args.hi
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    c = hi - (hi - lo) * inv_phi
    d = lo + (hi - lo) * inv_phi
    fc, hc, sc, Fc = evaluate(c, target_h, target_s, target_f)
    fd, hd, sd, Fd = evaluate(d, target_h, target_s, target_f)
    eval_rows = [(c, fc, hc, sc, Fc), (d, fd, hd, sd, Fd)]
    iter_rows = []
    best_yield, best_loss = (c, fc) if fc < fd else (d, fd)

    print_header(yield_min_true)
    for it in range(args.iters):
        start = time.perf_counter()
        if fc < fd:
            hi = d
            d, fd = c, fc
            hd, sd, Fd = hc, sc, Fc
            c = hi - (hi - lo) * inv_phi
            fc, hc, sc, Fc = evaluate(c, target_h, target_s, target_f)
            eval_rows.append((c, fc, hc, sc, Fc))
        else:
            lo = c
            c, fc = d, fd
            hc, sc, Fc = hd, sd, Fd
            d = lo + (hi - lo) * inv_phi
            fd, hd, sd, Fd = evaluate(d, target_h, target_s, target_f)
            eval_rows.append((d, fd, hd, sd, Fd))

        best_yield, best_loss = min([(best_yield, best_loss), (c, fc), (d, fd)],
                                   key=lambda row: row[1])
        elapsed = time.perf_counter() - start
        width = hi - lo
        iter_rows.append((it, lo, hi, c, fc, d, fd,
                          best_yield, best_loss, width, elapsed))
        
        print(f"{it:3d} {lo:10.4f} {hi:10.4f} {best_yield:12.4f} "
              f"{best_loss:12.4e} {width:10.4f} {elapsed:8.2f}")

    pred_h, pred_s, pred_f = forward_observables(best_yield)
    best_total, best_h, best_s, best_f = weighted_loss(
        pred_h, pred_s, pred_f, target_h, target_s, target_f)

    eval_out = np.array(eval_rows, dtype=np.float32)
    iter_out = np.array(iter_rows, dtype=np.float32)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    out_path = os.path.join(DATA_DIR, "yield_search_result.npz")
    np.savez(out_path,
             evaluations=eval_out,
             iterations=iter_out,
             best_yield_min=np.float32(best_yield),
             best_loss=np.float32(best_total),
             best_h_loss=np.float32(best_h),
             best_s_loss=np.float32(best_s),
             best_F_loss=np.float32(best_f),
             yield_min_true=np.float32(yield_min_true),
             pred_h=pred_h,
             pred_s=pred_s,
             pred_F_mean=pred_f,
             target_h=target_h,
             target_s=target_s,
             target_F_mean=target_f)

    np.save(os.path.join(DATA_DIR, "yield_search_log.npy"), eval_out[:, :2])

    print("-" * 72)
    print(f"  best yield_min:  {best_yield:.6f}")
    print(f"  true yield_min:  {yield_min_true:.6f}")
    print(f"  abs error:       {abs(best_yield - yield_min_true):.6f}")
    print(f"  final loss:      {best_total:.8e}")
    print(f"  saved:           {out_path}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()