"""Evaluate inverse-method robustness across several target E values.

This script generates synthetic targets with different Young's moduli, then
evaluates the trained NN+FD inverse model and the derivative-free baseline on
each target.

Examples:

    python code/evaluate_E_robustness.py --E_values 300,350,400,450,500
    python code/evaluate_E_robustness.py --methods baseline --baseline_iters 18
"""
import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--E_values", default="300,350,400,450,500",
                    help="comma-separated target Young's modulus values")
parser.add_argument("--methods", choices=("nn", "baseline", "all"),
                    default="all")
parser.add_argument("--baseline_iters", type=int, default=18)
parser.add_argument("--lo", type=float, default=50.0)
parser.add_argument("--hi", type=float, default=800.0)
parser.add_argument("--model_dir", default="",
                    help="trained NN directory; defaults to data/trained_model")
parser.add_argument("--out", default="E_robustness_results",
                    help="output file stem under data/")
args = parser.parse_args()

import taichi as ti
import sim_config as scfg

scfg.cfg.init_taichi()

import mpm_sim as sim
import observables as obs
from nn_layers import Linear


cfg = scfg.cfg
DATA_DIR = scfg.DATA_DIR
MODEL_DIR = args.model_dir or scfg.MODEL_DIR
PLOT_DIR = os.path.join(DATA_DIR, "plots", "robustness")

nn_input = ti.field(dtype=float, shape=(1, 1, 1, cfg.n_input),
                    needs_grad=False)

fc1 = None
fc2 = None


def parse_E_values():
    values = []
    for item in args.E_values.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("No E values were provided.")
    return values


def init_nn_model():
    global fc1, fc2
    fc1 = Linear(n_models=1, batch_size=1, n_steps=1,
                 n_input=cfg.n_input, n_hidden=cfg.n_hidden,
                 n_output=cfg.n_hidden, needs_grad=False,
                 activation=False)
    fc2 = Linear(n_models=1, batch_size=1, n_steps=1,
                 n_input=cfg.n_hidden, n_hidden=cfg.n_output,
                 n_output=cfg.n_output, needs_grad=False,
                 activation=True)

    p1 = os.path.join(MODEL_DIR, "fc1.pkl")
    p2 = os.path.join(MODEL_DIR, "fc2.pkl")
    if not os.path.exists(p1) or not os.path.exists(p2):
        raise FileNotFoundError(
            f"Missing trained NN weights in {MODEL_DIR}. Run training first.")
    fc1.load_weights(p1, model_id=0)
    fc2.load_weights(p2, model_id=0)
    print(f"Loaded NN model from {MODEL_DIR}/")


def run_obs_kernels(step):
    obs.zero_mean_acc(step)
    obs.accum_mean(step)
    obs.copy_mean_to_h(step)
    obs.zero_cov_acc(step)
    obs.accum_cov(step)
    obs.zero_F_acc(step)
    obs.accum_F(step)


def forward_observables(E_value):
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


def make_features(h_np, s_np):
    features = np.zeros(cfg.n_input, dtype=np.float32)
    features[0] = h_np[0]
    features[1] = h_np[-1]
    features[2] = h_np.max() - h_np.min()
    features[3] = np.mean([np.trace(s_np[t]) for t in range(h_np.shape[0])])
    features[4] = np.max([np.trace(s_np[t]) for t in range(h_np.shape[0])])
    features[5] = abs(h_np[-1] - h_np[0]) / max(
        h_np.shape[0] * cfg.substeps_per_step * cfg.dt, 1e-8)
    return features


def predict_nn_E(h_np, s_np):
    features = make_features(h_np, s_np)
    nn_input.from_numpy(features.reshape(1, 1, 1, cfg.n_input))
    fc1.clear()
    fc2.clear()
    fc1.forward(0, nn_input)
    fc2.forward(0, fc1.output)
    sim.copy_nn_to_material_params(fc2.output)
    return float(sim.E_pred[None])


def weighted_loss(pred_h, pred_s, pred_f, target_h, target_s, target_f):
    h_loss = np.mean((pred_h - target_h) ** 2)
    s_loss = np.sum((pred_s - target_s) ** 2) / cfg.n_steps
    f_loss = np.sum((pred_f - target_f) ** 2) / cfg.n_steps
    total = h_loss + 10.0 * s_loss + 5.0 * f_loss
    return float(total), float(h_loss), float(s_loss), float(f_loss)


def evaluate_E(E_value, target_h, target_s, target_f):
    pred_h, pred_s, pred_f = forward_observables(E_value)
    return weighted_loss(pred_h, pred_s, pred_f, target_h, target_s, target_f)


def baseline_search(target_h, target_s, target_f):
    lo, hi = args.lo, args.hi
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    c = hi - (hi - lo) * inv_phi
    d = lo + (hi - lo) * inv_phi
    fc, _, _, _ = evaluate_E(c, target_h, target_s, target_f)
    fd, _, _, _ = evaluate_E(d, target_h, target_s, target_f)
    best_E, best_loss = (c, fc) if fc < fd else (d, fd)

    for _ in range(args.baseline_iters):
        if fc < fd:
            hi = d
            d, fd = c, fc
            c = hi - (hi - lo) * inv_phi
            fc, _, _, _ = evaluate_E(c, target_h, target_s, target_f)
        else:
            lo = c
            c, fc = d, fd
            d = lo + (hi - lo) * inv_phi
            fd, _, _, _ = evaluate_E(d, target_h, target_s, target_f)
        best_E, best_loss = min([(best_E, best_loss), (c, fc), (d, fd)],
                                key=lambda row: row[1])
    return best_E, best_loss


def save_results(rows):
    os.makedirs(DATA_DIR, exist_ok=True)
    stem = args.out
    csv_path = os.path.join(DATA_DIR, f"{stem}.csv")
    npz_path = os.path.join(DATA_DIR, f"{stem}.npz")
    fields = [
        "method", "E_true", "E_pred", "abs_error", "rel_error",
        "loss", "h_loss", "s_loss", "F_loss", "seconds",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    np.savez(npz_path, rows=np.array(
        [[r["E_true"], r["E_pred"], r["abs_error"], r["rel_error"],
          r["loss"], r["h_loss"], r["s_loss"], r["F_loss"], r["seconds"]]
         for r in rows], dtype=np.float32),
        methods=np.array([r["method"] for r in rows]))
    print(f"Saved {csv_path}")
    print(f"Saved {npz_path}")
    return csv_path


def plot_results(rows):
    os.makedirs(PLOT_DIR, exist_ok=True)
    methods = sorted(set(row["method"] for row in rows))

    plt.figure(figsize=(7.2, 4.4))
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        subset.sort(key=lambda row: row["E_true"])
        plt.plot([row["E_true"] for row in subset],
                 [row["E_pred"] for row in subset],
                 marker="o", linewidth=2, label=method)
    all_E = [row["E_true"] for row in rows]
    plt.plot([min(all_E), max(all_E)], [min(all_E), max(all_E)],
             color="#333333", linestyle="--", linewidth=1.5, label="ideal")
    plt.xlabel("Target E")
    plt.ylabel("Estimated E")
    plt.title("Robustness Across Target Stiffness")
    plt.legend()
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "E_prediction_vs_target.png")
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Saved {path}")

    plt.figure(figsize=(7.2, 4.4))
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        subset.sort(key=lambda row: row["E_true"])
        plt.plot([row["E_true"] for row in subset],
                 [row["abs_error"] for row in subset],
                 marker="o", linewidth=2, label=method)
    plt.xlabel("Target E")
    plt.ylabel("|Estimated E - Target E|")
    plt.title("Absolute Parameter Error")
    plt.legend()
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "E_abs_error.png")
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Saved {path}")


def main():
    e_values = parse_E_values()
    use_nn = args.methods in ("nn", "all")
    use_baseline = args.methods in ("baseline", "all")
    if use_nn:
        init_nn_model()

    rows = []
    print("\n" + "=" * 78)
    print("  ROBUSTNESS EVALUATION")
    print("=" * 78)
    print(f"{'method':>10} {'E_true':>10} {'E_pred':>10} "
          f"{'abs_err':>10} {'loss':>12} {'seconds':>9}")
    print("-" * 78)

    for E_true in e_values:
        target_h, target_s, target_f = forward_observables(E_true)

        if use_nn:
            start = time.perf_counter()
            E_pred = predict_nn_E(target_h, target_s)
            pred_h, pred_s, pred_f = forward_observables(E_pred)
            total, h_loss, s_loss, f_loss = weighted_loss(
                pred_h, pred_s, pred_f, target_h, target_s, target_f)
            elapsed = time.perf_counter() - start
            rows.append(dict(method="nn_fd", E_true=E_true, E_pred=E_pred,
                             abs_error=abs(E_pred - E_true),
                             rel_error=abs(E_pred - E_true) / E_true,
                             loss=total, h_loss=h_loss, s_loss=s_loss,
                             F_loss=f_loss, seconds=elapsed))
            print(f"{'nn_fd':>10} {E_true:10.3f} {E_pred:10.3f} "
                  f"{abs(E_pred - E_true):10.3f} {total:12.4e} "
                  f"{elapsed:9.2f}")

        if use_baseline:
            start = time.perf_counter()
            E_pred, _ = baseline_search(target_h, target_s, target_f)
            pred_h, pred_s, pred_f = forward_observables(E_pred)
            total, h_loss, s_loss, f_loss = weighted_loss(
                pred_h, pred_s, pred_f, target_h, target_s, target_f)
            elapsed = time.perf_counter() - start
            rows.append(dict(method="baseline", E_true=E_true, E_pred=E_pred,
                             abs_error=abs(E_pred - E_true),
                             rel_error=abs(E_pred - E_true) / E_true,
                             loss=total, h_loss=h_loss, s_loss=s_loss,
                             F_loss=f_loss, seconds=elapsed))
            print(f"{'baseline':>10} {E_true:10.3f} {E_pred:10.3f} "
                  f"{abs(E_pred - E_true):10.3f} {total:12.4e} "
                  f"{elapsed:9.2f}")

    print("=" * 78)
    save_results(rows)
    plot_results(rows)


if __name__ == "__main__":
    main()
