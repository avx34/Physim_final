"""NN+autodiff inverse-physics entry point.

Instead of a finite-difference estimate of dL/dE, this script wraps the
full MPM forward simulation inside `ti.ad.Tape` so that reverse-mode AD
propagates gradients through the physics kernels back to the NN weights.

Usage
-----
  python code/gen_target_data.py
  python code/inverse_train_ad.py --train
  python code/inverse_train_ad.py --infer
"""
import argparse
import os
import time
import traceback
import numpy as np
import taichi as ti


parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true",
                    help="train the NN (full autodiff through simulation)")
parser.add_argument("--infer", action="store_true",
                    help="run inference with a trained NN")
parser.add_argument("--quick_test", action="store_true",
                    help="run 2 training epochs for a fast smoke test")
parser.add_argument("--tiny", action="store_true",
                    help="run with fewer particles/steps to avoid OOM")
parser.add_argument("--epochs", type=int, default=100,
                    help="number of NN+AD training epochs")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="AdamW learning rate")
parser.add_argument("--resume", type=str, default="",
                    help="model directory for inference")
args = parser.parse_args()

TRAIN = args.train or args.quick_test
INFER = args.infer

import sim_config as scfg

# Use a distinct model/data directory to avoid overwriting NN+FD results
scfg.cfg.init_taichi()

if args.tiny:
    scfg.cfg.apply_tiny()

import mpm_sim as sim
import observables as obs
from nn_layers import Linear, AdamW

cfg = scfg.cfg
# Override DATA_DIR/MODEL_DIR to keep NN+AD results separate
DATA_DIR = os.path.join(os.path.dirname(scfg.DATA_DIR), "data_ad")
MODEL_DIR = os.path.join(DATA_DIR, "trained_model")

nn_input = ti.field(dtype=float, shape=(1, 1, 1, cfg.n_input),
                    needs_grad=False)

fc1: Linear = None
fc2: Linear = None
optimizer: AdamW = None

E_true = 200.0
nu_true = 0.4
target_data_npz = None


def init_nn_model():
    global fc1, fc2, optimizer

    fc1 = Linear(n_models=1, batch_size=1, n_steps=1,
                 n_input=cfg.n_input, n_hidden=cfg.n_hidden,
                 n_output=cfg.n_hidden, needs_grad=True,
                 activation=False)
    fc2 = Linear(n_models=1, batch_size=1, n_steps=1,
                 n_input=cfg.n_hidden, n_hidden=cfg.n_output,
                 n_output=cfg.n_output, needs_grad=True,
                 activation=True)

    if TRAIN:
        fc1.weights_init()
        fc2.weights_init()
        params = fc1.parameters() + fc2.parameters()
        optimizer = AdamW(params=params, lr=args.lr,
                          betas=(0.9, 0.999), weight_decay=1e-5)
        n_params = sum(p.shape[0] * p.shape[1]
                       if len(p.shape) > 1 else p.shape[0] for p in params)
        print(f"NN initialised. Params: {n_params}, lr={args.lr:g}")
        return

    model_dir = args.resume or MODEL_DIR
    p1 = os.path.join(model_dir, "fc1.pkl")
    p2 = os.path.join(model_dir, "fc2.pkl")
    if not os.path.exists(p1) or not os.path.exists(p2):
        raise FileNotFoundError(
            f"No trained model found in {model_dir}. Run --train first.")
    fc1.load_weights(p1, model_id=0)
    fc2.load_weights(p2, model_id=0)
    print(f"Model loaded from {model_dir}/")


def load_target_data(path=None):
    global E_true, nu_true, target_data_npz
    if path is None:
        # Use the common target data from the main data dir
        path = os.path.join(scfg.DATA_DIR, "target_trajectory.npz")

    data = np.load(path)
    target_data_npz = data
    h_np = data["h"].astype(np.float32)[:cfg.n_steps]
    s_np = data["s"].astype(np.float32)[:cfg.n_steps]
    f_np = data["F_mean"].astype(np.float32)[:cfg.n_steps]
    E_true = float(data["E_true"])
    nu_true = float(data["nu_true"])

    obs.target_h.from_numpy(h_np)
    obs.target_s.from_numpy(s_np)
    obs.target_F_mean.from_numpy(f_np)

    features = np.zeros(cfg.n_input, dtype=np.float32)
    features[0] = h_np[0]
    features[1] = h_np[-1]
    features[2] = h_np.max() - h_np.min()
    features[3] = np.mean([np.trace(s_np[t]) for t in range(h_np.shape[0])])
    features[4] = np.max([np.trace(s_np[t]) for t in range(h_np.shape[0])])
    features[5] = abs(h_np[-1] - h_np[0]) / max(
        h_np.shape[0] * cfg.substeps_per_step * cfg.dt, 1e-8)
    nn_input.from_numpy(features.reshape(1, 1, 1, cfg.n_input))

    print(f"Loaded target trajectory: {h_np.shape[0]} steps")
    print(f"  True params: E={E_true:.1f}, nu={nu_true:.3f}")
    if all(key in data for key in ("x0", "v0", "C0", "F0")):
        warmup = int(data["warmup_steps"]) if "warmup_steps" in data else -1
        print(f"  Initial state: warm-up snapshot (warmup_steps={warmup})")
    else:
        print("  Initial state: default particle initializer")
    print(f"  NN input features: {features}")


def run_obs_kernels(step):
    obs.zero_mean_acc(step)
    obs.accum_mean(step)
    obs.copy_mean_to_h(step)
    obs.zero_cov_acc(step)
    obs.accum_cov(step)
    obs.zero_F_acc(step)
    obs.accum_F(step)


def predict_E_from_nn():
    fc1.clear()
    fc2.clear()
    fc1.forward(0, nn_input)
    fc2.forward(0, fc1.output)
    sim.copy_nn_to_material_params(fc2.output)
    return float(sim.E_pred[None])


def train():
    """Train NN weights using full autodiff through MPM simulation."""
    print("\n" + "=" * 72)
    print("  NN + AUTODIFF PHYSICS GRADIENT")
    print("=" * 72)

    max_epochs = 2 if args.quick_test else args.epochs
    params = fc1.parameters() + fc2.parameters()
    losses = []
    train_log = []

    print(f"  max_epochs = {max_epochs}")
    print(f"  lr = {args.lr:g}")
    print(f"  method: full Taichi reverse-mode AD through MPM sim")
    print(f"  n_steps = {cfg.n_steps}, n_particles = {cfg.n_particles}")
    print("-" * 72)
    print(f"{'epoch':>5} {'loss':>12} {'E':>10} {'|E-E*|':>10} "
          f"{'max|g|':>12} {'time':>8}")
    print("-" * 72)

    oom_detected = False

    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()

        # Reset gradient state
        optimizer.zero_grad()
        sim.zero_grad(sim.E_pred)
        sim.zero_grad(sim.mu_tmp)
        sim.zero_grad(sim.lambda_tmp)
        fc1.clear()
        fc2.clear()
        fc1.clear_io_grad()
        fc2.clear_io_grad()

        # Re-init simulation to a clean state
        sim.init_from_target_data(target_data_npz)
        obs.loss[None] = 0.0

        try:
            with ti.ad.Tape(loss=obs.loss):
                # NN forward
                fc1.forward(0, nn_input)
                fc2.forward(0, fc1.output)

                # Copy NN output to material parameter E
                sim.copy_nn_to_material_params(fc2.output)
                sim.compute_lame_params()

                # Full MPM simulation — all inside AD tape
                for step in range(cfg.n_steps):
                    for _ in range(cfg.substeps_per_step):
                        sim.differentiable_substep()
                    run_obs_kernels(step)
                    obs.compute_step_loss(step)
        except Exception as e:
            print(f"\n  [ERROR] AD tape failed at epoch {epoch}: {e}")
            traceback.print_exc()
            print("  Falling back to finite-difference gradient for "
                  "remaining epochs.")
            oom_detected = True

            # Re-init without AD and compute loss for logging
            sim.init_from_target_data(target_data_npz)
            fc1.clear()
            fc2.clear()
            fc1.forward(0, nn_input)
            fc2.forward(0, fc1.output)
            sim.copy_nn_to_material_params(fc2.output)
            sim.compute_lame_params()
            obs.loss[None] = 0.0
            for step in range(cfg.n_steps):
                for _ in range(cfg.substeps_per_step):
                    sim.differentiable_substep()
                run_obs_kernels(step)
                obs.compute_step_loss(step)

            loss_current = float(obs.loss[None])
            E_after = float(sim.E_pred[None])
            epoch_time = time.perf_counter() - epoch_start
            losses.append(loss_current)
            train_log.append((
                epoch, loss_current, E_after, abs(E_after - E_true),
                0.0, epoch_time,
            ))
            print(f"{epoch:5d} {loss_current:12.4e} {E_after:10.3f} "
                  f"{abs(E_after - E_true):10.3f} {'N/A':>12} "
                  f"{epoch_time:8.2f}")
            break

        # Read gradient magnitude for diagnostics
        g_max = 0.0
        for w in params:
            g_arr = w.grad.to_numpy()
            g_max = max(g_max, float(np.max(np.abs(g_arr))))
            if np.any(np.isnan(g_arr)):
                print(f"  [WARN] NaN in weight gradient after epoch {epoch}!")

        optimizer.step()

        # Forward pass without AD for logging
        sim.init_from_target_data(target_data_npz)
        fc1.clear()
        fc2.clear()
        fc1.forward(0, nn_input)
        fc2.forward(0, fc1.output)
        sim.copy_nn_to_material_params(fc2.output)
        sim.compute_lame_params()
        obs.loss[None] = 0.0
        for step in range(cfg.n_steps):
            for _ in range(cfg.substeps_per_step):
                sim.differentiable_substep()
            run_obs_kernels(step)
            obs.compute_step_loss(step)

        loss_current = float(obs.loss[None])
        E_after = float(sim.E_pred[None])
        epoch_time = time.perf_counter() - epoch_start
        losses.append(loss_current)
        train_log.append((
            epoch,
            float(loss_current),
            E_after,
            abs(E_after - E_true),
            float(g_max),
            float(epoch_time),
        ))

        if epoch % 5 == 0 or epoch == max_epochs - 1:
            print(f"{epoch:5d} {loss_current:12.4e} {E_after:10.3f} "
                  f"{abs(E_after - E_true):10.3f} {g_max:12.4e} "
                  f"{epoch_time:8.2f}")

    if oom_detected:
        print("\n  [NOTE] Full AD ran into an error (likely OOM). "
              "Try --tiny for lighter settings.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    fc1.dump_weights(os.path.join(MODEL_DIR, "fc1.pkl"))
    fc2.dump_weights(os.path.join(MODEL_DIR, "fc2.pkl"))
    np.save(os.path.join(DATA_DIR, "loss_history.npy"),
            np.array(losses, dtype=np.float32))

    if train_log:
        train_log_arr = np.array(train_log, dtype=np.float32)
        np.savez(os.path.join(DATA_DIR, "training_log.npz"),
                 epoch=train_log_arr[:, 0],
                 loss=train_log_arr[:, 1],
                 E_pred=train_log_arr[:, 2],
                 E_abs_error=train_log_arr[:, 3],
                 max_grad=train_log_arr[:, 4],
                 epoch_time=train_log_arr[:, 5],
                 E_true=np.float32(E_true),
                 nu_true=np.float32(nu_true),
                 mode="nn_ad")

    print("-" * 72)
    print(f"Model saved to {MODEL_DIR}/")
    print(f"Training log saved to {DATA_DIR}/training_log.npz")
    return losses


def infer():
    print("\n" + "=" * 72)
    print("  NN+AD INFERENCE")
    print("=" * 72)

    E_pred = predict_E_from_nn()
    print(f"NN predicted: E={E_pred:.4f}")
    print(f"True:         E={E_true:.4f}")

    sim.init_from_target_data(target_data_npz)
    sim.E_pred[None] = E_pred
    sim.compute_lame_params()

    pred_h = np.zeros(cfg.n_steps, dtype=np.float32)
    pred_s = np.zeros((cfg.n_steps, 3, 3), dtype=np.float32)
    pred_f = np.zeros((cfg.n_steps, 3, 3), dtype=np.float32)
    pred_x = np.zeros((cfg.n_steps, cfg.n_particles, cfg.dim),
                      dtype=np.float32)

    for step in range(cfg.n_steps):
        for _ in range(cfg.substeps_per_step):
            sim.differentiable_substep()
        run_obs_kernels(step)
        pred_h[step] = obs.pred_h[step]
        pred_s[step] = obs.pred_s[step].to_numpy()
        pred_f[step] = obs.pred_F_mean[step].to_numpy()
        pred_x[step] = sim.x.to_numpy()

    target_h = obs.target_h.to_numpy()
    target_s = obs.target_s.to_numpy()
    target_f = obs.target_F_mean.to_numpy()
    mse_h = np.mean((pred_h - target_h) ** 2)
    mse_s = np.mean((pred_s - target_s) ** 2)
    mse_f = np.mean((pred_f - target_f) ** 2)
    print(f"MSE(h)={mse_h:.8e}, MSE(s)={mse_s:.8e}, MSE(F)={mse_f:.8e}")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(os.path.join(DATA_DIR, "predicted_trajectory.npz"),
             h=pred_h,
             s=pred_s,
             F_mean=pred_f,
             x=pred_x,
             target_h=target_h,
             target_s=target_s,
             target_F_mean=target_f,
             E_pred=np.float32(E_pred),
             E_true=np.float32(E_true),
             nu_true=np.float32(nu_true),
             mse_h=np.float32(mse_h),
             mse_s=np.float32(mse_s),
             mse_F=np.float32(mse_f))
    print(f"Predicted trajectory saved to {DATA_DIR}/predicted_trajectory.npz")


if __name__ == "__main__":
    load_target_data()
    init_nn_model()

    if TRAIN:
        train()
    elif INFER:
        infer()
    else:
        print("Please specify --train, --quick_test, or --infer")
