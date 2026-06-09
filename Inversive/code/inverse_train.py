"""
inverse_train.py — inverse-physics learning entry point.

Usage
-----
  python code/gen_target_data.py                   # 1) generate target trajectory
  python code/inverse_train.py --train             # 2) train the NN
  python code/inverse_train.py --infer             # 3) run inference with trained NN
  python code/inverse_train.py --quick_test --tiny # fast smoke-test
"""
import argparse
import os
import time
import numpy as np
import taichi as ti

# ---------------------------------------------------------------------------
# 0.  CLI — parsed first so we can configure before Ti init
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true",
                    help="train the NN to predict material params")
parser.add_argument("--infer", action="store_true",
                    help="run inference with a trained NN")
parser.add_argument("--quick_test", action="store_true",
                    help="quick test: run 2 training epochs and exit")
parser.add_argument("--tiny", action="store_true",
                    help="tiny test: run with minimal particles for fast verification")
parser.add_argument("--epochs", type=int, default=0,
                    help="override max epochs (0 = use default)")
parser.add_argument("--resume", type=str, default="",
                    help="resume from saved model directory")
parser.add_argument("--random_init", action="store_true",
                    help="use the original stochastic particle initialization")
parser.add_argument("--seg_len", type=int, default=2,
                    help="number of simulation steps covered by each Tape")
parser.add_argument("--lr", type=float, default=3e-3,
                    help="AdamW learning rate")
parser.add_argument("--fd_train", action="store_true",
                    help="train NN with finite-difference dL/dE instead of MPM AD")
parser.add_argument("--fd_eps", type=float, default=1.0,
                    help="finite-difference epsilon for --fd_train")
args = parser.parse_args()

TRAIN = args.train or args.quick_test
INFER = args.infer

# ---------------------------------------------------------------------------
# 1.  Taichi init (MUST happen before modules that create fields are imported)
# ---------------------------------------------------------------------------
import sim_config as scfg

scfg.cfg.deterministic_init = not args.random_init
scfg.cfg.init_taichi()

if args.tiny:
    scfg.cfg.apply_tiny()

import mpm_sim as sim
import observables as obs
from nn_layers import Linear, AdamW

# local aliases
cfg = scfg.cfg
DATA_DIR = scfg.DATA_DIR
MODEL_DIR = scfg.MODEL_DIR

# ---------------------------------------------------------------------------
# 2.  NN input field (no gradient)
# ---------------------------------------------------------------------------
nn_input = ti.field(dtype=float, shape=(1, 1, 1, cfg.n_input),
                    needs_grad=False)
surrogate_loss = ti.field(dtype=float, shape=(), needs_grad=True)


# ==============================================================================
# 3.  Initialise the NN model
# ==============================================================================
fc1: Linear = None
fc2: Linear = None
optimizer: AdamW = None
model_dir: str = ""


def init_nn_model():
    global fc1, fc2, optimizer, model_dir

    steps_nn = 1                  # single forward call
    n_hidden_nn = cfg.n_hidden

    fc1 = Linear(n_models=1, batch_size=1, n_steps=steps_nn,
                 n_input=cfg.n_input, n_hidden=n_hidden_nn, n_output=n_hidden_nn,
                 needs_grad=True, activation=False)
    fc2 = Linear(n_models=1, batch_size=1, n_steps=steps_nn,
                 n_input=n_hidden_nn, n_hidden=cfg.n_output,
                 n_output=cfg.n_output,
                 needs_grad=True, activation=True)

    if TRAIN:
        fc1.weights_init()
        fc2.weights_init()
        params = fc1.parameters() + fc2.parameters()
        optimizer = AdamW(params=params, lr=args.lr,
                          betas=(0.9, 0.999), weight_decay=1e-5)
        n_params = sum(p.shape[0] * p.shape[1]
                       if len(p.shape) > 1 else p.shape[0] for p in params)
        print(f"NN initialised.  Params: {n_params}, lr={args.lr:g}")
    else:
        model_dir = args.resume or MODEL_DIR
        p1 = f"{model_dir}/fc1.pkl"
        p2 = f"{model_dir}/fc2.pkl"
        if not os.path.exists(p1) or not os.path.exists(p2):
            print(f"[ERROR] No trained model found at {model_dir}/")
            print("  Run `python inverse_train.py --train` first, then infer.")
            import sys; sys.exit(1)
        fc1.load_weights(p1, model_id=0)
        fc2.load_weights(p2, model_id=0)
        print(f"Model loaded from {model_dir}/")


# ==============================================================================
# 4.  Load target trajectory & compute input features
# ==============================================================================
E_true = 200.0
nu_true = 0.4
target_features: np.ndarray = None


def load_target_data(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "target_trajectory.npz")
    global E_true, nu_true, target_features

    data = np.load(path)
    h_np = data["h"].astype(np.float32)
    s_np = data["s"].astype(np.float32)
    F_np = data["F_mean"].astype(np.float32)
    E_true = float(data["E_true"])
    nu_true = float(data["nu_true"])

    n = min(h_np.shape[0], cfg.n_steps)
    h_np = h_np[:n]; s_np = s_np[:n]; F_np = F_np[:n]

    print(f"Loaded target trajectory: {n} steps (using {cfg.n_steps})")
    print(f"  True params: E={E_true:.1f}, nu={nu_true:.3f}")
    print(f"  h range: [{h_np.min():.4f}, {h_np.max():.4f}]")

    obs.target_h.from_numpy(h_np)
    obs.target_s.from_numpy(s_np)
    obs.target_F_mean.from_numpy(F_np)

    # 6 input features
    features = np.zeros(6, dtype=np.float32)
    features[0] = h_np[0]
    features[1] = h_np[-1]
    features[2] = h_np.max() - h_np.min()
    features[3] = np.mean([np.trace(s_np[t]) for t in range(n)])
    features[4] = np.max([np.trace(s_np[t]) for t in range(n)])
    features[5] = abs(h_np[-1] - h_np[0]) / max(
        n * cfg.substeps_per_step * cfg.dt, 1e-8)

    nn_input.from_numpy(features.reshape(1, 1, 1, cfg.n_input))
    target_features = features
    print(f"  NN input features: {features}")
    return features


# ==============================================================================
# 5.  Training loop
# ==============================================================================

def run_obs_kernels(step):
    """Call all split-kernel observation chains for one timestep."""
    obs.zero_mean_acc(step)
    obs.accum_mean(step)
    obs.copy_mean_to_h(step)
    obs.zero_cov_acc(step)
    obs.accum_cov(step)
    obs.zero_F_acc(step)
    obs.accum_F(step)


def forward_loss_for_E(E_value):
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


@ti.kernel
def set_surrogate_loss(dL_dE: float):
    # This makes d(surrogate_loss)/d(theta) = dL/dE * dE/d(theta)
    # while avoiding reverse-mode AD through the MPM rollout.
    surrogate_loss[None] = dL_dE * sim.E_pred[None]


def predict_E_from_nn():
    fc1.clear()
    fc2.clear()
    fc1.forward(0, nn_input)
    fc2.forward(0, fc1.output)
    sim.copy_nn_to_material_params(fc2.output)
    return float(sim.E_pred[None])


def train_finite_difference():
    print("\n" + "=" * 72)
    print("  NN TRAINING WITH FINITE-DIFFERENCE PHYSICS GRADIENT")
    print("=" * 72)

    max_epochs = 2 if args.quick_test else (args.epochs if args.epochs > 0 else 80)
    losses = []
    train_log = []
    params = fc1.parameters() + fc2.parameters()

    print(f"  max_epochs = {max_epochs}")
    print(f"  lr = {args.lr:g}, fd_eps = {args.fd_eps:g}")
    print("-" * 72)
    print(f"{'epoch':>5} {'loss':>12} {'E':>10} {'|E-E*|':>10} "
          f"{'dL/dE':>12} {'time':>8}")
    print("-" * 72)

    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()

        E_current = predict_E_from_nn()
        loss_current = forward_loss_for_E(E_current)
        loss_plus = forward_loss_for_E(E_current + args.fd_eps)
        loss_minus = forward_loss_for_E(E_current - args.fd_eps)
        dL_dE = (loss_plus - loss_minus) / (2.0 * args.fd_eps)

        optimizer.zero_grad()
        sim.zero_grad(sim.E_pred)
        surrogate_loss[None] = 0.0
        surrogate_loss.grad[None] = 0.0
        fc1.clear()
        fc2.clear()
        fc1.clear_io_grad()
        fc2.clear_io_grad()

        with ti.ad.Tape(loss=surrogate_loss):
            fc1.forward(0, nn_input)
            fc2.forward(0, fc1.output)
            sim.copy_nn_to_material_params(fc2.output)
            set_surrogate_loss(float(dL_dE))

        g_max = 0.0
        for w in params:
            g_arr = w.grad.to_numpy()
            gm = np.max(np.abs(g_arr))
            if gm > g_max:
                g_max = gm
            if np.any(np.isnan(g_arr)):
                print(f"  [WARN] NaN in weight gradient after epoch {epoch}!")

        optimizer.step()
        optimizer.zero_grad()

        E_after = predict_E_from_nn()
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
                  f"{abs(E_after - E_true):10.3f} {dL_dE:12.4e} "
                  f"{epoch_time:8.2f}")

        if abs(dL_dE) < 1e-8:
            print(f"  [STOP] finite-difference gradient is tiny at epoch {epoch}")
            break

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
                 mode="finite_difference")
    print("-" * 72)
    print(f"Model saved to {MODEL_DIR}/")
    print(f"Training log saved to {DATA_DIR}/training_log.npz")
    return losses


def train():
    if args.fd_train:
        return train_finite_difference()

    print("\n" + "=" * 60)
    print("  TRAINING  (multi-segment gradient accumulation)")
    print("=" * 60)

    max_epochs = 2 if args.quick_test else (args.epochs if args.epochs > 0 else 300)

    seg_len = args.seg_len
    num_segments = (cfg.n_steps + seg_len - 1) // seg_len

    print(f"  max_epochs = {max_epochs}")
    print(f"  n_steps = {cfg.n_steps},  seg_len = {seg_len}  "
          f"→ {num_segments} segments (Tape covers ≤{seg_len} steps each)")
    losses = []
    train_log = []

    grad_sensitive = [sim.x, sim.v, sim.C, sim.F, sim.grid_v, sim.grid_m,
                      sim.mu_tmp, sim.lambda_tmp, sim.E_pred,
                      obs.pred_h, obs.pred_s, obs.pred_F_mean,
                      obs.mean_tmp, obs.loss]

    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()
        sim.init_particles()

        for _f in grad_sensitive:
            sim.zero_grad(_f)

        epoch_loss = 0.0

        for seg in range(num_segments):
            seg_start = seg * seg_len
            seg_end = min(seg_start + seg_len, cfg.n_steps)

            fc1.clear()
            fc2.clear()
            fc1.clear_io_grad()
            fc2.clear_io_grad()
            obs.loss[None] = 0.0

            for _f in grad_sensitive:
                sim.zero_grad(_f)

            with ti.ad.Tape(loss=obs.loss):
                fc1.forward(0, nn_input)
                fc2.forward(0, fc1.output)
                sim.copy_nn_to_material_params(fc2.output)
                sim.compute_lame_params()

                for step in range(seg_start, seg_end):
                    for _ in range(cfg.substeps_per_step):
                        sim.differentiable_substep()
                    run_obs_kernels(step)
                    obs.compute_step_loss(step)

            epoch_loss += obs.loss[None]

        g_max = 0.0
        for w in fc1.parameters() + fc2.parameters():
            g_arr = w.grad.to_numpy()
            gm = np.max(np.abs(g_arr))
            if gm > g_max:
                g_max = gm
            if np.any(np.isnan(g_arr)):
                print(f"  [WARN] NaN in weight gradient after epoch {epoch}!")

        optimizer.step()
        optimizer.zero_grad()

        fc1.clear()
        fc2.clear()
        fc1.forward(0, nn_input)
        fc2.forward(0, fc1.output)
        sim.copy_nn_to_material_params(fc2.output)
        e_after_update = float(sim.E_pred[None])
        epoch_time = time.perf_counter() - epoch_start

        if np.isnan(epoch_loss):
            print(f"  [ERROR] NaN loss at epoch {epoch}, stopping training")
            break

        losses.append(epoch_loss)
        train_log.append((
            epoch,
            float(epoch_loss),
            e_after_update,
            abs(e_after_update - E_true),
            float(g_max),
            float(epoch_time),
        ))

        if epoch % 30 == 0 or epoch == max_epochs - 1:
            print(f"  epoch {epoch:4d}  loss={epoch_loss:.6f}  "
                  f"E={e_after_update:.1f}  "
                  f"|E-E_true|={abs(e_after_update - E_true):.2f}  "
                  f"max|grad|={g_max:.6f}  time={epoch_time:.2f}s")

    # final report
    print(f"\n{'='*60}")
    print(f"  True:      E={E_true:.1f}  (ν fixed = {cfg.NU_FIXED:.4f})")
    print(f"  Predicted: E={sim.E_pred[None]:.1f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    fc1.dump_weights(os.path.join(MODEL_DIR, "fc1.pkl"))
    fc2.dump_weights(os.path.join(MODEL_DIR, "fc2.pkl"))
    print(f"Model saved to {MODEL_DIR}/")

    np.save(os.path.join(DATA_DIR, "loss_history.npy"),
            np.array(losses, dtype=np.float32))
    print(f"Loss history saved to {DATA_DIR}/loss_history.npy")

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
                 nu_true=np.float32(nu_true))
        print(f"Training log saved to {DATA_DIR}/training_log.npz")
    return losses


# ==============================================================================
# 6.  Inference
# ==============================================================================

def infer():
    print("\n" + "=" * 60)
    print("  INFERENCE")
    print("=" * 60)

    fc1.clear()
    fc2.clear()
    sim.init_particles()

    fc1.forward(0, nn_input)
    fc2.forward(0, fc1.output)
    sim.copy_nn_to_material_params(fc2.output)
    sim.compute_lame_params()

    print(f"NN predicted: E={sim.E_pred[None]:.2f}")
    print(f"True:          E={E_true:.1f}")
    print("Running simulation with predicted params ...")

    pred_h_arr = np.zeros(cfg.n_steps, dtype=np.float32)
    pred_s_arr = np.zeros((cfg.n_steps, 3, 3), dtype=np.float32)
    pred_F_arr = np.zeros((cfg.n_steps, 3, 3), dtype=np.float32)

    for step in range(cfg.n_steps):
        for _ in range(cfg.substeps_per_step):
            sim.differentiable_substep()
        run_obs_kernels(step)
        pred_h_arr[step] = obs.pred_h[step]
        pred_s_arr[step] = obs.pred_s[step].to_numpy()
        pred_F_arr[step] = obs.pred_F_mean[step].to_numpy()

    mse_h = np.mean((pred_h_arr - obs.target_h.to_numpy()) ** 2)
    target_s_arr = obs.target_s.to_numpy()
    target_F_arr = obs.target_F_mean.to_numpy()
    mse_s = np.mean((pred_s_arr - target_s_arr) ** 2)
    mse_F = np.mean((pred_F_arr - target_F_arr) ** 2)
    print(f"MSE(h)={mse_h:.6f}, MSE(s)={mse_s:.6f}, MSE(F)={mse_F:.6f}")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez(os.path.join(DATA_DIR, "predicted_trajectory.npz"),
             h=pred_h_arr,
             s=pred_s_arr,
             F_mean=pred_F_arr,
             target_h=obs.target_h.to_numpy(),
             target_s=target_s_arr,
             target_F_mean=target_F_arr,
             E_pred=np.float32(sim.E_pred[None]),
             E_true=np.float32(E_true),
             nu_true=np.float32(nu_true),
             mse_h=np.float32(mse_h),
             mse_s=np.float32(mse_s),
             mse_F=np.float32(mse_F))
    print(f"Predicted trajectory saved to {DATA_DIR}/predicted_trajectory.npz")

    return pred_h_arr, pred_s_arr, pred_F_arr


# ==============================================================================
# 7.  Entry point
# ==============================================================================

if __name__ == "__main__":
    load_target_data()
    init_nn_model()

    if TRAIN or args.quick_test:
        train()
    elif INFER:
        infer()
    else:
        print("Please specify --train, --quick_test, or --infer")
        print("  Step 1: python gen_target_data.py")
        print("  Step 2: python inverse_train.py --train")
        print("  Step 3: python inverse_train.py --infer")
