"""
inverse_train_camera.py — NN+FD inverse-physics using CAMERA observations.

Instead of matching 3D observables (h, s, F_mean), this script projects
the MPM particles through a perspective camera at each observation step,
computes 2D projected statistics (mean, 2×2 covariance, depth), and uses
the MSE between predicted and target camera-space observables as the loss.

The training paradigm is identical to ``inverse_train.py``:
    NN predicts E → MPM simulation → camera projection → observables → loss
    dL/dE is estimated via finite differences.

Usage
-----
    python code/gen_camera_target_data.py               # step 1: generate data
    python code/inverse_train_camera.py --train          # step 2: train
    python code/inverse_train_camera.py --infer          # step 3: evaluate
"""
import argparse
import os
import time
import numpy as np
import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true",
                    help="train the NN to predict Young's modulus")
parser.add_argument("--infer", action="store_true",
                    help="run inference with a trained NN")
parser.add_argument("--quick_test", action="store_true",
                    help="run 2 training epochs for a fast smoke test")
parser.add_argument("--tiny", action="store_true",
                    help="run with fewer particles/steps for smoke testing")
parser.add_argument("--epochs", type=int, default=100,
                    help="number of NN+FD training epochs")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="AdamW learning rate")
parser.add_argument("--fd_eps", type=float, default=1.0,
                    help="finite-difference epsilon for dL/dE")
parser.add_argument("--resume", type=str, default="",
                    help="model directory for inference")
args = parser.parse_args()

TRAIN = args.train or args.quick_test
INFER = args.infer

import sim_config as scfg

scfg.cfg.init_taichi()

if args.tiny:
    scfg.cfg.apply_tiny()

import mpm_sim as sim
import camera_observables as cobs
from camera_module import Camera
from nn_layers import Linear, AdamW

cfg = scfg.cfg
CAM_DATA_DIR = scfg.CAM_DATA_DIR
CAM_MODEL_DIR = scfg.CAM_MODEL_DIR

# ── Camera instance (created once, used throughout) ──
camera = Camera(position=cfg.cam_position, lookat=cfg.cam_lookat,
                fov_deg=cfg.cam_fov_deg, aspect_ratio=cfg.cam_aspect_ratio,
                n_particles=cfg.n_particles, needs_grad=False)

# ── NN fields ──
nn_input = ti.field(dtype=float, shape=(1, 1, 1, cfg.n_input_camera),
                    needs_grad=False)
surrogate_loss = ti.field(dtype=float, shape=(), needs_grad=True)

fc1: Linear = None
fc2: Linear = None
optimizer: AdamW = None

E_true = 200.0
nu_true = 0.4
target_data_npz = None


# ═══════════════════════════════════════════════════════════════════════════════
#  NN model initialisation
# ═══════════════════════════════════════════════════════════════════════════════
def init_nn_model():
    global fc1, fc2, optimizer

    fc1 = Linear(n_models=1, batch_size=1, n_steps=1,
                 n_input=cfg.n_input_camera, n_hidden=cfg.n_hidden,
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
        print(f"  n_input_camera = {cfg.n_input_camera} (camera features)")
        return

    model_dir = args.resume or CAM_MODEL_DIR
    p1 = os.path.join(model_dir, "fc1.pkl")
    p2 = os.path.join(model_dir, "fc2.pkl")
    if not os.path.exists(p1) or not os.path.exists(p2):
        raise FileNotFoundError(
            f"No trained model found in {model_dir}. Run --train first.")
    fc1.load_weights(p1, model_id=0)
    fc2.load_weights(p2, model_id=0)
    print(f"Model loaded from {model_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════
def load_camera_target_data(path=None):
    """Load camera trajectory data and populate target observables."""
    global E_true, nu_true, target_data_npz
    if path is None:
        path = os.path.join(CAM_DATA_DIR, "camera_target_trajectory.npz")

    data = np.load(path)
    target_data_npz = data

    proj_mean_np = data["proj_mean"].astype(np.float32)[:cfg.n_steps]
    proj_cov_np = data["proj_cov"].astype(np.float32)[:cfg.n_steps]
    proj_depth_np = data["proj_depth"].astype(np.float32)[:cfg.n_steps]
    E_true = float(data["E_true"])
    nu_true = float(data["nu_true"])

    # Populate camera target fields
    for t in range(cfg.n_steps):
        cobs.target_proj_mean[t] = ti.Vector([proj_mean_np[t, 0],
                                              proj_mean_np[t, 1]])
        cobs.target_proj_cov[t] = ti.Matrix([
            [proj_cov_np[t, 0, 0], proj_cov_np[t, 0, 1]],
            [proj_cov_np[t, 1, 0], proj_cov_np[t, 1, 1]],
        ])
        cobs.target_proj_depth[t] = proj_depth_np[t]

    # Extract NN input features from camera trajectory
    features = Camera.compute_camera_features(
        proj_mean_np, proj_cov_np, proj_depth_np,
        cfg.n_steps, cfg.substeps_per_step, cfg.dt)

    nn_input.from_numpy(features.reshape(1, 1, 1, cfg.n_input_camera))

    print(f"Loaded camera target trajectory: {proj_mean_np.shape[0]} steps")
    print(f"  True params: E={E_true:.1f}, nu={nu_true:.3f}")
    if all(key in data for key in ("x0", "v0", "C0", "F0")):
        warmup = int(data["warmup_steps"]) if "warmup_steps" in data else -1
        print(f"  Initial state: warm-up snapshot (warmup_steps={warmup})")
    else:
        print("  Initial state: default particle initializer")
    print(f"  Camera NN input features ({cfg.n_input_camera}):")
    for i, f in enumerate(features):
        print(f"    [{i}] {f:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Forward simulation + camera projection for one E value
# ═══════════════════════════════════════════════════════════════════════════════
def run_camera_obs_kernels(step):
    """Project particles → compute 2D statistics → accumulate."""
    # Project 3D positions through camera (fills camera.proj_2d, camera.proj_depth)
    camera.project_kernel(sim.x)
    # Compute camera-space statistics (reads camera.proj_2d, camera.proj_depth)
    cobs.run_camera_obs_kernels(step, camera)


def forward_loss_for_E(E_value):
    """Run full MPM simulation + camera projection; return camera loss."""
    sim.init_from_target_data(target_data_npz)
    sim.E_pred[None] = float(E_value)
    sim.compute_lame_params()
    cobs.cam_loss[None] = 0.0

    for step in range(cfg.n_steps):
        for _ in range(cfg.substeps_per_step):
            sim.differentiable_substep()
        run_camera_obs_kernels(step)
        cobs.compute_camera_step_loss(step)

    return float(cobs.cam_loss[None])


# ═══════════════════════════════════════════════════════════════════════════════
#  NN forward pass
# ═══════════════════════════════════════════════════════════════════════════════
def predict_E_from_nn():
    fc1.clear()
    fc2.clear()
    fc1.forward(0, nn_input)
    fc2.forward(0, fc1.output)
    sim.copy_nn_to_material_params(fc2.output)
    return float(sim.E_pred[None])


@ti.kernel
def set_surrogate_loss(dL_dE: float):
    surrogate_loss[None] = dL_dE * sim.E_pred[None]


# ═══════════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════════
def train():
    print("\n" + "=" * 72)
    print("  NN + FD PHYSICS GRADIENT  (CAMERA OBSERVABLES)")
    print("=" * 72)

    max_epochs = 2 if args.quick_test else args.epochs
    params = fc1.parameters() + fc2.parameters()
    losses = []
    train_log = []

    print(f"  max_epochs = {max_epochs}")
    print(f"  lr = {args.lr:g}, fd_eps = {args.fd_eps:g}")
    print(f"  camera: pos={cfg.cam_position}, lookat={cfg.cam_lookat}")
    print(f"  camera: fov={cfg.cam_fov_deg}°, aspect={cfg.cam_aspect_ratio}")
    print("-" * 72)
    print(f"{'epoch':>5} {'loss':>12} {'E':>10} {'|E-E*|':>10} "
          f"{'dL/dE':>12} {'time':>8}")
    print("-" * 72)

    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()

        # Finite-difference gradient estimate through camera loss
        E_current = predict_E_from_nn()
        loss_current = forward_loss_for_E(E_current)
        loss_plus = forward_loss_for_E(E_current + args.fd_eps)
        loss_minus = forward_loss_for_E(E_current - args.fd_eps)
        dL_dE = (loss_plus - loss_minus) / (2.0 * args.fd_eps)

        # Back-propagate through NN using surrogate loss trick
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
            g_max = max(g_max, float(np.max(np.abs(g_arr))))
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

    os.makedirs(CAM_MODEL_DIR, exist_ok=True)
    fc1.dump_weights(os.path.join(CAM_MODEL_DIR, "fc1.pkl"))
    fc2.dump_weights(os.path.join(CAM_MODEL_DIR, "fc2.pkl"))
    np.save(os.path.join(CAM_DATA_DIR, "loss_history.npy"),
            np.array(losses, dtype=np.float32))

    if train_log:
        train_log_arr = np.array(train_log, dtype=np.float32)
        np.savez(os.path.join(CAM_DATA_DIR, "training_log.npz"),
                 epoch=train_log_arr[:, 0],
                 loss=train_log_arr[:, 1],
                 E_pred=train_log_arr[:, 2],
                 E_abs_error=train_log_arr[:, 3],
                 max_grad=train_log_arr[:, 4],
                 epoch_time=train_log_arr[:, 5],
                 E_true=np.float32(E_true),
                 nu_true=np.float32(nu_true),
                 mode="nn_fd_camera")

    print("-" * 72)
    print(f"Model saved to {CAM_MODEL_DIR}/")
    print(f"Training log saved to {CAM_DATA_DIR}/training_log.npz")
    return losses


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════════════════
def infer():
    print("\n" + "=" * 72)
    print("  NN+FD CAMERA INFERENCE")
    print("=" * 72)

    E_pred = predict_E_from_nn()
    print(f"NN predicted: E={E_pred:.4f}")
    print(f"True:         E={E_true:.4f}")

    sim.init_from_target_data(target_data_npz)
    sim.E_pred[None] = E_pred
    sim.compute_lame_params()

    pred_proj_mean = np.zeros((cfg.n_steps, 2), dtype=np.float32)
    pred_proj_cov = np.zeros((cfg.n_steps, 2, 2), dtype=np.float32)
    pred_proj_depth = np.zeros(cfg.n_steps, dtype=np.float32)
    pred_x = np.zeros((cfg.n_steps, cfg.n_particles, cfg.dim),
                      dtype=np.float32)

    for step in range(cfg.n_steps):
        for _ in range(cfg.substeps_per_step):
            sim.differentiable_substep()
        run_camera_obs_kernels(step)
        pred_proj_mean[step] = cobs.pred_proj_mean[step].to_numpy()
        pred_proj_cov[step] = cobs.pred_proj_cov[step].to_numpy()
        pred_proj_depth[step] = cobs.pred_proj_depth[step]
        pred_x[step] = sim.x.to_numpy()

    # Load target for comparison
    target_proj_mean_np = np.zeros((cfg.n_steps, 2), dtype=np.float32)
    target_proj_cov_np = np.zeros((cfg.n_steps, 2, 2), dtype=np.float32)
    target_proj_depth_np = np.zeros(cfg.n_steps, dtype=np.float32)
    for t in range(cfg.n_steps):
        target_proj_mean_np[t] = cobs.target_proj_mean[t].to_numpy()
        target_proj_cov_np[t] = cobs.target_proj_cov[t].to_numpy()
        target_proj_depth_np[t] = cobs.target_proj_depth[t]

    mse_mean = np.mean((pred_proj_mean - target_proj_mean_np) ** 2)
    mse_cov = np.mean((pred_proj_cov - target_proj_cov_np) ** 2)
    mse_depth = np.mean((pred_proj_depth - target_proj_depth_np) ** 2)
    print(f"MSE(proj_mean)={mse_mean:.8e}, MSE(proj_cov)={mse_cov:.8e}, "
          f"MSE(depth)={mse_depth:.8e}")

    os.makedirs(CAM_DATA_DIR, exist_ok=True)
    np.savez(os.path.join(CAM_DATA_DIR, "predicted_trajectory.npz"),
             proj_mean=pred_proj_mean,
             proj_cov=pred_proj_cov,
             proj_depth=pred_proj_depth,
             x=pred_x,
             target_proj_mean=target_proj_mean_np,
             target_proj_cov=target_proj_cov_np,
             target_proj_depth=target_proj_depth_np,
             E_pred=np.float32(E_pred),
             E_true=np.float32(E_true),
             nu_true=np.float32(nu_true),
             mse_proj_mean=np.float32(mse_mean),
             mse_proj_cov=np.float32(mse_cov),
             mse_depth=np.float32(mse_depth))
    print(f"Predicted trajectory saved to "
          f"{CAM_DATA_DIR}/predicted_trajectory.npz")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_camera_target_data()
    init_nn_model()

    if TRAIN:
        train()
    elif INFER:
        infer()
    else:
        print("Please specify --train, --quick_test, or --infer")
