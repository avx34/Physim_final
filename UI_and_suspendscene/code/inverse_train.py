"""NN+finite-difference inverse-physics entry point.

Usage
-----
  python code/get_target_data.py
  python code/inverse_train.py --train
  python code/inverse_train.py --infer
"""
import argparse
import os
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true",
                    help="train the NN to predict one material parameter")
parser.add_argument("--infer", action="store_true",
                    help="run inference with a trained NN")
parser.add_argument("--quick_test", action="store_true",
                    help="run 2 training epochs for a fast smoke test")
parser.add_argument("--tiny", action="store_true",
                    help="run with fewer particles/steps for smoke testing")
parser.add_argument("--epochs", type=int, default=100,
                    help="number of NN+FD training epochs")
parser.add_argument("--lr", type=float, default=None,
                    help="AdamW learning rate; defaults to 3e-3")
parser.add_argument("--fd_eps", type=float, default=None,
                    help="legacy finite-difference epsilon for single-parameter runs")
parser.add_argument("--fd_eps_E", type=float, default=1.0,
                    help="finite-difference epsilon for Young's modulus")
parser.add_argument("--fd_eps_nu", type=float, default=1e-3,
                    help="finite-difference epsilon for Poisson ratio")
parser.add_argument("--learn_param", choices=("E", "nu", "both"), default="E",
                    help="material parameter(s) to infer")
parser.add_argument("--obs_mode", choices=("full", "external", "height"),
                    default="external",
                    help="loss observables: external=0.1h+1000s, full=h+10s+5F, height=h")
parser.add_argument("--obs_alpha_h", type=float, default=None,
                    help="override height loss weight")
parser.add_argument("--obs_alpha_s", type=float, default=None,
                    help="override covariance loss weight")
parser.add_argument("--obs_alpha_F", type=float, default=None,
                    help="override mean-deformation loss weight")
parser.add_argument("--grad_stop", type=float, default=1e-10,
                    help="stop when finite-difference gradient norm is below this value")
parser.add_argument("--resume", type=str, default="",
                    help="model directory for inference")
args = parser.parse_args()

if args.lr is None:
    args.lr = 3e-3

# 修复fd_eps赋值bug
if args.fd_eps is not None:
    if args.learn_param == "nu":
        args.fd_eps_nu = args.fd_eps
    else:
        args.fd_eps_E = args.fd_eps

TRAIN = args.train or args.quick_test
INFER = args.infer

import taichi as ti
import sim_config as scfg

if args.learn_param == "both":
    scfg.cfg.n_output = 2

scfg.cfg.init_taichi()

if args.tiny:
    scfg.cfg.apply_tiny()

import mpm_sim as sim
import observables as obs
from nn_layers import Linear, AdamW

cfg = scfg.cfg
DATA_DIR = scfg.DATA_DIR
MODEL_DIR = scfg.MODEL_DIR
ACTIVE_MODEL_DIR = (MODEL_DIR if args.learn_param == "E"
                    else os.path.join(DATA_DIR, f"trained_model_{args.learn_param}"))

nn_input = ti.field(dtype=float, shape=(1, 1, 1, cfg.n_input),
                    needs_grad=False)
surrogate_loss = ti.field(dtype=float, shape=(), needs_grad=True)

fc1: Linear = None
fc2: Linear = None
optimizer: AdamW = None

# 占位初始值，会被npz文件真实数值覆盖
E_true = 0.0
nu_true = 0.0
target_data_npz = None


def obs_weights():
    if args.obs_mode == "full":
        alpha_h, alpha_s, alpha_F = 1.0, 10.0, 5.0
    elif args.obs_mode == "external":
        alpha_h, alpha_s, alpha_F = 0.1, 1000.0, 0.0
    else:
        alpha_h, alpha_s, alpha_F = 1.0, 0.0, 0.0
    if args.obs_alpha_h is not None:
        alpha_h = args.obs_alpha_h
    if args.obs_alpha_s is not None:
        alpha_s = args.obs_alpha_s
    if args.obs_alpha_F is not None:
        alpha_F = args.obs_alpha_F
    return alpha_h, alpha_s, alpha_F


def weighted_mse(mse_h, mse_s, mse_f):
    alpha_h, alpha_s, alpha_F = obs_weights()
    return float(alpha_h * mse_h + alpha_s * mse_s + alpha_F * mse_f)


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

    model_dir = args.resume or ACTIVE_MODEL_DIR
    p1 = os.path.join(model_dir, "fc1.pkl")
    p2 = os.path.join(model_dir, "fc2.pkl")
    if not os.path.exists(p1) or not os.path.exists(p2):
        raise FileNotFoundError(
            f"No trained model found in {model_dir}. Run --train first.")
    fc1.load_weights(p1, model_id=0)
    fc2.load_weights(p2, model_id=0)
    print(f"Model loaded from {model_dir}/")


def active_param_names():
    if args.learn_param == "both":
        return ("E", "nu")
    return (args.learn_param,)


def true_param_values():
    if args.learn_param == "E":
        return np.array([E_true], dtype=np.float32)
    if args.learn_param == "nu":
        return np.array([nu_true], dtype=np.float32)
    return np.array([E_true, nu_true], dtype=np.float32)


def current_param_values():
    if args.learn_param == "E":
        return np.array([float(sim.E_pred[None])], dtype=np.float32)
    if args.learn_param == "nu":
        return np.array([float(sim.nu_pred[None])], dtype=np.float32)
    return np.array([float(sim.E_pred[None]), float(sim.nu_pred[None])],
                    dtype=np.float32)


def active_bounds_and_eps():
    table = {
        "E": (cfg.E_MIN, cfg.E_MAX, args.fd_eps_E),
        "nu": (cfg.NU_MIN, cfg.NU_MAX, args.fd_eps_nu),
    }
    return [table[name] for name in active_param_names()]


def set_material_params(E_value=None, nu_value=None):
    if E_value is None:
        E_value = E_true
    if nu_value is None:
        nu_value = nu_true
    sim.E_pred[None] = float(E_value)
    sim.nu_pred[None] = float(nu_value)
    sim.compute_lame_params()


def load_target_data(path=None):
    global E_true, nu_true, target_data_npz
    if path is None:
        path = os.path.join(DATA_DIR, "target_trajectory.npz")

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
    print(f"  Observation mode: {args.obs_mode}")
    if any(key in data for key in ("noise_h", "noise_s", "noise_F")):
        noise_h = float(data["noise_h"]) if "noise_h" in data else 0.0
        noise_s = float(data["noise_s"]) if "noise_s" in data else 0.0
        noise_F = float(data["noise_F"]) if "noise_F" in data else 0.0
        noise_seed = int(data["noise_seed"]) if "noise_seed" in data else -1
        print(f"  Target noise: h={noise_h:g}, s={noise_s:g}, "
              f"F={noise_F:g}, seed={noise_seed}")
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


def unpack_active_params(param_values):
    values = np.asarray(param_values, dtype=np.float32).reshape(-1)
    if args.learn_param == "E":
        return float(values[0]), float(nu_true)
    if args.learn_param == "nu":
        return float(E_true), float(values[0])
    return float(values[0]), float(values[1])


def forward_loss_for_params(param_values):
    sim.init_from_target_data(target_data_npz)
    E_value, nu_value = unpack_active_params(param_values)
    set_material_params(E_value, nu_value)
    obs.loss[None] = 0.0

    for step in range(cfg.n_steps):
        for _ in range(cfg.substeps_per_step):
            sim.differentiable_substep()
        run_obs_kernels(step)
        alpha_h, alpha_s, alpha_F = obs_weights()
        obs.compute_step_loss_weighted(step, alpha_h, alpha_s, alpha_F)

    return float(obs.loss[None])


def copy_nn_to_active_param():
    if args.learn_param == "E":
        sim.copy_nn_to_material_params(fc2.output)
        sim.nu_pred[None] = float(nu_true)
    elif args.learn_param == "nu":
        sim.copy_nn_to_nu(fc2.output)
        sim.E_pred[None] = float(E_true)
    else:
        sim.copy_nn_to_E_nu(fc2.output)


def predict_params_from_nn():
    fc1.clear()
    fc2.clear()
    fc1.forward(0, nn_input)
    fc2.forward(0, fc1.output)
    copy_nn_to_active_param()
    return current_param_values()


@ti.kernel
def set_surrogate_loss_E(dL_dparam: float):
    surrogate_loss[None] = dL_dparam * sim.E_pred[None]


@ti.kernel
def set_surrogate_loss_nu(dL_dparam: float):
    surrogate_loss[None] = dL_dparam * sim.nu_pred[None]


@ti.kernel
def set_surrogate_loss_E_nu(dL_dE: float, dL_dnu: float):
    surrogate_loss[None] = dL_dE * sim.E_pred[None] + dL_dnu * sim.nu_pred[None]


def finite_difference_gradient(param_values):
    param_values = np.asarray(param_values, dtype=np.float32).reshape(-1)
    grad = np.zeros_like(param_values)
    loss_current = forward_loss_for_params(param_values)
    for i, (lo, hi, eps) in enumerate(active_bounds_and_eps()):
        plus = param_values.copy()
        minus = param_values.copy()
        plus[i] = min(hi, float(param_values[i] + eps))
        minus[i] = max(lo, float(param_values[i] - eps))
        if plus[i] == minus[i]:
            grad[i] = 0.0
            continue
        loss_plus = forward_loss_for_params(plus)
        loss_minus = forward_loss_for_params(minus)
        grad[i] = (loss_plus - loss_minus) / (plus[i] - minus[i])
    return loss_current, grad


def train():
    train_start = time.perf_counter()
    print("\n" + "=" * 72)
    print("  NN + FINITE-DIFFERENCE PHYSICS GRADIENT")
    print("=" * 72)

    max_epochs = 2 if args.quick_test else args.epochs
    params = fc1.parameters() + fc2.parameters()
    losses = []
    train_log = []

    print(f"  max_epochs = {max_epochs}")
    print(f"  learn_param = {args.learn_param}")
    print(f"  obs_mode = {args.obs_mode}")
    print(f"  lr = {args.lr:g}, fd_eps_E = {args.fd_eps_E:g}, "
          f"fd_eps_nu = {args.fd_eps_nu:g}")
    print(f"  obs weights: alpha_h = {obs_weights()[0]:g}, "
          f"alpha_s = {obs_weights()[1]:g}, "
          f"alpha_F = {obs_weights()[2]:g}, grad_stop = {args.grad_stop:g}")
    print("-" * 72)
    print(f"{'epoch':>5} {'loss':>12} {'E':>10} {'nu':>9} "
          f"{'E_err':>10} {'nu_err':>9} {'|grad|':>10} {'time':>8}")
    print("-" * 72)

    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()

        param_current = predict_params_from_nn()
        loss_current, dL_dparams = finite_difference_gradient(param_current)

        optimizer.zero_grad()
        sim.zero_grad(sim.E_pred)
        sim.zero_grad(sim.nu_pred)
        surrogate_loss[None] = 0.0
        surrogate_loss.grad[None] = 0.0
        fc1.clear()
        fc2.clear()
        fc1.clear_io_grad()
        fc2.clear_io_grad()

        with ti.ad.Tape(loss=surrogate_loss):
            fc1.forward(0, nn_input)
            fc2.forward(0, fc1.output)
            copy_nn_to_active_param()
            if args.learn_param == "E":
                set_surrogate_loss_E(float(dL_dparams[0]))
            elif args.learn_param == "nu":
                set_surrogate_loss_nu(float(dL_dparams[0]))
            else:
                set_surrogate_loss_E_nu(float(dL_dparams[0]),
                                        float(dL_dparams[1]))

        g_max = 0.0
        for w in params:
            g_arr = w.grad.to_numpy()
            g_max = max(g_max, float(np.max(np.abs(g_arr))))
            if np.any(np.isnan(g_arr)):
                print(f"  [WARN] NaN in weight gradient after epoch {epoch}!")

        optimizer.step()
        optimizer.zero_grad()

        param_after = predict_params_from_nn()
        E_after = float(sim.E_pred[None])
        nu_after = float(sim.nu_pred[None])
        E_err = abs(E_after - E_true)
        nu_err = abs(nu_after - nu_true)
        grad_norm = float(np.linalg.norm(dL_dparams))
        grad_E = float(dL_dparams[0]) if args.learn_param in ("E", "both") else 0.0
        grad_nu = (float(dL_dparams[0]) if args.learn_param == "nu"
                   else float(dL_dparams[1]) if args.learn_param == "both"
                   else 0.0)
        epoch_time = time.perf_counter() - epoch_start
        losses.append(loss_current)
        train_log.append((
            epoch,
            float(loss_current),
            float(param_after[0]),
            float(np.linalg.norm(param_after - true_param_values())),
            float(g_max),
            float(epoch_time),
            E_after,
            nu_after,
            E_err,
            nu_err,
            grad_norm,
            grad_E,
            grad_nu,
        ))

        if epoch % 5 == 0 or epoch == max_epochs - 1:
            print(f"{epoch:5d} {loss_current:12.4e} {E_after:10.3f} "
                  f"{nu_after:9.4f} {E_err:10.3f} {nu_err:9.4f} "
                  f"{grad_norm:10.3e} "
                  f"{epoch_time:8.2f}")

        if args.grad_stop > 0.0 and grad_norm < args.grad_stop:
            print(f"  [STOP] finite-difference gradient tiny at epoch {epoch}")
            break

    os.makedirs(ACTIVE_MODEL_DIR, exist_ok=True)
    fc1.dump_weights(os.path.join(ACTIVE_MODEL_DIR, "fc1.pkl"))
    fc2.dump_weights(os.path.join(ACTIVE_MODEL_DIR, "fc2.pkl"))
    np.save(os.path.join(DATA_DIR, "loss_history.npy"),
            np.array(losses, dtype=np.float32))
    train_runtime = time.perf_counter() - train_start

    if train_log:
        train_log_arr = np.array(train_log, dtype=np.float32)
        np.savez(os.path.join(DATA_DIR, "training_log.npz"),
                 epoch=train_log_arr[:, 0],
                 loss=train_log_arr[:, 1],
                 param_pred=train_log_arr[:, 2],
                 param_abs_error=train_log_arr[:, 3],
                 max_grad=train_log_arr[:, 4],
                 epoch_time=train_log_arr[:, 5],
                 E_pred=train_log_arr[:, 6],
                 nu_pred=train_log_arr[:, 7],
                 E_abs_error=train_log_arr[:, 8],
                 nu_abs_error=train_log_arr[:, 9],
                 fd_grad_norm=train_log_arr[:, 10],
                 fd_grad_E=train_log_arr[:, 11],
                 fd_grad_nu=train_log_arr[:, 12],
                 E_true=np.float32(E_true),
                 nu_true=np.float32(nu_true),
                 learn_param=args.learn_param,
                 obs_mode=args.obs_mode,
                 obs_alpha_h=np.float32(obs_weights()[0]),
                 obs_alpha_s=np.float32(obs_weights()[1]),
                 obs_alpha_F=np.float32(obs_weights()[2]),
                 lr=np.float32(args.lr),
                 fd_eps_E=np.float32(args.fd_eps_E),
                 fd_eps_nu=np.float32(args.fd_eps_nu),
                 grad_stop=np.float32(args.grad_stop),
                 max_epochs=np.int32(max_epochs),
                 train_runtime_sec=np.float32(train_runtime),
                 n_forward_evals=np.int32(len(train_log_arr)
                                          * (2 * len(active_param_names()) + 1)),
                 n_input=np.int32(cfg.n_input),
                 n_hidden=np.int32(cfg.n_hidden),
                 n_output=np.int32(cfg.n_output),
                 mode="nn_fd")

    print("-" * 72)
    print(f"Model saved to {ACTIVE_MODEL_DIR}/")
    print(f"Training runtime: {train_runtime:.2f}s")
    print(f"Training log saved to {DATA_DIR}/training_log.npz")
    return losses


def infer():
    print("\n" + "=" * 72)
    print("  NN+FD INFERENCE")
    print("=" * 72)

    param_pred = predict_params_from_nn()
    E_pred = float(sim.E_pred[None])
    nu_pred = float(sim.nu_pred[None])
    print(f"Learned parameter: {args.learn_param}")
    print(f"NN predicted: E={E_pred:.4f}, nu={nu_pred:.5f}")
    print(f"True:         E={E_true:.4f}, nu={nu_true:.5f}")

    sim.init_from_target_data(target_data_npz)
    E_value, nu_value = unpack_active_params(param_pred)
    set_material_params(E_value, nu_value)

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
    weighted_loss = weighted_mse(mse_h, mse_s, mse_f)
    print(f"MSE(h)={mse_h:.8e}, MSE(s)={mse_s:.8e}, MSE(F)={mse_f:.8e}")
    print(f"Weighted loss ({args.obs_mode})={weighted_loss:.8e}")

    clean_payload = {}
    if all(key in target_data_npz for key in ("h_clean", "s_clean",
                                              "F_mean_clean")):
        clean_h = target_data_npz["h_clean"].astype(np.float32)[:cfg.n_steps]
        clean_s = target_data_npz["s_clean"].astype(np.float32)[:cfg.n_steps]
        clean_f = target_data_npz["F_mean_clean"].astype(np.float32)[:cfg.n_steps]
        clean_mse_h = np.mean((pred_h - clean_h) ** 2)
        clean_mse_s = np.mean((pred_s - clean_s) ** 2)
        clean_mse_f = np.mean((pred_f - clean_f) ** 2)
        clean_weighted_loss = weighted_mse(clean_mse_h, clean_mse_s,
                                           clean_mse_f)
        clean_payload = dict(
            target_h_clean=clean_h,
            target_s_clean=clean_s,
            target_F_mean_clean=clean_f,
            clean_mse_h=np.float32(clean_mse_h),
            clean_mse_s=np.float32(clean_mse_s),
            clean_mse_F=np.float32(clean_mse_f),
            clean_weighted_loss=np.float32(clean_weighted_loss),
        )
        print("Clean-target MSE: "
              f"h={clean_mse_h:.8e}, s={clean_mse_s:.8e}, "
              f"F={clean_mse_f:.8e}")

    os.makedirs(DATA_DIR, exist_ok=True)
    payload = dict(h=pred_h,
                   s=pred_s,
                   F_mean=pred_f,
                   x=pred_x,
                   target_h=target_h,
                   target_s=target_s,
                   target_F_mean=target_f,
                   E_pred=np.float32(E_pred),
                   nu_pred=np.float32(nu_pred),
                   E_true=np.float32(E_true),
                   nu_true=np.float32(nu_true),
                   learn_param=args.learn_param,
                   obs_mode=args.obs_mode,
                   obs_alpha_h=np.float32(obs_weights()[0]),
                   obs_alpha_s=np.float32(obs_weights()[1]),
                   obs_alpha_F=np.float32(obs_weights()[2]),
                   model_dir=ACTIVE_MODEL_DIR,
                   lr=np.float32(args.lr),
                   fd_eps_E=np.float32(args.fd_eps_E),
                   fd_eps_nu=np.float32(args.fd_eps_nu),
                   mse_h=np.float32(mse_h),
                   mse_s=np.float32(mse_s),
                   mse_F=np.float32(mse_f),
                   weighted_loss=np.float32(weighted_loss),
                   **clean_payload)
    for key in ("noise_h", "noise_s", "noise_F", "noise_seed"):
        if key in target_data_npz:
            payload[key] = target_data_npz[key]
    np.savez(os.path.join(DATA_DIR, "predicted_trajectory.npz"), **payload)
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