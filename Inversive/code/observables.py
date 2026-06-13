"""Trajectory observables and loss computation.

The target/predicted observables are:
  h       mean particle height
  s       3x3 particle-position covariance
  F_mean  mean deformation gradient

Loss weights are configurable at call time so experiments can use full
simulator observables or only externally observable trajectory statistics.
"""
import taichi as ti
from sim_config import cfg
import mpm_sim as sim


NP_FLOAT = float(cfg.n_particles)
NS_FLOAT = float(cfg.n_steps)


# Target observables loaded from target_trajectory.npz.
target_h = ti.field(dtype=float, shape=cfg.n_steps, needs_grad=False)
target_s = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float,
                           shape=cfg.n_steps, needs_grad=False)
target_F_mean = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float,
                                shape=cfg.n_steps, needs_grad=False)


# Predicted observables from the current rollout.
pred_h = ti.field(dtype=float, shape=cfg.n_steps, needs_grad=True)
pred_s = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float,
                         shape=cfg.n_steps, needs_grad=True)
pred_F_mean = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float,
                              shape=cfg.n_steps, needs_grad=True)

# Shared mean accumulator used by height and covariance kernels.
mean_tmp = ti.Vector.field(cfg.dim, dtype=float,
                           shape=cfg.n_steps, needs_grad=True)

loss = ti.field(dtype=float, shape=(), needs_grad=True)


@ti.kernel
def zero_mean_acc(t: ti.i32):
    for _ in range(1):
        mean_tmp[t] = ti.Vector.zero(float, 3)


@ti.kernel
def accum_mean(t: ti.i32):
    for i in range(cfg.n_particles):
        mean_tmp[t] += sim.x[i] / NP_FLOAT


@ti.kernel
def copy_mean_to_h(t: ti.i32):
    for _ in range(1):
        pred_h[t] = mean_tmp[t][1]


@ti.kernel
def zero_cov_acc(t: ti.i32):
    for _ in range(1):
        for j, k in ti.static(ti.ndrange(3, 3)):
            pred_s[t][j, k] = 0.0


@ti.kernel
def accum_cov(t: ti.i32):
    for i in range(cfg.n_particles):
        for j, k in ti.static(ti.ndrange(3, 3)):
            pred_s[t][j, k] += ((sim.x[i][j] - mean_tmp[t][j])
                                * (sim.x[i][k] - mean_tmp[t][k])) / NP_FLOAT


@ti.kernel
def zero_F_acc(t: ti.i32):
    for _ in range(1):
        for j, k in ti.static(ti.ndrange(3, 3)):
            pred_F_mean[t][j, k] = 0.0


@ti.kernel
def accum_F(t: ti.i32):
    for i in range(cfg.n_particles):
        for j, k in ti.static(ti.ndrange(3, 3)):
            pred_F_mean[t][j, k] += sim.F[i][j, k] / NP_FLOAT


@ti.func
def _accum_step_loss(t: ti.i32, alpha_h: float,
                     alpha_s: float, alpha_F: float):
    h_diff = pred_h[t] - target_h[t]
    loss[None] += alpha_h * h_diff * h_diff / NS_FLOAT

    s_frob = 0.0
    for j, k in ti.static(ti.ndrange(3, 3)):
        d = pred_s[t][j, k] - target_s[t][j, k]
        s_frob += d * d
    loss[None] += alpha_s * s_frob / NS_FLOAT

    F_frob = 0.0
    for j, k in ti.static(ti.ndrange(3, 3)):
        d = pred_F_mean[t][j, k] - target_F_mean[t][j, k]
        F_frob += d * d
    loss[None] += alpha_F * F_frob / NS_FLOAT


@ti.kernel
def compute_step_loss(t: ti.i32):
    """Full observable loss: h + 10*s + 5*F_mean."""
    for _ in range(1):
        _accum_step_loss(t, 1.0, 10.0, 5.0)


@ti.kernel
def compute_step_loss_weighted(t: ti.i32, alpha_h: float,
                               alpha_s: float, alpha_F: float):
    """Observable loss with configurable h, s, and F_mean weights."""
    for _ in range(1):
        _accum_step_loss(t, alpha_h, alpha_s, alpha_F)
