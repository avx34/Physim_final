"""
observables.py — trajectory observables and loss computation.

Creates fields for target/predicted observables (h, s, F_mean) and all
split-kernel accumulation patterns that avoid the cuModuleLoadDataEx PTX crash.

Depends on `sim_config.cfg` and `mpm_sim.*` for particle fields.
"""
import taichi as ti
from sim_config import cfg
import mpm_sim as sim   # provides x, F


# ==============================================================================
#  Pre-computed constants
# ==============================================================================
NP_FLOAT = float(cfg.n_particles)
NS_FLOAT = float(cfg.n_steps)


# ==============================================================================
#  Target observables (loaded from .npz, constant)
# ==============================================================================
target_h = ti.field(dtype=float, shape=cfg.n_steps, needs_grad=False)
target_s = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float,
                           shape=cfg.n_steps, needs_grad=False)
target_F_mean = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float,
                                shape=cfg.n_steps, needs_grad=False)


# ==============================================================================
#  Predicted observables (gradient-tracked)
# ==============================================================================
pred_h = ti.field(dtype=float, shape=cfg.n_steps, needs_grad=True)
pred_s = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float,
                         shape=cfg.n_steps, needs_grad=True)
pred_F_mean = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float,
                              shape=cfg.n_steps, needs_grad=True)

# Global accumulator to pass mean vector between mean kernels and cov/F kernels.
# This avoids local-accumulator patterns that trigger cuModuleLoadDataEx crashes.
mean_tmp = ti.Vector.field(cfg.dim, dtype=float,
                           shape=cfg.n_steps, needs_grad=True)

# Loss accumulator
loss = ti.field(dtype=float, shape=(), needs_grad=True)


# ==============================================================================
#  Mean height kernels  (zero → accum → copy)
# ==============================================================================
@ti.kernel
def zero_mean_acc(t: ti.i32):
    """Zero the mean accumulator (for _ in range(1) for AD safety)."""
    for _ in range(1):
        mean_tmp[t] = ti.Vector.zero(float, 3)


@ti.kernel
def accum_mean(t: ti.i32):
    """Accumulate particle positions into global field (PURE LOOP)."""
    for i in range(cfg.n_particles):
        mean_tmp[t] += sim.x[i] / NP_FLOAT


@ti.kernel
def copy_mean_to_h(t: ti.i32):
    """Extract height component from mean accumulator."""
    for _ in range(1):
        pred_h[t] = mean_tmp[t][1]


# ==============================================================================
#  Covariance kernels  (zero → accum)
# ==============================================================================
@ti.kernel
def zero_cov_acc(t: ti.i32):
    """Zero the covariance accumulator."""
    for _ in range(1):
        for j, k in ti.static(ti.ndrange(3, 3)):
            pred_s[t][j, k] = 0.0


@ti.kernel
def accum_cov(t: ti.i32):
    """Accumulate outer products into global covariance field (PURE LOOP)."""
    for i in range(cfg.n_particles):
        for j, k in ti.static(ti.ndrange(3, 3)):
            pred_s[t][j, k] += ((sim.x[i][j] - mean_tmp[t][j])
                                * (sim.x[i][k] - mean_tmp[t][k])) / NP_FLOAT


# ==============================================================================
#  Mean deformation gradient kernels  (zero → accum)
# ==============================================================================
@ti.kernel
def zero_F_acc(t: ti.i32):
    """Zero the mean-F accumulator."""
    for _ in range(1):
        for j, k in ti.static(ti.ndrange(3, 3)):
            pred_F_mean[t][j, k] = 0.0


@ti.kernel
def accum_F(t: ti.i32):
    """Accumulate deformation gradients (PURE LOOP)."""
    for i in range(cfg.n_particles):
        for j, k in ti.static(ti.ndrange(3, 3)):
            pred_F_mean[t][j, k] += sim.F[i][j, k] / NP_FLOAT


# ==============================================================================
#  Loss
# ==============================================================================
@ti.kernel
def compute_step_loss(t: ti.i32):
    """MSE from step t:  h-err + α_s·||s_err||² + α_F·||F_err||²."""
    for _ in range(1):
        alpha_s = 10.0
        alpha_F = 5.0

        h_diff = pred_h[t] - target_h[t]
        loss[None] += h_diff * h_diff / NS_FLOAT

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
