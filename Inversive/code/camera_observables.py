"""
camera_observables.py — camera-space trajectory observables and loss.

Creates fields for target/predicted camera observables (2D projected mean,
2×2 covariance, mean depth) and all split-kernel accumulation patterns.

Kernels accept the camera's per-particle projection fields as ``ti.template()``
arguments, so they work with any ``Camera`` instance without creating extra
scratch fields.

Depends on ``sim_config.cfg`` and ``mpm_sim.*`` for particle fields.

The split-kernel pattern (zero → accum → copy) avoids the cuModuleLoadDataEx
PTX crash observed in Taichi 1.7.x on Windows/CUDA.
"""
import taichi as ti
from sim_config import cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre-computed constants
# ═══════════════════════════════════════════════════════════════════════════════
NP_FLOAT = float(cfg.n_particles)
NS_FLOAT = float(cfg.n_steps)


# ═══════════════════════════════════════════════════════════════════════════════
#  Target camera observables (loaded from .npz, constant)
# ═══════════════════════════════════════════════════════════════════════════════
target_proj_mean = ti.Vector.field(2, dtype=float, shape=cfg.n_steps,
                                   needs_grad=False)
target_proj_cov = ti.Matrix.field(2, 2, dtype=float, shape=cfg.n_steps,
                                  needs_grad=False)
target_proj_depth = ti.field(dtype=float, shape=cfg.n_steps,
                              needs_grad=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Predicted camera observables (gradient-tracked)
# ═══════════════════════════════════════════════════════════════════════════════
pred_proj_mean = ti.Vector.field(2, dtype=float, shape=cfg.n_steps,
                                 needs_grad=True)
pred_proj_cov = ti.Matrix.field(2, 2, dtype=float, shape=cfg.n_steps,
                                needs_grad=True)
pred_proj_depth = ti.field(dtype=float, shape=cfg.n_steps,
                            needs_grad=True)

# 2D mean accumulator for split-kernel covariance computation
mean_2d_tmp = ti.Vector.field(2, dtype=float, shape=cfg.n_steps,
                               needs_grad=True)

# Camera-space loss accumulator
cam_loss = ti.field(dtype=float, shape=(), needs_grad=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Mean 2D projection kernels  (zero → accum → copy)
# ═══════════════════════════════════════════════════════════════════════════════
@ti.kernel
def zero_proj_mean_acc(t: ti.i32):
    """Zero the 2D mean accumulator."""
    for _ in range(1):
        mean_2d_tmp[t] = ti.Vector.zero(float, 2)


@ti.kernel
def accum_proj_mean(t: ti.i32, proj_2d: ti.template()):
    """Accumulate 2D projected positions from camera into global mean field.

    Parameters
    ----------
    t : int
        Time step index.
    proj_2d : ti.Vector.field(2, ...)
        Camera's per-particle 2D projection field (e.g. ``camera.proj_2d``).
    """
    for i in range(cfg.n_particles):
        mean_2d_tmp[t] += proj_2d[i] / NP_FLOAT


@ti.kernel
def copy_mean_to_proj(t: ti.i32):
    """Copy accumulated 2D mean to predicted observable."""
    for _ in range(1):
        pred_proj_mean[t] = mean_2d_tmp[t]


# ═══════════════════════════════════════════════════════════════════════════════
#  2D Covariance kernels  (zero → accum)
# ═══════════════════════════════════════════════════════════════════════════════
@ti.kernel
def zero_proj_cov_acc(t: ti.i32):
    """Zero the 2D covariance accumulator."""
    for _ in range(1):
        for j, k in ti.static(ti.ndrange(2, 2)):
            pred_proj_cov[t][j, k] = 0.0


@ti.kernel
def accum_proj_cov(t: ti.i32, proj_2d: ti.template()):
    """Accumulate 2×2 outer products into global covariance field.

    Parameters
    ----------
    t : int
        Time step index.
    proj_2d : ti.Vector.field(2, ...)
        Camera's per-particle 2D projection field.
    """
    for i in range(cfg.n_particles):
        for j, k in ti.static(ti.ndrange(2, 2)):
            pred_proj_cov[t][j, k] += (
                (proj_2d[i][j] - mean_2d_tmp[t][j])
                * (proj_2d[i][k] - mean_2d_tmp[t][k])
            ) / NP_FLOAT


# ═══════════════════════════════════════════════════════════════════════════════
#  Depth mean kernel
# ═══════════════════════════════════════════════════════════════════════════════
@ti.kernel
def zero_depth_acc(t: ti.i32):
    """Zero the depth accumulator."""
    for _ in range(1):
        pred_proj_depth[t] = 0.0


@ti.kernel
def accum_depth(t: ti.i32, proj_depth: ti.template()):
    """Accumulate per-particle depth into mean depth.

    Parameters
    ----------
    t : int
        Time step index.
    proj_depth : ti.field(float, ...)
        Camera's per-particle depth field (e.g. ``camera.proj_depth``).
    """
    for i in range(cfg.n_particles):
        pred_proj_depth[t] += proj_depth[i] / NP_FLOAT


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera-space loss
# ═══════════════════════════════════════════════════════════════════════════════
@ti.kernel
def compute_camera_step_loss(t: ti.i32):
    """MSE from step t:  α_mean·||proj_mean_err||² + α_cov·||cov_err||²_F.

    The loss is normalised by n_steps so the total is an average over time.
    Weights are chosen to balance the typical magnitude of mean (in [0,1])
    and covariance-trace (typically ≪ 1) terms.
    """
    for _ in range(1):
        alpha_mean = 100.0
        alpha_cov = 500.0
        alpha_depth = 10.0

        # 2D mean error
        mean_sq = 0.0
        for j in ti.static(range(2)):
            d = pred_proj_mean[t][j] - target_proj_mean[t][j]
            mean_sq += d * d
        cam_loss[None] += alpha_mean * mean_sq / NS_FLOAT

        # 2D covariance Frobenius error
        cov_frob = 0.0
        for j, k in ti.static(ti.ndrange(2, 2)):
            d = pred_proj_cov[t][j, k] - target_proj_cov[t][j, k]
            cov_frob += d * d
        cam_loss[None] += alpha_cov * cov_frob / NS_FLOAT

        # depth mean error
        d_depth = pred_proj_depth[t] - target_proj_depth[t]
        cam_loss[None] += alpha_depth * d_depth * d_depth / NS_FLOAT


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: run all camera-observation kernels for one step
# ═══════════════════════════════════════════════════════════════════════════════
def run_camera_obs_kernels(step, camera):
    """Run the full camera observable pipeline for one simulation step.

    Parameters
    ----------
    step : int
        Current simulation step index.
    camera : Camera
        Camera instance whose ``proj_2d`` and ``proj_depth`` fields have
        already been filled by ``camera.project_kernel(sim.x)``.
    """
    zero_proj_mean_acc(step)
    accum_proj_mean(step, camera.proj_2d)
    copy_mean_to_proj(step)
    zero_proj_cov_acc(step)
    accum_proj_cov(step, camera.proj_2d)
    zero_depth_acc(step)
    accum_depth(step, camera.proj_depth)
