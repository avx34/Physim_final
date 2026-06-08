"""
gen_target_data.py
~~~~~~~~~~~~~~~~~~
Run the MPM soft-body forward simulation once with known material parameters,
record per-timestep observables h (mean particle height), s (3×3 position
covariance), and F_mean (mean deformation gradient), and save to
target_trajectory.npz.

Shares configuration with the training pipeline via `sim_config.cfg`.
"""
import os
import numpy as np
import taichi as ti
from sim_config import cfg, DATA_DIR

# True material params — what the inverse system tries to recover
E_true, nu_true = 400.0, 0.4
mu_0_true = E_true / (2.0 * (1.0 + nu_true))
lambda_0_true = E_true * nu_true / ((1.0 + nu_true) * (1.0 - 2.0 * nu_true))

# Init with pure GPU (no gradient tracking needed for forward pass)
ti.init(arch=ti.gpu, random_seed=42)

# ---------------------------------------------------------------------------
# Taichi fields (no needs_grad — forward-only)
# ---------------------------------------------------------------------------
x = ti.Vector.field(cfg.dim, dtype=float, shape=cfg.n_particles)
v = ti.Vector.field(cfg.dim, dtype=float, shape=cfg.n_particles)
C = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float, shape=cfg.n_particles)
F = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float, shape=cfg.n_particles)

grid_v = ti.Vector.field(cfg.dim, dtype=float, shape=(cfg.n_grid,) * 3)
grid_m = ti.field(dtype=float, shape=(cfg.n_grid,) * 3)

# ---------------------------------------------------------------------------
# SDF helpers (forward-only copies — no gradient needed)
# ---------------------------------------------------------------------------
@ti.func
def box_sdf(p, c, e):
    d = ti.abs(p - c) - e
    out_dist = ti.math.length(ti.max(d, 0.0))
    in_dist = ti.min(ti.max(d[0], ti.max(d[1], d[2])), 0.0)
    return out_dist + in_dist


@ti.func
def get_sdf(p):
    d = p[1] - cfg.ground_y
    wall_lx = p[0] - (-0.45)
    wall_rx = 1.45 - p[0]
    wall_lz = p[2] - (-0.45)
    wall_rz = 1.45 - p[2]
    c1 = ti.Vector([0.5, 0.15, 0.7])
    e1 = ti.Vector([0.3, 0.05, 0.1])
    c2 = ti.Vector([0.5, 0.25, 0.5])
    e2 = ti.Vector([0.3, 0.05, 0.1])
    res = d
    res = ti.min(res, box_sdf(p, c1, e1))
    res = ti.min(res, box_sdf(p, c2, e2))
    res = ti.min(res, wall_lx)
    res = ti.min(res, wall_rx)
    res = ti.min(res, wall_lz)
    res = ti.min(res, wall_rz)
    return res


@ti.func
def get_sdf_normal(p):
    eps = 1e-4
    dx_val = get_sdf(p + ti.Vector([eps, 0.0, 0.0])) \
           - get_sdf(p - ti.Vector([eps, 0.0, 0.0]))
    dy_val = get_sdf(p + ti.Vector([0.0, eps, 0.0])) \
           - get_sdf(p - ti.Vector([0.0, eps, 0.0]))
    dz_val = get_sdf(p + ti.Vector([0.0, 0.0, eps])) \
           - get_sdf(p - ti.Vector([0.0, 0.0, eps]))
    n = ti.Vector([dx_val, dy_val, dz_val])
    length = n.norm()
    if length == 0:
        n = ti.Vector([0.0, 1.0, 0.0])
    else:
        n = n / length
    return n


# ---------------------------------------------------------------------------
# MPM substep (forward-only)
# ---------------------------------------------------------------------------
@ti.kernel
def init():
    for i in range(cfg.n_particles):
        x[i] = [ti.random() * 0.16 + 0.42,
                ti.random() * 0.16 + 0.12,
                ti.random() * 0.16 + 0.42]
        v[i] = [0.0, -8.0, 0.0]
        F[i] = ti.Matrix.identity(float, 3)
        C[i] = ti.Matrix.zero(float, 3, 3)


@ti.kernel
def substep():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0

    for p in x:
        pos = x[p] + ti.Vector([0.5, 0.5, 0.5])
        base = (pos * cfg.inv_dx - 0.5).cast(int)
        fx = pos * cfg.inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2]

        new_F = (ti.Matrix.identity(float, 3) + cfg.dt * C[p]) @ F[p]
        F[p] = new_F

        J = new_F.determinant()
        R, _ = ti.polar_decompose(new_F, ti.f32)

        cauchy = (2.0 * mu_0_true * (new_F - R) @ new_F.transpose()
                  + ti.Matrix.identity(float, 3) * lambda_0_true * J * (J - 1.0))
        stress = -(cfg.dt * cfg.p_vol * 4.0 * cfg.inv_dx * cfg.inv_dx) * cauchy
        affine = stress + cfg.p_mass * C[p]

        sdf = get_sdf(x[p])
        penalty_force = ti.Vector.zero(float, 3)
        if sdf < 0.0:
            n = get_sdf_normal(x[p])
            f_n = -sdf * cfg.penalty_k * cfg.p_mass
            v_n = v[p].dot(n)
            if v_n < 0.0:
                f_n -= v_n * cfg.penalty_damp * cfg.p_mass
            penalty_force = f_n * n

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * cfg.dx
            weight = w[i][0] * w[j][1] * w[k][2]
            momentum = weight * (cfg.p_mass * v[p] + affine @ dpos
                                 + penalty_force * cfg.dt)
            grid_v[base + offset] += momentum
            grid_m[base + offset] += weight * cfg.p_mass

    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0.0:
            grid_v[i, j, k] = (1.0 / grid_m[i, j, k]) * grid_v[i, j, k]
            grid_v[i, j, k][1] -= cfg.dt * 9.8

    for p in x:
        pos = x[p] + ti.Vector([0.5, 0.5, 0.5])
        base = (pos * cfg.inv_dx - 0.5).cast(int)
        fx = pos * cfg.inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2]

        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = offset.cast(float) - fx
            weight = w[i][0] * w[j][1] * w[k][2]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4.0 * cfg.inv_dx * weight * g_v.outer_product(dpos)

        v[p] = new_v
        C[p] = new_C
        x[p] += cfg.dt * v[p]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init()

    h_list = []
    s_list = []
    F_list = []

    print("Running forward simulation to generate target trajectory ...")
    for step in range(cfg.n_steps):
        for _ in range(cfg.substeps_per_step):
            substep()

        pos_np = x.to_numpy()
        h = np.mean(pos_np[:, 1])
        s = np.cov(pos_np.T)
        h_list.append(h)
        s_list.append(s)

        F_np = F.to_numpy()
        mean_F = np.mean(F_np, axis=0)
        F_list.append(mean_F)

        if step % 8 == 0:
            print(f"  step {step:3d}/{cfg.n_steps}  h={h:.4f}  "
                  f"trace(s)={np.trace(s):.6f}  "
                  f"det(F_mean)={np.linalg.det(mean_F):.4f}")

    h_arr = np.array(h_list, dtype=np.float32)
    s_arr = np.array(s_list, dtype=np.float32)
    F_arr = np.array(F_list, dtype=np.float32)

    out_path = os.path.join(DATA_DIR, "target_trajectory.npz")
    np.savez(out_path, h=h_arr, s=s_arr, F_mean=F_arr,
             E_true=E_true, nu_true=nu_true,
             dt=cfg.dt, substeps_per_step=cfg.substeps_per_step,
             n_steps=cfg.n_steps)
    print(f"Saved target trajectory to {out_path}")
    print(f"  shape h: {h_arr.shape}, s: {s_arr.shape}, F_mean: {F_arr.shape}")
    print(f"  h range: [{h_arr.min():.4f}, {h_arr.max():.4f}]")
    print(f"  final trace(s): {np.trace(s_arr[-1]):.6f}")
