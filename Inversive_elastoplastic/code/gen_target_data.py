"""
gen_target_data.py
~~~~~~~~~~~~~~~~~~
Run the MPM soft-body forward simulation once with known material parameters,
record per-timestep observables h (mean particle height), s (3×3 position
covariance), and F_mean (mean deformation gradient), and save to
target_trajectory.npz.

Shares configuration with the training pipeline via `sim_config.cfg`.
"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--yield_min", type=float, default=0.95,
                    help="target plasticity yield minimum constraint (e.g., 0.6 ~ 1.0)")
parser.add_argument("--E", type=float, default=400.0,
                    help="target Young's modulus")
parser.add_argument("--nu", type=float, default=0.4,
                    help="target Poisson ratio")
parser.add_argument("--out", type=str, default="target_trajectory.npz",
                    help="output file name under data/, or an absolute path")
parser.add_argument("--no_particles", action="store_true",
                    help="do not save per-step particle positions")
parser.add_argument("--warmup_steps", type=int, default=170,
                    help="simulation steps to run before recording target data")
args = parser.parse_args()

import taichi as ti
from sim_config import cfg, DATA_DIR

# True material params — what the inverse system tries to recover
yield_min_true = args.yield_min
E_true = args.E
nu_true = args.nu
mu_0_true = E_true / (2.0 * (1.0 + nu_true))
lambda_0_true = E_true * nu_true / ((1.0 + nu_true) * (1.0 - 2.0 * nu_true))

# No gradient tracking is needed for the forward pass, but using the shared
# initializer keeps CPU-only machines usable.
cfg.init_taichi()

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
@ti.func
def deterministic_unit(i, salt):
    phase = (ti.cast(i + 1, float) * (12.9898 + 17.23 * ti.cast(salt, float))
             + ti.cast(cfg.init_seed, float) * 0.12345)
    value = ti.sin(phase) * 43758.5453
    return value - ti.floor(value)


@ti.kernel
def init():
    for i in range(cfg.n_particles):
        x[i] = [
            cfg.init_base_x + deterministic_unit(i, 0) * cfg.init_extent,
            cfg.init_base_y + deterministic_unit(i, 1) * cfg.init_extent,
            cfg.init_base_z + deterministic_unit(i, 2) * cfg.init_extent,
        ]
        v[i] = [0.0, cfg.init_v_y, 0.0]
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

        F_trial = (ti.Matrix.identity(float, 3) + cfg.dt * C[p]) @ F[p]
        U, Sigma, V = ti.svd(F_trial, ti.f32)
        for i in ti.static(range(3)):
            Sigma[i, i] = ti.max(yield_min_true, ti.min(cfg.yield_max, Sigma[i, i]))
        F_e = U @ Sigma @ V.transpose()
        new_F = F_e
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
    os.makedirs(DATA_DIR, exist_ok=True)
    init()

    print("Running warm-up simulation before recording target trajectory ...")
    for step in range(args.warmup_steps):
        for _ in range(cfg.substeps_per_step):
            substep()
        if step % 10 == 0 or step == args.warmup_steps - 1:
            pos_np = x.to_numpy()
            print(f"  warmup {step + 1:3d}/{args.warmup_steps}  "
                  f"h={np.mean(pos_np[:, 1]):.4f}")

    x0 = x.to_numpy().astype(np.float32)
    v0 = v.to_numpy().astype(np.float32)
    C0 = C.to_numpy().astype(np.float32)
    F0 = F.to_numpy().astype(np.float32)

    h_list = []
    s_list = []
    F_list = []
    x_list = []

    print("Running forward simulation to generate target trajectory ...")
    for step in range(cfg.n_steps):
        for _ in range(cfg.substeps_per_step):
            substep()

        pos_np = x.to_numpy()
        if not args.no_particles:
            x_list.append(pos_np.astype(np.float32))
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

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(DATA_DIR, out_path)

    payload = dict(h=h_arr, s=s_arr, F_mean=F_arr,
                   x0=x0, v0=v0, C0=C0, F0=F0,
                   yield_min_true=yield_min_true, nu_true=nu_true,
                   dt=cfg.dt, substeps_per_step=cfg.substeps_per_step,
                   n_steps=cfg.n_steps,
                   warmup_steps=args.warmup_steps)
    if x_list:
        payload["x"] = np.stack(x_list, axis=0).astype(np.float32)

    np.savez(out_path, **payload)
    print(f"Saved target trajectory to {out_path}")
    print(f"  shape h: {h_arr.shape}, s: {s_arr.shape}, F_mean: {F_arr.shape}")
    if x_list:
        print(f"  shape x: {payload['x'].shape}")
    print(f"  saved warm-up state: x0/v0/C0/F0")
    print(f"  h range: [{h_arr.min():.4f}, {h_arr.max():.4f}]")
    print(f"  final trace(s): {np.trace(s_arr[-1]):.6f}")
    if np.max(np.abs(F_arr - np.eye(3, dtype=np.float32))) < 1e-4:
        print("  [WARN] F_mean stayed near identity; the recorded window may "
              "still be mostly pre-contact motion.")
