"""
gen_camera_target_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Run the MPM soft-body forward simulation once with known material parameters,
project particles through a perspective camera at each step, record camera-
space observables (2D projected mean, 2×2 covariance, mean depth, and
full projected positions), and save to ``camera_target_trajectory.npz``.

Shares configuration with the training pipeline via ``sim_config.cfg``.

Usage
-----
    python code/gen_camera_target_data.py
    python code/gen_camera_target_data.py --E 400 --nu 0.4
    python code/gen_camera_target_data.py --cam_pos 0.7,1.8,2.0 --cam_look 0.5,0.15,0.6
"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--E", type=float, default=400.0,
                    help="target Young's modulus")
parser.add_argument("--nu", type=float, default=0.4,
                    help="target Poisson ratio")
parser.add_argument("--out", type=str, default="camera_target_trajectory.npz",
                    help="output file name under data_camera/, or an absolute path")
parser.add_argument("--no_particles", action="store_true",
                    help="do not save per-step particle positions")
parser.add_argument("--warmup_steps", type=int, default=170,
                    help="simulation steps to run before recording target data")
parser.add_argument("--cam_pos", type=str, default="0.7,1.8,2.0",
                    help="camera position as x,y,z")
parser.add_argument("--cam_look", type=str, default="0.5,0.15,0.6",
                    help="camera look-at point as x,y,z")
parser.add_argument("--cam_fov", type=float, default=45.0,
                    help="camera vertical FOV in degrees")
args = parser.parse_args()

import taichi as ti
from sim_config import cfg, CAM_DATA_DIR

# True material params — what the inverse system tries to recover
E_true, nu_true = args.E, args.nu
mu_0_true = E_true / (2.0 * (1.0 + nu_true))
lambda_0_true = E_true * nu_true / ((1.0 + nu_true) * (1.0 - 2.0 * nu_true))

# Parse camera parameters
cam_pos = tuple(float(x) for x in args.cam_pos.split(","))
cam_look = tuple(float(x) for x in args.cam_look.split(","))

# No gradient tracking needed for forward pass
cfg.init_taichi()

# ──────────────────────────────────────────────────────────────────────────────
#  Import camera module after ti.init() so fields can be created
# ──────────────────────────────────────────────────────────────────────────────
from camera_module import Camera

# ──────────────────────────────────────────────────────────────────────────────
#  Taichi fields (forward-only — no needs_grad)
# ──────────────────────────────────────────────────────────────────────────────
x = ti.Vector.field(cfg.dim, dtype=float, shape=cfg.n_particles)
v = ti.Vector.field(cfg.dim, dtype=float, shape=cfg.n_particles)
C = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float, shape=cfg.n_particles)
F = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float, shape=cfg.n_particles)

grid_v = ti.Vector.field(cfg.dim, dtype=float, shape=(cfg.n_grid,) * 3)
grid_m = ti.field(dtype=float, shape=(cfg.n_grid,) * 3)


# ──────────────────────────────────────────────────────────────────────────────
#  SDF helpers (forward-only)
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
#  Deterministic particle init
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
#  MPM substep (forward-only)
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(CAM_DATA_DIR, exist_ok=True)
    init()

    # Create camera
    camera = Camera(position=cam_pos, lookat=cam_look,
                    fov_deg=args.cam_fov, aspect_ratio=cfg.cam_aspect_ratio,
                    n_particles=cfg.n_particles, needs_grad=False)

    print("Camera parameters:")
    print(f"  position: {cam_pos}")
    print(f"  lookat:   {cam_look}")
    print(f"  fov:      {args.cam_fov}°")
    print(f"  aspect:   {cfg.cam_aspect_ratio}")

    # ── Warm-up phase ──
    print("\nRunning warm-up simulation before recording target trajectory ...")
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

    # ── Recording phase ──
    # Camera-space observables
    proj_mean_list = []      # (n_steps, 2)
    proj_cov_list = []       # (n_steps, 2, 2)
    proj_depth_list = []     # (n_steps,)
    proj_2d_list = []        # (n_steps, n_particles, 2)  — absolute positions
    x_list = []              # (n_steps, n_particles, 3)  — 3D reference
    h_list = []              # (n_steps,) — 3D height for comparison
    n_valid_list = []        # (n_steps,) — visible particle count

    print("\nRunning forward simulation, recording camera projections ...")
    for step in range(cfg.n_steps):
        for _ in range(cfg.substeps_per_step):
            substep()

        pos_np = x.to_numpy()

        # Camera projection (NumPy path for data generation)
        screen_np, depth_np, valid_np = camera.project_numpy(pos_np)

        # Compute 2D statistics from visible particles
        mean_2d, cov_2d, n_valid = Camera.compute_2d_statistics(
            screen_np, valid_np)

        depth_mean = np.mean(depth_np[valid_np]) if n_valid > 0 else -1.0

        proj_mean_list.append(mean_2d)
        proj_cov_list.append(cov_2d)
        proj_depth_list.append(depth_mean)
        proj_2d_list.append(screen_np.astype(np.float32))
        n_valid_list.append(n_valid)

        # 3D reference
        if not args.no_particles:
            x_list.append(pos_np.astype(np.float32))
        h = np.mean(pos_np[:, 1])
        h_list.append(h)

        if step % 8 == 0:
            cov_trace = np.trace(cov_2d) if n_valid > 0 else 0.0
            print(f"  step {step:3d}/{cfg.n_steps}  "
                  f"h={h:.4f}  proj_mean=({mean_2d[0]:.4f},{mean_2d[1]:.4f})  "
                  f"tr(cov_2d)={cov_trace:.6f}  n_valid={n_valid}")

    # ── Pack and save ──
    proj_mean_arr = np.array(proj_mean_list, dtype=np.float32)
    proj_cov_arr = np.array(proj_cov_list, dtype=np.float32)
    proj_depth_arr = np.array(proj_depth_list, dtype=np.float32)
    proj_2d_arr = np.stack(proj_2d_list, axis=0).astype(np.float32)
    h_arr = np.array(h_list, dtype=np.float32)
    n_valid_arr = np.array(n_valid_list, dtype=np.int32)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(CAM_DATA_DIR, out_path)

    payload = dict(
        # camera-space observables
        proj_mean=proj_mean_arr,
        proj_cov=proj_cov_arr,
        proj_depth=proj_depth_arr,
        proj_2d=proj_2d_arr,
        n_valid=n_valid_arr,

        # 3D reference
        h=h_arr,

        # initial state
        x0=x0, v0=v0, C0=C0, F0=F0,

        # metadata
        E_true=E_true, nu_true=nu_true,
        dt=cfg.dt, substeps_per_step=cfg.substeps_per_step,
        n_steps=cfg.n_steps,
        warmup_steps=args.warmup_steps,
        cam_position=np.array(cam_pos, dtype=np.float32),
        cam_lookat=np.array(cam_look, dtype=np.float32),
        cam_fov=float(args.cam_fov),
        cam_aspect=float(cfg.cam_aspect_ratio),
    )
    if x_list:
        payload["x"] = np.stack(x_list, axis=0).astype(np.float32)

    np.savez(out_path, **payload)
    print(f"\nSaved camera target trajectory to {out_path}")
    print(f"  proj_mean:  {proj_mean_arr.shape}  (T × 2)")
    print(f"  proj_cov:   {proj_cov_arr.shape}   (T × 2 × 2)")
    print(f"  proj_depth: {proj_depth_arr.shape}  (T,)")
    print(f"  proj_2d:    {proj_2d_arr.shape}     (T × N × 2)")
    if x_list:
        print(f"  x (3D):     {payload['x'].shape}  (T × N × 3)")
    print(f"  saved warm-up state: x0/v0/C0/F0")
    print(f"  h range: [{h_arr.min():.4f}, {h_arr.max():.4f}]")
    print(f"  proj_mean X range: [{proj_mean_arr[:, 0].min():.4f}, "
          f"{proj_mean_arr[:, 0].max():.4f}]")
    print(f"  proj_mean Y range: [{proj_mean_arr[:, 1].min():.4f}, "
          f"{proj_mean_arr[:, 1].max():.4f}]")
    print(f"  min visible particles: {n_valid_arr.min()}/{cfg.n_particles}")
