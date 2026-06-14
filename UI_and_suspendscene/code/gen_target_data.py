"""
gen_target_data.py
完全对齐用户提供的悬挂Demo物理逻辑，无自定义颜色修改，重力0.5，初始化/约束/SDF1:1复刻
"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--E", type=float, default=800.0,
                    help="target Young's modulus")
parser.add_argument("--nu", type=float, default=0.4,
                    help="target Poisson ratio")
parser.add_argument("--out", type=str, default="hang_target.npz",
                    help="output file name under data/, or an absolute path")
parser.add_argument("--no_particles", action="store_true",
                    help="do not save per-step particle positions")
parser.add_argument("--warmup_steps", type=int, default=170,
                    help="simulation steps to run before recording target data")
parser.add_argument("--noise_h", type=float, default=0.0,
                    help="Gaussian std added to recorded mean height h")
parser.add_argument("--noise_s", type=float, default=0.0,
                    help="Gaussian std added to recorded covariance s")
parser.add_argument("--noise_F", type=float, default=0.0,
                    help="Gaussian std added to recorded mean deformation F")
parser.add_argument("--noise_seed", type=int, default=0,
                    help="random seed for target-observation noise")
parser.add_argument("--visual_check", action="store_true",
                    help="Pop Taichi window to preview hanging scene, skip saving npz")
args = parser.parse_args()

import taichi as ti
# 完全复用你Demo里的全局固定参数，不再依赖sim_config，彻底对齐
ti.init(arch=ti.gpu)
dim = 3
n_particles = 4096
n_grid = 128
dx = 2.0 / n_grid
inv_dx = 1 / dx
dt = 1e-4
substeps = int(2e-3 // dt)

p_vol = (dx * 0.5)**3
p_rho = 1
p_mass = p_vol * p_rho

ground_y = 0.1
penalty_k = 1e5
penalty_damp = 2e3

# 材质场
E_ctrl = ti.field(dtype=ti.f32, shape=())
nu_ctrl = ti.field(dtype=ti.f32, shape=())
E_ctrl[None] = args.E
nu_ctrl[None] = args.nu

# 粒子场 完全和Demo一致
x = ti.Vector.field(dim, dtype=float, shape=n_particles)
v = ti.Vector.field(dim, dtype=float, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
x_init = ti.Vector.field(dim, dtype=float, shape=n_particles)
particle_colors = ti.Vector.field(dim, dtype=float, shape=n_particles)
is_fixed = ti.field(dtype=int, shape=n_particles)

grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))

box_min = np.array([-0.45, 0.05, -0.45], dtype=np.float32)
box_max = np.array([1.45, 1.45, 1.45], dtype=np.float32)

# 场景网格构建函数 原样复制
def get_box_mesh(c, e):
    v = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                v.append([c[0] + (2*i-1)*e[0], c[1] + (2*j-1)*e[1], c[2] + (2*k-1)*e[2]])
    idx = [0,1,3, 0,3,2, 4,6,7, 4,7,5, 0,4,5, 0,5,1, 2,3,7, 2,7,6, 0,2,6, 0,6,4, 1,5,7, 1,7,3]
    return np.array(v, dtype=np.float32), np.array(idx, dtype=np.int32)

v_list, i_list = [], []
offset = 0
for cx, ex in [
    ([0.5, 0.05, 0.5], [1.0, 0.05, 1.0]),
    ([0.5, 0.15, 0.7], [0.3, 0.05, 0.1]),
    ([0.5, 0.25, 0.5], [0.3, 0.05, 0.1]),
]:
    bv, bi = get_box_mesh(cx, ex)
    v_list.append(bv)
    i_list.append(bi + offset)
    offset += 8
scene_verts_np = np.concatenate(v_list)
scene_inds_np = np.concatenate(i_list)
scene_verts = ti.Vector.field(3, dtype=float, shape=scene_verts_np.shape[0])
scene_inds = ti.field(dtype=int, shape=scene_inds_np.shape[0])
scene_verts.from_numpy(scene_verts_np)
scene_inds.from_numpy(scene_inds_np)

# SDF函数 完全原样复制
@ti.func
def box_sdf(p, c, e):
    d = ti.abs(p - c) - e
    out_dist = ti.math.length(ti.max(d, 0.0))
    in_dist = ti.min(ti.max(d[0], ti.max(d[1], d[2])), 0.0)
    return out_dist + in_dist

@ti.func
def get_sdf(p):
    d = p[1] - 0.1
    c1 = ti.Vector([0.5, 0.15, 0.7])
    e1 = ti.Vector([0.3, 0.05, 0.1])
    d1 = box_sdf(p, c1, e1)
    c2 = ti.Vector([0.5, 0.25, 0.5])
    e2 = ti.Vector([0.3, 0.05, 0.1])
    d2 = box_sdf(p, c2, e2)
    return ti.min(ti.min(d, d1), d2)

@ti.func
def get_sdf_normal(p):
    eps = 1e-4
    dx = get_sdf(p + ti.Vector([eps, 0.0, 0.0])) - get_sdf(p - ti.Vector([eps, 0.0, 0.0]))
    dy = get_sdf(p + ti.Vector([0.0, eps, 0.0])) - get_sdf(p - ti.Vector([0.0, eps, 0.0]))
    dz = get_sdf(p + ti.Vector([0.0, 0.0, eps])) - get_sdf(p - ti.Vector([0.0, 0.0, eps]))
    n = ti.Vector([dx, dy, dz])
    len_n = n.norm()
    return n / len_n if len_n > 0 else ti.Vector([0.0, 1.0, 0.0])

# 粒子初始化内核 原样复制
@ti.kernel
def init_particles_raw():
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.2 + 0.4,
            ti.random() * 0.2 + 0.9,
            ti.random() * 0.2 + 0.45
        ]
        v[i] = [0.0, 0.0, 0.0]
        F[i] = ti.Matrix.identity(float, 3)
        C[i] = ti.Matrix.zero(float, 3, 3)
        is_fixed[i] = 0
        particle_colors[i] = ti.Vector([0.06, 0.85, 0.87])

# 初始化两角固定逻辑 完全照搬你的代码
def init():
    init_particles_raw()
    pos_np = x.to_numpy()

    x_min, x_max = pos_np[:, 0].min(), pos_np[:, 0].max()
    y_max = pos_np[:, 1].max()
    z_min = pos_np[:, 2].min()

    back_left_corner = np.array([x_min, y_max, z_min])
    back_right_corner = np.array([x_max, y_max, z_min])

    dist_to_left = np.linalg.norm(pos_np - back_left_corner, axis=1)
    dist_to_right = np.linalg.norm(pos_np - back_right_corner, axis=1)

    left_fixed_indices = np.argsort(dist_to_left)[:100]
    right_fixed_indices = np.argsort(dist_to_right)[:100]
    fixed_indices = np.concatenate([left_fixed_indices, right_fixed_indices])

    fixed_mask = np.zeros(n_particles, dtype=np.int32)
    fixed_mask[fixed_indices] = 1
    is_fixed.from_numpy(fixed_mask)

    # 全程统一青色，不修改任何粒子颜色
    colors_np = np.tile(np.array([0.06, 0.85, 0.87]), (n_particles, 1))
    particle_colors.from_numpy(colors_np.astype(np.float32))
    x_init.copy_from(x)

# 边界约束函数原样复制
@ti.kernel
def confine_particles(min_x: float, min_y: float, min_z: float, max_x: float, max_y: float, max_z: float):
    for i in range(n_particles):
        if is_fixed[i] == 1:
            continue
        for axis in ti.static(range(3)):
            bounds_min = [min_x, min_y, min_z]
            bounds_max = [max_x, max_y, max_z]
            if x[i][axis] < bounds_min[axis]:
                x[i][axis] = bounds_min[axis]
                if v[i][axis] < 0:
                    v[i][axis] = 0.0
            elif x[i][axis] > bounds_max[axis]:
                x[i][axis] = bounds_max[axis]
                if v[i][axis] > 0:
                    v[i][axis] = 0.0

# 核心substep 100%复刻，重力dt*0.5不变
@ti.kernel
def substep():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0

    E_now = E_ctrl[None]
    nu_now = ti.min(nu_ctrl[None], 0.495)
    mu_0_now = E_now / (2 * (1 + nu_now))
    lambda_0_now = E_now * nu_now / ((1 + nu_now) * (1 - 2 * nu_now))

    # P2G
    for p in x:
        pos = x[p] + ti.Vector([0.5, 0.5, 0.5])
        base = (pos * inv_dx - 0.5).cast(int)
        fx = pos * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        new_F = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]
        F[p] = new_F

        J = new_F.determinant()
        R, S = ti.polar_decompose(new_F, ti.f32)
        cauchy = 2 * mu_0_now * (new_F - R) @ new_F.transpose() + ti.Matrix.identity(float, 3) * lambda_0_now * J * (J - 1)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + p_mass * C[p]

        sdf = get_sdf(x[p])
        penalty_force = ti.Vector.zero(float, 3)
        if sdf < 0:
            normal = get_sdf_normal(x[p])
            f_n = -sdf * penalty_k * p_mass
            v_n = v[p].dot(normal)
            if v_n < 0:
                f_n -= v_n * penalty_damp * p_mass
            penalty_force = f_n * normal

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            momentum_inc = weight * (p_mass * v[p] + affine @ dpos + penalty_force * dt)
            grid_v[base + offset] += momentum_inc
            grid_m[base + offset] += weight * p_mass

    # 重力严格0.5，和你Demo一致
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]
        grid_v[i, j, k][1] -= dt * 0.5

    # G2P + 固定粒子锁死逻辑完全照搬
    for p in x:
        pos = x[p] + ti.Vector([0.5, 0.5, 0.5])
        base = (pos * inv_dx - 0.5).cast(int)
        fx = pos * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = offset.cast(float) - fx
            weight = w[i][0] * w[j][1] * w[k][2]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        if is_fixed[p] == 1:
            v[p] = ti.Vector([0.0, 0.0, 0.0])
            C[p] = ti.Matrix.zero(float, 3, 3)
            x[p] = x_init[p]
            F[p] = ti.Matrix.identity(float, 3)
        else:
            v[p] = new_v
            C[p] = new_C
            x[p] += dt * v[p]

# 噪声处理函数不变
def add_observation_noise(h_arr, s_arr, F_arr):
    if args.noise_h == 0.0 and args.noise_s == 0.0 and args.noise_F == 0.0:
        return h_arr.copy(), s_arr.copy(), F_arr.copy()
    rng = np.random.default_rng(args.noise_seed)
    h_noisy = h_arr.astype(np.float32).copy()
    s_noisy = s_arr.astype(np.float32).copy()
    F_noisy = F_arr.astype(np.float32).copy()
    if args.noise_h > 0.0:
        h_noisy += rng.normal(0.0, args.noise_h, size=h_noisy.shape).astype(np.float32)
    if args.noise_s > 0.0:
        raw = rng.normal(0.0, args.noise_s, size=s_noisy.shape).astype(np.float32)
        sym = 0.5 * (raw + np.swapaxes(raw, -1, -2))
        s_noisy += sym
    if args.noise_F > 0.0:
        F_noisy += rng.normal(0.0, args.noise_F, size=F_noisy.shape).astype(np.float32)
    return h_noisy, s_noisy, F_noisy

# 可视化校验窗口：完全使用原始统一青色，不改动粒子颜色
def visual_scene_check():
    window = ti.ui.Window("Verify Exact Hang Scene", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    cam_target = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    cam_pos0 = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    cam_radius = float(np.linalg.norm(cam_pos0 - cam_target))
    cam_yaw = float(np.arctan2(cam_pos0[0]-cam_target[0], cam_pos0[2]-cam_target[2]))
    cam_pitch = float(np.arcsin((cam_pos0[1]-cam_target[1]) / cam_radius))
    cam_fov = 45.0
    last_rmb = None

    print("Visual check: same color for all particles, right mouse drag rotate")
    while window.running:
        mx, my = window.get_cursor_pos()
        if window.is_pressed(ti.ui.RMB):
            if last_rmb is None:
                last_rmb = (mx, my)
            else:
                cam_yaw += (mx - last_rmb[0]) * 4.0
                cam_pitch = max(-1.35, min(1.35, cam_pitch + (my - last_rmb[1]) * 4.0))
                last_rmb = (mx, my)
        else:
            last_rmb = None

        camera.position(
            cam_target[0] + cam_radius * np.sin(cam_yaw) * np.cos(cam_pitch),
            cam_target[1] + cam_radius * np.sin(cam_pitch),
            cam_target[2] + cam_radius * np.cos(cam_yaw) * np.cos(cam_pitch)
        )
        camera.lookat(cam_target[0], cam_target[1], cam_target[2])
        camera.fov(cam_fov)

        # 完整仿真步进，和主逻辑一致
        for _ in range(substeps):
            substep()
            confine_particles(box_min[0], box_min[1], box_min[2], box_max[0], box_max[1], box_max[2])

        scene.set_camera(camera)
        scene.mesh(scene_verts, indices=scene_inds, color=(0.4, 0.4, 0.45))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1,1,1))
        scene.ambient_light((0.5, 0.5, 0.5))
        # 沿用原始particle_colors，全部青色，无变色
        scene.particles(x, radius=0.006, per_vertex_color=particle_colors)
        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    init()

    # 可视化校验分支
    if args.visual_check:
        visual_scene_check()
        exit()

    # 预热仿真
    print(f"Warmup {args.warmup_steps} steps")
    for step in range(args.warmup_steps):
        for _ in range(substeps):
            substep()
            confine_particles(box_min[0], box_min[1], box_min[2], box_max[0], box_max[1], box_max[2])
        if step % 10 == args.warmup_steps % 10:
            pos_np = x.to_numpy()
            print(f"Warm step {step+1} mean h: {np.mean(pos_np[:,1]):.4f}")

    # 保存初始状态
    x0 = x.to_numpy().astype(np.float32)
    v0 = v.to_numpy().astype(np.float32)
    C0 = C.to_numpy().astype(np.float32)
    F0 = F.to_numpy().astype(np.float32)

    h_list, s_list, F_list, x_list = [], [], [], []
    print("Recording target trajectory frames")
    # 这里固定记录帧数200，如需修改直接改range(200)
    total_record_frames = 200
    for frame in range(total_record_frames):
        for _ in range(substeps):
            substep()
            confine_particles(box_min[0], box_min[1], box_min[2], box_max[0], box_max[1], box_max[2])
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
        if frame % 8 == 0:
            print(f"Rec frame {frame}/{total_record_frames-1} h={h:.4f} trace(s)={np.trace(s):.6f} det(F)={np.linalg.det(mean_F):.4f}")

    # 打包数据
    h_arr = np.array(h_list, dtype=np.float32)
    s_arr = np.array(s_list, dtype=np.float32)
    F_arr = np.array(F_list, dtype=np.float32)
    h_clean, s_clean, F_clean = h_arr.copy(), s_arr.copy(), F_arr.copy()
    h_arr, s_arr, F_arr = add_observation_noise(h_arr, s_arr, F_arr)

    out_path = args.out
    payload = {
        "h": h_arr, "s": s_arr, "F_mean": F_arr,
        "h_clean": h_clean, "s_clean": s_clean, "F_mean_clean": F_clean,
        "x0":x0, "v0":v0, "C0":C0, "F0":F0,
        "E_true": args.E, "nu_true": args.nu,
        "dt": dt, "substeps_per_step": substeps,
        "n_steps": total_record_frames,
        "warmup_steps": args.warmup_steps,
        "noise_h":args.noise_h, "noise_s":args.noise_s, "noise_F":args.noise_F,
        "noise_seed":args.noise_seed
    }
    if x_list:
        payload["x"] = np.stack(x_list, axis=0).astype(np.float32)
    np.savez(os.path.join("data", out_path), **payload)
    print(f"Saved target file to data/{out_path}")
    print(f"Recorded frames count: {total_record_frames}")