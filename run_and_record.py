"""
运行 MPM 软体仿真，每秒保存一张摄像机视角的照片（matplotlib 渲染）。

用法:  python run_and_record.py [--duration 30] [--out frames/]
"""
import argparse
import os
import time
import taichi as ti
import numpy as np

# ── 参数 ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=4.0,
                    help="仿真物理时长（秒），默认 4")
parser.add_argument("--out", type=str, default="frames", help="输出目录")
parser.add_argument("--dpi", type=int, default=120, help="输出图像分辨率")
args = parser.parse_args()

ti.init(arch=ti.gpu)

dim = 3
n_particles = 4096
n_grid = 128
dx = 2.0 / n_grid
inv_dx = 1 / dx
dt = 1e-4
substeps = int(2e-3 // dt)         # 20 substeps/render-frame

p_vol = (dx * 0.5) ** 3
p_rho = 1
p_mass = p_vol * p_rho

E, nu = 400, 0.4
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

ground_y = 0.1
penalty_k = 1e5
penalty_damp = 2e3

# 仿真时间参数
sim_dt_per_frame = dt * substeps    # 0.002 s per render frame
capture_interval = 0.1              # 每 0.1 秒拍一帧
render_frames_per_capture = int(capture_interval / sim_dt_per_frame)  # 100 帧 = 0.2 秒
total_captures = 20                 # 总共拍 20 张

# ── 粒子 & 网格字段 ──────────────────────────────────────────
x = ti.Vector.field(dim, dtype=float, shape=n_particles)
v = ti.Vector.field(dim, dtype=float, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)

grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))


# ── SDF ──────────────────────────────────────────────────────
@ti.func
def box_sdf(p, c, e):
    d = ti.abs(p - c) - e
    out_dist = ti.math.length(ti.max(d, 0.0))
    in_dist = ti.min(ti.max(d[0], ti.max(d[1], d[2])), 0.0)
    return out_dist + in_dist


@ti.func
def get_sdf(p):
    d = p[1] - 0.1
    wall_lx = p[0] - (-0.45)
    wall_rx = 1.45 - p[0]
    wall_lz = p[2] - (-0.45)
    wall_rz = 1.45 - p[2]
    c1 = ti.Vector([0.5, 0.15, 0.7])
    e1 = ti.Vector([0.3, 0.05, 0.1])
    d1 = box_sdf(p, c1, e1)
    c2 = ti.Vector([0.5, 0.25, 0.5])
    e2 = ti.Vector([0.3, 0.05, 0.1])
    d2 = box_sdf(p, c2, e2)
    res = d
    res = ti.min(res, d1)
    res = ti.min(res, d2)
    res = ti.min(res, wall_lx)
    res = ti.min(res, wall_rx)
    res = ti.min(res, wall_lz)
    res = ti.min(res, wall_rz)
    return res


@ti.func
def get_sdf_normal(p):
    eps = 1e-4
    dx_v = (get_sdf(p + ti.Vector([eps, 0.0, 0.0]))
            - get_sdf(p - ti.Vector([eps, 0.0, 0.0])))
    dy_v = (get_sdf(p + ti.Vector([0.0, eps, 0.0]))
            - get_sdf(p - ti.Vector([0.0, eps, 0.0])))
    dz_v = (get_sdf(p + ti.Vector([0.0, 0.0, eps]))
            - get_sdf(p - ti.Vector([0.0, 0.0, eps])))
    n = ti.Vector([dx_v, dy_v, dz_v])
    len_n = n.norm()
    if len_n == 0:
        n = ti.Vector([0.0, 1.0, 0.0])
    else:
        n = n / len_n
    return n


# ── MPM 积分 ─────────────────────────────────────────────────
@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.2 + 0.4,
                ti.random() * 0.2 + 0.9,
                ti.random() * 0.2 + 0.45]
        v[i] = [0, -2.0, 0]
        F[i] = ti.Matrix.identity(float, 3)
        C[i] = ti.Matrix.zero(float, 3, 3)


@ti.kernel
def substep():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0

    for p in x:
        pos = x[p] + ti.Vector([0.5, 0.5, 0.5])
        base = (pos * inv_dx - 0.5).cast(int)
        fx = pos * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1) ** 2,
             0.5 * (fx - 0.5) ** 2]

        new_F = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]
        F[p] = new_F

        J = new_F.determinant()
        R, S = ti.polar_decompose(new_F, ti.f32)
        cauchy = (2 * mu_0 * (new_F - R) @ new_F.transpose()
                  + ti.Matrix.identity(float, 3) * lambda_0 * J * (J - 1))
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
            momentum_inc = weight * (p_mass * v[p] + affine @ dpos
                                     + penalty_force * dt)
            grid_v[base + offset] += momentum_inc
            grid_m[base + offset] += weight * p_mass

    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]
            grid_v[i, j, k][1] -= dt * 9.8

    for p in x:
        pos = x[p] + ti.Vector([0.5, 0.5, 0.5])
        base = (pos * inv_dx - 0.5).cast(int)
        fx = pos * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1) ** 2,
             0.5 * (fx - 0.5) ** 2]

        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = offset.cast(float) - fx
            weight = w[i][0] * w[j][1] * w[k][2]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * v[p]


# ── 投影 & 渲染 ──────────────────────────────────────────────
def project_particles(positions, cam_pos, cam_lookat, fov_deg=45.0):
    """透视投影：把 3D 粒子投影到 2D 屏幕。返回 (N, 2) 和深度。"""
    cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    z_axis = cam_pos - cam_lookat
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(cam_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    tan_half = np.tan(np.deg2rad(fov_deg * 0.5))

    rel = positions - cam_pos                        # (N, 3)
    cam_x = rel @ x_axis
    cam_y = rel @ y_axis
    cam_z = rel @ z_axis                              # 视线方向

    depth = -cam_z
    valid = depth > 0.01

    screen = np.full((positions.shape[0], 2), -1.0, dtype=np.float32)
    ndc_x = np.zeros_like(cam_x)
    ndc_y = np.zeros_like(cam_y)
    ndc_x[valid] = cam_x[valid] / (depth[valid] * tan_half)
    ndc_y[valid] = cam_y[valid] / (depth[valid] * tan_half)
    screen[valid, 0] = (ndc_x[valid] + 1.0) * 0.5
    screen[valid, 1] = (ndc_y[valid] + 1.0) * 0.5
    return screen, depth, valid


def render_frame(positions, cam_pos, cam_lookat, figsize=(8, 8), dpi=120):
    """用 matplotlib 渲染粒子 + 场景线框，返回 RGBA numpy 数组。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    screen_coords, depths, valid = project_particles(
        positions, cam_pos, cam_lookat)

    # ── 场景 box 的 3D 角点 ──
    boxes = [
        ([0.5, 0.05, 0.5],  [1.0, 0.05, 1.0]),   # 地面
        ([0.5, 0.15, 0.7],  [0.3, 0.05, 0.1]),   # 阶梯1
        ([0.5, 0.25, 0.5],  [0.3, 0.05, 0.1]),   # 阶梯2
    ]

    # ── 2D 绘图 ──
    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # 渲染场景 box 线框（投影后画矩形）
    for (cx, cy, cz), (ex, ey, ez) in boxes:
        corners = np.array([
            [cx - ex, cy - ey, cz - ez],
            [cx + ex, cy - ey, cz - ez],
            [cx + ex, cy + ey, cz - ez],
            [cx - ex, cy + ey, cz - ez],
            [cx - ex, cy - ey, cz + ez],
            [cx + ex, cy - ey, cz + ez],
            [cx + ex, cy + ey, cz + ez],
            [cx - ex, cy + ey, cz + ez],
        ])
        proj, _, vld = project_particles(corners, cam_pos, cam_lookat)
        proj = proj[vld]
        if len(proj) < 4:
            continue
        # 找到投影后的 bounding rect
        xmin, ymin = proj.min(axis=0)
        xmax, ymax = proj.max(axis=0)
        if xmin < 0 or xmax > 1 or ymin < 0 or ymax > 1:
            xmin = max(xmin, 0); xmax = min(xmax, 1)
            ymin = max(ymin, 0); ymax = min(ymax, 1)
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=True, facecolor="#4a4a4e",
                             edgecolor="#6a6a70", linewidth=1.2,
                             alpha=0.7)
        ax.add_patch(rect)

    # 渲染粒子（按深度排序，远的先画）
    vld = valid
    if vld.sum() > 0:
        order = np.argsort(depths[vld])[::-1]  # 远的先画
        idx_vld = np.where(vld)[0][order]
        sx = screen_coords[idx_vld, 0]
        sy = screen_coords[idx_vld, 1]

        # 过滤屏幕外的点
        in_view = (sx >= 0) & (sx <= 1) & (sy >= 0) & (sy <= 1)
        ax.scatter(sx[in_view], sy[in_view],
                   s=1.5, c="#0fd9d9", edgecolors="none",
                   alpha=0.7, rasterized=True)

    # 时间戳
    sim_time = sim_frame * sim_dt_per_frame
    ax.text(0.01, 0.99, f"t = {sim_time:.2f}s",
            transform=ax.transAxes, fontsize=11,
            color="white", fontfamily="monospace",
            verticalalignment="top")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    rgba = buf.reshape(h, w, 4)
    plt.close(fig)
    return rgba


# ── 主程序 ───────────────────────────────────────────────────
os.makedirs(args.out, exist_ok=True)

init()
print(f"[初始化] {n_particles} 粒子, {n_grid}^3 网格")
print(f"[物理] dt={dt}, substeps={substeps}, "
      f"每渲染帧= {sim_dt_per_frame:.4f}s 物理时间")
print(f"[录制] 每 {capture_interval}s 拍一帧, 共 {total_captures} 张, "
      f"{total_captures * capture_interval:.1f}s 物理时间")
print(f"[输出] {args.out}/")

# 摄像机参数（高位俯视，稍偏移）
cam_pos = np.array([0.7, 1.8, 2.0], dtype=np.float32)
cam_lookat = np.array([0.5, 0.15, 0.6], dtype=np.float32)

sim_frame = 0
capture_count = 0
next_capture_frame = render_frames_per_capture

t_start = time.perf_counter()

while capture_count < total_captures:
    # 物理步进
    for _ in range(substeps):
        substep()
    sim_frame += 1

    # 每秒捕获一帧
    if sim_frame >= next_capture_frame:
        positions = x.to_numpy()
        rgba = render_frame(positions, cam_pos, cam_lookat,
                            dpi=args.dpi)

        capture_count += 1
        fname = os.path.join(args.out, f"frame_{capture_count:04d}.png")
        from PIL import Image
        Image.fromarray(rgba, "RGBA").convert("RGB").save(fname)

        elapsed = time.perf_counter() - t_start
        sim_t = sim_frame * sim_dt_per_frame
        print(f"  [{elapsed:6.1f}s 墙钟 | {sim_t:5.1f}s 物理] "
              f"已存 {fname}")

        next_capture_frame += render_frames_per_capture

total_elapsed = time.perf_counter() - t_start
print(f"\n录制完成: {capture_count} 帧, "
      f"墙钟时间 {total_elapsed:.1f}s, "
      f"物理时间 {sim_frame * sim_dt_per_frame:.1f}s")
print(f"帧保存位置: {args.out}/")
