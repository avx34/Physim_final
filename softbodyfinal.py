import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

dim = 3
n_particles = 4096
n_grid = 128
dx = 2.0 / n_grid #网格拉宽但分辨率不变
inv_dx = 1 / dx
dt = 1e-4
substeps = int(2e-3 // dt)

p_vol = (dx * 0.5)**3
p_rho = 1
p_mass = p_vol * p_rho

# 弹性体材质参数 (Neo-Hookean)
E, nu = 40, 0.4
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

# 碰撞参数
ground_y = 0.1
penalty_k = 1e5     # penalty 刚度系数
penalty_damp = 2e3  # penalty 阻尼系数，消耗碰撞能量

# 粒子
x = ti.Vector.field(dim, dtype=float, shape=n_particles) # 粒子位置
v = ti.Vector.field(dim, dtype=float, shape=n_particles) # 粒子速度
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles) # 速度梯度
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles) # 形变梯度


# 网格
grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))

# 构建场景渲染网格
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
    ([0.5, 0.05, 0.5], [1.0, 0.05, 1.0]),  # 地面
    ([0.5, 0.15, 0.7], [0.3, 0.05, 0.1]),  # 阶梯1
    ([0.5, 0.25, 0.5], [0.3, 0.05, 0.1]),  # 阶梯2
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

box_min = np.array([-0.45, 0.05, -0.45], dtype=np.float32)
box_max = np.array([1.45, 1.45, 1.45], dtype=np.float32)

E_ctrl = ti.field(dtype=ti.f32, shape=())
nu_ctrl = ti.field(dtype=ti.f32, shape=())
E_ctrl[None] = 40.0
nu_ctrl[None] = 0.4

dragging_object = False
drag_plane_point = np.zeros(3, dtype=np.float32)
drag_plane_normal = np.zeros(3, dtype=np.float32)
drag_last_target = np.zeros(3, dtype=np.float32)
init_x_np = None
init_v_np = None
init_C_np = None
init_F_np = None


def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def spherical_camera(target, yaw, pitch, radius):
    offset = np.array([
        radius * np.sin(yaw) * np.cos(pitch),
        radius * np.sin(pitch),
        radius * np.cos(yaw) * np.cos(pitch),
    ], dtype=np.float32)
    return target + offset


def get_camera_basis(cam_pos, cam_lookat):
    forward = normalize(cam_lookat - cam_pos)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = normalize(right)
    up = normalize(np.cross(right, forward))
    return forward, right, up


def screen_to_ray(mouse_x, mouse_y, cam_pos, cam_lookat, fov_deg, aspect):
    forward, right, up = get_camera_basis(cam_pos, cam_lookat)
    ndc_x = mouse_x * 2.0 - 1.0
    ndc_y = mouse_y * 2.0 - 1.0
    tan_half_fov = np.tan(np.deg2rad(fov_deg * 0.5))
    ray_dir = normalize(forward + ndc_x * aspect * tan_half_fov * right + ndc_y * tan_half_fov * up)
    return cam_pos.copy(), ray_dir


def ray_aabb_intersection(ray_origin, ray_dir, bounds_min, bounds_max):
    t_min = -1e9
    t_max = 1e9
    for axis in range(3):
        if abs(ray_dir[axis]) < 1e-8:
            if ray_origin[axis] < bounds_min[axis] or ray_origin[axis] > bounds_max[axis]:
                return None
            continue
        inv_dir = 1.0 / ray_dir[axis]
        t1 = (bounds_min[axis] - ray_origin[axis]) * inv_dir
        t2 = (bounds_max[axis] - ray_origin[axis]) * inv_dir
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return None
    if t_max < 0.0:
        return None
    return t_min if t_min >= 0.0 else t_max


def ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal):
    denom = float(np.dot(ray_dir, plane_normal))
    if abs(denom) < 1e-8:
        return None
    t = float(np.dot(plane_point - ray_origin, plane_normal) / denom)
    if t < 0.0:
        return None
    return ray_origin + ray_dir * t


def pick_object(mouse_x, mouse_y, cam_pos, cam_lookat, fov_deg, aspect):
    ray_origin, ray_dir = screen_to_ray(mouse_x, mouse_y, cam_pos, cam_lookat, fov_deg, aspect)
    particle_pos = x.to_numpy()
    bounds_min = particle_pos.min(axis=0) - 0.03
    bounds_max = particle_pos.max(axis=0) + 0.03
    t = ray_aabb_intersection(ray_origin, ray_dir, bounds_min, bounds_max)
    if t is None:
        return False, None
    return True, ray_origin + ray_dir * t


@ti.kernel
def translate_particles(dx: float, dy: float, dz: float, vx: float, vy: float, vz: float):
    delta = ti.Vector([dx, dy, dz])
    vel = ti.Vector([vx, vy, vz])
    for i in range(n_particles):
        x[i] += delta
        v[i] = vel
        C[i] = ti.Matrix.zero(float, 3, 3)


@ti.kernel
def confine_particles(min_x: float, min_y: float, min_z: float, max_x: float, max_y: float, max_z: float):
    for i in range(n_particles):
        if x[i][0] < min_x:
            x[i][0] = min_x
            if v[i][0] < 0:
                v[i][0] = 0.0
        elif x[i][0] > max_x:
            x[i][0] = max_x
            if v[i][0] > 0:
                v[i][0] = 0.0

        if x[i][1] < min_y:
            x[i][1] = min_y
            if v[i][1] < 0:
                v[i][1] = 0.0
        elif x[i][1] > max_y:
            x[i][1] = max_y
            if v[i][1] > 0:
                v[i][1] = 0.0

        if x[i][2] < min_z:
            x[i][2] = min_z
            if v[i][2] < 0:
                v[i][2] = 0.0
        elif x[i][2] > max_z:
            x[i][2] = max_z
            if v[i][2] > 0:
                v[i][2] = 0.0


def reset_simulation():
    global dragging_object
    dragging_object = False
    if init_x_np is not None:
        x.from_numpy(init_x_np)
    if init_v_np is not None:
        v.from_numpy(init_v_np)
    if init_C_np is not None:
        C.from_numpy(init_C_np)
    if init_F_np is not None:
        F.from_numpy(init_F_np)
    drag_plane_point[:] = 0.0
    drag_plane_normal[:] = 0.0
    drag_last_target[:] = 0.0

@ti.func
def box_sdf(p, c, e):
    d = ti.abs(p - c) - e
    out_dist = ti.math.length(ti.max(d, 0.0))
    in_dist = ti.min(ti.max(d[0], ti.max(d[1], d[2])), 0.0)
    return out_dist + in_dist

@ti.func
def get_sdf(p):
    # 地面
    d = p[1] - 0.1
    
    # 四面墙
    wall_lx = p[0] - (-0.45)
    wall_rx = 1.45 - p[0]
    wall_lz = p[2] - (-0.45)
    wall_rz = 1.45 - p[2]
    
    # 阶梯 1
    c1 = ti.Vector([0.5, 0.15, 0.7])
    e1 = ti.Vector([0.3, 0.05, 0.1])
    d1 = box_sdf(p, c1, e1)
    
    # 阶梯 2
    c2 = ti.Vector([0.5, 0.25, 0.5])
    e2 = ti.Vector([0.3, 0.05, 0.1])
    d2 = box_sdf(p, c2, e2)
    
    # 取全部表面的布尔并集 (min)
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
    # 数值梯度法求解 SDF 法线
    eps = 1e-4
    dx = get_sdf(p + ti.Vector([eps, 0.0, 0.0])) - get_sdf(p - ti.Vector([eps, 0.0, 0.0]))
    dy = get_sdf(p + ti.Vector([0.0, eps, 0.0])) - get_sdf(p - ti.Vector([0.0, eps, 0.0]))
    dz = get_sdf(p + ti.Vector([0.0, 0.0, eps])) - get_sdf(p - ti.Vector([0.0, 0.0, eps]))
    n = ti.Vector([dx, dy, dz])
    len_n = n.norm()
    if len_n == 0:
        n = ti.Vector([0.0, 1.0, 0.0])
    else:
        n = n / len_n
    return n

@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.2 + 0.4, 
            ti.random() * 0.2 + 0.9,
            ti.random() * 0.2 + 0.45
        ]
        v[i] = [0, -2.0, 0]  # 初始向下速度
        F[i] = ti.Matrix.identity(float, 3)
        C[i] = ti.Matrix.zero(float, 3, 3)

@ti.kernel
def substep():
    # 每一步开始时重置网格
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
        base = (pos * inv_dx - 0.5).cast(int) # 粒子所在网格单元的左下角索引
        fx = pos * inv_dx - base.cast(float) # 粒子在网格单元内的相对位置
        
        # 二次 B-样条 权重, 把粒子状态分散到周围 3x3x3 个网格节点上
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        
        # 更新形变梯度 F
        new_F = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]
        F[p] = new_F
        
        # Neo-Hookean 模型计算应力
        J = new_F.determinant()
        R, S = ti.polar_decompose(new_F, ti.f32) # 极分解得到旋转矩阵 R
        cauchy = 2 * mu_0_now * (new_F - R) @ new_F.transpose() + ti.Matrix.identity(float, 3) * lambda_0_now * J * (J - 1)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        
        affine = stress + p_mass * C[p]
        
        # 基于全局复合 SDF 解析场景碰撞
        sdf = get_sdf(x[p])
        penalty_force = ti.Vector.zero(float, 3)
        if sdf < 0:
            normal = get_sdf_normal(x[p])
            
            # 沿表面法线施加推斥力
            f_n = -sdf * penalty_k * p_mass
            
            # 在法向投影相对速度，模拟阻尼
            v_n = v[p].dot(normal)
            if v_n < 0:
                f_n -= v_n * penalty_damp * p_mass
                
            penalty_force = f_n * normal
        
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            
            # 动量增量中累加了: 粒子惯性 + 弹性应力产生的力矩阵 + penalty碰撞力
            momentum_inc = weight * (p_mass * v[p] + affine @ dpos + penalty_force * dt)
            grid_v[base + offset] += momentum_inc
            grid_m[base + offset] += weight * p_mass

    # 网格操作
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            # 动量转速度
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]
            # 施加重力
            grid_v[i, j, k][1] -= dt * 9.8 
            
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
            
        v[p] = new_v
        C[p] = new_C
        x[p] += dt * v[p]

def main():
    global dragging_object
    init()
    global init_x_np, init_v_np, init_C_np, init_F_np
    init_x_np = x.to_numpy()
    init_v_np = v.to_numpy()
    init_C_np = C.to_numpy()
    init_F_np = F.to_numpy()
    
    window = ti.ui.Window("Taichi 3D MPM Softbody", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    gui = window.get_gui()
    
    cam_target = np.array([0.5, 0.1, 0.5], dtype=np.float32)
    cam_pos0 = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    cam_offset = cam_pos0 - cam_target
    cam_radius = float(np.linalg.norm(cam_offset))
    cam_yaw = float(np.arctan2(cam_offset[0], cam_offset[2]))
    cam_pitch = float(np.arcsin(cam_offset[1] / cam_radius))
    cam_fov = 45.0
    last_rmb_mouse = None
    last_t_pressed = False
    
    while window.running:
        t_pressed = window.is_pressed('t') or window.is_pressed('T')
        if t_pressed and not last_t_pressed:
            reset_simulation()
        last_t_pressed = t_pressed

        mouse_pos = window.get_cursor_pos()
        mouse_x, mouse_y = mouse_pos

        E_ctrl[None] = gui.slider_float("E", float(E_ctrl[None]), 20.0, 200.0)
        nu_ctrl[None] = gui.slider_float("nu", float(nu_ctrl[None]), 0.1, 1.0)

        if window.is_pressed(ti.ui.RMB):
            if last_rmb_mouse is None:
                last_rmb_mouse = mouse_pos
            else:
                dx_mouse = mouse_pos[0] - last_rmb_mouse[0]
                dy_mouse = mouse_pos[1] - last_rmb_mouse[1]
                cam_yaw += dx_mouse * 4.0
                cam_pitch = max(-1.35, min(1.35, cam_pitch + dy_mouse * 4.0))
                last_rmb_mouse = mouse_pos
        else:
            last_rmb_mouse = None

        cam_pos = spherical_camera(cam_target, cam_yaw, cam_pitch, cam_radius)
        camera.position(cam_pos[0], cam_pos[1], cam_pos[2])
        camera.lookat(cam_target[0], cam_target[1], cam_target[2])
        camera.fov(cam_fov)

        if window.is_pressed(ti.ui.LMB):
            if not dragging_object:
                hit, hit_point = pick_object(mouse_pos[0], mouse_pos[1], cam_pos, cam_target, cam_fov, 1.0)
                if hit:
                    dragging_object = True
                    drag_plane_point[:] = hit_point
                    drag_last_target[:] = hit_point
                    drag_plane_normal[:] = normalize(cam_target - cam_pos)
            else:
                ray_origin, ray_dir = screen_to_ray(mouse_pos[0], mouse_pos[1], cam_pos, cam_target, cam_fov, 1.0)
                target = ray_plane_intersection(ray_origin, ray_dir, drag_plane_point, drag_plane_normal)
                if target is not None:
                    delta = target - drag_last_target
                    if np.linalg.norm(delta) > 1e-8:
                        frame_dt = substeps * dt
                        if frame_dt > 0:
                            translate_particles(
                                float(delta[0]), float(delta[1]), float(delta[2]),
                                float(delta[0] / frame_dt), float(delta[1] / frame_dt), float(delta[2] / frame_dt)
                            )
                    drag_last_target[:] = target
        else:
            dragging_object = False

        for _ in range(substeps):
            substep()
            confine_particles(
                float(box_min[0]), float(box_min[1]), float(box_min[2]),
                float(box_max[0]), float(box_max[1]), float(box_max[2])
            )

        scene.set_camera(camera)
        # 场景环境mesh
        scene.mesh(scene_verts, indices=scene_inds, color=(0.4, 0.4, 0.45))

        # 光照
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        
        # 渲染粒子
        scene.particles(x, radius=0.005, color=(0.06, 0.85, 0.87))
        
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()