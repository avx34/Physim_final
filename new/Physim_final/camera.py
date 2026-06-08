import taichi as ti
import numpy as np
import os

ti.init(arch=ti.gpu)

dim = 3
# 1. 简化粒子数量为 100
n_particles = 100
n_grid = 32  # 粒子变少，网格分辨率也可以适当调小以加速
dx = 2.0 / n_grid 
inv_dx = 1 / dx
dt = 1e-4
substeps = int(2e-3 // dt)

p_vol = (dx * 0.5)**3
p_rho = 1
p_mass = p_vol * p_rho

# 弹性体材质参数 (Neo-Hookean)
E, nu = 200, 0.4
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

# 碰撞参数
ground_y = 0.1
penalty_k = 1e5     # penalty 刚度系数
penalty_damp = 2e3  # penalty 阻尼系数

# 粒子数据场
x = ti.Vector.field(dim, dtype=float, shape=n_particles) 
v = ti.Vector.field(dim, dtype=float, shape=n_particles) 
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles) 
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles) 

# --- 新增：专门用来存储二维投影坐标的场 ---
# 存储每个粒子归一化到 [0, 1] 的 2D 投影坐标 (x, y)。如果超出边界或在相机背面，则为 (-1, -1)
particle_pos_2d = ti.Vector.field(2, dtype=float, shape=n_particles)

# 网格
grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))

# 构建场景渲染网格（保持原样）
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
        # 100个粒子，稍微聚集得紧密一点点
        x[i] = [
            ti.random() * 0.1 + 0.45, 
            ti.random() * 0.1 + 0.9,
            ti.random() * 0.1 + 0.5
        ]
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
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        
        new_F = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]
        F[p] = new_F
        
        J = new_F.determinant()
        R, S = ti.polar_decompose(new_F, ti.f32)
        cauchy = 2 * mu_0 * (new_F - R) @ new_F.transpose() + ti.Matrix.identity(float, 3) * lambda_0 * J * (J - 1)
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

    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]
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


# --- 新增：相机投影计算 Kernel ---
@ti.kernel
def project_particles(
    cam_pos: ti.types.vector(3, float),
    cam_lookat: ti.types.vector(3, float),
    fov: float,
    aspect_ratio: float
):
    # 默认的 Up 向量
    cam_up = ti.Vector([0.0, 1.0, 0.0])
    
    # 计算相机坐标系的基向量 (LookAt 矩阵原理)
    z_axis = (cam_pos - cam_lookat).normalized() 
    x_axis = cam_up.cross(z_axis).normalized()   
    y_axis = z_axis.cross(x_axis).normalized()   

    tan_half_fov = ti.tan(fov * 0.5 * 3.14159265 / 180.0)

    for i in range(n_particles):
        p_world = x[i]
        p_rel = p_world - cam_pos
        
        # 转换到相机空间
        p_cam_x = p_rel.dot(x_axis)
        p_cam_y = p_rel.dot(y_axis)
        p_cam_z = p_rel.dot(z_axis)

        z_depth = -p_cam_z # 视线前方深度为正

        if z_depth > 0.01: # 避免除以0或相机背面的粒子
            # 透视投影变换到归一化坐标 (NDC)
            ndc_x = p_cam_x / (z_depth * tan_half_fov * aspect_ratio)
            ndc_y = p_cam_y / (z_depth * tan_half_fov)

            # 映射到 [0, 1] 视口坐标空间 (左下角为 0,0 ； 右上角为 1,1)
            screen_x = (ndc_x + 1.0) * 0.5
            screen_y = (ndc_y + 1.0) * 0.5
            
            particle_pos_2d[i] = ti.Vector([screen_x, screen_y])
        else:
            particle_pos_2d[i] = ti.Vector([-1.0, -1.0]) # 不可见标记


def main():
    init()
    
    window = ti.ui.Window("Taichi 3D MPM Softbody", (800, 800))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    # 固定的相机参数
    c_pos = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    c_lookat = np.array([0.5, 0.1, 0.5], dtype=np.float32)
    c_fov = 45.0
    c_aspect = 1.0 # 800 / 800
    
    camera.position(c_pos[0], c_pos[1], c_pos[2])
    camera.lookat(c_lookat[0], c_lookat[1], c_lookat[2])
    camera.fov(c_fov)
    
    # 创建保存目录
    output_dir = "./projection_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # 用于在内存中缓存前 100 帧数据的列表
    all_frames_data = []
    frame_count = 0
    max_frames = 100

    print("开始仿真并记录投影位置...")

    while window.running:
        for _ in range(substeps):
            substep()

        # 允许鼠标右键调整视角，但为了导出的数据和初始相机一致，我们使用定义的固定参数
        # 如果你想追踪用户实时调整的视角，可以解开下面三行注释：
        # c_pos = np.array(camera.get_postsation(), dtype=np.float32) # 注：GGUI未直接暴露此接口，建议使用固定相机
        
        # 执行 2D 投影计算
        project_particles(c_pos, c_lookat, c_fov, c_aspect)

        # 收集当前帧的 2D 坐标 (shape: 100 x 2)
        if frame_count < max_frames:
            current_frame_2d = particle_pos_2d.to_numpy()
            all_frames_data.append(current_frame_2d)
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"已记录 {frame_count} / {max_frames} 帧")
            
            if frame_count == max_frames:
                # 攒满 100 帧，一次性保存为一个大数组 (Shape: 100 x 100 x 2)
                final_array = np.stack(all_frames_data, axis=0)
                np.save(os.path.join(output_dir, "trajectories_2d.npy"), final_array)
                # 同时存一份人类可读的 txt（选存，展示第0帧前5个粒子作为示例）
                np.savetxt(os.path.join(output_dir, "frame_0_preview.txt"), final_array[0], fmt="%.4f", header="X_screen Y_screen")
                print(f"数据已成功保存至 {output_dir}/trajectories_2d.npy !")

        # 正常渲染 3D 场景
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.mesh(scene_verts, indices=scene_inds, color=(0.4, 0.4, 0.45))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        
        # 渲染缩减后的 100 个粒子
        scene.particles(x, radius=0.015, color=(0.06, 0.85, 0.87)) # 把 radius 放大一点方便看清
        
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()