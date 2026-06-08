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
E, nu = 400, 0.4
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
        cauchy = 2 * mu_0 * (new_F - R) @ new_F.transpose() + ti.Matrix.identity(float, 3) * lambda_0 * J * (J - 1)
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
    init()
    
    window = ti.ui.Window("Taichi 3D MPM Softbody", (800, 800))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    camera.position(0.5, 1.0, 2.0)
    camera.lookat(0.5, 0.1, 0.5)
    
    while window.running:
        for _ in range(substeps):
            substep()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
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