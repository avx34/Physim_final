"""
mpm_sim.py — differentiable 3D MPM simulation module.
Modified for two-corner hanging fixed particle scene, gravity=0.5, match user demo
Depends on `sim_config.cfg` — must import after `cfg.init_taichi()`.
"""
import taichi as ti
from sim_config import cfg
import numpy as np

# ==============================================================================
#  Material parameter fields
# ==============================================================================
E_pred = ti.field(dtype=float, shape=(), needs_grad=True)
nu_pred = ti.field(dtype=float, shape=(), needs_grad=True)
mu_tmp = ti.field(dtype=float, shape=(), needs_grad=True)
lambda_tmp = ti.field(dtype=float, shape=(), needs_grad=True)

# 新增：悬挂固定粒子标记场
is_fixed = ti.field(dtype=int, shape=cfg.n_particles, needs_grad=False)
# 备份初始悬挂坐标
x_init = ti.Vector.field(cfg.dim, dtype=float, shape=cfg.n_particles, needs_grad=False)


# ==============================================================================
#  Particle & grid fields
# ==============================================================================
x = ti.Vector.field(cfg.dim, dtype=float, shape=cfg.n_particles,
                     needs_grad=True)
v = ti.Vector.field(cfg.dim, dtype=float, shape=cfg.n_particles,
                     needs_grad=True)
C = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float,
                     shape=cfg.n_particles, needs_grad=True)
F = ti.Matrix.field(cfg.dim, cfg.dim, dtype=float,
                     shape=cfg.n_particles, needs_grad=True)

grid_v = ti.Vector.field(cfg.dim, dtype=float,
                          shape=(cfg.n_grid,) * 3, needs_grad=True)
grid_m = ti.field(dtype=float, shape=(cfg.n_grid,) * 3, needs_grad=True)

ti.root.lazy_grad()


# ==============================================================================
#  AD-safe polar decomposition (Newton iteration)
# ==============================================================================
@ti.func
def polar_newton(A):
    """Return the rotation matrix R from polar decomposition F = R·S.

    Uses Newton iteration on R⁻¹ = Rᵀ to avoid cast_bits issues with
    Taichi 1.7.4 autodiff.  10 iterations with epsilon regularisation.
    """
    R_mat = A
    eps_reg = 1e-6
    for _ in ti.static(range(10)):
        R_reg = R_mat + eps_reg * ti.Matrix.identity(float, 3)
        R_mat = 0.5 * (R_mat + R_reg.inverse().transpose())
    return R_mat


# ==============================================================================
#  SDF helpers (完全不变，和你demo一致)
# ==============================================================================
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


# ==============================================================================
#  MPM init kernels (改成你悬挂粒子分布)
# ==============================================================================
@ti.func
def deterministic_unit(i, salt):
    phase = (ti.cast(i + 1, float) * (12.9898 + 17.23 * ti.cast(salt, float))
             + ti.cast(cfg.init_seed, float) * 0.12345)
    value = ti.sin(phase) * 43758.5453
    return value - ti.floor(value)

# 第一步：原始粒子分布 0.4~0.6 / 0.9~1.1 /0.45~0.65
@ti.kernel
def init_particles_raw():
    for i in range(cfg.n_particles):
        x[i] = [
            deterministic_unit(i, 0) * 0.2 + 0.4,
            deterministic_unit(i, 1) * 0.2 + 0.9,
            deterministic_unit(i, 2) * 0.2 + 0.45,
        ]
        v[i] = [0.0, 0.0, 0.0]
        F[i] = ti.Matrix.identity(float, 3)
        C[i] = ti.Matrix.zero(float, 3, 3)
        is_fixed[i] = 0

# 第二步：筛选后方两角各100粒子标记固定
def init_particles():
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

    fixed_mask = np.zeros(cfg.n_particles, dtype=np.int32)
    fixed_mask[fixed_indices] = 1
    is_fixed.from_numpy(fixed_mask)
    # 备份初始悬挂坐标
    x_init.copy_from(x)


def load_state_from_numpy(x_np, v_np, C_np, F_np):
    """Load a saved simulation state as the rollout initial condition."""
    x.from_numpy(x_np)
    v.from_numpy(v_np)
    C.from_numpy(C_np)
    F.from_numpy(F_np)
    # 加载状态后也要刷新固定标记与初始坐标
    pos_np = x.to_numpy()
    x_min, x_max = pos_np[:, 0].min(), pos_np[:, 0].max()
    y_max = pos_np[:, 1].max()
    z_min = pos_np[:, 2].min()
    bl = np.array([x_min, y_max, z_min])
    br = np.array([x_max, y_max, z_min])
    dl = np.linalg.norm(pos_np - bl, axis=1)
    dr = np.linalg.norm(pos_np - br, axis=1)
    il = np.argsort(dl)[:100]
    ir = np.argsort(dr)[:100]
    mask = np.zeros(cfg.n_particles, int)
    mask[il] = 1
    mask[ir] = 1
    is_fixed.from_numpy(mask)
    x_init.copy_from(x)


def init_from_target_data(data):
    """Use warm-up state from target data when present, else fall back."""
    required = ("x0", "v0", "C0", "F0")
    if all(key in data for key in required):
        load_state_from_numpy(
            data["x0"].astype("float32"),
            data["v0"].astype("float32"),
            data["C0"].astype("float32"),
            data["F0"].astype("float32"),
        )
        return True
    init_particles()
    return False


# ==============================================================================
#  NN映射、拉梅系数 完全保留官方原版不动
# ==============================================================================
@ti.kernel
def copy_nn_to_material_params(fc2_output: ti.template()):
    """Extract E from NN output, map (-1,1) → (50, 800)."""
    for _ in range(1):
        out_E = fc2_output[0, 0, 0, 0]
        E_pred[None] = cfg.E_MIN + (out_E + 1.0) * 0.5 * (
            cfg.E_MAX - cfg.E_MIN)


@ti.kernel
def copy_nn_to_nu(fc2_output: ti.template()):
    """Extract nu from NN output and map (-1, 1) to configured nu range."""
    for _ in range(1):
        out_nu = fc2_output[0, 0, 0, 0]
        nu_pred[None] = cfg.NU_MIN + (out_nu + 1.0) * 0.5 * (
            cfg.NU_MAX - cfg.NU_MIN)


@ti.kernel
def copy_nn_to_E_nu(fc2_output: ti.template()):
    """Extract E and nu from a two-output NN head."""
    for _ in range(1):
        out_E = fc2_output[0, 0, 0, 0]
        out_nu = fc2_output[0, 0, 0, 1]
        E_pred[None] = cfg.E_MIN + (out_E + 1.0) * 0.5 * (
            cfg.E_MAX - cfg.E_MIN)
        nu_pred[None] = cfg.NU_MIN + (out_nu + 1.0) * 0.5 * (
            cfg.NU_MAX - cfg.NU_MIN)


@ti.kernel
def compute_lame_params():
    """Compute Lamé parameters from the active E and nu fields."""
    nu = ti.min(ti.max(nu_pred[None], cfg.NU_MIN), cfg.NU_MAX)
    mu_tmp[None] = E_pred[None] / (2.0 * (1.0 + nu))
    lambda_tmp[None] = (E_pred[None] * nu
                        / ((1.0 + nu) * (1.0 - 2.0 * nu)))


@ti.kernel
def substep_reset_grid():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0


@ti.kernel
def substep_p2g():
    for p in x:
        pos = x[p] + ti.Vector([0.5, 0.5, 0.5])
        base = ti.cast(pos * cfg.inv_dx - 0.5, ti.i32)
        fx = pos * cfg.inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2]

        new_F = (ti.Matrix.identity(float, 3) + cfg.dt * C[p]) @ F[p]
        F[p] = new_F

        J = new_F.determinant()
        R = polar_newton(new_F)

        cauchy = (2.0 * mu_tmp[None] * (new_F - R) @ new_F.transpose()
                  + ti.Matrix.identity(float, 3)
                  * lambda_tmp[None] * J * (J - 1.0))
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
            grid_idx = base + offset
            dpos = (ti.cast(offset, float) - fx) * cfg.dx
            weight = w[i][0] * w[j][1] * w[k][2]
            if (grid_idx[0] >= 0 and grid_idx[0] < cfg.n_grid
                    and grid_idx[1] >= 0 and grid_idx[1] < cfg.n_grid
                    and grid_idx[2] >= 0 and grid_idx[2] < cfg.n_grid):
                momentum = weight * (cfg.p_mass * v[p] + affine @ dpos
                                     + penalty_force * cfg.dt)
                grid_v[grid_idx] += momentum
                grid_m[grid_idx] += weight * cfg.p_mass


@ti.kernel
def substep_grid_g2p():
    # 重力修改为0.5，替换原版9.8
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0.0:
            inv_m = 1.0 / ti.max(grid_m[i, j, k], 1e-10)
            grid_v[i, j, k] = inv_m * grid_v[i, j, k]
            grid_v[i, j, k][1] -= cfg.dt * 0.5
    for p in x:
        pos = x[p] + ti.Vector([0.5, 0.5, 0.5])
        base = ti.cast(pos * cfg.inv_dx - 0.5, ti.i32)
        fx = pos * cfg.inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            grid_idx = base + offset
            dpos = ti.cast(offset, float) - fx
            weight = w[i][0] * w[j][1] * w[k][2]
            if (grid_idx[0] >= 0 and grid_idx[0] < cfg.n_grid
                    and grid_idx[1] >= 0 and grid_idx[1] < cfg.n_grid
                    and grid_idx[2] >= 0 and grid_idx[2] < cfg.n_grid):
                g_v = grid_v[grid_idx]
                new_v += weight * g_v
                new_C += 4.0 * cfg.inv_dx * weight * g_v.outer_product(dpos)

        # 关键：固定粒子锁死位置、速度、形变梯度，和你demo逻辑完全一致
        if is_fixed[p] == 1:
            v[p] = ti.Vector([0.0, 0.0, 0.0])
            C[p] = ti.Matrix.zero(float, 3, 3)
            x[p] = x_init[p]
            F[p] = ti.Matrix.identity(float, 3)
        else:
            v[p] = new_v
            C[p] = new_C
            x[p] += cfg.dt * v[p]


def differentiable_substep():
    substep_reset_grid()
    substep_p2g()
    substep_grid_g2p()


# ==============================================================================
#  Gradient helper
# ==============================================================================
@ti.kernel
def zero_grad(f: ti.template()):
    for I in ti.grouped(f):
        f.grad[I] = 0.0