"""
mpm_sim.py — differentiable 3D MPM simulation module.

Creates all particle/grid fields and velocity/position kernels.
Depends on `sim_config.cfg` — must import after `cfg.init_taichi()`.
"""
import taichi as ti
from sim_config import cfg


# ==============================================================================
#  Material parameter fields
# ==============================================================================
E_pred = ti.field(dtype=float, shape=(), needs_grad=True)
mu_tmp = ti.field(dtype=float, shape=(), needs_grad=True)
lambda_tmp = ti.field(dtype=float, shape=(), needs_grad=True)


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
#  SDF helpers
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
#  MPM kernels
# ==============================================================================
@ti.func
def deterministic_unit(i, salt):
    phase = (ti.cast(i + 1, float) * (12.9898 + 17.23 * ti.cast(salt, float))
             + ti.cast(cfg.init_seed, float) * 0.12345)
    value = ti.sin(phase) * 43758.5453
    return value - ti.floor(value)


@ti.kernel
def init_particles():
    for i in range(cfg.n_particles):
        x[i] = [
            cfg.init_base_x + deterministic_unit(i, 0) * cfg.init_extent,
            cfg.init_base_y + deterministic_unit(i, 1) * cfg.init_extent,
            cfg.init_base_z + deterministic_unit(i, 2) * cfg.init_extent,
        ]
        v[i] = [0.0, cfg.init_v_y, 0.0]
        F[i] = ti.Matrix.identity(float, 3)
        C[i] = ti.Matrix.zero(float, 3, 3)


def load_state_from_numpy(x_np, v_np, C_np, F_np):
    """Load a saved simulation state as the rollout initial condition."""
    x.from_numpy(x_np)
    v.from_numpy(v_np)
    C.from_numpy(C_np)
    F.from_numpy(F_np)


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


@ti.kernel
def copy_nn_to_material_params(fc2_output: ti.template()):
    """Extract E from NN output, map (-1,1) → (50, 800)."""
    for _ in range(1):
        out_E = fc2_output[0, 0, 0, 0]
        E_pred[None] = 50.0 + (out_E + 1.0) * 0.5 * 750.0


@ti.kernel
def compute_lame_params():
    """Compute Lamé parameters from E_pred with fixed ν."""
    mu_tmp[None] = E_pred[None] / (2.0 * (1.0 + cfg.NU_FIXED))
    lambda_tmp[None] = (E_pred[None] * cfg.NU_FIXED
                        / ((1.0 + cfg.NU_FIXED) * (1.0 - 2.0 * cfg.NU_FIXED)))


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
            dpos = (ti.cast(offset, float) - fx) * cfg.dx
            weight = w[i][0] * w[j][1] * w[k][2]
            momentum = weight * (cfg.p_mass * v[p] + affine @ dpos
                                 + penalty_force * cfg.dt)
            grid_v[base + offset] += momentum
            grid_m[base + offset] += weight * cfg.p_mass


@ti.kernel
def substep_grid_g2p():
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0.0:
            inv_m = 1.0 / ti.max(grid_m[i, j, k], 1e-10)
            grid_v[i, j, k] = inv_m * grid_v[i, j, k]
            grid_v[i, j, k][1] -= cfg.dt * 9.8
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
            dpos = ti.cast(offset, float) - fx
            weight = w[i][0] * w[j][1] * w[k][2]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4.0 * cfg.inv_dx * weight * g_v.outer_product(dpos)
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
