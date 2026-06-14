"""
camera_module.py — modular perspective-camera projection for MPM particles.

Provides a self-contained `Camera` class that projects 3D particle positions
to 2D screen coordinates via a perspective camera model.  The module supports
both **Taichi-kernel** projection (fast, differentiable when used inside a
`ti.ad.Tape`) and **NumPy** projection (for offline data generation).

All Taichi field allocations happen inside class ``__init__``, so this module
is safe to import before ``ti.init()`` — you just can't instantiate a Camera
until Taichi is initialised.

Usage
-----
    from camera_module import Camera

    cam = Camera(position=(0.7, 1.8, 2.0), lookat=(0.5, 0.15, 0.6),
                 fov_deg=45.0, aspect_ratio=1.0, n_particles=8192)

    # Taichi kernel projection (for training / AD)
    cam.project_kernel(sim_x)   # fills cam.proj_2d, cam.proj_valid, cam.proj_depth

    # NumPy projection (for offline data export)
    screen, depth, valid = cam.project_numpy(x_np)
"""

import taichi as ti
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera class
# ═══════════════════════════════════════════════════════════════════════════════

@ti.data_oriented
class Camera:
    """Perspective camera for 3D→2D particle projection.

    Parameters
    ----------
    position : tuple[float, float, float]
        World-space camera position.
    lookat : tuple[float, float, float]
        Point the camera looks at (defines view direction).
    fov_deg : float
        Vertical field-of-view in degrees.
    aspect_ratio : float
        Viewport width / height.
    n_particles : int
        Number of particles to project (allocates Taichi fields).
    needs_grad : bool
        Whether 2D projection fields need gradient tracking (for AD training).
    """

    def __init__(self, position, lookat, fov_deg=45.0, aspect_ratio=1.0,
                 n_particles=8192, needs_grad=False):
        # ── camera intrinsics & extrinsics ──
        self.position = np.array(position, dtype=np.float32)
        self.lookat = np.array(lookat, dtype=np.float32)
        self.fov_deg = float(fov_deg)
        self.aspect_ratio = float(aspect_ratio)
        self.n_particles = n_particles

        # precompute camera basis (world-space)
        self._compute_basis()

        # ── Taichi output fields ──
        self.proj_2d = ti.Vector.field(2, dtype=float, shape=n_particles,
                                       needs_grad=needs_grad)
        self.proj_valid = ti.field(dtype=float, shape=n_particles,
                                   needs_grad=False)
        self.proj_depth = ti.field(dtype=float, shape=n_particles,
                                   needs_grad=needs_grad)

        # constant scalar fields for kernel access
        self._cam_pos = ti.Vector.field(3, dtype=float, shape=())
        self._cam_x_axis = ti.Vector.field(3, dtype=float, shape=())
        self._cam_y_axis = ti.Vector.field(3, dtype=float, shape=())
        self._cam_z_axis = ti.Vector.field(3, dtype=float, shape=())
        self._tan_half_fov = ti.field(dtype=float, shape=())
        self._cam_aspect = ti.field(dtype=float, shape=())

        self._cam_pos.from_numpy(self.position)
        self._cam_x_axis.from_numpy(self.x_axis)
        self._cam_y_axis.from_numpy(self.y_axis)
        self._cam_z_axis.from_numpy(self.z_axis)
        self._tan_half_fov[None] = self.tan_half_fov
        self._cam_aspect[None] = self.aspect_ratio

    # ── basis computation ──────────────────────────────────────────────

    def _compute_basis(self):
        """Compute camera coordinate system (world → camera basis)."""
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        z_axis = self.position - self.lookat
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-8:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            z_axis /= z_norm

        x_axis = np.cross(up, z_axis)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            # gimbal-lock guard: camera looking straight up/down
            alt_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            x_axis = np.cross(alt_up, z_axis)
            x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            x_axis /= x_norm

        y_axis = np.cross(z_axis, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-8:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            y_axis /= y_norm

        self.x_axis = x_axis.astype(np.float32)
        self.y_axis = y_axis.astype(np.float32)
        self.z_axis = z_axis.astype(np.float32)
        self.tan_half_fov = np.tan(np.deg2rad(self.fov_deg * 0.5))

    # ── Taichi projection kernel ──────────────────────────────────────

    @ti.kernel
    def project_kernel(self, x: ti.template()):
        """Project 3D particle positions to 2D screen coordinates.

        Reads ``x`` (world-space positions, shape (n_particles, 3)) and fills
        ``self.proj_2d``, ``self.proj_depth``, and ``self.proj_valid``.

        Particles behind the camera or at depth ≤ 1e-4 are marked invalid
        (proj_2d = (-1, -1), valid = 0).
        """
        cam_pos = self._cam_pos[None]
        cam_x_axis = self._cam_x_axis[None]
        cam_y_axis = self._cam_y_axis[None]
        cam_z_axis = self._cam_z_axis[None]
        tan_half = self._tan_half_fov[None]
        aspect = self._cam_aspect[None]

        for i in range(self.n_particles):
            p_rel = x[i] - cam_pos

            # transform to camera space
            cam_x = p_rel.dot(cam_x_axis)
            cam_y = p_rel.dot(cam_y_axis)
            cam_z = p_rel.dot(cam_z_axis)

            depth = -cam_z  # positive in front of camera

            if depth > 1e-4:
                ndc_x = cam_x / (depth * tan_half * aspect)
                ndc_y = cam_y / (depth * tan_half)

                screen_x = (ndc_x + 1.0) * 0.5
                screen_y = (ndc_y + 1.0) * 0.5

                self.proj_2d[i] = ti.Vector([screen_x, screen_y])
                self.proj_valid[i] = 1.0
                self.proj_depth[i] = depth
            else:
                self.proj_2d[i] = ti.Vector([-1.0, -1.0])
                self.proj_valid[i] = 0.0
                self.proj_depth[i] = -1.0

    # ── NumPy projection (offline / data-gen) ─────────────────────────

    def project_numpy(self, x_np):
        """Project 3D positions → 2D using NumPy (no Taichi kernel).

        Parameters
        ----------
        x_np : np.ndarray, shape (N, 3)
            World-space particle positions.

        Returns
        -------
        screen : np.ndarray, shape (N, 2)
            Screen coordinates in [0,1]², (-1,-1) if invalid.
        depth : np.ndarray, shape (N,)
            Positive depth in front of camera, -1 if invalid.
        valid : np.ndarray, shape (N,) bool
            True for particles in front of camera.
        """
        rel = x_np - self.position                     # (N, 3)
        cam_x = rel @ self.x_axis
        cam_y = rel @ self.y_axis
        cam_z = rel @ self.z_axis

        depth = -cam_z
        valid = depth > 1e-4

        screen = np.full((x_np.shape[0], 2), -1.0, dtype=np.float32)
        ndc_x = np.zeros_like(cam_x)
        ndc_y = np.zeros_like(cam_y)
        ndc_x[valid] = cam_x[valid] / (depth[valid] * self.tan_half_fov
                                        * self.aspect_ratio)
        ndc_y[valid] = cam_y[valid] / (depth[valid] * self.tan_half_fov)
        screen[valid, 0] = (ndc_x[valid] + 1.0) * 0.5
        screen[valid, 1] = (ndc_y[valid] + 1.0) * 0.5

        depth_out = np.full(x_np.shape[0], -1.0, dtype=np.float32)
        depth_out[valid] = depth[valid]

        return screen, depth_out, valid

    # ── statistics helpers ────────────────────────────────────────────

    @staticmethod
    def compute_2d_statistics(screen_2d, valid_mask=None):
        """Compute mean and 2×2 covariance of visible 2D projected points.

        Parameters
        ----------
        screen_2d : np.ndarray, shape (N, 2)
            2D screen coordinates.
        valid_mask : np.ndarray, shape (N,) bool or None
            Visibility mask. If None, all points are considered visible.

        Returns
        -------
        mean_2d : np.ndarray, shape (2,)
        cov_2d : np.ndarray, shape (2, 2)
        n_valid : int
        """
        if valid_mask is None:
            pts = screen_2d
        else:
            pts = screen_2d[valid_mask]

        if len(pts) == 0:
            return np.zeros(2, dtype=np.float32), np.zeros((2, 2),
                                                           dtype=np.float32), 0

        mean_2d = np.mean(pts, axis=0).astype(np.float32)
        cov_2d = np.cov(pts.T).astype(np.float32)
        if cov_2d.ndim == 0:  # single point edge case
            cov_2d = np.zeros((2, 2), dtype=np.float32)
        return mean_2d, cov_2d, len(pts)

    @staticmethod
    def compute_camera_features(proj_mean_traj, proj_cov_traj,
                                depth_mean_traj,
                                n_steps, substeps_per_step, dt):
        """Extract 8 NN input features from a camera trajectory.

        Mirrors the 3D feature-extraction pattern in ``inverse_train.py``
        but operates on 2D projected statistics.

        Parameters
        ----------
        proj_mean_traj : np.ndarray, shape (T, 2)
            Per-step mean of 2D projected positions.
        proj_cov_traj : np.ndarray, shape (T, 2, 2)
            Per-step 2×2 covariance of projected positions.
        depth_mean_traj : np.ndarray, shape (T,)
            Per-step mean depth.
        n_steps, substeps_per_step, dt : int, int, float
            Simulation timing parameters.

        Returns
        -------
        features : np.ndarray, shape (8,)
        """
        features = np.zeros(8, dtype=np.float32)

        # 2D projected mean statistics (screen-X and screen-Y)
        features[0] = proj_mean_traj[0, 0]       # initial mean X
        features[1] = proj_mean_traj[0, 1]       # initial mean Y
        features[2] = proj_mean_traj[-1, 0]      # final mean X
        features[3] = proj_mean_traj[-1, 1]      # final mean Y

        # covariance trace statistics
        cov_trace = np.array([np.trace(proj_cov_traj[t])
                              for t in range(proj_cov_traj.shape[0])])
        features[4] = np.mean(cov_trace)         # mean cov trace
        features[5] = np.max(cov_trace)          # max cov trace

        # depth range
        features[6] = np.max(depth_mean_traj) - np.min(depth_mean_traj)

        # mean-Y rate of change (analogous to height rate in 3D mode)
        features[7] = abs(proj_mean_traj[-1, 1] - proj_mean_traj[0, 1]) / \
            max(n_steps * substeps_per_step * dt, 1e-8)

        return features
