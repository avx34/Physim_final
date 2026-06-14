"""
sim_config.py — shared simulation configuration and Taichi initialisation.

All configurable parameters live in a single SimConfig instance (`cfg`).
Import this module first, then call `cfg.init_taichi()` before importing
modules that create Taichi fields (mpm_sim, observables).
"""
import os
import shutil
import taichi as ti

# ---- project paths ----
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJ_ROOT, "data")
MODEL_DIR = os.path.join(DATA_DIR, "trained_model")


class SimConfig:
    """MPM simulation and inverse-learning configuration."""

    def __init__(self):
        # ---- Simulation constants ----
        self.dim = 3
        self.n_particles = 8192
        self.n_grid = 64

        # time stepping
        self.dt = 5e-4
        self.substeps_per_step = 2
        self.n_steps = 64

        # material
        self.p_rho = 1.0

        # collision
        self.ground_y = 0.1
        self.penalty_k = 1e5
        self.penalty_damp = 2e3
        self.mu_friction = 0.4

        # plasticity
        self.yield_max = 1.05

        # Deterministic initial condition: removes per-epoch physical
        # initialization noise while keeping an irregular particle cloud.
        self.init_base_x = 0.42
        self.init_base_y = 0.85
        self.init_base_z = 0.42
        self.init_extent = 0.16
        self.init_v_y = -2.0
        self.init_seed = 42
        self.warmup_steps = 150

        # NN architecture 
        self.n_input = 7
        self.n_hidden = 32
        self.n_output = 1
        self.NU_FIXED = 0.4
        self.E_FIXED = 400

        # derived (computed once at init)
        self._recompute_derived()

    def _recompute_derived(self):
        self.dx = 2.0 / self.n_grid
        self.inv_dx = 1.0 / self.dx
        self.p_vol = (self.dx * 0.5) ** 3
        self.p_mass = self.p_vol * self.p_rho

    def apply_tiny(self, n_particles=512, n_grid=32, n_steps=8):
        """Override for fast smoke-test validation."""
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.n_steps = n_steps
        self.substeps_per_step = 2
        self._recompute_derived()
        print(f"[TINY MODE] n_particles={self.n_particles} "
              f"n_grid={self.n_grid} n_steps={self.n_steps}")

    # ------------------------------------------------------------------
    # Taichi init
    # ------------------------------------------------------------------
    def init_taichi(self):
        """Purge stale cache and initialise Taichi with CUDA backend."""
        self._clear_cache()
        for name, candidate in [("cuda", ti.cuda), ("cpu", ti.cpu)]:
            try:
                ti.init(arch=candidate, device_memory_fraction=0.5,
                        random_seed=42)
                print(f"[ARCH] Using {name} backend")
                return
            except Exception as e:
                print(f"[ARCH] {name} not available: {e}")
                try:
                    ti.reset()
                except Exception:
                    pass
        raise RuntimeError(
            "No Taichi backend available (tried cuda, cpu)")

    @staticmethod
    def _clear_cache():
        cache_dirs = [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "taichi"),
            os.path.join(os.environ.get("TEMP", ""), "taichi"),
        ]
        for cd in cache_dirs:
            if os.path.isdir(cd):
                try:
                    shutil.rmtree(cd)
                    print(f"[CACHE] Removed stale Taichi cache: {cd}")
                except Exception:
                    pass


# singleton
cfg = SimConfig()
