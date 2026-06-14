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
_PROJ_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(_PROJ_ROOT, "data")
MODEL_DIR = os.path.join(DATA_DIR, "trained_model")


class SimConfig:
    """MPM simulation and inverse-learning configuration."""

    def __init__(self):
        # ---- 和gen_target_data完全对齐的仿真尺寸 ----
        self.dim = 3
        self.n_particles = 4096
        self.n_grid = 128

        # time stepping 1:1匹配生成数据代码
        self.dt = 1e-4
        self.substeps_per_step = 20
        self.n_steps = 200

        # material 新增内核需要的E、NU边界
        self.p_rho = 1.0
        self.E_MIN = 50.0
        self.E_MAX = 800.0
        self.NU_MIN = 0.0
        self.NU_MAX = 0.495

        # collision
        self.ground_y = 0.1
        self.penalty_k = 1e5
        self.penalty_damp = 2e3

        # 悬挂初始化种子
        self.init_seed = 42
        self.warmup_steps = 170

        # NN architecture  (ν固定0.4，只训练E)
        self.n_input = 6
        self.n_hidden = 32
        self.n_output = 1
        self.NU_FIXED = 0.4

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
        self.substeps_per_step = 20
        self._recompute_derived()
        print(f"[TINY MODE] n_particles={self.n_particles} "
              f"n_grid={self.n_grid} n_steps={self.n_steps}")

    # ------------------------------------------------------------------
    # Taichi init
    # ------------------------------------------------------------------
    def init_taichi(self):
        """Purge stale cache and initialise Taichi backend"""
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