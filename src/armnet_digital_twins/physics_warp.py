"""The "warp" engine (physics.py): all of a twin's worlds stepped together on the GPU by MuJoCo Warp.

Each physics step is one launch of a CUDA graph of mujoco_warp.step. The per-world buffers for contacts, constraint
rows and CCD contacts are fixed when the engine is built (Task.warp_buffers). A world that needs more silently loses
contacts, so every step ORs MuJoCo Warp's overflow flags into a per-world mask, and a world with a lossy overflow or a
non-finite state is bad from then on (ok False).

Results agree closely with CPU MuJoCo but not exactly (float32, and MuJoCo Warp's own convex collision), so contact
outcomes that are on a knife edge can differ between engines. Two runs on the same GPU also differ slightly, as the
GPU's atomic additions don't run in a fixed order."""
from __future__ import annotations

import mujoco
import numpy as np

from .physics import STATE_FIELDS

# overflow flags that lose data (constraint rows or contacts dropped). The others only mean that a solver did not
# fully converge within its iteration limit in some step, which CPU MuJoCo also allows silently.
LOSSY_OVERFLOW = {1: "nefc", 2: "njmax_nnz", 4: "broadphase", 8: "narrowphase", 16: "ccd", 32: "hfield", 128: "nvmax"}
_MASK = sum(LOSSY_OVERFLOW)
_modules = None
_accumulate = None


def modules():
    """(warp, mujoco_warp), imported once."""
    global _modules
    if _modules is None:
        try:
            import warp as wp
            import mujoco_warp as mjw
        except ImportError as e:
            raise ImportError("the warp engine needs MuJoCo Warp: pip install armnet-digital-twins[warp]") from e
        if hasattr(wp, "LOG_WARNING"):
            wp.config.log_level = wp.LOG_WARNING      # no "Module ... load" line per kernel
        wp.init()
        _modules = (wp, mjw)
    return _modules


def _accumulate_kernel(wp):
    global _accumulate
    if _accumulate is None:
        @wp.kernel
        def accumulate(flags: wp.array(dtype=wp.int32), seen: wp.array(dtype=wp.int32)):
            i = wp.tid()
            seen[i] = seen[i] | flags[i]
        _accumulate = accumulate
    return _accumulate


class WarpPhysics:
    """nconmax / njmax / nccdmax: per-world buffer sizes (contacts, constraint rows, CCD contacts); None for MuJoCo
    Warp's defaults."""
    name = "warp"

    def __init__(self, model, n_worlds: int = 1, *, nconmax: int | None = None, njmax: int | None = None,
                 nccdmax: int | None = None):
        wp, mjw = modules()
        self.wp, self.mjw, self.model, self.n_worlds = wp, mjw, model, int(n_worlds)
        self.m = mjw.put_model(model)
        self.m.opt.warn_overflow = 0                  # checked per world through the overflow mask instead
        d0 = mujoco.MjData(model)
        mujoco.mj_forward(model, d0)
        self.d = mjw.put_data(model, d0, nworld=self.n_worlds, nconmax=nconmax, njmax=njmax, nccdmax=nccdmax)
        self._seen = wp.zeros(self.n_worlds, dtype=wp.int32, device=self.d.qpos.device)
        self._bad = np.zeros(self.n_worlds, bool)
        self._graph = None

    def load(self, states):
        self.mjw.reset_data(self.m, self.d)
        self._seen.zero_()
        for f in STATE_FIELDS:
            if f != "act" or self.model.na:
                getattr(self.d, f).assign(np.stack([s[f] for s in states]).astype(np.float32))
        self.d.time.assign(np.array([s["time"] for s in states], np.float32))
        self._bad[:] = False

    def restart(self):
        self.d.qvel.zero_()
        self.d.time.zero_()

    def set_ctrl(self, ctrl):
        self.d.ctrl.assign(np.asarray(ctrl, np.float32).reshape(self.n_worlds, -1))

    def _step_once(self):
        self.mjw.step(self.m, self.d)
        self.wp.launch(_accumulate_kernel(self.wp), dim=self.n_worlds, inputs=[self.d.overflow, self._seen])

    def step(self, n: int, worlds=None):
        """Every world steps (the GPU does them together); `worlds` only matters to the CPU engine."""
        wp = self.wp
        if n <= 0:
            return
        if self._graph is None:                       # the first step compiles the kernels; then capture one step
            self._step_once()
            n -= 1
            if self.d.qpos.device.is_cuda:
                with wp.ScopedCapture() as capture:
                    self._step_once()
                self._graph = capture.graph
        for _ in range(n):
            if self._graph is not None:
                wp.capture_launch(self._graph)
            else:
                self._step_once()

    @property
    def qpos(self) -> np.ndarray:
        return self.d.qpos.numpy().astype(np.float64)

    @property
    def ok(self) -> np.ndarray:
        q, v = self.d.qpos.numpy(), self.d.qvel.numpy()
        self._bad |= ~(np.isfinite(q).all(1) & np.isfinite(v).all(1)) | ((self._seen.numpy() & _MASK) != 0)
        return ~self._bad

    def overflows(self) -> list[list[str]]:
        """Per world, the buffers that overflowed since the last load."""
        seen = self._seen.numpy()
        return [[n for b, n in LOSSY_OVERFLOW.items() if s & b] for s in seen]

    def close(self):
        self._graph = None
