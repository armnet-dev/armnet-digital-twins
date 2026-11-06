"""Physics engines: what steps a twin's worlds. Everything else (building the scene, placing objects before t=0,
grading, rendering) runs on the host MjModel / MjData and is the same for every engine.

An engine holds n_worlds copies of one model and offers

  load(states)      start world i from states[i] (snapshot() of a host MjData)
  restart()         zero every world's velocities and clock, keeping its positions (t=0 after a settle)
  set_ctrl(ctrl)    (n_worlds, nu) actuator controls, held until the next set_ctrl
  step(n, worlds)   n physics steps of the given worlds (default all)
  qpos              (n_worlds, nq) positions, on the host
  ok                (n_worlds,) bool, False from the first step whose state can't be trusted

Engines:
  "mujoco"  CPU MuJoCo: one MjData per world, worlds stepped in parallel threads. Each world steps exactly as it
            would alone, so results don't depend on the number of worlds. A world goes bad on MuJoCo's instability
            warnings (mj_step has then reset it) and stops stepping there.
  "warp"    MuJoCo Warp: all worlds stepped together on the GPU (pip install armnet-digital-twins[warp]).
            physics_warp.WarpPhysics."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

ENGINES = ("mujoco", "warp")
# MuJoCo's instability warnings: after any of them mj_step has reset mjData to the model defaults
BAD_WARNINGS = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQPOS)
# what a step depends on in our scenes, besides the clock (no mocap bodies, applied forces or user data)
STATE_FIELDS = ("qpos", "qvel", "act", "qacc_warmstart", "ctrl")


def n_unstable(d) -> int:
    return sum(int(d.warning[k].number) for k in BAD_WARNINGS)


def snapshot(d) -> dict:
    """A host MjData's state, as engines load it."""
    return dict(time=float(d.time), **{f: getattr(d, f).copy() for f in STATE_FIELDS})


class MujocoPhysics:
    """threads: worker threads stepping the worlds (default: one per world, up to the CPU count)."""
    name = "mujoco"

    def __init__(self, model, n_worlds: int = 1, *, threads: int | None = None):
        self.model, self.n_worlds = model, int(n_worlds)
        self.datas = [mujoco.MjData(model) for _ in range(self.n_worlds)]
        self._warnings = np.zeros(self.n_worlds, int)
        self._ok = np.ones(self.n_worlds, bool)
        threads = min(self.n_worlds, int(threads or os.cpu_count() or 1))
        self._pool = ThreadPoolExecutor(threads) if threads > 1 else None
        self._threads = threads

    def load(self, states):
        for i, (d, s) in enumerate(zip(self.datas, states)):
            mujoco.mj_resetData(self.model, d)
            for f in STATE_FIELDS:
                getattr(d, f)[:] = s[f]
            d.time = s["time"]
            self._warnings[i] = n_unstable(d)
        self._ok[:] = True

    def restart(self):
        for d in self.datas:
            d.qvel[:] = 0.0
            d.time = 0.0

    def set_ctrl(self, ctrl):
        for d, c in zip(self.datas, np.asarray(ctrl, np.float64)):
            d.ctrl[:] = c

    def step(self, n: int, worlds=None):
        idx = [i for i in (range(self.n_worlds) if worlds is None else worlds) if self._ok[i]]
        if n <= 0 or not idx:
            return

        def run(chunk):                 # mj_step releases the GIL, so the threads step in parallel
            m = self.model
            for i in chunk:
                d, w = self.datas[i], self._warnings[i]
                for _ in range(n):
                    mujoco.mj_step(m, d)
                    if n_unstable(d) != w:
                        self._ok[i] = False
                        break

        if self._pool is None or len(idx) == 1:
            run(idx)
        else:
            list(self._pool.map(run, [idx[k::self._threads] for k in range(min(self._threads, len(idx)))]))

    @property
    def qpos(self) -> np.ndarray:
        return np.stack([d.qpos for d in self.datas])

    @property
    def ok(self) -> np.ndarray:
        return self._ok.copy()

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


def make_physics(engine: str, model, n_worlds: int = 1, **options):
    """An engine for n_worlds copies of model. options go to the engine (MujocoPhysics / WarpPhysics)."""
    if engine == "mujoco":
        return MujocoPhysics(model, n_worlds, **options)
    if engine == "warp":
        from .physics_warp import WarpPhysics
        return WarpPhysics(model, n_worlds, **options)
    raise ValueError(f"unknown engine {engine!r}; available: {ENGINES}")
