"""DigitalTwin: a task in its cell as a MuJoCo simulation, driven the way the real cell is.

Each control tick (1 / task.fps) takes one physical command per arm (6 joint angles in rad + gripper 0..1), which
reaches the arm's joint controller after the cell's command delay. Camera frames come out as the cell's cameras
deliver them: colour matched, and each from its camera's latency ago. Objects move only through contact; the scene
is placed and settled before t=0 and the task's verifier grades every state after that.

A twin runs n_worlds copies of the scene in lockstep, each with its own episode, on one physics engine (physics.py):
CPU MuJoCo or MuJoCo Warp on the GPU. Only the stepping runs on the engine; building the scene, the task's placement,
grading and rendering use the host MjModel, so tasks, resets and verifiers are the same on every engine.

One world: reset, step, get_observation, get_joint_positions, verifier, result().
Many worlds: reset_worlds, step_worlds, get_observations, get_world_joint_positions, verifiers, result(world=i)."""
from __future__ import annotations

import atexit
import itertools
import math
import os
import weakref
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import cv2
import mujoco
import numpy as np

from .cell import Cell
from .physics import make_physics, snapshot
from .resets import stage
from .task import Task, make_task


class Arm:
    """One arm's joints and actuators in the scene, in the cell's physical units."""

    def __init__(self, model, cell: Cell, side: str):
        m = model
        self.cell, self.side = cell, side
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint{j + 1}") for j in range(7)]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.range = np.array([m.jnt_range[j] for j in jid[:6]])
        self.actuators = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_joint{j + 1}") for j in range(6)]
        self.actuators.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper"))
        self.offset = cell.joint_offsets[side].copy()

    def measured(self, qpos) -> np.ndarray:
        """The arm's pose in a state: 6 joint angles (rad) + gripper 0..1."""
        q = qpos[self.qadr]
        return np.r_[q[:6] - self.offset, self.cell.gripper_value(q[6])].astype(np.float32)

    def ctrl(self, pose) -> np.ndarray:
        """Its 7 actuator controls for a pose: joint targets (clamped to the joint stops) and the finger opening."""
        pose = np.asarray(pose, np.float64).reshape(7)
        return np.r_[np.clip(pose[:6] + self.offset, self.range[:, 0], self.range[:, 1]), self.cell.finger_opening(pose[6])]

    def place(self, data, pose):
        """Pre-t0: the arm at `pose` and its controller holding it there."""
        data.qpos[self.qadr[:6]] = pose[:6] + self.offset
        g = self.cell.finger_opening(pose[6])
        data.qpos[self.qadr[6]] = g
        data.qpos[self.qadr[6] + 1] = g
        data.ctrl[self.actuators] = self.ctrl(pose)


class Frames(Mapping):
    """Camera frames of one tick, rendered when first read."""

    def __init__(self, twin: "DigitalTwin", qpos: dict):
        self._twin, self._qpos, self._img = twin, qpos, {}

    def __getitem__(self, cam):
        if cam not in self._img:
            self._img[cam] = self._twin.render(cam, qpos=self._qpos[cam])
        return self._img[cam]

    def __iter__(self):
        return iter(self._qpos)

    def __len__(self):
        return len(self._qpos)


class DigitalTwin:
    """task: a task name or Task. n_worlds: copies of the scene run in lockstep. engine: "mujoco" (CPU, worlds in
    parallel threads) or "warp" (MuJoCo Warp on the GPU); engine_options go to the engine (physics.py), on top of the
    task's warp_buffers for "warp". task_options / cell_overrides: physical options of the task (its constructor
    arguments) and overrides merged into the cell's config, both applied when the scene is built. image_size:
    (height, width) of rendered camera frames, by default the cell cameras' resolution."""

    def __init__(self, task: str | Task = "insert_gear_in_gripper", *, n_worlds: int = 1, engine: str = "mujoco",
                 engine_options: dict | None = None, task_options: dict | None = None, cell_overrides: dict | None = None,
                 image_size: tuple | None = None):
        self.task = make_task(task, **(task_options or {})) if isinstance(task, str) else task
        self.cell = Cell(self.task.cell, cell_overrides)
        self.xml = self.cell.mjcf(self.task.scene(self.cell))
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.n_worlds, self.engine = int(n_worlds), engine
        options = {**(self.task.warp_buffers if engine == "warp" else {}), **(engine_options or {})}
        self.physics = make_physics(engine, self.model, self.n_worlds, **options)
        self._place_data, self._render_data = mujoco.MjData(self.model), mujoco.MjData(self.model)
        threads = min(self.n_worlds, os.cpu_count() or 1)              # worlds are graded in parallel
        self._grade_data = [mujoco.MjData(self.model) for _ in range(threads)]
        self._grade_pool = ThreadPoolExecutor(threads) if threads > 1 else None
        self.fps = float(self.task.fps)
        self.command_delay_s = self.cell.command_delay_s
        self.left_arm, self.right_arm = Arm(self.model, self.cell, "left"), Arm(self.model, self.cell, "right")
        self._actuators = self.left_arm.actuators + self.right_arm.actuators
        self.verifiers = [self.task.verifier(self.model, self.cell, self.fps) for _ in range(self.n_worlds)]
        h = self.model.opt.timestep
        self.substeps = math.ceil((1.0 / self.fps - 1e-9) / h)          # physics steps per tick
        self.latency_ticks = {c: math.ceil((s - 1e-6) / (self.substeps * h)) for c, s in self.cell.camera_latency_s.items()}
        # A command reaches the controller `lag` physics steps after its tick starts, so step j of a tick runs the
        # command sent `back` ticks earlier. Runs of steps with the same command are (back, steps) segments.
        lag = math.ceil((self.command_delay_s - 1e-9) / h) if self.command_delay_s > 0 else 0
        backs = [max(0, math.ceil((lag - j) / self.substeps)) for j in range(self.substeps)]
        self._segments = [(b, len(list(g))) for b, g in itertools.groupby(backs)]
        self._sent = np.zeros((max(backs) + 1, self.n_worlds, self.model.nu))     # controls of recent ticks, newest first
        self.image_size = tuple(image_size or (self.cell.height, self.cell.width))
        self._renderer = None
        self._history = [[self.model.qpos0.copy()] for _ in range(self.n_worlds)]   # per world: qpos at t=0, after each tick
        self.reset_infos = [None] * self.n_worlds
        self.in_episode = np.zeros(self.n_worlds, bool)
        self.world_unstable = np.zeros(self.n_worlds, bool)

    # ---------------- reset (before t=0) ----------------
    def reset_worlds(self, reset_indices: Sequence[int] | None = None, *, seed: int | None = None, worlds=None,
                     arm_states=None, staging: tuple = (0.0, 0.0), settle_steps: int = 1500) -> list[dict]:
        """Start an episode in each of the first len(reset_indices) worlds (default: reset i in world i, in every
        world); any others stay idle. worlds / arm_states: one per episode, as reset's world and arm_state (None
        entries take the default). The worlds settle together. Returns what was placed in each."""
        resets = list(range(self.n_worlds)) if reset_indices is None else [int(r) for r in reset_indices]
        n = len(resets)
        if not 0 < n <= self.n_worlds:
            raise ValueError(f"{n} resets for a twin of {self.n_worlds} worlds")
        worlds = [None] * n if worlds is None else list(worlds)
        arm_states = [None] * n if arm_states is None else list(arm_states)
        spec = self.task.reset_spec
        seed = spec.seed if seed is None else int(seed)
        m, d = self.model, self._place_data
        states, placed = [], []
        for reset_index, world, arm_state in zip(resets, worlds, arm_states):
            reset = None
            if world is None:
                reset = spec.sample(self.cell.top_camera, self.cell.mat_top, reset_index=reset_index, seed=seed)
                world = reset.world
            world = stage(world, xy_sd=staging[0], yaw_sd=staging[1], seed=seed, reset_index=reset_index)
            mujoco.mj_resetData(m, d)
            if arm_state is None:
                poses = self.rest_pose()
            else:
                st = np.asarray(arm_state, float)
                poses = self.cell.from_normalized(st[:7]), self.cell.from_normalized(st[7:])
            for arm, pose in zip((self.left_arm, self.right_arm), poses):
                arm.place(d, pose)
            info = self.task.place(m, d, world, self.cell)
            mujoco.mj_forward(m, d)
            states.append(snapshot(d))
            placed.append((reset_index, world, reset, info))
        states += [states[0]] * (self.n_worlds - n)                    # idle worlds hold a copy of the first
        p = self.physics
        p.load(states)
        p.step(settle_steps, range(n))
        settled = p.ok
        p.restart()
        q = p.qpos
        for i in range(self.n_worlds):
            self._history[i] = [q[i].copy()]
            self.verifiers[i].reset()
        self._sent[:] = np.stack([s["ctrl"] for s in states])
        self.in_episode[:] = np.arange(self.n_worlds) < n
        self.world_unstable[:] = self.in_episode & ~settled
        self.reset_infos = [dict(reset_index=r, seed=seed, world=w, settle_ok=bool(settled[i]), reset=rs, **info)
                            for i, (r, w, rs, info) in enumerate(placed)] + [None] * (self.n_worlds - n)
        return self.reset_infos[:n]

    def reset(self, reset_index: int = 0, *, seed: int | None = None, world: dict | None = None,
              arm_state=None, staging: tuple = (0.0, 0.0), settle_steps: int = 1500) -> dict:
        """Start an episode. The objects go to the task's reset layout for (seed, reset_index), or to `world`
        ({object: {"xy": [x, y], "yaw": rad}}) if given; staging = (m, rad) of Gaussian operator placement noise per
        object. The arms start at the cell's rest pose, or at arm_state (14 values in the normalized dataset frame:
        left 6 joints + gripper, right 6 joints + gripper). The scene then settles for settle_steps physics steps and
        the clock starts at 0. Returns what was placed."""
        self._one_world()
        return self.reset_worlds([reset_index], seed=seed, worlds=[world], arm_states=[arm_state], staging=staging,
                                 settle_steps=settle_steps)[0]

    def rest_pose(self) -> tuple[np.ndarray, np.ndarray]:
        out = []
        for arm, pose in zip((self.left_arm, self.right_arm), self.cell.rest_pose()):
            pose = pose.copy()
            pose[:6] = np.clip(pose[:6], arm.range[:, 0], arm.range[:, 1])
            out.append(pose)
        return tuple(out)

    # ---------------- running ----------------
    def step_worlds(self, left, right, worlds: Sequence[int] | None = None) -> np.ndarray:
        """One control tick of the given worlds (default: every world in an episode that hasn't gone unstable):
        left / right (n_worlds, 7) physical commands, clipped to the cell's limits as the cell does, then 1 / fps of
        physics. The other worlds hold still and their commands are ignored. Returns each world's verifier level of
        its new state (NaN for worlds not stepped)."""
        run = [int(i) for i in (range(self.n_worlds) if worlds is None else worlds)
               if self.in_episode[i] and not self.world_unstable[i]]
        sent = self._sent[0].copy()
        for i in run:
            sent[i, self._actuators] = np.r_[self.left_arm.ctrl(self.cell.clip_command(left[i])),
                                             self.right_arm.ctrl(self.cell.clip_command(right[i]))]
        self._sent = np.concatenate([sent[None], self._sent[:-1]])
        for back, steps in self._segments:
            self.physics.set_ctrl(self._sent[back])
            self.physics.step(steps, run)
        q, ok = self.physics.qpos, self.physics.ok
        for i in run:
            self._history[i].append(q[i].copy())
            self.world_unstable[i] = not ok[i]
        levels = np.full(self.n_worlds, np.nan)
        T = len(self._grade_data)

        def grade(k):                                 # mj_forward releases the GIL
            for i in run[k::T]:
                levels[i] = self.grade(q[i], world=i, data=self._grade_data[k])

        if self._grade_pool is None or len(run) == 1:
            for i in run:
                levels[i] = self.grade(q[i], world=i)
        else:
            list(self._grade_pool.map(grade, range(min(T, len(run)))))
        return levels

    def step(self, left, right) -> float:
        """One control tick: the two physical commands (clipped to the cell's limits, as the cell does), then
        1 / fps of physics. Returns the verifier's level of the new state."""
        self._one_world()
        if not self.in_episode[0]:
            raise RuntimeError("reset() before step()")
        return float(self.step_worlds(np.asarray(left)[None], np.asarray(right)[None])[0])

    def grade(self, qpos, *, world: int = 0, data=None) -> float:
        """Feed a state to a world's verifier (on a separate MjData, so the simulation itself is untouched)."""
        g = self._grade_data[0] if data is None else data
        g.qpos[:] = qpos
        mujoco.mj_forward(self.model, g)
        return self.verifiers[world].update(g)

    def get_observations(self) -> list[Frames]:
        """Per world, {camera: HxWx3 uint8 RGB} as the cell delivers them: each frame from its camera's latency ago."""
        return [Frames(self, {c: h[max(0, len(h) - 1 - self.latency_ticks[c])] for c in self.cell.cameras}) for h in self._history]

    def get_world_joint_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """(left, right): every world's measured 7-D arm poses, (n_worlds, 7), physical units."""
        q = [h[-1] for h in self._history]
        return np.stack([self.left_arm.measured(x) for x in q]), np.stack([self.right_arm.measured(x) for x in q])

    def get_observation(self) -> Frames:
        """{camera: HxWx3 uint8 RGB} as the cell delivers them: each frame from its camera's latency ago."""
        self._one_world()
        return self.get_observations()[0]

    def get_joint_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """(left, right) measured 7-D arm poses, physical units."""
        self._one_world()
        left, right = self.get_world_joint_positions()
        return left[0], right[0]

    # ---------------- state and results ----------------
    def world_qpos(self, world: int = 0) -> np.ndarray:
        """(ticks + 1, nq): a world's state at t=0 and after each tick of its episode."""
        return np.array(self._history[world])

    @property
    def world_ticks(self) -> np.ndarray:
        return np.array([len(h) - 1 for h in self._history])

    @property
    def world_done(self) -> np.ndarray:
        """Per world: its task succeeded (held for the verifier's success_hold_s) or its physics went unstable."""
        return np.array([v.done for v in self.verifiers]) | self.world_unstable

    @property
    def level_names(self) -> dict:
        """{level: name} of the task's verifier."""
        return self.verifiers[0].levels

    def result(self, n_ticks: int | None = None, *, world: int = 0) -> dict:
        r = self.verifiers[world].result(n_ticks)
        info = {k: v for k, v in (self.reset_infos[world] or {}).items() if k != "reset"}
        return dict(**info, **r, ticks=len(self._history[world]) - 1, unstable=bool(self.world_unstable[world]))

    def _one_world(self):
        if self.n_worlds != 1:
            raise RuntimeError(f"this twin runs {self.n_worlds} worlds: use reset_worlds, step_worlds, get_observations "
                               "and get_world_joint_positions")

    @property
    def verifier(self):
        self._one_world()
        return self.verifiers[0]

    @property
    def unstable(self) -> bool:
        self._one_world()
        return bool(self.world_unstable[0])

    @property
    def done(self) -> bool:
        """The task succeeded (held for the verifier's success_hold_s) or the physics went unstable."""
        self._one_world()
        return bool(self.world_done[0])

    @property
    def reset_info(self) -> dict | None:
        self._one_world()
        return self.reset_infos[0]

    @property
    def qpos_history(self) -> list:
        self._one_world()
        return self._history[0]

    @property
    def ticks(self) -> int:
        self._one_world()
        return len(self._history[0]) - 1

    @property
    def time(self) -> float:
        return self.ticks * self.substeps * float(self.model.opt.timestep)

    # ---------------- rendering ----------------
    def render(self, camera: str | mujoco.MjvCamera = "top", qpos=None, size: tuple | None = None, *, world: int = 0) -> np.ndarray:
        """A camera image (HxWx3 uint8 RGB) of a world's current state or of `qpos`. Cell cameras are colour matched
        to the real ones. camera: a cell camera name, "overview" (behind the arms) or a mujoco.MjvCamera. size:
        (height, width) to resize to."""
        if self._renderer is None:                 # one Renderer per process: several corrupt each other on EGL
            self._renderer = mujoco.Renderer(self.model, *self.image_size)
            # Renderer.__del__ during interpreter shutdown raises EGL errors. atexit runs hooks last in, first out, so
            # this one closes the renderer before MuJoCo's own hook (registered with its first context) ends EGL.
            ref = weakref.ref(self._renderer)
            atexit.register(lambda: (r := ref()) is not None and r.close())
        rd = self._render_data
        rd.qpos[:] = self._history[world][-1] if qpos is None else qpos
        mujoco.mj_kinematics(self.model, rd)
        mujoco.mj_camlight(self.model, rd)
        cam = overview_camera() if isinstance(camera, str) and camera == "overview" else camera
        self._renderer.update_scene(rd, camera=cam)
        img = self._renderer.render()
        if isinstance(camera, str) and camera in self.cell.cameras:
            img = self.cell.color_correct(img, camera)
        if size is not None and tuple(size) != img.shape[:2]:
            img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        return img

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._grade_pool is not None:
            self._grade_pool.shutdown()
            self._grade_pool = None
        self.physics.close()


def overview_camera() -> mujoco.MjvCamera:
    """A free camera high behind the two arms, at an angle that keeps the mast off the workspace."""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.3, -0.25, 0.08]
    cam.distance, cam.azimuth, cam.elevation = 1.3, 20.0, -40.0
    return cam
