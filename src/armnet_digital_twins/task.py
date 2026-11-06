"""Tasks: the objects a task adds to a cell, how a reset places them, and how an episode is graded.

A task lives in tasks/<name>/ with its reset.json (resets.ResetSpec) and assets, and subclasses Task:

    @register_task
    class MyTask(Task):
        name = "my_task"
        prompt = "Do the thing"

        def scene(self, cell) -> TaskScene: ...                # MJCF of the task's objects
        def place(self, model, data, world, cell) -> dict: ...  # put them at a reset's world poses (before t=0)
        def verifier(self, model, cell, fps) -> Verifier: ...   # grade the episode

Resets and grading then work the same for every task: DigitalTwin.reset samples the task's reset layout, places the
objects and settles the scene; DigitalTwin.step feeds every state to the verifier."""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .resets import ResetSpec

TASKS_DIR = Path(__file__).parent / "tasks"
TASKS: dict = {}


@dataclass
class TaskScene:
    """A task's part of the MuJoCo scene.

    cell_priority    geom priority for the cell's own collision geoms (table, mats, arms, fingertip pads). A task that
                     gives one object pair its own friction through geom priority sets this above that pair's, so
                     contacts with the cell keep the usual mixing.
    support_contype  contype of the table and mats (conaffinity 1), for tasks that keep some geoms off the table."""
    name: str
    defaults: str = ""
    assets: list = field(default_factory=list)
    bodies: str = ""
    excludes: list = field(default_factory=list)
    cell_priority: int = 0
    support_contype: int = 1


class Task:
    name: str = ""
    cell: str = "yam_bimanual"
    prompt: str = ""
    fps: float = 20.0
    episode_s: float = 60.0
    dataset: str | None = None      # public LeRobot dataset of real episodes of the task, if any
    # MuJoCo Warp buffers per world (physics_warp.py): nconmax / njmax / nccdmax, or {} for MuJoCo Warp's defaults.
    # A world that outgrows them is marked unstable.
    warp_buffers: dict = {}

    @property
    def root(self) -> Path:
        return TASKS_DIR / self.name

    @cached_property
    def reset_spec(self) -> ResetSpec:
        return ResetSpec.load(self.root / "reset.json")

    @cached_property
    def episode_layouts(self) -> dict:
        """{episode index: world poses} of the real dataset's episodes at t=0, if the task ships them."""
        p = self.root / "episode_layouts.json"
        return {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}

    def scene(self, cell) -> TaskScene:
        raise NotImplementedError

    def place(self, model, data, world: dict, cell) -> dict:
        raise NotImplementedError

    def verifier(self, model, cell, fps: float):
        raise NotImplementedError


def register_task(cls):
    TASKS[cls.name] = cls
    return cls


def make_task(name: str, **options) -> Task:
    if name not in TASKS and (TASKS_DIR / name / "__init__.py").exists():
        importlib.import_module(f"{__package__}.tasks.{name}")
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {sorted(p.name for p in TASKS_DIR.iterdir() if (p / '__init__.py').exists())}")
    return TASKS[name](**options)
