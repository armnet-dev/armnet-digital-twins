"""MuJoCo digital twins of armnet robot cells."""
import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")    # headless rendering; must be set before mujoco is imported

from .cell import Cell
from .physics import ENGINES
from .resets import Reset, ResetSpec
from .rollout import Episode, run_episode, run_episodes, summarize
from .task import Task, TaskScene, make_task, register_task
from .twin import DigitalTwin
from .verifier import Verifier

__version__ = "0.1.0"
__all__ = ["ENGINES", "Cell", "DigitalTwin", "Episode", "Reset", "ResetSpec", "Task", "TaskScene", "Verifier",
           "make_task", "register_task", "run_episode", "run_episodes", "summarize"]
