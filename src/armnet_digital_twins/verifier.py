"""Episode grading, shared by every task. A task's verifier looks at the scene after each control tick; the episode's
score, outcome and stage times follow from that history the same way for every task."""
from __future__ import annotations

import numpy as np


class Verifier:
    """Grades one episode from the scene state after each control tick.

    A task subclasses this and defines
      levels      {value: name} of its progress levels; the highest is success, the lowest failure
                  (e.g. {1.0: "inserted", 0.5: "on holder", 0.0: "fail"})
      level(d)    the level of a state (an MjData after mj_forward, so its contacts are filled)
      stages(d)   optional named sub-goals that are true or false in a state (e.g. grasped, lifted); one is reached
                  once it holds for stage_hold_s
      details(d)  optional numbers about the final state

    The episode succeeds once the top level has held for more than success_hold_s: the operator would stop the robot
    there, so `done` turns true. Otherwise it gets the highest level held for more than half of its last second.

    Every world of a twin has its own verifier, and worlds are graded in parallel threads, so a verifier keeps its
    episode's state on itself."""

    levels: dict = {1.0: "success", 0.0: "fail"}
    success_hold_s: float = 1.0
    stage_hold_s: float = 0.2

    def __init__(self, model, fps: float):
        self.model, self.fps = model, float(fps)
        self.reset()

    def level(self, data) -> float:
        raise NotImplementedError

    def stages(self, data) -> dict:
        return {}

    def details(self, data) -> dict:
        return {}

    def reset(self):
        self.history, self.stage_history, self.final = [], {}, {}
        self.done, self._run = False, 0

    def update(self, data) -> float:
        """Grade the state after a tick. Returns its level."""
        lv = float(self.level(data))
        self.history.append(lv)
        for k, v in self.stages(data).items():
            self.stage_history.setdefault(k, []).append(bool(v))
        self.final = self.details(data)
        self._run = self._run + 1 if lv == max(self.levels) else 0
        if self._run > int(round(self.success_hold_s * self.fps)):
            self.done = True
        return lv

    def score(self, n_ticks: int | None = None) -> float:
        """The episode's grade. Ticks not run up to n_ticks (an episode cut short) count as the lowest level."""
        lv = np.asarray(self.history, float)
        if n_ticks is not None and len(lv) < n_ticks:
            lv = np.r_[lv, np.full(n_ticks - len(lv), min(self.levels))]
        top, hold, run = max(self.levels), int(round(self.success_hold_s * self.fps)), 0
        for v in lv:
            run = run + 1 if v == top else 0
            if run > hold:
                return top
        last = lv[-int(self.fps):]
        for value in sorted(self.levels, reverse=True)[:-1]:
            if len(last) and (last >= value).mean() > 0.5:
                return value
        return min(self.levels)

    def result(self, n_ticks: int | None = None) -> dict:
        """score, outcome, success; seconds spent at each level; the first time each stage and level was reached
        (None if never); the task's details of the final state."""
        s = self.score(n_ticks)
        lv = np.asarray(self.history, float)
        n_hold = max(1, int(round(self.stage_hold_s * self.fps)))
        stages = {k: _first_held(np.asarray(v, bool), n_hold, self.fps) for k, v in self.stage_history.items()}
        for value in sorted(self.levels)[1:]:
            stages[self.levels[value]] = _first_held(lv >= value, 1, self.fps)
        return dict(score=s, outcome=self.levels[s], success=s == max(self.levels),
                    time_s={self.levels[v]: round(float((lv == v).sum()) / self.fps, 2) for v in sorted(self.levels, reverse=True)[:-1]},
                    stages=stages, **self.final)


def _first_held(flags: np.ndarray, n: int, fps: float):
    run = 0
    for k, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= n:
            return round((k - n + 1) / fps, 2)
    return None
