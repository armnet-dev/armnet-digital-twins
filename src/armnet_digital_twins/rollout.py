"""Running episodes: a policy in the loop with the twin, and summaries over many episodes.

A policy is any object with
  reset(seed)               called at the start of each episode, with a seed for its sampling noise
  __call__(obs, left, right) -> (left_cmd, right_cmd)
                            once per control tick: obs = {camera: image}, left / right = measured 7-D arm poses and
                            the commands in physical units (6 joint angles in rad + gripper 0..1)
A batched policy drives many worlds at once (e.g. one model forward pass for all of them):
  reset_worlds(seeds)       one seed per world
  act(worlds, obs, left, right) -> (left_cmds, right_cmds)
                            worlds: the indices of the worlds still running, obs: their observations, left / right:
                            (len(worlds), 7) poses; returns (len(worlds), 7) commands per arm"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .policies import PerWorld


@dataclass
class Episode:
    result: dict
    qpos: np.ndarray        # (ticks + 1, nq): the state at t=0 and after each tick
    commands: np.ndarray    # (ticks, 14): the command sent at each tick (left 7, right 7; physical units)
    levels: np.ndarray      # (ticks,): the verifier's level after each tick
    reset: Any = None       # the sampled resets.Reset, if the episode started from the task's reset layout


def episode_seed(seed: int, reset_index: int) -> int:
    """The sampling-noise seed of an episode: fixed by the run's seed and the reset, so reruns reproduce."""
    return int(seed) * 1_000_003 + int(reset_index)


def _batched(policy, episodes: list[int]):
    if hasattr(policy, "act") and hasattr(policy, "reset_worlds"):
        return policy
    if isinstance(policy, (list, tuple)):
        return PerWorld([policy[e] for e in episodes])
    return PerWorld([policy] if len(episodes) == 1 else [copy.deepcopy(policy) for _ in episodes])


def run_episodes(twin, policy, reset_indices, *, seed: int | None = None, worlds=None, arm_states=None,
                 staging=(0.0, 0.0), episode_s: float | None = None, stop_on_success: bool = True, on_tick=None,
                 on_episode=None) -> list[Episode]:
    """One episode per reset index, run twin.n_worlds at a time. Each starts as DigitalTwin.reset_worlds sets it up
    (worlds / arm_states: one per episode, or None), then the policy drives it for episode_s (default: the task's) or
    until the task succeeds. policy: a batched policy, a list of policies (one per episode), or one policy (copied
    for each world when several run at once). on_tick(twin, k) is called after each tick; on_episode(i, episode) as
    each batch of episodes ends, i indexing reset_indices."""
    resets = [int(r) for r in reset_indices]
    n = int(round((episode_s or twin.task.episode_s) * twin.fps))
    out = []
    for start in range(0, len(resets), twin.n_worlds):
        ix = list(range(start, min(len(resets), start + twin.n_worlds)))
        infos = twin.reset_worlds([resets[i] for i in ix], seed=seed, staging=staging,
                                  worlds=None if worlds is None else [worlds[i] for i in ix],
                                  arm_states=None if arm_states is None else [arm_states[i] for i in ix])
        pol = _batched(policy, ix)
        pol.reset_worlds([episode_seed(info["seed"], info["reset_index"]) for info in infos])
        cmds = np.zeros((n, len(ix), 14), np.float32)
        for k in range(n):
            live = [w for w in range(len(ix)) if not (twin.world_unstable[w] or (stop_on_success and twin.verifiers[w].done))]
            if not live:
                break
            obs = twin.get_observations()
            left, right = twin.get_world_joint_positions()
            cl, cr = pol.act(live, [obs[w] for w in live], left[live], right[live])
            left[live], right[live] = cl, cr
            twin.step_worlds(left, right, worlds=live)
            cmds[k, live] = np.c_[cl, cr]
            if on_tick is not None:
                on_tick(twin, k)
        for w, i in enumerate(ix):
            result = twin.result(n_ticks=n, world=w)
            ticks = result["ticks"]
            result["stopped_s"] = round(ticks / twin.fps, 2) if ticks < n else None
            ep = Episode(result=result, qpos=twin.world_qpos(w), commands=cmds[:ticks, w].copy(),
                         levels=np.array(twin.verifiers[w].history), reset=twin.reset_infos[w]["reset"])
            out.append(ep)
            if on_episode is not None:
                on_episode(i, ep)
    return out


def run_episode(twin, policy, *, reset_index: int = 0, seed: int | None = None, world: dict | None = None,
                arm_state=None, staging=(0.0, 0.0), episode_s: float | None = None, stop_on_success: bool = True,
                on_tick=None) -> Episode:
    """Reset the twin (see DigitalTwin.reset), then run the policy for episode_s (default: the task's) or until the
    task succeeds. on_tick(twin, k) is called after each tick."""
    return run_episodes(twin, policy, [reset_index], seed=seed, worlds=[world], arm_states=[arm_state], staging=staging,
                        episode_s=episode_s, stop_on_success=stop_on_success, on_tick=on_tick)[0]


def summarize(results: list[dict]) -> dict:
    """Success rate, mean score and the share of episodes reaching each stage."""
    n = len(results)
    if not n:
        return dict(episodes=0)
    stages = {k: sum(r["stages"].get(k) is not None for r in results) / n for k in results[0]["stages"]}
    return dict(episodes=n, success_rate=sum(r["success"] for r in results) / n,
                mean_score=float(np.mean([r["score"] for r in results])), stages=stages,
                unstable=sum(bool(r.get("unstable")) for r in results))
