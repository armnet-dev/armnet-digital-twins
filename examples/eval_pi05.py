"""Evaluate pi0.5 on insert_gear_in_gripper in the digital twin.

Each episode starts from the task's reset for (seed, reset index), the same layout the real cell's reset prompt shows
for that seed and index, and runs for up to 60 s at 20 Hz. The policy sees the cell's three cameras and joint
states and its actions drive the arms through the cell's controller. An episode succeeds once the gear has stayed
inserted for 1 s. Episodes run --worlds at a time, the policy querying all of them in one batch. Results go to
<out>/results.json; --render also writes each episode's video and reset prompt.

  python examples/eval_pi05.py                          # 50 episodes, 10 at a time on the CPU
  python examples/eval_pi05.py --engine warp            # physics on the GPU (pip install armnet-digital-twins[warp])
  python examples/eval_pi05.py --episodes 3 --render    # a few, with videos"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from armnet_digital_twins import ENGINES, DigitalTwin, run_episodes, summarize
from armnet_digital_twins.policies import LeRobotPolicy
from armnet_digital_twins.resets import draw
from armnet_digital_twins.video import write_rollout

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--policy", default="fanqi-robo/pi05_insert_gear_in_gripper_b0", help="Hub repo or local checkpoint")
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--first-reset", type=int, default=0, help="reset index of the first episode")
ap.add_argument("--seed", type=int, default=None, help="reset and policy-noise seed (default: the task's, 42)")
ap.add_argument("--episode-s", type=float, default=None, help="episode length (default 60 s)")
ap.add_argument("--prompt", default=None, help="language instruction (default: the task's)")
ap.add_argument("--n-action-steps", type=int, default=None, help="actions run per chunk (default: the checkpoint's)")
ap.add_argument("--engine", choices=ENGINES, default="mujoco", help="physics engine")
ap.add_argument("--worlds", type=int, default=10, help="episodes run at once")
ap.add_argument("--max-batch", type=int, default=None, help="largest policy batch (default: all worlds)")
ap.add_argument("--device", default=None)
ap.add_argument("--render", action="store_true", help="write each episode's video and reset prompt image")
ap.add_argument("--out", default=None, help="output folder (default outputs/eval_pi05_<time>)")
args = ap.parse_args()

twin = DigitalTwin("insert_gear_in_gripper", n_worlds=min(args.worlds, args.episodes), engine=args.engine)
task = twin.task
seed = task.reset_spec.seed if args.seed is None else args.seed
policy = LeRobotPolicy(args.policy, twin.cell, dataset=task.dataset, prompt=args.prompt or task.prompt, device=args.device,
                       n_action_steps=args.n_action_steps, max_batch=args.max_batch)
out = Path(args.out or f"outputs/eval_pi05_{time.strftime('%Y%m%d_%H%M%S')}")
out.mkdir(parents=True, exist_ok=True)
print(f"{args.policy}: {policy.describe()}\n{args.episodes} episodes from reset {args.first_reset}, seed {seed}, "
      f"{twin.n_worlds} at a time on {args.engine} -> {out}", flush=True)

resets = list(range(args.first_reset, args.first_reset + args.episodes))
results, t_start = [], time.time()


def save(k, ep):
    i, r = resets[k], ep.result
    results.append(r)
    stages = ", ".join(f"{s} {t:.1f}s" for s, t in r["stages"].items() if t is not None) or "no stage reached"
    print(f"reset {i:3d}: {r['outcome']:9s} score {r['score']:g}  ({stages})  {time.time() - t_start:.0f} s"
          + ("  UNSTABLE" if r["unstable"] else ""), flush=True)
    if args.render:
        prompt = draw(twin.render("top", qpos=ep.qpos[0]), ep.reset, task.reset_spec.calibration, f"reset {i}  seed {seed}")
        cv2.imwrite(str(out / f"reset_{i:03d}.jpg"), prompt[:, :, ::-1])
        write_rollout(twin, ep.qpos, out / f"rollout_{i:03d}.mp4", title=f"reset {i}", levels=ep.levels)
    np.savez_compressed(out / f"rollout_{i:03d}.npz", qpos=ep.qpos, commands=ep.commands, fps=twin.fps)
    (out / "results.json").write_text(json.dumps(dict(policy=args.policy, seed=seed, prompt=policy.prompt, engine=args.engine,
                                                      worlds=twin.n_worlds, summary=summarize(results), episodes=results),
                                                 indent=1))


run_episodes(twin, policy, resets, seed=seed, episode_s=args.episode_s, on_episode=save)

s = summarize(results)
n = s["episodes"]
print(f"\n{args.policy} on {task.name}: {round(s['success_rate'] * n)}/{n} succeeded ({100 * s['success_rate']:.0f}%), "
      f"mean score {s['mean_score']:.3f}, {time.time() - t_start:.0f} s")
print("stages reached: " + ", ".join(f"{k} {round(v * n)}/{n}" for k, v in s["stages"].items()))
print(f"results: {out / 'results.json'}")
