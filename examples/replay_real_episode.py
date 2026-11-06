"""Replay a real episode of villekuosmanen/insert_gear_in_gripper in the digital twin.

The objects start where they were in the real episode, the arms at its first logged joint state, and both arms are
driven open loop by its logged actions. Prints the verifier's grade of the sim replay and writes a video with the
real and the sim camera views side by side (rows: top, left wrist, right wrist). Open loop, small differences between
the real and the sim arm add up, so most replays miss the grasp; episode 10 picks the gear up and puts it on the holder.

  python examples/replay_real_episode.py --episode 10
  python examples/replay_real_episode.py --episode 34 --seconds 15 --video outputs/ep34.mp4"""
import argparse
import json
from pathlib import Path

from armnet_digital_twins import ENGINES, DigitalTwin, run_episode
from armnet_digital_twins.datasets import LeRobotEpisode
from armnet_digital_twins.policies import ReplayPolicy
from armnet_digital_twins.video import write_side_by_side

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--episode", type=int, default=10, help="dataset episode index")
ap.add_argument("--seconds", type=float, default=None, help="replay only the first N seconds")
ap.add_argument("--engine", choices=ENGINES, default="mujoco", help="physics engine")
ap.add_argument("--video", default=None, help="output MP4 (default outputs/replay_ep<N>.mp4)")
ap.add_argument("--no-video", dest="make_video", action="store_false")
args = ap.parse_args()

twin = DigitalTwin("insert_gear_in_gripper", engine=args.engine)
task = twin.task
if args.episode not in task.episode_layouts:
    raise SystemExit(f"no t=0 layout for episode {args.episode}; episodes with one: {sorted(task.episode_layouts)}")
ep = LeRobotEpisode(task.dataset, args.episode)
assert ep.fps == twin.fps, (ep.fps, twin.fps)
n = ep.length if args.seconds is None else min(ep.length, int(round(args.seconds * ep.fps)))

print(f"replaying episode {args.episode} of {task.dataset}: {n / ep.fps:.1f} s open loop", flush=True)
episode = run_episode(twin, ReplayPolicy(ep.actions, twin.cell), reset_index=args.episode, world=task.episode_layouts[args.episode],
                      arm_state=ep.states[0], episode_s=n / ep.fps, stop_on_success=False)
r = episode.result
print(f"sim: {r['outcome']} (score {r['score']:g}); stages reached: "
      + ", ".join(f"{k} at {t:.1f} s" if t is not None else f"{k} never" for k, t in r["stages"].items()))
print("result:", json.dumps({k: r[k] for k in ("score", "outcome", "time_s", "stages", "gear_vs_pocket_mm", "holder_tilt_deg")}))

if args.make_video:
    out = Path(args.video or f"outputs/replay_ep{args.episode:02d}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    tile = (360, 640)
    print("decoding the real camera videos ...", flush=True)
    real = {c: ep.frames(c, size=tile, n=n) for c in twin.cell.cameras}
    print(f"rendering {n} sim frames per camera ...", flush=True)
    write_side_by_side(twin, episode.qpos, real, out, levels=episode.levels, tile=tile)
    print(f"wrote {out}: rows top / left_wrist / right_wrist, columns real | sim")
