from armnet_digital_twins import DigitalTwin, run_episode, run_episodes  # first: picks the headless GL backend
from armnet_digital_twins.tasks.insert_gear_in_gripper import SEAT_Z

import mujoco
import numpy as np
import pytest


@pytest.fixture(scope="module")
def twin():
    return DigitalTwin("insert_gear_in_gripper")


class Sweep:
    """Scripted policy: swings every joint around the rest pose and opens and closes the grippers."""

    def __init__(self, rest):
        self.rest = rest

    def reset(self, seed=None):
        self.k = 0

    def __call__(self, obs, left, right):
        s = np.sin(self.k / 20.0)
        self.k += 1
        out = []
        for p in self.rest:
            q = p.copy()
            q[:6] += 0.15 * s
            q[6] = 0.5 + 0.4 * s
            out.append(q)
        return tuple(out)


def test_scene(twin):
    for cam in twin.cell.cameras:
        assert mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0
    assert twin.model.nu == 14


def test_joint_units_roundtrip(twin):
    c = twin.cell
    n = np.array([-50.0, 0.0, 20.0, 99.0, -99.0, 10.0, 40.0])
    assert np.allclose(c.to_normalized(c.from_normalized(n)), n, atol=1e-3)


def test_reset_is_deterministic(twin):
    a = twin.reset(3)
    q = twin.qpos_history[0].copy()
    b = twin.reset(3)
    assert a["world"] == b["world"] and np.array_equal(q, twin.qpos_history[0])
    assert twin.reset(4)["world"] != a["world"]


def test_holding_the_rest_pose_is_stable(twin):
    twin.reset(0)
    left, right = twin.get_joint_positions()
    for _ in range(20):
        twin.step(left, right)
    assert not twin.unstable and twin.verifier.history == [0.0] * 20
    assert np.abs(twin.get_joint_positions()[0] - left).max() < 0.02


def test_verifier_grades_a_seated_gear_inserted(twin):
    twin.reset(0)
    m, v = twin.model, twin.verifier
    d = mujoco.MjData(m)
    d.qpos[:] = twin.qpos_history[0]
    mujoco.mj_forward(m, d)
    a = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gear_free")]
    d.qpos[a + 3:a + 7] = d.xquat[v.jaw0]                # gear axis along the jaw axis
    for _ in range(2):                                   # move the gear centre onto the pocket centre
        mujoco.mj_forward(m, d)
        c, _, slide = v.gear_in_jaw_frame(d)
        d.qpos[a:a + 3] += d.xmat[v.jaw0].reshape(3, 3) @ (np.array([slide / 2, 0.0, SEAT_Z]) - c)
    mujoco.mj_forward(m, d)
    assert v.level(d) == 1.0
    d.qpos[a + 2] += 0.01
    mujoco.mj_forward(m, d)
    assert v.level(d) < 1.0


def test_episode_score(twin):
    v = twin.verifier
    hold = int(v.success_hold_s * twin.fps)
    v.reset()
    v.history = [0.0] * 50 + [1.0] * (hold + 1) + [0.0] * 50
    assert v.score() == 1.0                              # inserted for longer than the hold at any time
    v.history = [0.0] * 50 + [0.5] * 20
    assert v.score() == 0.5                              # on the holder through the last second
    v.history = [1.0] * 5 + [0.0] * 30
    assert v.score() == 0.0
    assert v.score(n_ticks=200) == 0.0


def test_worlds_run_like_separate_twins(twin):
    resets = [0, 1, 2, 3]                                # over 3 worlds: a full batch, then one with idle worlds
    batch = DigitalTwin("insert_gear_in_gripper", n_worlds=3)
    eps = run_episodes(batch, Sweep(batch.rest_pose()), resets, episode_s=1.0)
    batch.close()
    for r, ep in zip(resets, eps):
        one = run_episode(twin, Sweep(twin.rest_pose()), reset_index=r, episode_s=1.0)
        assert ep.result["reset_index"] == r
        assert np.array_equal(ep.qpos, one.qpos) and np.array_equal(ep.levels, one.levels)
        assert np.array_equal(ep.commands, one.commands)


def test_warp_matches_mujoco(twin):
    pytest.importorskip("mujoco_warp")
    import warp as wp
    if not wp.is_cuda_available():
        pytest.skip("MuJoCo Warp needs a CUDA GPU")
    gpu = DigitalTwin("insert_gear_in_gripper", n_worlds=2, engine="warp")
    eps = run_episodes(gpu, Sweep(gpu.rest_pose()), [0, 1], episode_s=1.0)
    gpu.close()
    m = twin.model
    arms = np.r_[twin.left_arm.qadr, twin.right_arm.qadr]
    objects = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
               for j in ("gear_free", "stand_free", "jaw0_free")]
    for r, ep in zip([0, 1], eps):
        one = run_episode(twin, Sweep(twin.rest_pose()), reset_index=r, episode_s=1.0)
        assert not ep.result["unstable"] and ep.qpos.shape == one.qpos.shape
        assert np.abs(ep.qpos[:, arms] - one.qpos[:, arms]).max() < 1e-4                  # rad
        for a in objects:
            assert np.abs(ep.qpos[:, a:a + 3] - one.qpos[:, a:a + 3]).max() < 1e-4       # m
        assert np.array_equal(ep.levels, one.levels)


def test_render(twin):
    try:
        img = twin.render("top")
    except Exception as e:                               # no OpenGL backend on this machine
        pytest.skip(f"rendering unavailable: {e}")
    assert img.shape == (twin.cell.height, twin.cell.width, 3) and img.dtype == np.uint8
