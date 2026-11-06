"""insert_gear_in_gripper: pick up an 18-tooth printed gear from the mat and push it down between the two toothed
jaws of a printed gripper (a rack-and-pinion jaw pair standing upright in its holder).

Objects: the holder stand, the jaw pair (jaw1 slides along the racks relative to jaw0, with a little outward play as
in the printed part) and the gear. A reset places the holder and the gear at the reset layout's poses on the mat.

Grading (InsertGearVerifier): 1 = inserted (the gear seated flat in the pocket between the racks, teeth meshed, the
holder upright), 0.5 = on the holder but not inserted, 0 = anything else. Stages: grasped (both fingertips of one arm
on the gear) and lifted (grasped and 15 mm above the mat)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import mujoco
import numpy as np

from ...cell import fmt
from ...task import Task, TaskScene, register_task
from ...verifier import Verifier

# Seated gear (gear centre in the jaw0 frame): flat on the jaw plate, centred between the racks at x = slide / 2, y = 0.
# Along the racks a gear counts while both racks still engage it.
SEAT_Z, SEAT_DZ, SEAT_TILT_DEG, SEAT_ALONG, SEAT_ACROSS = 0.054, 0.001, 5.0, 0.0125, 0.0015
HOLDER_UP_DEG = 15.0         # inserted also needs the jaw axis within this of vertical
PICK_LIFT = 0.015            # lifted: gear this far above resting on the mat
RACK1_Y = 0.0155             # jaw1's rack pitch line in the jaw1 frame
JAW_DY = 0.003533            # each jaw's offset toward the other, centring both racks on the gear's pitch circle
JAW_CLOSED_SLIDE = 0.04286   # jaw slide where the two gripping faces meet
# The assembled holder relative to its centre on the mat: stand and jaw0 poses (x y z, quat w x y z) and the jaw slide
# as the pair rests in the stand.
HOLDER_POSE = {"stand": [0.0, 0.0, -1.9420931152625686e-07, 0.9999999999998005, -1.6991948009438529e-09, -6.315300182076619e-07, 0.0],
               "jaw0": [-0.0002036263784216441, -0.021463267410539787, 0.0049980883167455815, 0.7070306622986701,
                        7.734261211374802e-06, -9.831952564929885e-06, 0.7071828917705932]}
HOLDER_SLIDE = 0.04292573760811399
# Mass properties of the printed parts (body frames): each part's mesh inertia scaled to its weighed mass, gear 2 g,
# stand and each jaw 11 g. Their collision geoms carry no mass.
INERTIAL = {
    "gear": '<inertial pos="6.60608e-08 6.7165e-20 0.002828" mass="0.002" fullinertia="9.837e-08 9.837e-08 1.854e-07 9.538e-25 -2.219e-13 8.097e-25"/>',
    "stand": '<inertial pos="-6.30929e-05 -0.000110217 0.0170998" mass="0.011" fullinertia="3.243e-06 4.254e-06 3.229e-06 -1.928e-11 3.049e-09 1.853e-09"/>',
    "jaw0": '<inertial pos="0.0239114 -0.00222816 0.0408008" mass="0.011" fullinertia="4.917e-06 2.818e-06 3.648e-06 -3.056e-07 1.013e-07 2.993e-07"/>',
    "jaw1": '<inertial pos="-0.0239114 0.00576116 0.0408007" mass="0.011" fullinertia="4.917e-06 2.818e-06 3.648e-06 -3.056e-07 -1.013e-07 -2.993e-07"/>',
}
GEAR_CLEAR_STEP, GEAR_CLEAR_MARGIN, GEAR_CLEAR_MAX = 0.002, 0.001, 0.08
RGBA_JAW, RGBA_STAND, RGBA_GEAR = (0.07, 0.07, 0.07, 1), (0.93, 0.93, 0.93, 1), (0.06, 0.06, 0.06, 1)


def _quat_z(yaw):
    return (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))


@register_task
class InsertGearInGripper(Task):
    """Options (all physical, applied before t=0):
      friction              sliding, torsional and rolling friction of the printed parts
      gear_holder_friction  friction of the gear against the holder (None: `friction`)
      holder_mat_friction   friction of the holder's stand on the mat (None: the mat's)
      holder_play_deg       how far jaw1 can lean outward as a gear is pushed in (0: a rigid pair)
      holder_play_mm        how far jaw1 can move away from jaw0 across the racks
      jaw_slide_friction    dry friction (N) of jaw1 sliding along the racks"""
    name = "insert_gear_in_gripper"
    prompt = "Insert a gear into the parallel gripper"
    dataset = "villekuosmanen/insert_gear_in_gripper"
    # the stand's boxes and the jaws' convex pieces overlap as bounding boxes: ~450 broadphase pairs per world, though
    # only ~35 are contacts
    warp_buffers = {"nconmax": 450, "njmax": 1200, "nccdmax": 128}

    def __init__(self, *, friction=(0.35, 0.005, 0.0001), gear_holder_friction=(0.1, 0.005, 0.0001),
                 holder_mat_friction=(0.35, 0.005, 0.0001), holder_play_deg=1.5, holder_play_mm=0.0, jaw_slide_friction=0.2):
        self.friction, self.gear_holder_friction = tuple(friction), gear_holder_friction and tuple(gear_holder_friction)
        self.holder_mat_friction = holder_mat_friction and tuple(holder_mat_friction)
        self.holder_play_deg, self.holder_play_mm, self.jaw_slide_friction = holder_play_deg, holder_play_mm, jaw_slide_friction

    # ---------------- scene ----------------
    def scene(self, cell) -> TaskScene:
        A = self.root / "assets"
        coll = sorted((A / "coll").glob("*.stl"), key=lambda p: p.name)
        assets = [f'<mesh name="{p.stem}" file="{p}" scale="0.001 0.001 0.001"/>' for p in coll]
        assets += [f'<mesh name="{n}_vis" file="{A / "vis" / (n + ".stl")}" scale="0.001 0.001 0.001"/>' for n in ("jaw0", "jaw1", "stand", "gear")]

        def coll_geoms(prefix, rgba, pos=(0, 0, 0), attr=""):
            return "".join(f'<geom class="obj_coll" type="mesh" mesh="{p.stem}" pos="{fmt(pos)}" density="0"{attr} rgba="{fmt(rgba)}"/>'
                           for p in coll if p.stem.startswith(prefix + "_"))

        gear_attr = f' priority="1" friction="{fmt(self.gear_holder_friction)}"' if self.gear_holder_friction else ""
        gear = (INERTIAL["gear"] + f'<geom class="obj_vis" type="mesh" mesh="gear_vis" rgba="{fmt(RGBA_GEAR)}"/>'
                + coll_geoms("gear", RGBA_GEAR, attr=gear_attr))
        # Stand: the printed part's 2 mm slabs, each as boxes. The bottom slab layer stands on the mat as one convex foot
        # (contype 2 keeps those slabs off the mat, which collides through contype 4).
        boxes = json.loads((A / "stand_boxes.json").read_text())
        zbot = min(b[2] - b[5] for b in boxes)
        bottom = [abs((b[2] - b[5]) - zbot) < 1e-4 for b in boxes]
        off_mat = ' contype="2" conaffinity="1"'
        stand = "".join(f'<geom class="obj_coll" type="box" pos="{b[0]:.5f} {b[1]:.5f} {b[2]:.5f}" size="{b[3]:.5f} {b[4]:.5f} {b[5]:.5f}" '
                        f'density="0" rgba="{fmt(RGBA_STAND)}"{off_mat if bot else ""}/>'
                        for b, bot in zip(boxes, bottom))
        corners = [(b[0] + sx * b[3], b[1] + sy * b[4], b[2] + sz * b[5]) for b, bot in zip(boxes, bottom) if bot
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        assets.append(f'<mesh name="stand_foot" vertex="{" ".join(f"{v:.6f}" for c in corners for v in c)}"/>')
        # the foot is the stand's only contact with the mat, so its friction is the stand-on-mat friction
        foot_attr = f' priority="3" friction="{fmt(self.holder_mat_friction)}"' if self.holder_mat_friction else ""
        stand += (f'<geom name="stand_foot" class="obj_coll" type="mesh" mesh="stand_foot" density="0" contype="0" conaffinity="4"'
                  f'{foot_attr} rgba="{fmt(RGBA_STAND)}"/>')
        jaw1_joints = (f'<joint name="jaw_slide" type="slide" axis="1 0 0" range="-0.03 {JAW_CLOSED_SLIDE}" damping="0.05" '
                       f'frictionloss="{self.jaw_slide_friction:g}" ref="0"/>')
        # The 11 g jaw is too light for MuJoCo's mass-scaled soft limits: armature on the play joints and the stiffest
        # stable limit keep its stops hard.
        lim = 'solreflimit="0.004 1" solimplimit="0.95 0.99 0.0005"'
        if self.holder_play_mm > 0:
            jaw1_joints += (f'<joint name="jaw_play_y" type="slide" axis="0 1 0" range="0 {self.holder_play_mm * 1e-3:.6f}" '
                            f'armature="0.05" damping="0.5" frictionloss="0.02" {lim}/>')
        if self.holder_play_deg > 0:
            jaw1_joints += (f'<joint name="jaw_play_roll" type="hinge" axis="1 0 0" pos="0 0 0" range="{-np.deg2rad(self.holder_play_deg):.6f} 0" '
                            f'armature="5e-5" damping="0.002" frictionloss="0.001" {lim}/>')
        bodies = f"""
    <body name="stand" pos="{fmt((0.42, -0.305, 0.0))}" quat="{fmt(_quat_z(0.0))}">
      <freejoint name="stand_free"/>
      {INERTIAL["stand"]}
      <geom class="obj_vis" type="mesh" mesh="stand_vis" rgba="{fmt(RGBA_STAND)}"/>
      {stand}
    </body>
    <body name="jaw0" pos="{fmt((0.42, -0.305, 0.0))}" quat="{fmt(_quat_z(1.5707963))}">
      <freejoint name="jaw0_free"/>
      {INERTIAL["jaw0"]}
      <geom class="obj_vis" type="mesh" mesh="jaw0_vis" pos="0 {JAW_DY:.6f} 0" rgba="{fmt(RGBA_JAW)}"/>
      {coll_geoms("jaw0", RGBA_JAW, (0, JAW_DY, 0))}
      <body name="jaw1" pos="0 {-JAW_DY:.6f} 0">
        {jaw1_joints}
        {INERTIAL["jaw1"]}
        <geom class="obj_vis" type="mesh" mesh="jaw1_vis" rgba="{fmt(RGBA_JAW)}"/>
        {coll_geoms("jaw1", RGBA_JAW)}
      </body>
    </body>
    <body name="gear" pos="{fmt((0.30, -0.20, 0.0))}" quat="{fmt(_quat_z(0.0))}">
      <freejoint name="gear_free"/>
      {gear}
    </body>
    """
        defaults = (f'<default class="obj_coll"><geom group="3" condim="4" friction="{fmt(self.friction)}" density="500.0" '
                    f'contype="1" conaffinity="1"/></default>\n    '
                    '<default class="obj_vis"><geom contype="0" conaffinity="0" group="2" density="0"/></default>')
        return TaskScene(name=self.name, defaults=defaults, assets=assets, bodies=bodies, excludes=[("jaw0", "jaw1")],
                         cell_priority=2 if self.gear_holder_friction else 0, support_contype=4)

    # ---------------- reset ----------------
    def place(self, model, data, world, cell) -> dict:
        """The holder and the gear at the reset's world poses on the mat. A gear drawn on the holder's footprint is
        moved straight away from the holder until clear of it, as an operator would set it down next to it."""
        m, d = model, data
        adr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        (hx, hy), hyaw = world["holder"]["xy"], world["holder"]["yaw"]
        Rz = np.array([[np.cos(hyaw), -np.sin(hyaw), 0], [np.sin(hyaw), np.cos(hyaw), 0], [0, 0, 1]])
        qz = np.array([np.cos(hyaw / 2), 0, 0, np.sin(hyaw / 2)])
        for n in ("stand", "jaw0"):
            p, q, a = np.array(HOLDER_POSE[n][:3]), np.array(HOLDER_POSE[n][3:]), adr(n + "_free")
            d.qpos[a:a + 3] = np.array([hx, hy, cell.mat_top]) + Rz @ p
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, qz, q)
            d.qpos[a + 3:a + 7] = out
        d.qpos[adr("jaw_slide")] = HOLDER_SLIDE
        (gx, gy), gyaw, a = world["gear"]["xy"], world["gear"]["yaw"], adr("gear_free")
        d.qpos[a:a + 3] = [gx, gy, cell.mat_top + 0.0002]
        d.qpos[a + 3:a + 7] = [np.cos(gyaw / 2), 0, 0, np.sin(gyaw / 2)]
        return dict(gear_moved_mm=self._clear_gear(m, d, np.array([hx, hy])))

    def _clear_gear(self, m, d, holder_xy) -> float:
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        g, holder = bid("gear"), {bid(n) for n in ("stand", "jaw0", "jaw1")}
        a = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gear_free")]
        moved = 0.0
        while moved < GEAR_CLEAR_MAX:
            mujoco.mj_forward(m, d)
            hit = False
            for i in range(d.ncon):
                b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
                if g in (b1, b2) and (b2 if b1 == g else b1) in holder and d.contact[i].dist < GEAR_CLEAR_MARGIN:
                    hit = True
                    break
            if not hit:
                break
            v = d.qpos[a:a + 2] - holder_xy
            n = np.linalg.norm(v)
            d.qpos[a:a + 2] += GEAR_CLEAR_STEP * (v / n if n > 1e-6 else np.array([1.0, 0.0]))
            moved += GEAR_CLEAR_STEP
        return round(moved * 1e3, 1)

    def verifier(self, model, cell, fps) -> "InsertGearVerifier":
        return InsertGearVerifier(model, fps, mat_top=cell.mat_top)


class InsertGearVerifier(Verifier):
    levels = {1.0: "inserted", 0.5: "on holder", 0.0: "fail"}

    def __init__(self, model, fps, *, mat_top):
        m = model
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        jnt = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.gear, self.jaw0, self.jaw1, self.stand = bid("gear"), bid("jaw0"), bid("jaw1"), bid("stand")
        self.tips = [(bid(f"{s}_tip_left"), bid(f"{s}_tip_right")) for s in ("left", "right")]
        self.slide_adr = m.jnt_qposadr[jnt("jaw_slide")]
        self.play_adr = {n: (m.jnt_qposadr[jnt(n)] if jnt(n) >= 0 else None) for n in ("jaw_play_y", "jaw_play_roll")}
        self.mat_top = mat_top
        super().__init__(model, fps)

    def gear_in_jaw_frame(self, d):
        """Gear centre (mid-thickness) in the jaw0 frame, measured from the midpoint between the racks; the angle
        between the gear and jaw axes (deg, either face up); the jaw slide."""
        Rj, Rg = d.xmat[self.jaw0].reshape(3, 3), d.xmat[self.gear].reshape(3, 3)
        c = Rj.T @ (d.xpos[self.gear] + 0.003 * Rg[:, 2] - d.xpos[self.jaw0])
        ay, ar = self.play_adr["jaw_play_y"], self.play_adr["jaw_play_roll"]
        dy = float(d.qpos[ay]) if ay is not None else 0.0
        th = float(d.qpos[ar]) if ar is not None else 0.0
        c[1] -= (dy + RACK1_Y * (np.cos(th) - 1) - SEAT_Z * np.sin(th)) / 2
        tilt = np.degrees(np.arccos(min(1.0, abs(float(Rg[:, 2] @ Rj[:, 2])))))
        return c, tilt, float(d.qpos[self.slide_adr])

    def level(self, d) -> float:
        m = self.model
        c, tilt, slide = self.gear_in_jaw_frame(d)
        upright = d.xmat[self.jaw0][8] >= np.cos(np.radians(HOLDER_UP_DEG))
        if (upright and abs(c[2] - SEAT_Z) <= SEAT_DZ and tilt <= SEAT_TILT_DEG and abs(c[0] - slide / 2) <= SEAT_ALONG
                and abs(c[1]) <= SEAT_ACROSS):
            return 1.0
        # on the holder: touching it, over the jaw pair's footprint and in its upper part
        if not (-0.019 <= c[0] <= slide + 0.019 and abs(c[1]) <= 0.037 and c[2] >= 0.045):
            return 0.0
        holder = (self.jaw0, self.jaw1, self.stand)
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if self.gear in (b1, b2) and (b2 if b1 == self.gear else b1) in holder:
                return 0.5
        return 0.0

    def stages(self, d) -> dict:
        b = self.model.geom_bodyid[d.contact.geom[:d.ncon]]
        other = np.where(b[:, 0] == self.gear, b[:, 1], np.where(b[:, 1] == self.gear, b[:, 0], -1))
        grasped = any(np.isin(pair, other).all() for pair in self.tips)
        lifted = grasped and d.xpos[self.gear][2] + 0.003 * d.xmat[self.gear][8] > self.mat_top + 0.003 + PICK_LIFT
        return {"grasped": bool(grasped), "lifted": bool(lifted)}

    def details(self, d) -> dict:
        c, tilt, slide = self.gear_in_jaw_frame(d)
        return dict(gear_vs_pocket_mm=[round(float(v), 1) for v in ((c[0] - slide / 2) * 1e3, c[1] * 1e3, (c[2] - SEAT_Z) * 1e3, tilt)],
                    holder_tilt_deg=round(float(np.degrees(np.arccos(np.clip(d.xmat[self.jaw0][8], -1, 1)))), 1))
