"""The bimanual YAM cell: its calibrated geometry, cameras and controller, the conversions between the cell's joint
units, and its part of the MuJoCo scene.

Joint units. The robot reports and is commanded in physical units: 6 joint angles (rad) and the gripper opening 0..1
per arm. LeRobot datasets of the cell store the normalized frame instead: each joint's trusted range [lo, hi] mapped
onto -100..100 and the gripper onto 0..100. In the sim each joint angle is the reported one plus the arm's calibrated
joint-zero offset, and the finger opening is linear in the gripper value."""
from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .camera import PinholeCamera

CELLS_DIR = Path(__file__).parent / "cells"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "wrist_yaw")
SIDES = ("left", "right")
CAMERAS = ("top", "left_wrist", "right_wrist")

ARM_RGBA = {"base": "0.08 0.08 0.09 1", "link1": "0.08 0.08 0.09 1", "link2": "0.92 0.92 0.92 1",
            "link3": "0.92 0.92 0.92 1", "link4": "0.08 0.08 0.09 1", "link5": "0.1 0.1 0.11 1",
            "gripper": "0.1 0.1 0.11 1", "tip_left": "0.06 0.06 0.06 1", "tip_right": "0.06 0.06 0.06 1",
            "d405_wrist_linear_4310": "0.08 0.08 0.09 1"}
COLLISION_MESHES = ("link2", "link3", "link4", "link5", "gripper")
# controller "gravcomp" group of each arm body
GRAVCOMP_GROUP = {"link1": "link1", "link2": "link2", "link3": "link3", "link4": "link4", "link5": "link5",
                  "gripper": "gripper", "tip_left": "gripper", "tip_right": "gripper", "camera": "gripper"}


def fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Cell:
    """A robot cell loaded from cells/<name>/cell.json. `overrides` is merged into that config (nested dicts merge),
    e.g. Cell(overrides={"controller": {"delay_s": 0.06}})."""

    def __init__(self, name: str = "yam_bimanual", overrides: dict | None = None):
        self.name, self.root = name, CELLS_DIR / name
        self.config = _merge(json.loads((self.root / "cell.json").read_text()), overrides or {})
        c = self.config
        self.table_z = float(c["table"]["z"])
        self.mat_top = self.table_z + float(c["table"]["mat_thickness"])
        self.cameras = CAMERAS
        self.width, self.height = c["cameras"]["resolution"]
        self.top_camera = PinholeCamera(c["cameras"]["top"], self.width, self.height)
        self.camera_latency_s = dict(c["cameras"]["latency_s"])
        self.command_delay_s = float(c["controller"]["delay_s"])
        self.joint_limits = np.array([c["arms"]["joint_limits"][j] for j in JOINTS], float)        # (6, 2) rad
        self.joint_offsets = {s: np.deg2rad(c["arms"]["joint_zero_offset_deg"][s]) for s in SIDES}
        g = c["arms"]["gripper"]
        self.grip_m_per_unit, self.grip_offset, self.finger_max = g["m_per_unit"], g["offset_m"], g["max_m"]
        self._lut = {cam: np.array(v, np.uint8) for cam, v in json.loads((self.root / "camera_lut.json").read_text()).items()}

    # ---------------- joint units ----------------
    def to_normalized(self, pose) -> np.ndarray:
        """Physical 7-D arm pose -> the normalized frame of the cell's datasets."""
        p = np.asarray(pose, np.float64).reshape(7)
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        return np.r_[np.clip((p[:6] - lo) / (hi - lo), 0, 1) * 200.0 - 100.0, np.clip(p[6], 0, 1) * 100.0]

    def from_normalized(self, values) -> np.ndarray:
        """Normalized 7-D arm pose -> physical (rad, gripper 0..1)."""
        v = np.asarray(values, np.float64).reshape(7)
        lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
        return np.r_[lo + np.clip((v[:6] + 100.0) / 200.0, 0, 1) * (hi - lo), np.clip(v[6] / 100.0, 0, 1)].astype(np.float32)

    def clip_command(self, pose) -> np.ndarray:
        """A physical command clipped to the cell's trusted joint limits and the gripper's 0..1, as the cell does."""
        out = np.asarray(pose, np.float32).reshape(7).copy()
        out[:6] = np.clip(out[:6], self.joint_limits[:, 0], self.joint_limits[:, 1])
        out[6] = np.clip(out[6], 0.0, 1.0)
        return out

    def finger_opening(self, gripper: float) -> float:
        """Gripper 0..1 -> each finger's slide (m)."""
        return float(np.clip(np.clip(gripper, 0, 1) * 100.0 * self.grip_m_per_unit + self.grip_offset, 0, self.finger_max))

    def gripper_value(self, finger: float) -> float:
        """Each finger's slide (m) -> gripper 0..1."""
        return float(np.clip((finger - self.grip_offset) / self.grip_m_per_unit / 100.0, 0.0, 1.0))

    def rest_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Where both arms start an episode (physical units): the cell's pose after a reset."""
        r = np.asarray(self.config["arms"]["rest_pose_normalized"], float)
        return self.from_normalized(r[:7]), self.from_normalized(r[7:])

    def color_correct(self, img: np.ndarray, camera: str) -> np.ndarray:
        """A rendered frame through the camera's calibrated colour curves."""
        lut = self._lut.get(camera)
        if lut is None:
            return img.copy()
        return np.stack([lut[ch][img[..., ch]] for ch in range(3)], -1)

    # ---------------- scene ----------------
    def mjcf(self, task_scene) -> str:
        """The full MuJoCo scene: this cell plus a task's objects (tasks.TaskScene)."""
        c, A = self.config, self.root / "assets"
        prio = f' priority="{task_scene.cell_priority}"' if task_scene.cell_priority else ""
        support = f'contype="{task_scene.support_contype}" conaffinity="1"'
        visual_lights = ('<quality shadowsize="1024"/>\n    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" '
                         'specular="0.1 0.1 0.1"/>')
        world_lights = ('<light name="sun" pos="0.5 -0.3 2.0" dir="-0.2 0 -1" directional="true" diffuse="0.7 0.7 0.7" '
                        'specular="0.2 0.2 0.2" castshadow="false"/>\n    <light name="fill" pos="-0.5 -0.3 1.5" '
                        'dir="0.5 0 -1" directional="false" diffuse="0.35 0.35 0.35" castshadow="false"/>')
        meshes = [f'<mesh name="{n}" file="{A / (n + ".stl")}"/>' for n in COLLISION_MESHES]
        meshes += [f'<mesh name="{p.stem}" file="{p}" scale="1 1 1"/>' for p in self._pads()]
        meshes += [f'<mesh name="{p.stem}_vis" file="{p}"/>' for p in sorted((A / "vis").glob("*.stl"))]
        t = c["table"]
        rep = 1.0 / t["mat_texture_size"]
        mat_assets = (f'<texture name="mattex" type="2d" file="{A / t["mat_texture"]}"/>'
                      f'<material name="mat_mat" texture="mattex" texrepeat="{rep:.3f} {rep:.3f}" texuniform="true" '
                      f'rgba="{fmt([min(1.0, v / 0.78) for v in t["mat_rgba"][:3]] + [1])}" reflectance="0"/>')
        table = (f'<geom name="table" type="box" size="{(1.3 - t["x0"]) / 2:.4f} 1.0 0.02" pos="{(1.3 + t["x0"]) / 2:.4f} '
                 f'-0.305 {self.table_z - 0.02}" material="table_mat" friction="{fmt(t["friction"])}" condim="4"{prio} {support}/>')
        ac = c["controller"]
        acts = []
        for side in SIDES:
            kp = [ac["eta"][side] * v for v in ac["kp"]]
            kd = [ac["eta"][side] * v for v in ac["kd"]]
            for j in range(6):
                acts.append(f'<general name="{side}_joint{j + 1}" joint="{side}_joint{j + 1}" biastype="affine" '
                            f'gainprm="{kp[j]:.6g}" biasprm="0 -{kp[j]:.6g} -{kd[j]:.6g}" ctrllimited="false"/>')
            g = c["arms"]["gripper"]
            acts.append(f'<general name="{side}_gripper" joint="{side}_joint7" biastype="affine" gainprm="{g["kp"]}" '
                        f'biasprm="0 -{g["kp"]} -{g["kd"]}" forcerange="-{g["force"]} {g["force"]}" ctrllimited="false"/>')
        sensors = [f'<{k} name="{side}_tip_{tip}_{k}" site="{side}_tip_{tip}_ft"/>'
                   for side in SIDES for tip in SIDES for k in ("force", "torque")]
        excludes = []
        for s in SIDES:
            excludes += [(f"{s}_gripper", f"{s}_tip_left"), (f"{s}_gripper", f"{s}_tip_right"), (f"{s}_tip_left", f"{s}_tip_right")]
        for a, b in (("link5", "gripper"), ("link4", "link5"), ("link3", "link4"), ("link2", "link3")):
            excludes += [(f"{s}_{a}", f"{s}_{b}") for s in SIDES]
        excludes += list(task_scene.excludes)
        ts = c["solref_timeconst"]
        pad_f = fmt(c["arms"]["pad_friction"])
        nl = "\n"
        return f"""<mujoco model="{self.name}_{task_scene.name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{c['timestep']}" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="0" gravity="0 0 -9.81"/>
  <size memory="64M"/>
  <visual>
    <global offwidth="{self.width}" offheight="{self.height}"/>
    <map znear="0.005" zfar="20"/>
    {visual_lights}
  </visual>
  <default>
    <geom solref="{ts} 1" solimp="0.95 0.995 0.001"/>
    <default class="arm_visual"><geom type="mesh" contype="0" conaffinity="0" group="2"/></default>
    <default class="arm_coll"><geom type="mesh" group="3" contype="1" conaffinity="1" condim="{4 if prio else 3}" friction="0.6 0.005 0.0001" rgba="0.5 0.5 0.5 0.3"{prio}/></default>
    <default class="pad_coll"><geom type="mesh" group="3" contype="1" conaffinity="1" condim="{c['arms']['pad_condim']}" friction="{pad_f}" rgba="0.3 0.6 0.9 0.3" solref="{ts} 1" solimp="0.95 0.995 0.001" margin="0.0003"{prio}/></default>
    {task_scene.defaults}
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.65 0.7" rgb2="0.3 0.32 0.35" width="256" height="256"/>
    <texture name="tabletex" type="2d" builtin="flat" rgb1="0.95 0.95 0.94" width="64" height="64"/>
    <material name="table_mat" texture="tabletex" reflectance="0.05"/>
    <material name="emit" emission="1" rgba="1 1 1 1"/>
    {mat_assets}
    {nl.join(meshes + list(task_scene.assets))}
  </asset>
  <worldbody>
    {world_lights}
    {table}
    {self._room_xml(prio, support)}
    {self._robot_xml()}
    {self._top_camera_xml()}
    {task_scene.bodies}
  </worldbody>
  <equality>
    <joint joint1="left_joint7" joint2="left_joint8" polycoef="0 1 0 0 0"/>
    <joint joint1="right_joint7" joint2="right_joint8" polycoef="0 1 0 0 0"/>
  </equality>
  <contact>
    {nl.join(f'<exclude body1="{a}" body2="{b}"/>' for a, b in excludes)}
  </contact>
  <actuator>
    {nl.join(acts)}
  </actuator>
  <sensor>
    {nl.join(sensors)}
  </sensor>
</mujoco>
"""

    def _pads(self):
        return [p for s in SIDES for p in sorted((self.root / "assets" / "pads").glob(f"tip_{s}_*.stl"))]

    def _robot_xml(self) -> str:
        """Both arms: visual meshes, convex collision meshes (fingertip pads as many convex pieces), the cell's
        link masses, wrist-camera payloads, joint friction and gravity compensation, wrist cameras, fingertip sites."""
        c = self.config
        left_base = ET.parse(self.root / "robot.xml").getroot().find("worldbody").find("body")
        pads = {s: sorted((self.root / "assets" / "pads").glob(f"tip_{s}_*.stl")) for s in SIDES}
        armature = c["arms"]["armature"]

        def process(body):
            for jn in body.findall("joint"):
                nm = jn.get("name", "")
                if nm[-1:].isdigit() and int(nm[-1]) <= 6 and ("left_joint" in nm or "right_joint" in nm):
                    jn.set("armature", str(armature[int(nm[-1]) - 1]))
                    jn.set("damping", "0.05")
                elif nm.endswith("joint7") or nm.endswith("joint8"):
                    jn.set("armature", "0.01")
                    jn.set("damping", "1.0")
            for g in list(body.findall("geom")):
                mesh = g.get("mesh")
                g.set("class", "arm_visual")
                g.set("rgba", ARM_RGBA.get(mesh, g.get("rgba")))
                if mesh in ("tip_left", "tip_right"):
                    for f in pads[mesh.split("_")[1]]:
                        ET.SubElement(body, "geom", {"class": "pad_coll", "type": "mesh", "mesh": f.stem,
                                                     "pos": g.get("pos"), "quat": g.get("quat")})
                elif mesh in COLLISION_MESHES:
                    col = copy.deepcopy(g)
                    col.set("class", "arm_coll")
                    col.attrib.pop("rgba", None)
                    body.append(col)
                if mesh == "d405_wrist_linear_4310":       # black bracket, grey camera body
                    g.set("mesh", "d405_wrist_mount_vis")
                    part = copy.deepcopy(g); part.set("mesh", "d405_wrist_body_vis"); part.set("rgba", "0.80 0.80 0.81 1"); body.append(part)
                elif mesh == "link3":                       # white forearm, black elbow plates and wrist-end housing
                    g.set("mesh", "link3_arm_vis")
                    for n in ("link3_plates_vis", "link3_wrist_vis"):
                        part = copy.deepcopy(g); part.set("mesh", n); part.set("rgba", "0.08 0.08 0.09 1"); body.append(part)
                elif mesh:
                    g.set("mesh", mesh + "_vis")
            for child in body.findall("body"):
                process(child)

        process(left_base)
        self._controller_xml(left_base)
        a = c["arms"]
        rbase = next(b for b in left_base.iter("body") if b.get("name") == "right_base")
        rbase.set("pos", f"{a['right_base_dx']:.5f} {-a['separation']:.5f} 0")
        if a["right_base_yaw"]:
            y = a["right_base_yaw"]
            rbase.set("quat", f"{np.cos(y / 2):.8f} 0 0 {np.sin(y / 2):.8f}")
        cams = c["cameras"]
        for side in SIDES:
            name = f"{side}_wrist"
            b = next(x for x in left_base.iter("body") if x.get("name") == f"{side}_camera")
            off = cams["wrist_offsets"].get(name, {})
            q = np.array([0.0, 1.0, 0.0, 0.0])     # optical frame (+z forward, +y down) -> MuJoCo camera (-z forward, +y up)
            if name in cams["wrist_rot"]:
                xq = (Rotation.from_quat([q[1], q[2], q[3], q[0]]) * Rotation.from_euler("xyz", cams["wrist_rot"][name])).as_quat()
                q = np.array([xq[3], xq[0], xq[1], xq[2]])
            ET.SubElement(b, "camera", {"name": name, "pos": off.get("pos", "0 0 0"), "quat": off.get("quat", fmt(q)),
                                        "fovy": f"{off.get('fovy', cams['wrist_fovy'])}", "resolution": f"{self.width} {self.height}"})
        for side in SIDES:
            for tip in SIDES:
                b = next(x for x in left_base.iter("body") if x.get("name") == f"{side}_tip_{tip}")
                ET.SubElement(b, "site", {"name": f"{side}_tip_{tip}_ft", "pos": "0 0 0", "size": "0.004", "rgba": "0 1 0 0.3"})
        for site in left_base.iter("site"):
            site.set("group", "4")
        return ET.tostring(left_base, encoding="unicode")

    def _controller_xml(self, left_base):
        """The cell's controller and arm dynamics: link masses x mass, a payload body at each wrist camera,
        body gravity compensation (the controller's, scaled by eta like every motor torque), joint friction."""
        ac = self.config["controller"]
        bodies = {b.get("name"): b for b in left_base.iter("body")}
        for side in SIDES:
            s, eta = ac["mass"][side], ac["eta"][side]
            for n, grp in GRAVCOMP_GROUP.items():
                b = bodies[f"{side}_{n}"]
                inert = b.find("inertial")
                inert.set("mass", f"{float(inert.get('mass')) * s:.6g}")
                inert.set("diaginertia", " ".join(f"{float(v) * s:.6g}" for v in inert.get("diaginertia").split()))
                b.set("gravcomp", f"{eta * ac['gravcomp'][grp] / s:.6g}")
            cam = bodies[f"{side}_camera"]
            mp = max(float(ac["payload"][side]), 1e-6)
            pb = ET.SubElement(cam, "body", name=f"{side}_payload")
            ET.SubElement(pb, "inertial", pos=cam.find("inertial").get("pos"), mass=f"{mp:.6g}",
                          diaginertia=" ".join([f"{mp * 2e-4:.6g}"] * 3))
            for j in range(6):
                jn = next(x for x in left_base.iter("joint") if x.get("name") == f"{side}_joint{j + 1}")
                jn.set("damping", f"{ac['damp'][j]:.6g}")
                jn.set("frictionloss", f"{ac['fric'][j]:.6g}")

    def _top_camera_xml(self) -> str:
        """The top camera at its calibrated pose, on a telescopic pole standing on the table under the camera body."""
        cam = self.top_camera
        cq = np.zeros(4)
        mujoco.mju_mat2Quat(cq, cam.R.ravel())
        box_h, box_z = 0.0125, 0.013
        mount = cam.pos + cam.R @ np.array([0.0, -box_h, box_z])
        ball = mount - np.array([0, 0, 0.016])
        top = ball[2] - 0.022
        secs = [(0.0, 0.45 * top, 0.013), (0.45 * top, 0.75 * top, 0.011), (0.75 * top, top, 0.0095)]
        blk, red, nc = "0.08 0.08 0.08 1", "0.75 0.08 0.08 1", 'contype="0" conaffinity="0" group="2"'
        pole = "".join(f'<geom type="cylinder" fromto="0 0 {a:.4f} 0 0 {b:.4f}" size="{r}" rgba="{blk}" {nc}/>' for a, b, r in secs)
        pole += "".join(f'<geom type="cylinder" fromto="0 0 {z - 0.008:.4f} 0 0 {z + 0.008:.4f}" size="{r + 0.0025}" rgba="{blk}" {nc}/>'
                        f'<geom type="cylinder" fromto="0 0 {z - 0.0095:.4f} 0 0 {z - 0.008:.4f}" size="{r + 0.0026}" rgba="{red}" {nc}/>'
                        for (_, z, r) in secs[:2])
        pole += (f'<geom type="cylinder" fromto="0 0 {top:.4f} 0 0 {top + 0.012:.4f}" size="0.012" rgba="{blk}" {nc}/>'
                 f'<geom type="sphere" pos="0 0 {top + 0.004:.4f}" size="0.0055" rgba="{red}" {nc}/>'
                 f'<geom type="sphere" pos="0 0 {ball[2]:.4f}" size="0.011" rgba="{blk}" {nc}/>'
                 f'<geom type="cylinder" fromto="0 0 {top + 0.012:.4f} 0 0 {ball[2]:.4f}" size="0.007" rgba="{blk}" {nc}/>'
                 f'<geom type="cylinder" fromto="0 0 {ball[2]:.4f} 0 0 {mount[2]:.4f}" size="0.006" rgba="{blk}" {nc}/>'
                 f'<geom type="cylinder" fromto="0 0 0 0 0 0.006" size="0.035" rgba="{blk}" {nc}/>')
        return f"""<body name="top_pole" pos="{mount[0]:.4f} {mount[1]:.4f} 0">
      {pole}
    </body>
    <body name="top_camera" pos="{fmt(cam.pos)}" quat="{fmt(cq)}">
      <geom type="box" size="0.045 {box_h} {box_h}" pos="0 0 {box_z}" rgba="0.92 0.92 0.92 1" {nc}/>
      <geom type="box" size="0.042 0.010 0.0006" pos="0 0 0.0003" rgba="0.05 0.05 0.05 1" {nc}/>
      <camera name="top" pos="0 0 0" quat="1 0 0 0" fovy="{cam.fovy:.3f}" resolution="{self.width} {self.height}"/>
    </body>"""

    def _room_xml(self, prio: str, support: str) -> str:
        """Table mats (the only part of the room with contact), the cable grommet, back and side walls, the LED strips
        at the foot of the back wall, wall brackets and the rail under the arm bases."""
        c = self.config
        E, t, tz, mt = c["room"], c["table"], self.table_z, c["table"]["mat_thickness"]
        nc = 'contype="0" conaffinity="0" group="2"'
        wall = fmt(E["wall_rgba"])
        out = ""
        ys = [E["mat_y0"]] + sorted(E["seams_y"], reverse=True) + [E["mat_y1"]]
        cx, hx = (E["mat_x0"] + E["mat_x1"]) / 2, (E["mat_x1"] - E["mat_x0"]) / 2
        for i in range(len(ys) - 1):
            ya = ys[i] - (t["mat_gap"] / 2 if i > 0 else 0.0003)
            yb = ys[i + 1] + (t["mat_gap"] / 2 if i < len(ys) - 2 else 0.0003)
            out += (f'<geom name="mat{i}" type="box" size="{hx:.4f} {(ya - yb) / 2:.4f} {mt / 2}" pos="{cx:.4f} {(ya + yb) / 2:.4f} '
                    f'{tz + mt / 2:.4f}" material="mat_mat" friction="{fmt(t["friction"])}" condim="4"{prio} {support}/>')
        gx, gy, gr = E["grommet"]
        out += f'<geom name="grommet_rim" type="cylinder" size="{gr + 0.006:.4f} 0.0004" pos="{gx:.4f} {gy:.4f} {tz + 0.0004:.4f}" rgba="0.42 0.42 0.44 1" {nc}/>'
        out += f'<geom name="grommet" type="cylinder" size="{gr:.4f} 0.0005" pos="{gx:.4f} {gy:.4f} {tz + 0.0006:.4f}" rgba="0.03 0.03 0.03 1" {nc}/>'
        wx, yl, yr = E["wall_x"], E["wall_y_left"], E["wall_y_right"]
        sy, rx = E["wall_seam_y"], E["wall_right_x"]
        out += f'<geom name="wall_back" type="box" size="0.01 {(yl - sy) / 2 + 0.01:.4f} 0.45" pos="{wx + 0.01:.4f} {(yl + sy) / 2:.4f} {tz + 0.45:.4f}" rgba="{wall}" {nc}/>'
        out += f'<geom name="wall_back_right" type="box" size="0.008 {(sy - yr) / 2 + 0.02:.4f} 0.445" pos="{rx + 0.008:.4f} {(sy + yr) / 2:.4f} {tz + 0.01 + 0.445:.4f}" rgba="{wall}" {nc}/>'
        out += f'<geom name="led_right" type="box" size="0.003 {(sy - yr) / 2:.4f} 0.004" pos="{rx + 0.004:.4f} {(sy + yr) / 2:.4f} {tz + 0.004:.4f}" rgba="0.80 0.90 1.0 1" material="emit" {nc}/>'
        for i, (bx, by) in enumerate(E["brackets"]):
            out += f'<geom name="bracket{i}" type="box" size="0.012 0.018 0.016" pos="{bx:.4f} {by:.4f} {tz + 0.016:.4f}" rgba="0.60 0.61 0.63 1" {nc}/>'
        out += f'<geom name="wall_left" type="box" size="{(wx - E["side_x0"]) / 2:.4f} 0.01 0.45" pos="{(wx + E["side_x0"]) / 2:.4f} {yl + 0.01:.4f} {tz + 0.45:.4f}" rgba="{wall}" {nc}/>'
        out += f'<geom name="wall_right" type="box" size="{(wx - E["side_x0"]) / 2:.4f} 0.01 0.45" pos="{(wx + E["side_x0"]) / 2:.4f} {yr - 0.01:.4f} {tz + 0.45:.4f}" rgba="{wall}" {nc}/>'
        out += f'<geom name="led" type="box" size="0.004 {(yl - sy) / 2:.4f} 0.004" pos="{wx - 0.004:.4f} {(yl + sy) / 2:.4f} {tz + 0.006:.4f}" rgba="1 1 1 1" material="emit" {nc}/>'
        a = c["arms"]
        if t["rail_width"] > 0 and tz < 0:
            rl = np.hypot(a["right_base_dx"], a["separation"]) / 2 + 0.15
            out += (f'<geom name="rail" type="box" size="{rl:.4f} {t["rail_width"] / 2:.4f} {-tz / 2:.4f}" '
                    f'pos="{a["right_base_dx"] / 2:.4f} {-a["separation"] / 2:.4f} {tz / 2:.4f}" '
                    f'euler="0 0 {np.arctan2(-a["separation"], a["right_base_dx"]):.5f}" rgba="0.06 0.06 0.065 1" {nc}/>')
        return out
