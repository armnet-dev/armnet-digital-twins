"""Scene resets, drawn the way a cell prompts its operator to reset the scene.

A task's reset is an armnet-core manual-reset config: where each object goes (a bounding box or a point with a
heading), sampled inside the cell's manipulation area in top-camera image coordinates. The draw is keyed by
(seed, reset index, cell, scene calibration, task), so a given seed and reset index give the same layout every time,
in the sim and on the cell. Here the layout is mapped onto the table through the calibrated top camera, giving each
object a world pose {"xy": [x, y], "yaw": rad}; the task then places its bodies there before t=0.

reset.json:
  task_slug, cell_id, seed      the sampler's key and default seed
  scene_calibration             manipulation area: vertices (normalized image coordinates) of the top camera frame
  manual_reset_config           armnet_core.manual_reset.ManualResetConfig JSON
  objects                       {element name: {"object": key in the world dict, "height": m above the mat,
                                "yaw_from_arrow_deg": optional}}: the height of the object's part the image point marks
                                (bounding box corners are on the mat), and the object's yaw relative to the prompt's arrow
  staging_offset_xy             optional [x, y] (m, world): where operators set the objects relative to the prompted
                                point, applied to every object"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from armnet_core.manual_reset import ManualResetConfig, ManualResetLayout, sample_manual_reset_layout
from armnet_core.scene_calibration import SceneCalibration

from .camera import PinholeCamera

ELEMENT_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948"]


@dataclass
class Reset:
    """One sampled reset: the image-space layout (what the operator is shown) and each object's world pose."""
    seed: int
    reset_index: int
    layout: ManualResetLayout
    world: dict


@dataclass
class ResetSpec:
    task_slug: str
    cell_id: str
    seed: int
    calibration: SceneCalibration
    config: ManualResetConfig
    objects: dict
    staging_offset_xy: tuple = (0.0, 0.0)

    @classmethod
    def load(cls, path) -> "ResetSpec":
        r = json.loads(Path(path).read_text())
        c = r["scene_calibration"]
        calib = SceneCalibration(id=c["id"], cell_id=r["cell_id"], camera_name="top", vertices=[tuple(v) for v in c["vertices"]],
                                 source_frame_width=int(c["source_frame_width"]), source_frame_height=int(c["source_frame_height"]),
                                 created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        return cls(task_slug=r["task_slug"], cell_id=r["cell_id"], seed=int(r["seed"]), calibration=calib,
                   config=ManualResetConfig.model_validate(r["manual_reset_config"]), objects=r["objects"],
                   staging_offset_xy=tuple(r.get("staging_offset_xy", (0.0, 0.0))))

    def sample(self, camera: PinholeCamera, mat_top: float, *, reset_index: int, seed: int | None = None) -> Reset:
        seed = self.seed if seed is None else int(seed)
        layout = sample_manual_reset_layout(self.config, self.calibration, seed=seed, reset_index=int(reset_index),
                                            cell_id=self.cell_id, task_slug=self.task_slug)
        return Reset(seed, int(reset_index), layout, self.to_world(layout, camera, mat_top))

    def to_world(self, layout: ManualResetLayout, camera: PinholeCamera, mat_top: float) -> dict:
        """World poses on the mat of a layout's elements, through the calibrated top camera, moved by the staging
        offset to where the operators set them."""
        W, H = layout.source_frame_width, layout.source_frame_height
        sx, sy = camera.width / W, camera.height / H
        dx, dy = self.staging_offset_xy
        world = {}
        for el in layout.elements:
            obj = self.objects[el.name]
            z = mat_top + obj["height"]
            c = camera.ray_plane(el.center[0] * W * sx, el.center[1] * H * sy, z)
            f = camera.ray_plane(el.arrow.end[0] * W * sx, el.arrow.end[1] * H * sy, z)
            yaw = math.atan2(f[1] - c[1], f[0] - c[0]) + math.radians(obj.get("yaw_from_arrow_deg", 0.0))
            pose = dict(xy=[float(c[0] + dx), float(c[1] + dy)], yaw=float(math.atan2(math.sin(yaw), math.cos(yaw))))
            if el.type == "bounding_box":
                pose["corners"] = [[float(p[0] + dx), float(p[1] + dy)] for p in
                                   (camera.ray_plane(u * W * sx, v * H * sy, mat_top) for u, v in el.corners)]
            world[obj["object"]] = pose
        return world


def stage(world: dict, *, xy_sd: float, yaw_sd: float, seed: int, reset_index: int) -> dict:
    """`world` as an operator stages it: each object moved by a Gaussian (xy_sd m per axis) and turned (yaw_sd rad)
    off its prompted pose. The draw depends only on (seed, reset_index)."""
    if not (xy_sd or yaw_sd):
        return world
    rng = np.random.default_rng([int(seed), int(reset_index)])
    out = {}
    for k in sorted(world):
        w = dict(world[k])
        c, d, a = np.array(w["xy"]), rng.normal(0.0, xy_sd, 2), rng.normal(0.0, yaw_sd)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        w["xy"], w["yaw"] = (c + d).tolist(), float(w["yaw"] + a)
        if "corners" in w:
            w["corners"] = [(c + d + R @ (np.array(p) - c)).tolist() for p in w["corners"]]
        out[k] = w
    return out


def describe(reset: Reset) -> str:
    lines = [f"reset {reset.reset_index} (seed {reset.seed}):"]
    for el, (key, w) in zip(reset.layout.elements, reset.world.items()):
        lines.append(f"  {el.name:20s} image ({el.center[0]:.3f}, {el.center[1]:.3f}) -> {key} at x={w['xy'][0]:.3f} "
                     f"y={w['xy'][1]:.3f} m, yaw {math.degrees(w['yaw']):+6.1f} deg")
    return "\n".join(lines)


def draw(img: np.ndarray, reset: Reset, calibration: SceneCalibration, title: str | None = None) -> np.ndarray:
    """The reset prompt over an RGB top-camera image: manipulation area, element outlines and front arrows, legend."""
    out = img.copy()
    h, w = out.shape[:2]
    px = lambda p: (int(round(p[0] * w)), int(round(p[1] * h)))
    poly = np.array([px(v) for v in calibration.vertices], np.int32)
    wash = out.copy()
    cv2.fillPoly(wash, [poly], (255, 255, 255))
    out = cv2.addWeighted(wash, 0.12, out, 0.88, 0)
    cv2.polylines(out, [poly], True, (255, 255, 255), 2, cv2.LINE_AA)
    s = max(1, w // 640)
    for i, el in enumerate(reset.layout.elements):
        hexc = ELEMENT_COLORS[i % len(ELEMENT_COLORS)]
        col = tuple(int(hexc[k:k + 2], 16) for k in (1, 3, 5))
        if el.type == "bounding_box":
            cv2.polylines(out, [np.array([px(p) for p in el.corners], np.int32)], True, col, 2 * s, cv2.LINE_AA)
        else:
            cv2.ellipse(out, px(el.center), (max(2, int(el.radius_x * w)), max(2, int(el.radius_y * h))), 0, 0, 360,
                        col, 2 * s, cv2.LINE_AA)
        cv2.line(out, px(el.arrow.start), px(el.arrow.end), col, 2 * s, cv2.LINE_AA)
        cv2.rectangle(out, (10 * s, (14 + 22 * i) * s), (24 * s, (28 + 22 * i) * s), col, -1)
        cv2.putText(out, el.name, (30 * s, (27 + 22 * i) * s), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (255, 255, 255), s, cv2.LINE_AA)
    if title:
        cv2.putText(out, title, (10 * s, h - 12 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (255, 255, 255), s, cv2.LINE_AA)
    return out
