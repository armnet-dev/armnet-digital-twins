"""Pinhole model of a calibrated cell camera: world pose from [x, y, z, yaw, pitch, roll, fovy], projection and
ray casting between image pixels and the world."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class PinholeCamera:
    """params = [x, y, z, yaw, pitch, roll, fovy_deg]: position (m), ZYX Euler angles of the camera's (forward, left,
    up) frame (rad) and the vertical field of view, for a width x height image."""

    def __init__(self, params, width: int, height: int):
        self.params = [float(v) for v in params]
        self.width, self.height = int(width), int(height)
        x, y, z, yaw, pitch, roll = self.params[:6]
        Rw = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
        self.pos = np.array([x, y, z])
        self.R = np.column_stack([-Rw[:, 1], Rw[:, 2], -Rw[:, 0]])     # MuJoCo camera axes: x right, y up, z back
        self.fovy = self.params[6]
        self.focal = (self.height / 2) / np.tan(np.deg2rad(self.fovy) / 2)

    def ray(self, u: float, v: float):
        """World ray (origin, unit direction) through pixel (u, v)."""
        d = self.R @ np.array([(u - self.width / 2) / self.focal, -(v - self.height / 2) / self.focal, -1.0])
        return self.pos, d / np.linalg.norm(d)

    def ray_plane(self, u: float, v: float, z: float) -> np.ndarray:
        """Where the ray through pixel (u, v) meets the horizontal plane at height z."""
        o, d = self.ray(u, v)
        return o + (z - o[2]) / d[2] * d

    def project(self, points) -> np.ndarray:
        """(N, 3) world points -> (N, 2) pixels."""
        Xc = (np.atleast_2d(points) - self.pos) @ self.R
        return np.stack([self.width / 2 + self.focal * Xc[:, 0] / -Xc[:, 2],
                         self.height / 2 - self.focal * Xc[:, 1] / -Xc[:, 2]], 1)
