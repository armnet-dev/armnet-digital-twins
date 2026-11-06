"""Videos of episodes: the twin's cameras over a rollout, and real camera videos next to their sim replay."""
from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg, or put ffmpeg on PATH")
        return exe


class VideoWriter:
    """H.264 MP4 from RGB frames of one size."""

    def __init__(self, path, fps: float, crf: int = 20):
        self.path, self.fps, self.crf, self._p = str(path), fps, crf, None

    def write(self, frame: np.ndarray):
        if self._p is None:
            h, w = frame.shape[:2]
            self._p = subprocess.Popen([ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                        "-s", f"{w}x{h}", "-r", f"{self.fps:g}", "-i", "-", "-c:v", "libx264",
                                        "-pix_fmt", "yuv420p", "-crf", str(self.crf), self.path], stdin=subprocess.PIPE)
        self._p.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())

    def close(self):
        if self._p is not None:
            self._p.stdin.close()
            if self._p.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def label(img: np.ndarray, text: str, sub: str | None = None) -> np.ndarray:
    """A caption bar on top of a tile and an optional line at its bottom."""
    img = img.copy()
    s = 0.45 * img.shape[1] / 640 + 0.1
    bar = int(26 * s / 0.55)
    cv2.rectangle(img, (0, 0), (img.shape[1], bar), (0, 0, 0), -1)
    cv2.putText(img, text, (6, int(bar * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def level_caption(names: dict, levels, k: int) -> str | None:
    """The verifier's level after tick k, once it is above the lowest (whose name is the failed outcome)."""
    if levels is None or k == 0 or levels[k - 1] <= min(names):
        return None
    return names[levels[k - 1]]


def write_rollout(twin, qpos: np.ndarray, path, *, title: str = "", levels=None, tile=(360, 640)):
    """A rollout's recorded states as a 2 x 2 grid: top camera, an overview from behind the arms, both wrist cameras.
    levels: the verifier's level after each tick, shown in the caption."""
    names = twin.level_names
    with VideoWriter(path, twin.fps) as vw:
        for k, q in enumerate(qpos):
            t = f"t = {k / twin.fps:5.1f} s"
            if (s := level_caption(names, levels, k)) is not None:
                t += f"   {s}"
            tiles = [label(twin.render(c, qpos=q, size=tile), f"{title}  {c}" if c == "top" else c, t if c == "top" else None)
                     for c in ("top", "overview", "left_wrist", "right_wrist")]
            vw.write(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0))


def write_side_by_side(twin, qpos: np.ndarray, real: dict, path, *, levels=None, tile=(360, 640)):
    """Rows: the cell cameras; columns: the real episode's frames, then the sim's. real = {camera: (n, H, W, 3)}.
    Sim frame k is what the camera delivers at tick k (from its latency ago), like the real frame logged at tick k."""
    cams = [c for c in twin.cell.cameras if c in real]
    n = min(min(len(v) for v in real.values()), len(qpos))
    names = twin.level_names
    with VideoWriter(path, twin.fps) as vw:
        for k in range(n):
            rows = []
            for c in cams:
                q = qpos[max(0, k - twin.latency_ticks[c])]
                im = cv2.resize(real[c][k], (tile[1], tile[0]), interpolation=cv2.INTER_AREA)
                sub = level_caption(names, levels, k) if c == cams[0] else None
                rows.append(np.concatenate([label(im, f"real  {c}", f"t = {k / twin.fps:5.1f} s"),
                                            label(twin.render(c, qpos=q, size=tile), f"sim  {c}", sub)], 1))
            vw.write(np.concatenate(rows, 0))
