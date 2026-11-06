"""Episodes of LeRobot (v3) datasets on the Hugging Face Hub: logged joint states, actions and camera videos."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

from .video import ffmpeg_exe


class LeRobotEpisode:
    """states / actions: (length, D) as logged (for the YAM cell: the normalized frame, left 7 then right 7)."""

    def __init__(self, repo_id: str, episode_index: int, *, revision: str | None = None):
        self.repo_id, self.episode_index, self.revision = repo_id, int(episode_index), revision
        meta = Path(snapshot_download(repo_id, repo_type="dataset", revision=revision, allow_patterns=["meta/*", "meta/**"]))
        self.info = json.loads((meta / "meta" / "info.json").read_text())
        self.fps = float(self.info["fps"])
        eps = pd.concat([pd.read_parquet(f) for f in sorted((meta / "meta" / "episodes").rglob("*.parquet"))])
        rows = eps[eps.episode_index == self.episode_index]
        if rows.empty:
            raise KeyError(f"{repo_id} has no episode {episode_index} ({self.info['total_episodes']} episodes)")
        self.row = rows.iloc[0]
        path = self.info["data_path"].format(chunk_index=int(self.row["data/chunk_index"]), file_index=int(self.row["data/file_index"]))
        df = pd.read_parquet(self._download(path), columns=["episode_index", "frame_index", "observation.state", "action"])
        g = df[df.episode_index == self.episode_index].sort_values("frame_index")
        self.states = np.stack(g["observation.state"].values).astype(np.float64)
        self.actions = np.stack(g["action"].values).astype(np.float64)
        self.length = len(g)

    def _download(self, filename: str) -> str:
        return hf_hub_download(self.repo_id, filename, repo_type="dataset", revision=self.revision)

    @property
    def cameras(self) -> list[str]:
        return [k.split(".")[-1] for k, v in self.info["features"].items() if v.get("dtype") == "video"]

    def frames(self, camera: str, size: tuple | None = None, n: int | None = None) -> np.ndarray:
        """The episode's frames of one camera, (n, H, W, 3) uint8 RGB. size: (height, width) to scale to."""
        key = f"observation.images.{camera}"
        path = self.info["video_path"].format(video_key=key, chunk_index=int(self.row[f"videos/{key}/chunk_index"]),
                                              file_index=int(self.row[f"videos/{key}/file_index"]))
        n = self.length if n is None else min(int(n), self.length)
        h, w = size or self.info["features"][key]["shape"][:2]
        cmd = [ffmpeg_exe(), "-v", "error", "-ss", f"{float(self.row[f'videos/{key}/from_timestamp']):.3f}", "-i", self._download(path),
               "-frames:v", str(n), "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        f = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(-1, h, w, 3)
        if len(f) < n:
            f = np.concatenate([f, np.repeat(f[-1:], n - len(f), 0)])
        return f[:n]
