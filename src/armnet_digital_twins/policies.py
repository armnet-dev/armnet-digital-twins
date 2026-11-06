"""Policies for rollout.run_episode(s): open-loop replay of logged actions, and LeRobot policies (pi0.5, ACT, ...)."""
from __future__ import annotations

import os

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "wrist_yaw")


class PerWorld:
    """Single-world policies as one batched policy: policies[i] drives world i."""

    def __init__(self, policies):
        self.policies = list(policies)

    def reset_worlds(self, seeds):
        for p, s in zip(self.policies, seeds):
            p.reset(s)

    def act(self, worlds, obs, left, right):
        out = [self.policies[w](o, l, r) for w, o, l, r in zip(worlds, obs, left, right)]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])


class ReplayPolicy:
    """Logged actions (the dataset's normalized frame, left 7 then right 7), one per tick, ignoring the cameras."""

    def __init__(self, actions, cell):
        self.actions, self.cell, self.k = np.asarray(actions, float), cell, 0

    def reset(self, seed=None):
        self.k = 0

    def __call__(self, obs, left, right):
        a = self.actions[min(self.k, len(self.actions) - 1)]
        self.k += 1
        return self.cell.from_normalized(a[:7]), self.cell.from_normalized(a[7:])


class LeRobotPolicy:
    """A chunked LeRobot policy trained on the cell's LeRobot data, run the way the cell runs it: each query sees the
    current joint state (normalized frame) and the newest camera frames and returns a chunk of actions, of which
    n_action_steps are executed before the next query. Flow-matching policies (pi0, pi0.5) draw their denoising noise
    from a generator seeded by the episode seed and the tick, so a rerun gives the same rollout.

    Batched over worlds: the worlds that need a new chunk at a tick are queried together, up to max_batch at a time
    (default all), and each world's noise is its own, so its chunk doesn't depend on the batch.

    path: Hub repo or local checkpoint. dataset: the LeRobot dataset whose features (cameras, joint names) the policy
    was trained on. Needs `pip install armnet-digital-twins[pi05]`."""

    def __init__(self, path: str, cell, *, dataset: str, prompt: str, device: str | None = None,
                 n_action_steps: int | None = None, max_batch: int | None = None, robot_type: str = "lerobot/bimanual_yam"):
        import dataclasses

        try:
            import torch
            from lerobot import policies as _registered  # noqa: F401  (registers the policy types)
        except ImportError as e:
            raise ImportError("LeRobotPolicy needs LeRobot: pip install armnet-digital-twins[pi05]") from e
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.feature_utils import build_dataset_frame
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.utils.constants import OBS_STR

        self.torch, self.cell, self.prompt, self.robot_type = torch, cell, prompt, robot_type
        self._build, self._prep, self._obs = build_dataset_frame, prepare_observation_for_inference, OBS_STR
        local = path if os.path.isdir(path) else snapshot_download(path)
        cfg = PreTrainedConfig.from_pretrained(local)
        cfg.pretrained_path = local
        if n_action_steps is not None:
            cfg = dataclasses.replace(cfg, n_action_steps=int(n_action_steps))
        self.meta = LeRobotDatasetMetadata(dataset)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg.device = str(self.device)
        self.policy = make_policy(cfg, ds_meta=self.meta)
        self.policy.eval()
        self.policy.to(self.device)
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=local,
                                                       preprocessor_overrides={"device_processor": {"device": str(self.device)}})
        self.cfg, self.horizon = self.policy.config, int(self.policy.config.n_action_steps)
        self.cameras = [k.split(".")[-1] for k in self.meta.features if k.startswith("observation.images.")]
        names = list(self.meta.features["action"]["names"])
        self.out_ix = {s: [names.index(f"{s}_{j}.pos") for j in JOINTS] + [names.index(f"{s}_gripper.pos")] for s in ("left", "right")}
        # keep the normalized state inside the envelope the policy was trained on (+-5 sigma; +-1 for bounded modes)
        mode = str(getattr(cfg, "normalization_mapping", {}).get("STATE", "MEAN_STD")).split(".")[-1].upper()
        self.state_clamp = 5.0 if mode == "MEAN_STD" else (None if mode == "IDENTITY" else 1.0)
        self.max_batch = max_batch
        self.reset_worlds([None])

    def describe(self) -> str:
        return (f"{self.cfg.type} on {self.device}, chunks of {getattr(self.cfg, 'chunk_size', '?')} executing "
                f"{self.horizon}, cameras {self.cameras}")

    def reset(self, seed=None):
        self.reset_worlds([seed])

    def __call__(self, obs, left, right):
        cl, cr = self.act([0], [obs], np.asarray(left)[None], np.asarray(right)[None])
        return cl[0], cr[0]

    def reset_worlds(self, seeds):
        self.policy.reset()
        self.seeds, self.queues, self.ticks = list(seeds), [[] for _ in seeds], [0] * len(seeds)

    def act(self, worlds, obs, left, right):
        need = [j for j, w in enumerate(worlds) if not self.queues[w]]
        size = self.max_batch or max(1, len(need))
        for s in range(0, len(need), size):
            part = need[s:s + size]
            chunks = self._predict([obs[j] for j in part], [left[j] for j in part], [right[j] for j in part],
                                   [worlds[j] for j in part])
            for j, a in zip(part, chunks):
                self.queues[worlds[j]] = [(self.cell.from_normalized(a[t, self.out_ix["left"]]),
                                           self.cell.from_normalized(a[t, self.out_ix["right"]])) for t in range(len(a))]
        out = []
        for w in worlds:
            out.append(self.queues[w].pop(0))
            self.ticks[w] += 1
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])

    def _frame(self, obs, left, right):
        robot_obs = {}
        for side, pose in (("left", left), ("right", right)):
            n = self.cell.to_normalized(pose)
            for i, j in enumerate(JOINTS):
                robot_obs[f"{side}_{j}.pos"] = float(n[i])
            robot_obs[f"{side}_gripper.pos"] = float(n[6])
        for c in self.cameras:
            robot_obs[c] = obs[c]
        frame = self._build(self.meta.features, robot_obs, prefix=self._obs)
        frame = self.pre(self._prep(frame, self.device, self.prompt, self.robot_type))
        if self.state_clamp is not None and "observation.state" in frame:
            frame["observation.state"] = frame["observation.state"].clamp(-self.state_clamp, self.state_clamp)
        return frame

    def _noise(self, worlds):
        torch = self.torch
        seeds = [self.seeds[w] for w in worlds]
        if None in seeds or not hasattr(getattr(self.policy, "model", None), "sample_noise"):
            return {}
        shape = (1, self.cfg.chunk_size, self.cfg.max_action_dim)
        return {"noise": torch.cat([torch.randn(shape, generator=torch.Generator(device=self.device).manual_seed(int(s) * 100_003 + self.ticks[w]),
                                                device=self.device, dtype=torch.float32) for s, w in zip(seeds, worlds)])}

    def _predict(self, obs, left, right, worlds) -> np.ndarray:
        """(len(worlds), n_action_steps, action dim) normalized actions: one forward pass for the batch."""
        torch = self.torch
        frames = [self._frame(o, l, r) for o, l, r in zip(obs, left, right)]
        batch = {}
        for key, v in frames[0].items():
            vals = [f[key] for f in frames]
            batch[key] = torch.cat(vals, 0) if torch.is_tensor(v) else sum(vals, []) if isinstance(v, list) else v
        with torch.inference_mode():
            a = self.policy.predict_action_chunk(batch, **self._noise(worlds))[:, :self.horizon]
            B, T, D = a.shape
            return self.post(a.reshape(B * T, D)).reshape(B, T, D).float().cpu().numpy()
