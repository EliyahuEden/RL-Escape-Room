"""Episode replay capture for the web frontend.

Two pieces:

* :class:`ReplayRecorder` — buffers frames for one episode at a time and
  persists them as ``results/replays/room{N}/<id>.json`` plus an
  ``index.json`` catalogue.  Only a spread of *milestone* training
  episodes is kept (first, spread, last) plus every greedy evaluation
  episode, so disk usage stays bounded while still showing how the
  behaviour evolved.

* :class:`RecordingEnv` — a transparent wrapper around any room env.
  The training code in ``rl/algos`` only ever calls ``reset()`` /
  ``step()`` / attribute access, so wrapping the env lets us record
  frame-by-frame replays **without modifying the algorithms at all**.

Each replay file: ``{"meta": {...}, "frames": [{...}, ...]}``.
Frame keys are compact: ``t`` step, ``a`` action, ``r`` step reward,
``cum`` cumulative reward, ``done`` flag, ``ev`` event list, plus
room-specific fields produced by the frame builders in
:mod:`backend.training.frames`.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

from backend.utils.config import replays_dir
from backend.utils.serialization import load_json, save_json


class ReplayRecorder:
    def __init__(self, room_id: int, clear: bool = False):
        self.room_id = room_id
        self.dir = replays_dir(room_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        if clear:
            for f in self.dir.glob("*.json"):
                f.unlink()
        idx = load_json(self.dir / "index.json", default={"episodes": []})
        self.index: List[dict] = idx["episodes"]
        self.milestones: set[int] = set()
        self._frames: Optional[list] = None
        self._id: Optional[str] = None

    # -- milestone planning --------------------------------------------------
    def plan_milestones(self, total_episodes: int, n_marks: int = 8) -> None:
        marks = {1, 2, total_episodes}
        for i in range(1, n_marks + 1):
            marks.add(max(1, round(i * total_episodes / n_marks)))
        self.milestones = marks

    def wants(self, episode_idx: int) -> bool:
        return episode_idx in self.milestones

    # -- one episode ----------------------------------------------------------
    def start(self, replay_id: str) -> None:
        self._id = replay_id
        self._frames = []

    @property
    def recording(self) -> bool:
        return self._frames is not None

    def add_frame(self, frame) -> None:
        """Accepts a single frame dict or a list of frames (e.g. an expanded
        football kick flight)."""
        if self._frames is None:
            return
        if isinstance(frame, list):
            self._frames.extend(frame)
        else:
            self._frames.append(frame)

    def discard(self) -> None:
        self._frames, self._id = None, None

    def finish(self, meta: dict) -> None:
        if self._frames is None or self._id is None:
            return
        meta = {**meta, "id": self._id, "room": self.room_id,
                "frames": len(self._frames),
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_json(self.dir / f"{self._id}.json",
                  {"meta": meta, "frames": self._frames})
        self.index = [e for e in self.index if e["id"] != self._id]
        self.index.append({k: meta.get(k) for k in
                           ("id", "kind", "label", "episode", "reward", "steps",
                            "success", "fail_reason", "frames", "recorded_at")})
        self._write_index()
        self.discard()

    def _write_index(self) -> None:
        order = {"train": 0, "eval": 1}
        self.index.sort(key=lambda e: (order.get(e.get("kind"), 2),
                                       e.get("episode") or 0))
        save_json(self.dir / "index.json", {"episodes": self.index})

    # -- catalogue access -----------------------------------------------------
    @staticmethod
    def list_episodes(room_id: int) -> List[dict]:
        idx = load_json(replays_dir(room_id) / "index.json",
                        default={"episodes": []})
        return idx["episodes"]

    @staticmethod
    def load_replay(room_id: int, replay_id: str) -> Optional[dict]:
        # ids are generated internally, but sanitise: this feeds a file path
        safe = "".join(ch for ch in replay_id if ch.isalnum() or ch in "_-")
        return load_json(replays_dir(room_id) / f"{safe}.json")


class RecordingEnv:
    """Wraps a room env and records milestone episodes during training.

    ``frame_fn(env, action, reward, cum, done, info)`` builds one frame
    dict; ``meta_fn(env)`` may add per-episode meta (e.g. the layout for
    randomised traffic).  All other attribute access is delegated to the
    wrapped env, so the training loops in ``rl/algos`` see an identical
    interface.
    """

    def __init__(self, env, recorder: ReplayRecorder,
                 frame_fn: Callable, meta_fn: Optional[Callable] = None,
                 label_fn: Optional[Callable[[int], str]] = None):
        object.__setattr__(self, "_wrapped", env)
        self.recorder = recorder
        self.frame_fn = frame_fn
        self.meta_fn = meta_fn
        self.label_fn = label_fn
        self.episode = 0
        self.cum = 0.0
        self.ep_steps = 0
        self.ep_success = False
        self.ep_fail: Optional[str] = None

    # delegate everything the algorithms touch (n_actions, obs_dim, ...)
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    @property
    def env(self):
        return self._wrapped

    # -- gym-style interface ---------------------------------------------------
    def reset(self):
        self._finish_pending()
        self.episode += 1
        out = self._wrapped.reset()
        self.cum = 0.0
        self.ep_steps = 0
        self.ep_success = False
        self.ep_fail = None
        if self.recorder.wants(self.episode):
            self.recorder.start(f"train_{self.episode:05d}")
            self.recorder.add_frame(
                self.frame_fn(self._wrapped, None, 0.0, 0.0, False, {}))
        return out

    def step(self, action):
        state, reward, done, info = self._wrapped.step(action)
        self.cum += reward
        self.ep_steps += 1
        if info.get("success") or info.get("escaped"):
            self.ep_success = True
        for key in ("caught", "crash", "collision"):
            if info.get(key):
                self.ep_fail = key
        if done and not self.ep_success and not self.ep_fail and info.get("event"):
            self.ep_fail = str(info["event"])[:28]
        if self.recorder.recording:
            self.recorder.add_frame(
                self.frame_fn(self._wrapped, int(action), float(reward),
                              round(self.cum, 2), bool(done), info))
            if done:
                self._finish_pending()
        return state, reward, done, info

    # -- internals ---------------------------------------------------------------
    def _finish_pending(self) -> None:
        """Persist a buffered episode (also handles max-steps cutoffs where
        the training loop abandons the episode without ``done=True``)."""
        if not self.recorder.recording:
            return
        label = (self.label_fn(self.episode) if self.label_fn
                 else f"Training episode {self.episode}")
        meta = {
            "kind": "train",
            "episode": self.episode,
            "label": label,
            "reward": round(self.cum, 1),
            "steps": self.ep_steps,
            "success": self.ep_success,
            "fail_reason": None if self.ep_success else (self.ep_fail or "timeout"),
        }
        if self.meta_fn is not None:
            meta.update(self.meta_fn(self._wrapped))
        self.recorder.finish(meta)

    def close(self) -> None:
        self._finish_pending()
