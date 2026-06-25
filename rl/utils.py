"""Training-side utilities shared by every room: seeding, metric logging and
smoothing.  Trajectories (used for episode replay) are plain lists of
room-specific *frame* dicts; the rendering code in :mod:`ui.render` and each
room's UI module knows how to draw them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def set_global_seed(seed: Optional[int]) -> random.Random:
    """Seed numpy + python RNGs and return a dedicated ``random.Random``."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    return random.Random(seed)


def moving_average(values, window: int = 50) -> np.ndarray:
    """Centred-ish moving average aligned to the original length (``mode=same``)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    window = int(max(1, min(window, arr.size)))
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


@dataclass
class TrainResult:
    """Everything the UI needs after training a room.

    The per-episode lists power the *learning* and *exploration* graphs; the
    ``episode_replays`` stores the actual trajectory from every training
    episode. ``snapshots`` keeps labelled greedy checkpoints such as
    ``"Early"`` / ``"Mid"`` / ``"Final"`` for quick comparison.
    """

    episode_rewards: List[float] = field(default_factory=list)
    episode_steps: List[int] = field(default_factory=list)
    episode_success: List[bool] = field(default_factory=list)
    epsilon: List[float] = field(default_factory=list)
    # algorithm-specific series, e.g. {"td_error": [...]} or {"value_delta": [...]}
    extra: Dict[str, List[float]] = field(default_factory=dict)
    # label -> trajectory (list of frame dicts)
    snapshots: Dict[str, List[dict]] = field(default_factory=dict)
    # episode index -> actual training trajectory (list of frame dicts)
    episode_replays: List[List[dict]] = field(default_factory=list)
    # final artefact: a greedy policy table, Q-table or trained agent
    policy: Any = None
    info: Dict[str, Any] = field(default_factory=dict)

    def log_episode(self, reward: float, steps: int, success: bool, epsilon: float = 0.0) -> None:
        self.episode_rewards.append(float(reward))
        self.episode_steps.append(int(steps))
        self.episode_success.append(bool(success))
        self.epsilon.append(float(epsilon))

    @property
    def num_episodes(self) -> int:
        return len(self.episode_rewards)

    def success_rate(self, last: int = 100) -> float:
        if not self.episode_success:
            return 0.0
        tail = self.episode_success[-last:]
        return float(np.mean(tail))
