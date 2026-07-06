"""Chart-ready metric series built from :class:`rl.utils.TrainResult`.

The algorithms in ``rl/algos`` already log everything per episode into a
``TrainResult`` (rewards, steps, success flags, epsilon, plus algorithm
extras such as TD error or DQN loss).  This module converts that object
into the JSON the frontend charts expect, both for live polling during
training and for the persisted ``results/metrics/room{N}.json``.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

from rl.utils import TrainResult

from backend.utils.config import metrics_path
from backend.utils.serialization import load_json, save_json

# rl extra-series labels -> stable series keys used by the frontend
_EXTRA_KEYS = {
    "mean |TD error|": "td_error",
    "training loss": "loss",
    "Camera detections / episode": "camera_hits",
    "Times caught by guards / episode": "caught",
    "Trap hits / episode": "trap_hits",
    "Crashes / episode": "crashes",
    "Shortcut tiles used / episode": "shortcuts",
}

_EXTRA_LABELS = {v: k for k, v in _EXTRA_KEYS.items()}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def moving_average(values: List[float], window: int = 25) -> List[float]:
    out, acc, q = [], 0.0, []
    for v in values:
        q.append(v)
        acc += v
        if len(q) > window:
            acc -= q.pop(0)
        out.append(acc / len(q))
    return out


def windowed_rate(flags: List[bool], window: int = 50) -> List[float]:
    return moving_average([1.0 if f else 0.0 for f in flags], window)


def build_series(result: TrainResult) -> Dict[str, list]:
    """All chart series keyed by name (episode-indexed unless noted)."""
    n = result.num_episodes
    out: Dict[str, list] = {"episode": list(range(1, n + 1))}
    if n:
        out["reward"] = [round(r, 2) for r in result.episode_rewards]
        out["reward_avg"] = [round(r, 2) for r in moving_average(out["reward"], 25)]
        out["steps"] = result.episode_steps
        out["success_rate"] = [round(r, 3) for r in
                               windowed_rate(result.episode_success, 50)]
        out["failure_rate"] = [round(1.0 - r, 3) for r in out["success_rate"]]
        if any(e > 0 for e in result.epsilon):
            out["epsilon"] = [round(e, 4) for e in result.epsilon]
    for label, series in result.extra.items():
        if not series:
            continue
        key = _EXTRA_KEYS.get(label, _slug(label))
        out[key] = [None if v is None else round(float(v), 4) for v in series]
    return out


def build_summary(result: TrainResult, algorithm: str,
                  train_time: float = 0.0) -> dict:
    rewards = result.episode_rewards
    n = len(rewards)
    best_ep = max(range(n), key=lambda i: rewards[i]) if n else None
    last = rewards[-50:]
    return {
        "algorithm": algorithm,
        "episodes": n,
        "best_reward": round(max(rewards), 2) if rewards else None,
        "best_episode": (best_ep + 1) if best_ep is not None else None,
        "best_steps": result.episode_steps[best_ep] if best_ep is not None else None,
        "avg_reward_last50": round(sum(last) / len(last), 2) if last else None,
        "success_rate_last50": round(result.success_rate(50), 3) if n else None,
        "train_time": round(train_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def downsample(series: dict, max_points: int = 600) -> dict:
    """Uniformly subsample long series so live polling stays light."""
    out = {}
    for k, v in series.items():
        if isinstance(v, list) and len(v) > max_points:
            step = len(v) / max_points
            idx = [int(i * step) for i in range(max_points)]
            if idx[-1] != len(v) - 1:
                idx.append(len(v) - 1)
            out[k] = [v[i] for i in idx]
        else:
            out[k] = v
    return out


def save_metrics(room_id: int, algorithm: str, params: dict,
                 series: dict, summary: dict,
                 eval_summary: Optional[dict] = None) -> None:
    save_json(metrics_path(room_id), {
        "room_id": room_id,
        "algorithm": algorithm,
        "params": params,
        "series": series,
        "summary": summary,
        "eval": eval_summary,
    })


def load_metrics(room_id: int) -> Optional[dict]:
    return load_json(metrics_path(room_id))
