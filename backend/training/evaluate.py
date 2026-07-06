"""Post-training evaluation using the persisted models.

* :func:`evaluate_room` — reload the saved policy/Q-table/network, run N
  greedy episodes, re-record them as eval replays and refresh the saved
  metrics' ``eval`` summary.
* :func:`results_summary` — the cross-room comparison used by the room
  cards and the Results page.
"""
from __future__ import annotations

import pickle

import numpy as np

from backend.training import train as tr
from backend.training.replay_recorder import ReplayRecorder
from backend.utils.config import info_path, metrics_path, model_path, replays_dir
from backend.utils.serialization import load_json, save_json


class NotTrainedError(RuntimeError):
    pass


def _load_act_fn(room_id: int, params: dict):
    """Rebuild a greedy action function from the saved model artefacts."""
    if room_id == 1:
        path = model_path(1)
        if not path.exists():
            raise NotTrainedError("Room 1 has no saved policy — train it first.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        policy = data["policy"]
        return lambda s: int(policy.get(s, 0))

    if room_id in (2, 3):
        path = model_path(room_id)
        if not path.exists():
            raise NotTrainedError(f"Room {room_id} has no saved Q-table — train it first.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        Q = {s: np.asarray(q) for s, q in data["Q"].items()}
        return lambda s: int(np.argmax(Q[s])) if s in Q else 0

    # rooms 4 & 5 — DQN checkpoint
    import torch
    from rl.algos.dqn import MLP

    path = model_path(room_id, "pt")
    if not path.exists():
        raise NotTrainedError(f"Room {room_id} has no saved network — train it first.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MLP(ckpt["obs_dim"], ckpt["n_actions"], tuple(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def act(obs):
        with torch.no_grad():
            q = model(torch.as_tensor(obs, dtype=torch.float32))
            return int(torch.argmax(q).item())
    return act


def _clear_eval_replays(room_id: int) -> None:
    rdir = replays_dir(room_id)
    if not rdir.exists():
        return
    for f in rdir.glob("eval_*.json"):
        f.unlink()
    idx = load_json(rdir / "index.json", default=None)
    if idx:
        idx["episodes"] = [e for e in idx["episodes"] if e.get("kind") != "eval"]
        save_json(rdir / "index.json", idx)


def evaluate_room(room_id: int, n_episodes: int = 10) -> dict:
    info = load_json(info_path(room_id), default=None)
    if info is None:
        raise NotTrainedError(f"Room {room_id} has not been trained yet.")
    params = tr.merge_params(room_id, info.get("params"))
    act = _load_act_fn(room_id, params)
    env = tr.make_env(room_id, params)

    _clear_eval_replays(room_id)
    recorder = ReplayRecorder(room_id, clear=False)
    freekick = room_id == 4 and params.get("mode") == "freekick"
    max_steps = 6 if freekick else params["max_steps"]
    summary = tr.run_greedy_eval(room_id, env, act, recorder, max_steps,
                                 n=n_episodes)

    # refresh the eval block inside the saved metrics + model info
    metrics = load_json(metrics_path(room_id), default=None)
    if metrics is not None:
        metrics["eval"] = summary
        save_json(metrics_path(room_id), metrics)
    info["eval"] = summary
    save_json(info_path(room_id), info)
    return summary


def results_summary() -> dict:
    """One row per room for the comparison table / room cards."""
    rooms = []
    for rid, meta in tr.ROOMS.items():
        metrics = load_json(metrics_path(rid), default=None)
        row = {
            "id": rid,
            "name": meta["name"],
            "algorithm": meta["algorithm"],
            "difficulty": meta["difficulty"],
            "trained": metrics is not None,
            "episodes": None, "best_reward": None, "avg_reward_last50": None,
            "success_rate": None, "eval_success_rate": None,
            "train_time": None, "timestamp": None,
        }
        if metrics:
            s = metrics.get("summary") or {}
            e = metrics.get("eval") or {}
            row.update(
                episodes=s.get("episodes"),
                best_reward=s.get("best_reward"),
                avg_reward_last50=s.get("avg_reward_last50"),
                success_rate=s.get("success_rate_last50"),
                eval_success_rate=e.get("success_rate"),
                eval_avg_reward=e.get("avg_reward"),
                train_time=s.get("train_time"),
                timestamp=s.get("timestamp"),
                stopped=s.get("stopped", False),
            )
        rooms.append(row)
    return {"rooms": rooms}
