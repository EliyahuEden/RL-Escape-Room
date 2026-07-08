"""Training orchestration for the five rooms.

This module owns the room **registry** (metadata + hyperparameter schemas
that the frontend renders as controls) and one trainer per algorithm
family.  The trainers call the *existing* algorithms in ``rl/algos``
unchanged — replays are captured by wrapping the environment
(:class:`~backend.training.replay_recorder.RecordingEnv`), and metrics
come from the :class:`rl.utils.TrainResult` the algorithms already build.

CLI (every algorithm can be tested independently)::

    python -m backend.training.train --room 3 --set episodes=300
"""
from __future__ import annotations

import pickle
import random
import time
from typing import Callable, Optional

import numpy as np

from rl.algos import dp as dp_mod
from rl.algos import td_core
from rl.algos import dqn as dqn_mod
from rl.envs.museum import DEFAULT_LAYOUT as MUSEUM_DEFAULT, MuseumEnv, generate_museum_layout
from rl.envs.obstacles import ObstacleEnv
from rl.envs.pacman import DEFAULT_LAYOUT as PACMAN_DEFAULT, PacmanEnv, generate_pacman_layout
from rl.envs.racing import DEFAULT_LAYOUT as RACING_DEFAULT, RacingEnv, generate_racing_layout
from rl.envs.football import FootballEnv, FreeKickEnv
from rl.utils import TrainResult

from backend.training import frames as fr
from backend.training.replay_recorder import RecordingEnv, ReplayRecorder
from backend.utils import metrics as mt
from backend.utils.config import (config_path, ensure_dirs, info_path,
                                  model_path, policy_path)
from backend.utils.serialization import load_json, save_json


class StopRequested(Exception):
    """Raised from a progress callback to abort training cleanly."""


# ===========================================================================
#  Hyperparameter schemas (rendered as controls by the frontend)
# ===========================================================================
def _p(key, label, type_, default, minv=None, maxv=None, step=None,
       options=None, help_=""):
    return {"key": key, "label": label, "type": type_, "default": default,
            "min": minv, "max": maxv, "step": step, "options": options,
            "help": help_}


def _td_params(episodes=800, max_steps=200, eps_end=0.05, eps_end_help="",
               alpha=0.1, gamma=0.95):
    return [
        _p("alpha", "Learning rate α", "float", alpha, 0.01, 1.0, 0.01,
           help_="How strongly each TD error updates Q(s,a)."),
        _p("gamma", "Discount γ", "float", gamma, 0.5, 0.999, 0.005,
           help_="How much future reward matters."),
        _p("eps_start", "Initial ε", "float", 1.0, 0.0, 1.0, 0.05,
           help_="Starting exploration rate."),
        _p("eps_end", "Minimum ε", "float", eps_end, 0.0, 0.5, 0.01,
           help_=eps_end_help),
        _p("eps_decay", "ε decay / episode", "float", 0.995, 0.9, 1.0, 0.001),
        _p("episodes", "Episodes", "int", episodes, 100, 5000, 50),
        _p("max_steps", "Max steps / episode", "int", max_steps, 50, 500, 25),
    ]


def _dqn_params(episodes=400, max_steps=300, max_steps_rng=(60, 800),
                lr=0.001, gamma=0.99, exploration_fraction=0.5, batch_size=64):
    return [
        _p("lr", "Learning rate", "float", lr, 0.0001, 0.01, 0.0001),
        _p("gamma", "Discount γ", "float", gamma, 0.8, 0.999, 0.005),
        _p("eps_start", "Initial ε", "float", 1.0, 0.0, 1.0, 0.05),
        _p("eps_end", "Minimum ε", "float", 0.05, 0.0, 0.3, 0.01),
        _p("exploration_fraction", "Exploration fraction", "float", exploration_fraction,
           0.1, 1.0, 0.05,
           help_="Fraction of training over which ε decays to its minimum."),
        _p("batch_size", "Batch size", "int", batch_size, 16, 256, 16),
        _p("buffer_size", "Replay buffer size", "int", 50000, 5000, 200000, 5000),
        _p("target_update", "Target net update (steps)", "int", 500, 50, 5000, 50),
        _p("episodes", "Episodes", "int", episodes, 100, 2500, 50),
        _p("max_steps", "Max steps / episode", "int", max_steps,
           max_steps_rng[0], max_steps_rng[1], 10),
    ]


_SEED = _p("seed", "Random seed", "int", 0, 0, 9999, 1)
_MAP = [
    _p("map_mode", "Map source", "select", "classic",
       options=["classic", "generated"],
       help_="classic = the curated hand-made map; generated = random items "
             "on the same floor plan."),
    _p("map_seed", "Map seed", "int", 10, 0, 9999, 1),
]


ROOMS: dict[int, dict] = {
    1: {
        "name": "Pacman Maze",
        "subtitle": "ROOM 01 · KNOWN MODEL",
        "algorithm": "Dynamic Programming",
        "algo_id": "dp",
        "difficulty": 1,
        "type": "pacman",
        "accent": "#ffd23f",
        "icon": "pacman",
        "description": (
            "A 10×10 arcade maze with a fully-known model. Collect every coin "
            "— only then does the exit door unlock. Icy tiles deflect moves "
            "sideways, and an optional guard patrols or chases. Because the "
            "transition model is known, Value/Policy Iteration compute the "
            "optimal policy offline — no exploration at all."),
        "state": "(cell, coin-bitmask [, guard]) — position + coins still on the board",
        "actions": "4 — Up / Down / Left / Right",
        "rewards": "step −1 · coin +10 · exit +100 · locked door −10 · slip −5 · wall −5 · guard −50",
        "params": [
            _p("method", "DP method", "select", "value_iteration",
               options=["value_iteration", "policy_iteration"]),
            _p("gamma", "Discount γ", "float", 0.95, 0.5, 0.999, 0.005),
            _p("theta", "Convergence threshold θ", "float", 0.0001, 0.000001, 0.1, 0.0001),
            _p("max_iterations", "Max DP iterations", "int", 250, 25, 1000, 25),
            _p("eval_episodes", "Evaluation episodes", "int", 20, 5, 100, 5),
            _p("max_steps", "Max steps / episode", "int", 200, 50, 500, 25),
            *_MAP,
            _p("n_coins", "Coins (generated map)", "int", 4, 2, 8, 1),
            _p("n_slippery", "Ice tiles (generated map)", "int", 5, 0, 15, 1),
            _p("slip_prob", "Slip probability", "float", 0.2, 0.0, 0.5, 0.05),
            _p("guard_enabled", "Enable guard", "bool", True),
            _p("guard_mode", "Guard behaviour", "select", "patrol",
               options=["patrol", "chase"],
               help_="patrol = fixed route (small state space). chase = hunts "
                     "the agent (much bigger state space, slower to solve)."),
            _SEED,
        ],
    },
    2: {
        "name": "Museum Heist",
        "subtitle": "ROOM 02 · ON-POLICY",
        "algorithm": "SARSA",
        "algo_id": "sarsa",
        "difficulty": 2,
        "type": "museum",
        "accent": "#a78bfa",
        "icon": "diamond",
        "description": (
            "Sneak through the museum, steal the diamond from the vault, and "
            "escape — past camera zones, laser traps, patrolling guards and "
            "slippery marble. One camera sighting raises the ALARM: every "
            "guard abandons its patrol and hunts you for the rest of the "
            "heist. The model is unknown: SARSA learns cautiously from its "
            "own ε-greedy experience."),
        "state": "(cell, has_diamond, guard phase — or guard positions once the alarm is up)",
        "actions": "4 — Up / Down / Left / Right",
        "rewards": "step −1 · diamond +30 · escape +100 · camera −50 + manhunt · trap −15 · guard −50 · slip −5",
        "params": _td_params(episodes=2000) + [
            _p("slip_prob", "Slip probability", "float", 0.1, 0.0, 0.5, 0.05),
            _p("alarm_enabled", "Camera alarm system", "bool", True,
               help_="One camera sighting makes the guards chase you for the "
                     "rest of the episode."),
            *_MAP, _SEED,
        ],
    },
    3: {
        "name": "Street Racing",
        "subtitle": "ROOM 03 · OFF-POLICY",
        "algorithm": "Q-Learning",
        "algo_id": "qlearning",
        "difficulty": 3,
        "type": "racing",
        "accent": "#ff4d5a",
        "icon": "car",
        "description": (
            "A Grand Prix against a rival: your Q-Learning car versus a "
            "SARSA car trained on the same circuit with the same settings. "
            "Both must cross checkpoint 1, then checkpoint 2, then the "
            "finish — and every gate exists twice: once on the express "
            "lane that hugs the crash barriers, once on the safe ring "
            "road. SARSA (on-policy) prices the barrier risk in and takes "
            "the long way; Q-Learning (off-policy) learns the barrier-"
            "hugging lap and wins. Cliff walking, staged as a race."),
        "state": "(cell, next-checkpoint) — each car tracks its own lap progress",
        "actions": "4 — Up / Down / Left / Right",
        "rewards": "step −1 · checkpoint +40 · finish +200 · crash −200 · gravel −5 · locked finish −10 · fastest car wins the race",
        "params": _td_params(episodes=1200, eps_end=0.15, alpha=0.2,
                             eps_end_help="Kept high on purpose: lingering "
                             "exploration is what makes the barrier lane "
                             "dangerous for the on-policy SARSA rival.") + [
            _p("slip_prob", "Oil slip probability", "float", 0.2, 0.0, 0.5, 0.05),
            *_MAP, _SEED,
        ],
    },
    4: {
        "name": "Football Striker",
        "subtitle": "ROOM 04 · CONTINUOUS STATE",
        "algorithm": "DQN",
        "algo_id": "dqn",
        "difficulty": 4,
        "type": "football",
        "accent": "#34d399",
        "icon": "ball",
        "description": (
            "A continuous 10×10 m pitch. Dribble past chasing defenders, "
            "reach the shooting zone, and kick — choosing power and curve — "
            "past a keeper patrolling the goal mouth. The ball physically "
            "flies on its own. State is continuous (x, y, Vx, Vy + opponents), "
            "so a neural network approximates Q(s, a). Free-kick mode trains "
            "3D set pieces over a defensive wall."),
        "state": "x, y, Vx, Vy, keeper position/direction, defender vectors — continuous",
        "actions": "15 — 9 moves (incl. diagonals) + 6 kicks (power × curve)",
        "rewards": "goal +300 · save −10 · miss −30 · tackle −50 · progress shaping · step −0.3",
        "params": [
            _p("mode", "Game mode", "select", "match",
               options=["match", "freekick"],
               help_="match = dribble + shoot. freekick = 3D set piece over a wall."),
            _p("kick_spot", "Free-kick spot", "select", "random",
               options=["random", "center", "left", "right", "near", "far"],
               help_="Where the free kick is taken from (free-kick mode only). "
                     "'random' (default) samples a fresh continuous spot every "
                     "episode — any distance and angle — so one policy learns to "
                     "score from anywhere. The wall auto-scales with distance."),
        ] + _dqn_params(episodes=800, max_steps=160, max_steps_rng=(60, 300),
                        lr=0.0005, gamma=0.98, exploration_fraction=0.65,
                        batch_size=128) + [
            _p("n_defenders", "Defenders (match)", "int", 3, 1, 5, 1,
               help_="Chasing defenders in match mode. In free-kick mode the "
                     "wall size is set automatically by how far out the kick is."),
            _p("keeper_speed", "Keeper patrol speed (m/s)", "float", 1.5, 0.5, 4.0, 0.25),
            _SEED,
        ],
    },
    5: {
        "name": "Cross the Road",
        "subtitle": "ROOM 05 · SENSORS + TRAFFIC",
        "algorithm": "DQN + sensors",
        "algo_id": "dqn",
        "difficulty": 5,
        "type": "crossroad",
        "accent": "#38bdf8",
        "icon": "chicken",
        "description": (
            "Guide the chicken across 10 m of moving traffic. Cars flow in "
            "lanes with alternating directions and wrap around, so nothing "
            "can be memorised — the agent senses the nearest cars inside its "
            "sensor range (relative position, velocity, closeness) and must "
            "learn when to dash, dodge or wait. Trained policies can be "
            "tested on brand-new traffic patterns."),
        "state": "x, y, Vx, Vy, goal direction + 6 sensor slots × 4 features",
        "actions": "9 — 8 directions (incl. diagonals) + Stay",
        "rewards": "cross +250 · collision −140 · off-road −40 · near-miss penalty · progress shaping",
        "params": _dqn_params(episodes=600, max_steps=350, max_steps_rng=(120, 800)) + [
            _p("n_cars", "Number of cars", "int", 14, 4, 24, 1),
            _p("car_speed", "Traffic speed (m/s)", "float", 1.35, 0.7, 2.4, 0.05),
            _p("player_speed", "Chicken speed (m/s)", "float", 1.6, 1.0, 2.6, 0.1),
            _p("sensor_range", "Sensor range (m)", "float", 3.5, 1.5, 6.0, 0.5),
            _p("randomize", "New traffic every episode", "bool", True,
               help_="Off = the same traffic pattern repeats (easier, memorisable)."),
            _SEED,
        ],
    },
}


def default_params(room_id: int) -> dict:
    return {p["key"]: p["default"] for p in ROOMS[room_id]["params"]}


def merge_params(room_id: int, user: Optional[dict]) -> dict:
    """Saved config + defaults + user overrides, clamped to schema ranges."""
    out = default_params(room_id)
    saved = load_json(config_path(room_id), default=None)
    schema = {p["key"]: p for p in ROOMS[room_id]["params"]}
    for source in (saved or {}), (user or {}):
        for k, v in source.items():
            if k not in schema:
                continue
            p = schema[k]
            try:
                if p["type"] == "int":
                    v = int(v)
                elif p["type"] == "float":
                    v = float(v)
                elif p["type"] == "bool":
                    v = bool(v)
                elif p["type"] == "select" and p["options"]:
                    if v not in p["options"]:
                        continue
            except (TypeError, ValueError):
                continue
            if p["type"] in ("int", "float"):
                if p["min"] is not None:
                    v = max(p["min"], v)
                if p["max"] is not None:
                    v = min(p["max"], v)
            out[k] = v
    return out


# ===========================================================================
#  Environment construction (the layouts/rewards live in rl/envs)
# ===========================================================================
def make_env(room_id: int, params: dict):
    seed = params.get("seed", 0)
    if room_id == 1:
        layout = (generate_pacman_layout(seed=params["map_seed"],
                                         n_coins=params["n_coins"],
                                         n_slippery=params["n_slippery"],
                                         guard_enabled=params["guard_enabled"])
                  if params["map_mode"] == "generated" else PACMAN_DEFAULT)
        return PacmanEnv(layout=layout, slip_prob=params["slip_prob"],
                         max_steps=params["max_steps"],
                         guard_enabled=params["guard_enabled"],
                         guard_mode=params["guard_mode"], seed=seed)
    if room_id == 2:
        layout = (generate_museum_layout(seed=params["map_seed"])
                  if params["map_mode"] == "generated" else MUSEUM_DEFAULT)
        return MuseumEnv(layout=layout, slip_prob=params["slip_prob"],
                         max_steps=params["max_steps"],
                         alarm_enabled=params["alarm_enabled"], seed=seed)
    if room_id == 3:
        layout = (generate_racing_layout(seed=params["map_seed"])
                  if params["map_mode"] == "generated" else RACING_DEFAULT)
        return RacingEnv(layout=layout, slip_prob=params["slip_prob"],
                         max_steps=params["max_steps"], seed=seed)
    if room_id == 4:
        if params.get("mode") == "freekick":
            # wall size is dynamic (set by kick distance), not by n_defenders
            return FreeKickEnv(keeper_speed=params["keeper_speed"],
                               kick_spot=params.get("kick_spot", "random"),
                               seed=seed)
        return FootballEnv(n_defenders=params["n_defenders"],
                           keeper_speed=params["keeper_speed"],
                           max_steps=params["max_steps"], seed=seed)
    if room_id == 5:
        return ObstacleEnv(n_cars=params["n_cars"], car_speed=params["car_speed"],
                           player_speed=params["player_speed"],
                           sensor_range=params["sensor_range"],
                           randomize=params["randomize"], seed=seed)
    raise ValueError(f"unknown room {room_id}")


FRAME_FNS = {1: fr.pacman_frame, 2: fr.museum_frame, 3: fr.racing_frame,
             4: fr.football_frame, 5: fr.crossroad_frame}
LAYOUT_FNS = {1: fr.pacman_layout, 2: fr.museum_layout, 3: fr.racing_layout,
              4: fr.football_layout, 5: fr.crossroad_layout}


def room_layout(room_id: int, params: Optional[dict] = None) -> dict:
    """Static layout for the canvas renderer (built from saved/default params)."""
    if params is None:
        info = load_json(info_path(room_id), default=None)
        params = merge_params(room_id, (info or {}).get("params"))
    env = make_env(room_id, params)
    return LAYOUT_FNS[room_id](env)


# ===========================================================================
#  Shared greedy evaluation
#  Runs ``n`` episodes for a stable success rate, but only saves the first
#  ``record_max`` as replays (so a big, steady metric doesn't flood the reel).
# ===========================================================================
def run_greedy_eval(room_id: int, env, act_fn: Callable, recorder: ReplayRecorder,
                    max_steps: int, n: int = 10, label: str = "Greedy eval",
                    record_max: Optional[int] = None) -> dict:
    frame_fn = FRAME_FNS[room_id]
    layout_fn = LAYOUT_FNS[room_id]
    if record_max is None:
        record_max = n
    wins, rewards, steps_list = 0, [], []
    for ep in range(1, n + 1):
        rec = ep <= record_max
        state = env.reset()
        if rec:
            recorder.start(f"eval_{ep:04d}")
            recorder.add_frame(frame_fn(env, None, 0.0, 0.0, False, {}))
        cum, steps, done = 0.0, 0, False
        success, fail = False, None
        while not done and steps < max_steps:
            a = act_fn(state)
            state, r, done, info = env.step(a)
            cum += r
            steps += 1
            if rec:
                recorder.add_frame(frame_fn(env, int(a), float(r), round(cum, 2),
                                            bool(done), info))
            if info.get("success") or info.get("escaped"):
                success = True
            for k in ("caught", "crash", "collision"):
                if info.get(k):
                    fail = k
            if done and not success and not fail and info.get("event"):
                fail = str(info["event"])[:28]
        wins += success
        rewards.append(cum)
        steps_list.append(steps)
        if rec:
            recorder.finish({"kind": "eval", "episode": ep,
                             "label": f"{label} {ep}", "reward": round(cum, 1),
                             "steps": steps, "success": success,
                             "fail_reason": None if success else (fail or "timeout"),
                             "layout": layout_fn(env)})
    return {"episodes": n, "success_rate": round(wins / n, 3),
            "avg_reward": round(float(np.mean(rewards)), 2),
            "avg_steps": round(float(np.mean(steps_list)), 1)}


# ===========================================================================
#  Room 3 race evaluation — Q-Learning car vs the SARSA rival, side by side
# ===========================================================================
def run_race_eval(env, rival_env, act_agent, act_rival, recorder,
                  max_steps: int, n: int = 10) -> dict:
    """Race the two greedy policies on the same track.

    Both cars step simultaneously in independent copies of the environment
    (each collects its own boost pads).  The replay stores the rival's
    position in ``rv`` on every frame; the last frame gets a
    ``won race`` / ``lost race`` event for the renderer.
    """
    frame_fn = FRAME_FNS[3]
    layout_fn = LAYOUT_FNS[3]
    wins, finishes, rewards, steps_list = 0, 0, [], []
    for ep in range(1, n + 1):
        # different slip draws per race, reproducible across runs
        env.rng = random.Random(ep)
        rival_env.rng = random.Random(1000 + ep)
        s_a = env.reset()
        s_r = rival_env.reset()
        recorder.start(f"eval_{ep:04d}")
        frame = frame_fn(env, None, 0.0, 0.0, False, {})
        frame["rv"] = list(rival_env.pos)
        recorder.add_frame(frame)

        cum, steps, done, info = 0.0, 0, False, {}
        rival_done, rival_steps, rival_fail = False, None, None
        while not done and steps < max_steps:
            a = act_agent(s_a)
            s_a, r, done, info = env.step(a)
            cum += r
            steps += 1
            rival_events = []
            if not rival_done:
                ra = act_rival(s_r)
                s_r, _, rival_done, rinfo = rival_env.step(ra)
                if rinfo.get("success"):
                    rival_steps = steps
                    rival_events.append("rival finished")
                elif rinfo.get("crash"):
                    rival_fail = "crash"
                    rival_events.append("rival crashed")
            frame = frame_fn(env, int(a), float(r), round(cum, 2),
                             bool(done), info)
            frame["rv"] = list(rival_env.pos)
            frame["ev"] = frame["ev"] + rival_events
            recorder.add_frame(frame)

        success = bool(info.get("success"))
        rival_finished = rival_steps is not None
        won = success and (not rival_finished or steps <= rival_steps)
        finishes += success
        wins += won
        rewards.append(cum)
        steps_list.append(steps)
        if success:
            frame["ev"] = frame["ev"] + (["won race"] if won else ["lost race"])
        fail = "crash" if info.get("crash") else None

        # epilogue: let the rival finish its lap on camera while the
        # outcome banner shows — a race should end with both cars home
        outcome_ev = list(frame["ev"])
        epilogue = 0
        while done and not rival_done and epilogue < 30:
            ra = act_rival(s_r)
            s_r, _, rival_done, rinfo = rival_env.step(ra)
            epilogue += 1
            ef = frame_fn(env, None, 0.0, round(cum, 2), True, {})
            ef["ev"] = list(outcome_ev)
            ef["rv"] = list(rival_env.pos)
            if rinfo.get("success"):
                rival_steps = steps + epilogue
                ef["ev"] = ef["ev"] + ["rival finished"]
            elif rinfo.get("crash"):
                ef["ev"] = ef["ev"] + ["rival crashed"]
            recorder.add_frame(ef)
        recorder.finish({
            "kind": "eval", "episode": ep,
            "label": f"Race {ep} — {'WON' if won else 'LOST'}",
            "reward": round(cum, 1), "steps": steps, "success": success,
            "fail_reason": None if success else (fail or "timeout"),
            "race": "won" if won else "lost",
            "rival_steps": rival_steps, "rival_fail": rival_fail,
            "layout": layout_fn(env),
        })
    return {"episodes": n,
            "success_rate": round(finishes / n, 3),
            "beat_rival": round(wins / n, 3),
            "avg_reward": round(float(np.mean(rewards)), 2),
            "avg_steps": round(float(np.mean(steps_list)), 1)}


# ===========================================================================
#  ROOM 1 — Dynamic Programming
# ===========================================================================
def _export_policy_json(room_id: int, env, entries: dict, flags: list) -> None:
    """entries: (cell, flag_idx) -> (value, action).  flags: list of labels."""
    size = env.rows
    out = {"type": "grid_policy", "size": size,
           "walls": sorted([list(w) for w in env.walls]), "flags": []}
    for fi, flag_label in enumerate(flags):
        vals = [[None] * size for _ in range(size)]
        acts = [[None] * size for _ in range(size)]
        for r in range(size):
            for c in range(size):
                e = entries.get(((r, c), fi))
                if e is not None:
                    vals[r][c] = round(float(e[0]), 2)
                    acts[r][c] = int(e[1])
        out["flags"].append({"label": flag_label, "values": vals, "actions": acts})
    save_json(policy_path(room_id), out)


def train_room1(params: dict, progress=None, stop=None) -> dict:
    env = make_env(1, params)
    mdp = env.build_mdp()
    algo = f"Dynamic Programming ({params['method'].replace('_', ' ')})"
    t0 = time.time()

    n_states = len(mdp.states)

    def cb(iteration, V, policy):
        if progress:
            progress(0, params["eval_episodes"], None,
                     f"solving MDP — sweep {iteration} over {n_states} states")
        if stop is not None and stop.is_set():
            raise StopRequested()

    solver = (dp_mod.value_iteration if params["method"] == "value_iteration"
              else dp_mod.policy_iteration)
    V, policy, diag = solver(mdp, gamma=params["gamma"], theta=params["theta"],
                             max_iter=params["max_iterations"], iteration_cb=cb)

    # ---- greedy evaluation episodes (stochastic slips → variance) ----------
    result = TrainResult()
    recorder = ReplayRecorder(1, clear=True)
    frame_fn = FRAME_FNS[1]
    layout = fr.pacman_layout(env)
    default_a = mdp.actions[0]

    for ep in range(1, params["eval_episodes"] + 1):
        if stop is not None and stop.is_set():
            break
        state = env.reset()
        rec = ep <= 10 or ep == params["eval_episodes"]
        if rec:
            recorder.start(f"eval_{ep:04d}")
            recorder.add_frame(frame_fn(env, None, 0.0, 0.0, False, {}))
        cum, steps, done, success, fail = 0.0, 0, False, False, None
        while not done and steps < params["max_steps"]:
            a = policy.get(state, default_a)
            state, r, done, info = env.step(a)
            cum += r
            steps += 1
            if rec:
                recorder.add_frame(frame_fn(env, int(a), float(r),
                                            round(cum, 2), bool(done), info))
            if info.get("escaped"):
                success = True
            if info.get("caught"):
                fail = "caught"
        result.log_episode(cum, steps, success, 0.0)
        if rec:
            recorder.finish({"kind": "eval", "episode": ep,
                             "label": f"Greedy run {ep}", "reward": round(cum, 1),
                             "steps": steps, "success": success,
                             "fail_reason": None if success else (fail or "timeout"),
                             "layout": layout})
        if progress:
            progress(ep, params["eval_episodes"],
                     lambda: mt.build_series(result), "evaluating optimal policy")

    # ---- diagnostics + persistence ------------------------------------------
    series = mt.build_series(result)
    series["dp_iteration"] = list(range(1, len(diag["value_delta"]) + 1))
    series["dp_delta"] = [round(float(d), 6) for d in diag["value_delta"]]
    series["dp_start_value"] = [round(float(v), 2) for v in diag["start_value"]]
    if diag["policy_changes"]:
        series["dp_policy_changes"] = diag["policy_changes"]

    summary = mt.build_summary(result, algo, time.time() - t0)
    summary["dp_iterations"] = len(diag["value_delta"])
    eval_summary = {"episodes": result.num_episodes,
                    "success_rate": round(result.success_rate(1000), 3),
                    "avg_reward": round(float(np.mean(result.episode_rewards)), 2)
                    if result.episode_rewards else None,
                    "avg_steps": round(float(np.mean(result.episode_steps)), 1)
                    if result.episode_steps else None}
    mt.save_metrics(1, algo, params, series, summary, eval_summary)

    with open(model_path(1), "wb") as f:
        pickle.dump({"V": V, "policy": policy}, f)
    save_json(info_path(1), {"room": 1, "algorithm": algo, "params": params,
                             "eval": eval_summary})

    # policy view: value + arrows for "all coins left" vs "all collected"
    best: dict = {}
    for s in mdp.states:
        if s == "CAUGHT":
            continue
        cell, mask = s[0], s[1]
        for fi, target in ((0, env.full_mask), (1, 0)):
            if mask == target:
                key = (cell, fi)
                if key not in best or V[s] > best[key][0]:
                    best[key] = (V[s], policy[s])
    _export_policy_json(1, env, best, ["All coins on board", "All coins collected"])
    return summary


# ===========================================================================
#  ROOMS 2 & 3 — tabular TD control (SARSA / Q-Learning)
# ===========================================================================
def train_td(room_id: int, params: dict, progress=None, stop=None) -> dict:
    algo = "SARSA" if room_id == 2 else "Q-Learning"
    env = make_env(room_id, params)
    recorder = ReplayRecorder(room_id, clear=True)
    recorder.plan_milestones(params["episodes"])
    layout_fn = LAYOUT_FNS[room_id]
    wrapped = RecordingEnv(env, recorder, FRAME_FNS[room_id],
                           meta_fn=lambda e: {"layout": layout_fn(e)})
    t0 = time.time()
    holder: dict = {}
    report_every = max(1, params["episodes"] // 200)

    def cb(ep, total, res):
        holder["result"] = res
        if progress and (ep % report_every == 0 or ep == total):
            progress(ep, total, lambda: mt.build_series(res), "training")
        if stop is not None and stop.is_set():
            raise StopRequested()

    stopped = False
    try:
        result = td_core.train(
            wrapped, algo="sarsa" if room_id == 2 else "qlearning",
            episodes=params["episodes"], alpha=params["alpha"],
            gamma=params["gamma"], eps_start=params["eps_start"],
            eps_end=params["eps_end"], eps_decay=params["eps_decay"],
            max_steps=params["max_steps"], seed=params["seed"],
            progress_cb=cb, record_replays=False, snapshot_fracs=())
    except StopRequested:
        stopped = True
        result = holder.get("result", TrainResult())
    wrapped.close()

    series = mt.build_series(result)

    # ---- Room 3: train the SARSA rival on the same track ------------------
    rival_result = None
    if room_id == 3 and not stopped and result.policy:
        def rival_cb(ep, total, res):
            holder["rival"] = res
            if progress and (ep % report_every == 0 or ep == total):
                def merged():
                    s = dict(series)
                    rs = mt.build_series(res)
                    s["rival_reward_avg"] = rs.get("reward_avg")
                    s["rival_success_rate"] = rs.get("success_rate")
                    return s
                progress(ep, total, merged,
                         "training the SARSA rival on the same track")
            if stop is not None and stop.is_set():
                raise StopRequested()

        def takes_safe_route(Qtable) -> bool:
            """Slip-free greedy rollout — the SARSA rival must not only FINISH
            but do it via the SAFE detour (up and around the barriers). That
            contrast — off-policy Q hugging the cliff, on-policy SARSA going the
            long way — is the whole point of the room, so if this seed's rival
            just learned the express we reject it and try another seed."""
            probe = make_env(3, params)
            probe.grid.slip_prob = 0.0
            barrier_row = min((r for r, _ in probe.crash), default=probe.rows)
            s, done, steps, pinfo = probe.reset(), False, 0, {}
            detoured = False
            while not done and steps < params["max_steps"]:
                q = Qtable.get(s)
                s, _, done, pinfo = probe.step(
                    int(np.argmax(q)) if q is not None else 0)
                steps += 1
                if probe.pos[0] < barrier_row:   # climbed above the barriers
                    detoured = True
            return bool(pinfo.get("success")) and detoured

        try:
            # SARSA's greedy trace varies with the seed; retry until one both
            # finishes AND takes the safe detour (the intended contrast). Each
            # retry is a full but fast tabular train, so we can afford plenty.
            for attempt in range(10):
                rival_result = td_core.train(
                    make_env(3, params), algo="sarsa",
                    episodes=params["episodes"], alpha=params["alpha"],
                    gamma=params["gamma"], eps_start=params["eps_start"],
                    eps_end=params["eps_end"], eps_decay=params["eps_decay"],
                    max_steps=params["max_steps"],
                    seed=params["seed"] + 1 + attempt,
                    progress_cb=rival_cb, record_replays=False,
                    snapshot_fracs=())
                if rival_result.policy and takes_safe_route(rival_result.policy):
                    break
        except StopRequested:
            stopped = True
            rival_result = None

    summary = mt.build_summary(result, algo, time.time() - t0)
    eval_summary = None

    if not stopped and result.policy:
        Q = result.policy  # dict: state -> np.array of action values

        def act(state):
            q = Q.get(state)
            return int(np.argmax(q)) if q is not None else 0

        model_dump = {"Q": {s: q.tolist() for s, q in Q.items()}}

        if rival_result is not None and rival_result.policy:
            rival_Q = rival_result.policy
            rs = mt.build_series(rival_result)
            series["rival_reward_avg"] = rs.get("reward_avg")
            series["rival_success_rate"] = rs.get("success_rate")
            summary["rival_algorithm"] = "SARSA"
            summary["rival_success_rate_last50"] = round(
                rival_result.success_rate(50), 3)
            model_dump["rival_Q"] = {s: q.tolist() for s, q in rival_Q.items()}

            def rival_act(state):
                q = rival_Q.get(state)
                return int(np.argmax(q)) if q is not None else 0

            eval_summary = run_race_eval(env, make_env(3, params),
                                         act, rival_act, recorder,
                                         params["max_steps"], n=10)
        else:
            eval_summary = run_greedy_eval(room_id, env, act, recorder,
                                           params["max_steps"], n=10)

        with open(model_path(room_id), "wb") as f:
            pickle.dump(model_dump, f)
        save_json(info_path(room_id), {"room": room_id, "algorithm": algo,
                                       "params": params, "eval": eval_summary})
        _export_td_policy(room_id, env, Q)

    summary["stopped"] = stopped
    mt.save_metrics(room_id, algo, params, series, summary, eval_summary)
    return summary


def _export_td_policy(room_id: int, env, Q: dict) -> None:
    """Best value/action per cell for the room's flag settings."""
    if room_id == 2:
        flag_of = lambda s: int(s[1])
        labels = ["Before the diamond", "Diamond in hand"]
    else:
        flag_of = lambda s: min(int(s[1]), env.n_gates)
        labels = ([f"To checkpoint {i + 1}" for i in range(env.n_gates)]
                  + ["All checkpoints — finish open"])
    best: dict = {}
    for s, q in Q.items():
        cell, fi = s[0], flag_of(s)
        v = float(np.max(q))
        key = (cell, fi)
        if key not in best or v > best[key][0]:
            best[key] = (v, int(np.argmax(q)))
    _export_policy_json(room_id, env, best, labels)


# ===========================================================================
#  ROOMS 4 & 5 — DQN (function approximation, continuous state)
# ===========================================================================
def train_dqn_room(room_id: int, params: dict, progress=None, stop=None) -> dict:
    algo = ("DQN (free kick)" if room_id == 4 and params.get("mode") == "freekick"
            else "DQN")
    env = make_env(room_id, params)
    freekick = room_id == 4 and params.get("mode") == "freekick"
    # a free kick is a single kick per episode, so train on many more (1-step)
    # episodes for enough gradient updates — 8x converges (~65%), 4x is noisy
    n_episodes = params["episodes"] * 8 if freekick else params["episodes"]
    max_steps = 1 if freekick else params["max_steps"]
    recorder = ReplayRecorder(room_id, clear=True)
    recorder.plan_milestones(n_episodes)
    layout_fn = LAYOUT_FNS[room_id]
    wrapped = RecordingEnv(env, recorder, FRAME_FNS[room_id],
                           meta_fn=lambda e: {"layout": layout_fn(e)})
    t0 = time.time()
    holder: dict = {}
    report_every = max(1, n_episodes // 200)

    def cb(ep, total, res):
        holder["result"] = res
        if progress and (ep % report_every == 0 or ep == total):
            progress(ep, total, lambda: mt.build_series(res), "training")
        if stop is not None and stop.is_set():
            raise StopRequested()

    stopped = False
    try:
        result = dqn_mod.train(
            wrapped, episodes=n_episodes, hidden=(128, 128),
            lr=params["lr"], gamma=params["gamma"],
            batch_size=params["batch_size"], buffer_size=params["buffer_size"],
            eps_start=params["eps_start"], eps_end=params["eps_end"],
            exploration_fraction=params["exploration_fraction"],
            learn_start=500 if freekick else 1000,
            target_update=params["target_update"], max_steps=max_steps,
            seed=params["seed"], progress_cb=cb,
            record_replays=False, snapshot_fracs=())
    except StopRequested:
        stopped = True
        result = holder.get("result", TrainResult())
    wrapped.close()

    series = mt.build_series(result)
    summary = mt.build_summary(result, algo, time.time() - t0)
    eval_summary = None

    if not stopped and result.policy is not None:
        import torch
        model = result.policy
        model.eval()

        def act(obs):
            with torch.no_grad():
                q = model(torch.as_tensor(obs, dtype=torch.float32))
                return int(torch.argmax(q).item())

        # evaluate over many episodes for a stable success rate, but only keep
        # a handful as replays. The free kick is one quick kick per episode, so
        # it can afford far more eval episodes.
        n_eval = 500 if freekick else 60
        eval_summary = run_greedy_eval(room_id, env, act, recorder,
                                       max_steps, n=n_eval, record_max=15)
        torch.save({"state_dict": model.state_dict(),
                    "obs_dim": env.obs_dim, "n_actions": env.n_actions,
                    "hidden": [128, 128]}, model_path(room_id, "pt"))
        save_json(info_path(room_id), {"room": room_id, "algorithm": algo,
                                       "params": params, "eval": eval_summary})

    summary["stopped"] = stopped
    mt.save_metrics(room_id, algo, params, series, summary, eval_summary)
    return summary


# ===========================================================================
#  Dispatch
# ===========================================================================
def run_training(room_id: int, user_params: Optional[dict] = None,
                 progress=None, stop=None) -> dict:
    ensure_dirs()
    params = merge_params(room_id, user_params)
    if room_id == 1:
        return train_room1(params, progress, stop)
    if room_id in (2, 3):
        return train_td(room_id, params, progress, stop)
    if room_id in (4, 5):
        return train_dqn_room(room_id, params, progress, stop)
    raise ValueError(f"unknown room {room_id}")


# ===========================================================================
#  CLI — test any room's training independently
# ===========================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train a single escape room.")
    ap.add_argument("--room", type=int, required=True, choices=list(ROOMS))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override any hyperparameter, e.g. --set episodes=300")
    args = ap.parse_args()

    overrides: dict = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        overrides[k] = v

    last = {"t": 0.0}

    def cli_progress(ep, total, series_fn, msg):
        if time.time() - last["t"] > 1.0 or ep == total:
            last["t"] = time.time()
            print(f"[room {args.room}] {msg}  {ep}/{total}")

    out = run_training(args.room, overrides, progress=cli_progress)
    print("SUMMARY:", out)
