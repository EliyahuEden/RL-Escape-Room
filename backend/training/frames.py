"""Per-room replay frame builders + static layout snapshots.

The canvas renderers in the frontend consume two things:

* a **layout** — the static geometry of the room (walls, coins, cameras,
  lanes...) captured once per training run and attached to every replay's
  meta, so replays always render against the exact map they were played on;
* a stream of **frames** — the dynamic state per step, kept compact:

    common      t, a (action), r (step reward), cum, done, ev (events)
    pacman      p [r,c] · coins remaining · guard [r,c] · open (door)
    museum      p [r,c] · d (has diamond) · guards [[r,c]] · alarm
    racing      p [r,c] · b (boosters remaining) · open (finish unlocked)
    football    p [x,y] · defs [[x,y]] · keeper [x,y] · shoot (in area)
                (+ kick flights expanded into extra ball-flight frames)
    crossroad   p [x,y] · cars [[x,y]] (metadata lives in the layout)
"""
from __future__ import annotations

from typing import List, Optional

from rl.envs.football import FootballEnv, FreeKickEnv
from rl.envs.museum import MuseumEnv
from rl.envs.obstacles import ObstacleEnv
from rl.envs.pacman import PacmanEnv
from rl.envs.racing import RacingEnv


def _round2(x) -> float:
    return round(float(x), 2)


def _pt(xy) -> List[float]:
    return [_round2(xy[0]), _round2(xy[1])]


# ===========================================================================
#  Event extraction (frontend shows these as popup badges)
# ===========================================================================
_EVENT_FLAGS = [
    ("slipped", "slip"), ("hit_wall", "wall"), ("door_bump", "locked door"),
    ("camera", "camera!"), ("trap", "trap"), ("caught", "caught"),
    ("alarm_triggered", "alarm!"), ("crash", "crash"),
    ("checkpoint", "checkpoint"),
    ("shortcut", "shortcut"), ("entered_shoot", "shooting zone"),
    ("dodge", "dodged"),
]

GOOD_EVENTS = {"coin", "diamond", "checkpoint", "escaped", "finished", "goal",
               "crossed", "dodged", "shooting zone", "shortcut"}


def events_from(info: dict, reward: float = 0.0) -> List[str]:
    ev = []
    for key, name in _EVENT_FLAGS:
        if info.get(key):
            ev.append(name)
    if info.get("escaped"):
        ev.append("escaped")
    if info.get("success") and "escaped" not in ev:
        ev.append(info.get("event", "success").split(" ")[0].lower()
                  if info.get("event") else "success")
    if info.get("event") and not info.get("success"):
        # short human event strings from the continuous rooms
        evt = str(info["event"])
        if len(evt) <= 26:
            ev.append(evt)
    return ev


# ===========================================================================
#  Room 1 — Pacman
# ===========================================================================
def pacman_layout(env: PacmanEnv) -> dict:
    return {
        "type": "pacman", "size": env.rows,
        "walls": sorted(env.walls),
        "coins": list(env.coins),
        "slippery": sorted(env.slippery),
        "start": list(env.start),
        "door": list(env.door),
        "guard": ({"mode": env.guard_mode, "route": [list(c) for c in env.guard_route]}
                  if env.guard_enabled else None),
    }


def pacman_frame(env: PacmanEnv, action, reward, cum, done, info) -> dict:
    ev = events_from(info)
    idx = env.coin_index.get(env.pos)
    if idx is not None and not (env.mask >> idx) & 1 and reward and reward > 0:
        ev.append("coin")
    return {
        "t": env.steps, "a": action, "r": _round2(reward or 0.0), "cum": cum,
        "done": done, "ev": ev,
        "p": list(env.pos),
        "coins": [list(c) for c in env.remaining_coins(env.mask)],
        "guard": list(env.guard_pos) if env.guard_enabled else None,
        "open": env.mask == 0,
    }


# ===========================================================================
#  Room 2 — Museum Heist
# ===========================================================================
def museum_layout(env: MuseumEnv) -> dict:
    return {
        "type": "museum", "size": env.rows,
        "walls": sorted(env.walls),
        "cameras": sorted(env.cameras),
        "camera_devices": sorted(env.camera_devices),
        "traps": sorted(env.traps),
        "slippery": sorted(env.slippery),
        "start": list(env.start),
        "exit": list(env.exit),
        "diamond": list(env.diamond),
        "guard_routes": [[list(c) for c in route] for route in env.guard_routes],
    }


def museum_frame(env: MuseumEnv, action, reward, cum, done, info) -> dict:
    ev = events_from(info)
    if info.get("success"):
        ev = [e for e in ev if e != "success"] + ["escaped"]
    if env.has_diamond and reward and reward >= env.r_diamond - 1.5:
        ev.append("diamond")
    return {
        "t": env.steps, "a": action, "r": _round2(reward or 0.0), "cum": cum,
        "done": done, "ev": ev,
        "p": list(env.pos),
        "d": int(env.has_diamond),
        "guards": [list(g) for g in env.guard_positions()],
        "alarm": env.alarmed,
    }


# ===========================================================================
#  Room 3 — Racing
# ===========================================================================
def racing_layout(env: RacingEnv) -> dict:
    return {
        "type": "racing", "size": env.rows,
        "walls": sorted(env.walls),
        "oil": sorted(env.oil),
        "mud": sorted(env.mud),
        "crash": sorted(env.crash),
        "checkpoints": [sorted([list(c) for c in gate])
                        for gate in env.checkpoints],
        "start": list(env.start),
        "finish": list(env.finish),
    }


def racing_frame(env: RacingEnv, action, reward, cum, done, info) -> dict:
    ev = events_from(info)
    if info.get("success"):
        ev = [e for e in ev if e != "success"] + ["finished"]
    return {
        "t": env.steps, "a": action, "r": _round2(reward or 0.0), "cum": cum,
        "done": done, "ev": ev,
        "p": list(env.pos),
        "ncp": env.next_cp,
        "open": env.finish_unlocked(env.next_cp),
    }


# ===========================================================================
#  Room 4 — Football (match + free kick share a frame shape)
# ===========================================================================
def football_layout(env) -> dict:
    base = {
        "type": "football", "W": env.W, "H": env.H,
        "goal_lo": env.goal_lo, "goal_hi": env.goal_hi,
    }
    if isinstance(env, FreeKickEnv):
        base.update(mode="freekick", keeper_x=env.keeper_x_pos,
                    shoot_x=0.0, kick=[env.kick_x, env.kick_y],
                    wall_x=env.wall_x)
    else:
        base.update(mode="match", keeper_x=env.keeper_x, shoot_x=env.shoot_x)
    return base


def _flight_frames(env, flight, t, cum, done, outcome_ev, base) -> List[dict]:
    """Expand a recorded ball flight into extra frames (every few ticks) so
    the shot animates in the replay viewer."""
    out = []
    step = max(1, len(flight) // 26)
    ticks = flight[::step]
    if flight and ticks[-1] is not flight[-1]:
        ticks.append(flight[-1])
    for i, tick in enumerate(ticks):
        last = i == len(ticks) - 1
        f = dict(base)
        f.update({
            "t": t, "a": None, "r": 0.0, "cum": cum,
            "done": done and last, "ev": outcome_ev if last else [],
            "fl": True,
            "ball": _pt(tick["ball"]),
            "keeper": _pt(tick["keeper"]),
        })
        if "ball_z" in tick:
            f["z"] = _round2(tick["ball_z"])
        out.append(f)
    return out


def football_frame(env, action, reward, cum, done, info) -> List[dict]:
    """Returns a list: several frames when the step contains a ball flight."""
    if isinstance(env, FreeKickEnv):
        player = [env.kick_x, env.kick_y]
        defenders = [list(w) for w in env.wall_players]
        keeper = [env.keeper_x_pos, env.keeper_y]
        in_shoot = True
    else:
        player = [env.x, env.y]
        defenders = [list(d) for d in env.defenders]
        keeper = [env.keeper_x, env.keeper_y]
        in_shoot = env.x >= env.shoot_x

    base = {
        "p": _pt(player),
        "defs": [_pt(d) for d in defenders],
        "keeper": _pt(keeper),
        "shoot": bool(in_shoot),
    }
    ev = events_from(info)
    t = env.attempts if isinstance(env, FreeKickEnv) else env.steps
    frame = dict(base)
    # while a flight follows, the episode visually ends on the LAST flight
    # frame — marking the kick frame done would flash "NO GOAL" pre-shot
    frame.update({"t": t, "a": action, "r": _round2(reward or 0.0), "cum": cum,
                  "done": done and not env._flight,
                  "ev": [] if env._flight else ev})

    if env._flight:
        t = frame["t"]
        return [frame] + _flight_frames(env, env._flight, t, cum, done, ev, base)
    return [frame]


# ===========================================================================
#  Room 5 — Cross the Road
# ===========================================================================
def crossroad_layout(env: ObstacleEnv) -> dict:
    return {
        "type": "crossroad", "W": env.W, "H": env.H,
        "goal_x": env.goal_x,
        "road_x_min": env.road_x_min,
        "road_x_max": env.road_x_max,
        "lane_xs": [_round2(x) for x in env.lane_xs],
        "sensor_range": env.sensor_range,
        "start": [_round2(env.start[0]), _round2(env.start[1])],
        "cars_meta": [{"w": c.width, "h": c.height, "color": c.color,
                       "dir": c.direction} for c in env.cars],
    }


def crossroad_frame(env: ObstacleEnv, action, reward, cum, done, info) -> dict:
    ev = events_from(info)
    if info.get("success"):
        ev = [e for e in ev if e != "success"] + ["crossed"]
    return {
        "t": env.steps, "a": action, "r": _round2(reward or 0.0), "cum": cum,
        "done": done, "ev": ev,
        "p": [_round2(env.x), _round2(env.y)],
        "cars": [[_round2(c.x), _round2(c.y)] for c in env.cars],
    }


# ===========================================================================
#  Action names (shown in the replay state monitor)
# ===========================================================================
GRID_ACTIONS = ["Up", "Down", "Left", "Right"]
FOOTBALL_ACTIONS = [
    "Up", "Down", "Left", "Right", "Stay",
    "Up-Left", "Up-Right", "Down-Left", "Down-Right",
    "Kick soft straight", "Kick soft curve-L", "Kick soft curve-R",
    "Kick hard straight", "Kick hard curve-L", "Kick hard curve-R",
]
FREEKICK_ACTIONS = [f"{aim} {power} {curve}"
                    for aim in ("low", "mid", "high")
                    for power in ("soft", "hard")
                    for curve in ("curve-L", "straight", "curve-R")]
CROSSROAD_ACTIONS = ["Up", "Down", "Left", "Right", "Stay"]


def action_names(room_id: int, mode: Optional[str] = None) -> List[str]:
    if room_id in (1, 2, 3):
        return GRID_ACTIONS
    if room_id == 4:
        return FREEKICK_ACTIONS if mode == "freekick" else FOOTBALL_ACTIONS
    return CROSSROAD_ACTIONS
