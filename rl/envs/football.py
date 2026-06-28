"""Room 4 — Football Final Shot (DQN, continuous state).

A 10x10 m pitch.  The player (with the ball) starts on the left and must dribble
past **chasing defenders** (random start positions every episode), get into the
**shooting area**, and then **kick the ball into the goal** past a goalkeeper who
**patrols side to side** across the goal mouth.

The kick is a real physical event: the ball *leaves the player* and flies on its
own.  The player chooses the **power** (soft / hard → ball speed) and the **curve**
(bend left / straight / right → a Magnus-style sideways acceleration during
flight).  The ball only counts if it crosses the goal line inside the mouth and
the patrolling keeper is not there to stop it — so the player must time the shot
for when the keeper has drifted away, and/or bend the ball around him.

Core player state is ``(x, y, vx, vy)``; the observation also includes the
keeper's position + patrol direction and each defender's relative position.

Actions (11)
------------
0 Up · 1 Down · 2 Left · 3 Right · 4 Stay ·
5 soft+straight · 6 soft+curve-left · 7 soft+curve-right ·
8 hard+straight · 9 hard+curve-left · 10 hard+curve-right
A kick taken inside the area ends the episode (goal / save / miss); a kick from
outside the area is a wasted touch (small penalty, play continues).
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

import numpy as np

UP, DOWN, LEFT, RIGHT, STAY, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = range(9)
_D = 1.0 / math.sqrt(2.0)  # diagonals are unit-normalised so they aren't faster
_VEL = {
    UP: (0.0, 1.0), DOWN: (0.0, -1.0), LEFT: (-1.0, 0.0), RIGHT: (1.0, 0.0),
    STAY: (0.0, 0.0),
    UP_LEFT: (-_D, _D), UP_RIGHT: (_D, _D), DOWN_LEFT: (-_D, -_D), DOWN_RIGHT: (_D, -_D),
}
# kick action -> (power_key, curve_sign)  (actions 9..14, after the 9 move actions)
_KICKS = {
    9: ("soft", 0.0), 10: ("soft", -1.0), 11: ("soft", 1.0),
    12: ("hard", 0.0), 13: ("hard", -1.0), 14: ("hard", 1.0),
}


class FootballEnv:
    def __init__(
        self,
        n_defenders: int = 3,
        randomize: bool = True,
        dt: float = 0.02,
        action_repeat: int = 10,
        player_speed: float = 1.0,
        def_speed: float = 0.6,
        keeper_speed: float = 1.5,
        keeper_reach: float = 0.7,
        shoot_x: float = 6.5,
        max_steps: int = 120,
        soft_speed: float = 7.0,
        hard_speed: float = 13.0,
        curve_acc: float = 9.0,
        max_flight_ticks: int = 220,
        tackle_radius: float = 0.5,
        danger_radius: float = 1.5,
        # rewards
        r_goal: float = 300.0,
        r_save: float = -10.0,
        r_miss: float = -30.0,
        r_wasted: float = -5.0,
        r_tackle: float = -50.0,
        r_oob: float = -30.0,
        r_enter_shoot: float = 10.0,
        r_dodge: float = 20.0,
        r_step: float = -0.3,
        r_dwell: float = -1.0,
        r_timeout: float = -25.0,
        k_progress: float = 3.0,
        seed: Optional[int] = None,
    ) -> None:
        self.W = self.H = 10.0
        self.goal_lo, self.goal_hi = 3.5, 6.5
        self.goal_cy = 0.5 * (self.goal_lo + self.goal_hi)
        self.keeper_x = self.W - 0.3
        self.k_lo, self.k_hi = self.goal_lo + 0.3, self.goal_hi - 0.3
        self.n_defenders = int(n_defenders)
        self.randomize = randomize
        self.dt = dt
        self.dt_eff = dt * action_repeat
        self.player_speed = player_speed
        self.def_speed = def_speed
        self.keeper_speed = keeper_speed
        self.keeper_reach = keeper_reach
        self.shoot_x = shoot_x
        self.power = {"soft": soft_speed, "hard": hard_speed}
        self.curve_acc = curve_acc
        self.max_flight_ticks = max_flight_ticks
        self.tackle_radius = tackle_radius
        self.danger_radius = danger_radius
        self.r_goal, self.r_save, self.r_miss = r_goal, r_save, r_miss
        self.r_wasted, self.r_tackle, self.r_oob = r_wasted, r_tackle, r_oob
        self.r_enter_shoot, self.r_dodge = r_enter_shoot, r_dodge
        self.r_step, self.k_progress = r_step, k_progress
        self.r_dwell = r_dwell
        self.r_timeout = r_timeout
        self.max_steps = int(max_steps)
        self.rng = random.Random(seed)

        self.n_actions = 15  # 9 moves (incl. diagonals) + 6 kicks
        self.obs_dim = 10 + 3 * self.n_defenders
        self._diag = math.hypot(self.W, self.H)
        self.reset()

    # -- setup --------------------------------------------------------------
    def _spawn_defenders(self) -> List[List[float]]:
        defs: List[List[float]] = []
        tries = 0
        while len(defs) < self.n_defenders and tries < 300:
            tries += 1
            if self.randomize:
                dx = self.rng.uniform(3.0, self.shoot_x + 0.5)
                dy = self.rng.uniform(1.0, self.H - 1.0)
            else:
                dx = 4.5
                dy = 1.5 + 7.0 * (len(defs) / max(1, self.n_defenders - 1))
            if math.hypot(dx - self.x, dy - self.y) > 2.0 and \
               all(math.hypot(dx - ox, dy - oy) > 1.2 for ox, oy in defs):
                defs.append([dx, dy])
        return defs

    def reset(self):
        self.x = 1.0
        self.y = self.rng.uniform(3.0, 7.0) if self.randomize else 5.0
        self.vx = self.vy = 0.0
        self.keeper_y = self.goal_cy
        self.kdir = 1.0 if self.rng.random() < 0.5 else -1.0
        self.defenders = self._spawn_defenders()
        self.steps = 0
        self._entered_shoot = False
        self._in_danger = False
        self._prev_goal_dist = self._goal_dist()
        self._flight = None
        return self._obs()

    # -- helpers ------------------------------------------------------------
    def _goal_dist(self) -> float:
        return math.hypot(self.W - self.x, self.goal_cy - self.y)

    def _min_def_dist(self) -> float:
        if not self.defenders:
            return self._diag
        return min(math.hypot(d[0] - self.x, d[1] - self.y) for d in self.defenders)

    def _advance_keeper(self, dt: float) -> None:
        self.keeper_y += self.keeper_speed * dt * self.kdir
        if self.keeper_y < self.k_lo:
            self.keeper_y = self.k_lo
            self.kdir = 1.0
        elif self.keeper_y > self.k_hi:
            self.keeper_y = self.k_hi
            self.kdir = -1.0

    def _move_defenders(self) -> None:
        step = self.def_speed * self.dt_eff
        for d in self.defenders:
            dx, dy = self.x - d[0], self.y - d[1]
            dist = math.hypot(dx, dy) or 1.0
            d[0] = min(max(d[0] + step * dx / dist, 0.0), self.W)
            d[1] = min(max(d[1] + step * dy / dist, 0.0), self.H)

    # -- observation --------------------------------------------------------
    def _obs(self) -> np.ndarray:
        feats = [
            self.x / self.W, self.y / self.H, self.vx, self.vy,
            (self.W - self.x) / self.W, (self.goal_cy - self.y) / self.H,
            1.0 if self.x >= self.shoot_x else 0.0,
            (self.keeper_y - self.goal_lo) / (self.goal_hi - self.goal_lo),
            self.kdir,
            self._goal_dist() / self._diag,
        ]
        defs = sorted(self.defenders, key=lambda d: math.hypot(d[0] - self.x, d[1] - self.y))
        for d in defs:
            feats += [(d[0] - self.x) / self.W, (d[1] - self.y) / self.H,
                      math.hypot(d[0] - self.x, d[1] - self.y) / self._diag]
        return np.asarray(feats, dtype=np.float32)

    # -- the kick: ball leaves the player and flies -------------------------
    def _resolve_kick(self, power_key: str, curve_sign: float):
        speed = self.power[power_key]
        # aim straight at the goal line on the player's own line (clamped into the
        # mouth) so a straight shot stays on target; curve then bends it from there.
        target_y = min(max(self.y, self.goal_lo + 0.2), self.goal_hi - 0.2)
        dvec = np.array([self.W - self.x, target_y - self.y], dtype=float)
        n = np.linalg.norm(dvec) or 1.0
        vel = dvec / n * speed
        bx, by = self.x, self.y
        flight = [{"ball": (bx, by), "keeper": (self.keeper_x, self.keeper_y)}]
        outcome = "miss"
        passed_keeper = False
        for _ in range(self.max_flight_ticks):
            v = math.hypot(*vel) or 1.0
            perp = np.array([-vel[1], vel[0]]) / v  # +90° rotation
            vel = vel + curve_sign * self.curve_acc * perp * self.dt
            bx += vel[0] * self.dt
            by += vel[1] * self.dt
            self._advance_keeper(self.dt)
            flight.append({"ball": (bx, by), "keeper": (self.keeper_x, self.keeper_y)})
            if not passed_keeper and bx >= self.keeper_x:
                passed_keeper = True
                if abs(by - self.keeper_y) <= self.keeper_reach:
                    outcome = "save"
                    break
            if bx >= self.W:
                outcome = "goal" if self.goal_lo <= by <= self.goal_hi else "miss"
                break
            if by < 0 or by > self.H or bx < 0:
                outcome = "miss"
                break
        self._flight = flight
        return outcome

    # -- step ---------------------------------------------------------------
    def step(self, action: int):
        self.steps += 1
        reward = self.r_step
        info = {"success": False}
        self._flight = None

        if action in _KICKS:  # a kick
            if self.x < self.shoot_x:
                reward += self.r_wasted
                info["event"] = "shot from too far — wasted"
                return self._obs(), reward, False, info
            power_key, curve_sign = _KICKS[action]
            outcome = self._resolve_kick(power_key, curve_sign)
            info["kick"] = (power_key, curve_sign)
            if outcome == "goal":
                reward += self.r_goal
                info["event"] = "GOAL!"
                info["success"] = True
            elif outcome == "save":
                reward += self.r_save
                info["event"] = "saved by the keeper"
            else:
                reward += self.r_miss
                info["event"] = "missed the goal"
            return self._obs(), reward, True, info

        # movement (dribble)
        vx, vy = _VEL[action]
        self.vx, self.vy = float(vx), float(vy)
        self.x += vx * self.player_speed * self.dt_eff
        self.y += vy * self.player_speed * self.dt_eff
        self._advance_keeper(self.dt_eff)  # keeper keeps patrolling

        if self.x > self.W:
            self.x = self.W
        if self.x < 0 or self.y < 0 or self.y > self.H:
            self.x = min(max(self.x, 0.0), self.W)
            self.y = min(max(self.y, 0.0), self.H)
            reward += self.r_oob
            info["event"] = "out of bounds"
            return self._obs(), reward, True, info

        gd = self._goal_dist()
        reward += self.k_progress * (self._prev_goal_dist - gd)
        self._prev_goal_dist = gd
        self._move_defenders()

        if self.x >= self.shoot_x:
            if not self._entered_shoot:
                self._entered_shoot = True
                reward += self.r_enter_shoot
                info["entered_shoot"] = True
            reward += self.r_dwell  # standing in the area without shooting is costly

        mind = self._min_def_dist()
        if mind <= self.tackle_radius:
            reward += self.r_tackle
            info["event"] = "tackled — ball lost"
            return self._obs(), reward, True, info
        if self._in_danger and mind > self.danger_radius:
            reward += self.r_dodge
            info["dodge"] = True
        self._in_danger = mind <= self.danger_radius

        if self.steps >= self.max_steps:  # shot clock: dawdling is a turnover
            reward += self.r_timeout
            info["event"] = "ran out of time"
            return self._obs(), reward, True, info
        return self._obs(), reward, False, info

    # -- rendering ----------------------------------------------------------
    def render_state(self) -> dict:
        state = {
            "x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy,
            "ball": (self.x, self.y), "ball_in_flight": False,
            "defenders": [tuple(d) for d in self.defenders],
            "keeper": (self.keeper_x, self.keeper_y),
            "in_shoot": self.x >= self.shoot_x,
        }
        if self._flight is not None:
            state["flight"] = self._flight
        return state


# ─── Free Kick Mode ─────────────────────────────────────────────────────────

_FK_AIMS = ("low", "mid", "high")
_FK_POWERS = ("soft", "hard")
_FK_CURVES = ("left", "straight", "right")

def _fk_action_table():
    table = {}
    idx = 0
    for aim in _FK_AIMS:
        for power in _FK_POWERS:
            for curve in _FK_CURVES:
                table[idx] = (aim, power, curve)
                idx += 1
    return table

_FK_ACTIONS = _fk_action_table()


class FreeKickEnv:
    """Free kick: the player stands at a fixed spot and kicks the ball toward
    the goal.  A **wall** of defenders stands between the player and the goal,
    and a keeper patrols the goal mouth.

    The ball has 3D physics: it travels in (x, y) on the pitch and also rises
    and falls in z (height).  High shots arc over the wall but are slower and
    less accurate; low shots are fast but can be blocked by the wall.  Curve
    bends the ball sideways (Magnus effect) to go around the keeper.

    Actions (18): 3 aims (low/mid/high) × 2 powers (soft/hard) × 3 curves
    (left/straight/right).  Each episode is a single kick — the agent picks
    one action and the ball flight resolves immediately.
    """

    def __init__(
        self,
        n_wall: int = 3,
        kick_x: float = 5.0,
        kick_y: float = 5.0,
        wall_x: float = 7.5,
        keeper_speed: float = 1.5,
        keeper_reach: float = 0.7,
        soft_speed: float = 7.0,
        hard_speed: float = 13.0,
        curve_acc: float = 14.0,
        max_flight_ticks: int = 250,
        wall_height: float = 1.2,
        wall_block_radius: float = 0.3,
        r_goal: float = 300.0,
        r_save: float = 20.0,
        r_miss: float = -15.0,
        r_blocked: float = -30.0,
        r_post: float = -5.0,
        max_attempts: int = 5,
        randomize: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.W = self.H = 10.0
        self.goal_lo, self.goal_hi = 3.5, 6.5
        self.goal_cy = 0.5 * (self.goal_lo + self.goal_hi)
        self.keeper_x_pos = self.W - 0.3
        self.k_lo = self.goal_lo + 0.3
        self.k_hi = self.goal_hi - 0.3
        self.n_wall = int(n_wall)
        self.kick_x_base = kick_x
        self.kick_y_base = kick_y
        self.kick_x = kick_x
        self.kick_y = kick_y
        self.wall_x_offset = 2.0
        self.wall_x = wall_x
        self.keeper_speed = keeper_speed
        self.keeper_reach = keeper_reach
        self.power = {"soft": soft_speed, "hard": hard_speed}
        self.curve_acc = curve_acc
        self.max_flight_ticks = max_flight_ticks
        self.wall_height = wall_height
        self.wall_block_radius = wall_block_radius
        self.r_goal = r_goal
        self.r_save = r_save
        self.r_miss = r_miss
        self.r_blocked = r_blocked
        self.r_post = r_post
        self.randomize = randomize
        self.rng = random.Random(seed)

        self.max_attempts = max_attempts
        self.attempts = 0
        self.dt = 0.02
        self.n_actions = len(_FK_ACTIONS)
        self.obs_dim = 4 + 2 * self.n_wall
        self.reset()

    def _spawn_wall(self) -> List[List[float]]:
        center = self.kick_y
        spacing = 0.9
        wall = []
        for i in range(self.n_wall):
            wy = center + (i - (self.n_wall - 1) / 2.0) * spacing
            if self.randomize:
                wy += self.rng.uniform(-0.15, 0.15)
            wall.append([self.wall_x, min(max(wy, 1.0), self.H - 1.0)])
        return wall

    def reset(self):
        self.wall_x = self.kick_x + self.wall_x_offset
        self.wall_x = min(self.wall_x, self.W - 1.5)
        self.keeper_y = self.goal_cy
        self.kdir = 1.0 if self.rng.random() < 0.5 else -1.0
        self.wall_players = self._spawn_wall()
        self.attempts = 0
        self._flight = None
        return self._obs()

    def _obs(self) -> np.ndarray:
        feats = [
            (self.keeper_y - self.goal_lo) / (self.goal_hi - self.goal_lo),
            self.kdir,
            self.kick_x / self.W,
            self.kick_y / self.H,
        ]
        for wx, wy in self.wall_players:
            feats += [(wx - self.kick_x) / self.W, (wy - self.kick_y) / self.H]
        return np.asarray(feats, dtype=np.float32)

    def _resolve_kick(self, aim: str, power_key: str, curve_dir: str):
        speed = self.power[power_key]
        curve_sign = {"left": -1.0, "straight": 0.0, "right": 1.0}[curve_dir]

        launch_angles = {"low": 5.0, "mid": 18.0, "high": 32.0}
        angle_deg = launch_angles[aim]
        if self.randomize:
            angle_deg += self.rng.uniform(-2.0, 2.0)
        angle_rad = math.radians(angle_deg)

        horiz_speed = speed * math.cos(angle_rad)
        vz = speed * math.sin(angle_rad)

        target_y = min(max(self.kick_y, self.goal_lo + 0.2), self.goal_hi - 0.2)
        dvec = np.array([self.W - self.kick_x, target_y - self.kick_y], dtype=float)
        n = np.linalg.norm(dvec) or 1.0
        vel = dvec / n * horiz_speed

        bx, by, bz = self.kick_x, self.kick_y, 0.0
        gravity = 9.8
        flight = [{"ball": (bx, by), "ball_z": bz,
                   "keeper": (self.keeper_x_pos, self.keeper_y)}]
        outcome = "miss"

        for _ in range(self.max_flight_ticks):
            v = math.hypot(*vel) or 1.0
            perp = np.array([-vel[1], vel[0]]) / v
            vel = vel + curve_sign * self.curve_acc * perp * self.dt

            bx += vel[0] * self.dt
            by += vel[1] * self.dt
            vz -= gravity * self.dt
            bz += vz * self.dt
            if bz < 0:
                bz = 0.0

            self.keeper_y += self.keeper_speed * self.dt * self.kdir
            if self.keeper_y < self.k_lo:
                self.keeper_y = self.k_lo
                self.kdir = 1.0
            elif self.keeper_y > self.k_hi:
                self.keeper_y = self.k_hi
                self.kdir = -1.0

            flight.append({"ball": (bx, by), "ball_z": bz,
                           "keeper": (self.keeper_x_pos, self.keeper_y)})

            for wx, wy in self.wall_players:
                if (abs(bx - wx) < self.wall_block_radius and
                        abs(by - wy) < self.wall_block_radius and
                        bz < self.wall_height):
                    self._flight = flight
                    return "blocked"

            if bx >= self.keeper_x_pos and bz < 2.5:
                if abs(by - self.keeper_y) <= self.keeper_reach:
                    outcome = "save"
                    break

            if bx >= self.W:
                if self.goal_lo <= by <= self.goal_hi and bz < 2.44:
                    outcome = "goal"
                else:
                    outcome = "miss"
                break

            if by < 0 or by > self.H or bx < 0:
                outcome = "miss"
                break

        self._flight = flight
        return outcome

    def step(self, action: int):
        aim, power_key, curve_dir = _FK_ACTIONS[action]
        outcome = self._resolve_kick(aim, power_key, curve_dir)
        self.attempts += 1

        if outcome == "goal":
            reward = self.r_goal
            event = "GOAL!"
            success = True
            done = True
        elif outcome == "save":
            reward = self.r_save
            event = "Saved by the keeper!"
            success = False
            done = self.attempts >= self.max_attempts
        elif outcome == "blocked":
            reward = self.r_blocked
            event = "Blocked by the wall!"
            success = False
            done = self.attempts >= self.max_attempts
        else:
            reward = self.r_miss
            event = "Missed the goal!"
            success = False
            done = self.attempts >= self.max_attempts

        if done and not success:
            event += " (out of attempts)"

        info = {"success": success, "event": event, "outcome": outcome,
                "kick": (aim, power_key, curve_dir)}
        return self._obs(), reward, done, info

    def render_state(self) -> dict:
        state = {
            "x": self.kick_x, "y": self.kick_y, "vx": 0, "vy": 0,
            "ball": (self.kick_x, self.kick_y), "ball_in_flight": False,
            "defenders": [tuple(w) for w in self.wall_players],
            "keeper": (self.keeper_x_pos, self.keeper_y),
            "in_shoot": True,
            "mode": "freekick",
        }
        if self._flight is not None:
            state["flight"] = self._flight
        return state
