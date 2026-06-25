"""Room 5 (optional) — Dynamic Obstacles (DQN + look-ahead sensors).

A continuous 10x10 m room.  The player must cross from the left to the **exit**
on the right while avoiding circular **obstacles** of width 0.5 m (radius 0.25).
The number and positions of the obstacles are **dynamic**: a fresh random layout
is generated every episode, so the agent cannot memorise a map — it must learn a
reactive avoidance policy from its **sensors**.

Observation (the controllable part)
-----------------------------------
The agent always sees its own dynamics ``(x, y, vx, vy)`` and the direction to
the exit.  On top of that it senses the nearest obstacles whose **centre** lies
within ``sensor_range`` metres of its own centre (the "see X metres ahead"
control).  Each sensor slot reports the obstacle's relative position and how
close it is; empty slots read as "clear".  Lowering ``sensor_range`` makes the
agent short-sighted and the task much harder.

Because the policy is purely sensor-based it generalises, so after training you
can drop it into a brand-new random room and watch it cope.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np

UP, DOWN, LEFT, RIGHT, STAY = range(5)
_VEL = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0), STAY: (0, 0)}


class ObstacleEnv:
    def __init__(
        self,
        n_obstacles: int = 6,
        obstacle_radius: float = 0.25,
        player_radius: float = 0.2,
        sensor_range: float = 3.0,
        max_sensed: int = 4,
        randomize: bool = True,
        dt: float = 0.02,
        action_repeat: int = 10,
        player_speed: float = 1.0,
        # rewards
        r_goal: float = 200.0,
        r_collision: float = -100.0,
        r_oob: float = -30.0,
        r_near: float = -0.5,
        r_step: float = -0.1,
        k_progress: float = 3.0,
        near_radius: float = 0.6,
        seed: Optional[int] = None,
    ) -> None:
        self.W = self.H = 10.0
        self.n_obstacles = int(n_obstacles)
        self.obstacle_radius = obstacle_radius
        self.player_radius = player_radius
        self.sensor_range = sensor_range
        self.max_sensed = int(max_sensed)
        self.randomize = randomize
        self.dt_eff = dt * action_repeat
        self.player_speed = player_speed
        self.r_goal, self.r_collision, self.r_oob = r_goal, r_collision, r_oob
        self.r_near, self.r_step, self.k_progress = r_near, r_step, k_progress
        self.near_radius = near_radius
        self.rng = random.Random(seed)

        self.goal_x = self.W - 0.25
        self.goal_cy = self.H / 2.0
        self.start = (0.5, self.H / 2.0)

        self.n_actions = 5
        self.obs_dim = 6 + 3 * self.max_sensed
        self._diag = math.hypot(self.W, self.H)
        self.obstacles: List[Tuple[float, float]] = []
        self._fixed = None
        self.reset()

    # -- layout -------------------------------------------------------------
    def _generate_obstacles(self) -> List[Tuple[float, float]]:
        obs: List[Tuple[float, float]] = []
        tries = 0
        while len(obs) < self.n_obstacles and tries < 500:
            tries += 1
            x = self.rng.uniform(2.0, self.W - 1.5)
            y = self.rng.uniform(0.6, self.H - 0.6)
            if all(math.hypot(x - ox, y - oy) > 1.0 for ox, oy in obs):
                obs.append((x, y))
        return obs

    def reset(self):
        if self.randomize or self._fixed is None:
            self.obstacles = self._generate_obstacles()
            if not self.randomize:
                self._fixed = self.obstacles
        else:
            self.obstacles = self._fixed
        self.x, self.y = self.start
        self.vx = self.vy = 0.0
        self.steps = 0
        self._prev_dx = self.goal_x - self.x
        return self._obs()

    # -- helpers ------------------------------------------------------------
    def _min_obstacle_dist(self) -> float:
        if not self.obstacles:
            return self._diag
        return min(math.hypot(self.x - ox, self.y - oy) for ox, oy in self.obstacles)

    def _collision(self) -> bool:
        thresh = self.player_radius + self.obstacle_radius
        return any(math.hypot(self.x - ox, self.y - oy) <= thresh
                   for ox, oy in self.obstacles)

    def _obs(self) -> np.ndarray:
        feats = [self.x / self.W, self.y / self.H, self.vx, self.vy,
                 (self.goal_x - self.x) / self.W, (self.goal_cy - self.y) / self.H]
        sensed = sorted(
            ((math.hypot(self.x - ox, self.y - oy), ox, oy) for ox, oy in self.obstacles),
            key=lambda t: t[0],
        )
        slots = 0
        for dist, ox, oy in sensed:
            if dist > self.sensor_range or slots >= self.max_sensed:
                break
            feats += [(ox - self.x) / self.sensor_range,
                      (oy - self.y) / self.sensor_range,
                      1.0 - dist / self.sensor_range]
            slots += 1
        feats += [0.0, 0.0, 0.0] * (self.max_sensed - slots)  # empty = "clear"
        return np.asarray(feats, dtype=np.float32)

    # -- step ---------------------------------------------------------------
    def step(self, action: int):
        self.steps += 1
        reward = self.r_step
        info = {"success": False}

        vx, vy = _VEL[action]
        self.vx, self.vy = float(vx), float(vy)
        self.x += vx * self.player_speed * self.dt_eff
        self.y += vy * self.player_speed * self.dt_eff

        if self.x < 0 or self.y < 0 or self.y > self.H:
            self.x = min(max(self.x, 0.0), self.W)
            self.y = min(max(self.y, 0.0), self.H)
            reward += self.r_oob
            info["event"] = "out of bounds"
            return self._obs(), reward, True, info

        if self._collision():
            reward += self.r_collision
            info["event"] = "hit an obstacle"
            return self._obs(), reward, True, info

        if self.x >= self.goal_x:
            reward += self.r_goal
            info["event"] = "reached the exit"
            info["success"] = True
            return self._obs(), reward, True, info

        dx = self.goal_x - self.x
        reward += self.k_progress * (self._prev_dx - dx)
        self._prev_dx = dx
        if self._min_obstacle_dist() < self.near_radius:
            reward += self.r_near  # discourage grazing obstacles

        return self._obs(), reward, False, info

    # -- rendering ----------------------------------------------------------
    def render_state(self) -> dict:
        return {
            "x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy,
            "obstacles": list(self.obstacles),
            "obstacle_radius": self.obstacle_radius,
            "sensor_range": self.sensor_range,
            "goal_x": self.goal_x,
        }
