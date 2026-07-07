"""Room 5 — Chicken road crossing (DQN + look-ahead sensors).

A continuous 10x10 m road.  The chicken starts on the left sidewalk and must
reach the right edge while traffic moves vertically through the box.  Cars are
spawned in lanes with alternating up/down directions and wrap from just outside
one edge back to the other, so the policy has to learn when to move, dodge, or
wait instead of memorising a static layout.

Observation (the controllable part)
-----------------------------------
The chicken sees its own dynamics ``(x, y, vx, vy)`` and the direction to the
far sidewalk.  It also senses the nearest cars inside ``sensor_range``.  Each
sensor slot reports the car's relative centre, vertical velocity, and closeness.
Empty slots read as "clear".  The policy is therefore reactive and can be tested
on fresh traffic patterns after training.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

UP, DOWN, LEFT, RIGHT, STAY, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT = range(9)
_D = 1.0 / math.sqrt(2.0)  # diagonals are unit-normalised so they aren't faster
_VEL = {
    UP: (0.0, 1.0), DOWN: (0.0, -1.0), LEFT: (-1.0, 0.0), RIGHT: (1.0, 0.0),
    STAY: (0.0, 0.0),
    UP_LEFT: (-_D, _D), UP_RIGHT: (_D, _D), DOWN_LEFT: (-_D, -_D), DOWN_RIGHT: (_D, -_D),
}


@dataclass
class Car:
    x: float
    y: float
    direction: float
    speed: float
    width: float
    height: float
    color: str

    def copy(self) -> "Car":
        return Car(self.x, self.y, self.direction, self.speed, self.width, self.height, self.color)


class ObstacleEnv:
    def __init__(
        self,
        n_cars: Optional[int] = None,
        *,
        n_obstacles: Optional[int] = None,
        car_width: float = 0.55,
        car_height: float = 1.05,
        player_radius: float = 0.2,
        sensor_range: float = 3.5,
        max_sensed: int = 6,
        randomize: bool = True,
        dt: float = 0.02,
        action_repeat: int = 10,
        player_speed: float = 1.6,
        car_speed: float = 1.35,
        # rewards
        r_goal: float = 250.0,
        r_collision: float = -140.0,
        r_oob: float = -40.0,
        r_near: float = -0.6,
        r_step: float = -0.08,
        k_progress: float = 4.0,
        near_radius: float = 0.55,
        seed: Optional[int] = None,
    ) -> None:
        self.W = self.H = 10.0
        # Backwards-compatible alias for old UI/session configs.
        if n_cars is None:
            n_cars = n_obstacles if n_obstacles is not None else 14
        self.n_cars = int(n_cars)
        self.car_width = float(car_width)
        self.car_height = float(car_height)
        self.player_radius = player_radius
        self.sensor_range = sensor_range
        self.max_sensed = int(max_sensed)
        self.randomize = randomize
        self.dt = dt
        self.action_repeat = int(action_repeat)
        self.player_speed = player_speed
        self.car_speed = car_speed
        self.max_car_speed = max(0.1, car_speed * 1.5)
        self.r_goal, self.r_collision, self.r_oob = r_goal, r_collision, r_oob
        self.r_near, self.r_step, self.k_progress = r_near, r_step, k_progress
        self.near_radius = near_radius
        self.rng = random.Random(seed)

        self.goal_x = self.W - 0.25
        self.goal_cy = self.H / 2.0
        self.start = (0.45, self.H / 2.0)
        self.road_x_min = 1.25
        self.road_x_max = self.goal_x - 0.65
        self.wrap_margin = max(1.0, self.car_height)
        self.lane_xs: List[float] = []

        self.n_actions = 9  # 4 orthogonal + 4 diagonal moves + Stay
        self.obs_dim = 6 + 4 * self.max_sensed
        self._diag = math.hypot(self.W, self.H)
        self.cars: List[Car] = []
        self._fixed: Optional[List[Car]] = None
        self.reset()

    # -- layout -------------------------------------------------------------
    def _generate_cars(self) -> List[Car]:
        cars: List[Car] = []
        lane_count = min(8, max(4, math.ceil(self.n_cars / 2)))
        lane_min = self.road_x_min + self.car_width * 0.9
        lane_max = self.road_x_max - self.car_width * 0.6
        self.lane_xs = np.linspace(lane_min, lane_max, lane_count).tolist()
        per_lane = [self.n_cars // lane_count] * lane_count
        for lane_idx in range(self.n_cars % lane_count):
            per_lane[lane_idx] += 1

        colors = ["#ef4444", "#f97316", "#8b5cf6", "#06b6d4", "#22c55e", "#facc15"]
        span = self.H + 2 * self.wrap_margin
        for lane_idx, count in enumerate(per_lane):
            if count <= 0:
                continue
            direction = 1.0 if lane_idx % 2 == 0 else -1.0
            lane_speed = self.car_speed * self.rng.uniform(0.78, 1.28)
            spacing = span / count
            phase = self.rng.uniform(0.0, spacing)
            for car_idx in range(count):
                y = -self.wrap_margin + ((phase + car_idx * spacing) % span)
                cars.append(
                    Car(
                        x=self.lane_xs[lane_idx] + self.rng.uniform(-0.08, 0.08),
                        y=y,
                        direction=direction,
                        speed=lane_speed,
                        width=self.car_width,
                        height=self.car_height,
                        color=colors[(lane_idx + car_idx) % len(colors)],
                    )
                )
        return cars

    def reset(self):
        if self.randomize or self._fixed is None:
            self.cars = self._generate_cars()
            if not self.randomize:
                self._fixed = [car.copy() for car in self.cars]
        else:
            self.cars = [car.copy() for car in self._fixed]
        self.x, self.y = self.start
        self.vx = self.vy = 0.0
        self.steps = 0
        self._prev_dx = self.goal_x - self.x
        return self._obs()

    # -- helpers ------------------------------------------------------------
    def _wrap_car(self, car: Car) -> None:
        span = self.H + 2 * self.wrap_margin
        if car.direction > 0 and car.y - car.height / 2 > self.H + self.wrap_margin:
            car.y -= span
        elif car.direction < 0 and car.y + car.height / 2 < -self.wrap_margin:
            car.y += span

    def _advance_cars(self, dt: float) -> None:
        for car in self.cars:
            car.y += car.direction * car.speed * dt
            self._wrap_car(car)

    def _distance_to_car(self, car: Car) -> float:
        left = car.x - car.width / 2
        right = car.x + car.width / 2
        bottom = car.y - car.height / 2
        top = car.y + car.height / 2
        closest_x = min(max(self.x, left), right)
        closest_y = min(max(self.y, bottom), top)
        return math.hypot(self.x - closest_x, self.y - closest_y)

    def _min_car_dist(self) -> float:
        if not self.cars:
            return self._diag
        return min(self._distance_to_car(car) for car in self.cars)

    def _collision(self) -> bool:
        return any(self._distance_to_car(car) <= self.player_radius for car in self.cars)

    def _obs(self) -> np.ndarray:
        feats = [
            self.x / self.W,
            self.y / self.H,
            self.vx,
            self.vy,
            (self.goal_x - self.x) / self.W,
            (self.goal_cy - self.y) / self.H,
        ]
        sensed = sorted(
            ((self._distance_to_car(car), car) for car in self.cars),
            key=lambda t: t[0],
        )
        slots = 0
        for dist, car in sensed:
            if dist > self.sensor_range or slots >= self.max_sensed:
                break
            feats += [
                (car.x - self.x) / self.sensor_range,
                (car.y - self.y) / self.sensor_range,
                (car.direction * car.speed) / self.max_car_speed,
                1.0 - dist / self.sensor_range,
            ]
            slots += 1
        feats += [0.0, 0.0, 0.0, 0.0] * (self.max_sensed - slots)  # empty = "clear"
        return np.asarray(feats, dtype=np.float32)

    # -- step ---------------------------------------------------------------
    def step(self, action: int):
        self.steps += 1
        reward = self.r_step
        info = {"success": False}

        vx, vy = _VEL[int(action)]
        self.vx, self.vy = float(vx), float(vy)

        for _ in range(self.action_repeat):
            self.x += vx * self.player_speed * self.dt
            self.y += vy * self.player_speed * self.dt
            self._advance_cars(self.dt)

            if self.x < 0 or self.y < 0 or self.y > self.H:
                self.x = min(max(self.x, 0.0), self.W)
                self.y = min(max(self.y, 0.0), self.H)
                reward += self.r_oob
                info["event"] = "ran off the road"
                return self._obs(), reward, True, info

            if self._collision():
                reward += self.r_collision
                info["event"] = "hit by traffic"
                return self._obs(), reward, True, info

            if self.x >= self.goal_x:
                reward += self.r_goal
                info["event"] = "crossed the road"
                info["success"] = True
                return self._obs(), reward, True, info

        dx = self.goal_x - self.x
        reward += self.k_progress * (self._prev_dx - dx)
        self._prev_dx = dx

        nearest = self._min_car_dist()
        if nearest < self.near_radius:
            reward += self.r_near * (1.0 - nearest / self.near_radius)

        return self._obs(), reward, False, info

    # -- rendering ----------------------------------------------------------
    def render_state(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "cars": [
                {
                    "x": car.x,
                    "y": car.y,
                    "direction": car.direction,
                    "speed": car.speed,
                    "width": car.width,
                    "height": car.height,
                    "color": car.color,
                }
                for car in self.cars
            ],
            "sensor_range": self.sensor_range,
            "goal_x": self.goal_x,
            "road_x_min": self.road_x_min,
            "road_x_max": self.road_x_max,
            "lane_xs": list(self.lane_xs),
        }
