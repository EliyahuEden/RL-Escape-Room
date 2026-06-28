"""Room 3 - Street Race (Q-Learning, unknown model).

A car races through a city street circuit.  The finish line is **locked** until
the car collects enough **boosters** scattered around the track — just like
Room 1's Pacman must collect all coins before the door opens.

The streets wind through city blocks and are littered with hazards:

* **Oil spills** — slippery; deflect movement sideways with probability
  ``slip_prob``, potentially into barriers.
* **Mud patches** — slow the car (step penalty).
* **Crash barriers** — driving into one ends the race (terminal).
* **Boosters** — one-time speed pads.  Collect at least ``min_boosters``
  to unlock the finish line.

The model is unknown — Q-Learning learns the track from sampled experience.

State
-----
``(cell, booster_mask)`` — position plus a bitmask of boosters already
collected.  The final state is crossing the (unlocked) finish line.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

from rl.envs.grid_base import ACTIONS, DOWN, GridBase, LEFT, RIGHT, UP, path_length, reachable, shortest_path

Cell = Tuple[int, int]
_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# '#'=barrier '.'=track 'S'=start 'F'=finish 'B'=booster
# 'O'=oil 'M'=mud 'X'=crash barrier

# ── Street layout (walls only) ──────────────────────────────────────────────
#
#   City blocks form a grid of streets.  Items are placed on top.
#
TRACK_BASE_LAYOUT = [
    "S.........",   # 0: starting straight
    ".##.##.##.",   # 1: city blocks
    "..........",   # 2: cross street
    ".##.##.##.",   # 3: city blocks
    "..........",   # 4: main boulevard
    ".##.##.##.",   # 5: city blocks
    "..........",   # 6: cross street
    ".##.##.##.",   # 7: city blocks
    "..........",   # 8: final straight
    ".##.##..#F",   # 9: finish
]

# ── Default layout (hand-placed items) ──────────────────────────────────────
DEFAULT_LAYOUT = [
    "S.B.......",   # 0: start, booster
    ".##.##.##.",   # 1: city blocks
    "..OO..B...",   # 2: oil patch, booster
    ".##.##.##.",   # 3: city blocks
    "..M...X...",   # 4: mud, crash barrier
    ".##.##.##.",   # 5: city blocks
    "..B..OO...",   # 6: booster, oil
    ".##.##.##.",   # 7: city blocks
    "..M.....B.",   # 8: mud, booster
    ".##.##..#F",   # 9: finish
]


def _put(layout: List[List[str]], cell: Cell, ch: str) -> None:
    layout[cell[0]][cell[1]] = ch


def _parse_layout(layout: List[str]) -> Dict:
    rows, cols = len(layout), len(layout[0])
    walls: Set[Cell] = set()
    oil: Set[Cell] = set()
    mud: Set[Cell] = set()
    crash: Set[Cell] = set()
    shortcut: Set[Cell] = set()
    boosters: List[Cell] = []
    start: Cell = (0, 0)
    finish: Cell = (rows - 1, cols - 1)

    for r, line in enumerate(layout):
        for c, ch in enumerate(line):
            cell = (r, c)
            if ch == "#":
                walls.add(cell)
            elif ch == "S":
                start = cell
            elif ch == "F":
                finish = cell
            elif ch == "B":
                boosters.append(cell)
            elif ch == "M":
                mud.add(cell)
            elif ch == "O":
                oil.add(cell)
                shortcut.add(cell)
            elif ch == "R":
                shortcut.add(cell)
            elif ch == "X":
                crash.add(cell)

    return {
        "rows": rows,
        "cols": cols,
        "walls": walls,
        "oil": oil,
        "mud": mud,
        "crash": crash,
        "shortcut": shortcut,
        "boosters": boosters,
        "start": start,
        "finish": finish,
    }


def racing_layout_stats(layout: List[str]) -> Dict[str, object]:
    parsed = _parse_layout(layout)
    rows, cols = parsed["rows"], parsed["cols"]
    walls = parsed["walls"]
    oil = parsed["oil"]
    crash = parsed["crash"]
    start, finish = parsed["start"], parsed["finish"]

    short = shortest_path(walls | crash, start, finish, rows, cols)
    safe = shortest_path(walls, start, finish, rows, cols, extra_blocked=oil | crash)
    short_len = path_length(short)
    safe_len = path_length(safe)
    return {
        "short_len": short_len,
        "safe_len": safe_len,
        "safe_gap": None if short_len is None or safe_len is None else safe_len - short_len,
        "oil": len(oil),
        "mud": len(parsed["mud"]),
        "boosters": len(parsed["boosters"]),
        "crash": len(crash),
        "shortcut_cells": len(parsed["shortcut"]),
    }


def generate_racing_layout(
    seed: int = 0,
    n_oil: int = 5,
    n_mud: int = 3,
    n_boosters: int = 4,
    n_crash: int = 2,
) -> List[str]:
    """Generate a street race by placing items on the fixed city layout."""
    rng = random.Random(seed)
    R = C = 10
    start = (0, 0)
    finish = (R - 1, C - 1)

    base = _parse_layout(TRACK_BASE_LAYOUT)
    walls = base["walls"]
    layout = [list(row) for row in TRACK_BASE_LAYOUT]

    free = [
        (r, c) for r in range(R) for c in range(C)
        if (r, c) not in walls and (r, c) != start and (r, c) != finish
    ]

    path = shortest_path(walls, start, finish, R, C)
    on_path = set(path or []) - {start, finish}

    off_path = [c for c in free if c not in on_path]
    rng.shuffle(off_path)

    for cell in off_path[:n_oil]:
        _put(layout, cell, "O")

    for cell in off_path[n_oil:n_oil + n_crash]:
        _put(layout, cell, "X")

    booster_pool = off_path[n_oil + n_crash:]
    rng.shuffle(booster_pool)
    for cell in booster_pool[:n_boosters]:
        _put(layout, cell, "B")

    mud_candidates = [c for c in list(on_path) if layout[c[0]][c[1]] == "."]
    rng.shuffle(mud_candidates)
    for cell in mud_candidates[:n_mud]:
        _put(layout, cell, "M")

    return ["".join(row) for row in layout]


class RacingEnv:
    def __init__(
        self,
        layout: List[str] = None,
        slip_prob: float = 0.2,
        max_steps: int = 200,
        min_boosters: int = 3,
        r_step: float = -1.0,
        r_finish: float = 200.0,
        r_boost: float = 20.0,
        r_mud: float = -5.0,
        r_crash: float = -200.0,
        r_offtrack: float = -30.0,
        r_wall: float = -5.0,
        r_finish_locked: float = -10.0,
        seed: int = None,
    ) -> None:
        self.layout = layout or DEFAULT_LAYOUT
        parsed = _parse_layout(self.layout)
        self.rows = parsed["rows"]
        self.cols = parsed["cols"]
        self.r_step = r_step
        self.r_finish = r_finish
        self.r_boost = r_boost
        self.r_mud = r_mud
        self.r_crash = r_crash
        self.r_offtrack = r_offtrack
        self.r_wall = r_wall
        self.r_finish_locked = r_finish_locked
        self.min_boosters = min_boosters
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.walls = parsed["walls"]
        self.oil = parsed["oil"]
        self.mud = parsed["mud"]
        self.crash = parsed["crash"]
        self.shortcut_cells = parsed["shortcut"]
        self.boosters = parsed["boosters"]
        self.start = parsed["start"]
        self.finish = parsed["finish"]
        self.boost_index: Dict[Cell, int] = {c: i for i, c in enumerate(self.boosters)}
        self.grid = GridBase(self.rows, self.cols, walls=self.walls, slippery=self.oil, slip_prob=slip_prob)

        self.n_actions = len(ACTIONS)
        self.pos: Cell = self.start
        self.bmask: int = 0
        self.steps = 0

    def collected_count(self, bmask: int) -> int:
        return bin(bmask).count("1")

    def finish_unlocked(self, bmask: int) -> bool:
        return self.collected_count(bmask) >= self.min_boosters

    def remaining_boosters(self, bmask: int) -> List[Cell]:
        return [c for c, i in self.boost_index.items() if not (bmask >> i) & 1]

    def reset(self):
        self.pos = self.start
        self.bmask = 0
        self.steps = 0
        return (self.pos, self.bmask)

    def step(self, action: int):
        ncell, slipped, hit_wall = self.grid.sample_cell(self.pos, action, self.rng)
        reward = self.r_step
        self.steps += 1

        if (slipped and hit_wall) or (ncell in self.crash):
            reward += self.r_crash
            if ncell in self.crash:
                self.pos = ncell
            info = {
                "slipped": slipped,
                "crash": True,
                "success": False,
                "shortcut": self.pos in self.shortcut_cells,
            }
            return (self.pos, self.bmask), reward, True, info

        if hit_wall:
            dr, dc = _DELTA[action]
            target = (self.pos[0] + dr, self.pos[1] + dc)
            reward += self.r_offtrack if not self.grid.in_bounds(target) else self.r_wall
        else:
            if ncell in self.mud:
                reward += self.r_mud
            idx = self.boost_index.get(ncell)
            if idx is not None and not (self.bmask >> idx) & 1:
                reward += self.r_boost
                self.bmask |= 1 << idx

        finished = False
        if ncell == self.finish:
            if self.finish_unlocked(self.bmask):
                reward += self.r_finish
                finished = True
            else:
                reward += self.r_finish_locked
                ncell = self.pos

        self.pos = ncell
        done = finished or self.steps >= self.max_steps
        info = {
            "slipped": slipped,
            "crash": False,
            "success": finished,
            "shortcut": ncell in self.shortcut_cells,
        }
        return (self.pos, self.bmask), reward, done, info

    def render_state(self) -> dict:
        return {
            "pos": self.pos,
            "bmask": self.bmask,
            "finish_unlocked": self.finish_unlocked(self.bmask),
        }
