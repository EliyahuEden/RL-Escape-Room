"""Room 3 — Racing Track (Q-Learning, unknown model).

The car must reach the finish line **F** as fast as possible.  The track is a
10x10 grid presented as a circuit: a long, clean main route (down the right
side) and a short-cut straight down the middle that is littered with **oil**
(slippery, big penalty) and **mud** (slows you down).  **Boosters** give a one-off
bonus.  Driving into the barrier ("leaving the track") is heavily penalised.

Q-Learning is **off-policy**: it evaluates the greedy policy, so it is happy to
learn the fastest line — including the risky short-cut — making a nice contrast
with the cautious SARSA agent of Room 2.

State
-----
``(cell, booster_mask)`` — position plus a bitmask of boosters already taken
(so a booster pays out only once and cannot be farmed).  The terminal/final
state is crossing the finish line.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from rl.envs.grid_base import (ACTIONS, DOWN, GridBase, LEFT, RIGHT, UP,
                               path_length, reachable, shortest_path)

Cell = Tuple[int, int]
_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# '#'=barrier '.'=track 'S'=start 'F'=finish 'B'=booster 'M'=mud 'O'=oil 'X'=crash zone
# A serpentine circuit: the safe racing line snakes the full track (~54 steps),
# while a vertical short-cut straight down column 5 (oil + mud) reaches the finish
# in ~18 steps. Q-Learning chases the short-cut; raise the oil penalty / slip and
# it grows more cautious. Watch both lines with the SARSA-vs-Q race below.
DEFAULT_LAYOUT = [
    "S.B..O...B",
    "#####O###.",
    "..M.......",
    ".####O####",
    "..B.......",
    "#####M###.",
    "...O...B..",
    ".####M####",
    "..M..O....",
    "#########F",
]

RACING_BASE_LAYOUT = [
    "S.........",
    "#####.###.",
    "..........",
    ".####.####",
    "..........",
    "#####.###.",
    "..........",
    ".####.####",
    "..........",
    "#########F",
]


def _put(layout: List[List[str]], cell: Cell, ch: str) -> None:
    layout[cell[0]][cell[1]] = ch


def generate_racing_layout(
    seed: int = 0,
    n_oil: int = 5,
    n_mud: int = 4,
    n_boosters: int = 4,
) -> List[str]:
    """Generate a **fresh random track** every seed.

    Random wall segments carve a different circuit each time; a BFS guarantees
    start→finish is connected. Oil/mud/boosters are then scattered on the track,
    and the narrow passages the walls create are exactly where an oil slip can
    turn into a crash.
    """
    rng = random.Random(seed)
    R = C = 10
    start, finish = (0, 0), (R - 1, C - 1)

    for _attempt in range(400):
        reserved = {start, finish}
        for base in (start, finish):
            for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                reserved.add((base[0] + dr, base[1] + dc))
        walls = set()
        for _ in range(rng.randint(7, 11)):
            length = rng.randint(2, 4)
            if rng.random() < 0.5:
                r, c0 = rng.randint(0, R - 1), rng.randint(0, C - length)
                seg = [(r, c0 + i) for i in range(length)]
            else:
                c, r0 = rng.randint(0, C - 1), rng.randint(0, R - length)
                seg = [(r0 + i, c) for i in range(length)]
            walls.update(cell for cell in seg if cell not in reserved)
        if finish not in reachable(walls, start, R, C):
            continue

        p1 = shortest_path(walls, start, finish, R, C)   # the short-cut
        if not p1:
            continue
        interior = [c for c in p1 if c not in (start, finish)]
        if len(interior) < 5:
            continue
        mid = len(interior) // 2
        span = min(len(interior), max(3, n_oil))
        lo = max(0, mid - span // 2)
        oil_stretch = interior[lo:lo + span]
        oil = set(oil_stretch)
        after = interior[lo + span:]                     # tail of the short-cut

        # a SAFE route avoiding the oil must exist and be meaningfully longer
        safe = shortest_path(walls, start, finish, R, C, extra_blocked=oil)
        if not safe:
            continue
        safe_set = set(safe)
        short_len, safe_len = path_length(p1), path_length(safe)
        if safe_len - short_len < 3:
            continue

        # guarantee every oil cell on the short-cut is crash-risky: it must have a
        # wall/edge neighbour, otherwise drop a crash zone on a free neighbour that
        # lies off both routes (so the safe route stays clear).
        def neighbours(cell):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yield (cell[0] + dr, cell[1] + dc)

        crash = set()
        risky = True
        for o in oil_stretch:
            if any(not (0 <= n[0] < R and 0 <= n[1] < C) or n in walls or n in crash
                   for n in neighbours(o)):
                continue                                  # already wall/edge-flanked
            spot = next((n for n in neighbours(o)
                         if 0 <= n[0] < R and 0 <= n[1] < C and n not in walls
                         and n not in p1 and n not in safe_set and n not in reserved), None)
            if spot is None:
                risky = False
                break
            crash.add(spot)
        if not risky:
            continue

        layout = [["." for _ in range(C)] for _ in range(R)]
        for cell in walls:
            _put(layout, cell, "#")
        for cell in oil:
            _put(layout, cell, "O")
        for cell in crash:
            _put(layout, cell, "X")
        # boost just past the short-cut (reward for surviving the risk)
        boosters = [after[0]] if after else []
        # extra random boosters off both routes
        safe_set = set(safe)
        pool = [(r, c) for r in range(R) for c in range(C)
                if layout[r][c] == "." and (r, c) not in oil and (r, c) not in crash
                and (r, c) not in (start, finish) and (r, c) not in p1 and (r, c) not in safe_set]
        rng.shuffle(pool)
        boosters += pool[:max(0, n_boosters - len(boosters))]
        for cell in boosters:
            _put(layout, cell, "B")
        # mud slows the SAFE route, sharpening the risk/reward trade-off
        mud_cells = [c for c in safe if c not in (start, finish) and layout[c[0]][c[1]] == "."]
        rng.shuffle(mud_cells)
        for cell in mud_cells[:n_mud]:
            _put(layout, cell, "M")
        _put(layout, start, "S")
        _put(layout, finish, "F")
        return ["".join(row) for row in layout]

    return [row[:] for row in DEFAULT_LAYOUT]  # guaranteed-valid fallback


class RacingEnv:
    def __init__(
        self,
        layout: List[str] = None,
        slip_prob: float = 0.3,
        max_steps: int = 200,
        r_step: float = -1.0,
        r_finish: float = 150.0,
        r_boost: float = 20.0,
        r_mud: float = -5.0,
        r_crash: float = -100.0,
        r_offtrack: float = -30.0,
        r_wall: float = -5.0,
        seed: int = None,
    ) -> None:
        self.layout = layout or DEFAULT_LAYOUT
        self.rows = len(self.layout)
        self.cols = len(self.layout[0])
        self.r_step = r_step
        self.r_finish = r_finish
        self.r_boost = r_boost
        self.r_mud = r_mud
        self.r_crash = r_crash
        self.r_offtrack = r_offtrack
        self.r_wall = r_wall
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        walls, oil = set(), set()
        self.mud, boosters = set(), []
        self.crash: set = set()
        self.start: Cell = (0, 0)
        self.finish: Cell = (self.rows - 1, 0)
        for r, line in enumerate(self.layout):
            for c, ch in enumerate(line):
                cell = (r, c)
                if ch == "#":
                    walls.add(cell)
                elif ch == "S":
                    self.start = cell
                elif ch == "F":
                    self.finish = cell
                elif ch == "B":
                    boosters.append(cell)
                elif ch == "M":
                    self.mud.add(cell)
                elif ch == "O":
                    oil.add(cell)
                elif ch == "X":
                    self.crash.add(cell)
        self.walls = walls
        self.oil = oil
        self.boosters = boosters
        self.boost_index: Dict[Cell, int] = {c: i for i, c in enumerate(boosters)}
        self.grid = GridBase(self.rows, self.cols, walls=walls,
                             slippery=oil, slip_prob=slip_prob)
        # the middle column is the risky short-cut; track which cells the car uses
        self.shortcut_col = self.cols // 2
        self.shortcut_cells = {(r, self.shortcut_col) for r in range(self.rows)
                               if (r, self.shortcut_col) not in walls}

        self.n_actions = len(ACTIONS)
        self.pos: Cell = self.start
        self.bmask: int = 0
        self.steps = 0

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

        # CRASH: a slip that throws the car into a wall/edge, or driving into a
        # crash zone — ends the episode.
        if (slipped and hit_wall) or (ncell in self.crash):
            reward += self.r_crash
            if ncell in self.crash:
                self.pos = ncell
            info = {"slipped": slipped, "crash": True, "success": False,
                    "shortcut": self.pos in self.shortcut_cells}
            return (self.pos, self.bmask), reward, True, info

        if hit_wall:  # blocked but kept control: bumped a barrier / nudged the edge
            dr, dc = _DELTA[action]
            target = (self.pos[0] + dr, self.pos[1] + dc)
            reward += self.r_offtrack if not self.grid.in_bounds(target) else self.r_wall
        else:
            if ncell in self.mud:
                reward += self.r_mud
            idx = self.boost_index.get(ncell)
            if idx is not None and not (self.bmask >> idx) & 1:
                reward += self.r_boost
                self.bmask |= (1 << idx)

        finished = ncell == self.finish
        if finished:
            reward += self.r_finish

        self.pos = ncell
        done = finished or self.steps >= self.max_steps
        info = {"slipped": slipped, "crash": False, "success": finished,
                "shortcut": ncell in self.shortcut_cells}
        return (self.pos, self.bmask), reward, done, info

    def render_state(self) -> dict:
        return {"pos": self.pos, "bmask": self.bmask}
