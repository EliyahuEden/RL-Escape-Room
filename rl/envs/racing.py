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

from rl.envs.grid_base import ACTIONS, GridBase

Cell = Tuple[int, int]

# '#'=barrier '.'=track 'S'=start 'F'=finish 'B'=booster 'M'=mud 'O'=oil(ice)
# A serpentine circuit: the safe racing line snakes the full track (~54 steps),
# while a vertical short-cut straight down column 5 (oil + mud) reaches the finish
# in ~18 steps. Q-Learning chases the short-cut; raise the oil penalty / slip and
# it grows more cautious. Watch both lines with the SARSA-vs-Q race below.
DEFAULT_LAYOUT = [
    "SBB.......",
    "#####O###.",
    "..........",
    ".####O####",
    "..........",
    "#####M###.",
    "..........",
    ".####M####",
    "..........",
    "#########F",
]


class RacingEnv:
    def __init__(
        self,
        layout: List[str] = None,
        slip_prob: float = 0.2,
        max_steps: int = 200,
        r_step: float = -1.0,
        r_finish: float = 100.0,
        r_boost: float = 15.0,
        r_mud: float = -10.0,
        r_oil: float = -20.0,
        r_offtrack: float = -30.0,
        seed: int = None,
    ) -> None:
        self.layout = layout or DEFAULT_LAYOUT
        self.rows = len(self.layout)
        self.cols = len(self.layout[0])
        self.r_step = r_step
        self.r_finish = r_finish
        self.r_boost = r_boost
        self.r_mud = r_mud
        self.r_oil = r_oil
        self.r_offtrack = r_offtrack
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        walls, oil = set(), set()
        self.mud, boosters = set(), []
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
        self.walls = walls
        self.oil = oil
        self.boosters = boosters
        self.boost_index: Dict[Cell, int] = {c: i for i, c in enumerate(boosters)}
        self.grid = GridBase(self.rows, self.cols, walls=walls,
                             slippery=oil, slip_prob=slip_prob)

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
        ncell, slipped = self.grid.sample_cell(self.pos, action, self.rng)
        reward = self.r_step
        offtrack = ncell == self.pos  # a blocked move = crash into the barrier
        if offtrack:
            reward += self.r_offtrack
        if slipped:
            reward += self.r_oil
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
        self.steps += 1
        done = finished or self.steps >= self.max_steps
        info = {"slipped": slipped, "offtrack": offtrack, "success": finished}
        return (self.pos, self.bmask), reward, done, info

    def render_state(self) -> dict:
        return {"pos": self.pos, "bmask": self.bmask}
