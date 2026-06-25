"""Room 2 — Museum Heist (SARSA, unknown model).

A thief must grab the **diamond** and reach the **exit** while avoiding security
**cameras** and **traps**.  The model is unknown, so the agent learns from
experience.

The room is laid out as a *cliff-walking* problem to expose the difference
between on-policy and off-policy control:

* The bottom row is a line of **cameras** (the "cliff").  Walking into one trips
  the alarm: a big penalty and the thief is dragged back to the start.
* The row just above the cameras (row 8) is the **shortest** corridor from the
  diamond to the exit — the tempting *edge* path.
* A safer detour one row higher is only a couple of steps longer.

Because SARSA evaluates the policy it actually follows (ε-greedy), it learns to
keep a safety margin from the cliff; Q-Learning (Room 3) learns to hug the edge.

State
-----
``(cell, has_diamond)`` with ``has_diamond`` ∈ {0, 1}; 200 states in total.
The terminal/final state is reaching the exit **with** the diamond.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from rl.envs.grid_base import ACTIONS, GridBase

Cell = Tuple[int, int]

# '#'=wall '.'=floor 'S'=start 'X'=exit 'G'=diamond 'C'=camera 'T'=trap '~'=ice

# Easy: open hall with a camera cliff along the bottom (the original layout).
EASY_LAYOUT = [
    "..........",
    "..........",
    "..~~~.....",
    "..~~~.....",
    "....T.....",
    "..........",
    "..........",
    "..........",
    "G.........",
    "SCCCCCCCCX",
]

# Hard: a comb-maze museum. The diamond sits on the left edge; the *short* way
# out runs east along the icy edge straight into camera cells, while the only
# camera-free escape is a long detour up through the maze and down the far side
# (≈27 steps vs ≈11). SARSA learns the safe detour; Q-Learning would hug the edge.
HARD_LAYOUT = [
    "..........",
    ".#.#.#.#..",
    ".#.#.#.#..",
    ".#.#.#.#..",
    ".#T#.#.#..",
    ".#.#.#.#..",
    ".#.#.#.#..",
    ".#.#.#.#..",
    "G~~~~~~CC.",
    "SCCCCCCCCX",
]

LAYOUTS = {"Museum maze (hard)": HARD_LAYOUT, "Open hall (easy)": EASY_LAYOUT}
DEFAULT_LAYOUT = HARD_LAYOUT


class MuseumEnv:
    def __init__(
        self,
        layout: List[str] = None,
        slip_prob: float = 0.05,
        max_steps: int = 200,
        r_step: float = -1.0,
        r_diamond: float = 30.0,
        r_exit: float = 100.0,
        r_camera: float = -25.0,
        r_trap: float = -20.0,
        r_slip: float = -5.0,
        r_exit_early: float = -5.0,
        seed: int = None,
    ) -> None:
        self.layout = layout or DEFAULT_LAYOUT
        self.rows = len(self.layout)
        self.cols = len(self.layout[0])
        self.r_step = r_step
        self.r_diamond = r_diamond
        self.r_exit = r_exit
        self.r_camera = r_camera
        self.r_trap = r_trap
        self.r_slip = r_slip
        self.r_exit_early = r_exit_early
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        walls, slippery = set(), set()
        self.cameras, self.traps = set(), set()
        self.start: Cell = (self.rows - 1, 0)
        self.exit: Cell = (self.rows - 1, self.cols - 1)
        self.diamond: Cell = (0, 0)
        for r, line in enumerate(self.layout):
            for c, ch in enumerate(line):
                cell = (r, c)
                if ch == "#":
                    walls.add(cell)
                elif ch == "S":
                    self.start = cell
                elif ch == "X":
                    self.exit = cell
                elif ch == "G":
                    self.diamond = cell
                elif ch == "C":
                    self.cameras.add(cell)
                elif ch == "T":
                    self.traps.add(cell)
                elif ch == "~":
                    slippery.add(cell)
        self.walls = walls
        self.slippery = slippery
        self.grid = GridBase(self.rows, self.cols, walls=walls,
                             slippery=slippery, slip_prob=slip_prob)

        self.n_actions = len(ACTIONS)
        self.pos: Cell = self.start
        self.has_diamond: int = 0
        self.steps = 0

    # -- core dynamics ------------------------------------------------------
    def reset(self):
        self.pos = self.start
        self.has_diamond = 0
        self.steps = 0
        return (self.pos, self.has_diamond)

    def step(self, action: int):
        ncell, slipped = self.grid.sample_cell(self.pos, action, self.rng)
        reward = self.r_step
        if slipped:
            reward += self.r_slip
        caught = False
        success = False

        if ncell in self.cameras:           # alarm: penalty + dragged to start
            reward += self.r_camera
            ncell = self.start
            caught = True
        elif ncell in self.traps:
            reward += self.r_trap
        elif ncell == self.diamond and not self.has_diamond:
            reward += self.r_diamond
            self.has_diamond = 1
        elif ncell == self.exit:
            if self.has_diamond:
                reward += self.r_exit
                success = True
            else:                            # door locked without the loot
                reward += self.r_exit_early
                ncell = self.pos

        self.pos = ncell
        self.steps += 1
        done = success or self.steps >= self.max_steps
        info = {"slipped": slipped, "caught": caught, "success": success}
        return (self.pos, self.has_diamond), reward, done, info

    # -- rendering helper ---------------------------------------------------
    def render_state(self) -> dict:
        return {"pos": self.pos, "has_diamond": self.has_diamond}
