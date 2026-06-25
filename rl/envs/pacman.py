"""Room 1 — Pacman (Dynamic Programming, known model).

The agent roams a 10x10 maze, must collect **every** coin, and only then may
leave through the door.  Because the environment model is fully known we can
solve it exactly with Dynamic Programming.

State
-----
``(cell, mask)`` where ``cell`` is ``(row, col)`` and ``mask`` is a bitmask of
the coins **still on the board** (bit *i* set ⇒ coin *i* not yet collected).
The single terminal / final state is standing on the door with ``mask == 0``.

Dynamics
--------
4-connected moves; illegal moves keep the agent in place.  Icy (slippery) cells
randomise the action with probability ``slip_prob`` — these stochastic
transitions are part of the *known* model handed to DP.  While any coin remains
the door is locked: trying to step onto it bumps the agent back and costs
``r_door_early``.

Rewards (defaults match the project spec)
-----------------------------------------
``r_step=-1`` per move, ``r_coin=+10`` per coin, ``r_exit=+100`` for escaping,
``r_door_early=-10`` for trying the locked door, ``r_slip=-5`` when a slip occurs.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from rl.algos.dp import TabularMDP
from rl.envs.grid_base import ACTIONS, GridBase

Cell = Tuple[int, int]

# Default maze.  '#'=wall  '.'=floor  'S'=start  'D'=door  'o'=coin  '~'=ice
DEFAULT_LAYOUT = [
    "S....#...o",
    ".###.#.##.",
    ".#o..#...." ,
    ".#.#.##.#.",
    "...#~~..#.",
    ".#.#~#.#..",
    ".#...#.#.o",
    "..o#.#.#.#",
    "#..#...#..",
    ".#...##..D",
]


class PacmanEnv:
    def __init__(
        self,
        layout: List[str] = None,
        slip_prob: float = 0.15,
        max_steps: int = 200,
        r_step: float = -1.0,
        r_coin: float = 10.0,
        r_exit: float = 100.0,
        r_door_early: float = -10.0,
        r_slip: float = -5.0,
        seed: int = None,
    ) -> None:
        self.layout = layout or DEFAULT_LAYOUT
        self.rows = len(self.layout)
        self.cols = len(self.layout[0])
        self.r_step = r_step
        self.r_coin = r_coin
        self.r_exit = r_exit
        self.r_door_early = r_door_early
        self.r_slip = r_slip
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        walls, coins, slippery = set(), [], set()
        self.start: Cell = (0, 0)
        self.door: Cell = (self.rows - 1, self.cols - 1)
        for r, line in enumerate(self.layout):
            for c, ch in enumerate(line):
                cell = (r, c)
                if ch == "#":
                    walls.add(cell)
                elif ch == "S":
                    self.start = cell
                elif ch == "D":
                    self.door = cell
                elif ch == "o":
                    coins.append(cell)
                elif ch == "~":
                    slippery.add(cell)
        self.walls = walls
        self.coins = coins
        self.slippery = slippery
        self.coin_index: Dict[Cell, int] = {cell: i for i, cell in enumerate(coins)}
        self.full_mask = (1 << len(coins)) - 1
        self.grid = GridBase(self.rows, self.cols, walls=walls,
                             slippery=slippery, slip_prob=slip_prob)

        # episode state
        self.pos: Cell = self.start
        self.mask: int = self.full_mask
        self.steps: int = 0

    # -- helpers ------------------------------------------------------------
    def remaining_coins(self, mask: int) -> List[Cell]:
        return [c for c, i in self.coin_index.items() if (mask >> i) & 1]

    def _apply(self, cell: Cell, mask: int, ncell: Cell, slipped: bool):
        """Resolve door/coins/escape for one realised move. Returns
        ``(ncell, nmask, reward, terminal, info)``."""
        door_bump = False
        if ncell == self.door and mask != 0:
            ncell = cell  # locked door: bump back
            door_bump = True
        escaped = (ncell == self.door and mask == 0)

        nmask = mask
        reward = self.r_step
        if slipped:
            reward += self.r_slip
        if door_bump:
            reward += self.r_door_early
        idx = self.coin_index.get(ncell)
        if idx is not None and (mask >> idx) & 1:
            nmask = mask & ~(1 << idx)
            reward += self.r_coin
        if escaped:
            reward += self.r_exit
        return ncell, nmask, reward, escaped, {"slipped": slipped, "door_bump": door_bump}

    # -- sampling interface (for replay / model-free comparison) -----------
    def reset(self):
        self.pos = self.start
        self.mask = self.full_mask
        self.steps = 0
        return (self.pos, self.mask)

    def step(self, action: int):
        ncell, slipped = self.grid.sample_cell(self.pos, action, self.rng)
        ncell, nmask, reward, escaped, info = self._apply(self.pos, self.mask, ncell, slipped)
        self.pos, self.mask = ncell, nmask
        self.steps += 1
        done = escaped or self.steps >= self.max_steps
        info["escaped"] = escaped
        return (self.pos, self.mask), reward, done, info

    # -- explicit model (for Dynamic Programming) --------------------------
    def build_mdp(self) -> TabularMDP:
        free_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                      if (r, c) not in self.walls]
        n_masks = 1 << len(self.coins)
        states = [(cell, mask) for mask in range(n_masks) for cell in free_cells]
        terminal = (self.door, 0)
        terminals = {terminal}

        P: Dict = {}
        for state in states:
            cell, mask = state
            P[state] = {}
            if state in terminals:
                for a in ACTIONS:
                    P[state][a] = [(1.0, state, 0.0, True)]
                continue
            for a in ACTIONS:
                outs = []
                for prob, ncell, slipped in self.grid.cell_transitions(cell, a):
                    rc, rmask, reward, escaped, _ = self._apply(cell, mask, ncell, slipped)
                    nstate = (rc, rmask)
                    outs.append((prob, nstate, reward, escaped))
                P[state][a] = outs
        return TabularMDP(states=states, actions=ACTIONS, P=P,
                          start=(self.start, self.full_mask), terminals=terminals)

    # -- connectivity sanity check -----------------------------------------
    def reachable_cells(self):
        seen = {self.start}
        stack = [self.start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if (0 <= nb[0] < self.rows and 0 <= nb[1] < self.cols
                        and nb not in self.walls and nb not in seen):
                    seen.add(nb)
                    stack.append(nb)
        return seen
