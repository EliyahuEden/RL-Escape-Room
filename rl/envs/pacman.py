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


PACMAN_BASE_LAYOUT = [
    "S....#....",
    ".###.#.##.",
    ".#...#....",
    ".#.#.##.#.",
    "...#....#.",
    ".#.#.#.#..",
    ".#...#.#..",
    "...#.#.#.#",
    "#..#...#..",
    ".#...##..D",
]


def _put(layout: List[List[str]], cell: Cell, ch: str) -> None:
    layout[cell[0]][cell[1]] = ch


def generate_pacman_layout(
    seed: int = 0,
    n_coins: int = 4,
    n_slippery: int = 5,
    guard_enabled: bool = False,
) -> List[str]:
    """Generate a Pacman map by reusing the maze walls and randomising items."""
    rng = random.Random(seed)
    layout = [list(row) for row in PACMAN_BASE_LAYOUT]
    free = [
        (r, c)
        for r, row in enumerate(layout)
        for c, ch in enumerate(row)
        if ch == "."
    ]
    start = (0, 0)
    door = (9, 9)
    safe = {start, door}
    coin_pool = [c for c in free if abs(c[0] - start[0]) + abs(c[1] - start[1]) > 3]
    coins = rng.sample(coin_pool, k=min(n_coins, len(coin_pool)))
    safe.update(coins)
    for cell in coins:
        _put(layout, cell, "o")

    slip_pool = [c for c in free if c not in safe]
    for cell in rng.sample(slip_pool, k=min(n_slippery, len(slip_pool))):
        _put(layout, cell, "~")

    if guard_enabled:
        guard_pool = [
            c for c in free
            if c not in safe and abs(c[0] - start[0]) + abs(c[1] - start[1]) >= 8
        ] or [c for c in free if c not in safe]
        if guard_pool:
            _put(layout, rng.choice(guard_pool), "A")

    return ["".join(row) for row in layout]


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
        guard_enabled: bool = False,
        guard_start: Cell = None,
        guard_speed: int = 1,
        r_guard: float = -75.0,
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
        self.guard_enabled = bool(guard_enabled)
        self.guard_speed = max(1, int(guard_speed))
        self.r_guard = r_guard
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        walls, coins, slippery = set(), [], set()
        self.start: Cell = (0, 0)
        self.door: Cell = (self.rows - 1, self.cols - 1)
        self.guard_start: Cell = guard_start or (self.rows - 2, self.cols - 2)
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
                elif ch == "A":
                    self.guard_start = cell
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
        self.guard_pos: Cell = self.guard_start
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

    def _move_guard_once(self, guard: Cell, target: Cell) -> Cell:
        candidates = [guard]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (guard[0] + dr, guard[1] + dc)
            if self.grid.in_bounds(nb) and nb not in self.walls and nb != self.door:
                candidates.append(nb)
        return min(
            candidates,
            key=lambda c: (abs(c[0] - target[0]) + abs(c[1] - target[1]), c[0], c[1]),
        )

    def _move_guard(self, guard: Cell, target: Cell) -> Cell:
        for _ in range(self.guard_speed):
            guard = self._move_guard_once(guard, target)
            if guard == target:
                break
        return guard

    def _state(self):
        if self.guard_enabled:
            return (self.pos, self.mask, self.guard_pos)
        return (self.pos, self.mask)

    def _apply_guard(self, cell: Cell, mask: int, guard: Cell, ncell: Cell, slipped: bool):
        ncell, nmask, reward, escaped, info = self._apply(cell, mask, ncell, slipped)
        caught = False
        nguard = guard
        if self.guard_enabled and not escaped:
            if ncell == guard:
                caught = True
            else:
                nguard = self._move_guard(guard, ncell)
                caught = nguard == ncell
            if caught:
                reward += self.r_guard
        info["caught"] = caught
        return ncell, nmask, nguard, reward, escaped, caught, info

    # -- sampling interface (for replay / model-free comparison) -----------
    def reset(self):
        self.pos = self.start
        self.mask = self.full_mask
        self.guard_pos = self.guard_start
        self.steps = 0
        return self._state()

    def step(self, action: int):
        ncell, slipped = self.grid.sample_cell(self.pos, action, self.rng)
        ncell, nmask, nguard, reward, escaped, caught, info = self._apply_guard(
            self.pos, self.mask, self.guard_pos, ncell, slipped
        )
        self.pos, self.mask, self.guard_pos = ncell, nmask, nguard
        self.steps += 1
        done = escaped or caught or self.steps >= self.max_steps
        info["escaped"] = escaped
        return self._state(), reward, done, info

    # -- explicit model (for Dynamic Programming) --------------------------
    def build_mdp(self) -> TabularMDP:
        free_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                      if (r, c) not in self.walls]
        n_masks = 1 << len(self.coins)
        if self.guard_enabled:
            states = [
                (cell, mask, guard)
                for mask in range(n_masks)
                for cell in free_cells
                for guard in free_cells
            ]
            terminals = {
                s for s in states
                if (s[0] == self.door and s[1] == 0) or s[0] == s[2]
            }
            start = (self.start, self.full_mask, self.guard_start)
        else:
            states = [(cell, mask) for mask in range(n_masks) for cell in free_cells]
            terminal = (self.door, 0)
            terminals = {terminal}
            start = (self.start, self.full_mask)

        P: Dict = {}
        for state in states:
            if self.guard_enabled:
                cell, mask, guard = state
            else:
                cell, mask = state
                guard = self.guard_start
            P[state] = {}
            if state in terminals:
                for a in ACTIONS:
                    P[state][a] = [(1.0, state, 0.0, True)]
                continue
            for a in ACTIONS:
                outs = []
                for prob, ncell, slipped in self.grid.cell_transitions(cell, a):
                    rc, rmask, rguard, reward, escaped, caught, _ = self._apply_guard(
                        cell, mask, guard, ncell, slipped
                    )
                    nstate = (rc, rmask, rguard) if self.guard_enabled else (rc, rmask)
                    outs.append((prob, nstate, reward, escaped or caught))
                P[state][a] = outs
        return TabularMDP(states=states, actions=ACTIONS, P=P,
                          start=start, terminals=terminals)

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
