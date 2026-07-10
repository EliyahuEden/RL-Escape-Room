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
#   'A'=guard start.  Designed so there are TWO independent routes from start to
#   door (a verified cycle) and the exit has two open approaches — the guard can
#   no longer trap the agent by camping the single doorway, and the agent can
#   choose its lane. The guard starts DEAD CENTRE so a chasing guard pressures
#   the agent from the very first move.
DEFAULT_LAYOUT = [
    "S.o#######",
    "....##..o#",
    ".##.###..#",
    "..#.~##.##",
    "..~#A.#.##",
    "#..##.~...",
    "..o.#####.",
    "##.~#####.",
    "##....o...",
    "######...D",
]


# Same wall skeleton as DEFAULT_LAYOUT (items stripped) — the generator reuses
# these corridors and scatters coins / ice / guard on top.
PACMAN_BASE_LAYOUT = [
    "S..#######",
    "....##...#",
    ".##.###..#",
    "..#..##.##",
    "...#..#.##",
    "#..##.....",
    "....#####.",
    "##..#####.",
    "##........",
    "######...D",
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
        # start the guard near the middle of the maze so a chasing guard is a
        # threat from the first move (not parked next to the exit)
        central = [c for c in free
                   if c not in safe and 3 <= c[0] <= 6 and 3 <= c[1] <= 6]
        guard_pool = central or [c for c in free if c not in safe]
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
        r_wall: float = -5.0,
        guard_enabled: bool = False,
        guard_start: Cell = None,
        guard_speed: int = 1,
        guard_mode: str = "chase",
        r_guard: float = -50.0,
        patrol_len: int = 5,
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
        self.r_wall = r_wall
        self.guard_enabled = bool(guard_enabled)
        self.guard_speed = max(1, int(guard_speed))
        self.guard_mode = guard_mode if guard_mode in ("chase", "patrol") else "chase"
        self.r_guard = r_guard
        self.patrol_len = max(2, int(patrol_len))
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

        # Fixed, fully-known patrol route (a back-and-forth corridor walk). The
        # guard's position is a deterministic function of its patrol step, so the
        # whole thing is compatible with Dynamic Programming.
        self.guard_route = (self._patrol_route(self.guard_start)
                            if self.guard_enabled else [self.guard_start])
        self.guard_period = len(self.guard_route)

        # episode state
        self.pos: Cell = self.start
        self.mask: int = self.full_mask
        self.guard_phase: int = 0
        self.guard_pos: Cell = self.guard_route[0]
        self.steps: int = 0

    # -- helpers ------------------------------------------------------------
    def remaining_coins(self, mask: int) -> List[Cell]:
        return [c for c, i in self.coin_index.items() if (mask >> i) & 1]

    def _apply(self, cell: Cell, mask: int, ncell: Cell, slipped: bool, hit_wall: bool):
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
        if hit_wall:
            reward += self.r_wall
        if door_bump:
            reward += self.r_door_early
        idx = self.coin_index.get(ncell)
        if idx is not None and (mask >> idx) & 1:
            nmask = mask & ~(1 << idx)
            reward += self.r_coin
        if escaped:
            reward += self.r_exit
        return ncell, nmask, reward, escaped, {"slipped": slipped, "door_bump": door_bump,
                                               "hit_wall": hit_wall}

    # -- fixed patrol guard -------------------------------------------------
    def _patrol_route(self, start: Cell) -> List[Cell]:
        """A deterministic back-and-forth corridor patrol through ``start``.

        The route is fully known and never depends on the agent, so it stays
        compatible with Dynamic Programming.
        """
        def free(c: Cell) -> bool:
            return (self.grid.in_bounds(c) and c not in self.walls
                    and c != self.door and c not in self.coin_index and c != self.start)

        row = [start]
        c = start[1] - 1
        while free((start[0], c)):
            row.insert(0, (start[0], c)); c -= 1
        c = start[1] + 1
        while free((start[0], c)):
            row.append((start[0], c)); c += 1
        col = [start]
        r = start[0] - 1
        while free((r, start[1])):
            col.insert(0, (r, start[1])); r -= 1
        r = start[0] + 1
        while free((r, start[1])):
            col.append((r, start[1])); r += 1

        seg = row if len(row) >= len(col) else col
        seg = seg[:max(2, self.patrol_len)]
        if len(seg) <= 1:
            return [start]
        # ping-pong so the guard oscillates along the corridor
        return seg + seg[-2:0:-1]

    def guard_at(self, phase: int) -> Cell:
        return self.guard_route[phase % self.guard_period]

    def _next_phase(self, phase: int) -> int:
        return (phase + self.guard_speed) % self.guard_period

    def _resolve_guard(self, cell: Cell, ncell: Cell, phase: int, escaped: bool):
        """Advance the patrol one step and test for a catch.

        Caught if the agent lands on the guard's new cell, or if the two swap
        cells (cross past each other). Returns ``(next_phase, caught)``.
        """
        nphase = self._next_phase(phase)
        if escaped:
            return nphase, False
        g_old = self.guard_at(phase)
        g_new = self.guard_at(nphase)
        caught = (ncell == g_new) or (ncell == g_old and g_new == cell)
        return nphase, caught

    # -- reactive (chasing) guard -------------------------------------------
    def _chase_once(self, guard: Cell, target: Cell) -> Cell:
        """One deterministic greedy step that minimises distance to the player."""
        candidates = [guard]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (guard[0] + dr, guard[1] + dc)
            if self.grid.in_bounds(nb) and nb not in self.walls and nb != self.door:
                candidates.append(nb)
        return min(candidates,
                   key=lambda c: (abs(c[0] - target[0]) + abs(c[1] - target[1]), c[0], c[1]))

    def _chase(self, guard: Cell, target: Cell) -> Cell:
        for _ in range(self.guard_speed):
            guard = self._chase_once(guard, target)
            if guard == target:
                break
        return guard

    def _resolve_chase(self, cell: Cell, ncell: Cell, guard: Cell, escaped: bool):
        """Move the chasing guard toward the player's new cell; test for a catch.

        Deterministic (a fixed function of the state) → still DP-compatible, with
        the guard's *position* carried in the state. Returns ``(next_guard, caught)``.
        """
        if escaped:
            return guard, False
        if ncell == guard:                       # walked straight into the guard
            return guard, True
        nguard = self._chase(guard, ncell)
        caught = (nguard == ncell) or (nguard == cell and guard == ncell)
        return nguard, caught

    def _state(self):
        if not self.guard_enabled:
            return (self.pos, self.mask)
        if self.guard_mode == "chase":
            return (self.pos, self.mask, self.guard_pos)
        return (self.pos, self.mask, self.guard_phase)

    # -- sampling interface (for replay / model-free comparison) -----------
    def reset(self):
        self.pos = self.start
        self.mask = self.full_mask
        self.guard_phase = 0
        self.guard_pos = self.guard_start if self.guard_mode == "chase" else self.guard_at(0)
        self.steps = 0
        return self._state()

    def step(self, action: int):
        ncell, slipped, hit_wall = self.grid.sample_cell(self.pos, action, self.rng)
        ncell, nmask, reward, escaped, info = self._apply(
            self.pos, self.mask, ncell, slipped, hit_wall
        )
        caught = False
        if self.guard_enabled:
            if self.guard_mode == "chase":
                self.guard_pos, caught = self._resolve_chase(
                    self.pos, ncell, self.guard_pos, escaped)
            else:
                self.guard_phase, caught = self._resolve_guard(
                    self.pos, ncell, self.guard_phase, escaped)
                self.guard_pos = self.guard_at(self.guard_phase)
            if caught:
                reward += self.r_guard
        info["caught"] = caught
        info["escaped"] = escaped
        self.pos, self.mask = ncell, nmask
        self.steps += 1
        done = escaped or caught or self.steps >= self.max_steps
        return self._state(), reward, done, info

    # -- explicit model (for Dynamic Programming) --------------------------
    def build_mdp(self) -> TabularMDP:
        free_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                      if (r, c) not in self.walls]
        n_masks = 1 << len(self.coins)

        if not self.guard_enabled:
            states = [(cell, mask) for mask in range(n_masks) for cell in free_cells]
            terminals = {(self.door, 0)}
            start = (self.start, self.full_mask)
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
                    for prob, ncell, slipped, hit_wall in self.grid.cell_transitions(cell, a):
                        rc, rmask, reward, escaped, _ = self._apply(
                            cell, mask, ncell, slipped, hit_wall)
                        outs.append((prob, (rc, rmask), reward, escaped))
                    P[state][a] = outs
            return TabularMDP(states=states, actions=ACTIONS, P=P,
                              start=start, terminals=terminals)

        # --- guard enabled: state = (cell, mask, guard_rep) ----------------
        # guard_rep is the guard's CELL in chase mode, or its patrol PHASE in
        # patrol mode; both are deterministic functions of the state, so DP works.
        CAUGHT = "CAUGHT"  # single absorbing "game over" state
        if self.guard_mode == "chase":
            guard_reps = free_cells
            start_g = self.guard_start
            resolve = self._resolve_chase
        else:
            guard_reps = list(range(self.guard_period))
            start_g = 0
            resolve = self._resolve_guard

        states = [(cell, mask, g) for mask in range(n_masks)
                  for cell in free_cells for g in guard_reps]
        states.append(CAUGHT)
        terminals = {s for s in states
                     if s != CAUGHT and s[0] == self.door and s[1] == 0}
        terminals.add(CAUGHT)
        start = (self.start, self.full_mask, start_g)

        P = {}
        for state in states:
            P[state] = {}
            if state in terminals:
                for a in ACTIONS:
                    P[state][a] = [(1.0, state, 0.0, True)]
                continue
            cell, mask, g = state
            for a in ACTIONS:
                outs = []
                for prob, ncell, slipped, hit_wall in self.grid.cell_transitions(cell, a):
                    rc, rmask, reward, escaped, _ = self._apply(
                        cell, mask, ncell, slipped, hit_wall)
                    ng, caught = resolve(cell, rc, g, escaped)
                    if caught:
                        outs.append((prob, CAUGHT, reward + self.r_guard, True))
                    else:
                        outs.append((prob, (rc, rmask, ng), reward, escaped))
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
