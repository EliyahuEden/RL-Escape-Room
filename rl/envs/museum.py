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

from rl.envs.grid_base import ACTIONS, GridBase, path_length, reachable, shortest_path

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

MUSEUM_BASE_LAYOUT = [
    "..........",
    ".#.#.#.#..",
    ".#.#.#.#..",
    ".#.#.#.#..",
    ".#.#.#.#..",
    ".#.#.#.#..",
    ".#.#.#.#..",
    ".#.#.#.#..",
    "G.........",
    "S........X",
]


def _put(layout: List[List[str]], cell: Cell, ch: str) -> None:
    layout[cell[0]][cell[1]] = ch


def museum_guard_route(start: Cell, walls, exit_cell: Cell, rows: int, cols: int) -> List[Cell]:
    """Fixed back-and-forth patrol along the longest free row/column run through
    ``start`` (shared by the env and the layout generator)."""
    def free(cell: Cell) -> bool:
        return (0 <= cell[0] < rows and 0 <= cell[1] < cols
                and cell not in walls and cell != exit_cell)

    row = [(start[0], c) for c in range(start[1] - 2, start[1] + 3) if free((start[0], c))]
    col = [(r, start[1]) for r in range(start[0] - 2, start[0] + 3) if free((r, start[1]))]
    route = sorted(set(row if len(row) >= len(col) else col))
    if len(route) <= 1:
        return [start]
    return route + route[-2:0:-1]


def generate_museum_layout(
    seed: int = 0,
    n_cameras: int = 9,
    n_traps: int = 2,
    n_slippery: int = 6,
    n_guards: int = 2,
) -> List[str]:
    """Generate a **fresh random museum** every seed.

    Random wall *segments* form different galleries each time; a BFS guarantees
    the start, diamond and exit stay connected before any hazards (which are all
    passable) are scattered on top.
    """
    rng = random.Random(seed)
    R = C = 10
    start, exit_cell = (R - 1, 0), (R - 1, C - 1)

    for _attempt in range(400):
        diamond = (rng.randint(0, 5), rng.randint(1, C - 1))
        if diamond in (start, exit_cell):
            continue
        reserved = {start, exit_cell, diamond}
        for base in (start, exit_cell, diamond):
            for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                reserved.add((base[0] + dr, base[1] + dc))
        walls = set()
        for _ in range(rng.randint(6, 10)):
            length = rng.randint(2, 4)
            if rng.random() < 0.5:
                r, c0 = rng.randint(0, R - 1), rng.randint(0, C - length)
                seg = [(r, c0 + i) for i in range(length)]
            else:
                c, r0 = rng.randint(0, C - 1), rng.randint(0, R - length)
                seg = [(r0 + i, c) for i in range(length)]
            walls.update(cell for cell in seg if cell not in reserved)

        seen = reachable(walls, start, R, C)
        if diamond not in seen or exit_cell not in seen:
            continue
        p_sd = shortest_path(walls, start, diamond, R, C)
        p_de = shortest_path(walls, diamond, exit_cell, R, C)
        if not p_sd or not p_de:
            continue

        # interior cells of the SHORT route → these become the danger we make the
        # short path pass through (a "wall" of cameras + a trap), with guards nearby.
        ordered = p_sd + p_de[1:]
        interior = [c for c in ordered if c not in (start, exit_cell, diamond)]
        if len(interior) < 5:
            continue
        mid = len(interior) // 2
        span = min(len(interior), max(3, n_cameras + n_traps))
        lo = max(0, mid - span // 2)
        stretch = interior[lo:lo + span]
        traps = stretch[-n_traps:] if n_traps else []
        cameras = [c for c in stretch if c not in traps]
        if len(cameras) < 2:
            continue
        outside = [c for c in interior if c not in stretch]
        guard_starts = outside[:n_guards]

        guard_cells = set()
        for g in guard_starts:
            guard_cells.update(museum_guard_route(g, walls, exit_cell, R, C))
        danger = (set(cameras) | set(traps) | guard_cells) - {start, exit_cell, diamond}

        # a danger-free SAFE route must exist and be meaningfully longer
        s_sd = shortest_path(walls, start, diamond, R, C, extra_blocked=danger)
        s_de = shortest_path(walls, diamond, exit_cell, R, C, extra_blocked=danger)
        if not s_sd or not s_de:
            continue
        short_len = path_length(p_sd) + path_length(p_de)
        safe_len = path_length(s_sd) + path_length(s_de)
        if safe_len - short_len < 3:
            continue

        layout = [["." for _ in range(C)] for _ in range(R)]
        for cell in walls:
            _put(layout, cell, "#")
        for cell in cameras:
            _put(layout, cell, "C")
        for cell in traps:
            _put(layout, cell, "T")
        for cell in guard_starts:
            _put(layout, cell, "P")
        # icy tiles only off the safe route, for flavour
        safe_cells = set(s_sd) | set(s_de) | danger | {start, exit_cell, diamond}
        slip_pool = [(r, c) for r in range(R) for c in range(C)
                     if layout[r][c] == "." and (r, c) not in safe_cells]
        rng.shuffle(slip_pool)
        for cell in slip_pool[:n_slippery]:
            _put(layout, cell, "~")
        _put(layout, start, "S")
        _put(layout, exit_cell, "X")
        _put(layout, diamond, "G")
        return ["".join(row) for row in layout]

    return [row[:] for row in EASY_LAYOUT]  # guaranteed-valid fallback

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
        r_exit: float = 120.0,
        r_camera: float = -25.0,
        r_trap: float = -20.0,
        r_guard: float = -50.0,
        r_slip: float = -5.0,
        r_wall: float = -5.0,
        r_exit_early: float = -10.0,
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
        self.r_guard = r_guard
        self.r_slip = r_slip
        self.r_wall = r_wall
        self.r_exit_early = r_exit_early
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        walls, slippery = set(), set()
        self.cameras, self.traps = set(), set()
        self.guard_starts: List[Cell] = []
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
                elif ch == "P":
                    self.guard_starts.append(cell)
        self.walls = walls
        self.slippery = slippery
        self.grid = GridBase(self.rows, self.cols, walls=walls,
                             slippery=slippery, slip_prob=slip_prob)
        self.guard_routes = [self._guard_route(cell) for cell in self.guard_starts]
        self.guard_period = max([len(route) for route in self.guard_routes] or [1])

        self.n_actions = len(ACTIONS)
        self.pos: Cell = self.start
        self.has_diamond: int = 0
        self.guard_phase = 0
        self.steps = 0

    # -- patrol guards ------------------------------------------------------
    def _guard_route(self, start: Cell) -> List[Cell]:
        return museum_guard_route(start, self.walls, self.exit, self.rows, self.cols)

    def guard_positions(self, phase: int = None) -> List[Cell]:
        if phase is None:
            phase = self.guard_phase
        return [route[(phase + i * 2) % len(route)] for i, route in enumerate(self.guard_routes)]

    def _state(self):
        if self.guard_routes:
            return (self.pos, self.has_diamond, self.guard_phase)
        return (self.pos, self.has_diamond)

    # -- core dynamics ------------------------------------------------------
    def reset(self):
        self.pos = self.start
        self.has_diamond = 0
        self.guard_phase = 0
        self.steps = 0
        return self._state()

    def step(self, action: int):
        ncell, slipped, hit_wall = self.grid.sample_cell(self.pos, action, self.rng)
        reward = self.r_step
        if slipped:
            reward += self.r_slip
        if hit_wall:
            reward += self.r_wall

        camera = ncell in self.cameras
        trap = ncell in self.traps
        success = False
        if camera:                           # camera vision zone: penalty, NOT terminal
            reward += self.r_camera
        elif trap:
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

        # patrol guards advance; being caught ends the heist
        old_guards = set(self.guard_positions())
        self.guard_phase = (self.guard_phase + 1) % self.guard_period
        new_guards = set(self.guard_positions())
        caught = False
        if not success and (ncell in old_guards or ncell in new_guards):
            reward += self.r_guard
            caught = True

        self.pos = ncell
        self.steps += 1
        done = success or caught or self.steps >= self.max_steps
        info = {"slipped": slipped, "hit_wall": hit_wall, "camera": camera,
                "trap": trap, "caught": caught, "success": success}
        return self._state(), reward, done, info

    # -- rendering helper ---------------------------------------------------
    def render_state(self) -> dict:
        return {
            "pos": self.pos,
            "has_diamond": self.has_diamond,
            "guards": self.guard_positions(),
        }
