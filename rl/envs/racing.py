"""Room 3 - Grand Prix street circuit (Q-Learning vs. a SARSA rival).

A lap with **checkpoint gates**: the car must cross checkpoint 1, then
checkpoint 2, and only then does the finish line open.  Every gate exists
in TWO places — one cell on the **express lane** and one on the **ring
road** — so there are two complete ways to drive the lap:

* the **express lane** (bottom row) runs straight from start to finish,
  hugging a wall of **crash barriers** — one sideways move into them ends
  the race.  Driven greedily it is perfectly safe *and* the fastest lap
  on the map;
* the **ring road** detours around the barriers — several cells longer,
  through the gravel traps, but nowhere near anything terminal.

The track is the classic *cliff walking* problem staged as a race.  Under
ε-greedy exploration the express lane is exactly a cliff edge: every step
beside the barriers risks an exploratory move into them.  **SARSA**
(on-policy) prices that risk into its Q-values and settles on the ring
road; **Q-Learning** (off-policy) backs up the greedy value and learns the
express lane.  Racing the two learned policies side by side, Q-Learning
wins on lap time — the textbook off-policy/on-policy split.

Other hazards: **oil spills** (slippery — deflect sideways with
probability ``slip_prob``; slipping into a wall = crash) and **gravel
traps** (step penalty).

State
-----
``(cell, next_checkpoint)`` — position plus the index of the next gate to
cross (0, 1, ... n_gates).  The final state is crossing the finish line
after all gates.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

from rl.envs.grid_base import ACTIONS, DOWN, GridBase, LEFT, RIGHT, UP, path_length, reachable, shortest_path

Cell = Tuple[int, int]
_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# '#'=off-track '.'=track 'S'=start 'F'=finish '1'/'2'/'3'=checkpoint gates
# 'O'=oil 'M'=gravel trap 'X'=crash barrier 'R'=risky racing line marker

# ── Street layout (structure: city blocks + the barrier line) ──────────────
#
#   Rows 0-6: city grid.  Row 7: the open ring road (safe detour).
#   Row 8: construction crash barriers (the "cliff").  Row 9: the express
#   lane — the shortest line on the map, right along the barriers.
#
TRACK_BASE_LAYOUT = [
    "..........",   # 0: north boulevard
    ".##.##.##.",   # 1: city blocks
    "..........",   # 2: cross street
    ".##.##.##.",   # 3: city blocks
    "..........",   # 4: mid boulevard
    ".##.##.##.",   # 5: city blocks
    "..........",   # 6: ring boulevard (the safe detour)
    "..........",   # 7: buffer row — keeps the detour away from the cliff
    "..XXXXXX..",   # 8: construction barriers — the cliff
    "S........F",   # 9: express lane
]

# ── Default layout (hand-placed items) ──────────────────────────────────────
#    Each checkpoint gate has one cell on the ring road (row 6) and one on
#    the express lane (row 9) at the same column — two ways to drive the lap.
DEFAULT_LAYOUT = [
    "..........",   # 0: north side
    ".##.##.##.",   # 1: infield blocks
    "..........",   # 2: cross link
    ".##.##.##.",   # 3: infield blocks
    "...O..O...",   # 4: oil punishes the deep-north alternative
    ".##.##.##.",   # 5: infield blocks
    "..M1.M.2..",   # 6: ring road — gravel traps + safe checkpoint cells
    "..........",   # 7: buffer row (clean)
    "..XXXXXX..",   # 8: crash barriers — the cliff
    "SRR1RRR2RF",   # 9: express lane (R = racing line) + risky checkpoint cells
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
    gate_cells: Dict[int, Set[Cell]] = {}
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
            elif ch in "123":
                gate_cells.setdefault(int(ch), set()).add(cell)
            elif ch == "M":
                mud.add(cell)
            elif ch == "O":
                oil.add(cell)
                shortcut.add(cell)
            elif ch == "R":
                shortcut.add(cell)
            elif ch == "X":
                crash.add(cell)

    checkpoints = [gate_cells[k] for k in sorted(gate_cells)]
    return {
        "rows": rows,
        "cols": cols,
        "walls": walls,
        "oil": oil,
        "mud": mud,
        "crash": crash,
        "shortcut": shortcut,
        "checkpoints": checkpoints,
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
    # the "safe" line avoids oil AND the barrier-hugging racing line itself
    safe = shortest_path(walls, start, finish, rows, cols,
                         extra_blocked=oil | crash | parsed["shortcut"])
    short_len = path_length(short)
    safe_len = path_length(safe)
    return {
        "short_len": short_len,
        "safe_len": safe_len,
        "safe_gap": None if short_len is None or safe_len is None else safe_len - short_len,
        "oil": len(oil),
        "mud": len(parsed["mud"]),
        "checkpoints": len(parsed["checkpoints"]),
        "crash": len(crash),
        "shortcut_cells": len(parsed["shortcut"]),
    }


def generate_racing_layout(
    seed: int = 0,
    n_oil: int = 2,
    n_mud: int = 2,
    n_gates: int = 2,
    n_crash: int = 8,          # kept for signature compat; the barrier line is structural
) -> List[str]:
    """Generate a Grand Prix lap on the fixed cliff-track structure.

    The express lane / barrier line / ring road skeleton never changes —
    that asymmetry IS the lesson — but every seed shuffles the checkpoint
    gate columns, gravel traps and oil, so each map still plays
    differently.  Each gate gets one express-lane cell and one ring-road
    cell at the same column: two complete ways to drive the lap.
    """
    rng = random.Random(seed)
    layout = [list(row) for row in TRACK_BASE_LAYOUT]

    # racing line along the express lane
    for c in range(1, 9):
        layout[9][c] = "R"

    # checkpoint gates: gate 1 early-mid, gate 2 mid-late; both rows.
    # (columns 3-7 keep every inter-gate value chain short enough for
    # exploration along the cliff to actually learn it)
    g1 = rng.randint(3, 4)
    g2 = rng.randint(6, 7)
    gate_cols = [g1, g2][:max(1, n_gates)]
    for gi, col in enumerate(gate_cols, start=1):
        layout[9][col] = str(gi)
        layout[6][col] = str(gi)

    # gravel traps on the ring road, avoiding the gate cells
    trap_pool = [c for c in range(1, 9) if layout[6][c] == "."]
    rng.shuffle(trap_pool)
    for c in trap_pool[:max(0, n_mud)]:
        layout[6][c] = "M"

    # oil only at open intersections where a slip deflects onto open road
    # (row 4 mid boulevard, between street gaps — never beside a wall)
    oil_spots = [(4, c) for c in (3, 6)
                 if layout[3][c] == "." and layout[5][c] == "."]
    for cell in rng.sample(oil_spots, k=min(n_oil, len(oil_spots))):
        _put(layout, cell, "O")

    result = ["".join(row) for row in layout]

    # sanity: express lane must be the unique 9-step line; a crash-free
    # detour must exist and be meaningfully longer
    parsed = _parse_layout(result)
    blocked = parsed["walls"] | parsed["crash"]
    express = path_length(shortest_path(blocked, parsed["start"], parsed["finish"], 10, 10))
    safe = path_length(shortest_path(blocked, parsed["start"], parsed["finish"], 10, 10,
                                     extra_blocked=parsed["shortcut"] | parsed["oil"]))
    if express != 9 or safe is None or safe < express + 3:
        return [row[:] for row in DEFAULT_LAYOUT]
    return result


class RacingEnv:
    def __init__(
        self,
        layout: List[str] = None,
        slip_prob: float = 0.2,
        max_steps: int = 200,
        r_step: float = -1.0,
        r_finish: float = 200.0,
        r_checkpoint: float = 40.0,
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
        self.r_checkpoint = r_checkpoint
        self.r_mud = r_mud
        self.r_crash = r_crash
        self.r_offtrack = r_offtrack
        self.r_wall = r_wall
        self.r_finish_locked = r_finish_locked
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.walls = parsed["walls"]
        self.oil = parsed["oil"]
        self.mud = parsed["mud"]
        self.crash = parsed["crash"]
        self.shortcut_cells = parsed["shortcut"]
        self.checkpoints: List[Set[Cell]] = parsed["checkpoints"]
        self.n_gates = len(self.checkpoints)
        self.start = parsed["start"]
        self.finish = parsed["finish"]
        self.grid = GridBase(self.rows, self.cols, walls=self.walls, slippery=self.oil, slip_prob=slip_prob)

        self.n_actions = len(ACTIONS)
        self.pos: Cell = self.start
        self.next_cp: int = 0
        self.steps = 0

    def finish_unlocked(self, next_cp: int) -> bool:
        return next_cp >= self.n_gates

    def reset(self):
        self.pos = self.start
        self.next_cp = 0
        self.steps = 0
        return (self.pos, self.next_cp)

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
            return (self.pos, self.next_cp), reward, True, info

        checkpoint = False
        if hit_wall:
            dr, dc = _DELTA[action]
            target = (self.pos[0] + dr, self.pos[1] + dc)
            reward += self.r_offtrack if not self.grid.in_bounds(target) else self.r_wall
        else:
            if ncell in self.mud:
                reward += self.r_mud
            if (self.next_cp < self.n_gates
                    and ncell in self.checkpoints[self.next_cp]):
                reward += self.r_checkpoint
                self.next_cp += 1
                checkpoint = True

        finished = False
        if ncell == self.finish:
            if self.finish_unlocked(self.next_cp):
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
            "checkpoint": checkpoint,
            "shortcut": ncell in self.shortcut_cells,
        }
        return (self.pos, self.next_cp), reward, done, info

    def render_state(self) -> dict:
        return {
            "pos": self.pos,
            "next_cp": self.next_cp,
            "finish_unlocked": self.finish_unlocked(self.next_cp),
        }
