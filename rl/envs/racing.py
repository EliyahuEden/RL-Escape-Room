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

# '#'=grass infield '.'=track 'S'=start 'F'=finish '1'/'2'/'3'=checkpoint gates
# 'O'=oil 'M'=gravel trap 'X'=crash barrier 'R'=risky racing line marker

# ── Grand-Prix circuit (10×10) ──────────────────────────────────────────────
#
#   A proper F1-style ribbon around a grass infield, centred in the grid with
#   grass run-off top and bottom (the first three rooms stay on a 10×10 grid):
#     * row 8 — the MAIN STRAIGHT (the express lane): S → F straight along a
#       wall of TecPro crash barriers (row 7). Shortest line on the map, but
#       one twitch up into the barriers ends the race — the "cliff".
#     * cols 0 & 9 + the back straight (row 1) — the outer loop (the safe way
#       round): longer, through the gravel run-off, nowhere near anything
#       terminal — a SARSA rival prices the barrier risk in and takes it.
#     * rows 1-2 — a CHICANE that kinks the back straight into the infield.
#   Every gate sits on BOTH lines (one cell on the main straight, one on the
#   outer loop) so there are two complete ways to drive from start to finish.
#
TRACK_BASE_LAYOUT = [
    "##########",  # 0: grass run-off (top)
    "....##....",  # 1: back straight, split by the chicane gap
    ".##....##.",  # 2: chicane apex (dips into the infield)
    ".########.",  # 3: outer loop (cols 0 & 9) around the grass infield
    ".########.",  # 4
    ".########.",  # 5
    ".########.",  # 6
    ".XXXXXXXX.",  # 7: crash barriers — the cliff (open ends = pit exits)
    "S........F",  # 8: main straight — the express lane
    "##########",  # 9: grass run-off (bottom)
]

# ── Default layout (hand-placed gates + gravel) ─────────────────────────────
#    Gate 1 / 2 each have a cell on the main straight (row 8) AND on the outer
#    loop (row 1), so the express line and the safe loop both complete the lap.
DEFAULT_LAYOUT = [
    "##########",  # 0: grass run-off (top)
    "..1.##.2..",  # 1: back straight — safe checkpoint cells + chicane gap
    ".##....##.",  # 2: chicane apex
    ".########M",  # 3: gravel run-off on the right-hand straight
    "O########O",  # 4: oil slicks on the outer-loop straights (slippery, survivable)
    "M########.",  # 5: gravel run-off on the left-hand straight
    ".########O",  # 6: another oil slick on the right-hand straight
    ".XXXXXXXX.",  # 7: crash barriers — the cliff
    "SRRR1RR2RF",  # 8: main straight (R = racing line) + risky checkpoints
    "##########",  # 9: grass run-off (bottom)
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
    # the "safe" line avoids the barrier-hugging racing line itself (oil is
    # survivable now — a slip there bounces off the grass, it does not crash)
    safe = shortest_path(walls, start, finish, rows, cols,
                         extra_blocked=crash | parsed["shortcut"])
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
    n_oil: int = 3,
    n_mud: int = 2,
    n_gates: int = 2,
    n_crash: int = 10,         # kept for signature compat; the barrier line is structural
) -> List[str]:
    """Generate a Grand-Prix lap on the fixed F1-circuit structure.

    The main straight / barrier line / outer loop skeleton never changes —
    that asymmetry IS the lesson — but every seed shuffles the checkpoint
    gate columns and gravel run-off, so each map still plays differently.
    Each gate gets one main-straight cell and one outer-loop cell, so both
    lines complete the lap.
    """
    rng = random.Random(seed)
    layout = [list(row) for row in TRACK_BASE_LAYOUT]
    N = len(layout)              # 10

    # structural rows (derived, so this survives layout tweaks)
    express_row = next(r for r, row in enumerate(TRACK_BASE_LAYOUT) if "S" in row)
    back_row = next(r for r, row in enumerate(TRACK_BASE_LAYOUT)
                    if "." in row[2:N - 2])   # topmost track straight

    # racing line along the main straight
    for c in range(1, N - 1):
        if layout[express_row][c] == ".":
            layout[express_row][c] = "R"

    # checkpoint gates: gate 1 early, gate 2 late, well separated so each
    # inter-gate chain along the cliff stays short enough to learn.
    g1 = rng.randint(3, 4)
    g2 = rng.randint(6, 7)
    express_cols = [g1, g2][:max(1, n_gates)]
    for gi, col in enumerate(express_cols, start=1):
        layout[express_row][col] = str(gi)

    # matching gate cells on the outer loop (back straight), same order
    # left→right, skipping the chicane gap in the middle of the back straight
    ring_cols = [rng.choice([2, 3]), rng.choice([6, 7])][:max(1, n_gates)]
    for gi, col in enumerate(ring_cols, start=1):
        if layout[back_row][col] == ".":
            layout[back_row][col] = str(gi)

    # gravel run-off on the two vertical straights (cols 0 and N-1)
    mud_slots = [(r, c) for c in (0, N - 1)
                 for r in range(back_row + 2, express_row - 1)]
    rng.shuffle(mud_slots)
    placed = 0
    for r, c in mud_slots:
        if placed >= n_mud:
            break
        if layout[r][c] == ".":
            layout[r][c] = "M"
            placed += 1

    # oil slicks on the outer-loop straights. A slip on oil bounces off the
    # grass run-off (survivable) — only the barrier line crashes — so oil is a
    # visible, slowing hazard on the safe route, not an instant-death trap.
    oil_slots = [(r, c) for c in (0, N - 1)
                 for r in range(back_row + 2, express_row - 1)]
    rng.shuffle(oil_slots)
    placed_oil = 0
    for r, c in oil_slots:
        if placed_oil >= n_oil:
            break
        if layout[r][c] == ".":
            layout[r][c] = "O"
            placed_oil += 1

    result = ["".join(row) for row in layout]

    # sanity: a crash-free outer loop must exist and be meaningfully longer
    # than the straight-line main straight; otherwise fall back to the default
    parsed = _parse_layout(result)
    blocked = parsed["walls"] | parsed["crash"]
    express = path_length(shortest_path(blocked, parsed["start"], parsed["finish"], N, N))
    safe = path_length(shortest_path(blocked, parsed["start"], parsed["finish"], N, N,
                                     extra_blocked=parsed["shortcut"]))
    if express is None or safe is None or safe < express + 6:
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
        r_slip: float = -5.0,
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
        self.r_slip = r_slip
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

        # Only the crash BARRIERS end the race (the "cliff"). Oil that slides the
        # car into the grass run-off makes it lose control and bounce — costly,
        # but survivable — so oil is a real hazard without being instant death.
        if ncell in self.crash:
            reward += self.r_crash
            self.pos = ncell
            info = {
                "slipped": slipped,
                "crash": True,
                "success": False,
                "shortcut": self.pos in self.shortcut_cells,
            }
            return (self.pos, self.next_cp), reward, True, info

        if slipped:
            reward += self.r_slip          # oil made the car lose grip

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
