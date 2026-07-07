"""Room 2 - Museum Heist (SARSA, unknown model).

The robber starts at the museum entrance, must sneak through galleries to
reach the **diamond** in the vault, and escape through the exit — all while
avoiding security cameras, laser traps, and patrol guards.

The model is unknown — SARSA learns the environment from sampled experience.

The museum layout features:
* Gallery rooms connected by corridors with **pillars** and **exhibits**
* A **vault** at the top where the diamond is kept
* **Camera surveillance zones** covering key corridors (heavy penalty — and
  one sighting triggers the **alarm**: every guard abandons its patrol and
  chases the robber for the rest of the heist)
* **Laser traps** near the vault (penalty)
* **Patrol guards** roaming the galleries (terminal if caught)
* **Slippery marble floors** that can deflect movement

The generated layouts keep the same museum floor plan (walls stay fixed) but
randomize where the diamond, cameras, traps, guards, and slippery tiles are
placed — so every seed gives a different heist challenge on a familiar map.

State
-----
``(cell, has_diamond)`` plus the guard information when guards exist: the
patrol ``guard_phase`` while the museum is quiet, or — once the alarm has
been raised — the alarm flag together with the guards' actual positions
(the chase makes guard cells depend on history, so they must be part of the
state to keep it Markov).  The final state is reaching the exit with the
diamond.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

from rl.envs.grid_base import ACTIONS, GridBase, path_length, reachable, shortest_path

Cell = Tuple[int, int]

# '#'=wall '.'=floor 'S'=start 'X'=exit 'G'=diamond
# 'K'=camera device 'V'=camera vision 'T'=trap '~'=slippery 'P'=patrol guard

# ── Museum floor plan (walls only) ──────────────────────────────────────────
#
#   The fixed architecture: vault at top, galleries, pillars, corridors.
#   Items (diamond, cameras, traps, guards, slippery) are placed on top.
#
MUSEUM_BASE_LAYOUT = [
    "###...####",   # 0: vault room
    "###...####",   # 1: vault antechamber
    "..........",   # 2: upper gallery
    ".##....##.",   # 3: gallery divider walls
    "..........",   # 4: main exhibition hall
    ".#.#..#.#.",   # 5: pillars / exhibit cases
    "..........",   # 6: lower gallery
    ".#......#.",   # 7: corridor walls
    "..........",   # 8: lobby
    "S........X",   # 9: entrance and exit
]

# ── Curated default layout (hand-placed items on the floor plan) ────────────
DEFAULT_LAYOUT = [
    "###.G.####",   # 0: diamond in vault
    "###...####",   # 1: vault antechamber
    "~..P.....~",   # 2: slippery marble, guard patrol
    ".##.TT.##.",   # 3: laser traps guarding vault approach
    "..........",   # 4: main exhibition hall
    ".#.#..#.#.",   # 5: pillars / exhibit cases
    "......P...",   # 6: lower gallery, guard patrol
    "~#VVVVVV#~",   # 7: camera surveillance corridor
    "..VVVVVV..",   # 8: camera vision zone
    "S........X",   # 9: entrance and exit
]


def _put(layout: List[List[str]], cell: Cell, ch: str) -> None:
    layout[cell[0]][cell[1]] = ch


def _parse_layout(layout: List[str]) -> Dict:
    rows, cols = len(layout), len(layout[0])
    walls: Set[Cell] = set()
    camera_zones: Set[Cell] = set()
    camera_devices: Set[Cell] = set()
    traps: Set[Cell] = set()
    slippery: Set[Cell] = set()
    guard_starts: List[Cell] = []
    start: Cell = (rows - 1, 0)
    exit_cell: Cell = (rows - 1, cols - 1)
    diamond: Cell = (0, 0)

    for r, line in enumerate(layout):
        for c, ch in enumerate(line):
            cell = (r, c)
            if ch == "#":
                walls.add(cell)
            elif ch == "K":
                walls.add(cell)
                camera_devices.add(cell)
            elif ch in {"V", "C"}:
                camera_zones.add(cell)
            elif ch == "S":
                start = cell
            elif ch == "X":
                exit_cell = cell
            elif ch == "G":
                diamond = cell
            elif ch == "T":
                traps.add(cell)
            elif ch == "~":
                slippery.add(cell)
            elif ch == "P":
                guard_starts.append(cell)

    return {
        "rows": rows,
        "cols": cols,
        "walls": walls,
        "camera_zones": camera_zones,
        "camera_devices": camera_devices,
        "traps": traps,
        "slippery": slippery,
        "guard_starts": guard_starts,
        "start": start,
        "exit": exit_cell,
        "diamond": diamond,
    }


def museum_guard_route(start: Cell, walls, exit_cell: Cell, rows: int, cols: int,
                       patrol_len: int = 8) -> List[Cell]:
    """Back-and-forth patrol along the longest free row/column run from start."""

    def free(cell: Cell) -> bool:
        return (
            0 <= cell[0] < rows
            and 0 <= cell[1] < cols
            and cell not in walls
            and cell != exit_cell
        )

    half = patrol_len // 2

    row_cells = [start]
    for dc in range(1, half + 1):
        c = (start[0], start[1] - dc)
        if free(c):
            row_cells.insert(0, c)
        else:
            break
    for dc in range(1, half + 1):
        c = (start[0], start[1] + dc)
        if free(c):
            row_cells.append(c)
        else:
            break

    col_cells = [start]
    for dr in range(1, half + 1):
        c = (start[0] - dr, start[1])
        if free(c):
            col_cells.insert(0, c)
        else:
            break
    for dr in range(1, half + 1):
        c = (start[0] + dr, start[1])
        if free(c):
            col_cells.append(c)
        else:
            break

    route = row_cells if len(row_cells) >= len(col_cells) else col_cells
    if len(route) <= 1:
        return [start]
    return route + route[-2:0:-1]


def museum_layout_stats(layout: List[str]) -> Dict[str, object]:
    """Return route facts used by the UI and smoke tests."""
    parsed = _parse_layout(layout)
    walls = parsed["walls"]
    rows, cols = parsed["rows"], parsed["cols"]
    start, diamond, exit_cell = parsed["start"], parsed["diamond"], parsed["exit"]

    guard_cells: Set[Cell] = set()
    for guard in parsed["guard_starts"]:
        guard_cells.update(museum_guard_route(guard, walls, exit_cell, rows, cols))
    danger = parsed["camera_zones"] | parsed["traps"] | guard_cells
    danger -= {start, diamond, exit_cell}

    p_sd = shortest_path(walls, start, diamond, rows, cols)
    p_de = shortest_path(walls, diamond, exit_cell, rows, cols)
    s_sd = shortest_path(walls, start, diamond, rows, cols, extra_blocked=danger)
    s_de = shortest_path(walls, diamond, exit_cell, rows, cols, extra_blocked=danger)
    short_len = None if not p_sd or not p_de else path_length(p_sd) + path_length(p_de)
    safe_len = None if not s_sd or not s_de else path_length(s_sd) + path_length(s_de)
    return {
        "short_len": short_len,
        "safe_len": safe_len,
        "safe_gap": None if short_len is None or safe_len is None else safe_len - short_len,
        "danger_cells": len(danger),
        "camera_zones": len(parsed["camera_zones"]),
        "camera_devices": len(parsed["camera_devices"]),
        "traps": len(parsed["traps"]),
        "guards": len(parsed["guard_starts"]),
        "slippery": len(parsed["slippery"]),
    }


def generate_museum_layout(
    seed: int = 0,
    n_cameras: int = 7,
    n_traps: int = 2,
    n_slippery: int = 6,
    n_guards: int = 2,
) -> List[str]:
    """Generate a museum by placing items on the fixed floor plan.

    The wall structure stays the same (like a real building) but every seed
    gives a different diamond location, camera placement, trap positions,
    guard patrols, and slippery marble tiles.

    Since one camera sighting now triggers a rest-of-episode manhunt, every
    generated heist is **verified to have a camera/trap-free route**
    (start → diamond → exit); layouts are re-rolled — with progressively
    fewer cameras — until one exists.
    """
    rng = random.Random(seed)
    R = C = 10
    start = (R - 1, 0)
    exit_cell = (R - 1, C - 1)

    base = _parse_layout(MUSEUM_BASE_LAYOUT)
    walls = base["walls"]

    for attempt in range(60):
        # every 15 failed attempts, ease off one camera
        cams_now = max(3, n_cameras - attempt // 15)
        layout = [list(row) for row in MUSEUM_BASE_LAYOUT]

        free = [
            (r, c) for r in range(R) for c in range(C)
            if (r, c) not in walls and (r, c) != start and (r, c) != exit_cell
        ]

        vault_cells = [(r, c) for r, c in free if r <= 1]
        diamond = rng.choice(vault_cells) if vault_cells else free[0]
        _put(layout, diamond, "G")
        reserved = {start, exit_cell, diamond}

        placeable = [c for c in free if c not in reserved]
        rng.shuffle(placeable)

        path_sd = shortest_path(walls, start, diamond, R, C)
        path_de = shortest_path(walls, diamond, exit_cell, R, C)
        on_path = set(path_sd or []) | set(path_de or []) if path_sd and path_de else set()
        on_path -= reserved

        near_path = [c for c in placeable if c in on_path]
        off_path = [c for c in placeable if c not in on_path]

        cameras_placed: List[Cell] = []
        for cell in near_path[:cams_now]:
            _put(layout, cell, "V")
            cameras_placed.append(cell)
        remaining_cam = cams_now - len(cameras_placed)
        if remaining_cam > 0:
            for cell in off_path[:remaining_cam]:
                _put(layout, cell, "V")
                cameras_placed.append(cell)
                off_path = off_path[1:]

        used = set(cameras_placed)
        trap_pool = [c for c in near_path if c not in used]
        rng.shuffle(trap_pool)
        for cell in trap_pool[:n_traps]:
            _put(layout, cell, "T")
            used.add(cell)

        guard_pool = [c for c in placeable if c not in used and c not in on_path]
        rng.shuffle(guard_pool)
        for cell in guard_pool[:n_guards]:
            _put(layout, cell, "P")
            used.add(cell)

        slip_pool = [c for c in placeable if c not in used]
        rng.shuffle(slip_pool)
        for cell in slip_pool[:n_slippery]:
            _put(layout, cell, "~")

        # a stealth route must exist: both heist legs passable without
        # stepping into a camera zone, a laser trap, or a guard patrol lane
        guard_cells: Set[Cell] = set()
        for gcell in guard_pool[:n_guards]:
            guard_cells.update(museum_guard_route(gcell, walls, exit_cell, R, C))
        avoid = (set(cameras_placed) | set(trap_pool[:n_traps]) | guard_cells)
        avoid -= {start, diamond, exit_cell}
        ok_sd = shortest_path(walls, start, diamond, R, C, extra_blocked=avoid)
        ok_de = shortest_path(walls, diamond, exit_cell, R, C, extra_blocked=avoid)
        if ok_sd and ok_de:
            return ["".join(row) for row in layout]

    return list(DEFAULT_LAYOUT)  # extremely unlucky seed: curated fallback


class MuseumEnv:
    def __init__(
        self,
        layout: List[str] = None,
        slip_prob: float = 0.1,
        max_steps: int = 200,
        r_step: float = -1.0,
        r_diamond: float = 30.0,
        r_exit: float = 100.0,
        r_camera: float = -50.0,
        r_trap: float = -15.0,
        r_guard: float = -50.0,
        r_slip: float = -5.0,
        r_wall: float = -2.0,
        r_exit_early: float = -5.0,
        alarm_enabled: bool = True,
        seed: int = None,
    ) -> None:
        self.layout = layout or DEFAULT_LAYOUT
        parsed = _parse_layout(self.layout)
        self.rows = parsed["rows"]
        self.cols = parsed["cols"]
        self.r_step = r_step
        self.r_diamond = r_diamond
        self.r_exit = r_exit
        self.r_camera = r_camera
        self.r_trap = r_trap
        self.r_guard = r_guard
        self.r_slip = r_slip
        self.r_wall = r_wall
        self.r_exit_early = r_exit_early
        self.alarm_enabled = alarm_enabled and bool(parsed["guard_starts"])
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.walls = parsed["walls"]
        self.cameras = parsed["camera_zones"]
        self.camera_devices = parsed["camera_devices"]
        self.traps = parsed["traps"]
        self.slippery = parsed["slippery"]
        self.guard_starts = parsed["guard_starts"]
        self.start = parsed["start"]
        self.exit = parsed["exit"]
        self.diamond = parsed["diamond"]
        self.grid = GridBase(
            self.rows,
            self.cols,
            walls=self.walls,
            slippery=self.slippery,
            slip_prob=slip_prob,
        )
        self.guard_routes = [self._guard_route(cell) for cell in self.guard_starts]
        self.guard_period = max([len(route) for route in self.guard_routes] or [1])

        self.n_actions = len(ACTIONS)
        self.pos: Cell = self.start
        self.has_diamond: int = 0
        self.guard_phase = 0
        self.alarmed: bool = False
        self.guard_cells: List[Cell] = self._patrol_positions(0)
        self.steps = 0

    def _guard_route(self, start: Cell) -> List[Cell]:
        return museum_guard_route(start, self.walls, self.exit, self.rows, self.cols)

    def _patrol_positions(self, phase: int) -> List[Cell]:
        return [route[(phase + i * 2) % len(route)]
                for i, route in enumerate(self.guard_routes)]

    def guard_positions(self) -> List[Cell]:
        return list(self.guard_cells)

    # -- alarm chase ---------------------------------------------------------
    def _chase_step(self, guard: Cell, target: Cell) -> Cell:
        """One deterministic greedy step that closes distance to the robber.

        Deterministic (fixed tie-breaking) so the transition stays a function
        of the state — the guards' positions are carried in the state.
        Guards never block the exit cell.
        """
        candidates = [guard]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (guard[0] + dr, guard[1] + dc)
            if (self.grid.in_bounds(nb) and nb not in self.walls
                    and nb != self.exit):
                candidates.append(nb)
        return min(candidates,
                   key=lambda c: (abs(c[0] - target[0]) + abs(c[1] - target[1]),
                                  c[0], c[1]))

    def _state(self):
        if not self.guard_routes:
            return (self.pos, self.has_diamond)
        if self.alarm_enabled:
            if self.alarmed:
                return (self.pos, self.has_diamond, 1, tuple(self.guard_cells))
            return (self.pos, self.has_diamond, 0, self.guard_phase)
        return (self.pos, self.has_diamond, self.guard_phase)

    def reset(self):
        self.pos = self.start
        self.has_diamond = 0
        self.guard_phase = 0
        self.alarmed = False
        self.guard_cells = self._patrol_positions(0)
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
        alarm_triggered = False
        if camera:
            reward += self.r_camera
            if self.alarm_enabled and not self.alarmed:
                self.alarmed = True         # rest-of-episode manhunt
                alarm_triggered = True
        elif trap:
            reward += self.r_trap
        elif ncell == self.diamond and not self.has_diamond:
            reward += self.r_diamond
            self.has_diamond = 1
        elif ncell == self.exit:
            if self.has_diamond:
                reward += self.r_exit
                success = True
            else:
                reward += self.r_exit_early
                ncell = self.pos

        old_guards = list(self.guard_cells)
        caught = False
        if self.alarmed:
            # alarm raised: every guard abandons its patrol and hunts the robber
            self.guard_cells = [self._chase_step(gc, ncell) for gc in old_guards]
            if not success:
                caught = any(g == ncell for g in self.guard_cells) or any(
                    gn == self.pos and go == ncell        # crossed paths
                    for go, gn in zip(old_guards, self.guard_cells))
        else:
            self.guard_phase = (self.guard_phase + 1) % self.guard_period
            self.guard_cells = self._patrol_positions(self.guard_phase)
            if not success and (ncell in set(old_guards)
                                or ncell in set(self.guard_cells)):
                caught = True
        if caught:
            reward += self.r_guard

        self.pos = ncell
        self.steps += 1
        done = success or caught or self.steps >= self.max_steps
        info = {
            "slipped": slipped,
            "hit_wall": hit_wall,
            "camera": camera,
            "trap": trap,
            "caught": caught,
            "success": success,
            "alarm": self.alarmed,
            "alarm_triggered": alarm_triggered,
        }
        return self._state(), reward, done, info

    def render_state(self) -> dict:
        return {
            "pos": self.pos,
            "has_diamond": self.has_diamond,
            "guards": self.guard_positions(),
            "alarm": self.alarmed,
        }
