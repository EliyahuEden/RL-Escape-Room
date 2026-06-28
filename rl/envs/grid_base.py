"""Shared geometry and stochastic transition model for the 10x10 grid rooms.

Rooms 1-3 all live on a 10x10 grid and share:

* 4-connected movement (Up / Down / Left / Right).  An illegal move (into a
  wall or off the board) keeps the agent in place.
* "Slippery" / icy cells: when the agent *stands on* a slippery cell, its chosen
  action is, with probability ``slip_prob``, deflected **sideways** (perpendicular
  to the intended direction) — half the slip mass to each side.  So with
  ``slip_prob = 0.2`` the move is 80 % intended / 10 % slip-left / 10 % slip-right
  (Room 1 ice); ``slip_prob = 0.3`` gives 70 / 15 / 15 (Room 3 oil).  This makes
  the transitions stochastic in a directional, realistic way.

Room 1 (Dynamic Programming) needs the *explicit* model, so
:meth:`GridBase.cell_transitions` returns the full probability distribution over
next cells.  Rooms 2-3 (SARSA / Q-Learning) treat the model as unknown and only
*sample* from it through :meth:`GridBase.sample_cell`.  Both use the exact same
distribution, so a policy found by DP is directly comparable to one learned by
the model-free agents.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Iterable, List, Optional, Set, Tuple

# --- Action encoding -------------------------------------------------------
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS: Tuple[int, ...] = (UP, DOWN, LEFT, RIGHT)
ACTION_NAMES = ("Up", "Down", "Left", "Right")
ACTION_ARROWS = ("↑", "↓", "←", "→")  # ↑ ↓ ← →
_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
# perpendicular ("left", "right") deflections for the slip model
_PERP = {UP: (LEFT, RIGHT), DOWN: (RIGHT, LEFT), LEFT: (DOWN, UP), RIGHT: (UP, DOWN)}

Cell = Tuple[int, int]


def reachable(walls: Set[Cell], start: Cell, rows: int, cols: int) -> Set[Cell]:
    """4-connected flood fill from ``start`` over non-wall cells.

    Used by the random layout generators to guarantee a map is solvable before
    it is handed to a room.
    """
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < rows and 0 <= n[1] < cols and n not in walls and n not in seen:
                seen.add(n)
                q.append(n)
    return seen


def shortest_path(walls: Set[Cell], start: Cell, goal: Cell, rows: int, cols: int,
                  extra_blocked: Iterable[Cell] = ()) -> Optional[List[Cell]]:
    """BFS shortest path ``start``→``goal`` (inclusive) over non-blocked cells, or
    ``None`` if unreachable. ``extra_blocked`` is treated as impassable (used to
    compute a *danger-free* route by blocking camera/trap/oil cells)."""
    blocked = set(walls) | set(extra_blocked)
    if start == goal:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cur[0] + dr, cur[1] + dc)
            if (0 <= n[0] < rows and 0 <= n[1] < cols
                    and n not in blocked and n not in prev):
                prev[n] = cur
                if n == goal:
                    path = [n]
                    while path[-1] is not None:
                        path.append(prev[path[-1]])
                    return path[-2::-1]  # drop trailing None, reverse
                q.append(n)
    return None


def path_length(path: Optional[List[Cell]]) -> Optional[int]:
    """Number of steps (edges) in a path, or ``None``."""
    return None if path is None else len(path) - 1


class GridBase:
    """Geometry + slip dynamics shared by the three grid rooms."""

    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        walls: Optional[Iterable[Cell]] = None,
        slippery: Optional[Iterable[Cell]] = None,
        slip_prob: float = 0.2,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.walls = set(walls or ())
        self.slippery = set(slippery or ())
        self.slip_prob = float(slip_prob)

    # -- geometry -----------------------------------------------------------
    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, cell: Cell) -> bool:
        return cell in self.walls

    def is_slippery(self, cell: Cell) -> bool:
        return cell in self.slippery

    def _resolve(self, cell: Cell, action: int) -> Cell:
        """Apply one deterministic move; stay put if it hits a wall/boundary."""
        return self._resolve_hit(cell, action)[0]

    def _resolve_hit(self, cell: Cell, action: int) -> Tuple[Cell, bool]:
        """Deterministic move returning ``(next_cell, hit_wall)``.

        ``hit_wall`` is True when the move was blocked by a wall or the board
        boundary (so the agent stays put).
        """
        dr, dc = _DELTAS[action]
        nxt = (cell[0] + dr, cell[1] + dc)
        if not self.in_bounds(nxt) or nxt in self.walls:
            return cell, True
        return nxt, False

    # -- stochastic transition model ---------------------------------------
    def cell_transitions(self, cell: Cell, action: int) -> List[Tuple[float, Cell, bool, bool]]:
        """Full distribution over outcomes as ``(prob, next_cell, slipped, hit_wall)``.

        Used by Dynamic Programming, which needs the complete model.  On a
        slippery cell the move is deflected to the two perpendicular directions.
        """
        if cell not in self.slippery or self.slip_prob <= 0.0:
            nc, hit = self._resolve_hit(cell, action)
            return [(1.0, nc, False, hit)]
        nc, hit = self._resolve_hit(cell, action)
        out = [(1.0 - self.slip_prob, nc, False, hit)]
        half = self.slip_prob / 2.0
        for side in _PERP[action]:
            ncs, hits = self._resolve_hit(cell, side)
            out.append((half, ncs, True, hits))
        return out

    def sample_cell(self, cell: Cell, action: int,
                    rng: random.Random = random) -> Tuple[Cell, bool, bool]:
        """Sample one ``(next_cell, slipped, hit_wall)`` from the same distribution.

        Used by the model-free agents, which never see :meth:`cell_transitions`.
        """
        slipped = False
        if cell in self.slippery and rng.random() < self.slip_prob:
            left, right = _PERP[action]
            action = left if rng.random() < 0.5 else right
            slipped = True
        nc, hit = self._resolve_hit(cell, action)
        return nc, slipped, hit
