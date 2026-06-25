"""Shared geometry and stochastic transition model for the 10x10 grid rooms.

Rooms 1-3 all live on a 10x10 grid and share:

* 4-connected movement (Up / Down / Left / Right).  An illegal move (into a
  wall or off the board) keeps the agent in place.
* "Slippery" / icy cells: when the agent *stands on* a slippery cell, its chosen
  action is, with probability ``slip_prob``, replaced by a uniformly random
  direction.  This makes the transitions stochastic.

Room 1 (Dynamic Programming) needs the *explicit* model, so
:meth:`GridBase.cell_transitions` returns the full probability distribution over
next cells.  Rooms 2-3 (SARSA / Q-Learning) treat the model as unknown and only
*sample* from it through :meth:`GridBase.sample_cell`.  Both use the exact same
distribution, so a policy found by DP is directly comparable to one learned by
the model-free agents.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Optional, Tuple

# --- Action encoding -------------------------------------------------------
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS: Tuple[int, ...] = (UP, DOWN, LEFT, RIGHT)
ACTION_NAMES = ("Up", "Down", "Left", "Right")
ACTION_ARROWS = ("↑", "↓", "←", "→")  # ↑ ↓ ← →
_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

Cell = Tuple[int, int]


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
        dr, dc = _DELTAS[action]
        nxt = (cell[0] + dr, cell[1] + dc)
        if not self.in_bounds(nxt) or nxt in self.walls:
            return cell
        return nxt

    # -- stochastic transition model ---------------------------------------
    def cell_transitions(self, cell: Cell, action: int) -> List[Tuple[float, Cell, bool]]:
        """Full distribution over outcomes as ``(prob, next_cell, slipped)``.

        Used by Dynamic Programming, which needs the complete model.
        """
        if cell not in self.slippery or self.slip_prob <= 0.0:
            return [(1.0, self._resolve(cell, action), False)]
        out = [(1.0 - self.slip_prob, self._resolve(cell, action), False)]
        p = self.slip_prob / len(ACTIONS)
        for a in ACTIONS:
            out.append((p, self._resolve(cell, a), True))
        return out

    def sample_cell(self, cell: Cell, action: int, rng: random.Random = random) -> Tuple[Cell, bool]:
        """Sample a single ``(next_cell, slipped)`` from the same distribution.

        Used by the model-free agents, which never see :meth:`cell_transitions`.
        """
        if cell in self.slippery and rng.random() < self.slip_prob:
            action = rng.choice(ACTIONS)
            return self._resolve(cell, action), True
        return self._resolve(cell, action), False
