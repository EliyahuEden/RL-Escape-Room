"""Shared rendering helpers (Matplotlib) used by the room UI modules.

* :func:`render_grid`   -- draws one frame of a 10x10 grid room (Rooms 1-3).
* :func:`field_axes`    -- sets up a metres-based pitch for the continuous
                           rooms (Rooms 4-5); the rooms draw players/obstacles
                           on top.
* :func:`metrics_frame` -- turns a :class:`~rl.utils.TrainResult` into a tidy
                           ``pandas.DataFrame`` for the learning-curve charts.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend; Streamlit renders the Figure object
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Rectangle

from rl.utils import TrainResult, moving_average

Cell = Tuple[int, int]


def render_grid(
    rows: int,
    cols: int,
    *,
    walls: Iterable[Cell] = (),
    cell_styles: Optional[Dict[Cell, dict]] = None,
    agent: Optional[Cell] = None,
    agent_color: str = "#FFD23F",
    markers: Iterable[Tuple[Cell, str, str]] = (),
    title: str = "",
    figsize: Tuple[float, float] = (4.8, 4.8),
):
    """Render one grid frame and return the Matplotlib ``Figure``.

    ``cell_styles`` maps a cell to ``{"color": facecolor, "label": str,
    "label_color": str}``.  ``markers`` is a list of ``(cell, symbol, color)``
    drawn as centred text (handy for moving enemies / dynamic items).
    """
    cell_styles = cell_styles or {}
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at the top, like a screen
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)

    walls = set(walls)
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if cell in walls:
                face = "#222831"
            else:
                face = cell_styles.get(cell, {}).get("color", "#f4f4f4")
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=face,
                                   edgecolor="#cfcfcf", linewidth=0.6))
            label = cell_styles.get(cell, {}).get("label")
            if label:
                ax.text(c + 0.5, r + 0.5, label, ha="center", va="center",
                        fontsize=12,
                        color=cell_styles[cell].get("label_color", "#222"))

    for cell, symbol, color in markers:
        ax.text(cell[1] + 0.5, cell[0] + 0.5, symbol, ha="center", va="center",
                fontsize=15, color=color, fontweight="bold")

    if agent is not None:
        ax.add_patch(Circle((agent[1] + 0.5, agent[0] + 0.5), 0.34,
                            facecolor=agent_color, edgecolor="#333", linewidth=1.2,
                            zorder=5))
    fig.tight_layout(pad=0.3)
    return fig


def field_axes(width: float = 10.0, height: float = 10.0,
               figsize: Tuple[float, float] = (5.2, 5.2), grass: str = "#2e7d32"):
    """Create a metres-based pitch and return ``(fig, ax)``."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(Rectangle((0, 0), width, height, facecolor=grass, zorder=0))
    return fig, ax


def metrics_frame(result: TrainResult, smooth_window: int = 50) -> pd.DataFrame:
    """Tidy per-episode metrics with a smoothed reward column for plotting."""
    n = result.num_episodes
    data = {
        "episode": list(range(1, n + 1)),
        "reward": result.episode_rewards,
        "reward_smoothed": list(moving_average(result.episode_rewards, smooth_window)),
        "steps": result.episode_steps,
        "epsilon": result.epsilon,
        "success": [int(s) for s in result.episode_success],
    }
    df = pd.DataFrame(data).set_index("episode")
    df["success_rate"] = df["success"].rolling(smooth_window, min_periods=1).mean()
    return df
