"""Shared rendering helpers (Matplotlib) used by the room UI modules.

* :func:`render_grid`   -- draws one frame of a 10x10 grid room (Rooms 1-3).
* :func:`field_axes`    -- sets up a metres-based pitch for the continuous
                           rooms (Rooms 4-5); each room draws its entities on
                           top.
* :func:`metrics_frame` -- turns a :class:`~rl.utils.TrainResult` into a tidy
                           ``pandas.DataFrame`` for the learning-curve charts.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend; Streamlit renders the Figure object
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle, Wedge

from rl.utils import TrainResult, moving_average

Cell = Tuple[int, int]


def _cell_center(cell: Cell) -> Tuple[float, float]:
    return cell[1] + 0.5, cell[0] + 0.5


def _draw_icon(ax, cell: Cell, icon: str, color: str = "#222", size: float = 1.0) -> bool:
    """Draw a small vector icon inside a grid cell. Returns True if handled."""
    x, y = _cell_center(cell)
    s = 0.34 * size

    if icon.startswith("pacman"):
        if icon == "pacman_closed":
            ax.add_patch(Circle((x, y), s, facecolor="#ffd23f", edgecolor="#333", lw=1.2, zorder=6))
            return True
        angles = {
            "pacman_up": (125, 55),
            "pacman_down": (305, 235),
            "pacman_left": (215, 145),
            "pacman_right": (35, 325),
        }.get(icon, (35, 325))
        ax.add_patch(Wedge((x, y), s, angles[0], angles[1],
                           facecolor="#ffd23f", edgecolor="#333", lw=1.2, zorder=6))
        ax.add_patch(Circle((x, y - 0.12), 0.035, facecolor="#111", edgecolor="none", zorder=7))
        return True

    if icon == "ghost":
        ax.add_patch(Circle((x, y - 0.08), 0.24 * size, facecolor=color, edgecolor="#5d1010", lw=1, zorder=6))
        ax.add_patch(Rectangle((x - 0.24 * size, y - 0.08), 0.48 * size, 0.34 * size,
                               facecolor=color, edgecolor="#5d1010", lw=1, zorder=6))
        for ex in (x - 0.09, x + 0.09):
            ax.add_patch(Circle((ex, y - 0.09), 0.055, facecolor="white", edgecolor="none", zorder=7))
            ax.add_patch(Circle((ex + 0.015, y - 0.085), 0.022, facecolor="#111", edgecolor="none", zorder=8))
        return True

    if icon == "coin":
        ax.add_patch(Circle((x, y), 0.13 * size, facecolor="#f5b301", edgecolor="#8a6100", lw=1, zorder=5))
        return True

    if icon == "door":
        ax.add_patch(Rectangle((x - 0.17, y - 0.28), 0.34, 0.56, facecolor="#8d5524",
                               edgecolor="#3e2723", lw=1, zorder=5))
        ax.add_patch(Circle((x + 0.09, y), 0.025, facecolor="#ffd54f", edgecolor="none", zorder=6))
        return True

    if icon == "home":
        ax.add_patch(Rectangle((x - 0.19, y - 0.02), 0.38, 0.27, facecolor="#66bb6a",
                               edgecolor="#1b5e20", lw=1, zorder=5))
        ax.add_patch(Polygon([(x - 0.24, y - 0.02), (x, y - 0.27), (x + 0.24, y - 0.02)],
                             closed=True, facecolor="#2e7d32", edgecolor="#1b5e20", lw=1, zorder=6))
        return True

    if icon == "ice":
        for dx, dy in ((0.22, 0), (0, 0.22), (0.16, 0.16), (0.16, -0.16)):
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color="#1565c0", lw=1.4, zorder=5)
        return True

    if icon == "diamond":
        ax.add_patch(Polygon([(x, y - 0.26), (x + 0.25, y), (x, y + 0.26), (x - 0.25, y)],
                             closed=True, facecolor="#7dd3fc", edgecolor="#0369a1", lw=1.2, zorder=6))
        ax.plot([x - 0.16, x + 0.16], [y, y], color="white", lw=1, alpha=0.7, zorder=7)
        return True

    if icon == "camera":
        ax.add_patch(Rectangle((x - 0.24, y - 0.16), 0.48, 0.32, facecolor="#263238",
                               edgecolor="#000", lw=1, zorder=5))
        ax.add_patch(Rectangle((x - 0.14, y - 0.25), 0.2, 0.09, facecolor="#455a64",
                               edgecolor="#000", lw=1, zorder=5))
        ax.add_patch(Circle((x, y), 0.105, facecolor="#90caf9", edgecolor="#111", lw=1, zorder=6))
        return True

    if icon == "trap":
        ax.add_patch(Ellipse((x, y), 0.52, 0.32, facecolor="#212121", edgecolor="#000", lw=1, zorder=5))
        ax.add_patch(Ellipse((x, y - 0.03), 0.33, 0.16, facecolor="#424242", edgecolor="none", zorder=6))
        return True

    if icon == "police":
        ax.add_patch(Circle((x, y - 0.12), 0.13, facecolor="#f4c7a1", edgecolor="#5d4037", lw=1, zorder=6))
        ax.add_patch(Rectangle((x - 0.14, y - 0.24), 0.28, 0.08, facecolor="#1565c0",
                               edgecolor="#0d47a1", lw=1, zorder=7))
        ax.add_patch(Rectangle((x - 0.2, y + 0.01), 0.4, 0.25, facecolor="#1565c0",
                               edgecolor="#0d47a1", lw=1, zorder=6))
        return True

    if icon == "robber":
        ax.add_patch(Circle((x, y - 0.13), 0.14, facecolor="#f4c7a1", edgecolor="#111", lw=1, zorder=6))
        ax.add_patch(Rectangle((x - 0.13, y - 0.17), 0.26, 0.055, facecolor="#111", edgecolor="none", zorder=7))
        ax.add_patch(Rectangle((x - 0.21, y + 0.02), 0.42, 0.25, facecolor="#212121",
                               edgecolor="#000", lw=1, zorder=6))
        ax.plot([x - 0.18, x + 0.18], [y + 0.11, y + 0.11], color="white", lw=1, zorder=7)
        return True

    if icon == "car":
        ax.add_patch(Rectangle((x - 0.26, y - 0.16), 0.52, 0.32, facecolor=color,
                               edgecolor="#7f1d1d", lw=1, zorder=6))
        ax.add_patch(Rectangle((x - 0.09, y - 0.24), 0.18, 0.1, facecolor="#bbdefb",
                               edgecolor="#0d47a1", lw=0.8, zorder=7))
        for wx in (x - 0.17, x + 0.17):
            ax.add_patch(Circle((wx, y + 0.18), 0.055, facecolor="#111", edgecolor="none", zorder=7))
        return True

    if icon == "oil":
        ax.add_patch(Ellipse((x, y), 0.52, 0.28, angle=-15, facecolor="#111",
                             edgecolor="#000", lw=1, zorder=5))
        ax.add_patch(Ellipse((x + 0.07, y - 0.03), 0.2, 0.08, angle=-15,
                             facecolor="#546e7a", edgecolor="none", zorder=6))
        return True

    if icon == "mud":
        ax.add_patch(Ellipse((x, y), 0.55, 0.33, facecolor="#795548",
                             edgecolor="#4e342e", lw=1, zorder=5))
        return True

    if icon == "boost":
        ax.add_patch(Polygon([(x - 0.04, y - 0.3), (x + 0.16, y - 0.04),
                              (x + 0.03, y - 0.04), (x + 0.1, y + 0.3),
                              (x - 0.16, y + 0.02), (x - 0.02, y + 0.02)],
                             closed=True, facecolor="#ffd54f", edgecolor="#f57f17", lw=1, zorder=6))
        return True

    if icon == "flag":
        ax.plot([x - 0.18, x - 0.18], [y + 0.28, y - 0.28], color="#333", lw=1.2, zorder=6)
        for rr in range(2):
            for cc in range(3):
                face = "#111" if (rr + cc) % 2 == 0 else "white"
                ax.add_patch(Rectangle((x - 0.16 + cc * 0.105, y - 0.28 + rr * 0.1),
                                       0.105, 0.1, facecolor=face, edgecolor="#111", lw=0.3, zorder=6))
        return True

    if icon == "runner":
        ax.add_patch(Circle((x, y - 0.16), 0.08, facecolor="#f4c7a1", edgecolor="#5d4037", lw=0.8, zorder=7))
        ax.plot([x, x], [y - 0.08, y + 0.12], color=color, lw=2, zorder=7)
        ax.plot([x, x - 0.16], [y + 0.02, y + 0.18], color=color, lw=2, zorder=7)
        ax.plot([x, x + 0.16], [y + 0.02, y + 0.16], color=color, lw=2, zorder=7)
        return True

    if icon == "rock":
        ax.add_patch(Polygon([(x - 0.2, y + 0.16), (x - 0.26, y - 0.03),
                              (x - 0.1, y - 0.22), (x + 0.15, y - 0.2),
                              (x + 0.26, y + 0.02), (x + 0.1, y + 0.2)],
                             closed=True, facecolor=color, edgecolor="#263238", lw=1, zorder=5))
        return True

    if icon == "ball":
        ax.add_patch(Circle((x, y), 0.14, facecolor="white", edgecolor="#222", lw=1, zorder=7))
        ax.add_patch(Circle((x, y), 0.055, facecolor="#222", edgecolor="none", zorder=8))
        return True

    return False


def render_grid(
    rows: int,
    cols: int,
    *,
    walls: Iterable[Cell] = (),
    cell_styles: Optional[Dict[Cell, dict]] = None,
    agent: Optional[Cell] = None,
    agent_color: str = "#FFD23F",
    agent_symbol: Optional[str] = None,
    agent_symbol_color: str = "#111",
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
            icon = cell_styles.get(cell, {}).get("icon")
            if icon:
                _draw_icon(ax, cell, icon,
                           cell_styles[cell].get("label_color", "#222"),
                           cell_styles[cell].get("icon_size", 1.0))
            label = cell_styles.get(cell, {}).get("label")
            if label and not icon:
                ax.text(c + 0.5, r + 0.5, label, ha="center", va="center",
                        fontsize=12,
                        color=cell_styles[cell].get("label_color", "#222"))

    for marker in markers:
        cell, symbol, color = marker[:3]
        if _draw_icon(ax, cell, symbol, color):
            continue
        fontsize = marker[3] if len(marker) > 3 else 15
        ax.text(cell[1] + 0.5, cell[0] + 0.5, symbol, ha="center", va="center",
                fontsize=fontsize, color=color, fontweight="bold")

    if agent is not None:
        if agent_symbol and _draw_icon(ax, agent, agent_symbol, agent_color):
            pass
        else:
            ax.add_patch(Circle((agent[1] + 0.5, agent[0] + 0.5), 0.34,
                                facecolor=agent_color, edgecolor="#333", linewidth=1.2,
                                zorder=5))
        if agent_symbol and agent_symbol not in {
            "pacman_closed", "pacman_up", "pacman_down", "pacman_left", "pacman_right",
            "robber", "car", "runner",
        }:
            ax.text(agent[1] + 0.5, agent[0] + 0.5, agent_symbol,
                    ha="center", va="center", fontsize=19,
                    color=agent_symbol_color, fontweight="bold", zorder=6)
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
