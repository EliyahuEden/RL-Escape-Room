"""RL Escape Room — Streamlit entry point.

Run with::

    streamlit run app.py

The agent escapes a sequence of themed rooms, each driven by a different
reinforcement-learning algorithm:

    Room 1  Pacman          Dynamic Programming   (known model)
    Room 2  Museum Heist    SARSA                 (on-policy, cautious)
    Room 3  Racing          Q-Learning            (off-policy, greedy)
    Room 4  Football        DQN                   (continuous state)
    Room 5  Obstacles       DQN + sensors         (dynamic obstacles)

Each room module exposes a ``render()`` function.  Rooms are imported lazily so
a problem in one room never prevents the others from loading.
"""

from __future__ import annotations

import importlib
import sys
import traceback

import streamlit as st

st.set_page_config(page_title="RL Escape Room", page_icon="🚪", layout="wide")

# label -> module path under ui/
ROOMS = {
    "🏠 Overview": None,
    "🟡 Room 1 — Pacman · Dynamic Programming": "ui.room1_pacman",
    "💎 Room 2 — Museum Heist · SARSA": "ui.room2_museum",
    "🏎️ Room 3 — Racing · Q-Learning": "ui.room3_racing",
    "⚽ Room 4 — Football · DQN": "ui.room4_football",
    "🚧 Room 5 — Obstacles · DQN + sensors": "ui.room5_obstacles",
}

SHARED_MODULES = [
    "ui.render",
    "ui.common",
]


def overview() -> None:
    st.title("🚪 Reinforcement-Learning Escape Room")
    st.markdown(
        """
Escape five themed rooms — each one a different RL problem and algorithm.
Use the sidebar to enter a room, tune its hyper-parameters, **train** the agent,
watch the **learning graphs**, and **replay** episodes from different stages of
training to see what the policy learned.
        """
    )
    st.subheader("The rooms")
    st.table(
        {
            "Room": ["1 · Pacman", "2 · Museum Heist", "3 · Racing", "4 · Football", "5 · Obstacles"],
            "Algorithm": ["Dynamic Programming", "SARSA", "Q-Learning", "DQN", "DQN + sensors"],
            "Model": ["Known", "Unknown", "Unknown", "Unknown (continuous)", "Unknown (continuous)"],
            "Main task": [
                "Collect all coins, then exit",
                "Steal the diamond, avoid cameras/traps, escape",
                "Reach the finish line fast (boosters, mud, oil)",
                "Dodge defenders & the keeper, then score",
                "Navigate dynamic obstacles to the exit",
            ],
        }
    )
    st.info("Each room is self-contained: its hyper-parameters, graphs and replay "
            "live on its own page. Start with Room 1 on the left. ⬅️")


def main() -> None:
    st.sidebar.title("🚪 Escape Room")
    choice = st.sidebar.radio("Select a room", list(ROOMS.keys()), label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption("Tune parameters → Train → inspect graphs → replay episodes.")

    module_path = ROOMS[choice]
    if module_path is None:
        overview()
        return

    try:
        for shared_module in SHARED_MODULES:
            if shared_module in sys.modules:
                importlib.reload(sys.modules[shared_module])
        module = importlib.import_module(module_path)
        importlib.reload(module)  # pick up edits during development
        module.render()
    except Exception:  # pragma: no cover - surfaced in the UI
        st.error(f"Room failed to load ({module_path}).")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
