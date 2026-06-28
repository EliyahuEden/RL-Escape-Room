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


_CSS = """
<style>
.block-container { padding-top: 1.3rem; max-width: 1250px; }
h1, h2, h3 { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
h2 { border-bottom: 2px solid #e6e6f2; padding-bottom: .25rem; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#f7f6ff,#eef0fb); }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2 { font-size: 1.1rem; }
.stButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid #d4d4e8; transition: .15s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(106,90,205,.22); }
.stButton > button[kind="primary"] {
  background: linear-gradient(90deg,#6a5acd,#7e57c2); border: none; color: #fff;
}
[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #ededf7; border-radius: 14px;
  padding: 12px 16px; box-shadow: 0 2px 10px rgba(60,60,120,.07);
}
.hero {
  background: linear-gradient(120deg,#6a5acd,#7e57c2 55%,#26a69a);
  border-radius: 18px; padding: 24px 30px; margin-bottom: 18px; color: #fff;
  box-shadow: 0 10px 30px rgba(106,90,205,.30);
}
.hero h1 { color: #fff !important; margin: 0; font-size: 2.05rem; }
.hero p { color: #f1eeff; margin: .5rem 0 0; font-size: 1.03rem; max-width: 760px; }
.room-grid {
  display: grid; grid-template-columns: repeat(auto-fit,minmax(235px,1fr));
  gap: 14px; margin-top: 6px;
}
.room-card {
  background: #fff; border-radius: 16px; padding: 16px 18px; border: 1px solid #ededf7;
  box-shadow: 0 4px 16px rgba(60,60,120,.08); transition: .18s;
}
.room-card:hover { transform: translateY(-3px); box-shadow: 0 12px 26px rgba(106,90,205,.18); }
.room-card .emoji { font-size: 1.9rem; }
.room-card h4 { margin: .35rem 0 .25rem; }
.algo-tag {
  display: inline-block; color: #fff; border-radius: 999px; padding: 2px 10px;
  font-size: .72rem; font-weight: 700; letter-spacing: .02em;
}
.room-card p { color: #555; font-size: .88rem; margin: .25rem 0 0; line-height: 1.35; }
</style>
"""

_ROOM_CARDS = [
    ("🟡", "Room 1 · Pacman", "Dynamic Programming", "#f5b301",
     "Plan the optimal path in a fully-known maze: collect every coin, dodge the guard, escape."),
    ("💎", "Room 2 · Museum Heist", "SARSA", "#7e57c2",
     "Steal the diamond and slip past cameras and patrol guards — cautious on-policy learning."),
    ("🏎️", "Room 3 · Racing", "Q-Learning", "#e53935",
     "Race to the finish: gamble on the risky oil short-cut or play the long safe line."),
    ("⚽", "Room 4 · Football", "DQN", "#2e7d32",
     "Continuous control: dribble past defenders, time the shot, curve the ball past the keeper."),
    ("🚧", "Room 5 · Obstacles", "DQN + sensors", "#455a64",
     "Navigate dynamic obstacles with look-ahead sensors, then test on a brand-new random room."),
]


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def overview() -> None:
    st.markdown(
        '<div class="hero"><h1>🚪 Reinforcement-Learning Escape Room</h1>'
        '<p>Five themed rooms, five RL algorithms. Tune the hyper-parameters, train the '
        'agent with live learning graphs, and replay exactly what it learned at every stage.</p></div>',
        unsafe_allow_html=True,
    )
    cards = "".join(
        f'<div class="room-card"><div class="emoji">{emoji}</div>'
        f'<span class="algo-tag" style="background:{color}">{algo}</span>'
        f'<h4>{title}</h4><p>{desc}</p></div>'
        for emoji, title, algo, color, desc in _ROOM_CARDS
    )
    st.markdown(f'<div class="room-grid">{cards}</div>', unsafe_allow_html=True)
    st.write("")
    st.info("Pick a room from the sidebar ⬅️ — each is self-contained with its own "
            "parameters, graphs and episode replay. Room 1 is the gentlest start.")


def main() -> None:
    inject_css()
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
