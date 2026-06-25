"""Room 2 UI — Museum Heist solved with SARSA."""

from __future__ import annotations

from typing import Dict

import streamlit as st

from rl.algos import sarsa
from rl.envs.museum import LAYOUTS, MuseumEnv
from ui.common import learning_section, replay_player, train_with_live_progress
from ui.render import render_grid


def _draw(env: MuseumEnv, frame: dict):
    styles: Dict = {}
    for cell in env.slippery:
        styles[cell] = {"color": "#bfe3ff", "label": "❄", "label_color": "#1565c0"}
    for cell in env.cameras:
        styles[cell] = {"color": "#ff9aa2", "label": "📷", "label_color": "#7a0000"}
    for cell in env.traps:
        styles[cell] = {"color": "#ffd9a0", "label": "⚠", "label_color": "#7a4a00"}
    styles[env.start] = {"color": "#d7f0d7", "label": "S", "label_color": "#2e7d32"}
    has_d = frame.get("has_diamond", 0)
    styles[env.exit] = {"color": "#7CFC00" if has_d else "#d9a441",
                        "label": "EXIT", "label_color": "#222"}
    if not has_d:
        styles[env.diamond] = {"color": "#fff3b0", "label": "💎", "label_color": "#b8860b"}

    markers = []
    if has_d:
        markers.append((frame["pos"], "💎", "#b8860b"))
    title = "Diamond: ✓ — escaping" if has_d else "Diamond: ✗ — go steal it"
    if frame.get("success"):
        title = "🏆 Clean getaway!"
    return render_grid(env.rows, env.cols, walls=env.walls, cell_styles=styles,
                       agent=frame["pos"], agent_color="#2b2b2b", markers=markers,
                       title=title)


def render() -> None:
    st.title("💎 Room 2 — Museum Heist · SARSA")
    st.markdown(
        "Steal the 💎, dodge the cameras 📷 and traps ⚠, then escape. The bottom "
        "row of cameras is a **cliff**: step in and the alarm drags you back to "
        "start. SARSA is **on-policy**, so it accounts for its own ε-greedy "
        "exploration and learns a path with a safety margin from the cliff."
    )

    with st.sidebar:
        st.header("⚙️ SARSA parameters")
        layout_name = st.selectbox("Layout", list(LAYOUTS), index=0)
        layout = LAYOUTS[layout_name]
        hard = layout_name.startswith("Museum maze")
        episodes = st.slider("Episodes", 100, 4000, 1500 if hard else 800, 100)
        alpha = st.slider("Learning rate α", 0.01, 1.0, 0.2, 0.01)
        gamma = st.slider("Discount γ", 0.50, 0.999, 0.97 if hard else 0.95, 0.005)
        eps_start = st.slider("ε start", 0.0, 1.0, 1.0, 0.05)
        eps_end = st.slider("ε end", 0.0, 0.5, 0.02 if hard else 0.05, 0.01)
        eps_decay = st.slider("ε decay / episode", 0.90, 1.0, 0.997 if hard else 0.99, 0.001)
        slip_prob = st.slider("Slip probability", 0.0, 0.4, 0.05, 0.05)
        max_steps = st.slider("Max steps / episode", 50, 500, 200, 50)
        seed = st.number_input("Random seed", value=1, step=1)
        with st.expander("Reward shaping"):
            r_diamond = st.number_input("Diamond", value=30.0, step=5.0)
            r_exit = st.number_input("Exit", value=100.0, step=10.0)
            r_camera = st.number_input("Camera (alarm)", value=-25.0, step=5.0)
            r_trap = st.number_input("Trap", value=-20.0, step=5.0)
            r_step = st.number_input("Step cost", value=-1.0, step=1.0)
            r_slip = st.number_input("Slip", value=-5.0, step=1.0)
        train = st.button("🎓 Train SARSA", type="primary", use_container_width=True)

    env = MuseumEnv(layout=layout, slip_prob=slip_prob, max_steps=max_steps,
                    r_step=r_step, r_diamond=r_diamond, r_exit=r_exit,
                    r_camera=r_camera, r_trap=r_trap, r_slip=r_slip)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Museum")
        st.pyplot(_draw(env, {"pos": env.start, "has_diamond": 0}), clear_figure=True)
        st.caption(f"{len(env.cameras)} camera cells · {len(env.traps)} trap(s) · "
                   f"{len(env.slippery)} icy cells")

    if train:
        result = train_with_live_progress(
            lambda cb: sarsa.train(env, episodes=episodes, alpha=alpha, gamma=gamma,
                                   eps_start=eps_start, eps_end=eps_end,
                                   eps_decay=eps_decay, max_steps=max_steps,
                                   seed=int(seed), progress_cb=cb)
        )
        st.session_state["room2"] = result

    result = st.session_state.get("room2")
    with right:
        st.subheader("Result")
        if not result:
            st.info("Set parameters and press **Train SARSA**.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Success rate (last 100)", f"{result.success_rate(100):.0%}")
            m2.metric("Mean reward (last 100)",
                      f"{sum(result.episode_rewards[-100:]) / min(100, result.num_episodes):.0f}")
            fin = result.snapshots.get("Final")
            if fin:
                m3.metric("Greedy escape", f"{fin[-1]['step']} steps")

    if result:
        st.subheader("📈 Learning & exploration graphs")
        learning_section(result)
        st.subheader("🎬 Replay episodes from different training stages")
        replay_player(result.snapshots, lambda f: _draw(env, f), key="room2",
                      caption="Early vs Mid vs Final greedy policy.")
