"""Room 2 UI — Museum Heist solved with SARSA."""

from __future__ import annotations

import random
from typing import Dict

import streamlit as st

from rl.algos import sarsa
from rl.envs.museum import DEFAULT_LAYOUT, MuseumEnv, generate_museum_layout, museum_layout_stats
from ui.common import episode_replay_player, learning_section, train_with_live_progress
from ui.render import render_grid
from ui_canvas.toggle import use_canvas
from ui_canvas.room2_museum import draw_museum_canvas, replay_museum_canvas, frame_to_canvas as _museum_frame


def _draw(env: MuseumEnv, frame: dict):
    alarm_on = frame.get("alarm", False)
    styles: Dict = {}
    for cell in env.slippery:
        styles[cell] = {"color": "#e8d5b7", "icon": "ice", "label_color": "#8d6e63"}
    for cell in getattr(env, "camera_devices", set()):
        styles[cell] = {"color": "#263238", "icon": "camera", "label_color": "#90caf9"}
    cam_color = "#ff8a80" if alarm_on else "#ffcdd2"
    for cell in env.cameras:
        styles[cell] = {"color": cam_color, "label": "!", "label_color": "#c62828"}
    for cell in env.traps:
        styles[cell] = {"color": "#ffab91", "icon": "trap", "label_color": "#bf360c"}
    styles[env.start] = {"color": "#c8e6c9", "icon": "home", "label_color": "#2e7d32"}
    has_d = frame.get("has_diamond", 0)
    styles[env.exit] = {"color": "#7CFC00" if has_d else "#a1887f",
                        "icon": "door", "label_color": "#222"}
    if not has_d:
        styles[env.diamond] = {"color": "#e1f5fe", "icon": "diamond", "label_color": "#0277bd"}

    markers = []
    if has_d:
        markers.append((frame["pos"], "diamond", "#0277bd"))
    guard_color = "#d32f2f" if alarm_on else "#1565c0"
    for guard in frame.get("guards", []):
        markers.append((guard, "police", guard_color))

    title = "💎 Got the diamond — escape!" if has_d else "Sneak to the vault..."
    if alarm_on:
        title = "🚨 ALARM! Guards are rushing!" if not has_d else "🚨 ALARM! Escape NOW!"
    if frame.get("success"):
        title = "🎉 Clean getaway!"
    if frame.get("caught"):
        title = "🚨 Caught by security!"
    if frame.get("alarm_triggered"):
        title = "🚨 Camera triggered the alarm!"
    return render_grid(env.rows, env.cols, walls=env.walls, cell_styles=styles,
                       agent=frame["pos"], agent_color="#2b2b2b",
                       agent_symbol="robber", agent_symbol_color="#f5f5f5",
                       markers=markers,
                       title=title)


def render() -> None:
    st.title("💎 Room 2 — Museum Heist · SARSA")
    st.markdown(
        "You're a thief breaking into a museum. Sneak through the galleries, "
        "steal the 💎 **diamond** from the vault, and escape through the exit. "
        "Watch out for **security cameras** (!), **laser traps**, "
        "**patrol guards**, and **slippery marble floors**. "
        "If a camera spots you, it triggers the **alarm** — "
        "guards speed up and rush to catch you!\n\n"
        "The agent uses **SARSA** (on-policy TD control) to learn the safest "
        "route through the museum."
    )

    with st.sidebar:
        st.header("⚙️ SARSA parameters")
        episodes = st.slider("Episodes", 100, 4000, 1500, 100)
        alpha = st.slider("Learning rate α", 0.01, 1.0, 0.2, 0.01)
        gamma = st.slider("Discount γ", 0.50, 0.999, 0.97, 0.005)
        eps_start = st.slider("ε start", 0.0, 1.0, 1.0, 0.05)
        eps_end = st.slider("ε end", 0.0, 0.5, 0.02, 0.01)
        eps_decay = st.slider("ε decay / episode", 0.90, 1.0, 0.997, 0.001)
        slip_prob = st.slider("Marble slip chance", 0.0, 0.4, 0.1, 0.05)
        max_steps = st.slider("Max steps / episode", 50, 500, 200, 50)
        seed = st.number_input("Random seed", value=1, step=1)
        st.header("Museum layout")
        use_generated = st.checkbox("Generate random museum", value=False, key="room2_gen")
        if "room2_map_seed" not in st.session_state:
            st.session_state["room2_map_seed"] = random.randint(0, 99999)
        if st.button("🎲 New random museum", key="room2_newmap",
                     use_container_width=True, disabled=not use_generated):
            st.session_state["room2_map_seed"] = random.randint(0, 99999)
        map_seed = st.number_input("Map seed", step=1, key="room2_map_seed",
                                   disabled=not use_generated)
        n_cameras = st.slider("Camera zones", 0, 18, 10, disabled=not use_generated)
        n_traps = st.slider("Laser traps", 0, 8, 3, disabled=not use_generated)
        n_slippery = st.slider("Marble floors", 0, 18, 7, disabled=not use_generated)
        n_guards = st.slider("Patrol guards", 0, 5, 2, disabled=not use_generated)
        alarm_enabled = st.checkbox("Camera triggers alarm (guards speed up)", value=True)
        with st.expander("Reward shaping"):
            r_diamond = st.number_input("Steal diamond", value=30.0, step=5.0)
            r_exit = st.number_input("Escape with diamond", value=100.0, step=10.0)
            r_camera = st.number_input("Camera detection", value=-50.0, step=5.0)
            r_trap = st.number_input("Laser trap", value=-15.0, step=5.0)
            r_guard = st.number_input("Caught by guard", value=-50.0, step=5.0)
            r_exit_early = st.number_input("Exit without diamond", value=-5.0, step=1.0)
            r_step = st.number_input("Step cost", value=-1.0, step=1.0)
            r_slip = st.number_input("Slip on marble", value=-5.0, step=1.0)
            r_wall = st.number_input("Wall bump", value=-2.0, step=1.0)
        train = st.button("🎓 Train SARSA", type="primary", use_container_width=True)

    layout = (
        generate_museum_layout(int(map_seed), n_cameras, n_traps, n_slippery, n_guards)
        if use_generated
        else DEFAULT_LAYOUT
    )
    stats = museum_layout_stats(layout)
    env_config = dict(layout=layout, slip_prob=slip_prob, max_steps=max_steps,
                      r_step=r_step, r_diamond=r_diamond, r_exit=r_exit,
                      r_camera=r_camera, r_trap=r_trap, r_guard=r_guard,
                      r_slip=r_slip, r_wall=r_wall, r_exit_early=r_exit_early,
                      alarm_enabled=alarm_enabled, seed=int(seed))

    def make_env() -> MuseumEnv:
        return MuseumEnv(**env_config)

    env = make_env()

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Museum floor plan")
        if use_canvas():
            draw_museum_canvas(env, env.render_state())
        else:
            st.pyplot(_draw(env, env.render_state()), clear_figure=True)
        st.caption(f"{len(env.cameras)} camera zones · "
                   f"{len(env.traps)} traps · {len(env.slippery)} marble tiles · "
                   f"{len(env.guard_starts)} guards")

    if train:
        result = train_with_live_progress(
            lambda cb: sarsa.train(make_env(), episodes=episodes, alpha=alpha, gamma=gamma,
                                   eps_start=eps_start, eps_end=eps_end,
                                   eps_decay=eps_decay, max_steps=max_steps,
                                   seed=int(seed), progress_cb=cb)
        )
        st.session_state["room2"] = {"result": result, "env_config": env_config}

    store = st.session_state.get("room2")
    if isinstance(store, dict) and "result" in store:
        result = store["result"]
        trained_env = MuseumEnv(**store["env_config"])
    else:
        result = store
        trained_env = env
    with right:
        st.subheader("Heist result")
        if not result:
            st.info("Set parameters and press **Train SARSA** to plan the heist.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Escape rate (last 100)", f"{result.success_rate(100):.0%}")
            m2.metric("Mean reward (last 100)",
                      f"{sum(result.episode_rewards[-100:]) / min(100, result.num_episodes):.0f}")
            fin = result.snapshots.get("Final")
            if fin:
                m3.metric("Greedy heist", f"{fin[-1]['step']} steps")

    if result:
        st.subheader("📈 Learning & exploration graphs")
        learning_section(result)
        st.subheader("🎞️ Replay any heist attempt")
        if use_canvas() and result.episode_replays:
            n = len(result.episode_replays)
            ep = st.slider("Episode", 1, n, n, key="room2_ep_sel") - 1
            raw = result.episode_replays[ep]
            if raw:
                replay_museum_canvas(trained_env, raw, key=f"room2_cv_{ep}")
        else:
            episode_replay_player(
                result,
                lambda f: _draw(trained_env, f),
                key="room2_all_episodes",
                caption="Pick any training episode — watch the robber sneak, get caught, or escape cleanly.",
            )
