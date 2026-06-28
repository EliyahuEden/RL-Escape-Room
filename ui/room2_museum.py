"""Room 2 UI — Museum Heist solved with SARSA."""

from __future__ import annotations

import random
from typing import Dict

import streamlit as st

from rl.algos import sarsa
from rl.envs.museum import LAYOUTS, MuseumEnv, generate_museum_layout
from ui.common import episode_replay_player, learning_section, train_with_live_progress
from ui.render import render_grid


def _draw(env: MuseumEnv, frame: dict):
    styles: Dict = {}
    for cell in env.slippery:
        styles[cell] = {"color": "#bfe3ff", "icon": "ice", "label_color": "#1565c0"}
    for cell in env.cameras:
        styles[cell] = {"color": "#ff9aa2", "icon": "camera", "label_color": "#7a0000"}
    for cell in env.traps:
        styles[cell] = {"color": "#ffd9a0", "icon": "trap", "label_color": "#7a4a00"}
    styles[env.start] = {"color": "#d7f0d7", "icon": "home", "label_color": "#2e7d32"}
    has_d = frame.get("has_diamond", 0)
    styles[env.exit] = {"color": "#7CFC00" if has_d else "#d9a441",
                        "icon": "door", "label_color": "#222"}
    if not has_d:
        styles[env.diamond] = {"color": "#fff3b0", "icon": "diamond", "label_color": "#0369a1"}

    markers = []
    if has_d:
        markers.append((frame["pos"], "diamond", "#0369a1"))
    for guard in frame.get("guards", []):
        markers.append((guard, "police", "#1565c0"))
    title = "Diamond: ✓ — escaping" if has_d else "Diamond: ✗ — go steal it"
    if frame.get("success"):
        title = "Clean getaway!"
    if frame.get("caught"):
        title = "Caught by a guard!"
    return render_grid(env.rows, env.cols, walls=env.walls, cell_styles=styles,
                       agent=frame["pos"], agent_color="#2b2b2b",
                       agent_symbol="robber", agent_symbol_color="#f5f5f5",
                       markers=markers,
                       title=title)


def render() -> None:
    st.title("💎 Room 2 — Museum Heist · SARSA")
    st.markdown(
        "Steal the 💎, then escape through the exit. **Camera vision zones** cost a "
        "penalty each step you're seen (you keep moving), **traps** hurt, and a "
        "**patrol guard** that catches you **ends the heist** (big penalty). There's "
        "a short dangerous route past the cameras/guards and a longer safe one — "
        "SARSA is **on-policy**, so it accounts for its own ε-greedy exploration and "
        "learns the cautious path."
    )

    with st.sidebar:
        st.header("⚙️ SARSA parameters")
        map_mode = st.selectbox("Map source", ["Generated", "Fixed layout"], key="room2_map_mode")
        layout_name = st.selectbox("Fixed layout", list(LAYOUTS), index=0,
                                   disabled=map_mode == "Generated")
        hard = map_mode == "Generated" or layout_name.startswith("Museum maze")
        episodes = st.slider("Episodes", 100, 4000, 1500 if hard else 800, 100)
        alpha = st.slider("Learning rate α", 0.01, 1.0, 0.2, 0.01)
        gamma = st.slider("Discount γ", 0.50, 0.999, 0.97 if hard else 0.95, 0.005)
        eps_start = st.slider("ε start", 0.0, 1.0, 1.0, 0.05)
        eps_end = st.slider("ε end", 0.0, 0.5, 0.02 if hard else 0.05, 0.01)
        eps_decay = st.slider("ε decay / episode", 0.90, 1.0, 0.997 if hard else 0.99, 0.001)
        slip_prob = st.slider("Slip probability", 0.0, 0.4, 0.05, 0.05)
        max_steps = st.slider("Max steps / episode", 50, 500, 200, 50)
        seed = st.number_input("Random seed", value=1, step=1)
        st.header("Map (random each time)")
        if "room2_map_seed" not in st.session_state:
            st.session_state["room2_map_seed"] = random.randint(0, 99999)
        if st.button("🎲 New random layout", key="room2_newmap",
                     use_container_width=True, disabled=map_mode != "Generated"):
            st.session_state["room2_map_seed"] = random.randint(0, 99999)
        map_seed = st.number_input("Map seed", step=1, key="room2_map_seed")
        n_cameras = st.slider("Camera cells", 0, 18, 10, disabled=map_mode != "Generated")
        n_traps = st.slider("Traps", 0, 8, 3, disabled=map_mode != "Generated")
        n_slippery = st.slider("Slippery tiles", 0, 18, 7, disabled=map_mode != "Generated")
        n_guards = st.slider("Patrol guards", 0, 5, 2, disabled=map_mode != "Generated")
        with st.expander("Reward shaping"):
            r_diamond = st.number_input("Diamond", value=30.0, step=5.0)
            r_exit = st.number_input("Escape with diamond", value=120.0, step=10.0)
            r_camera = st.number_input("Camera vision zone", value=-25.0, step=5.0)
            r_trap = st.number_input("Trap", value=-20.0, step=5.0)
            r_guard = st.number_input("Caught by guard (ends run)", value=-50.0, step=5.0)
            r_exit_early = st.number_input("Exit without diamond", value=-10.0, step=1.0)
            r_step = st.number_input("Step cost", value=-1.0, step=1.0)
            r_slip = st.number_input("Slip", value=-5.0, step=1.0)
            r_wall = st.number_input("Wall hit", value=-5.0, step=1.0)
        train = st.button("🎓 Train SARSA", type="primary", use_container_width=True)

    layout = (
        generate_museum_layout(int(map_seed), n_cameras, n_traps, n_slippery, n_guards)
        if map_mode == "Generated"
        else LAYOUTS[layout_name]
    )
    env_config = dict(layout=layout, slip_prob=slip_prob, max_steps=max_steps,
                      r_step=r_step, r_diamond=r_diamond, r_exit=r_exit,
                      r_camera=r_camera, r_trap=r_trap, r_guard=r_guard,
                      r_slip=r_slip, r_wall=r_wall, r_exit_early=r_exit_early,
                      seed=int(seed))

    def make_env() -> MuseumEnv:
        return MuseumEnv(**env_config)

    env = make_env()

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Museum")
        st.pyplot(_draw(env, env.render_state()), clear_figure=True)
        st.caption("Every map has a short route through the danger + a longer safe one · "
                   f"{len(env.cameras)} camera cells · {len(env.traps)} trap(s) · "
                   f"{len(env.slippery)} icy cells · {len(env.guard_starts)} patrol guard(s)")

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
        st.subheader("🎞️ Replay any training episode")
        episode_replay_player(
            result,
            lambda f: _draw(trained_env, f),
            key="room2_all_episodes",
            caption="Pick any SARSA training episode, including exploratory failures and successful runs.",
        )
