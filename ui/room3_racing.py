"""Room 3 UI — Racing Track solved with Q-Learning (optionally compared with SARSA)."""

from __future__ import annotations

import random
import time
from typing import Dict

import pandas as pd
import streamlit as st

from rl.algos import qlearning, sarsa
from rl.envs.racing import DEFAULT_LAYOUT, RacingEnv, generate_racing_layout
from ui.common import episode_replay_player, learning_section, train_with_live_progress
from ui.render import render_grid
from rl.utils import moving_average


def _draw(env: RacingEnv, frame: dict):
    styles: Dict = {}
    for cell in env.oil:
        styles[cell] = {"color": "#444", "icon": "oil", "label_color": "#111"}
    for cell in env.mud:
        styles[cell] = {"color": "#8d6e63", "icon": "mud", "label_color": "#795548"}
    for cell in env.remaining_boosters(frame.get("bmask", 0)):
        styles[cell] = {"color": "#80deea", "icon": "boost", "label_color": "#f57f17"}
    for cell in getattr(env, "crash", set()):
        styles[cell] = {"color": "#ef5350", "label": "✕", "label_color": "#fff"}
    styles[env.start] = {"color": "#d7f0d7", "icon": "flag", "label_color": "#2e7d32"}
    styles[env.finish] = {"color": "#7CFC00", "icon": "flag", "label_color": "#222"}
    title = "Crashed!" if frame.get("crash") else ("Finished!" if frame.get("success") else "On the track")
    return render_grid(env.rows, env.cols, walls=env.walls, cell_styles=styles,
                       agent=frame["pos"], agent_color="#e53935",
                       agent_symbol="car", agent_symbol_color="#fff",
                       title=title)


def _track_styles(env: RacingEnv) -> Dict:
    styles: Dict = {}
    for cell in env.oil:
        styles[cell] = {"color": "#444", "icon": "oil", "label_color": "#111"}
    for cell in env.mud:
        styles[cell] = {"color": "#8d6e63", "icon": "mud", "label_color": "#795548"}
    for cell in env.boosters:
        styles[cell] = {"color": "#80deea", "icon": "boost", "label_color": "#f57f17"}
    for cell in getattr(env, "crash", set()):
        styles[cell] = {"color": "#ef5350", "label": "✕", "label_color": "#fff"}
    styles[env.start] = {"color": "#d7f0d7", "icon": "flag", "label_color": "#2e7d32"}
    styles[env.finish] = {"color": "#7CFC00", "icon": "flag", "label_color": "#222"}
    return styles


def _draw_race(env: RacingEnv, qf: dict, sf: dict):
    """Draw both cars on the same track: 🔴 Q-Learning, 🔵 SARSA."""
    markers = [(qf["pos"], "car", "#e53935"), (sf["pos"], "car", "#1565c0")]
    return render_grid(env.rows, env.cols, walls=env.walls, cell_styles=_track_styles(env),
                       agent=None, markers=markers, title="🔴 Q-Learning  vs  🔵 SARSA")


def _race_player(env: RacingEnv, qtraj, straj) -> None:
    """Animate both learned greedy policies racing on the same track at once."""
    n = max(len(qtraj), len(straj))
    q_last, s_last = len(qtraj) - 1, len(straj) - 1

    def outcome(traj):
        f = traj[-1]
        if f.get("success"):
            return "finished", f.get("step", 0)
        if f.get("crash"):
            return "crashed", f.get("step", 0)
        return "ran out of laps", f.get("step", 0)

    (qo, qs), (so, ss) = outcome(qtraj), outcome(straj)
    if qo == "finished" and (so != "finished" or qs <= ss):
        winner = "🔴 Q-Learning"
    elif so == "finished" and (qo != "finished" or ss < qs):
        winner = "🔵 SARSA"
    else:
        winner = "nobody finished"
    st.caption(f"🔴 **Q-Learning**: {qo} in {qs} steps · 🔵 **SARSA**: {so} in {ss} steps · "
               f"winner: **{winner}**  (each car holds its spot once done)")

    ph = st.empty()

    def show(i: int) -> None:
        ph.pyplot(_draw_race(env, qtraj[min(i, q_last)], straj[min(i, s_last)]),
                  clear_figure=True)

    c1, c2 = st.columns([1, 1])
    play = c1.button("▶ Start race", key="room3_race_go", use_container_width=True)
    idx = c2.slider("Race step", 0, max(1, n - 1), 0, key="room3_race_step")
    if play:
        for i in range(n):
            show(i)
            time.sleep(0.12)
    else:
        show(idx)


def render() -> None:
    st.title("🏎️ Room 3 — Racing Track · Q-Learning")
    st.markdown(
        "Reach the finish 🏁 as fast as possible. A long **safe route** snakes the "
        "track; a tempting **short-cut** cuts down the middle through **oil** (slip "
        "sideways) and **mud** (slow). On the narrow short-cut a slip can fling the "
        "car into a wall — a **crash** that ends the run (big penalty). **Q-Learning** "
        "is off-policy and chases the highest-value line: lower the slip probability "
        "to watch it gamble on the short-cut, raise it to watch it play safe."
    )

    with st.sidebar:
        st.header("⚙️ Q-Learning parameters")
        episodes = st.slider("Episodes", 100, 4000, 1500, 100)
        alpha = st.slider("Learning rate α", 0.01, 1.0, 0.2, 0.01)
        gamma = st.slider("Discount γ", 0.50, 0.999, 0.97, 0.005)
        eps_start = st.slider("ε start", 0.0, 1.0, 1.0, 0.05)
        eps_end = st.slider("ε end", 0.0, 0.5, 0.05, 0.01)
        eps_decay = st.slider("ε decay / episode", 0.90, 1.0, 0.997, 0.001)
        slip_prob = st.slider("Oil slip probability", 0.0, 0.5, 0.3, 0.05)
        max_steps = st.slider("Max steps / episode", 50, 500, 300, 50)
        seed = st.number_input("Random seed", value=2, step=1)
        compare = st.checkbox("Also train SARSA for comparison", value=False)
        st.header("Map difficulty")
        map_mode = st.selectbox("Map source", ["Generated", "Fixed layout"], key="room3_map_mode")
        if "room3_map_seed" not in st.session_state:
            st.session_state["room3_map_seed"] = random.randint(0, 99999)
        if st.button("🎲 New random layout", key="room3_newmap",
                     use_container_width=True, disabled=map_mode != "Generated"):
            st.session_state["room3_map_seed"] = random.randint(0, 99999)
        map_seed = st.number_input("Map seed", step=1, key="room3_map_seed")
        n_oil = st.slider("Oil tiles", 0, 12, 6, disabled=map_mode != "Generated")
        n_mud = st.slider("Mud tiles", 0, 12, 5, disabled=map_mode != "Generated")
        n_boosters = st.slider("Boosters", 0, 8, 4, disabled=map_mode != "Generated")
        with st.expander("Reward shaping"):
            r_finish = st.number_input("Finish", value=150.0, step=10.0)
            r_boost = st.number_input("Booster", value=20.0, step=5.0)
            r_mud = st.number_input("Mud", value=-5.0, step=1.0)
            r_crash = st.number_input("Crash (ends run)", value=-100.0, step=10.0)
            r_offtrack = st.number_input("Leave track edge", value=-30.0, step=5.0)
            r_wall = st.number_input("Wall hit", value=-5.0, step=1.0)
            r_step = st.number_input("Step cost", value=-1.0, step=1.0)
        train = st.button("🏁 Train", type="primary", use_container_width=True)

    layout = (
        generate_racing_layout(int(map_seed), n_oil, n_mud, n_boosters)
        if map_mode == "Generated"
        else DEFAULT_LAYOUT
    )
    env_config = dict(layout=layout, slip_prob=slip_prob, max_steps=max_steps,
                      r_step=r_step, r_finish=r_finish, r_boost=r_boost,
                      r_mud=r_mud, r_crash=r_crash, r_offtrack=r_offtrack,
                      r_wall=r_wall, seed=int(seed))

    def make_env():
        return RacingEnv(**env_config)

    env = make_env()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Track")
        st.pyplot(_draw(env, {"pos": env.start, "bmask": 0}), clear_figure=True)
        st.caption(f"{len(env.boosters)} boosters · {len(env.oil)} oil · "
                   f"{len(env.mud)} mud · {len(env.crash)} crash zones (✕) — "
                   f"every map has a safe route + a shorter oily short-cut")

    if train:
        common = dict(episodes=episodes, alpha=alpha, gamma=gamma, eps_start=eps_start,
                      eps_end=eps_end, eps_decay=eps_decay, max_steps=max_steps,
                      seed=int(seed))
        st.caption("Training Q-Learning…")
        res_q = train_with_live_progress(lambda cb: qlearning.train(make_env(), progress_cb=cb, **common))
        res_s = None
        if compare:
            st.caption("Training SARSA…")
            res_s = train_with_live_progress(lambda cb: sarsa.train(make_env(), progress_cb=cb, **common))
        st.session_state["room3"] = {"q": res_q, "sarsa": res_s, "env_config": env_config}

    store = st.session_state.get("room3")
    trained_env = RacingEnv(**store["env_config"]) if store and "env_config" in store else env
    with right:
        st.subheader("Result")
        if not store:
            st.info("Set parameters and press **Train**.")
        else:
            res_q = store["q"]
            fin = res_q.snapshots.get("Final")
            m1, m2, m3 = st.columns(3)
            m1.metric("Q success (last 100)", f"{res_q.success_rate(100):.0%}")
            if fin:
                m2.metric("Q greedy lap", f"{fin[-1]['step']} steps")
                m3.metric("Q greedy return", f"{fin[-1]['cum_reward']:+.0f}")
            if store["sarsa"]:
                rs = store["sarsa"]
                fs = rs.snapshots.get("Final")
                st.caption(f"SARSA: success {rs.success_rate(100):.0%}"
                           + (f", greedy lap {fs[-1]['step']} steps, "
                              f"return {fs[-1]['cum_reward']:+.0f}" if fs else ""))

    if store:
        res_q = store["q"]
        st.subheader("📈 Learning graphs")
        if store["sarsa"]:
            rs = store["sarsa"]
            df = pd.DataFrame({
                "Q-Learning": list(moving_average(res_q.episode_rewards, 50)),
                "SARSA": list(moving_average(rs.episode_rewards, 50)),
            })
            st.caption("Smoothed episode reward — Q-Learning vs SARSA")
            st.line_chart(df, height=260)
            steps_df = pd.DataFrame({
                "Q-Learning": list(moving_average(res_q.episode_steps, 50)),
                "SARSA": list(moving_average(rs.episode_steps, 50)),
            })
            st.caption("Smoothed steps-to-finish — lower (faster) is better")
            st.line_chart(steps_df, height=260)

            qtraj = res_q.snapshots.get("Final", [])
            straj = rs.snapshots.get("Final", [])
            if qtraj and straj:
                st.subheader("🏁 Watch both models race together")
                _race_player(trained_env, qtraj, straj)
        else:
            learning_section(res_q)

        st.subheader("🎞️ Replay any training episode")
        episode_replay_player(
            res_q,
            lambda f: _draw(trained_env, f),
            key="room3_q_all_episodes",
            caption="Q-Learning: pick any training episode and animate the actual exploratory lap.",
        )
        if store["sarsa"]:
            episode_replay_player(
                store["sarsa"],
                lambda f: _draw(trained_env, f),
                key="room3_sarsa_all_episodes",
                caption="SARSA comparison: replay any training episode on the same track.",
            )
