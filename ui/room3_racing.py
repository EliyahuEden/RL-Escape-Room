"""Room 3 UI — Racing Track solved with Q-Learning (optionally compared with SARSA)."""

from __future__ import annotations

import time
from typing import Dict

import pandas as pd
import streamlit as st

from rl.algos import qlearning, sarsa
from rl.envs.racing import RacingEnv
from ui.common import learning_section, replay_player, train_with_live_progress
from ui.render import render_grid
from rl.utils import moving_average


def _draw(env: RacingEnv, frame: dict):
    styles: Dict = {}
    for cell in env.oil:
        styles[cell] = {"color": "#444", "label": "🛢", "label_color": "#ffd"}
    for cell in env.mud:
        styles[cell] = {"color": "#8d6e63", "label": "M", "label_color": "#fff"}
    for cell in env.remaining_boosters(frame.get("bmask", 0)):
        styles[cell] = {"color": "#80deea", "label": "»", "label_color": "#006064"}
    styles[env.start] = {"color": "#d7f0d7", "label": "S", "label_color": "#2e7d32"}
    styles[env.finish] = {"color": "#7CFC00", "label": "🏁", "label_color": "#222"}
    title = "🏁 Finished!" if frame.get("success") else "On the track"
    return render_grid(env.rows, env.cols, walls=env.walls, cell_styles=styles,
                       agent=frame["pos"], agent_color="#e53935", title=title)


def _track_styles(env: RacingEnv) -> Dict:
    styles: Dict = {}
    for cell in env.oil:
        styles[cell] = {"color": "#444", "label": "🛢", "label_color": "#ffd"}
    for cell in env.mud:
        styles[cell] = {"color": "#8d6e63", "label": "M", "label_color": "#fff"}
    for cell in env.boosters:
        styles[cell] = {"color": "#80deea", "label": "»", "label_color": "#006064"}
    styles[env.start] = {"color": "#d7f0d7", "label": "S", "label_color": "#2e7d32"}
    styles[env.finish] = {"color": "#7CFC00", "label": "🏁", "label_color": "#222"}
    return styles


def _draw_race(env: RacingEnv, qf: dict, sf: dict):
    """Draw both cars on the same track: 🔴 Q-Learning, 🔵 SARSA."""
    markers = [(qf["pos"], "Q", "#e53935"), (sf["pos"], "S", "#1565c0")]
    return render_grid(env.rows, env.cols, walls=env.walls, cell_styles=_track_styles(env),
                       agent=None, markers=markers, title="🏁 Q (red) vs SARSA (blue)")


def _race_player(env: RacingEnv, qtraj, straj) -> None:
    n = max(len(qtraj), len(straj))
    qsteps, ssteps = qtraj[-1]["step"], straj[-1]["step"]
    winner = "🔴 Q-Learning" if qsteps < ssteps else ("🔵 SARSA" if ssteps < qsteps else "Tie")
    st.caption(f"🔴 Q-Learning finishes in **{qsteps}** steps · 🔵 SARSA in **{ssteps}** · "
               f"winner: **{winner}** (each car holds at the finish line)")
    ph = st.empty()

    def show(i: int) -> None:
        qf = qtraj[min(i, len(qtraj) - 1)]
        sf = straj[min(i, len(straj) - 1)]
        ph.pyplot(_draw_race(env, qf, sf), clear_figure=True)

    if st.button("▶ Start race", key="room3_race_go"):
        for i in range(n):
            show(i)
            time.sleep(0.12)
    else:
        show(st.slider("Race step", 0, n - 1, 0, key="room3_race_step"))


def render() -> None:
    st.title("🏎️ Room 3 — Racing Track · Q-Learning")
    st.markdown(
        "Reach the finish 🏁 as fast as possible. A clean **main route** runs down "
        "the right; a tempting **short-cut** cuts down the middle through oil 🛢 "
        "(slippery, costly) and mud (slow). **Q-Learning** is off-policy and chases "
        "the highest-value line — turn the slip/oil penalties down to watch it "
        "start gambling on the short-cut."
    )

    with st.sidebar:
        st.header("⚙️ Q-Learning parameters")
        episodes = st.slider("Episodes", 100, 4000, 1500, 100)
        alpha = st.slider("Learning rate α", 0.01, 1.0, 0.2, 0.01)
        gamma = st.slider("Discount γ", 0.50, 0.999, 0.97, 0.005)
        eps_start = st.slider("ε start", 0.0, 1.0, 1.0, 0.05)
        eps_end = st.slider("ε end", 0.0, 0.5, 0.05, 0.01)
        eps_decay = st.slider("ε decay / episode", 0.90, 1.0, 0.997, 0.001)
        slip_prob = st.slider("Oil slip probability", 0.0, 0.5, 0.2, 0.05)
        max_steps = st.slider("Max steps / episode", 50, 500, 300, 50)
        seed = st.number_input("Random seed", value=2, step=1)
        compare = st.checkbox("Also train SARSA for comparison", value=False)
        with st.expander("Reward shaping"):
            r_finish = st.number_input("Finish", value=100.0, step=10.0)
            r_boost = st.number_input("Booster", value=15.0, step=5.0)
            r_mud = st.number_input("Mud", value=-10.0, step=1.0)
            r_oil = st.number_input("Oil (slip)", value=-20.0, step=1.0)
            r_offtrack = st.number_input("Leave track", value=-30.0, step=5.0)
            r_step = st.number_input("Step cost", value=-1.0, step=1.0)
        train = st.button("🏁 Train", type="primary", use_container_width=True)

    def make_env():
        return RacingEnv(slip_prob=slip_prob, max_steps=max_steps, r_step=r_step,
                         r_finish=r_finish, r_boost=r_boost, r_mud=r_mud, r_oil=r_oil,
                         r_offtrack=r_offtrack)

    env = make_env()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Track")
        st.pyplot(_draw(env, {"pos": env.start, "bmask": 0}), clear_figure=True)
        st.caption(f"{len(env.boosters)} boosters · {len(env.oil)} oil · "
                   f"{len(env.mud)} mud cells")

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
        st.session_state["room3"] = {"q": res_q, "sarsa": res_s}

    store = st.session_state.get("room3")
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
        else:
            learning_section(res_q)

        if store["sarsa"]:
            st.subheader("🏁 Race — watch both learned policies run together")
            qtraj = res_q.snapshots.get("Final", [])
            straj = store["sarsa"].snapshots.get("Final", [])
            if qtraj and straj:
                _race_player(env, qtraj, straj)

        st.subheader("🎬 Replay")
        snaps = {f"Q-Learning · {k}": v for k, v in res_q.snapshots.items()}
        if store["sarsa"]:
            snaps.update({f"SARSA · {k}": v for k, v in store["sarsa"].snapshots.items()})
        replay_player(snaps, lambda f: _draw(env, f), key="room3",
                      caption="Compare the racing lines learned at each stage / by each algorithm.")
