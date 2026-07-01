"""Room 3 UI — Street Race solved with Q-Learning (optionally compared with SARSA)."""

from __future__ import annotations

import random
import time
from typing import Dict

import pandas as pd
import streamlit as st

from rl.algos import qlearning, sarsa
from rl.envs.racing import DEFAULT_LAYOUT, RacingEnv, generate_racing_layout, racing_layout_stats
from ui.common import episode_replay_player, learning_section, train_with_live_progress
from ui.render import render_grid
from rl.utils import moving_average
from ui_canvas.toggle import use_canvas
from ui_canvas.room3_racing import draw_racing_canvas, replay_racing_canvas


def _draw(env: RacingEnv, frame: dict):
    bmask = frame.get("bmask", 0)
    styles = _track_styles(env, bmask)
    unlocked = frame.get("finish_unlocked", env.finish_unlocked(bmask))
    collected = env.collected_count(bmask)
    needed = env.min_boosters

    if frame.get("crash"):
        title = "💥 Crashed!"
    elif frame.get("success"):
        title = "🏁 Finished!"
    elif unlocked:
        title = f"🏎️ Finish unlocked! Go go go! (⚡ {collected}/{len(env.boosters)})"
    else:
        title = f"🏎️ Collect boosters to unlock finish (⚡ {collected}/{needed} needed)"
    return render_grid(env.rows, env.cols, walls=env.walls, cell_styles=styles,
                       agent=frame["pos"], agent_color="#e53935",
                       agent_symbol="car", agent_symbol_color="#fff",
                       title=title)


def _track_styles(env: RacingEnv, bmask=None) -> Dict:
    styles: Dict = {
        (r, c): {"color": "#596471"}
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in env.walls
    }
    for cell in getattr(env, "shortcut_cells", set()) - env.oil:
        styles[cell] = {"color": "#6b7280"}
    for cell in env.oil:
        styles[cell] = {"color": "#444", "icon": "oil", "label_color": "#111"}
    for cell in env.mud:
        styles[cell] = {"color": "#8d6e63", "icon": "mud", "label_color": "#795548"}
    booster_cells = env.boosters if bmask is None else env.remaining_boosters(bmask)
    for cell in booster_cells:
        styles[cell] = {"color": "#80deea", "icon": "boost", "label_color": "#f57f17"}
    for cell in getattr(env, "crash", set()):
        styles[cell] = {"color": "#ef5350", "label": "✕", "label_color": "#fff"}
    styles[env.start] = {"color": "#d7f0d7", "icon": "flag", "label_color": "#2e7d32"}
    unlocked = env.finish_unlocked(bmask) if bmask is not None else False
    styles[env.finish] = {"color": "#7CFC00" if unlocked else "#d9a441",
                          "icon": "flag", "label_color": "#222"}
    return styles


def _draw_race(env: RacingEnv, qf: dict, sf: dict):
    """Draw both cars on the same track."""
    markers = [(qf["pos"], "car", "#e53935"), (sf["pos"], "car", "#1565c0")]
    return render_grid(env.rows, env.cols, walls=env.walls, cell_styles=_track_styles(env),
                       agent=None, markers=markers, title="🔴 Q-Learning  vs  🔵 SARSA")


def _race_player(env: RacingEnv, qtraj, straj) -> None:
    """Animate both learned greedy policies racing on the same track."""
    n = max(len(qtraj), len(straj))
    q_last, s_last = len(qtraj) - 1, len(straj) - 1

    def outcome(traj):
        f = traj[-1]
        if f.get("success"):
            return "finished", f.get("step", 0)
        if f.get("crash"):
            return "crashed", f.get("step", 0)
        return "timed out", f.get("step", 0)

    (qo, qs), (so, ss) = outcome(qtraj), outcome(straj)
    if qo == "finished" and (so != "finished" or qs <= ss):
        winner = "🔴 Q-Learning"
    elif so == "finished" and (qo != "finished" or ss < qs):
        winner = "🔵 SARSA"
    else:
        winner = "nobody finished"
    st.caption(f"🔴 **Q-Learning**: {qo} in {qs} steps · "
               f"🔵 **SARSA**: {so} in {ss} steps · "
               f"winner: **{winner}**")

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
    st.title("🏎️ Room 3 — Street Race · Q-Learning")
    st.markdown(
        "Race through a city circuit from start to finish. "
        "The finish line is **locked** until you collect enough **boosters** ⚡ — "
        "plan your route to grab them while dodging "
        "**oil spills** 🛢️ (slippery), "
        "**mud** (penalty), and **crash barriers** ✕ (race over).\n\n"
        "The agent uses **Q-Learning** (off-policy TD control) to learn the "
        "fastest racing line. Compare with SARSA to see the difference."
    )

    with st.sidebar:
        st.header("⚙️ Q-Learning parameters")
        episodes = st.slider("Episodes", 100, 4000, 2000, 100)
        alpha = st.slider("Learning rate α", 0.01, 1.0, 0.2, 0.01)
        gamma = st.slider("Discount γ", 0.50, 0.999, 0.97, 0.005)
        eps_start = st.slider("ε start", 0.0, 1.0, 1.0, 0.05)
        eps_end = st.slider("ε end", 0.0, 0.5, 0.05, 0.01)
        eps_decay = st.slider("ε decay / episode", 0.90, 1.0, 0.997, 0.001)
        slip_prob = st.slider("Oil slip chance", 0.0, 0.5, 0.2, 0.05)
        max_steps = st.slider("Max steps / episode", 50, 500, 200, 10)
        min_boosters = st.slider("Boosters needed to unlock finish", 1, 6, 3)
        seed = st.number_input("Random seed", value=2, step=1)
        compare = st.checkbox("Also train SARSA for comparison", value=False)
        st.header("Track layout")
        use_generated = st.checkbox("Generate random track", value=False, key="room3_gen")
        if "room3_map_seed" not in st.session_state:
            st.session_state["room3_map_seed"] = random.randint(0, 99999)
        if st.button("🎲 New random track", key="room3_newmap",
                     use_container_width=True, disabled=not use_generated):
            st.session_state["room3_map_seed"] = random.randint(0, 99999)
        map_seed = st.number_input("Map seed", step=1, key="room3_map_seed",
                                   disabled=not use_generated)
        n_oil = st.slider("Oil spills", 0, 12, 5, disabled=not use_generated)
        n_mud = st.slider("Mud patches", 0, 12, 3, disabled=not use_generated)
        n_boosters = st.slider("Boosters on track", 0, 8, 4, disabled=not use_generated)
        n_crash = st.slider("Crash barriers", 0, 6, 2, disabled=not use_generated)
        with st.expander("Reward shaping"):
            r_finish = st.number_input("Cross finish line", value=200.0, step=10.0)
            r_boost = st.number_input("Collect booster", value=20.0, step=5.0)
            r_mud = st.number_input("Mud", value=-5.0, step=1.0)
            r_crash = st.number_input("Crash (race over)", value=-200.0, step=10.0)
            r_offtrack = st.number_input("Off track edge", value=-30.0, step=5.0)
            r_wall = st.number_input("Wall hit", value=-5.0, step=1.0)
            r_finish_locked = st.number_input("Finish while locked", value=-10.0, step=1.0)
            r_step = st.number_input("Step cost", value=-1.0, step=1.0)
        train = st.button("🏁 Train", type="primary", use_container_width=True)

    layout = (
        generate_racing_layout(int(map_seed), n_oil, n_mud, n_boosters, n_crash)
        if use_generated
        else DEFAULT_LAYOUT
    )
    stats = racing_layout_stats(layout)
    env_config = dict(layout=layout, slip_prob=slip_prob, max_steps=max_steps,
                      min_boosters=min_boosters,
                      r_step=r_step, r_finish=r_finish, r_boost=r_boost,
                      r_mud=r_mud, r_crash=r_crash, r_offtrack=r_offtrack,
                      r_wall=r_wall, r_finish_locked=r_finish_locked,
                      seed=int(seed))

    def make_env():
        return RacingEnv(**env_config)

    env = make_env()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Track")
        if use_canvas():
            draw_racing_canvas(env, {"pos": env.start, "bmask": 0})
        else:
            st.pyplot(_draw(env, {"pos": env.start, "bmask": 0}), clear_figure=True)
        st.caption(f"{len(env.boosters)} boosters (need {min_boosters}) · "
                   f"{len(env.oil)} oil · {len(env.mud)} mud · "
                   f"{len(env.crash)} crash barriers")

    if train:
        common = dict(episodes=episodes, alpha=alpha, gamma=gamma, eps_start=eps_start,
                      eps_end=eps_end, eps_decay=eps_decay, max_steps=max_steps,
                      seed=int(seed))
        st.caption("Training Q-Learning...")
        res_q = train_with_live_progress(lambda cb: qlearning.train(make_env(), progress_cb=cb, **common))
        res_s = None
        if compare:
            st.caption("Training SARSA...")
            res_s = train_with_live_progress(lambda cb: sarsa.train(make_env(), progress_cb=cb, **common))
        st.session_state["room3"] = {"q": res_q, "sarsa": res_s, "env_config": env_config}

    store = st.session_state.get("room3")
    if isinstance(store, dict) and store.get("env_config") != env_config:
        st.session_state.pop("room3", None)
        store = None
    trained_env = RacingEnv(**store["env_config"]) if store and "env_config" in store else env
    with right:
        st.subheader("Result")
        if not store:
            st.info("Set parameters and press **Train** to start the race.")
        else:
            res_q = store["q"]
            fin = res_q.snapshots.get("Final")
            m1, m2, m3 = st.columns(3)
            m1.metric("Finish rate (last 100)", f"{res_q.success_rate(100):.0%}")
            if fin:
                m2.metric("Greedy lap", f"{fin[-1]['step']} steps")
                m3.metric("Greedy return", f"{fin[-1]['cum_reward']:+.0f}")
            if store["sarsa"]:
                rs = store["sarsa"]
                fs = rs.snapshots.get("Final")
                st.caption(f"SARSA: finish rate {rs.success_rate(100):.0%}"
                           + (f", greedy {fs[-1]['step']} steps, "
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
            st.caption("Smoothed steps — lower (faster) is better")
            st.line_chart(steps_df, height=260)

            qtraj = res_q.snapshots.get("Final", [])
            straj = rs.snapshots.get("Final", [])
            if qtraj and straj:
                st.subheader("🏁 Watch both cars race")
                _race_player(trained_env, qtraj, straj)
        else:
            learning_section(res_q)

        st.subheader("🎞️ Replay any race")
        if use_canvas() and res_q.episode_replays:
            n = len(res_q.episode_replays)
            ep = st.slider("Episode (Q-Learning)", 1, n, n, key="room3_q_ep_sel") - 1
            raw = res_q.episode_replays[ep]
            if raw:
                replay_racing_canvas(trained_env, raw, key=f"room3_qcv_{ep}")
        else:
            episode_replay_player(
                res_q,
                lambda f: _draw(trained_env, f),
                key="room3_q_all_episodes",
                caption="Q-Learning: pick any training race and watch the car collect boosters and race to the finish.",
            )
        if store["sarsa"]:
            if use_canvas() and store["sarsa"].episode_replays:
                n2 = len(store["sarsa"].episode_replays)
                ep2 = st.slider("Episode (SARSA)", 1, n2, n2, key="room3_s_ep_sel") - 1
                raw2 = store["sarsa"].episode_replays[ep2]
                if raw2:
                    replay_racing_canvas(trained_env, raw2, key=f"room3_scv_{ep2}")
            else:
                episode_replay_player(
                    store["sarsa"],
                    lambda f: _draw(trained_env, f),
                    key="room3_sarsa_all_episodes",
                    caption="SARSA: replay any race for comparison.",
                )
