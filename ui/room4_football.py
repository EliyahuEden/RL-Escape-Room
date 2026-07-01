"""Room 4 UI — Football solved with DQN: Dribble + Shoot or Free Kick mode."""

from __future__ import annotations

from typing import List

import streamlit as st
from matplotlib.patches import Circle, Polygon, Rectangle

from rl.algos import dqn
from rl.envs.football import FootballEnv, FreeKickEnv
from ui.common import episode_replay_player, learning_section, train_with_live_progress
from ui.render import field_axes
from ui_canvas.toggle import use_canvas
from ui_canvas.room4_football import draw_football_canvas, replay_football_canvas

HIDDEN_CHOICES = {"64, 64": (64, 64), "128, 128": (128, 128), "256, 128": (256, 128)}


def _draw(env, frame: dict):
    fig, ax = field_axes(env.W, env.H)
    is_fk = frame.get("mode") == "freekick" or isinstance(env, FreeKickEnv)

    if not is_fk:
        shoot_x = env.shoot_x
        ax.add_patch(Rectangle((shoot_x, 0), env.W - shoot_x, env.H,
                               facecolor="white", alpha=0.10, zorder=1))
        ax.plot([shoot_x, shoot_x], [0, env.H], color="white", ls="--", lw=1, alpha=0.5)
    goal_lo = env.goal_lo if hasattr(env, "goal_lo") else 3.5
    goal_hi = env.goal_hi if hasattr(env, "goal_hi") else 6.5
    ax.add_patch(Rectangle((env.W - 0.15, goal_lo), 0.15, goal_hi - goal_lo,
                           facecolor="white", zorder=2))

    kx, ky = frame["keeper"]
    kr = env.keeper_reach if hasattr(env, "keeper_reach") else 0.7
    ax.add_patch(Rectangle((kx - 0.15, ky - kr), 0.3, 2 * kr,
                           facecolor="#ffeb3b", edgecolor="#333", zorder=4))
    ax.text(kx, ky, "GK", ha="center", va="center", fontsize=8, fontweight="bold",
            color="#111", zorder=8)
    ax.add_patch(Circle((kx, ky - kr), 0.09, facecolor="#111", edgecolor="none", zorder=8))
    ax.add_patch(Circle((kx, ky + kr), 0.09, facecolor="#111", edgecolor="none", zorder=8))

    for dx, dy in frame.get("defenders", []):
        ax.add_patch(Circle((dx, dy), 0.3, facecolor="#e53935", edgecolor="#7a0000", zorder=4))
        ax.plot([dx - 0.16, dx + 0.16], [dy - 0.08, dy + 0.08],
                color="white", lw=2, zorder=8)

    px, py = frame["x"], frame["y"]
    ax.add_patch(Circle((px, py), 0.3, facecolor="#1e88e5", edgecolor="#fff", lw=1.5, zorder=5))
    ax.add_patch(Circle((px, py - 0.1), 0.08, facecolor="#bbdefb", edgecolor="#0d47a1", lw=0.8, zorder=8))
    ax.plot([px, px], [py - 0.02, py + 0.16], color="white", lw=1.8, zorder=8)
    ax.plot([px, px - 0.14], [py + 0.05, py + 0.18], color="white", lw=1.8, zorder=8)
    ax.plot([px, px + 0.14], [py + 0.05, py + 0.17], color="white", lw=1.8, zorder=8)

    bx, by = frame.get("ball", (px, py))
    ball_z = frame.get("ball_z", 0.0)
    lift = ball_z * 0.25
    ball_draw_y = by - lift
    ball_size = 0.16
    if ball_z > 0.1:
        shadow_size = max(0.06, 0.14 - ball_z * 0.02)
        ax.add_patch(Circle((bx, by), shadow_size, facecolor="#000", alpha=0.25, zorder=1))
        if lift > 0.05:
            ax.plot([bx, bx], [by, ball_draw_y + ball_size], color="#888",
                    lw=0.8, ls=":", alpha=0.5, zorder=6)
    ax.add_patch(Circle((bx, ball_draw_y), ball_size, facecolor="white", edgecolor="#222",
                        lw=1.2, zorder=10))
    ax.add_patch(Polygon([(bx, ball_draw_y - 0.06), (bx + 0.055, ball_draw_y - 0.02),
                          (bx + 0.035, ball_draw_y + 0.055), (bx - 0.035, ball_draw_y + 0.055),
                          (bx - 0.055, ball_draw_y - 0.02)],
                         closed=True, facecolor="#222", edgecolor="none", zorder=11))
    if ball_z > 0.3:
        ax.text(bx + 0.22, ball_draw_y, f"{ball_z:.1f}m", fontsize=7, color="#fff",
                fontweight="bold", ha="left", va="center", zorder=12)

    if not frame.get("ball_in_flight") and (frame.get("vx") or frame.get("vy")):
        ax.arrow(px, py, frame["vx"] * 0.6, frame["vy"] * 0.6, head_width=0.18,
                 color="white", zorder=6)

    ax.set_title(frame.get("event", "In play"), fontsize=11)
    fig.tight_layout(pad=0.2)
    return fig


def _expand_flight(frames: List[dict]) -> List[dict]:
    out: List[dict] = []
    for f in frames:
        flight = f.get("flight")
        if flight:
            last = len(flight) - 1
            for i, scene in enumerate(flight):
                out.append({
                    "x": f["x"], "y": f["y"], "vx": 0, "vy": 0,
                    "ball": scene["ball"], "ball_z": scene.get("ball_z", 0.0),
                    "ball_in_flight": True,
                    "keeper": scene["keeper"], "defenders": f.get("defenders", []),
                    "event": f.get("event", "shot!") if i == last else "ball away!",
                    "step": f.get("step"), "cum_reward": f.get("cum_reward"),
                    "mode": f.get("mode"),
                })
        else:
            out.append(f)
    return out


def render() -> None:
    st.title("⚽ Room 4 — Football · DQN")

    with st.sidebar:
        st.header("⚙️ Game mode")
        game_mode = st.selectbox("Mode", ["Dribble + Shoot", "Free Kick"], key="room4_mode")

    if game_mode == "Free Kick":
        _render_freekick()
    else:
        _render_dribble()


def _render_dribble() -> None:
    st.markdown(
        "Dribble past the red defenders (random positions each game), reach the "
        "shooting area, then **kick the ball into the goal**. The ball *leaves the "
        "player* and flies — you pick the **power** (soft/hard) and **curve** "
        "(bend left/straight/right). The yellow keeper **patrols side to side**, so "
        "time the shot for when he's drifted away, or bend it around him."
    )

    with st.sidebar:
        st.header("⚙️ DQN parameters")
        episodes = st.slider("Episodes", 100, 2000, 800, 50)
        hidden = HIDDEN_CHOICES[st.selectbox("Hidden layers", list(HIDDEN_CHOICES), index=1)]
        lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)
        gamma = st.slider("Discount γ", 0.80, 0.999, 0.99, 0.005)
        batch_size = st.select_slider("Batch size", [32, 64, 128, 256], value=64)
        eps_end = st.slider("ε end", 0.0, 0.3, 0.05, 0.01)
        expl = st.slider("Exploration fraction", 0.1, 1.0, 0.6, 0.05)
        max_steps = st.slider("Shot clock (steps per attempt)", 60, 300, 120, 20)
        seed = st.number_input("Random seed", value=0, step=1)
        st.markdown("**Difficulty**")
        n_def = st.slider("Defenders", 1, 5, 3)
        def_speed = st.slider("Defender speed (m/s)", 0.3, 1.2, 0.6, 0.05)
        keeper_speed = st.slider("Keeper patrol speed (m/s)", 0.5, 4.0, 1.5, 0.25)
        train = st.button("🧠 Train DQN", type="primary", use_container_width=True)

    env_config = dict(n_defenders=n_def, def_speed=def_speed,
                      keeper_speed=keeper_speed, max_steps=max_steps,
                      seed=int(seed))

    def make_env():
        return FootballEnv(**env_config)

    env = make_env()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Pitch")
        _f = env.render_state() | {"event": "Kick-off"}
        if use_canvas():
            draw_football_canvas(env, _f, title="Kick-off")
        else:
            st.pyplot(_draw(env, _f), clear_figure=True)
        st.caption("Blue = player · white = ball · red = defenders · yellow = keeper "
                   "(patrolling) · dashed line = shooting area")

    if train:
        result = train_with_live_progress(
            lambda cb: dqn.train(make_env(), episodes=episodes, hidden=hidden, lr=lr,
                                 gamma=gamma, batch_size=batch_size, eps_end=eps_end,
                                 exploration_fraction=expl, target_update=500,
                                 learn_start=1500, max_steps=max_steps + 5, seed=int(seed),
                                 progress_cb=cb),
            update_every=10,
        )
        st.session_state["room4"] = {"result": result, "env_config": env_config,
                                     "mode": "dribble"}

    store = st.session_state.get("room4")
    if isinstance(store, dict) and store.get("mode") != "dribble":
        store = None
    if isinstance(store, dict) and "result" in store:
        result = store["result"]
        trained_env = FootballEnv(**store["env_config"])
    else:
        result = store if not isinstance(store, dict) else None
        trained_env = env
    with right:
        st.subheader("Result")
        if not result:
            st.info("Set parameters and press **Train DQN**. (Training takes ~1–2 min.)")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Goal rate (last 50)", f"{result.success_rate(50):.0%}")
            fin = result.snapshots.get("Final")
            if fin:
                m2.metric("Final greedy", fin[-1].get("event", "—"))

    if result:
        st.subheader("📈 Learning graphs")
        learning_section(result)
        st.subheader("🎞️ Replay any training episode")
        if use_canvas() and result.episode_replays:
            n = len(result.episode_replays)
            ep = st.slider("Episode", 1, n, n, key="room4_ep_sel") - 1
            raw = _expand_flight(result.episode_replays[ep])
            if raw:
                replay_football_canvas(trained_env, raw, key=f"room4_cv_{ep}", expand_flight=False)
        else:
            episode_replay_player(
                result,
                lambda f: _draw(trained_env, f),
                key="room4_all_episodes",
                caption="Pick any DQN training episode; exploratory shots include the animated ball flight.",
                frame_transform=_expand_flight,
            )


def _render_freekick() -> None:
    st.markdown(
        "Take a **free kick** from a fixed position. A **wall** of red defenders "
        "stands between you and the goal, and the keeper patrols the goal mouth. "
        "Choose the **aim** (low/mid/high — high shots arc over the wall), "
        "**power** (soft/hard), and **curve** (left/straight/right — bends the ball "
        "around the keeper). The ball has real 3D physics: it rises, curves, "
        "and drops back down."
    )

    with st.sidebar:
        st.header("⚙️ DQN parameters")
        episodes = st.slider("Episodes", 100, 3000, 1500, 50, key="fk_episodes")
        hidden = HIDDEN_CHOICES[st.selectbox("Hidden layers", list(HIDDEN_CHOICES), index=1,
                                             key="fk_hidden")]
        lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3,
                              key="fk_lr")
        gamma = st.slider("Discount γ", 0.80, 0.999, 0.99, 0.005, key="fk_gamma")
        batch_size = st.select_slider("Batch size", [32, 64, 128, 256], value=64,
                                      key="fk_batch")
        eps_end = st.slider("ε end", 0.0, 0.3, 0.05, 0.01, key="fk_eps")
        expl = st.slider("Exploration fraction", 0.1, 1.0, 0.5, 0.05, key="fk_expl")
        seed = st.number_input("Random seed", value=0, step=1, key="fk_seed")
        st.markdown("**Free kick setup**")
        n_wall = st.slider("Wall players", 1, 5, 3, key="fk_wall")
        keeper_speed = st.slider("Keeper patrol speed", 0.5, 4.0, 1.5, 0.25, key="fk_ks")
        if "fk_kick_x" not in st.session_state:
            st.session_state["fk_kick_x"] = 5.0
            st.session_state["fk_kick_y"] = 5.0
        if st.button("🎲 New kick position", use_container_width=True, key="fk_newpos"):
            import random as _rng
            st.session_state["fk_kick_x"] = round(_rng.uniform(3.0, 7.0), 1)
            st.session_state["fk_kick_y"] = round(_rng.uniform(2.5, 7.5), 1)
        kick_x = st.session_state["fk_kick_x"]
        kick_y = st.session_state["fk_kick_y"]
        st.caption(f"Kicker at ({kick_x:.1f}, {kick_y:.1f})")
        train = st.button("🧠 Train DQN", type="primary", use_container_width=True,
                          key="fk_train")

    env_config = dict(n_wall=n_wall, kick_x=kick_x, kick_y=kick_y,
                      keeper_speed=keeper_speed, seed=int(seed))

    def make_env():
        return FreeKickEnv(**env_config)

    env = make_env()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Free kick setup")
        _fk_frame = env.render_state() | {"event": "Ready to kick"}
        if use_canvas():
            draw_football_canvas(env, _fk_frame, title="Ready to kick")
        else:
            st.pyplot(_draw(env, _fk_frame), clear_figure=True)
        st.caption(f"Blue = kicker at ({kick_x:.1f}, {kick_y:.1f}) · "
                   f"red wall = {n_wall} defenders · yellow = keeper · "
                   f"18 kick actions (3 aims × 2 powers × 3 curves)")

    if train:
        result = train_with_live_progress(
            lambda cb: dqn.train(make_env(), episodes=episodes, hidden=hidden, lr=lr,
                                 gamma=gamma, batch_size=batch_size, eps_end=eps_end,
                                 exploration_fraction=expl, target_update=200,
                                 learn_start=500, max_steps=10, seed=int(seed),
                                 progress_cb=cb),
            update_every=25,
        )
        st.session_state["room4"] = {"result": result, "env_config": env_config,
                                     "mode": "freekick"}

    store = st.session_state.get("room4")
    if isinstance(store, dict) and store.get("mode") != "freekick":
        store = None
    if isinstance(store, dict) and "result" in store:
        result = store["result"]
        trained_env = FreeKickEnv(**store["env_config"])
    else:
        result = store if not isinstance(store, dict) else None
        trained_env = env
    with right:
        st.subheader("Result")
        if not result:
            st.info("Set parameters and press **Train DQN**.")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Goal rate (last 50)", f"{result.success_rate(50):.0%}")
            fin = result.snapshots.get("Final")
            if fin:
                m2.metric("Final greedy", fin[-1].get("event", "—"))
                kick_info = fin[-1].get("kick")
                if kick_info:
                    aim, power, curve = kick_info
                    st.caption(f"Best shot: **{aim}** aim · **{power}** power · **{curve}** curve")

    if result:
        st.subheader("📈 Learning graphs")
        learning_section(result)
        st.subheader("🎞️ Replay any free kick")
        if use_canvas() and result.episode_replays:
            n = len(result.episode_replays)
            ep = st.slider("Episode", 1, n, n, key="room4_fk_ep_sel") - 1
            raw = _expand_flight(result.episode_replays[ep])
            if raw:
                replay_football_canvas(trained_env, raw, key=f"room4_fkcv_{ep}", expand_flight=False)
        else:
            episode_replay_player(
                result,
                lambda f: _draw(trained_env, f),
                key="room4_fk_episodes",
                caption="Pick any training kick and watch the ball flight — does it go over the wall? Around the keeper?",
                frame_transform=_expand_flight,
            )
