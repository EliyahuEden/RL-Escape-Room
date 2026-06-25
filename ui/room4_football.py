"""Room 4 UI — Football Final Shot solved with DQN (ball flight + patrolling keeper)."""

from __future__ import annotations

from typing import List

import streamlit as st
from matplotlib.patches import Circle, Polygon, Rectangle

from rl.algos import dqn
from rl.envs.football import FootballEnv
from ui.common import episode_replay_player, learning_section, train_with_live_progress
from ui.render import field_axes

HIDDEN_CHOICES = {"64, 64": (64, 64), "128, 128": (128, 128), "256, 128": (256, 128)}


def _draw(env: FootballEnv, frame: dict):
    fig, ax = field_axes(env.W, env.H)
    # shooting area
    ax.add_patch(Rectangle((env.shoot_x, 0), env.W - env.shoot_x, env.H,
                           facecolor="white", alpha=0.10, zorder=1))
    ax.plot([env.shoot_x, env.shoot_x], [0, env.H], color="white", ls="--", lw=1, alpha=0.5)
    # goal mouth
    ax.add_patch(Rectangle((env.W - 0.15, env.goal_lo), 0.15, env.goal_hi - env.goal_lo,
                           facecolor="white", zorder=2))
    # keeper (height ≈ its save zone)
    kx, ky = frame["keeper"]
    ax.add_patch(Rectangle((kx - 0.15, ky - env.keeper_reach), 0.3, 2 * env.keeper_reach,
                           facecolor="#ffeb3b", edgecolor="#333", zorder=4))
    ax.text(kx, ky, "GK", ha="center", va="center", fontsize=8, fontweight="bold",
            color="#111", zorder=8)
    ax.add_patch(Circle((kx, ky - env.keeper_reach), 0.09, facecolor="#111", edgecolor="none", zorder=8))
    ax.add_patch(Circle((kx, ky + env.keeper_reach), 0.09, facecolor="#111", edgecolor="none", zorder=8))
    # defenders
    for dx, dy in frame.get("defenders", []):
        ax.add_patch(Circle((dx, dy), 0.3, facecolor="#e53935", edgecolor="#7a0000", zorder=4))
        ax.plot([dx - 0.16, dx + 0.16], [dy - 0.08, dy + 0.08],
                color="white", lw=2, zorder=8)
    # player
    px, py = frame["x"], frame["y"]
    ax.add_patch(Circle((px, py), 0.3, facecolor="#1e88e5", edgecolor="#fff", lw=1.5, zorder=5))
    ax.add_patch(Circle((px, py - 0.1), 0.08, facecolor="#bbdefb", edgecolor="#0d47a1", lw=0.8, zorder=8))
    ax.plot([px, px], [py - 0.02, py + 0.16], color="white", lw=1.8, zorder=8)
    ax.plot([px, px - 0.14], [py + 0.05, py + 0.18], color="white", lw=1.8, zorder=8)
    ax.plot([px, px + 0.14], [py + 0.05, py + 0.17], color="white", lw=1.8, zorder=8)
    # ball (separate entity)
    bx, by = frame.get("ball", (px, py))
    ax.add_patch(Circle((bx, by), 0.16, facecolor="white", edgecolor="#222", lw=1.2, zorder=7))
    ax.add_patch(Polygon([(bx, by - 0.06), (bx + 0.055, by - 0.02),
                          (bx + 0.035, by + 0.055), (bx - 0.035, by + 0.055),
                          (bx - 0.055, by - 0.02)],
                         closed=True, facecolor="#222", edgecolor="none", zorder=9))
    # dribble velocity arrow
    if not frame.get("ball_in_flight") and (frame.get("vx") or frame.get("vy")):
        ax.arrow(px, py, frame["vx"] * 0.6, frame["vy"] * 0.6, head_width=0.18,
                 color="white", zorder=6)
    ax.set_title(frame.get("event", "In play"), fontsize=11)
    fig.tight_layout(pad=0.2)
    return fig


def _expand_flight(frames: List[dict]) -> List[dict]:
    """Expand each kick frame (which carries the ball-flight trajectory) into a
    sequence of sub-frames so the replay animates the ball leaving the player."""
    out: List[dict] = []
    for f in frames:
        flight = f.get("flight")
        if flight:
            last = len(flight) - 1
            for i, scene in enumerate(flight):
                out.append({
                    "x": f["x"], "y": f["y"], "vx": 0, "vy": 0,
                    "ball": scene["ball"], "ball_in_flight": True,
                    "keeper": scene["keeper"], "defenders": f.get("defenders", []),
                    "event": f.get("event", "shot!") if i == last else "ball away!",
                    "step": f.get("step"), "cum_reward": f.get("cum_reward"),
                })
        else:
            out.append(f)
    return out


def render() -> None:
    st.title("⚽ Room 4 — Football Final Shot · DQN")
    st.markdown(
        "Dribble past the red defenders (random positions each game), reach the "
        "shooting area, then **kick the ball into the goal**. The ball *leaves the "
        "player* and flies — you pick the **power** (soft/hard → speed) and **curve** "
        "(bend left/straight/right). The yellow keeper **patrols side to side**, so "
        "time the shot for when he's drifted away, or bend it around him. A Q-table "
        "is impractical here, so a neural net approximates Q(s, a)."
    )

    with st.sidebar:
        st.header("⚙️ DQN parameters")
        episodes = st.slider("Episodes", 100, 2000, 600, 50)
        hidden = HIDDEN_CHOICES[st.selectbox("Hidden layers", list(HIDDEN_CHOICES), index=1)]
        lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)
        gamma = st.slider("Discount γ", 0.80, 0.999, 0.99, 0.005)
        batch_size = st.select_slider("Batch size", [32, 64, 128, 256], value=64)
        eps_end = st.slider("ε end", 0.0, 0.3, 0.05, 0.01)
        expl = st.slider("Exploration fraction", 0.1, 1.0, 0.5, 0.05)
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
        st.pyplot(_draw(env, env.render_state() | {"event": "Kick-off"}), clear_figure=True)
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
        st.session_state["room4"] = {"result": result, "env_config": env_config}

    store = st.session_state.get("room4")
    if isinstance(store, dict) and "result" in store:
        result = store["result"]
        trained_env = FootballEnv(**store["env_config"])
    else:
        result = store
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
        episode_replay_player(
            result,
            lambda f: _draw(trained_env, f),
            key="room4_all_episodes",
            caption="Pick any DQN training episode; exploratory shots include the animated ball flight.",
            frame_transform=_expand_flight,
        )
