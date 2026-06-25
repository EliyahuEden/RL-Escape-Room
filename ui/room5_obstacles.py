"""Room 5 UI — Dynamic Obstacles solved with DQN + look-ahead sensors."""

from __future__ import annotations

import streamlit as st
from matplotlib.patches import Circle, Polygon, Rectangle

from rl.algos import dqn
from rl.envs.obstacles import ObstacleEnv
from ui.common import episode_replay_player, learning_section, replay_player, train_with_live_progress
from ui.render import field_axes

HIDDEN_CHOICES = {"64, 64": (64, 64), "128, 128": (128, 128), "256, 128": (256, 128)}


def _draw(env: ObstacleEnv, frame: dict):
    fig, ax = field_axes(env.W, env.H, grass="#eceff1")
    gx = frame["goal_x"]
    ax.add_patch(Rectangle((gx, 0), env.W - gx, env.H, facecolor="#66bb6a", alpha=0.6, zorder=1))
    ax.text(gx + 0.05, env.H / 2, "EXIT", rotation=90, va="center", ha="left",
            fontsize=10, color="#1b5e20")
    px, py = frame["x"], frame["y"]
    rng = frame["sensor_range"]
    ax.add_patch(Circle((px, py), rng, facecolor="#90caf9", alpha=0.15,
                        edgecolor="#42a5f5", ls="--", lw=1, zorder=1))
    import math
    for ox, oy in frame["obstacles"]:
        within = math.hypot(px - ox, py - oy) <= rng
        ax.add_patch(Circle((ox, oy), frame["obstacle_radius"],
                            facecolor="#455a64" if not within else "#ef6c00",
                            edgecolor="#212121", zorder=3))
        ax.add_patch(Polygon([(ox - 0.14, oy + 0.1), (ox - 0.18, oy - 0.04),
                              (ox - 0.05, oy - 0.15), (ox + 0.13, oy - 0.12),
                              (ox + 0.18, oy + 0.03), (ox + 0.06, oy + 0.15)],
                             closed=True,
                             facecolor="#78909c" if not within else "#ffb74d",
                             edgecolor="#263238", lw=0.7, zorder=4))
    ax.add_patch(Circle((px, py), env.player_radius, facecolor="#1e88e5",
                        edgecolor="#fff", lw=1.5, zorder=5))
    ax.add_patch(Circle((px, py - 0.07), 0.055, facecolor="#bbdefb", edgecolor="#0d47a1", lw=0.7, zorder=7))
    ax.plot([px, px], [py - 0.02, py + 0.09], color="white", lw=1.4, zorder=7)
    ax.plot([px, px - 0.1], [py + 0.03, py + 0.11], color="white", lw=1.4, zorder=7)
    ax.plot([px, px + 0.1], [py + 0.03, py + 0.1], color="white", lw=1.4, zorder=7)
    if frame["vx"] or frame["vy"]:
        ax.arrow(px, py, frame["vx"] * 0.6, frame["vy"] * 0.6, head_width=0.18,
                 color="#0d47a1", zorder=6)
    ax.set_title(frame.get("event", "Navigating"), fontsize=11)
    fig.tight_layout(pad=0.2)
    return fig


def render() -> None:
    st.title("🚧 Room 5 — Dynamic Obstacles · DQN + sensors")
    st.markdown(
        "Cross to the green **EXIT** while avoiding the 0.5 m obstacles. A **new "
        "random layout is generated every episode**, so the agent must learn a "
        "*reactive* policy from its **sensors** (the dashed circle = how far it can "
        "see). Obstacles inside the sensor range turn orange. Because the policy is "
        "sensor-based, you can drop it into a brand-new room at the end and watch it cope."
    )

    with st.sidebar:
        st.header("⚙️ DQN parameters")
        episodes = st.slider("Episodes", 100, 2000, 600, 50)
        hidden = HIDDEN_CHOICES[st.selectbox("Hidden layers", list(HIDDEN_CHOICES), index=1)]
        lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)
        gamma = st.slider("Discount γ", 0.80, 0.999, 0.99, 0.005)
        eps_end = st.slider("ε end", 0.0, 0.3, 0.05, 0.01)
        expl = st.slider("Exploration fraction", 0.1, 1.0, 0.5, 0.05)
        max_steps = st.slider("Max steps / episode", 100, 600, 300, 50)
        seed = st.number_input("Random seed", value=0, step=1)
        st.markdown("**Room / observation**")
        n_obs = st.slider("Number of obstacles", 1, 12, 6)
        sensor_range = st.slider("Sensor range (m ahead)", 1.0, 6.0, 3.0, 0.5)
        max_sensed = st.slider("Obstacles sensed", 1, 6, 4)
        train = st.button("🧠 Train DQN", type="primary", use_container_width=True)

    env_config = dict(n_obstacles=n_obs, sensor_range=sensor_range,
                      max_sensed=max_sensed, randomize=True)

    def make_env(s=None):
        return ObstacleEnv(**env_config, seed=s)

    env = make_env(int(seed))
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Room (random each episode)")
        st.pyplot(_draw(env, env.render_state() | {"event": "Start"}), clear_figure=True)
        st.caption(f"{n_obs} obstacles · sensor range {sensor_range} m · "
                   f"sensing {max_sensed} nearest")

    if train:
        result = train_with_live_progress(
            lambda cb: dqn.train(make_env(int(seed)), episodes=episodes, hidden=hidden,
                                 lr=lr, gamma=gamma, eps_end=eps_end,
                                 exploration_fraction=expl, learn_start=2000,
                                 target_update=1000, max_steps=max_steps,
                                 seed=int(seed), progress_cb=cb),
            update_every=10,
        )
        st.session_state["room5"] = {
            "result": result,
            "env_config": env_config,
            "training_seed": int(seed),
        }
        st.session_state.pop("room5_test", None)

    store = st.session_state.get("room5")
    if isinstance(store, dict) and "result" in store:
        result = store["result"]
        trained_env = ObstacleEnv(**store["env_config"], seed=store["training_seed"])
    else:
        result = store
        trained_env = env
    with right:
        st.subheader("Result")
        if not result:
            st.info("Set parameters and press **Train DQN**. (Training takes ~1–2 min.)")
        else:
            m1, m2 = st.columns(2)
            m1.metric("Exit rate (last 50)", f"{result.success_rate(50):.0%}")
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
            key="room5_all_episodes",
            caption="Pick any DQN training episode from the random rooms generated during training.",
        )

        st.subheader("🎲 Test the learned policy on a brand-new random room")
        c1, c2 = st.columns([1, 3])
        with c1:
            test_seed = st.number_input("Test seed", value=999, step=1)
            if st.button("Generate & test"):
                test_config = store["env_config"] if isinstance(store, dict) and "env_config" in store else env_config
                test_env = ObstacleEnv(**test_config, seed=int(test_seed))
                frames = dqn.greedy_rollout(test_env, result.policy, max_steps)
                st.session_state["room5_test"] = frames
        test_frames = st.session_state.get("room5_test")
        if test_frames:
            replay_player({"Random test room": test_frames},
                          lambda f: _draw(trained_env, f), key="room5_test",
                          caption=f"Greedy policy in an unseen room — "
                                  f"ended: {test_frames[-1].get('event', '—')}")
