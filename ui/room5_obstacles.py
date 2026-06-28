"""Room 5 UI — Chicken road crossing solved with DQN + look-ahead sensors."""

from __future__ import annotations

import math

import streamlit as st
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle

from rl.algos import dqn
from rl.envs.obstacles import ObstacleEnv
from ui.common import episode_replay_player, learning_section, replay_player, train_with_live_progress
from ui.render import field_axes

HIDDEN_CHOICES = {"64, 64": (64, 64), "128, 128": (128, 128), "256, 128": (256, 128)}


def _rect_distance(px: float, py: float, x: float, y: float, w: float, h: float) -> float:
    closest_x = min(max(px, x - w / 2), x + w / 2)
    closest_y = min(max(py, y - h / 2), y + h / 2)
    return math.hypot(px - closest_x, py - closest_y)


def _draw_chicken(ax, px: float, py: float, vx: float, vy: float) -> None:
    # Point the beak in the last horizontal direction, defaulting toward the goal.
    facing = -1 if vx < 0 else 1
    ax.add_patch(Ellipse((px - 0.02 * facing, py), 0.38, 0.3, angle=8 * facing,
                         facecolor="#fff7ed", edgecolor="#9a6a38", lw=1.1, zorder=8))
    ax.add_patch(Circle((px + 0.18 * facing, py + 0.08), 0.12,
                        facecolor="#fff7ed", edgecolor="#9a6a38", lw=1, zorder=9))
    ax.add_patch(Polygon([(px + 0.28 * facing, py + 0.08),
                          (px + 0.43 * facing, py + 0.14),
                          (px + 0.43 * facing, py + 0.02)],
                         closed=True, facecolor="#f97316", edgecolor="#c2410c",
                         lw=0.8, zorder=10))
    ax.add_patch(Polygon([(px + 0.1 * facing, py + 0.19),
                          (px + 0.17 * facing, py + 0.31),
                          (px + 0.24 * facing, py + 0.18)],
                         closed=True, facecolor="#ef4444", edgecolor="#991b1b",
                         lw=0.8, zorder=10))
    ax.add_patch(Circle((px + 0.22 * facing, py + 0.1), 0.018,
                        facecolor="#111827", edgecolor="none", zorder=11))
    for dx in (-0.06, 0.08):
        ax.plot([px + dx, px + dx - 0.06 * facing],
                [py - 0.16, py - 0.27], color="#f97316", lw=1.2, zorder=8)
        ax.plot([px + dx - 0.06 * facing, px + dx - 0.13 * facing],
                [py - 0.27, py - 0.27], color="#f97316", lw=1.2, zorder=8)


def _draw_car(ax, car: dict, highlighted: bool) -> None:
    x, y = car["x"], car["y"]
    w, h = car["width"], car["height"]
    direction = 1 if car.get("direction", 1) >= 0 else -1
    color = "#fb923c" if highlighted else car.get("color", "#ef4444")
    edge = "#fde68a" if highlighted else "#111827"
    lw = 2.0 if highlighted else 1.0

    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, edgecolor=edge, lw=lw, zorder=5,
    ))
    windshield_y = y + direction * h * 0.18
    ax.add_patch(Rectangle((x - w * 0.28, windshield_y - h * 0.11),
                           w * 0.56, h * 0.2,
                           facecolor="#bae6fd", edgecolor="#0369a1",
                           lw=0.7, zorder=6))
    for wx in (x - w * 0.46, x + w * 0.46):
        for wy in (y - h * 0.3, y + h * 0.3):
            ax.add_patch(Circle((wx, wy), 0.055,
                                facecolor="#111827", edgecolor="#020617",
                                lw=0.5, zorder=6))
    ax.add_patch(Polygon([(x, y + direction * h * 0.47),
                          (x - w * 0.12, y + direction * h * 0.34),
                          (x + w * 0.12, y + direction * h * 0.34)],
                         closed=True, facecolor="#f8fafc",
                         edgecolor="none", alpha=0.85, zorder=7))


def _draw(env: ObstacleEnv, frame: dict):
    fig, ax = field_axes(env.W, env.H, grass="#4b5563")
    px, py = frame["x"], frame["y"]
    rng = frame["sensor_range"]
    gx = frame["goal_x"]
    road_min = frame.get("road_x_min", env.road_x_min)
    road_max = frame.get("road_x_max", env.road_x_max)
    lane_xs = frame.get("lane_xs", [])

    ax.add_patch(Rectangle((0, 0), road_min, env.H, facecolor="#fbbf24", zorder=1))
    ax.add_patch(Rectangle((gx, 0), env.W - gx, env.H, facecolor="#86efac", zorder=1))
    ax.add_patch(Rectangle((road_min, 0), road_max - road_min, env.H,
                           facecolor="#4b5563", zorder=1))
    ax.text(0.08, env.H / 2, "START", rotation=90, va="center", ha="left",
            fontsize=9, color="#7c2d12", weight="bold", zorder=2)
    ax.text(gx + 0.04, env.H / 2, "GOAL", rotation=90, va="center", ha="left",
            fontsize=9, color="#14532d", weight="bold", zorder=2)

    if len(lane_xs) > 1:
        lane_edges = [(a + b) / 2 for a, b in zip(lane_xs[:-1], lane_xs[1:])]
        for idx, x in enumerate(lane_edges):
            marker_color = "#fef08a" if idx == len(lane_edges) // 2 else "#f8fafc"
            y = 0.2
            while y < env.H:
                ax.add_patch(Rectangle((x - 0.035, y), 0.07, 0.48,
                                       facecolor=marker_color, edgecolor="none",
                                       alpha=0.95, zorder=2))
                y += 1.0

    ax.add_patch(Circle((px, py), rng, facecolor="#bfdbfe", alpha=0.15,
                        edgecolor="#38bdf8", ls="--", lw=1.1, zorder=3))

    cars = frame.get("cars")
    if cars is None:
        cars = [
            {"x": ox, "y": oy, "direction": 1, "speed": 0.0,
             "width": 0.55, "height": 1.05, "color": "#ef4444"}
            for ox, oy in frame.get("obstacles", [])
        ]
    for car in cars:
        dist = _rect_distance(px, py, car["x"], car["y"], car["width"], car["height"])
        _draw_car(ax, car, highlighted=dist <= rng)

    _draw_chicken(ax, px, py, frame.get("vx", 0.0), frame.get("vy", 0.0))
    if frame.get("vx") or frame.get("vy"):
        ax.arrow(px, py, frame["vx"] * 0.55, frame["vy"] * 0.55,
                 head_width=0.16, color="#78350f", alpha=0.8, zorder=7)

    ax.set_title(frame.get("event", "Chicken crossing"), fontsize=11)
    fig.tight_layout(pad=0.2)
    return fig


def render() -> None:
    st.title("🐔 Room 5 — Cross the Road · DQN + sensors")
    st.markdown(
        "Guide the chicken from the left sidewalk to the far edge of the road. "
        "Cars stream vertically through alternating lanes and wrap in from off-map, "
        "so the agent must learn when to move up, down, left, right, or wait. "
        "The dashed circle shows the sensor range; cars inside it are highlighted."
    )

    with st.sidebar:
        st.header("⚙️ DQN parameters")
        episodes = st.slider("Episodes", 100, 2500, 800, 50)
        hidden = HIDDEN_CHOICES[st.selectbox("Hidden layers", list(HIDDEN_CHOICES), index=1)]
        lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)
        gamma = st.slider("Discount γ", 0.80, 0.999, 0.99, 0.005)
        eps_end = st.slider("ε end", 0.0, 0.3, 0.05, 0.01)
        expl = st.slider("Exploration fraction", 0.1, 1.0, 0.5, 0.05)
        max_steps = st.slider("Max steps / episode", 120, 800, 350, 50)
        seed = st.number_input("Random seed", value=0, step=1)
        st.markdown("**Road / observation**")
        n_cars = st.slider("Number of cars", 4, 24, 14)
        traffic_speed = st.slider("Traffic speed", 0.7, 2.4, 1.35, 0.05)
        chicken_speed = st.slider("Chicken speed", 1.0, 2.6, 1.6, 0.1)
        sensor_range = st.slider("Sensor range (m)", 1.0, 6.0, 3.5, 0.5)
        max_sensed = st.slider("Cars sensed", 1, 10, 6)
        train = st.button("🧠 Train DQN", type="primary", use_container_width=True)

    env_config = dict(n_cars=n_cars, sensor_range=sensor_range,
                      max_sensed=max_sensed, car_speed=traffic_speed,
                      player_speed=chicken_speed, randomize=True)

    def make_env(s=None):
        return ObstacleEnv(**env_config, seed=s)

    env = make_env(int(seed))
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Road crossing")
        st.pyplot(_draw(env, env.render_state() | {"event": "Start"}), clear_figure=True)
        st.caption(f"{n_cars} cars · traffic {traffic_speed:.2f} m/s · "
                   f"sensor range {sensor_range} m · sensing {max_sensed} nearest")

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
            m1.metric("Crossing rate (last 50)", f"{result.success_rate(50):.0%}")
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
            caption="Pick any DQN training episode and watch the chicken react to moving traffic.",
        )

        st.subheader("🎲 Test the learned policy on brand-new traffic")
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
            replay_player({"Random traffic": test_frames},
                          lambda f: _draw(trained_env, f), key="room5_test",
                          caption=f"Greedy policy in unseen traffic — "
                                  f"ended: {test_frames[-1].get('event', '—')}")
