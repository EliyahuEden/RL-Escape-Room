"""Room 1 UI — Pacman solved with Dynamic Programming."""

from __future__ import annotations

import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from rl.algos.dp import policy_iteration, value_iteration
from rl.envs.grid_base import DOWN, LEFT, RIGHT, UP
from rl.envs.pacman import DEFAULT_LAYOUT, PacmanEnv, generate_pacman_layout
from ui.common import sequence_replay_player
from ui.render import render_grid
from ui_canvas.toggle import use_canvas
from ui_canvas.room1_pacman import draw_pacman_canvas, replay_pacman_canvas, frame_to_canvas
from ui_canvas.canvas_core import render_canvas_replay
from ui_canvas.room1_pacman import DRAW_JS as PACMAN_DRAW_JS

_ARROW = {UP: (0.0, -0.3), DOWN: (0.0, 0.3), LEFT: (-0.3, 0.0), RIGHT: (0.3, 0.0)}


def _slice_state(env: PacmanEnv, cell):
    """The DP state for ``cell`` in the start configuration (no coins taken,
    guard at its starting position/phase) — the slice we visualise."""
    if not env.guard_enabled:
        return (cell, env.full_mask)
    guard_rep = env.guard_start if env.guard_mode == "chase" else 0
    return (cell, env.full_mask, guard_rep)


def _value_policy_fig(env: PacmanEnv, V: Dict, policy: Dict):
    """Final value heatmap + greedy-policy arrows over the maze."""
    grid = np.full((env.rows, env.cols), np.nan)
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in env.walls:
                continue
            s = _slice_state(env, (r, c))
            if s in V:
                grid[r, c] = V[s]
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    im = ax.imshow(grid, cmap="viridis")
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in env.walls or (r, c) == env.door:
                continue
            s = _slice_state(env, (r, c))
            a = policy.get(s)
            if a is None:
                continue
            dx, dy = _ARROW[a]
            ax.arrow(c, r, dx, dy, head_width=0.16, head_length=0.14,
                     fc="white", ec="#111", lw=0.5, length_includes_head=True, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Value heatmap + greedy policy (start config)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.3)
    return fig

SLIP_COLOR = "#bfe3ff"
COIN_COLOR = "#f5b301"
START_COLOR = "#d7f0d7"
MAX_DP_STATES = 80_000


def _estimate_state_count(env: PacmanEnv) -> int:
    free_cells = sum(
        1
        for r in range(env.rows)
        for c in range(env.cols)
        if (r, c) not in env.walls
    )
    masks = 1 << len(env.coins)
    if not env.guard_enabled:
        guard_multiplier = 1
    elif env.guard_mode == "chase":
        guard_multiplier = free_cells          # guard position carried in the state
    else:
        guard_multiplier = env.guard_period    # patrol phase carried in the state
    return free_cells * masks * guard_multiplier


def _record_rollout(env: PacmanEnv, policy: Dict, seed: int = 0,
                    max_steps: int = 300) -> List[dict]:
    env.rng = random.Random(seed)
    state = env.reset()
    frames, cum, done = [], 0.0, False
    frames.append({"pos": env.pos, "remaining": env.remaining_coins(env.mask),
                   "guard": env.guard_pos if env.guard_enabled else None,
                   "step": 0, "reward": 0.0, "cum_reward": 0.0,
                   "escaped": False, "caught": False})
    steps = 0
    while not done and steps < max_steps:
        action = policy.get(state, 0)
        state, reward, done, info = env.step(action)
        cum += reward
        steps += 1
        frames.append({"pos": env.pos, "remaining": env.remaining_coins(env.mask),
                       "guard": env.guard_pos if env.guard_enabled else None,
                       "step": steps, "reward": reward, "cum_reward": cum,
                       "action": action,
                       "escaped": info.get("escaped", False),
                       "caught": info.get("caught", False),
                       "slipped": info.get("slipped", False)})
    return frames


def _draw(env: PacmanEnv, frame: dict):
    styles: Dict = {}
    for cell in env.slippery:
        styles[cell] = {"color": SLIP_COLOR, "icon": "ice", "label_color": "#1565c0"}
    styles[env.start] = {"color": START_COLOR, "icon": "home", "label_color": "#2e7d32"}
    remaining = set(frame["remaining"])
    door_open = len(remaining) == 0
    styles[env.door] = {"color": "#7CFC00" if door_open else "#d9a441",
                        "icon": "door", "label_color": "#222"}
    for coin in remaining:
        styles[coin] = {"color": styles.get(coin, {}).get("color", "#f4f4f4"),
                        "icon": "coin", "label_color": COIN_COLOR}
    markers = []
    if frame.get("guard") is not None:
        markers.append((frame["guard"], "ghost", "#c62828"))
    mouth_open = {
        0: "pacman_up",
        1: "pacman_down",
        2: "pacman_left",
        3: "pacman_right",
    }.get(frame.get("action"), "pacman_right")
    mouth = mouth_open if frame.get("step", 0) % 2 == 0 else "pacman_closed"
    title = f"Coins left: {len(remaining)}"
    if frame.get("escaped"):
        title = "Escaped!"
    if frame.get("caught"):
        title = "Caught by the guard"
    return render_grid(env.rows, env.cols, walls=env.walls, cell_styles=styles,
                       agent=frame["pos"], agent_color="#ffd23f",
                       agent_symbol=mouth, agent_symbol_color="#111",
                       markers=markers, title=title)


def render() -> None:
    st.title("🟡 Room 1 — Pacman · Dynamic Programming")
    st.markdown(
        "The maze model is **fully known**, so we solve it exactly with Dynamic "
        "Programming. Collect **every coin** to unlock the door, then escape. "
        "Icy cells (❄) randomise the move with probability *slip*."
    )

    with st.sidebar:
        st.header("⚙️ DP parameters")
        method = st.selectbox("Algorithm", ["Value Iteration", "Policy Iteration"])
        gamma = st.slider("Discount γ", 0.50, 0.999, 0.95, 0.005)
        theta_exp = st.slider("Convergence threshold θ = 10^x", -6, -1, -4)
        theta = 10.0 ** theta_exp
        max_dp_iter = st.slider("Max DP iterations", 25, 500, 250, 25)
        st.header("Map")
        map_mode = st.selectbox("Map source", ["Generated", "Classic"], key="room1_map_mode")
        map_seed = st.number_input("Map seed", value=10, step=1, key="room1_map_seed")
        st.header("Guard")
        guard_enabled = st.checkbox("Enable guard", value=True)
        guard_behavior = st.selectbox("Guard behaviour",
                                      ["Chase (hunts you)", "Patrol (fixed loop)"],
                                      disabled=not guard_enabled)
        guard_mode = "chase" if guard_behavior.startswith("Chase") else "patrol"
        guard_speed = st.slider("Guard speed (steps/turn)", 1, 2, 1)
        patrol_len = st.slider("Patrol length", 2, 6, 5,
                               disabled=guard_mode != "patrol")
        # a chasing guard carries its position in the DP state (free_cells×),
        # a patrol guard only its phase — so chase needs fewer coins to stay small
        coin_max = (3 if guard_mode == "chase" else 5) if guard_enabled else 6
        if st.session_state.get("room1_coins", 3) > coin_max:
            st.session_state["room1_coins"] = coin_max
        default_coins = 2 if guard_mode == "chase" else min(3, coin_max)
        n_coins = st.slider("Coins", 2, coin_max, default_coins, key="room1_coins")
        if guard_mode == "chase":
            st.caption("🏃 The chase guard carries its **position** in the DP state, "
                       "so it's heavier — fewer coins keep the solve fast. Pick "
                       "**Patrol** for near-instant solves with more coins.")
        n_slippery = st.slider("Slippery tiles", 0, 18, 6)
        slip_prob = st.slider("Slip probability (sideways)", 0.0, 0.5, 0.20, 0.05)
        with st.expander("Reward shaping"):
            r_coin = st.number_input("Coin reward", value=10.0, step=1.0)
            r_exit = st.number_input("Exit reward", value=100.0, step=10.0)
            r_step = st.number_input("Step cost", value=-1.0, step=1.0)
            r_door_early = st.number_input("Early-door penalty", value=-10.0, step=1.0)
            r_slip = st.number_input("Slip penalty", value=-5.0, step=1.0)
            r_wall = st.number_input("Wall-hit penalty", value=-5.0, step=1.0)
            r_guard = st.number_input("Guard catch penalty", value=-50.0, step=5.0)
        solve = st.button("🧠 Solve room (DP)", type="primary", use_container_width=True)

    layout = (
        generate_pacman_layout(int(map_seed), n_coins, n_slippery, guard_enabled)
        if map_mode == "Generated"
        else DEFAULT_LAYOUT
    )
    env_config = dict(layout=layout, slip_prob=slip_prob, r_step=r_step,
                      r_coin=r_coin, r_exit=r_exit, r_door_early=r_door_early,
                      r_slip=r_slip, r_wall=r_wall, guard_enabled=guard_enabled,
                      guard_speed=guard_speed, guard_mode=guard_mode,
                      patrol_len=patrol_len, r_guard=r_guard)

    def make_env() -> PacmanEnv:
        return PacmanEnv(**env_config)

    env = make_env()
    estimated_states = _estimate_state_count(env)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Maze")
        _init_frame = {"pos": env.start, "remaining": env.coins,
                       "guard": env.guard_pos if env.guard_enabled else None,
                       "escaped": False}
        if use_canvas():
            draw_pacman_canvas(env, _init_frame)
        else:
            st.pyplot(_draw(env, _init_frame), clear_figure=True)
        st.caption(f"{len(env.coins)} coins · {len(env.slippery)} icy cells · "
                   f"{'guard enabled' if env.guard_enabled else 'no guard'} · start S → door EXIT")
        st.caption(f"Estimated DP states: {estimated_states:,}")
        if estimated_states > MAX_DP_STATES:
            st.warning(
                "This Room 1 setup is too large for interactive DP. "
                "Use fewer coins, disable the guard, or switch to a smaller map."
            )

    if solve and estimated_states > MAX_DP_STATES:
        st.error(
            f"Blocked a {estimated_states:,}-state DP run to keep the app responsive. "
            f"The interactive limit is {MAX_DP_STATES:,} states."
        )

    can_solve = (
        solve
        and estimated_states <= MAX_DP_STATES
    )

    if can_solve:
        mdp = env.build_mdp()
        iteration_replays: List[List[dict]] = []

        def collect_replay(i: int, _V: Dict, pol: Dict) -> None:
            iteration_replays.append(_record_rollout(make_env(), pol, seed=i))

        with st.spinner(f"Running {method} over {len(mdp.states)} states…"):
            if method == "Value Iteration":
                V, policy, diag = value_iteration(
                    mdp, gamma, theta, max_iter=max_dp_iter, iteration_cb=collect_replay
                )
                stages = {
                    "After 1 sweep": iteration_replays[0],
                    "After 5 sweeps": iteration_replays[min(4, len(iteration_replays) - 1)],
                    "Optimal (converged)": iteration_replays[-1],
                }
            else:
                V, policy, diag = policy_iteration(
                    mdp, gamma, theta, max_iter=max_dp_iter, iteration_cb=collect_replay
                )
                stages = {"Final policy": iteration_replays[-1]}
        st.session_state["room1"] = {
            "diag": diag, "start_value": V[mdp.start], "n_states": len(mdp.states),
            "snapshots": stages, "iteration_replays": iteration_replays,
            "method": method, "env_config": env_config, "V": V, "policy": policy,
        }

    data = st.session_state.get("room1")
    trained_env = PacmanEnv(**data["env_config"]) if data and "env_config" in data else env
    with right:
        st.subheader("Convergence")
        if not data:
            st.info("Set the parameters and press **Solve room (DP)**.")
        else:
            diag = data["diag"]
            m1, m2, m3 = st.columns(3)
            m1.metric("States", data["n_states"])
            m2.metric("Iterations", len(diag["value_delta"]))
            m3.metric("Start-state value", f"{data['start_value']:.2f}")
            final = data["snapshots"].get("Optimal (converged)") or \
                list(data["snapshots"].values())[-1]
            st.caption(f"Optimal greedy rollout: **{final[-1]['step']} steps**, "
                       f"return **{final[-1]['cum_reward']:+.0f}**.")

    if data:
        diag = data["diag"]
        st.subheader("📈 Learning / convergence graphs")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Bellman residual (max value change per sweep) — log scale")
            df = pd.DataFrame({"value_delta": diag["value_delta"]})
            df["value_delta"] = df["value_delta"].clip(lower=1e-9)
            st.line_chart(df, height=240)
        with c2:
            st.caption("Estimated value of the start state per sweep")
            st.line_chart(pd.DataFrame({"start_value": diag["start_value"]}), height=240)
        if diag["policy_changes"]:
            st.caption("Policy changes per sweep (→ 0 when the policy is stable)")
            st.line_chart(pd.DataFrame({"policy_changes": diag["policy_changes"]}),
                          height=200)

        if data.get("V") is not None:
            st.subheader("🗺️ Final value & policy")
            st.pyplot(_value_policy_fig(trained_env, data["V"], data["policy"]),
                      clear_figure=True)
            st.caption("Colour = optimal value of each cell at the start (no coins "
                       "collected, guard at patrol phase 0); arrows = the greedy "
                       "Dynamic-Programming action.")

        if data.get("iteration_replays"):
            st.subheader("🎞️ Replay every DP iteration")
            if use_canvas():
                n_iter = len(data["iteration_replays"])
                sel = st.slider("Iteration", 1, n_iter, n_iter, key="room1_iter_sel") - 1
                raw = data["iteration_replays"][sel]
                if raw:
                    replay_pacman_canvas(trained_env, raw, key=f"room1_canvas_{sel}")
            else:
                sequence_replay_player(
                    data["iteration_replays"],
                    lambda f: _draw(trained_env, f),
                    key="room1_all_iterations",
                    item_label="Iteration",
                    caption="Pick any DP sweep / policy improvement and animate its greedy rollout.",
            )
