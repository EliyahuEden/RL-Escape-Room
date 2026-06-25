"""Room 1 UI — Pacman solved with Dynamic Programming."""

from __future__ import annotations

import random
from typing import Dict, List

import pandas as pd
import streamlit as st

from rl.algos.dp import greedy_policy, policy_iteration, value_iteration
from rl.envs.pacman import PacmanEnv
from ui.common import replay_player
from ui.render import render_grid

SLIP_COLOR = "#bfe3ff"
COIN_COLOR = "#f5b301"
START_COLOR = "#d7f0d7"


def _record_rollout(env: PacmanEnv, policy: Dict, seed: int = 0,
                    max_steps: int = 300) -> List[dict]:
    env.rng = random.Random(seed)
    env.reset()
    frames, cum, done = [], 0.0, False
    state = (env.pos, env.mask)
    frames.append({"pos": env.pos, "remaining": env.remaining_coins(env.mask),
                   "step": 0, "reward": 0.0, "cum_reward": 0.0, "escaped": False})
    steps = 0
    while not done and steps < max_steps:
        action = policy.get(state, 0)
        state, reward, done, info = env.step(action)
        cum += reward
        steps += 1
        frames.append({"pos": env.pos, "remaining": env.remaining_coins(env.mask),
                       "step": steps, "reward": reward, "cum_reward": cum,
                       "escaped": info.get("escaped", False),
                       "slipped": info.get("slipped", False)})
    return frames


def _draw(env: PacmanEnv, frame: dict):
    styles: Dict = {}
    for cell in env.slippery:
        styles[cell] = {"color": SLIP_COLOR, "label": "❄", "label_color": "#1565c0"}
    styles[env.start] = {"color": START_COLOR, "label": "S", "label_color": "#2e7d32"}
    remaining = set(frame["remaining"])
    door_open = len(remaining) == 0
    styles[env.door] = {"color": "#7CFC00" if door_open else "#d9a441",
                        "label": "EXIT", "label_color": "#222"}
    for coin in remaining:
        styles[coin] = {"color": styles.get(coin, {}).get("color", "#f4f4f4"),
                        "label": "●", "label_color": COIN_COLOR}
    title = f"Coins left: {len(remaining)}"
    if frame.get("escaped"):
        title = "🏆 Escaped!"
    return render_grid(env.rows, env.cols, walls=env.walls, cell_styles=styles,
                       agent=frame["pos"], title=title)


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
        slip_prob = st.slider("Slip probability", 0.0, 0.5, 0.15, 0.05)
        with st.expander("Reward shaping"):
            r_coin = st.number_input("Coin reward", value=10.0, step=1.0)
            r_exit = st.number_input("Exit reward", value=100.0, step=10.0)
            r_step = st.number_input("Step cost", value=-1.0, step=1.0)
            r_door_early = st.number_input("Early-door penalty", value=-10.0, step=1.0)
            r_slip = st.number_input("Slip penalty", value=-5.0, step=1.0)
        solve = st.button("🧠 Solve room (DP)", type="primary", use_container_width=True)

    env = PacmanEnv(slip_prob=slip_prob, r_step=r_step, r_coin=r_coin, r_exit=r_exit,
                    r_door_early=r_door_early, r_slip=r_slip)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Maze")
        st.pyplot(_draw(env, {"pos": env.start, "remaining": env.coins,
                              "escaped": False}), clear_figure=True)
        st.caption(f"{len(env.coins)} coins · {len(env.slippery)} icy cells · "
                   f"start S → door EXIT")

    if solve:
        mdp = env.build_mdp()
        with st.spinner(f"Running {method} over {len(mdp.states)} states…"):
            if method == "Value Iteration":
                V, policy, diag = value_iteration(mdp, gamma, theta, max_iter=2000)
                # intermediate policies for replay snapshots
                V1, p1, _ = value_iteration(mdp, gamma, theta, max_iter=1)
                V5, p5, _ = value_iteration(mdp, gamma, theta, max_iter=5)
                stages = [("After 1 sweep", p1), ("After 5 sweeps", p5),
                          ("Optimal (converged)", policy)]
            else:
                V, policy, diag = policy_iteration(mdp, gamma, theta, max_iter=200)
                stages = [("Final policy", policy)]
        snapshots = {label: _record_rollout(env, pol) for label, pol in stages}
        st.session_state["room1"] = {
            "diag": diag, "start_value": V[mdp.start], "n_states": len(mdp.states),
            "snapshots": snapshots, "method": method, "slip_prob": slip_prob,
        }

    data = st.session_state.get("room1")
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

        st.subheader("🎬 Replay the policy at different stages")
        replay_player(data["snapshots"], lambda f: _draw(env, f), key="room1",
                      caption="Watch the DP policy improve from a single sweep to optimal.")
