"""Reusable Streamlit building blocks shared by every room:

* :func:`train_with_live_progress` -- runs a training callable while showing a
  live progress bar + reward curve.
* :func:`learning_section`          -- the standard set of learning/exploration
  graphs from a :class:`~rl.utils.TrainResult`.
* :func:`replay_player`             -- snapshot picker + step slider + animate
  button to replay recorded episodes.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List

import pandas as pd
import streamlit as st

from rl.utils import TrainResult, moving_average
from ui.render import metrics_frame


def train_with_live_progress(train_callable: Callable, update_every: int = 25) -> TrainResult:
    """Run ``train_callable(progress_cb)`` showing a live progress bar + curve."""
    bar = st.progress(0.0, text="Training…")
    live = st.empty()

    def cb(i: int, total: int, res: TrainResult) -> None:
        frac = 0.0 if total <= 0 else min(1.0, i / total)
        bar.progress(frac, text=f"Episode {i}/{total}")
        if res.episode_rewards and i % update_every == 0:
            live.line_chart(
                pd.DataFrame(
                    {
                        "reward": res.episode_rewards,
                        "smoothed": list(moving_average(res.episode_rewards, 50)),
                    }
                ),
                height=220,
            )

    result = train_callable(cb)
    bar.progress(1.0, text="Training complete")
    live.empty()
    return result


def learning_section(result: TrainResult, smooth_window: int = 50) -> None:
    """Render the standard learning + exploration graphs."""
    if result.num_episodes == 0:
        return
    df = metrics_frame(result, smooth_window)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Episode reward (raw + smoothed)")
        st.line_chart(df[["reward", "reward_smoothed"]], height=240)
    with c2:
        st.caption("Steps to finish the room")
        st.line_chart(df[["steps"]], height=240)

    c3, c4 = st.columns(2)
    with c3:
        st.caption("Success rate (rolling)")
        st.line_chart(df[["success_rate"]], height=240)
    with c4:
        if df["epsilon"].abs().sum() > 0:
            st.caption("Exploration rate ε (decay)")
            st.line_chart(df[["epsilon"]], height=240)
        else:
            st.caption("Exploration")
            st.write("This algorithm uses no ε-greedy exploration.")

    for name, series in result.extra.items():
        if series:
            st.caption(name)
            st.line_chart(pd.DataFrame({name: series}), height=200)


def replay_player(
    snapshots: Dict[str, List[dict]],
    draw_fn: Callable[[dict], object],
    key: str,
    caption: str = "",
) -> None:
    """Replay recorded episodes: pick a snapshot, scrub or animate the frames."""
    if not snapshots:
        st.info("Train the room first to record replay snapshots.")
        return

    labels = list(snapshots.keys())
    label = st.selectbox("Training stage", labels, key=f"{key}_stage")
    traj = snapshots[label]
    if not traj:
        st.warning("This snapshot has no frames.")
        return
    if caption:
        st.caption(caption)

    last = len(traj) - 1
    idx = st.slider("Step", 0, last, last, key=f"{key}_step")
    placeholder = st.empty()
    info = st.empty()

    def show(i: int) -> None:
        frame = traj[i]
        placeholder.pyplot(draw_fn(frame), clear_figure=True)
        msg = f"Step {i}/{last}"
        if "reward" in frame:
            msg += f" • step reward {frame['reward']:+.1f}"
        if "cum_reward" in frame:
            msg += f" • cumulative {frame['cum_reward']:+.1f}"
        info.write(msg)

    if st.button("▶ Animate episode", key=f"{key}_play"):
        for i in range(len(traj)):
            show(i)
            time.sleep(0.08)
    else:
        show(idx)
