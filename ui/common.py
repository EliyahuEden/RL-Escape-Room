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
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from rl.utils import TrainResult, moving_average
from ui.render import metrics_frame

# replay animation speed → per-frame delay (seconds)
_SPEEDS = {"0.5×": 0.16, "1×": 0.08, "2×": 0.04, "4×": 0.02}


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

    c1, c2 = st.columns([1, 1])
    play = c1.button("▶ Animate episode", key=f"{key}_play", use_container_width=True)
    delay = _SPEEDS[c2.select_slider("Speed", options=list(_SPEEDS), value="1×",
                                     key=f"{key}_speed", label_visibility="collapsed")]
    if play:
        for i in range(len(traj)):
            show(i)
            time.sleep(delay)
    else:
        show(idx)


def sequence_replay_player(
    trajectories: Sequence[List[dict]],
    draw_fn: Callable[[dict], object],
    key: str,
    item_label: str = "Episode",
    caption: str = "",
    frame_transform: Optional[Callable[[List[dict]], List[dict]]] = None,
    metrics: Optional[Callable[[int], str]] = None,
) -> None:
    """Replay any indexed sequence of trajectories with an item and step slider."""
    if not trajectories:
        st.info("Train the room first to record replay episodes.")
        return
    if caption:
        st.caption(caption)

    item_idx = st.slider(item_label, 1, len(trajectories), len(trajectories), key=f"{key}_item") - 1
    if metrics is not None:
        st.caption(metrics(item_idx))

    traj = trajectories[item_idx]
    if frame_transform is not None:
        traj = frame_transform(traj)
    if not traj:
        st.warning(f"This {item_label.lower()} has no frames.")
        return

    last = len(traj) - 1
    idx = st.slider("Step", 0, last, last, key=f"{key}_step")
    placeholder = st.empty()
    info = st.empty()

    def show(i: int) -> None:
        frame = traj[i]
        placeholder.pyplot(draw_fn(frame), clear_figure=True)
        msg = f"{item_label} {item_idx + 1}/{len(trajectories)} · Step {i}/{last}"
        if "reward" in frame:
            msg += f" · step reward {frame['reward']:+.1f}"
        if "cum_reward" in frame:
            msg += f" · cumulative {frame['cum_reward']:+.1f}"
        info.write(msg)

    c1, c2 = st.columns([1, 1])
    play = c1.button(f"▶ Animate {item_label.lower()}", key=f"{key}_play",
                     use_container_width=True)
    delay = _SPEEDS[c2.select_slider("Speed", options=list(_SPEEDS), value="1×",
                                     key=f"{key}_speed", label_visibility="collapsed")]
    if play:
        for i in range(len(traj)):
            show(i)
            time.sleep(delay)
    else:
        show(idx)


def episode_replay_player(
    result: TrainResult,
    draw_fn: Callable[[dict], object],
    key: str,
    caption: str = "",
    frame_transform: Optional[Callable[[List[dict]], List[dict]]] = None,
) -> None:
    """Replay the actual trajectory from any training episode."""

    def metric_line(i: int) -> str:
        reward = result.episode_rewards[i] if i < len(result.episode_rewards) else 0.0
        steps = result.episode_steps[i] if i < len(result.episode_steps) else 0
        success = result.episode_success[i] if i < len(result.episode_success) else False
        eps = result.epsilon[i] if i < len(result.epsilon) else 0.0
        return f"Reward {reward:+.1f} · steps {steps} · success {success} · ε {eps:.3f}"

    sequence_replay_player(
        result.episode_replays,
        draw_fn,
        key,
        item_label="Episode",
        caption=caption,
        frame_transform=frame_transform,
        metrics=metric_line,
    )
