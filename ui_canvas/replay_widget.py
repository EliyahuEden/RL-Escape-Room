"""Streamlit widget that replaces the Matplotlib replay player with Canvas.

Provides episode selection + Canvas animation for any room.
"""

from __future__ import annotations

from typing import Callable, List

import streamlit as st

from rl.utils import TrainResult


def canvas_episode_replay(
    result: TrainResult,
    frame_converter: Callable[[dict], dict],
    canvas_replay_fn: Callable[[list, str], None],
    key: str,
    caption: str = "",
) -> None:
    """Canvas-based episode replay widget.

    ``frame_converter`` converts a raw training frame dict to canvas format.
    ``canvas_replay_fn`` is called with (converted_frames, key) to render.
    """
    if not result.episode_replays:
        st.info("Train the room first to record replay episodes.")
        return
    if caption:
        st.caption(caption)

    n = len(result.episode_replays)
    ep_idx = st.slider("Episode", 1, n, n, key=f"{key}_ep") - 1

    reward = result.episode_rewards[ep_idx] if ep_idx < len(result.episode_rewards) else 0
    steps = result.episode_steps[ep_idx] if ep_idx < len(result.episode_steps) else 0
    success = result.episode_success[ep_idx] if ep_idx < len(result.episode_success) else False
    eps = result.epsilon[ep_idx] if ep_idx < len(result.epsilon) else 0
    st.caption(f"Reward {reward:+.1f} · {steps} steps · {'✓' if success else '✗'} · ε {eps:.3f}")

    raw_frames = result.episode_replays[ep_idx]
    if not raw_frames:
        st.warning("This episode has no frames.")
        return

    canvas_frames = [frame_converter(f) for f in raw_frames]
    canvas_replay_fn(canvas_frames, f"{key}_{ep_idx}")
