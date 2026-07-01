"""Helper to check if Canvas rendering mode is active."""

import streamlit as st


def use_canvas() -> bool:
    return bool(st.session_state.get("canvas_mode", False))
