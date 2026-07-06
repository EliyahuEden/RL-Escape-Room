"""RL Escape Room — web backend.

FastAPI wrapper around the existing :mod:`rl` package.  The RL logic
(environments + algorithms) lives in ``rl/`` and is the single source of
truth; this package only orchestrates training jobs, records metrics and
replays as JSON, and serves them to the React frontend.
"""
