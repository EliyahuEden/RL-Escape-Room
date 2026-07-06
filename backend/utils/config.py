"""Filesystem layout for training artefacts.

Everything the backend produces is written under ``results/`` so the
frontend (and the grader) can inspect plain JSON files:

    results/metrics/room{N}.json        per-episode series + summary
    results/replays/room{N}/*.json      frame-by-frame episode replays
    results/models/room{N}_*.pkl|.pt    trained Q-tables / networks
    results/policies/room{N}.json       value heatmap + policy arrows
    results/configs/room{N}.json        saved hyperparameter overrides
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
REPLAYS_DIR = RESULTS_DIR / "replays"
MODELS_DIR = RESULTS_DIR / "models"
POLICIES_DIR = RESULTS_DIR / "policies"
CONFIGS_DIR = RESULTS_DIR / "configs"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, METRICS_DIR, REPLAYS_DIR, MODELS_DIR,
              POLICIES_DIR, CONFIGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def metrics_path(room_id: int) -> Path:
    return METRICS_DIR / f"room{room_id}.json"


def replays_dir(room_id: int) -> Path:
    return REPLAYS_DIR / f"room{room_id}"


def model_path(room_id: int, ext: str = "pkl") -> Path:
    return MODELS_DIR / f"room{room_id}_model.{ext}"


def info_path(room_id: int) -> Path:
    return MODELS_DIR / f"room{room_id}_info.json"


def policy_path(room_id: int) -> Path:
    return POLICIES_DIR / f"room{room_id}.json"


def config_path(room_id: int) -> Path:
    return CONFIGS_DIR / f"room{room_id}.json"
