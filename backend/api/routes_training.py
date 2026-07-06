"""Training control + evaluation.

    POST /api/train/{room_id}           start a background training job
    GET  /api/train/{room_id}/status    live progress (polled by the UI)
    POST /api/train/{room_id}/stop      request stop
    POST /api/evaluate/{room_id}        greedy evaluation of the saved model
    GET  /api/results/summary           cross-room comparison
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.routes_rooms import check_room, clean
from backend.api.state import MANAGER
from backend.training import evaluate as ev

router = APIRouter()


class TrainRequest(BaseModel):
    params: dict = {}


class EvaluateRequest(BaseModel):
    episodes: int = 10


@router.post("/train/{room_id}")
def train(room_id: int, req: TrainRequest):
    check_room(room_id)
    if not MANAGER.start(room_id, req.params):
        raise HTTPException(409, "training already running for this room")
    return {"ok": True, "room": room_id, "state": "running"}


@router.get("/train/{room_id}/status")
def train_status(room_id: int):
    check_room(room_id)
    return clean(MANAGER.status(room_id))


@router.post("/train/{room_id}/stop")
def train_stop(room_id: int):
    check_room(room_id)
    return {"ok": MANAGER.stop(room_id)}


@router.post("/evaluate/{room_id}")
def evaluate(room_id: int, req: EvaluateRequest):
    check_room(room_id)
    if MANAGER.is_running(room_id):
        raise HTTPException(409, "room is currently training")
    try:
        result = ev.evaluate_room(room_id,
                                  n_episodes=max(1, min(req.episodes, 50)))
        return clean({"ok": True, "result": result})
    except ev.NotTrainedError as exc:
        raise HTTPException(400, str(exc))


@router.get("/results/summary")
def summary():
    return clean(ev.results_summary())
