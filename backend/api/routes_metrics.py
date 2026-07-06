"""Saved metrics + policy views.

    GET /api/metrics/{room_id}    full saved metric series + summary
    GET /api/policy/{room_id}     value heatmap / policy arrows (rooms 1-3)
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.routes_rooms import check_room, clean
from backend.utils.config import policy_path
from backend.utils.metrics import load_metrics
from backend.utils.serialization import load_json

router = APIRouter()


@router.get("/metrics/{room_id}")
def metrics(room_id: int):
    check_room(room_id)
    data = load_metrics(room_id)
    if data is None:
        return clean({"trained": False})
    return clean({"trained": True, **data})


@router.get("/policy/{room_id}")
def policy(room_id: int):
    check_room(room_id)
    data = load_json(policy_path(room_id), default=None)
    return clean({"available": data is not None, "policy": data})
