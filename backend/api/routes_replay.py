"""Episode replay routes.

    GET /api/replay/{room_id}                list recorded episodes
    GET /api/replay/{room_id}/{episode_id}   full frame-by-frame replay JSON
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.routes_rooms import check_room, clean
from backend.training.replay_recorder import ReplayRecorder

router = APIRouter()


@router.get("/replay/{room_id}")
def list_replays(room_id: int):
    check_room(room_id)
    return clean({"episodes": ReplayRecorder.list_episodes(room_id)})


@router.get("/replay/{room_id}/{episode_id}")
def get_replay(room_id: int, episode_id: str):
    check_room(room_id)
    data = ReplayRecorder.load_replay(room_id, episode_id)
    if data is None:
        raise HTTPException(404, "replay not found")
    return clean(data)
