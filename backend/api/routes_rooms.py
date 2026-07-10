"""Room catalogue + per-room detail + config storage.

    GET  /api/rooms                room list (cards): status, best reward, ...
    GET  /api/rooms/{room_id}      full detail: schema, layout, action names
    GET  /api/config/{room_id}     hyperparameter schema + effective values
    POST /api/config/{room_id}     persist hyperparameter overrides
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.api.state import MANAGER
from backend.training import evaluate as ev
from backend.training import frames as fr
from backend.training import train as tr
from backend.utils.config import config_path, info_path
from backend.utils.serialization import clean_jsonable, load_json, save_json

router = APIRouter()


def clean(payload) -> JSONResponse:
    return JSONResponse(clean_jsonable(payload))


def check_room(room_id: int) -> None:
    if room_id not in tr.ROOMS:
        raise HTTPException(404, f"room {room_id} does not exist")


def _card(room_id: int, summary_row: dict) -> dict:
    meta = tr.ROOMS[room_id]
    return {
        "id": room_id,
        "name": meta["name"],
        "subtitle": meta["subtitle"],
        "algorithm": meta["algorithm"],
        "algo_id": meta["algo_id"],
        "difficulty": meta["difficulty"],
        "type": meta["type"],
        "accent": meta["accent"],
        "icon": meta["icon"],
        "description": meta["description"],
        "state": meta["state"],
        "actions": meta["actions"],
        "rewards": meta["rewards"],
        "training": MANAGER.is_running(room_id),
        **{k: summary_row.get(k) for k in
           ("trained", "episodes", "best_reward", "avg_reward_last50",
            "success_rate", "eval_success_rate", "train_time", "timestamp")},
    }


@router.get("/rooms")
def rooms():
    summary = {r["id"]: r for r in ev.results_summary()["rooms"]}
    return clean({"rooms": [_card(rid, summary[rid]) for rid in tr.ROOMS]})


@router.get("/rooms/{room_id}")
def room_detail(room_id: int):
    check_room(room_id)
    summary = {r["id"]: r for r in ev.results_summary()["rooms"]}
    info = load_json(info_path(room_id), default=None)
    params = tr.merge_params(room_id, (info or {}).get("params"))
    mode = params.get("mode")
    detail = _card(room_id, summary[room_id])
    detail.update({
        "params": tr.ROOMS[room_id]["params"],
        "values": params,
        "layout": tr.room_layout(room_id, params),
        "action_names": fr.action_names(room_id, mode),
    })
    return clean(detail)


class PreviewRequest(BaseModel):
    values: dict = {}


@router.post("/rooms/{room_id}/preview")
def preview_layout(room_id: int, req: PreviewRequest):
    """Build the layout for a set of (unsaved) hyperparameter values so the UI
    can show a live preview of a generated map before training."""
    check_room(room_id)
    params = tr.merge_params(room_id, req.values)
    return clean({
        "layout": tr.room_layout(room_id, params),
        "values": params,
        "action_names": fr.action_names(room_id, params.get("mode")),
    })


class ConfigRequest(BaseModel):
    values: dict = {}


@router.get("/config/{room_id}")
def get_config(room_id: int):
    check_room(room_id)
    return clean({
        "schema": tr.ROOMS[room_id]["params"],
        "values": tr.merge_params(room_id, None),
        "saved": load_json(config_path(room_id), default={}),
    })


@router.post("/config/{room_id}")
def set_config(room_id: int, req: ConfigRequest):
    check_room(room_id)
    merged = tr.merge_params(room_id, req.values)
    keep = {k: merged[k] for k in req.values if k in merged}
    save_json(config_path(room_id), keep)
    return clean({"ok": True, "saved": keep, "effective": merged})
