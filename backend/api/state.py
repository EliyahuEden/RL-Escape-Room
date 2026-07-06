"""TrainingManager — one background training thread per room.

The trainer's progress callback pushes throttled, downsampled metric
series into the job dict under a lock; the frontend polls
``GET /api/train/{room_id}/status`` to draw live charts.
"""
from __future__ import annotations

import threading
import time
import traceback

from backend.training.train import run_training
from backend.utils.metrics import downsample


class TrainingManager:
    def __init__(self):
        self.jobs: dict[int, dict] = {}
        self.events: dict[int, threading.Event] = {}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    def is_running(self, room_id: int) -> bool:
        with self.lock:
            job = self.jobs.get(room_id)
            return bool(job and job["state"] == "running")

    def any_running(self) -> bool:
        with self.lock:
            return any(j["state"] == "running" for j in self.jobs.values())

    def status(self, room_id: int) -> dict:
        with self.lock:
            job = self.jobs.get(room_id)
            if not job:
                return {"state": "idle"}
            snap = dict(job)
            if job.get("started_at"):
                snap["elapsed"] = round(time.time() - job["started_at"], 1)
            return snap

    # ------------------------------------------------------------------
    def start(self, room_id: int, params: dict | None) -> bool:
        with self.lock:
            job = self.jobs.get(room_id)
            if job and job["state"] == "running":
                return False
            stop_event = threading.Event()
            self.events[room_id] = stop_event
            self.jobs[room_id] = {
                "state": "running", "episode": 0, "total": 0,
                "message": "initialising", "series": None, "summary": None,
                "error": None, "started_at": time.time(), "params": params or {},
            }
        thread = threading.Thread(target=self._run,
                                  args=(room_id, params, stop_event), daemon=True)
        thread.start()
        return True

    def stop(self, room_id: int) -> bool:
        with self.lock:
            ev = self.events.get(room_id)
            job = self.jobs.get(room_id)
            if ev and job and job["state"] == "running":
                ev.set()
                job["message"] = "stop requested"
                return True
        return False

    # ------------------------------------------------------------------
    def _run(self, room_id: int, params: dict | None, stop_event):
        last_push = [0.0]

        def progress(episode, total, series_fn, message=""):
            now = time.time()
            heavy = series_fn is not None and (now - last_push[0] > 0.35
                                               or episode == total)
            series = None
            if heavy:
                series = downsample(series_fn())
                last_push[0] = now
            with self.lock:
                job = self.jobs[room_id]
                job["episode"] = episode
                job["total"] = total
                job["message"] = message
                if series is not None:
                    job["series"] = series

        try:
            summary = run_training(room_id, params, progress=progress,
                                   stop=stop_event)
            with self.lock:
                job = self.jobs[room_id]
                stopped = stop_event.is_set() or summary.get("stopped")
                job["state"] = "stopped" if stopped else "finished"
                job["summary"] = summary
                job["message"] = ("training stopped by user"
                                  if stopped else "training complete")
        except Exception as exc:  # surfaced in the UI
            traceback.print_exc()
            with self.lock:
                job = self.jobs[room_id]
                job["state"] = "error"
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["message"] = "training failed"


MANAGER = TrainingManager()
