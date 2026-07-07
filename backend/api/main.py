"""RL Escape Room — FastAPI backend entry point.

Run from the project root::

    python -m backend.api.main            # http://localhost:8000

Serves the JSON API under ``/api/*`` and, when ``frontend/dist`` exists
(after ``npm run build``), also serves the built web app — one command
runs the whole project.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes_metrics import router as metrics_router
from backend.api.routes_replay import router as replay_router
from backend.api.routes_rooms import router as rooms_router
from backend.api.routes_training import router as training_router
from backend.utils.config import FRONTEND_DIST, reset_results


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # safety net: clear leftovers if the previous run crashed / was killed
    # before its shutdown hook could run
    reset_results()
    yield
    # session over: training data only lives while the server is running
    reset_results()


app = FastAPI(
    title="RL Escape Room",
    description="Five themed escape rooms, five RL algorithms — Dynamic "
                "Programming, SARSA, Q-Learning and DQN behind an arcade web UI.",
    version="2.0.0",
    lifespan=lifespan,
)

# the dev frontend (vite, port 5173) talks to this server on port 8000
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

app.include_router(rooms_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")
app.include_router(replay_router, prefix="/api")


@app.get("/api/health")
def health():
    try:
        import torch  # noqa: F401
        torch_ok = True
    except Exception:
        torch_ok = False
    return {"ok": True, "torch": torch_ok}


# ---------------------------------------------------------------------------
# Serve the built frontend (single-page app) when available
# ---------------------------------------------------------------------------
if (FRONTEND_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"),
              name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        target = FRONTEND_DIST / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def no_frontend():
        return {
            "message": "API is running, but the web app is not built yet. "
                       "Either run `npm run build` inside frontend/ and "
                       "restart this server, or start the dev frontend "
                       "(`npm run dev` inside frontend/) and open "
                       "http://localhost:5173",
            "api_docs": "http://127.0.0.1:8000/docs",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api.main:app", host="127.0.0.1", port=8000,
                log_level="info")
