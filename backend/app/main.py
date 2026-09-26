"""
SnapSight AI backend entrypoint.

Dev mode (two processes, hot reload on both sides):
    uvicorn app.main:app --reload --port 8000     # from backend/, venv active
    npm run dev                                    # from frontend/, separate terminal

Production/demo mode (Phase 5 -- one process, one port):
    cd frontend && npm run build                  # produces frontend/dist/
    cd backend  && uvicorn app.main:app --port 8000
    -> open http://localhost:8000 -- this same FastAPI process now also
       serves the built UI, no separate frontend server needed.
Run `python scripts/check_deployment.py` first to confirm the machine
you're demoing on is actually ready (system binaries, providers, models).
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import APP_NAME, APP_VERSION, CORS_ORIGINS, FRONTEND_DIST_DIR
from app.api import hardware, benchmark, documents, camera, vision, voice, history, settings

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hardware.router)
app.include_router(benchmark.router)
app.include_router(documents.router)
app.include_router(camera.router)
app.include_router(vision.router)
app.include_router(voice.router)
app.include_router(history.router)
app.include_router(settings.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}


# --------------------------------------------------------------------------
# Phase 5: serve the built frontend, if it exists, from this same process.
# Registered LAST so every /api/... route above still wins the match --
# FastAPI matches routes in registration order, and this catch-all would
# otherwise swallow every path including real API ones.
# --------------------------------------------------------------------------
if (FRONTEND_DIST_DIR / "index.html").exists():
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        # An unknown /api/... path is a real 404, not a client-side route --
        # never mask a typo'd or removed endpoint behind index.html.
        if full_path.startswith("api/"):
            raise HTTPException(404, f"Unknown API route: /{full_path}")
        # Everything else is a React Router client-side route (e.g. /history,
        # /settings on a hard refresh) -- always hand back index.html and let
        # the SPA's own router resolve it.
        return FileResponse(FRONTEND_DIST_DIR / "index.html")
else:

    @app.get("/")
    def frontend_not_built():
        return {
            "status": "backend-only",
            "message": (
                "No frontend build found. Run `npm run build` in frontend/ to serve the UI from "
                "this same process (see scripts/check_deployment.py), or run `npm run dev` "
                "separately on http://localhost:5173 for local development."
            ),
        }
