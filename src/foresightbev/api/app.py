"""Stage 8 backend. ``/api/v1`` is the typed API; ``/api`` stays for the frozen dashboard.

Every response comes from files written by the pipeline or from a fresh computation
on them. This service is observability only and is not in any control path.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from foresightbev import __version__
from foresightbev.api.jobs import JobManager
from foresightbev.api.logs import configure_logging
from foresightbev.api.store import RunStore
from foresightbev.api.showcase import router as showcase_router
from foresightbev.api.v1 import router
from foresightbev.config import Settings, load_settings
from foresightbev.errors import ForesightError

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("FORESIGHTBEV_DATA", ROOT / "experiments/end_to_end"))
FRONTEND = ROOT / "frontend/dist"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
LAYERS = {"bev": "bev", "cameras": "cameras", "forecast": "forecast"}
logger = logging.getLogger(__name__)

legacy = APIRouter()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logger.info("backend started, data directory %s", settings.data_dir)
        yield
        logger.info("backend stopping")

    application = FastAPI(title="Tavyra", version=__version__, description="Backend for stored and freshly computed pipeline outputs. Observability only, never in a control path.", lifespan=lifespan)
    application.state.settings = settings
    application.state.log_path = configure_logging(settings.data_dir / "logs", settings.log_level)
    application.state.store = RunStore(settings.data_dir)
    application.state.jobs = JobManager(settings.data_dir / "jobs")
    application.state.started = time.time()
    application.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["GET"])

    @application.middleware("http")
    async def trace(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        if request.url.path.startswith("/api"):
            logger.info("request", extra=dict(request_id=request.state.request_id, method=request.method, path=request.url.path, status=response.status_code,
                                              duration_ms=round(1000 * (time.perf_counter() - start), 1)))
        return response

    def body(request: Request, code: str, message: str, status: int) -> JSONResponse:
        return JSONResponse(status_code=status, content=dict(error=dict(code=code, message=message, request_id=getattr(request.state, "request_id", ""))),
                            headers={"X-Request-ID": getattr(request.state, "request_id", "")})

    @application.exception_handler(ForesightError)
    async def known_error(request: Request, error: ForesightError):
        logger.warning("%s: %s", error.code, error, extra=dict(request_id=getattr(request.state, "request_id", "")))
        return body(request, error.code, str(error), error.http_status)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        details = "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in error.errors())
        return body(request, "invalid_request", details, 422)

    @application.exception_handler(Exception)
    async def unexpected(request: Request, error: Exception):
        logger.exception("unhandled error", exc_info=error, extra=dict(request_id=getattr(request.state, "request_id", "")))
        return body(request, "internal_error", "Unexpected server error. See GET /api/v1/logs with this request id.", 500)

    application.include_router(showcase_router)
    application.include_router(router)
    application.include_router(legacy)
    if FRONTEND.is_dir():
        application.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
    return application


def _safe(name: str) -> str:
    if not _SAFE_NAME.match(name) or name in (".", ".."):
        raise HTTPException(404, "not found")
    return name


def _read_json(path: Path):
    if not path.is_file():
        raise HTTPException(404, f"{path.name} not found")
    return json.loads(path.read_text())


def _run_dir(run_id: str) -> Path:
    return DATA / "runs" / _safe(run_id)


@legacy.get("/api/health")
def health() -> dict:
    runs = DATA / "runs"
    return {"status": "ok", "data_directory_exists": DATA.is_dir(), "runs": sum(1 for p in runs.glob("*/manifest.json")) if runs.is_dir() else 0}


@legacy.get("/api/runs")
def list_runs() -> list[dict]:
    out = []
    for manifest in sorted((DATA / "runs").glob("*/manifest.json")):
        doc = json.loads(manifest.read_text())
        out.append({key: doc.get(key) for key in ("title", "kind", "scenario", "seed", "mode", "frames", "input", "caveat")}
                   | {"id": manifest.parent.name, "summary": doc.get("summary")})
    return out


@legacy.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _read_json(_run_dir(run_id) / "manifest.json")


@legacy.get("/api/runs/{run_id}/frames/{index}")
def get_frame(run_id: str, index: int) -> dict:
    return _read_json(_run_dir(run_id) / "frames" / f"{index:05d}.json")


@legacy.get("/api/runs/{run_id}/timeline")
def get_timeline(run_id: str) -> list[dict]:
    """One row per frame, for the decision strip and speed plot."""
    rows = []
    for path in sorted((_run_dir(run_id) / "frames").glob("*.json")):
        doc = json.loads(path.read_text())
        rows.append(dict(frame=doc["frame"], t=doc["t"], speed=doc["ego"]["speed"], status=doc["decision"]["status"],
                         selected=doc["decision"]["selected"], reasons=doc["decision"]["reasons"]))
    if not rows:
        raise HTTPException(404, "run has no frames")
    return rows


@legacy.get("/api/runs/{run_id}/images/{layer}/{index}")
def get_image(run_id: str, layer: str, index: int) -> FileResponse:
    if layer not in LAYERS:
        raise HTTPException(404, "unknown layer")
    path = _run_dir(run_id) / LAYERS[layer] / f"{index:05d}.png"
    if not path.is_file():
        raise HTTPException(404, "image not found")
    return FileResponse(path, media_type="image/png")


@legacy.get("/api/evaluations")
def list_evaluations() -> list[str]:
    return sorted(p.parent.name for p in (DATA / "results").glob("*/summary.json"))


@legacy.get("/api/evaluations/{name}")
def get_evaluation(name: str) -> dict:
    directory = DATA / "results" / _safe(name)
    out = {"summary": _read_json(directory / "summary.json"), "experiment": _read_json(directory / "experiment.json"),
           "environment": _read_json(directory / "environment.json")}
    for key in ("calibration", "real_sample", "learned_forecaster"):
        path = directory / f"{key}.json"
        out[key] = json.loads(path.read_text()) if path.is_file() else None
    return out


@legacy.get("/api/evaluations/{name}/report", response_class=PlainTextResponse)
def get_report(name: str) -> str:
    path = DATA / "results" / _safe(name) / "REPORT.md"
    if not path.is_file():
        raise HTTPException(404, "report not found")
    return path.read_text()


@legacy.get("/api/ros2")
def list_ros2() -> list[dict]:
    """Recorded ROS 2 closed-loop runs, without the per-frame decisions."""
    out = []
    for path in sorted((DATA / "results" / "ros2").glob("*.json")):
        doc = json.loads(path.read_text())
        out.append({k: v for k, v in doc.items() if k != "decisions"} | {"id": path.stem, "decisions": len(doc.get("decisions", []))})
    return out


app = create_app()
