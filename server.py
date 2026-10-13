"""Catchy Catcher processing engine.

FastAPI wrapper around :func:`pipeline.process_song_session`, designed to run on
the GPU desktop and be driven by the laptop client over a local network or a
Tailscale VPN.

The client works in **sessions** (one per song/idea) made of **takes** (one audio
file each). Each take is uploaded and processed synchronously: the client shows
a spinner while it waits, then records the next take. Takes are stored under
``sessions/<session id>/takes/<take id>/`` so a session can be reopened and its
takes re-fetched later.

Endpoints
---------
``GET    /health``
    Liveness + which torch device would be used.
``POST   /sessions/{session_id}/takes/{take_id}``
    Upload one take (multipart field ``file``). Processed by default; pass
    ``process=false`` to store the audio only. Optional ``index``, ``name``,
    ``process`` and ``silence_threshold`` query parameters.
``POST   /sessions/{session_id}/takes/{take_id}/process``
    Process a take that was previously stored unprocessed.
``PATCH  /sessions/{session_id}/takes/{take_id}``
    Rename a journal entry and/or toggle its ``bookmarked`` flag.
``GET    /sessions``
    List stored sessions as ``{"id", "name", "updated_at"}`` objects, ordered by
    id.
``POST   /sessions/{session_id}``
    Create an empty session folder ahead of recording the first take. Optional
    ``name`` query parameter for the display name.
``PATCH  /sessions/{session_id}``
    Rename a session (``name`` query parameter); takes are untouched.
``DELETE /sessions/{session_id}``
    Delete a session and every take inside it.
``GET    /sessions/{session_id}``
    Session manifest: the name, then every take in order with its status.
``GET    /sessions/{session_id}/takes/{take_id}``
    One take's metadata and (when finished) its result.
``GET    /sessions/{session_id}/takes/{take_id}/audio/{stem}``
    Stream a generated stem, where ``stem`` is ``vocals`` or ``background``.
``GET    /sessions/{session_id}/takes/{take_id}/original``
    Stream the raw recording of a take, processed or not.
``DELETE /sessions/{session_id}/takes/{take_id}``
    Remove a take and its files.
``POST   /process-session``
    Single-shot convenience endpoint from the original spec. Equivalent to
    creating a session named after the file with one take called ``take``.

Scrapbooks
----------
``GET    /scrapbooks``
    List scrapbooks (``{scrapbook_id, name, bpm, line_count, ...}``).
``POST   /scrapbooks``
    Create one from ``{name, bpm, lines}`` where each line is
    ``{session_id, take_id, line_index}``.
``GET    /scrapbooks/{scrapbook_id}``
    One scrapbook, with its lines and last render.
``PATCH  /scrapbooks/{scrapbook_id}``
    Update ``name``/``bpm``/``lines`` (edits invalidate the render).
``DELETE /scrapbooks/{scrapbook_id}``
    Delete a scrapbook and its mix.
``POST   /scrapbooks/{scrapbook_id}/render``
    Render (or return the cached) tempo-matched mix and word timeline.
``GET    /scrapbooks/{scrapbook_id}/audio``
    Stream a scrapbook's rendered mix.

Configuration is via environment variables:

============================== ==================================================
``CATCHYCATCHER_HOST``         Bind host (default ``0.0.0.0``).
``CATCHYCATCHER_PORT``         Bind port (default ``8000``).
``CATCHYCATCHER_UPLOAD_DIR``   Session storage root (default ``./sessions``).
``CATCHYCATCHER_CORS_ORIGINS`` Comma-separated origins, or ``*`` (default).
``CATCHYCATCHER_LOG_LEVEL``    Uvicorn log level (default ``info``; ``run.py``
                               uses ``warning`` unless ``--verbose``).
``CATCHYCATCHER_REQUEST_LOG``  Level for the per-request log line (default
                               ``info``); ``off`` silences it.
============================== ==================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import pipeline
import scrapbook

BASE_DIR = Path(__file__).resolve().parent

SESSIONS_DIR = Path(
    os.environ.get("CATCHYCATCHER_UPLOAD_DIR", BASE_DIR / "sessions")
).resolve()
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

SCRAPBOOKS_DIR = Path(
    os.environ.get("CATCHYCATCHER_SCRAPBOOK_DIR", BASE_DIR / "scrapbooks")
).resolve()
SCRAPBOOKS_DIR.mkdir(parents=True, exist_ok=True)

_raw_origins = os.environ.get("CATCHYCATCHER_CORS_ORIGINS", "*").strip()
if _raw_origins in ("", "*"):
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------
#
# Every request is written to ``catchycatcher.requests`` as it completes, so it
# is obvious when the laptop frontend (or the share tunnel) actually reaches the
# API, and how long the server took to answer. On by default; set
# ``CATCHYCATCHER_REQUEST_LOG=off`` to silence it. The logger gets its own
# handler rather than relying on uvicorn's logging config, so it works whichever
# way the server is started (``python server.py``, ``python run.py``, uvicorn).

_LOG_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

_REQUEST_LOG_LEVEL = os.environ.get("CATCHYCATCHER_REQUEST_LOG", "info").strip().lower()
request_logger = logging.getLogger("catchycatcher.requests")


def _configure_request_logging() -> None:
    level = _LOG_LEVELS.get(_REQUEST_LOG_LEVEL)
    if level is None:
        # Anything else (notably "off"/"none") disables request logging.
        request_logger.disabled = True
        return
    request_logger.setLevel(level)
    request_logger.propagate = False
    if not request_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
            )
        )
        request_logger.addHandler(handler)


_configure_request_logging()

# Demucs + WhisperX both want the whole GPU; serialize takes so two uploads
# cannot fight over VRAM and OOM each other.
PROCESS_LOCK = threading.Lock()

#: Stems a client is allowed to stream.
AUDIO_STEMS = ("vocals", "background")

app = FastAPI(
    title="Catchy Catcher Processing Engine",
    description="Isolate vocals, align words, group lyric lines and estimate BPM.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Credentials cannot be combined with a wildcard origin, and this API is
    # token-less, so keep it off.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id", "X-Take-Id"],
)


@app.middleware("http")
async def log_request(request: Request, call_next):
    """Log a one-line summary of every request once it has been handled."""
    started = time.perf_counter()
    client = request.client.host if request.client else "-"
    target = request.url.path
    if request.url.query:
        target = f"{target}?{request.url.query}"

    try:
        response = await call_next(request)
    except Exception:
        request_logger.exception(
            "%s %s from %s -> unhandled error (%.1f ms)",
            request.method,
            target,
            client,
            (time.perf_counter() - started) * 1000,
        )
        raise

    request_logger.info(
        "%s %s from %s -> %d (%.1f ms)",
        request.method,
        target,
        client,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def sanitize_id(value: str) -> str:
    """Turn arbitrary user input into a safe single path segment.

    Keeps letters, digits, ``.``, ``_`` and ``-``; everything else (including
    path separators and ``..``) collapses to ``_``. Falls back to ``session``.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or "session"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / sanitize_id(session_id)


def _take_dir(session_id: str, take_id: str) -> Path:
    return _session_dir(session_id) / "takes" / sanitize_id(take_id)


def _next_index(session_id: str) -> int:
    """One past the highest index already present in the session."""
    takes_dir = _session_dir(session_id) / "takes"
    highest = 0
    if takes_dir.is_dir():
        for take in takes_dir.iterdir():
            meta = _read_json(take / "meta.json") if take.is_dir() else None
            if isinstance(meta, dict):
                highest = max(highest, int(meta.get("index") or 0))
    return highest + 1


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------
#
# A session's id is its folder name (URL-safe, stable, never shown as the
# primary label) while its display name is free-form and lives in
# ``session.json``. Older sessions predate that file, so every reader falls
# back to the id.


def _session_meta_path(session_id: str) -> Path:
    return _session_dir(session_id) / "session.json"


def _clean_name(name: str | None, fallback: str) -> str:
    """Collapse whitespace and cap length, falling back when empty."""
    cleaned = " ".join((name or "").split())
    return cleaned[:120] or fallback


def session_name(session_id: str) -> str:
    """Display name for a session, or its sanitised id when none is stored."""
    sid = sanitize_id(session_id)
    meta = _read_json(_session_meta_path(sid))
    if isinstance(meta, dict):
        name = meta.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return sid


def _write_session_meta(
    session_id: str,
    *,
    name: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict[str, str]:
    """Merge-update ``session.json`` and return the record that was written.

    Any field left as ``None`` keeps its previous value (or a sensible default),
    so callers only have to pass what changed.
    """
    sid = sanitize_id(session_id)
    existing = _read_json(_session_meta_path(sid))
    if not isinstance(existing, dict):
        existing = {}

    prior_name = existing.get("name")
    prior_name = prior_name if isinstance(prior_name, str) and prior_name.strip() else sid

    now = _now()
    meta = {
        "session_id": sid,
        "name": _clean_name(name, prior_name),
        "created_at": created_at or existing.get("created_at") or now,
        "updated_at": updated_at or now,
    }
    _write_json(_session_meta_path(sid), meta)
    return meta


def _touch_session(session_id: str) -> None:
    """Bump a session's ``updated_at`` (its takes changed)."""
    if _session_dir(session_id).is_dir():
        _write_session_meta(session_id)


def session_updated_at(session_id: str) -> str:
    """Last-modified timestamp for a session, best-effort for old sessions."""
    sid = sanitize_id(session_id)
    meta = _read_json(_session_meta_path(sid))
    if isinstance(meta, dict):
        for key in ("updated_at", "created_at"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return value
    # Legacy session without metadata: fall back to the folder's own mtime.
    try:
        mtime = _session_dir(sid).stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Take processing
# ---------------------------------------------------------------------------


def _take_input_path(take_dir: Path, meta: dict[str, Any] | None = None) -> Path:
    """The stored recording for a take, whether or not it has been processed."""
    if isinstance(meta, dict):
        name = meta.get("input_filename")
        if isinstance(name, str) and name:
            candidate = take_dir / name
            if candidate.is_file():
                return candidate
    for candidate in sorted(take_dir.glob("input.*")):
        if candidate.is_file():
            return candidate
    raise HTTPException(status_code=404, detail="No stored audio for this take.")


def _save_take(
    session_id: str,
    take_id: str,
    file: UploadFile,
    index: int | None,
    name: str | None = None,
) -> tuple[str, str, Path, dict[str, Any], Path]:
    """Store the uploaded audio and its (unprocessed) metadata."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename on the uploaded file.")

    sid = sanitize_id(session_id)
    tid = sanitize_id(take_id)
    take_dir = _take_dir(sid, tid)
    take_dir.mkdir(parents=True, exist_ok=True)

    # Keep the original extension (defaulting to .m4a) for ffmpeg's benefit.
    suffix = Path(file.filename).suffix.lower() or ".m4a"
    input_name = f"input{suffix}"
    input_path = take_dir / input_name
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    meta_path = take_dir / "meta.json"
    previous = _read_json(meta_path) or {}
    if index is None:
        index = previous.get("index")
        if index is None:
            index = _next_index(sid)

    prior_name = previous.get("name")
    prior_name = prior_name if isinstance(prior_name, str) and prior_name.strip() else tid

    meta: dict[str, Any] = {
        "session_id": sid,
        "take_id": tid,
        "index": int(index),
        "name": _clean_name(name, prior_name),
        "filename": file.filename,
        "input_filename": input_name,
        "bookmarked": bool(previous.get("bookmarked", False)),
        "processed": False,
        "status": "recorded",
        "error": None,
        "created_at": previous.get("created_at") or _now(),
    }
    _write_json(meta_path, meta)
    _touch_session(sid)
    return sid, tid, take_dir, meta, meta_path


def _run_pipeline(
    sid: str,
    tid: str,
    take_dir: Path,
    meta: dict[str, Any],
    meta_path: Path,
    silence_threshold: float,
) -> JSONResponse:
    """Process an already-stored take and return the spec payload."""
    input_path = _take_input_path(take_dir, meta)
    meta["input_filename"] = input_path.name
    meta["status"] = "processing"
    _write_json(meta_path, meta)

    try:
        try:
            with PROCESS_LOCK:
                result = pipeline.process_song_session(
                    str(input_path),
                    output_dir=str(take_dir),
                    silence_threshold=silence_threshold,
                )
        except RuntimeError as exc:
            # Missing dependencies / missing input are server-side problems.
            _record_failure(meta_path, meta, str(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - surface the failure to the client
            _record_failure(meta_path, meta, str(exc))
            raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

        _write_json(take_dir / "result.json", result)
        meta["status"] = "done"
        meta["processed"] = True
        meta["finished_at"] = _now()
        _write_json(meta_path, meta)

        response = JSONResponse(result)
        response.headers["X-Session-Id"] = sid
        response.headers["X-Take-Id"] = tid
        return response
    finally:
        # Processing a take modifies the session; record when.
        _touch_session(sid)


def _take_json_response(payload: Any, meta: dict[str, Any]) -> JSONResponse:
    response = JSONResponse(payload)
    response.headers["X-Session-Id"] = str(meta.get("session_id", ""))
    response.headers["X-Take-Id"] = str(meta.get("take_id", ""))
    return response


def _process_take(
    session_id: str,
    take_id: str,
    file: UploadFile,
    index: int | None,
    silence_threshold: float,
    name: str | None = None,
) -> JSONResponse:
    """Store, process and record a single take, then return its result."""
    sid, tid, take_dir, meta, meta_path = _save_take(session_id, take_id, file, index, name)
    return _run_pipeline(sid, tid, take_dir, meta, meta_path, silence_threshold)


def _store_take(
    session_id: str,
    take_id: str,
    file: UploadFile,
    index: int | None,
    name: str | None = None,
) -> JSONResponse:
    """Store a take without processing it, returning its (raw) metadata."""
    _sid, _tid, _take_dir, meta, _meta_path = _save_take(
        session_id, take_id, file, index, name
    )
    return _take_json_response(meta, meta)



def _record_failure(meta_path: Path, meta: dict[str, Any], error: str) -> None:
    meta["status"] = "error"
    meta["error"] = error
    meta["finished_at"] = _now()
    _write_json(meta_path, meta)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict[str, Any]:
    """Small discovery document so a browser hitting the base url sees something."""
    return {
        "name": "Catchy Catcher Processing Engine",
        "endpoints": [
            "POST /sessions/{session_id}/takes/{take_id}",
            "POST /sessions/{session_id}/takes/{take_id}/process",
            "GET /sessions",
            "POST /sessions/{session_id}",
            "PATCH /sessions/{session_id}",
            "DELETE /sessions/{session_id}",
            "GET /sessions/{session_id}",
            "GET /sessions/{session_id}/takes/{take_id}",
            "PATCH /sessions/{session_id}/takes/{take_id}",
            "GET /sessions/{session_id}/takes/{take_id}/audio/{stem}",
            "GET /sessions/{session_id}/takes/{take_id}/original",
            "DELETE /sessions/{session_id}/takes/{take_id}",
            "POST /process-session",
            "GET /scrapbooks",
            "POST /scrapbooks",
            "GET /scrapbooks/{scrapbook_id}",
            "PATCH /scrapbooks/{scrapbook_id}",
            "DELETE /scrapbooks/{scrapbook_id}",
            "POST /scrapbooks/{scrapbook_id}/render",
            "GET /scrapbooks/{scrapbook_id}/audio",
        ],
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Reports the torch device that would be used."""
    return {"status": "ok", "device": pipeline.detect_device()}


@app.post("/sessions/{session_id}/takes/{take_id}")
def create_take(
    session_id: str,
    take_id: str,
    file: UploadFile = File(..., description="The recorded take."),
    index: int | None = Query(
        default=None,
        ge=1,
        description="1-based position of the take within the session. "
        "Defaults to one past the highest existing take.",
    ),
    name: str | None = Query(
        default=None,
        description="Display name for the entry. Defaults to the take id.",
    ),
    process: bool = Query(
        default=True,
        description="Run the pipeline now; ``false`` just stores the audio so it "
        "can be processed later via ``POST .../process``.",
    ),
    silence_threshold: float = Query(
        default=pipeline.SILENCE_THRESHOLD,
        gt=0,
        le=5,
        description="Seconds of pause that always starts a new lyric line. Other signals (relative pauses, punctuation, nearby line length) are combined into the same line-break score.",
    ),
) -> JSONResponse:
    """Store one take, processing it now unless ``process=false``."""
    if process:
        return _process_take(session_id, take_id, file, index, silence_threshold, name)
    return _store_take(session_id, take_id, file, index, name)


@app.patch("/sessions/{session_id}/takes/{take_id}")
def update_take(
    session_id: str,
    take_id: str,
    name: str | None = Query(default=None, min_length=1, description="New entry name."),
    bookmarked: bool | None = Query(default=None, description="Mark/unmark the entry."),
) -> dict[str, Any]:
    """Rename a journal entry and/or toggle its bookmark; audio is untouched."""
    sid = sanitize_id(session_id)
    tid = sanitize_id(take_id)
    take_dir = _take_dir(sid, tid)
    meta = _read_json(take_dir / "meta.json")
    if meta is None:
        raise HTTPException(status_code=404, detail="Unknown take.")
    if name is not None:
        meta["name"] = _clean_name(name, tid)
    if bookmarked is not None:
        meta["bookmarked"] = bool(bookmarked)
    _write_json(take_dir / "meta.json", meta)
    _touch_session(sid)
    return {
        "take_id": tid,
        "name": meta.get("name"),
        "bookmarked": bool(meta.get("bookmarked", False)),
    }


@app.post("/sessions/{session_id}/takes/{take_id}/process")
def process_take(
    session_id: str,
    take_id: str,
    silence_threshold: float = Query(
        default=pipeline.SILENCE_THRESHOLD,
        gt=0,
        le=5,
        description="Seconds of pause that always starts a new lyric line. Other signals (relative pauses, punctuation, nearby line length) are combined into the same line-break score.",
    ),
) -> JSONResponse:
    """Process a previously stored (unprocessed) take and return its result."""
    sid = sanitize_id(session_id)
    tid = sanitize_id(take_id)
    take_dir = _take_dir(sid, tid)
    meta = _read_json(take_dir / "meta.json")
    if meta is None:
        raise HTTPException(status_code=404, detail="Unknown take.")
    return _run_pipeline(sid, tid, take_dir, meta, take_dir / "meta.json", silence_threshold)


@app.get("/sessions")
def list_sessions() -> dict[str, Any]:
    """List stored sessions as ``{id, name, updated_at}``, ordered by id."""
    sessions: list[dict[str, str]] = []
    if SESSIONS_DIR.is_dir():
        for path in SESSIONS_DIR.iterdir():
            if path.is_dir():
                sessions.append(
                    {
                        "id": path.name,
                        "name": session_name(path.name),
                        "updated_at": session_updated_at(path.name),
                    }
                )
    sessions.sort(key=lambda item: item["id"])
    return {"sessions": sessions}


@app.post("/sessions/{session_id}")
def create_session(
    session_id: str,
    name: str | None = Query(
        default=None,
        description="Display name for the session. Defaults to its id.",
    ),
) -> dict[str, Any]:
    """Create an empty session folder so the client can record takes into it.

    Idempotent: posting an id that already exists leaves its takes untouched and
    reports ``created: false``. The display ``name`` is stored in
    ``session.json``; when omitted an existing name is kept (or the id is used).
    """
    sid = sanitize_id(session_id)
    session_dir = _session_dir(sid)
    created = not session_dir.is_dir()
    (session_dir / "takes").mkdir(parents=True, exist_ok=True)

    meta = _write_session_meta(sid, name=name)
    return {
        "session_id": sid,
        "name": meta["name"],
        "created": created,
        "created_at": meta["created_at"],
        "updated_at": meta["updated_at"],
    }


@app.patch("/sessions/{session_id}")
def rename_session(
    session_id: str,
    name: str = Query(..., min_length=1, description="New display name."),
) -> dict[str, Any]:
    """Rename a session's display name; its takes and files are untouched."""
    sid = sanitize_id(session_id)
    if not _session_dir(sid).is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown session {sid!r}.")

    meta = _write_session_meta(sid, name=name)
    return {"session_id": sid, "name": meta["name"], "updated_at": meta["updated_at"]}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    """Delete a session and every take inside it."""
    sid = sanitize_id(session_id)
    session_dir = _session_dir(sid)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown session {sid!r}.")
    shutil.rmtree(session_dir)
    return {"deleted": True, "session_id": sid}


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """Return a session's name and every take in it, ordered by index."""
    sid = sanitize_id(session_id)
    session_dir = _session_dir(sid)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown session {sid!r}.")

    takes: list[dict[str, Any]] = []
    takes_dir = session_dir / "takes"
    if takes_dir.is_dir():
        for take in takes_dir.iterdir():
            if not take.is_dir():
                continue
            meta = _read_json(take / "meta.json")
            if isinstance(meta, dict):
                takes.append(meta)

    takes.sort(key=lambda m: (m.get("index") or 0, m.get("created_at") or ""))
    return {
        "session_id": sid,
        "name": session_name(sid),
        "updated_at": session_updated_at(sid),
        "take_count": len(takes),
        "takes": takes,
    }


@app.get("/sessions/{session_id}/takes/{take_id}")
def get_take(session_id: str, take_id: str) -> dict[str, Any]:
    """Return one take's metadata plus its result, when available."""
    take_dir = _take_dir(session_id, take_id)
    meta = _read_json(take_dir / "meta.json")
    if meta is None:
        raise HTTPException(status_code=404, detail="Unknown take.")
    return {"meta": meta, "result": _read_json(take_dir / "result.json")}


@app.get("/sessions/{session_id}/takes/{take_id}/audio/{stem}")
def get_take_audio(session_id: str, take_id: str, stem: str) -> FileResponse:
    """Stream a generated stem (``vocals`` or ``background``)."""
    if stem not in AUDIO_STEMS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown stem {stem!r}; expected one of {AUDIO_STEMS}.",
        )
    path = _take_dir(session_id, take_id) / f"{stem}.m4a"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found for this take.")
    return FileResponse(path, media_type="audio/mp4", filename=f"{stem}.m4a")


@app.get("/sessions/{session_id}/takes/{take_id}/original")
def get_take_original(session_id: str, take_id: str) -> FileResponse:
    """Stream the raw recording of a take, processed or not."""
    take_dir = _take_dir(session_id, take_id)
    meta = _read_json(take_dir / "meta.json")
    if meta is None:
        raise HTTPException(status_code=404, detail="Unknown take.")
    path = _take_input_path(take_dir, meta)
    return FileResponse(path, filename=path.name)


@app.delete("/sessions/{session_id}/takes/{take_id}")
def delete_take(session_id: str, take_id: str) -> dict[str, Any]:
    """Delete a take and all of its files."""
    take_dir = _take_dir(session_id, take_id)
    if not take_dir.is_dir():
        raise HTTPException(status_code=404, detail="Unknown take.")
    shutil.rmtree(take_dir)
    _touch_session(session_id)
    return {"deleted": True, "session_id": sanitize_id(session_id), "take_id": sanitize_id(take_id)}


@app.post("/process-session")
def process_session_endpoint(
    file: UploadFile = File(..., description="The recorded .m4a session."),
    silence_threshold: float = Query(
        default=pipeline.SILENCE_THRESHOLD,
        gt=0,
        le=5,
        description="Seconds of pause that always starts a new lyric line. Other signals (relative pauses, punctuation, nearby line length) are combined into the same line-break score.",
    ),
) -> JSONResponse:
    """Single-shot endpoint from the original spec.

    Stores the upload as ``<filename stem>/take`` so it behaves exactly like a
    one-take session.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename on the uploaded file.")
    return _process_take(
        Path(file.filename).stem,
        "take",
        file,
        None,
        silence_threshold,
    )


# ---------------------------------------------------------------------------
# Scrapbooks
# ---------------------------------------------------------------------------
#
# A scrapbook is an ordered list of lines referenced from processed takes, plus
# one BPM every line is stretched to. It is the *only* place lines can be added,
# removed or reordered. Rendering stitches the isolated vocals into one mix; the
# result is cached against a signature of the BPM + line list.


def _scrapbook_dir(scrapbook_id: str) -> Path:
    return SCRAPBOOKS_DIR / sanitize_id(scrapbook_id)


def _scrapbook_meta_path(scrapbook_id: str) -> Path:
    return _scrapbook_dir(scrapbook_id) / "scrapbook.json"


def _normalize_lines(lines: Any) -> list[dict[str, Any]]:
    """Keep only well-formed ``{session_id, take_id, line_index}`` references."""
    if not isinstance(lines, list):
        return []
    normalized: list[dict[str, Any]] = []
    for ref in lines:
        if not isinstance(ref, dict):
            continue
        session_id = str(ref.get("session_id", "")).strip()
        take_id = str(ref.get("take_id", "")).strip()
        try:
            line_index = int(ref.get("line_index", -1))
        except (TypeError, ValueError):
            continue
        if not session_id or not take_id or line_index < 0:
            continue
        ref = {
            "session_id": sanitize_id(session_id),
            "take_id": sanitize_id(take_id),
            "line_index": line_index,
        }
        # Only keep lines that actually resolve to a processed take.
        try:
            _line_data(ref)
        except HTTPException:
            continue
        normalized.append(ref)
    return normalized


def _clean_bpm(bpm: Any, fallback: float = 120.0) -> float:
    try:
        value = float(bpm)
    except (TypeError, ValueError):
        return fallback
    if value <= 20 or value >= 400:
        return fallback
    return round(value, 1)


def _median_line_bpm(lines: list[dict[str, Any]], fallback: float = 120.0) -> float:
    """Median BPM of the lines' source takes, or ``fallback`` when unknown."""
    bpms: list[float] = []
    for ref in lines:
        try:
            bpms.append(float(_line_data(ref)["bpm"]))
        except HTTPException:
            continue
    usable = sorted(value for value in bpms if value > 20)
    if not usable:
        return fallback
    middle = len(usable) // 2
    if len(usable) % 2:
        return round(usable[middle], 1)
    return round((usable[middle - 1] + usable[middle]) / 2, 1)


def _scrapbook_signature(bpm: float, lines: list[dict[str, Any]]) -> str:
    payload = json.dumps({"bpm": bpm, "lines": lines}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_scrapbook(scrapbook_id: str) -> dict[str, Any]:
    meta = _read_json(_scrapbook_meta_path(scrapbook_id))
    if not isinstance(meta, dict):
        raise HTTPException(status_code=404, detail=f"Unknown scrapbook {scrapbook_id!r}.")
    return meta


def _scrapbook_summary(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "scrapbook_id": meta.get("scrapbook_id"),
        "name": meta.get("name"),
        "bpm": meta.get("bpm"),
        "bpm_manual": bool(meta.get("bpm_manual", False)),
        "line_count": len(meta.get("lines") or []),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "rendered": bool(meta.get("render")),
    }


def _line_data(ref: dict[str, Any]) -> dict[str, Any]:
    """Resolve one scrapbook line reference into what the renderer needs."""
    sid = sanitize_id(str(ref.get("session_id", "")))
    tid = sanitize_id(str(ref.get("take_id", "")))
    index = int(ref.get("line_index", 0))
    take_dir = _take_dir(sid, tid)
    result = _read_json(take_dir / "result.json")
    lyrics = result.get("lyrics") if isinstance(result, dict) else None
    if not isinstance(lyrics, list) or not 0 <= index < len(lyrics):
        raise HTTPException(
            status_code=400, detail=f"Line {sid}/{tid}/{index} is no longer available."
        )
    line = lyrics[index]
    vocals = take_dir / "vocals.m4a"
    if not vocals.is_file():
        raise HTTPException(status_code=400, detail=f"No isolated vocals for {sid}/{tid}.")

    previous = lyrics[index - 1] if index > 0 else None
    following = lyrics[index + 1] if index + 1 < len(lyrics) else None
    return {
        "bpm": float(line.get("bpm") or 120.0),
        "start": float(line["start time"]),
        "end": float(line["end time"]),
        "words": line.get("words") or [],
        "gap_before": (
            float(line["start time"]) - float(previous["end time"]) if previous else None
        ),
        "gap_after": (
            float(following["start time"]) - float(line["end time"]) if following else None
        ),
        "vocals": vocals,
    }


def _render_scrapbook(meta: dict[str, Any]) -> dict[str, Any]:
    sid = str(meta["scrapbook_id"])
    bpm = _clean_bpm(meta.get("bpm"))
    lines = meta.get("lines") or []
    signature = _scrapbook_signature(bpm, lines)
    mix = _scrapbook_dir(sid) / "mix.m4a"

    cached = meta.get("render")
    if isinstance(cached, dict) and cached.get("signature") == signature and mix.is_file():
        return cached

    if not lines:
        raise HTTPException(status_code=400, detail="Add at least one line before rendering.")

    entries = [_line_data(ref) for ref in lines]
    with PROCESS_LOCK:
        timeline, gaps, _mix = scrapbook.render(entries, bpm, _scrapbook_dir(sid))

    render = {
        "signature": signature,
        "audio": f"/scrapbooks/{sid}/audio",
        "bpm": bpm,
        "timeline": timeline,
        "gaps": gaps,
        "rendered_at": _now(),
    }
    meta["render"] = render
    meta["updated_at"] = _now()
    _write_json(_scrapbook_meta_path(sid), meta)
    return render


@app.get("/scrapbooks")
def list_scrapbooks() -> dict[str, Any]:
    """List scrapbooks, most recently updated first."""
    items = []
    if SCRAPBOOKS_DIR.is_dir():
        for path in SCRAPBOOKS_DIR.iterdir():
            if not path.is_dir():
                continue
            meta = _read_json(path / "scrapbook.json")
            if isinstance(meta, dict):
                items.append(_scrapbook_summary(meta))
    items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return {"scrapbooks": items}


@app.post("/scrapbooks")
def create_scrapbook(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Create a scrapbook from ``{name, lines}``.

    The BPM starts as the median of the lines' source takes (120 when empty) and
    keeps tracking it until the client sets ``bpm`` explicitly.
    """
    name = _clean_name(body.get("name"), "Scrapbook")
    lines = _normalize_lines(body.get("lines"))
    bpm = _median_line_bpm(lines)
    scrapbook_id = f"sb-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
    now = _now()
    meta = {
        "scrapbook_id": scrapbook_id,
        "name": name,
        "bpm": bpm,
        "bpm_manual": False,
        "lines": lines,
        "created_at": now,
        "updated_at": now,
    }
    _scrapbook_dir(scrapbook_id).mkdir(parents=True, exist_ok=True)
    _write_json(_scrapbook_meta_path(scrapbook_id), meta)
    return _scrapbook_summary(meta)


@app.get("/scrapbooks/{scrapbook_id}")
def get_scrapbook(scrapbook_id: str) -> dict[str, Any]:
    """Return a scrapbook with its lines and last render, if any."""
    meta = _load_scrapbook(scrapbook_id)
    result = _scrapbook_summary(meta)
    result["lines"] = meta.get("lines") or []
    result["render"] = meta.get("render")
    return result


@app.patch("/scrapbooks/{scrapbook_id}")
def update_scrapbook(
    scrapbook_id: str, body: dict[str, Any] = Body(default={})
) -> dict[str, Any]:
    """Update ``name``, ``bpm`` and/or ``lines``. Edits invalidate the render.

    Setting ``bpm`` marks it manual (it then stops tracking the lines); pass
    ``bpm_manual: false`` to hand it back to the automatic median.
    """
    meta = _load_scrapbook(scrapbook_id)
    manual = bool(meta.get("bpm_manual", False))
    if "bpm_manual" in body:
        manual = bool(body.get("bpm_manual"))

    if "name" in body:
        meta["name"] = _clean_name(body.get("name"), meta.get("name") or "Scrapbook")
    if "lines" in body:
        meta["lines"] = _normalize_lines(body.get("lines"))
    if "bpm" in body:
        meta["bpm"] = _clean_bpm(body.get("bpm"), meta.get("bpm") or 120.0)
        manual = True

    meta["bpm_manual"] = manual
    if not manual:
        meta["bpm"] = _median_line_bpm(meta.get("lines") or [], meta.get("bpm") or 120.0)

    meta["updated_at"] = _now()
    _write_json(_scrapbook_meta_path(scrapbook_id), meta)
    result = _scrapbook_summary(meta)
    result["lines"] = meta.get("lines") or []
    result["render"] = meta.get("render")
    return result


@app.delete("/scrapbooks/{scrapbook_id}")
def delete_scrapbook(scrapbook_id: str) -> dict[str, Any]:
    """Delete a scrapbook and its rendered mix."""
    sid = sanitize_id(scrapbook_id)
    path = _scrapbook_dir(sid)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown scrapbook {sid!r}.")
    shutil.rmtree(path)
    return {"deleted": True, "scrapbook_id": sid}


@app.post("/scrapbooks/{scrapbook_id}/render")
def render_scrapbook(scrapbook_id: str) -> dict[str, Any]:
    """Render (or return the cached) tempo-matched mix for a scrapbook."""
    meta = _load_scrapbook(scrapbook_id)
    return _render_scrapbook(meta)


@app.get("/scrapbooks/{scrapbook_id}/audio")
def get_scrapbook_audio(scrapbook_id: str) -> FileResponse:
    """Stream a scrapbook's rendered mix."""
    meta = _load_scrapbook(scrapbook_id)
    sid = str(meta["scrapbook_id"])
    mix = _scrapbook_dir(sid) / "mix.m4a"
    if not mix.is_file():
        raise HTTPException(status_code=404, detail="Scrapbook has not been rendered.")
    return FileResponse(mix, media_type="audio/mp4", filename=f"{sid}.m4a")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.environ.get("CATCHYCATCHER_HOST", "0.0.0.0"),
        port=int(os.environ.get("CATCHYCATCHER_PORT", "8000")),
        # run.py drops this to "warning" so a normal start is quiet; set it back
        # to "info" (or pass --verbose) to see requests and startup diagnostics.
        log_level=os.environ.get("CATCHYCATCHER_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
