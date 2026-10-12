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
    Upload one take (multipart field ``file``) and process it. Returns the schema
    from ``SYSTEM_INSTRUCTIONS.md``. Optional ``index`` and ``silence_threshold``
    query parameters.
``GET    /sessions``
    List stored sessions as ``{"id", "name"}`` objects, ordered by id.
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
``DELETE /sessions/{session_id}/takes/{take_id}``
    Remove a take and its files.
``POST   /process-session``
    Single-shot convenience endpoint from the original spec. Equivalent to
    creating a session named after the file with one take called ``take``.

Configuration is via environment variables:

============================== ==================================================
``CATCHYCATCHER_HOST``         Bind host (default ``0.0.0.0``).
``CATCHYCATCHER_PORT``         Bind port (default ``8000``).
``CATCHYCATCHER_UPLOAD_DIR``   Session storage root (default ``./sessions``).
``CATCHYCATCHER_CORS_ORIGINS`` Comma-separated origins, or ``*`` (default).
``CATCHYCATCHER_LOG_LEVEL``    Uvicorn log level (default ``info``; ``run.py``
                               uses ``warning`` unless ``--verbose``).
============================== ==================================================
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import pipeline

BASE_DIR = Path(__file__).resolve().parent

SESSIONS_DIR = Path(
    os.environ.get("CATCHYCATCHER_UPLOAD_DIR", BASE_DIR / "sessions")
).resolve()
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

_raw_origins = os.environ.get("CATCHYCATCHER_CORS_ORIGINS", "*").strip()
if _raw_origins in ("", "*"):
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

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


# ---------------------------------------------------------------------------
# Take processing
# ---------------------------------------------------------------------------


def _process_take(
    session_id: str,
    take_id: str,
    file: UploadFile,
    index: int | None,
    silence_threshold: float,
) -> JSONResponse:
    """Store, process and record a single take, then return its result."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename on the uploaded file.")

    sid = sanitize_id(session_id)
    tid = sanitize_id(take_id)
    take_dir = _take_dir(sid, tid)
    take_dir.mkdir(parents=True, exist_ok=True)

    # Keep the original extension (defaulting to .m4a) for ffmpeg's benefit.
    suffix = Path(file.filename).suffix.lower() or ".m4a"
    input_path = take_dir / f"input{suffix}"
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

    meta: dict[str, Any] = {
        "session_id": sid,
        "take_id": tid,
        "index": int(index),
        "filename": file.filename,
        "status": "processing",
        "error": None,
        "created_at": previous.get("created_at") or _now(),
    }
    _write_json(meta_path, meta)

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
    meta["finished_at"] = _now()
    _write_json(meta_path, meta)

    response = JSONResponse(result)
    response.headers["X-Session-Id"] = sid
    response.headers["X-Take-Id"] = tid
    return response


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
            "GET /sessions",
            "POST /sessions/{session_id}",
            "PATCH /sessions/{session_id}",
            "DELETE /sessions/{session_id}",
            "GET /sessions/{session_id}",
            "GET /sessions/{session_id}/takes/{take_id}",
            "GET /sessions/{session_id}/takes/{take_id}/audio/{stem}",
            "DELETE /sessions/{session_id}/takes/{take_id}",
            "POST /process-session",
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
    silence_threshold: float = Query(
        default=pipeline.SILENCE_THRESHOLD,
        gt=0,
        le=5,
        description="Seconds of pause that starts a new lyric line.",
    ),
) -> JSONResponse:
    """Process one take. Blocks until the stems and lyric timeline are ready."""
    return _process_take(session_id, take_id, file, index, silence_threshold)


@app.get("/sessions")
def list_sessions() -> dict[str, Any]:
    """List stored sessions as ``{id, name}`` objects, ordered by id."""
    sessions: list[dict[str, str]] = []
    if SESSIONS_DIR.is_dir():
        for path in SESSIONS_DIR.iterdir():
            if path.is_dir():
                sessions.append({"id": path.name, "name": session_name(path.name)})
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

    meta_path = _session_meta_path(sid)
    existing = _read_json(meta_path)
    if not isinstance(existing, dict):
        existing = {}
    prior = existing.get("name")
    prior = prior if isinstance(prior, str) and prior.strip() else sid

    display = _clean_name(name, prior)
    created_at = existing.get("created_at") or _now()
    _write_json(
        meta_path,
        {"session_id": sid, "name": display, "created_at": created_at},
    )
    return {
        "session_id": sid,
        "name": display,
        "created": created,
        "created_at": created_at,
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

    meta_path = _session_meta_path(sid)
    existing = _read_json(meta_path)
    if not isinstance(existing, dict):
        existing = {}

    display = _clean_name(name, session_name(sid))
    _write_json(
        meta_path,
        {
            "session_id": sid,
            "name": display,
            "created_at": existing.get("created_at") or _now(),
        },
    )
    return {"session_id": sid, "name": display}


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


@app.delete("/sessions/{session_id}/takes/{take_id}")
def delete_take(session_id: str, take_id: str) -> dict[str, Any]:
    """Delete a take and all of its files."""
    take_dir = _take_dir(session_id, take_id)
    if not take_dir.is_dir():
        raise HTTPException(status_code=404, detail="Unknown take.")
    shutil.rmtree(take_dir)
    return {"deleted": True, "session_id": sanitize_id(session_id), "take_id": sanitize_id(take_id)}


@app.post("/process-session")
def process_session_endpoint(
    file: UploadFile = File(..., description="The recorded .m4a session."),
    silence_threshold: float = Query(
        default=pipeline.SILENCE_THRESHOLD,
        gt=0,
        le=5,
        description="Seconds of pause that starts a new lyric line.",
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
