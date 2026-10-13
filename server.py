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
``PATCH  /sessions/{session_id}/takes/{take_id}/lines/{line_index}``
    Set (or clear) a line's display ``text`` override; timings are untouched.
``PUT    /sessions/{session_id}/takes/{take_id}/lyrics``
    Replace a take's lyric lines (re-splitting) and recompute their BPM.
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

Scribbles
---------
Scribbles are a journal shared across every session. They are stored as takes in
a reserved ``scribbles`` session (hidden from ``/sessions``), and these routes
mirror the take routes.

``GET    /scribbles``
    List scribbles (``{scribble_id, name, status, ...}``), newest first.
``POST   /scribbles/{scribble_id}``
    Upload a scribble; processed by default (``process=false`` just stores it).
``POST   /scribbles/text``
    Create a text-only scribble (no audio) from ``{name, lines}``.
``PUT    /scribbles/{scribble_id}/text``
    Replace a text scribble's lines.
``POST   /scribbles/{scribble_id}/process``
    Process a previously stored scribble.
``GET    /scribbles/{scribble_id}``
    One scribble's metadata plus its result, when finished.
``PATCH  /scribbles/{scribble_id}``
    Rename a scribble and/or toggle its bookmark.
``PATCH  /scribbles/{scribble_id}/lines/{line_index}``
    Set (or clear) one line's display ``text`` override.
``PUT    /scribbles/{scribble_id}/lyrics``
    Replace a scribble's lines (re-splitting) and recompute their BPM.
``GET    /scribbles/{scribble_id}/audio/{stem}``
    Stream a generated stem (``vocals`` or ``background``).
``GET    /scribbles/{scribble_id}/original``
    Stream the raw recording.
``DELETE /scribbles/{scribble_id}``
    Remove a scribble and its files.

Scrapbooks
----------
A scrapbook is scoped to one session (stored under that session's folder) and is
an ordered list of lines referenced from processed takes.

``GET    /sessions/{session_id}/scrapbooks``
    List a session's scrapbooks (``{scrapbook_id, name, line_count, ...}``).
``POST   /sessions/{session_id}/scrapbooks``
    Create one from ``{name, lines}`` where each line is
    ``{session_id, take_id, line_index}``.
``GET    /sessions/{session_id}/scrapbooks/{scrapbook_id}``
    One scrapbook, with its lines and last render.
``PATCH  /sessions/{session_id}/scrapbooks/{scrapbook_id}``
    Update ``name``/``lines`` (edits invalidate the render).
``DELETE /sessions/{session_id}/scrapbooks/{scrapbook_id}``
    Delete a scrapbook and its mix.
``POST   /sessions/{session_id}/scrapbooks/{scrapbook_id}/render``
    Render (or return the cached) tempo-matched mix and word timeline.
``GET    /sessions/{session_id}/scrapbooks/{scrapbook_id}/audio``
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

import difflib
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

# Scribbles are a journal shared across every session. They live in a reserved
# session folder so takes, stems, lyric edits and scrapbook line references all
# reuse the exact same storage and code paths. It is hidden from ``/sessions``.
SCRIBBLES_SESSION_ID = "scribbles"

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
            "PATCH /sessions/{session_id}/takes/{take_id}/lines/{line_index}",
            "PUT /sessions/{session_id}/takes/{take_id}/lyrics",
            "GET /sessions/{session_id}/takes/{take_id}/audio/{stem}",
            "GET /sessions/{session_id}/takes/{take_id}/original",
            "DELETE /sessions/{session_id}/takes/{take_id}",
            "POST /process-session",
            "GET /scribbles",
            "POST /scribbles/{scribble_id}",
            "POST /scribbles/text",
            "PUT /scribbles/{scribble_id}/text",
            "POST /scribbles/{scribble_id}/process",
            "GET /scribbles/{scribble_id}",
            "PATCH /scribbles/{scribble_id}",
            "PATCH /scribbles/{scribble_id}/lines/{line_index}",
            "PUT /scribbles/{scribble_id}/lyrics",
            "GET /scribbles/{scribble_id}/audio/{stem}",
            "GET /scribbles/{scribble_id}/original",
            "DELETE /scribbles/{scribble_id}",
            "GET /sessions/{session_id}/scrapbooks",
            "POST /sessions/{session_id}/scrapbooks",
            "GET /sessions/{session_id}/scrapbooks/{scrapbook_id}",
            "PATCH /sessions/{session_id}/scrapbooks/{scrapbook_id}",
            "DELETE /sessions/{session_id}/scrapbooks/{scrapbook_id}",
            "POST /sessions/{session_id}/scrapbooks/{scrapbook_id}/render",
            "GET /sessions/{session_id}/scrapbooks/{scrapbook_id}/audio",
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


_VOWELS = "aeiouy"


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9']", "", str(token).lower())


def _syllables(token: str) -> int:
    word = re.sub(r"[^a-z]", "", str(token).lower())
    if not word:
        return 1
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1 and not word.endswith(("le", "ee", "ye")):
        count -= 1
    return max(1, count)


def _retime_words(
    original_words: list[dict[str, Any]], new_text: str, line_start: float, line_end: float
) -> list[dict[str, Any]]:
    """Guess per-word timings for an edited line.

    Words present in both the original and the edit are anchors (their original
    times are kept); the words between them are guessed by spreading the
    available time in proportion to their syllable counts. When two anchors are
    jammed together (no room for the inserted words) a little time is borrowed
    from those anchors so the new words still get a slice.
    """
    tokens = [token for token in new_text.split() if token]
    if not tokens:
        return []

    original = [
        (str(word.get("word", "")), float(word["start"]), float(word["end"]))
        for word in original_words
        if "start" in word and "end" in word
    ]
    matcher = difflib.SequenceMatcher(
        a=[_normalize_token(word) for word, _start, _end in original],
        b=[_normalize_token(token) for token in tokens],
        autojunk=False,
    )
    times: list[tuple[float, float] | None] = [None] * len(tokens)
    for first, second, size in matcher.get_matching_blocks():
        for step in range(size):
            times[second + step] = (original[first + step][1], original[first + step][2])

    total = len(tokens)
    index = 0
    while index < total:
        if times[index] is not None:
            index += 1
            continue
        end = index
        while end < total and times[end] is None:
            end += 1

        previous = times[index - 1] if index > 0 else None
        following = times[end] if end < total else None
        left = previous[1] if previous else line_start
        right = following[0] if following else line_end
        span = right - left

        syllables = [_syllables(token) for token in tokens[index:end]]
        weight = sum(syllables) or (end - index)

        if span <= 0.02 and previous is not None and following is not None:
            # No room between the anchors: borrow a little from both of them.
            borrowed = [_syllables(tokens[index - 1])] + syllables + [_syllables(tokens[end])]
            borrowed_total = sum(borrowed)
            borrowed_span = following[1] - previous[0]
            cursor = previous[0]
            allocated: list[tuple[float, float]] = []
            for value in borrowed:
                share = borrowed_span * (value / borrowed_total) if borrowed_total else 0.0
                allocated.append((cursor, cursor + share))
                cursor += share
            times[index - 1] = allocated[0]
            for offset in range(index, end):
                times[offset] = allocated[1 + (offset - index)]
            times[end] = allocated[-1]
            index = end
            continue

        cursor = left
        for offset in range(index, end):
            share = span * (syllables[offset - index] / weight) if weight else 0.0
            times[offset] = (cursor, cursor + share)
            cursor += share
        index = end

    result: list[dict[str, Any]] = []
    for token, bounds in zip(tokens, times):
        start, finish = bounds if bounds is not None else (line_start, line_start)
        finish = max(start, finish)
        result.append(
            {
                "word": token,
                "start": round(start, 3),
                "end": round(finish, 3),
                "length": round(finish - start, 3),
            }
        )
    return result


@app.patch("/sessions/{session_id}/takes/{take_id}/lines/{line_index}")
def update_line_text(
    session_id: str,
    take_id: str,
    line_index: int,
    text: str = Query(
        default="",
        max_length=2000,
        description="New line text; empty restores the pipeline words.",
    ),
) -> dict[str, Any]:
    """Rewrite one lyric line's words from edited text.

    The words shared with the original are used as timing anchors and the rest
    are guessed by syllable count, so the stored per-word timings actually
    reflect what the line now says. Clearing the text restores the originals.
    """
    sid = sanitize_id(session_id)
    tid = sanitize_id(take_id)
    result_path = _take_dir(sid, tid) / "result.json"
    result = _read_json(result_path)
    lyrics = result.get("lyrics") if isinstance(result, dict) else None
    if not isinstance(lyrics, list) or not 0 <= line_index < len(lyrics):
        raise HTTPException(status_code=404, detail="Unknown line.")

    line = lyrics[line_index]
    display = text.strip()
    if display:
        if "original_words" not in line:
            line["original_words"] = line.get("words") or []
        line["words"] = _retime_words(
            line["original_words"], display, float(line["start time"]), float(line["end time"])
        )
        line["text"] = display
    else:
        if "original_words" in line:
            line["words"] = line.pop("original_words")
        line.pop("text", None)

    _write_json(result_path, result)
    _touch_session(sid)
    public = {key: value for key, value in line.items() if key != "original_words"}
    return {"take_id": tid, "line_index": line_index, "line": public}


def _clean_words(words: Any) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    if not isinstance(words, list):
        return cleaned
    for word in words:
        if not isinstance(word, dict):
            continue
        text = str(word.get("word", "")).strip()
        start = word.get("start")
        end = word.get("end")
        if not text or start is None or end is None:
            continue
        start = float(start)
        end = max(start, float(end))
        cleaned.append(
            {"word": text, "start": round(start, 3), "end": round(end, 3), "length": round(end - start, 3)}
        )
    return cleaned


@app.put("/sessions/{session_id}/takes/{take_id}/lyrics")
def replace_lyrics(
    session_id: str, take_id: str, body: dict[str, Any] = Body(default={})
) -> dict[str, Any]:
    """Replace a take's lyric lines (re-splitting) and recompute their BPM.

    Body: ``{"lines": [{"words": [...]}, ...]}``. Each new line's
    ``start time``/``end time`` come from its first/last word, and its BPM is
    estimated from the isolated vocal exactly like an auto-detected line.
    """
    lines_in = body.get("lines") if isinstance(body, dict) else None
    if not isinstance(lines_in, list) or not lines_in:
        raise HTTPException(status_code=400, detail="Provide at least one line.")

    sid = sanitize_id(session_id)
    tid = sanitize_id(take_id)
    take_dir = _take_dir(sid, tid)
    result_path = take_dir / "result.json"
    result = _read_json(result_path)
    if not isinstance(result, dict):
        raise HTTPException(status_code=404, detail="Unknown take.")
    vocals = take_dir / "vocals.m4a"
    if not vocals.is_file():
        raise HTTPException(status_code=404, detail="No isolated vocals for this take.")

    try:
        audio, sample_rate = pipeline.load_vocal_mono(str(vocals))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    new_lines: list[dict[str, Any]] = []
    for item in lines_in:
        words = _clean_words(item.get("words") if isinstance(item, dict) else None)
        if not words:
            continue
        start = words[0]["start"]
        end = words[-1]["end"]
        start_sample = max(0, int(start * sample_rate))
        end_sample = max(start_sample, int(end * sample_rate))
        line: dict[str, Any] = {
            "bpm": pipeline.estimate_line_bpm(audio[start_sample:end_sample], sample_rate),
            "start time": start,
            "end time": end,
            "words": words,
        }
        # A ``raw`` line is mid-re-split: shown as plain text (no bubble) until
        # the user carves bubbles out of it.
        if isinstance(item, dict) and item.get("raw"):
            line["raw"] = True
        new_lines.append(line)

    if not new_lines:
        raise HTTPException(status_code=400, detail="No usable words in the new lines.")

    result["lyrics"] = new_lines
    _write_json(result_path, result)
    _touch_session(sid)
    return {"lyrics": new_lines, "bpm": result.get("bpm")}


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
            if path.is_dir() and path.name != SCRIBBLES_SESSION_ID:
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
# Scribbles (a journal shared across every session)
# ---------------------------------------------------------------------------
#
# A scribble is exactly a take, but stored in the reserved ``scribbles`` session
# instead of a song session. That means stems, lyric edits and scrapbook line
# references all work with no special cases; these routes are thin, explicit
# aliases so the client (and the docs) never have to know the reserved id.


def _scribble_metas() -> list[dict[str, Any]]:
    takes_dir = _session_dir(SCRIBBLES_SESSION_ID) / "takes"
    scribbles: list[dict[str, Any]] = []
    if takes_dir.is_dir():
        for take in takes_dir.iterdir():
            meta = _read_json(take / "meta.json") if take.is_dir() else None
            if isinstance(meta, dict):
                scribbles.append(meta)
    scribbles.sort(
        key=lambda meta: (meta.get("created_at") or "", int(meta.get("index") or 0)),
        reverse=True,
    )
    return scribbles


@app.get("/scribbles")
def list_scribbles() -> dict[str, Any]:
    """List every scribble, newest first. Always ``200``, even when empty."""
    return {"scribbles": _scribble_metas()}


def _clean_text_lines(raw: Any) -> list[str]:
    """Normalise a text scribble's lines to trimmed, non-empty strings."""
    lines: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                item = item.get("text", "")
            text = str(item).strip()
            if text:
                lines.append(text[:2000])
    return lines


def _text_lyrics(lines: list[str]) -> list[dict[str, Any]]:
    return [
        {"bpm": 0.0, "start time": 0.0, "end time": 0.0, "words": [], "text": text}
        for text in lines
    ]


# NOTE: registered before ``/scribbles/{scribble_id}`` so the literal path wins.
@app.post("/scribbles/text")
def create_text_scribble(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Create a text-only scribble (no audio) from ``{name, lines}``."""
    name = _clean_name(body.get("name"), "Text scribble")
    lines = _clean_text_lines(body.get("lines"))
    scribble_id = f"sb-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
    take_dir = _take_dir(SCRIBBLES_SESSION_ID, scribble_id)
    take_dir.mkdir(parents=True, exist_ok=True)
    now = _now()
    meta = {
        "session_id": SCRIBBLES_SESSION_ID,
        "take_id": scribble_id,
        "index": _next_index(SCRIBBLES_SESSION_ID),
        "name": name,
        "bookmarked": False,
        "processed": True,
        "status": "done",
        "kind": "text",
        "error": None,
        "created_at": now,
        "finished_at": now,
    }
    _write_json(take_dir / "meta.json", meta)
    _write_json(take_dir / "result.json", {"kind": "text", "lyrics": _text_lyrics(lines)})
    _touch_session(SCRIBBLES_SESSION_ID)
    return meta


@app.post("/scribbles/{scribble_id}")
def create_scribble(
    scribble_id: str,
    file: UploadFile = File(..., description="The recorded scribble."),
    index: int | None = Query(default=None, ge=1),
    name: str | None = Query(default=None),
    process: bool = Query(default=True),
    silence_threshold: float = Query(default=pipeline.SILENCE_THRESHOLD, gt=0, le=5),
) -> JSONResponse:
    """Store a scribble, processing it now unless ``process=false``."""
    return create_take(
        SCRIBBLES_SESSION_ID, scribble_id, file, index, name, process, silence_threshold
    )


@app.post("/scribbles/{scribble_id}/process")
def process_scribble(
    scribble_id: str,
    silence_threshold: float = Query(default=pipeline.SILENCE_THRESHOLD, gt=0, le=5),
) -> JSONResponse:
    """Process a previously stored scribble and return its result."""
    return process_take(SCRIBBLES_SESSION_ID, scribble_id, silence_threshold)


@app.get("/scribbles/{scribble_id}")
def get_scribble(scribble_id: str) -> dict[str, Any]:
    """Return one scribble's metadata plus its result, when available."""
    return get_take(SCRIBBLES_SESSION_ID, scribble_id)


@app.put("/scribbles/{scribble_id}/text")
def update_text_scribble(
    scribble_id: str, body: dict[str, Any] = Body(default={})
) -> dict[str, Any]:
    """Replace a text scribble's lines."""
    take_dir = _take_dir(SCRIBBLES_SESSION_ID, scribble_id)
    meta = _read_json(take_dir / "meta.json")
    if not isinstance(meta, dict):
        raise HTTPException(status_code=404, detail="Unknown scribble.")
    if meta.get("kind") != "text":
        raise HTTPException(status_code=400, detail="Not a text scribble.")
    lines = _clean_text_lines(body.get("lines"))
    _write_json(take_dir / "result.json", {"kind": "text", "lyrics": _text_lyrics(lines)})
    _touch_session(SCRIBBLES_SESSION_ID)
    return {"lyrics": _text_lyrics(lines)}


@app.patch("/scribbles/{scribble_id}")
def update_scribble(
    scribble_id: str,
    name: str | None = Query(default=None, min_length=1),
    bookmarked: bool | None = Query(default=None),
) -> dict[str, Any]:
    """Rename a scribble and/or toggle its bookmark; audio is untouched."""
    return update_take(SCRIBBLES_SESSION_ID, scribble_id, name, bookmarked)


@app.patch("/scribbles/{scribble_id}/lines/{line_index}")
def update_scribble_line(
    scribble_id: str,
    line_index: int,
    text: str = Query(default="", max_length=2000),
) -> dict[str, Any]:
    """Rewrite one scribble line's words from edited text."""
    return update_line_text(SCRIBBLES_SESSION_ID, scribble_id, line_index, text)


@app.put("/scribbles/{scribble_id}/lyrics")
def replace_scribble_lyrics(
    scribble_id: str, body: dict[str, Any] = Body(default={})
) -> dict[str, Any]:
    """Replace a scribble's lyric lines and recompute their BPM."""
    return replace_lyrics(SCRIBBLES_SESSION_ID, scribble_id, body)


@app.get("/scribbles/{scribble_id}/audio/{stem}")
def get_scribble_audio(scribble_id: str, stem: str) -> FileResponse:
    """Stream a scribble's generated stem (``vocals`` or ``background``)."""
    return get_take_audio(SCRIBBLES_SESSION_ID, scribble_id, stem)


@app.get("/scribbles/{scribble_id}/original")
def get_scribble_original(scribble_id: str) -> FileResponse:
    """Stream a scribble's raw recording."""
    return get_take_original(SCRIBBLES_SESSION_ID, scribble_id)


@app.delete("/scribbles/{scribble_id}")
def delete_scribble(scribble_id: str) -> dict[str, Any]:
    """Delete a scribble and all of its files."""
    return delete_take(SCRIBBLES_SESSION_ID, scribble_id)


# ---------------------------------------------------------------------------
# Scrapbooks
# ---------------------------------------------------------------------------
#
# A scrapbook is an ordered list of lines referenced from processed takes. It is
# the *only* place lines can be added, removed or reordered. Each line keeps its
# recorded tempo; only the relative tempo between neighbours is adjusted when
# rendering. The result is cached against a signature of the line list.


def _scrapbook_dir(session_id: str, scrapbook_id: str) -> Path:
    """Scrapbooks live inside their session so deleting a session cleans them up."""
    return _session_dir(session_id) / "scrapbooks" / sanitize_id(scrapbook_id)


def _scrapbook_meta_path(session_id: str, scrapbook_id: str) -> Path:
    return _scrapbook_dir(session_id, scrapbook_id) / "scrapbook.json"


#: Section types a scrapbook line can be grouped under (UI colour-coding only).
SECTION_TYPES = ("chorus", "verse", "bridge", "other")


def _normalize_line_ref(ref: Any) -> dict[str, Any] | None:
    """Validate one bare line, snapshotting it if it arrived without ``data``."""
    if not isinstance(ref, dict):
        return None
    session_id = str(ref.get("session_id", "")).strip()
    take_id = str(ref.get("take_id", "")).strip()
    try:
        line_index = int(ref.get("line_index", -1))
    except (TypeError, ValueError):
        return None
    if not session_id or not take_id or line_index < 0:
        return None

    sid = sanitize_id(session_id)
    tid = sanitize_id(take_id)
    data = ref.get("data")
    if not isinstance(data, dict):
        try:
            resolved = _line_data({"session_id": sid, "take_id": tid, "line_index": line_index})
        except HTTPException:
            return None
        data = {key: value for key, value in resolved.items() if key != "vocals"}
    return {"session_id": sid, "take_id": tid, "line_index": line_index, "data": data}


def _normalize_group(entry: Any) -> dict[str, Any] | None:
    """Validate a section group: ``{group: {id, type, name}, lines: [...]}``."""
    if not isinstance(entry, dict):
        return None
    group = entry.get("group")
    if not isinstance(group, dict) or group.get("type") not in SECTION_TYPES:
        return None
    group_id = str(group.get("id", "")).strip()[:64]
    if not group_id:
        return None
    name = str(group.get("name", "")).strip()
    instance = str(group.get("instance", "")).strip()[:64]
    inner: list[dict[str, Any]] = []
    for line in entry.get("lines") or []:
        normalized = _normalize_line_ref(line)
        if normalized:
            inner.append(normalized)
    if not inner:
        return None
    return {
        "group": {
            "id": group_id,
            "type": str(group["type"]),
            "name": _clean_name(name or str(group["type"]).capitalize(), str(group["type"])),
            # A unique per-entry instance id, so two copies of a section stay
            # separate even when they sit next to each other.
            "instance": instance or f"{group_id}-{os.urandom(3).hex()}",
        },
        "lines": inner,
    }


def _normalize_lines(lines: Any) -> list[dict[str, Any]]:
    """Validate scrapbook entries, snapshotting each line as it is added.

    An entry is either a bare line or a section group (``{group, lines}``). A
    line sent without ``data`` is resolved from the journal *now* and its
    words/timings are stored with it, so later edits to the journal (including
    re-splitting) cannot change a scrapbook that already holds it.
    """
    if not isinstance(lines, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen_instances: set[str] = set()
    for entry in lines:
        if isinstance(entry, dict) and isinstance(entry.get("group"), dict):
            group = _normalize_group(entry)
            if group:
                # A split section (two entries sharing an instance) would break the
                # client; give the later one its own instance.
                instance = group["group"]["instance"]
                if instance in seen_instances:
                    instance = f"{group['group']['id']}-{os.urandom(3).hex()}"
                    group["group"]["instance"] = instance
                seen_instances.add(instance)
                normalized.append(group)
            continue
        line = _normalize_line_ref(entry)
        if line:
            normalized.append(line)
    return normalized


def _expand_lines(lines: Any) -> list[dict[str, Any]]:
    """Flatten the entry list into the line sequence that actually plays."""
    expanded: list[dict[str, Any]] = []
    if not isinstance(lines, list):
        return expanded
    for entry in lines:
        if isinstance(entry, dict) and isinstance(entry.get("group"), dict):
            expanded.extend(entry.get("lines") or [])
        elif isinstance(entry, dict):
            expanded.append(entry)
    return expanded


def _scrapbook_entry(ref: dict[str, Any]) -> dict[str, Any]:
    """A scrapbook line's snapshot (or live data) plus its vocals path."""
    sid = sanitize_id(str(ref.get("session_id", "")))
    tid = sanitize_id(str(ref.get("take_id", "")))
    index = int(ref.get("line_index", 0))
    vocals = _take_dir(sid, tid) / "vocals.m4a"
    if not vocals.is_file():
        raise HTTPException(status_code=400, detail=f"No isolated vocals for {sid}/{tid}.")

    data = ref.get("data")
    if not isinstance(data, dict):
        data = _line_data({"session_id": sid, "take_id": tid, "line_index": index})
    entry = {key: value for key, value in data.items() if key != "vocals"}
    entry["vocals"] = vocals
    entry["source"] = {"session_id": sid, "take_id": tid, "line_index": index}
    return entry


def _scrapbook_signature(lines: list[dict[str, Any]]) -> str:
    # Grouping is visual: the mix only depends on the flattened line sequence.
    payload = json.dumps({"lines": _expand_lines(lines)}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_scrapbook(session_id: str, scrapbook_id: str) -> dict[str, Any]:
    meta = _read_json(_scrapbook_meta_path(session_id, scrapbook_id))
    if not isinstance(meta, dict):
        raise HTTPException(status_code=404, detail=f"Unknown scrapbook {scrapbook_id!r}.")
    return meta


def _scrapbook_summary(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "scrapbook_id": meta.get("scrapbook_id"),
        "name": meta.get("name"),
        "line_count": len(_expand_lines(meta.get("lines") or [])),
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
    take_bpm = result.get("bpm") if isinstance(result, dict) else None
    return {
        "bpm": float(line.get("bpm") or 120.0),
        "text": line.get("text"),
        "take_bpm": float(take_bpm) if take_bpm else None,
        "prev_bpm": float(previous.get("bpm") or 0.0) if previous else None,
        "next_bpm": float(following.get("bpm") or 0.0) if following else None,
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


def _render_scrapbook(session_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    sid = str(meta["scrapbook_id"])
    lines = meta.get("lines") or []
    expanded = _expand_lines(lines)
    signature = _scrapbook_signature(lines)
    book_dir = _scrapbook_dir(session_id, sid)
    mix = book_dir / "mix.m4a"

    cached = meta.get("render")
    if isinstance(cached, dict) and cached.get("signature") == signature and mix.is_file():
        return cached

    if not expanded:
        raise HTTPException(status_code=400, detail="Add at least one line before rendering.")

    entries = [_scrapbook_entry(ref) for ref in expanded]
    with PROCESS_LOCK:
        timeline, gaps, _mix = scrapbook.render(entries, book_dir)

    render = {
        "signature": signature,
        "audio": f"/sessions/{sanitize_id(session_id)}/scrapbooks/{sid}/audio",
        "timeline": timeline,
        "gaps": gaps,
        "rendered_at": _now(),
    }
    # Merge into the *current* file: an edit may have landed while we rendered,
    # and its lines must not be overwritten by our older snapshot.
    meta_path = _scrapbook_meta_path(session_id, sid)
    current = _read_json(meta_path)
    if not isinstance(current, dict):
        current = meta
    current["render"] = render
    current["updated_at"] = _now()
    _write_json(meta_path, current)
    return render


@app.get("/sessions/{session_id}/scrapbooks")
def list_scrapbooks(session_id: str) -> dict[str, Any]:
    """List a session's scrapbooks, most recently updated first."""
    scrapbooks_dir = _session_dir(session_id) / "scrapbooks"
    items = []
    if scrapbooks_dir.is_dir():
        for path in scrapbooks_dir.iterdir():
            if not path.is_dir():
                continue
            meta = _read_json(path / "scrapbook.json")
            if isinstance(meta, dict):
                items.append(_scrapbook_summary(meta))
    items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return {"scrapbooks": items}


@app.post("/sessions/{session_id}/scrapbooks")
def create_scrapbook(session_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Create a scrapbook in ``session_id`` from ``{name, lines}``.

    There is no global BPM: each line keeps its recorded tempo and only the
    relative tempo between neighbours is adjusted when rendering.
    """
    sid = sanitize_id(session_id)
    name = _clean_name(body.get("name"), "Scrapbook")
    lines = _normalize_lines(body.get("lines"))
    scrapbook_id = f"sb-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
    now = _now()
    meta = {
        "scrapbook_id": scrapbook_id,
        "session_id": sid,
        "name": name,
        "lines": lines,
        "created_at": now,
        "updated_at": now,
    }
    _scrapbook_dir(sid, scrapbook_id).mkdir(parents=True, exist_ok=True)
    _write_json(_scrapbook_meta_path(sid, scrapbook_id), meta)
    return _scrapbook_summary(meta)


@app.get("/sessions/{session_id}/scrapbooks/{scrapbook_id}")
def get_scrapbook(session_id: str, scrapbook_id: str) -> dict[str, Any]:
    """Return a scrapbook with its lines and last render, if any."""
    meta = _load_scrapbook(session_id, scrapbook_id)
    result = _scrapbook_summary(meta)
    result["lines"] = meta.get("lines") or []
    result["render"] = meta.get("render")
    return result


@app.patch("/sessions/{session_id}/scrapbooks/{scrapbook_id}")
def update_scrapbook(
    session_id: str, scrapbook_id: str, body: dict[str, Any] = Body(default={})
) -> dict[str, Any]:
    """Update ``name`` and/or ``lines``. Edits invalidate the render."""
    meta = _load_scrapbook(session_id, scrapbook_id)
    if "name" in body:
        meta["name"] = _clean_name(body.get("name"), meta.get("name") or "Scrapbook")
    if "lines" in body:
        meta["lines"] = _normalize_lines(body.get("lines"))

    meta["updated_at"] = _now()
    _write_json(_scrapbook_meta_path(session_id, scrapbook_id), meta)
    result = _scrapbook_summary(meta)
    result["lines"] = meta.get("lines") or []
    result["render"] = meta.get("render")
    return result


@app.delete("/sessions/{session_id}/scrapbooks/{scrapbook_id}")
def delete_scrapbook(session_id: str, scrapbook_id: str) -> dict[str, Any]:
    """Delete a scrapbook and its rendered mix."""
    sid = sanitize_id(scrapbook_id)
    path = _scrapbook_dir(session_id, sid)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown scrapbook {sid!r}.")
    shutil.rmtree(path)
    return {"deleted": True, "scrapbook_id": sid}


@app.post("/sessions/{session_id}/scrapbooks/{scrapbook_id}/render")
def render_scrapbook(session_id: str, scrapbook_id: str) -> dict[str, Any]:
    """Render (or return the cached) tempo-matched mix for a scrapbook."""
    meta = _load_scrapbook(session_id, scrapbook_id)
    return _render_scrapbook(session_id, meta)


@app.get("/sessions/{session_id}/scrapbooks/{scrapbook_id}/audio")
def get_scrapbook_audio(session_id: str, scrapbook_id: str) -> FileResponse:
    """Stream a scrapbook's rendered mix."""
    meta = _load_scrapbook(session_id, scrapbook_id)
    sid = str(meta["scrapbook_id"])
    mix = _scrapbook_dir(session_id, sid) / "mix.m4a"
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
