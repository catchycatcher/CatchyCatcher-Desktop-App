# Catchy Catcher — Processing Engine (backend)

FastAPI server that turns freeform singing takes into separated stems and a
line-by-line lyric timeline. Runs on the GPU desktop; the laptop client talks
to it over the local network (or Tailscale).

The client works in **sessions** (one per song/idea) made of **takes** (one
audio file each). A take is uploaded and processed **synchronously** — the
client shows a spinner while it waits, then records the next take — and takes
are ~minutes apart, so the GPU is never contended.

```
take.m4a ──► Demucs (htdemucs) ──► vocals.m4a + background.m4a
                                      │
                                      ▼
                      WhisperX (medium + wav2vec2 alignment)
                                      │
                                      ▼
                       group words on a 0.4s pause threshold
                                      │
                                      ▼
                           per-line BPM (librosa) ──► JSON
```

The full specification lives in [`../SYSTEM_INSTRUCTIONS.md`](../SYSTEM_INSTRUCTIONS.md).

## Files

| File | Purpose |
| --- | --- |
| `pipeline.py` | `process_song_session()` — the whole DSP/AI pipeline for one take; returns the spec JSON. Caches models between takes. |
| `server.py` | FastAPI app: sessions, takes, stem streaming, health. |
| `run.py` | One-command launcher: starts the server and the share tunnel. |
| `tunnel.py` | Gradio share tunnel, driven by `run.py`. |
| `share_code.py` | Dependency-free base62 share-code maths (kept out of `tunnel.py` so it is testable). |
| `test_pipeline.py` | 24 `unittest` tests: grouping, BPM fallback, id sanitising, share codes, the full flow (heavy stack stubbed) and the HTTP API. |
| `setup_env.py` | One-command cross-platform installer (uv + Python + ffmpeg + deps). |
| `requirements.txt` | Python dependencies. |

## Setup

### One-command setup (Windows, macOS, Linux)

With any Python 3.8+ on `PATH`:

```sh
python setup_env.py
```

`setup_env.py` bootstraps the whole environment:

- installs [`uv`](https://docs.astral.sh/uv/) (and a managed Python 3.12 if the
  system interpreter is unusable);
- installs `ffmpeg` with the platform package manager (`winget` / `brew` /
  `apt` / `dnf` / `pacman` / `zypper` / `apk`);
- creates `.venv` and installs the pinned PyTorch build — **CUDA** when an
  NVIDIA GPU is detected, **CPU** otherwise, and **MPS** on macOS;
- installs the rest of `requirements.txt`;
- downloads the NLTK tokenizer data WhisperX alignment needs.

Useful flags: `--cpu`, `--python 3.11`, `--torch-index <url>`, `--skip-ffmpeg`,
`--skip-nltk`. The script is safe to re-run.

> On Windows, open a **new** terminal after the first run so the freshly
> installed `ffmpeg` shows up on `PATH`.

### Manual setup

`ffmpeg` must be on `PATH` (verify with `ffmpeg -version`).

```sh
cd backend
python3 -m venv .venv
source .venv/bin/activate

# CUDA build of PyTorch first (RTX 4060):
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# Everything else:
pip install -r requirements.txt
```

> **Keep PyTorch on the 2.8.0 line.** From torchaudio 2.9 on, `load`/`save`
> hard-require TorchCodec, which needs FFmpeg *shared* libraries (v4-7) that
> many systems do not have; the 2.8.x torchaudio still uses the
> soundfile/ffmpeg fallback `pipeline.py` relies on. `setup_env.py` pins this
> automatically.

> CPU-only machine? Install torch from the CPU index instead — the CUDA wheels
> are ~2.5 GB and useless without an NVIDIA GPU:
> `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`

On a machine without the GPU stack the server still starts — `/health` reports
`"device": "unavailable"` and take uploads return `503` with install
instructions instead of crashing.

Model weights (Demucs `htdemucs`, WhisperX `medium` + alignment) download on
first use and are cached under `~/.cache`.

## Run

### One-command start (recommended)

```sh
cd backend
python run.py
```

`run.py` uses the project's `.venv` automatically (no activation needed), starts
the API server, waits for `/health`, then opens a Gradio share tunnel and prints
a base62 **share code**. Paste that code into the frontend
(https://catchycatcher.github.io) to connect.

```
================================================================
  Catchy Catcher processing server
================================================================
  API       http://127.0.0.1:8000
  Docs      http://127.0.0.1:8000/docs
  Frontend  https://catchycatcher.github.io
================================================================
...
Running on public URL: https://<hex>.gradio.live
Share code (base-62): <code>
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--no-tunnel` | Skip the share tunnel; reach the API on the LAN only. |
| `--port 9000` | Listen on a different port. |
| `--host 127.0.0.1` | Bind to a specific interface. |

The share tunnel uses [`gradio`](https://gradio.app). If it is not installed,
`run.py` still starts the server and tells you the tunnel is unavailable — add it
with `pip install gradio`.

### Manual start

```sh
# from backend/, with the venv active
python server.py
# or
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://<desktop-ip>:8000/docs` for interactive Swagger UI.

### Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `CATCHYCATCHER_HOST` | `0.0.0.0` | Bind host. |
| `CATCHYCATCHER_PORT` | `8000` | Bind port. |
| `CATCHYCATCHER_UPLOAD_DIR` | `./sessions` | Session storage root. |
| `CATCHYCATCHER_CORS_ORIGINS` | `*` | Comma-separated allowed origins, or `*`. |

Because the frontend is hosted statically (GitHub Pages / `localhost:5173`),
requests are cross-origin. Lock this down in production, e.g.
`CATCHYCATCHER_CORS_ORIGINS="https://catchycatcher.github.io,http://localhost:5173"`.

## Storage layout

```
sessions/
  <session_id>/
    takes/
      <take_id>/
        input.m4a        # the uploaded take
        meta.json        # take_id, index, status, error, timestamps
        result.json      # the spec dict, once finished
        vocals.m4a
        background.m4a
```

`session_id` and `take_id` are sanitised to a single safe path segment, so
`../../etc` cannot escape the root.

## API

### `POST /sessions/{session_id}/takes/{take_id}`

`multipart/form-data` with a single field, `file` (the recorded take). Blocks
until the take is processed, then returns the spec schema.

Optional query parameters:

| Param | Default | Meaning |
| --- | --- | --- |
| `index` | one past the highest existing take | 1-based position within the session. |
| `silence_threshold` | `0.4` | Pause (seconds, `0 < x <= 5`) that starts a new lyric line. |

Response body:

```json
{
  "vocals": "/abs/path/sessions/song1/takes/3/vocals.m4a",
  "background": "/abs/path/sessions/song1/takes/3/background.m4a",
  "lyrics": [
    {
      "bpm": 120.0,
      "start time": 12.45,
      "end time": 15.82,
      "words": [
        { "word": "hello", "length": 0.37 },
        { "word": "world", "length": 0.33 }
      ]
    }
  ]
}
```

The sanitised ids are echoed in the `X-Session-Id` and `X-Take-Id` response
headers.

### Other endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | `{"status": "ok", "device": "cuda"}` |
| `GET` | `/sessions` | List session ids. |
| `POST` | `/sessions/{session_id}` | Create an empty session folder ahead of recording. Idempotent (`created: false` if it already exists). |
| `GET` | `/sessions/{session_id}` | Manifest: every take in order with status. |
| `GET` | `/sessions/{session_id}/takes/{take_id}` | One take's metadata + result. |
| `GET` | `/sessions/{session_id}/takes/{take_id}/audio/{stem}` | Stream `vocals` or `background` (`audio/mp4`). |
| `DELETE` | `/sessions/{session_id}/takes/{take_id}` | Delete a take and its files. |
| `POST` | `/process-session` | Single-shot compat endpoint (spec). Equivalent to a session named after the file with one take called `take`. |

### Example

```sh
# prepare an empty session folder before recording
curl -X POST "http://<desktop-ip>:8000/sessions/song1"

# send take 3 of a session called "song1"
curl -X POST "http://<desktop-ip>:8000/sessions/song1/takes/3" \
  -F "file=@take3.m4a" -o take3.json -D headers.txt

# which take/session did the server store it as?
grep -i 'x-session-id\|x-take-id' headers.txt

# reopen the whole session later (e.g. after a refresh)
curl http://<desktop-ip>:8000/sessions/song1

# play the isolated vocal for that take
curl -o vocals.m4a \
  "http://<desktop-ip>:8000/sessions/song1/takes/3/audio/vocals"
```

A take can take minutes, so make sure the client (and any proxy) allows a long
read timeout.

## Tests

```sh
source .venv/bin/activate
python -m unittest test_pipeline -v
```

20 tests, all with the heavy stack stubbed — no GPU, torch, demucs, whisperx or
librosa required.

## Networking

- Allow inbound TCP `8000` on the desktop firewall.
- Same Wi-Fi: connect the laptop to `http://<desktop-lan-ip>:8000`.
- Off-network: install [Tailscale](https://tailscale.com) on both machines and
  use the desktop's Tailscale IP/hostname. Prefer this over exposing the port.

## Notes on the pipeline

- **Model caching:** Demucs and WhisperX are loaded once and reused across every
  take (keyed by model/device/language), so consecutive takes skip the load
  penalty. `pipeline.clear_model_cache()` drops them.
- **Memory:** a take's Demucs output buffer is proportional to its length
  (~250 MB for 3 minutes at 44.1 kHz stereo), not the whole session.
  `apply_model` runs with `split=True`, chunking internally regardless.
- **Storage:** stems are exported as 128 kbps AAC (`.m4a`); temporary WAVs are
  deleted in a `finally` block.
- **Serialisation:** takes run one at a time behind a lock, so two uploads never
  fight over the 8 GB of VRAM.
- **Alignment:** WhisperX forced alignment gives millisecond word boundaries;
  words missing `start`/`end` are skipped.
- **Line grouping:** tune `silence_threshold` if phrases group too tightly or
  sparsely.
- **BPM:** estimated per line from the vocal stem (`start_bpm=110`). Lines
  shorter than `0.5s` (or that fail analysis) fall back to `120.0` bpm.
