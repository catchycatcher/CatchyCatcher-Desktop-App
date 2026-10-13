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
                      group words into lines (break model)
                                      │
                                      ▼
                          per-line BPM (librosa) ──► JSON
```

The full specification lives in [`../SYSTEM_INSTRUCTIONS.md`](../SYSTEM_INSTRUCTIONS.md).

## Files

| File | Purpose |
| --- | --- |
| `pipeline.py` | `process_song_session()` — the whole DSP/AI pipeline for one take; returns the spec JSON. Also holds `group_words_into_lines()`, the four-stage line-break model. Caches models between takes. |
| `phonetics.py` | Rhyme helpers backed by the CMU pronunciation dictionary (`pronouncing`), with a spelling fallback; used by the line-break model. |
| `server.py` | FastAPI app: sessions, takes, scrapbooks, stem streaming, health. |
| `scrapbook.py` | Scrapbook rendering: ffmpeg `atempo` time-stretch, gap blending, mix concat. |
| `run.py` | One-command launcher: sets the environment up if needed, then starts the server and the share tunnel. |
| `tunnel.py` | Gradio share tunnel, driven by `run.py`. |
| `share_code.py` | Dependency-free base62 share-code maths (kept out of `tunnel.py` so it is testable). |
| `test_pipeline.py` | 58 `unittest` tests: line grouping (including the frozen `example.m4a` timeline), phonetics, BPM fallback, id sanitising, share codes, the full flow (heavy stack stubbed) and the HTTP API. |
| `test_fixtures/example_words.json` | The WhisperX word timeline of `example.m4a`, captured once, used as the grouping regression case. |
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

You do not have to run this yourself: `python run.py` runs it automatically the
first time (whenever `.venv` is missing), so a fresh checkout really only needs
one command.

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

`run.py` uses the project's `.venv` automatically (no activation needed). If the
venv is missing it first runs `setup_env.py` for you, so on a fresh checkout
`python run.py` is the *only* command you need. It then starts the API server,
waits for `/health`, opens a Gradio share tunnel and prints a base62 **share
code**. Paste that code into the frontend
(https://catchycatcher.github.io) to connect.

The normal output is deliberately short and user focused — the share code is the
only thing on screen:

```
Starting Catchy Catcher...

Have https://catchycatcher.github.io open, then enter the share code below.

==========================================================
        YOUR SHARE CODE

        zz4-k1q

==========================================================
```

The code is printed grouped in fours (with a tidy tail, so there is never a
lone character). The hyphens are cosmetic — the frontend accepts the code with
or without them.

Pass `--verbose` to get the full startup detail instead (bind address, docs,
public tunnel url, expiry notice and uvicorn logs):

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
Share code (base-62): <grouped-code>
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--no-tunnel` | Skip the share tunnel; reach the API on the LAN only. |
| `--port 9000` | Listen on a different port. |
| `--host 127.0.0.1` | Bind to a specific interface. |
| `--verbose` | Show all startup details (API/tunnel urls, diagnostics, logs). |

The share tunnel uses [`gradio`](https://gradio.app) and is included in
`requirements.txt`. It is pinned below 6 (`gradio<6`) on purpose: Gradio 6
requires `huggingface-hub>=2`, which breaks WhisperX/transformers. If it is not
installed, `run.py` still starts the server and tells you the tunnel is
unavailable — add it with `pip install "gradio<6"`.

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
| `CATCHYCATCHER_SCRAPBOOK_DIR` | `./scrapbooks` | Scrapbook storage root. |
| `CATCHYCATCHER_CORS_ORIGINS` | `*` | Comma-separated allowed origins, or `*`. |
| `CATCHYCATCHER_LOG_LEVEL` | `info` | Uvicorn log level (`run.py` sets `warning` unless `--verbose`). |
| `CATCHYCATCHER_REQUEST_LOG` | `info` | Level for the per-request log line; `off` silences it. |

Because the frontend is hosted statically (GitHub Pages / `localhost:5173`),
requests are cross-origin. Lock this down in production, e.g.
`CATCHYCATCHER_CORS_ORIGINS="https://catchycatcher.github.io,http://localhost:5173"`.

## Storage layout

```
sessions/
  <session_id>/
    session.json         # display name, created_at, updated_at
    takes/
      <take_id>/
        input.m4a        # the uploaded take
        meta.json        # take_id, index, status, error, timestamps
        result.json      # the spec dict, once finished
        vocals.m4a
        background.m4a
```

A session's `id` is its folder name (URL-safe and stable); its **display name**
is free-form and lives in `session.json`. Sessions created before that file
existed simply fall back to showing the id.

`session_id` and `take_id` are sanitised to a single safe path segment, so
`../../etc` cannot escape the root.

Scrapbooks live beside the sessions:

```
scrapbooks/
  <scrapbook_id>/
    scrapbook.json       # name, bpm, ordered line refs, cached render
    mix.m4a              # the rendered, tempo-matched mix
    render/              # per-line/gap scratch WAVs (safe to delete)
```

A scrapbook line is just `{session_id, take_id, line_index}`; the audio is always
read fresh from the referenced take's `vocals.m4a`.

## API

### `POST /sessions/{session_id}/takes/{take_id}`

`multipart/form-data` with a single field, `file` (the recorded take). Blocks
until the take is processed, then returns the spec schema.

Optional query parameters:

| Param | Default | Meaning |
| --- | --- | --- |
| `index` | one past the highest existing take | 1-based position within the session. |
| `name` | the take id | Display name for the entry. |
| `process` | `true` | `false` stores the audio only; process later via `POST …/process`. |
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
| `GET` | `/sessions` | List sessions as `{"id", "name", "updated_at"}` objects, ordered by id. |
| `POST` | `/sessions/{session_id}` | Create an empty session folder ahead of recording. Idempotent (`created: false` if it already exists). Optional `?name=` sets the display name (kept on re-post when omitted). |
| `PATCH` | `/sessions/{session_id}` | Rename a session (`?name=…`); takes are untouched. |
| `DELETE` | `/sessions/{session_id}` | Delete a session and every take inside it. |
| `GET` | `/sessions/{session_id}` | Manifest: the session `name`, then every take in order with status. |
| `GET` | `/sessions/{session_id}/takes/{take_id}` | One take's metadata + result. |
| `PATCH` | `/sessions/{session_id}/takes/{take_id}` | Rename an entry and/or toggle its bookmark (`?name=…`, `?bookmarked=true|false`). |
| `POST` | `/sessions/{session_id}/takes/{take_id}/process` | Process a take that was stored with `process=false`. |
| `GET` | `/sessions/{session_id}/takes/{take_id}/audio/{stem}` | Stream `vocals` or `background` (`audio/mp4`). |
| `GET` | `/sessions/{session_id}/takes/{take_id}/original` | Stream the raw recording, processed or not. |
| `DELETE` | `/sessions/{session_id}/takes/{take_id}` | Delete a take and its files. |
| `POST` | `/process-session` | Single-shot compat endpoint (spec). Equivalent to a session named after the file with one take called `take`. |

### Scrapbooks

A scrapbook is an ordered list of lines lifted from processed takes, with one
BPM every line is stretched to (pitch-preserving). It is the only place lines can
be added, removed or reordered. Editing invalidates the cached render.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/scrapbooks` | List scrapbooks, most recently updated first. |
| `POST` | `/scrapbooks` | Create from `{name, lines}` (`lines` are `{session_id, take_id, line_index}`). The BPM starts as the median of the lines' takes. |
| `GET` | `/scrapbooks/{scrapbook_id}` | One scrapbook with its lines and last render. |
| `PATCH` | `/scrapbooks/{scrapbook_id}` | Update `name`/`bpm`/`lines`. Setting `bpm` pins it; `bpm_manual: false` hands it back to the automatic median. |
| `DELETE` | `/scrapbooks/{scrapbook_id}` | Delete a scrapbook and its mix. |
| `POST` | `/scrapbooks/{scrapbook_id}/render` | Render (or return the cached) mix + word timeline. |
| `GET` | `/scrapbooks/{scrapbook_id}/audio` | Stream the rendered mix (`audio/mp4`). |

Rendering cuts `[start time, end time]` from each line's isolated vocals, applies
`atempo = scrapbook_bpm / line_bpm`, and inserts a gap between each pair. The gap
is the mean of the pause that followed the first line and the pause that preceded
the second line in their own journals, each scaled by its own tempo change (with
a fallback when a line was at the edge of its journal).

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

58 tests, all with the heavy stack stubbed — no GPU, torch, demucs, whisperx or
librosa required. The example-song grouping tests read the committed word
timeline in `test_fixtures/`, so they are deterministic and offline (the
dictionary-backed phonetics tests skip themselves if `pronouncing` is absent).

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
- **Line grouping:** a four-stage break model. Every word boundary first gets a
  *likelihood* from the pause size (absolute and relative to the singer's other
  gaps) and the sentence/clause punctuation on the word before it. A rhyme pass
  then raises endings that near-rhyme with another nearby likely break, a
  line-length pass nudges gaps toward the size of the surrounding lines, and a
  final rolling threshold commits the breaks: a clear pause always breaks, an
  over-long line is cut at its strongest internal seam, and otherwise the
  threshold drifts to keep the break frequency near the song's typical line.
  A line ending on a comma or full stop also breaks once it is long enough,
  even when the singer runs straight through with no pause (a phrase can end on
  punctuation alone).
  Tune `silence_threshold` for overall tightness; the weights live in the
  `Line-grouping model` block at the top of `pipeline.py`.
- **BPM:** estimated per line from the vocal stem (`start_bpm=110`). Lines
  shorter than `0.5s` (or that fail analysis) fall back to `120.0` bpm.
