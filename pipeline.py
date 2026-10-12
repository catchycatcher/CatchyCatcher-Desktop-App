"""Catchy Catcher audio processing pipeline.

``process_song_session`` accepts a recorded ``.m4a`` and returns the
JSON-compatible payload described in ``SYSTEM_INSTRUCTIONS.md``::

    {
      "vocals": "path/to/vocals.m4a",
      "background": "path/to/background.m4a",
      "lyrics": [
        {
          "bpm": 120.0,
          "start time": 12.45,
          "end time": 15.82,
          "words": [
            {"word": "hello", "length": 0.37},
            {"word": "world", "length": 0.33}
          ]
        }
      ]
    }

The heavy dependencies (torch, torchaudio, demucs, whisperx, librosa) are
imported lazily so the API can start and report health on a machine where they
are not installed. ``process_song_session`` raises a ``RuntimeError`` with the
install instructions when a dependency is missing.

Pipeline stages:

1. Stem separation with Demucs (``htdemucs``) on CUDA when available.
2. Precision transcription + forced phoneme alignment with WhisperX.
3. Word grouping into lyric lines using a pause threshold.
4. Per-line BPM estimation with librosa.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from typing import Any, Callable, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Tunables (see "Implementation Guidelines" in SYSTEM_INSTRUCTIONS.md)
# ---------------------------------------------------------------------------

#: Pause in seconds between two words that starts a new lyric line.
SILENCE_THRESHOLD = 0.4

#: Tempo prior handed to librosa when estimating a line's BPM.
START_BPM = 110.0

#: BPM returned for lines that are too short or too quiet to analyse.
DEFAULT_BPM = 120.0

#: A line must be at least this many seconds long before we trust a BPM.
MIN_BPM_SLICE_SECONDS = 0.5

#: Sample rate Demucs operates at; exported stems and BPM slices use it.
DEFAULT_SAMPLE_RATE = 44100

#: htdemucs source order is [drums, bass, other, vocals].
VOCALS_INDEX = 3


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
#
# A session is a long-lived thing processed as many separate takes. Loading
# Demucs and WhisperX from scratch for every take would waste seconds and churn
# VRAM on each request, so model objects are kept alive between calls. The
# objects are only ever touched while ``server.PROCESS_LOCK`` is held, so the
# cache itself is safe to share.

_MODEL_CACHE: dict[tuple, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def clear_model_cache() -> None:
    """Forget cached models. Useful for tests or a manual VRAM reset."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _cached(key: tuple, factory: Callable[[], Any]) -> Any:
    """Return the cached object for ``key``, building it with ``factory`` once."""
    with _MODEL_CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = factory()
        return _MODEL_CACHE[key]


def detect_device() -> str:
    """Return ``"cuda"``, ``"cpu"`` or ``"unavailable"``.

    Cheap enough to call from a health endpoint: importing torch is the only
    cost, and a missing torch is reported rather than raised.
    """
    try:
        import torch
    except ImportError:
        return "unavailable"
    return "cuda" if torch.cuda.is_available() else "cpu"


def group_words_into_lines(
    words: Iterable[dict[str, Any]],
    silence_threshold: float = SILENCE_THRESHOLD,
) -> list[list[dict[str, Any]]]:
    """Split aligned words into lines wherever the gap between words is large.

    A new line starts when ``word["start"] - previous["end"] >= silence_threshold``.
    Pure Python and dependency-free so it can be unit tested without a GPU.
    """
    words = list(words)
    if not words:
        return []

    lines: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [words[0]]

    for previous, word in zip(words, words[1:]):
        if word["start"] - previous["end"] >= silence_threshold:
            lines.append(current)
            current = []
        current.append(word)

    lines.append(current)
    return lines


def _librosa_tempo_fn(librosa: Any) -> Callable[..., Any]:
    """Pick whichever ``tempo`` entry point the installed librosa exposes.

    librosa 0.10 moved ``beat.tempo`` to ``feature.tempo`` and added
    ``feature.rhythm.tempo``; older releases only have ``beat.tempo``.
    """
    rhythm = getattr(librosa.feature, "rhythm", None)
    if rhythm is not None and hasattr(rhythm, "tempo"):
        return rhythm.tempo
    if hasattr(librosa.feature, "tempo"):
        return librosa.feature.tempo
    return librosa.beat.tempo


def estimate_line_bpm(
    line_slice: "np.ndarray | None",
    sample_rate: int,
    start_bpm: float = START_BPM,
) -> float:
    """Median tempo of a short mono vocal slice, or :data:`DEFAULT_BPM`.

    Never raises: a line that is too short, silent, or that hits a librosa
    error simply falls back to :data:`DEFAULT_BPM`.
    """
    if line_slice is None or len(line_slice) < int(sample_rate * MIN_BPM_SLICE_SECONDS):
        return DEFAULT_BPM

    try:
        import librosa
    except ImportError:
        return DEFAULT_BPM

    try:
        onset_envelope = librosa.onset.onset_strength(y=line_slice, sr=sample_rate)
        tempo = _librosa_tempo_fn(librosa)(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            start_bpm=start_bpm,
        )
        return round(float(np.median(tempo)), 1)
    except Exception:  # noqa: BLE001 - a BPM estimate must never fail a session
        return DEFAULT_BPM


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _load_audio_via_ffmpeg(
    path: str,
    target_sr: int,
    torchaudio: Any,
) -> tuple[Any, int]:
    """Decode any container to a temporary WAV, then load it with torchaudio.

    Used as a fallback when torchaudio's own decoder cannot handle the input
    container (e.g. a torchaudio build without ffmpeg support).
    """
    fd, temp_wav = tempfile.mkstemp(suffix=".wav", prefix="catchy_decode_")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                path,
                "-ac",
                "2",
                "-ar",
                str(target_sr),
                "-f",
                "wav",
                temp_wav,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return torchaudio.load(temp_wav)
    finally:
        _safe_remove(temp_wav)


def _load_audio(path: str, target_sr: int, torchaudio: Any) -> tuple[Any, int]:
    """Load ``path`` as a 2-channel float tensor resampled to ``target_sr``.

    Demucs' ``htdemucs`` expects stereo input at ``target_sr`` (44.1 kHz), so
    mono files are duplicated and everything is resampled when necessary.
    """
    try:
        wav, sr = torchaudio.load(path)
    except Exception:  # noqa: BLE001 - fall back to an ffmpeg decode
        wav, sr = _load_audio_via_ffmpeg(path, target_sr, torchaudio)

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)
    elif wav.size(0) > 2:
        wav = wav[:2]
    elif wav.size(0) == 0:
        raise RuntimeError(f"Decoded no audio channels from {path!r}.")

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    return wav, sr


def _export_m4a(
    audio_tensor: Any,
    sample_rate: int,
    out_path: str,
    torchaudio: Any,
) -> None:
    """Write ``audio_tensor`` to ``out_path`` as 128 kbps AAC in an M4A container.

    The intermediate WAV is always removed, including on failure, so long
    sessions do not leave gigabytes of scratch data behind.
    """
    temp_wav = f"{out_path}.tmp.wav"
    try:
        torchaudio.save(temp_wav, audio_tensor.cpu(), sample_rate)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                temp_wav,
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                out_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        _safe_remove(temp_wav)


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _require_dependencies() -> None:
    """Import the heavy stack, raising one clear error if anything is missing."""
    missing: list[str] = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        missing.append("torchaudio")
    try:
        import demucs  # noqa: F401
    except ImportError:
        missing.append("demucs")
    try:
        import whisperx  # noqa: F401
    except ImportError:
        missing.append("whisperx")
    try:
        import librosa  # noqa: F401
    except ImportError:
        missing.append("librosa")

    if missing:
        raise RuntimeError(
            "Missing audio processing dependencies: "
            + ", ".join(missing)
            + ". Install them with:\n"
            "  pip install torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "  pip install -r requirements.txt"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def process_song_session(
    input_m4a_path: str,
    output_dir: str = "./output",
    *,
    silence_threshold: float = SILENCE_THRESHOLD,
    start_bpm: float = START_BPM,
) -> dict[str, Any]:
    """Run the full pipeline over ``input_m4a_path`` and return the result dict.

    Args:
        input_m4a_path: Source recording (any format ffmpeg can read).
        output_dir: Directory the ``vocals.m4a`` / ``background.m4a`` stems and
            their temporary files are written to.
        silence_threshold: Seconds of pause that starts a new lyric line.
        start_bpm: Tempo prior passed to librosa's BPM estimator.

    Returns:
        A JSON-compatible dict matching the schema in ``SYSTEM_INSTRUCTIONS.md``.

    Raises:
        RuntimeError: if a heavy dependency or the input file is missing.
        subprocess.CalledProcessError: if ffmpeg fails.
    """
    _require_dependencies()

    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    import whisperx

    if not os.path.isfile(input_m4a_path):
        raise RuntimeError(f"Input audio not found: {input_m4a_path!r}")

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 1. Stem separation (Demucs)
    # ------------------------------------------------------------------
    demucs_model = _cached(
        ("demucs", "htdemucs", device),
        lambda: get_model("htdemucs").to(device).eval(),
    )
    sample_rate = getattr(demucs_model, "samplerate", DEFAULT_SAMPLE_RATE)

    wav, sample_rate = _load_audio(input_m4a_path, sample_rate, torchaudio)

    try:
        with torch.no_grad():
            # `split=True` (the default) chunks the mix internally, so an
            # hours-long session is processed segment by segment instead of
            # exhausting the 8 GB of VRAM in one shot.
            sources = apply_model(
                demucs_model,
                wav.unsqueeze(0),
                device=device,
                progress=False,
            )[0]  # [sources, channels, samples]

        background_tensor = sources[0] + sources[1] + sources[2]
        vocals_tensor = sources[VOCALS_INDEX]
    finally:
        _empty_cuda_cache(torch)

    vocals_m4a = os.path.join(output_dir, "vocals.m4a")
    background_m4a = os.path.join(output_dir, "background.m4a")
    _export_m4a(vocals_tensor, sample_rate, vocals_m4a, torchaudio)
    _export_m4a(background_tensor, sample_rate, background_m4a, torchaudio)

    # The stem tensors are no longer needed.
    del sources
    _empty_cuda_cache(torch)

    # ------------------------------------------------------------------
    # 2. Precision transcription & phoneme alignment (WhisperX)
    # ------------------------------------------------------------------
    compute_type = "float16" if device == "cuda" else "int8"
    wx_model = _cached(
        ("whisperx", "medium", device, compute_type),
        lambda: whisperx.load_model(
            "medium", device=device, compute_type=compute_type
        ),
    )
    audio_data = whisperx.load_audio(vocals_m4a)

    transcription = wx_model.transcribe(audio_data, batch_size=16)
    language = transcription.get("language") or "en"

    align_model, align_metadata = _cached(
        ("whisperx-align", language, device),
        lambda: whisperx.load_align_model(language_code=language, device=device),
    )
    aligned_result = whisperx.align(
        transcription["segments"],
        align_model,
        align_metadata,
        audio_data,
        device=device,
    )

    all_words: list[dict[str, Any]] = []
    for segment in aligned_result.get("segments", []):
        for word in segment.get("words", []):
            if "start" not in word or "end" not in word:
                continue
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            start = round(float(word["start"]), 3)
            end = round(float(word["end"]), 3)
            all_words.append(
                {
                    "word": text,
                    "start": start,
                    "end": end,
                    "length": round(end - start, 3),
                }
            )

    # Models stay cached for the next take; only transient activations are freed.
    _empty_cuda_cache(torch)

    # ------------------------------------------------------------------
    # 3. Group words into lines (pause threshold)
    # ------------------------------------------------------------------
    grouped_lines = group_words_into_lines(all_words, silence_threshold)

    # ------------------------------------------------------------------
    # 4. Per-line BPM calculation (librosa)
    # ------------------------------------------------------------------
    vocal_np = vocals_tensor.mean(dim=0).cpu().numpy()

    lyrics_output: list[dict[str, Any]] = []
    for line_words in grouped_lines:
        line_start = line_words[0]["start"]
        line_end = line_words[-1]["end"]

        start_sample = max(0, int(line_start * sample_rate))
        end_sample = max(start_sample, int(line_end * sample_rate))
        line_slice = vocal_np[start_sample:end_sample]

        lyrics_output.append(
            {
                "bpm": estimate_line_bpm(line_slice, sample_rate, start_bpm),
                "start time": line_start,
                "end time": line_end,
                "words": [
                    {"word": w["word"], "length": w["length"]} for w in line_words
                ],
            }
        )

    return {
        "vocals": vocals_m4a,
        "background": background_m4a,
        "lyrics": lyrics_output,
    }


def _empty_cuda_cache(torch: Any) -> None:
    """Release cached VRAM between stages; a no-op on CPU-only installs."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
