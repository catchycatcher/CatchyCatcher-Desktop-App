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
3. Word grouping into lyric lines from a combined break score: pause size,
   relative pause, sentence/clause punctuation and nearby line length.
4. Per-line BPM estimation with librosa.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import tempfile
import threading
from typing import Any, Callable, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Tunables (see "Implementation Guidelines" in SYSTEM_INSTRUCTIONS.md)
# ---------------------------------------------------------------------------

#: Pause in seconds between two words that *always* starts a new lyric line.
SILENCE_THRESHOLD = 0.4

# --- Line-grouping model (see ``group_words_into_lines``) ------------------
#
# Whether a boundary is a line break is a *score*, not a single condition. Each
# signal contributes its own amount and they all add up; the line only breaks
# when the total clears zero. This lets a comma tip an otherwise ordinary pause
# over the line, and lets a long line be broken even without punctuation.
#
#   score = W_GAP   * (gap / silence_threshold)      # a real pause
#         + W_REL   * (gap / local_median_gap - 1)   # long for this singer
#         + W_PUNCT * punctuation_weight             # . ! ? vs , ; :
#         + W_LEN   * (line_seconds / nearby - 1)    # long for nearby lines
#         - BIAS
#
# ``group_words_into_lines`` runs twice: once to learn the typical line length,
# then again with the nearby-line signal enabled.

#: Weight of the absolute pause (``gap / silence_threshold``); 1.0 means a gap
#: exactly at the threshold contributes that whole weight.
GAP_WEIGHT = 1.0
#: Weight of "this pause is long compared to the surrounding words".
RELATIVE_GAP_WEIGHT = 0.8
#: Weight of sentence/clause punctuation.
PUNCTUATION_WEIGHT = 0.55
#: Weight of "this line is long compared to nearby lines".
LINE_LENGTH_WEIGHT = 0.9
#: Subtracted from the total; the boundary breaks when the score is >= 0.
BREAK_BIAS = 0.65
#: Caps that keep individual terms from dominating the score.
MAX_GAP_RATIO = 3.0
MAX_RELATIVE_RATIO = 4.0

#: Punctuation that very strongly suggests the line ends here.
SENTENCE_END_PUNCTUATION = ".!?…"
#: Punctuation that suggests a clause boundary (weaker than a full stop).
CLAUSE_PUNCTUATION = ",;:"
#: How many neighbouring gaps a gap is compared against.
GAP_WINDOW = 3
#: How many recently finished lines the nearby-length signal compares against.
LINE_WINDOW = 4
#: Hard ceilings; a line never grows past these.
MAX_LINE_SECONDS = 14.0
MAX_LINE_WORDS = 16
#: Closers ignored when looking for a word's trailing punctuation.
TRAILING_CLOSERS = "\"'”’)]}"


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


def _trailing_punctuation(word: str) -> str:
    """The word's final punctuation mark, ignoring quotes/brackets, or ``""``."""
    text = str(word).strip()
    while text and text[-1] in TRAILING_CLOSERS:
        text = text[:-1]
    if text and text[-1] in SENTENCE_END_PUNCTUATION + CLAUSE_PUNCTUATION:
        return text[-1]
    return ""


def _punct_weight(word: str) -> float:
    """How strongly a word's punctuation suggests the line ends: 1, 0.5 or 0."""
    mark = _trailing_punctuation(word)
    if not mark:
        return 0.0
    if mark in SENTENCE_END_PUNCTUATION:
        return 1.0
    if mark in CLAUSE_PUNCTUATION:
        return 0.5
    return 0.0


def _word_gaps(words: list[dict[str, Any]]) -> list[float]:
    """Non-negative pause between each consecutive pair of words."""
    gaps: list[float] = []
    for index in range(len(words) - 1):
        gap = float(words[index + 1]["start"]) - float(words[index]["end"])
        gaps.append(max(0.0, gap))
    return gaps


def _local_gap_scale(gaps: list[float], index: int, window: int = GAP_WINDOW) -> float:
    """Median of the *other* nearby gaps (how fast this singer usually moves).

    Returns 0.0 when there is no usable comparison, so the caller can treat the
    relative term as neutral instead of penalising the boundary.
    """
    low = max(0, index - window)
    high = min(len(gaps), index + window + 1)
    neighbours = [gaps[j] for j in range(low, high) if j != index and gaps[j] > 0]
    if not neighbours:
        neighbours = [gap for gap in gaps if gap > 0]
    if not neighbours:
        return 0.0
    return statistics.median(neighbours)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _break_score(
    words: list[dict[str, Any]],
    gaps: list[float],
    index: int,
    line_start: int,
    silence_threshold: float,
    nearby_seconds: float | None,
) -> float:
    """Total likelihood (>= 0 means "break here") for the boundary at ``index``.

    Every signal is folded in: the absolute pause, how the pause compares to the
    singer's nearby gaps, punctuation, and how long the line has grown compared
    to the lines around it.
    """
    gap = gaps[index]

    x_gap = 0.0
    if silence_threshold > 0:
        x_gap = _clamp(gap / silence_threshold, 0.0, MAX_GAP_RATIO)

    scale = _local_gap_scale(gaps, index)
    x_relative = 0.0
    if scale > 0:
        x_relative = _clamp(gap / scale, 0.0, MAX_RELATIVE_RATIO) - 1.0

    score = (
        GAP_WEIGHT * x_gap
        + RELATIVE_GAP_WEIGHT * x_relative
        + PUNCTUATION_WEIGHT * _punct_weight(str(words[index].get("word", "")))
    )

    if nearby_seconds and nearby_seconds > 0:
        line_seconds = float(words[index]["end"]) - float(words[line_start]["start"])
        score += LINE_LENGTH_WEIGHT * max(0.0, line_seconds / nearby_seconds - 1.0)

    return score - BREAK_BIAS


def _segment_lines(
    words: list[dict[str, Any]],
    gaps: list[float],
    silence_threshold: float,
    nearby_seconds: float | None,
    max_line_seconds: float,
    max_line_words: int,
) -> list[list[dict[str, Any]]]:
    """Greedily cut ``words`` into lines, tracking recent line lengths."""
    lines: list[list[dict[str, Any]]] = []
    recent_lengths: list[float] = []
    start = 0

    for index in range(len(gaps)):
        count = index - start + 1
        seconds = float(words[index]["end"]) - float(words[start]["start"])
        over_cap = count >= max_line_words or seconds >= max_line_seconds

        # "Nearby lines" = the lines we just finished (they tend to be similar).
        target = statistics.median(recent_lengths) if recent_lengths else nearby_seconds
        score = _break_score(words, gaps, index, start, silence_threshold, target)

        if over_cap or score >= 0.0:
            lines.append(words[start : index + 1])
            recent_lengths.append(float(words[index]["end"]) - float(words[start]["start"]))
            if len(recent_lengths) > LINE_WINDOW:
                recent_lengths.pop(0)
            start = index + 1

    lines.append(words[start:])
    return lines


def _median_line_seconds(lines: list[list[dict[str, Any]]]) -> float | None:
    durations = [
        float(line[-1]["end"]) - float(line[0]["start"]) for line in lines if line
    ]
    usable = [duration for duration in durations if duration > 0]
    if not usable:
        return None
    return statistics.median(usable)


def group_words_into_lines(
    words: Iterable[dict[str, Any]],
    silence_threshold: float = SILENCE_THRESHOLD,
    *,
    max_line_seconds: float = MAX_LINE_SECONDS,
    max_line_words: int = MAX_LINE_WORDS,
) -> list[list[dict[str, Any]]]:
    """Split aligned words into lyric lines from a weighted break score.

    Rather than any single trigger, every signal is combined into one score for
    each word boundary — the size of the pause, how that pause compares to the
    singer's surrounding gaps, whether the word ends a sentence or clause, and
    whether the line has grown longer than the lines around it. The boundary
    breaks when the combined score clears zero, so a comma or a full stop can
    tip an otherwise ordinary pause into a break, and a long line can break with
    no punctuation at all.

    Two passes are used: the first learns the typical line length, the second
    adds the nearby-line signal. Lines never exceed ``max_line_seconds`` /
    ``max_line_words``.

    Pure Python and dependency-free so it can be unit tested without a GPU.
    """
    words = list(words)
    if not words:
        return []
    if len(words) == 1:
        return [words]

    gaps = _word_gaps(words)

    first_pass = _segment_lines(
        words, gaps, silence_threshold, None, max_line_seconds, max_line_words
    )
    nearby_seconds = _median_line_seconds(first_pass)

    return _segment_lines(
        words, gaps, silence_threshold, nearby_seconds, max_line_seconds, max_line_words
    )


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
