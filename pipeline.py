"""Catchy Catcher audio processing pipeline.

``process_song_session`` accepts a recorded ``.m4a`` and returns the
JSON-compatible payload described in ``SYSTEM_INSTRUCTIONS.md``::

    {
      "vocals": "path/to/vocals.m4a",
      "background": "path/to/background.m4a",
      "bpm": 118.0,
      "lyrics": [
        {
          "bpm": 120.0,
          "start time": 12.45,
          "end time": 15.82,
          "words": [
            {"word": "hello", "start": 12.45, "end": 12.82, "length": 0.37},
            {"word": "world", "start": 12.82, "end": 13.15, "length": 0.33}
          ]
        }
      ]
    }

Each word carries its exact ``start``/``end`` (seconds) as well as ``length``,
so a client can highlight words as the audio plays; the spec's minimum fields
are unchanged. The top-level ``bpm`` is the tempo of the whole take (each lyric
line also carries its own ``bpm``).

The heavy dependencies (torch, torchaudio, demucs, whisperx, librosa) are
imported lazily so the API can start and report health on a machine where they
are not installed. ``process_song_session`` raises a ``RuntimeError`` with the
install instructions when a dependency is missing.

Pipeline stages:

1. Stem separation with Demucs (``htdemucs``) on CUDA when available.
2. Precision transcription + forced phoneme alignment with WhisperX.
3. Word grouping into lyric lines with a four-stage break model: seed each gap
   from pause size and punctuation, then a rhyme pass, a line-length pass and a
   rolling threshold decide where the lines actually break.
4. Per-line BPM estimation with librosa.
"""

from __future__ import annotations

import math
import os
import statistics
import subprocess
import tempfile
import threading
from typing import Any, Callable, Iterable

import numpy as np

import phonetics

# ---------------------------------------------------------------------------
# Tunables (see "Implementation Guidelines" in SYSTEM_INSTRUCTIONS.md)
# ---------------------------------------------------------------------------

#: Pause in seconds between two words that *always* starts a new lyric line.
SILENCE_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# Line-grouping model (see ``group_words_into_lines``)
# ---------------------------------------------------------------------------
#
# A word timeline has no line breaks in it: every boundary between two words is
# just a gap. Grouping them happens in four stages, each one writing a break
# *likelihood* (0..1) onto that boundary:
#
#   1. Seed    each gap gets a likelihood from how long the pause is, how it
#              compares to the singer's other pauses, and the sentence/clause
#              punctuation on the word before it (plus a nudge when the next
#              word opens a new sentence):
#
#                score = W_GAP   * (gap / silence_threshold)
#                      + gate * ( W_REL   * (gap / local_median_gap - 1)
#                               + W_PUNCT * punctuation_weight
#                               + W_NEXT  * next_sentence )
#                      - BIAS
#                likelihood = sigmoid(SHARPNESS * score)
#
#              ``gate`` fades the softer signals out below ``MIN_BREAK_GAP`` so
#              timing wobble cannot split a line, and a gap at the silence
#              threshold scores well above 0.5 on its own.
#
#   2. Rhyme   gaps whose word near-rhymes with the word at another *nearby*
#              likely break are pushed up together, so a rhyme lands on a line
#              ending instead of in the middle of one.
#
#   3. Length  gaps are nudged so the line they would close sits close to the
#              length of the lines around them.
#
#   4. Rolling the boundaries are finally committed left to right. A gap at or
#              over ``silence_threshold`` always breaks, a line that outgrows
#              its neighbours is force-broken, a line long enough to end on a
#              comma or full stop breaks even with no pause, and otherwise a gap
#              breaks when it clears a threshold that drifts to keep recent
#              lines near the song's typical line length. This makes the breaks
#              land at a sensible frequency.
#
# ``group_words_into_lines`` runs the rolling pass twice: once from the raw
# seeds to learn the typical line length, then again after stages 2-3.

#: Weight of the absolute pause (``gap / silence_threshold``).
GAP_WEIGHT = 1.0
#: Weight of "this pause is long compared to the singer's nearby gaps".
RELATIVE_GAP_WEIGHT = 0.3
#: Weight of sentence/clause punctuation on the word before the gap.
PUNCTUATION_WEIGHT = 0.5
#: Weight of "the next word opens a new sentence" (a capitalised word).
NEXT_SENTENCE_WEIGHT = 0.15
#: Subtracted from the score before the logistic squashes it into 0..1.
BREAK_BIAS = 0.9
#: Scales the score around zero: larger = more decisive likelihoods.
PROBABILITY_SHARPNESS = 4.0
#: Caps that keep individual terms from dominating the score.
MAX_GAP_RATIO = 3.0
MAX_RELATIVE_RATIO = 2.5
#: A pause shorter than this can never be broken by the softer signals alone
#: (relative pause, punctuation, line length). Below it, those signals are
#: scaled down toward nothing, so tiny timing wobbles do not split a line.
MIN_BREAK_GAP = 0.3
#: A line ending on sentence/clause punctuation may still break with almost no
#: pause once it has this many words -- a phrase can end on a comma or full
#: stop the singer runs straight through. Tiny clauses (``tell me,``) stay put.
PUNCT_BREAK_MIN_WORDS = 3

# --- Rhyme scheme ----------------------------------------------------------
#: Likelihood a gap needs before it counts as a candidate line ending for the
#: rhyme pass. Set at the base break threshold so a rhyme only reinforces
#: endings that are already plausible, rather than inventing new ones.
RHYME_CANDIDATE_PROBABILITY = 0.5
#: How many candidate breaks either side of one are inspected for a rhyme.
RHYME_WINDOW = 4
#: Extra likelihood a rhyming pair of endings gets, scaled by how near the
#: rhyme is.
RHYME_BONUS = 0.35
#: A pair of line endings scoring below this is not treated as a rhyme.
RHYME_MIN_SIMILARITY = 0.5

# --- Line-length consistency ----------------------------------------------
#: Likelihood nudge applied to gaps that would close an over/under-long line.
LENGTH_BONUS = 0.35
#: A line may reach this multiple of its neighbours before its internal gaps
#: are pushed up.
LENGTH_TOLERANCE = 1.15
#: A line under this multiple of its neighbours has its gaps pushed down.
LENGTH_SHORT_SLACK = 0.8

# --- Rolling selection -----------------------------------------------------
#: Likelihood a gap must clear before it becomes a line break.
BASE_BREAK_THRESHOLD = 0.5
#: The rolling threshold never drifts outside these bounds.
THRESHOLD_MIN = 0.35
THRESHOLD_MAX = 0.8
#: A line is force-broken once it reaches this multiple of the rolling target,
#: even when there is no pause or punctuation to break on.
FORCE_LENGTH_MULTIPLIER = 1.8
#: ... but never force-break a line shorter than this many words.
MIN_FORCE_WORDS = 3

#: Punctuation that very strongly suggests the line ends here.
SENTENCE_END_PUNCTUATION = ".!?…"
#: Punctuation that suggests a clause boundary (weaker than a full stop).
CLAUSE_PUNCTUATION = ",;:"
#: How many neighbouring gaps a gap is compared against.
GAP_WINDOW = 3
#: How many recently finished lines the rolling target is drawn from.
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


def _sigmoid(value: float) -> float:
    """Logistic squash into ``(0, 1)`` without overflowing on big scores."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _next_sentence_weight(words: list[dict[str, Any]], index: int) -> float:
    """1.0 when the word after the gap opens a new sentence, else 0.0.

    WhisperX keeps the case of the source, so a capitalised next word is a
    decent hint that a sentence just ended.
    """
    if index + 1 >= len(words):
        return 0.0
    following = str(words[index + 1].get("word", "")).strip()
    return 1.0 if following[:1].isupper() else 0.0


def _seed_break_probabilities(
    words: list[dict[str, Any]],
    gaps: list[float],
    silence_threshold: float,
) -> list[float]:
    """Stage 1: one break likelihood per gap from pause size and structure.

    The absolute pause, its size relative to the singer's nearby gaps, the
    punctuation on the word before the gap and a new-sentence hint all fold
    into one score, which the logistic turns into a 0..1 likelihood.
    """
    likelihoods: list[float] = []
    for index, gap in enumerate(gaps):
        x_gap = 0.0
        if silence_threshold > 0:
            x_gap = _clamp(gap / silence_threshold, 0.0, MAX_GAP_RATIO)

        scale = _local_gap_scale(gaps, index)
        x_relative = 0.0
        if scale > 0:
            x_relative = max(0.0, _clamp(gap / scale, 0.0, MAX_RELATIVE_RATIO) - 1.0)

        structure = (
            RELATIVE_GAP_WEIGHT * x_relative
            + PUNCTUATION_WEIGHT * _punct_weight(str(words[index].get("word", "")))
            + NEXT_SENTENCE_WEIGHT * _next_sentence_weight(words, index)
        )
        # Softer signals only count once there is a real pause to react to.
        if MIN_BREAK_GAP > 0:
            structure *= _clamp(gap / MIN_BREAK_GAP, 0.0, 1.0)

        score = GAP_WEIGHT * x_gap + structure - BREAK_BIAS
        likelihoods.append(_sigmoid(PROBABILITY_SHARPNESS * score))
    return likelihoods


def _near_rhyme_similarity(first: str, second: str) -> float:
    """How closely two words rhyme, 0..1 (pronunciation-dictionary backed).

    A perfect rhyme scores 1.0; sharing the rime vowel but not the ending
    (assonance) scores lower. See :mod:`phonetics` for the fallback used when
    the CMU dictionary is not installed.
    """
    return phonetics.rhyme_similarity(first, second)


def _apply_rhyme_scheme(
    words: list[dict[str, Any]],
    likelihoods: list[float],
) -> list[float]:
    """Stage 2: let a rhyme pull two nearby likely endings up together."""
    adjusted = list(likelihoods)
    candidates = [
        index
        for index, value in enumerate(likelihoods)
        if value >= RHYME_CANDIDATE_PROBABILITY
    ]
    for position, first in enumerate(candidates):
        neighbours = candidates[position + 1 : position + 1 + RHYME_WINDOW]
        for second in neighbours:
            similarity = _near_rhyme_similarity(
                str(words[first].get("word", "")),
                str(words[second].get("word", "")),
            )
            if similarity < RHYME_MIN_SIMILARITY:
                continue
            boost = RHYME_BONUS * similarity
            adjusted[first] = min(1.0, adjusted[first] + boost)
            adjusted[second] = min(1.0, adjusted[second] + boost)
    return adjusted


def _apply_line_length_consistency(
    words: list[dict[str, Any]],
    gaps: list[float],
    likelihoods: list[float],
    target_seconds: float | None,
) -> list[float]:
    """Stage 3: nudge gaps to close a line the size of the nearby ones.

    The gaps already likely to break anchor the lines; a boundary that could
    break is pushed up when the line it would close is too long and pushed down
    when the line is too short, so line sizes drift toward the local target.
    A weak seam is left alone: this stage only reinforces real candidates.
    """
    if not target_seconds or target_seconds <= 0:
        return list(likelihoods)

    adjusted = list(likelihoods)
    previous_anchor = -1
    for index, value in enumerate(likelihoods):
        if value < BASE_BREAK_THRESHOLD and index != len(likelihoods) - 1:
            continue
        line_start = previous_anchor + 1
        for gap_index in range(previous_anchor + 1, index + 1):
            seconds = float(words[gap_index]["end"]) - float(words[line_start]["start"])
            ratio = seconds / target_seconds
            # A gap with no real pause in it is never a good break, so the nudge
            # is faded out below ``MIN_BREAK_GAP`` just like the seed signals.
            viability = (
                _clamp(float(gaps[gap_index]) / MIN_BREAK_GAP, 0.0, 1.0)
                if MIN_BREAK_GAP > 0
                else 1.0
            )
            if ratio >= LENGTH_TOLERANCE and adjusted[gap_index] >= BASE_BREAK_THRESHOLD:
                # Only reinforce a boundary that could already break: the nudge
                # must not turn a weak seam into a new line all by itself.
                adjusted[gap_index] = min(
                    1.0,
                    adjusted[gap_index] + LENGTH_BONUS * _clamp(ratio - 1.0, 0.0, 1.0) * viability,
                )
            elif ratio <= LENGTH_SHORT_SLACK:
                adjusted[gap_index] = max(
                    0.0,
                    adjusted[gap_index] - LENGTH_BONUS * _clamp(1.0 - ratio, 0.0, 1.0) * viability,
                )
        previous_anchor = index
    return adjusted


def _rolling_threshold(recent: list[float], target: float | None) -> float:
    """The break threshold for the next line, drifting around the base value.

    Recent lines longer than the target push it down (so the singer breaks more
    often); lines shorter than the target push it up. The break frequency thus
    self-corrects toward the song's typical line length.
    """
    if not target or target <= 0:
        return BASE_BREAK_THRESHOLD
    usable = [length for length in recent if length > 0]
    if not usable:
        return BASE_BREAK_THRESHOLD
    median = statistics.median(usable)
    if median <= 0:
        return BASE_BREAK_THRESHOLD
    return _clamp(
        BASE_BREAK_THRESHOLD * (target / median), THRESHOLD_MIN, THRESHOLD_MAX
    )


def _best_internal_boundary(
    words: list[dict[str, Any]], gaps: list[float], start: int, index: int
) -> int:
    """The strongest boundary in ``[start, index)`` for a forced cap split.

    Biggest pause first, then punctuation strength, then the boundary nearest
    the middle of the line (so a uniform run splits evenly).
    """
    middle = (start + index) / 2
    return max(
        range(start, index),
        key=lambda j: (
            round(gaps[j], 3),
            _punct_weight(str(words[j].get("word", ""))),
            -abs(j - middle),
        ),
    )


def _best_likely_boundary(
    words: list[dict[str, Any]],
    gaps: list[float],
    likelihoods: list[float],
    start: int,
    index: int,
) -> int:
    """The best boundary in ``[start, index]`` to force-break an over-long line.

    Highest break likelihood first, then the pause size, punctuation and how
    close it sits to the middle of the line, so an over-long phrase breaks at
    its strongest natural seam instead of an arbitrary word.
    """
    middle = (start + index) / 2
    return max(
        range(start, index + 1),
        key=lambda j: (
            round(likelihoods[j], 3),
            round(gaps[j], 3),
            _punct_weight(str(words[j].get("word", ""))),
            -abs(j - middle),
        ),
    )


def _select_lines(
    words: list[dict[str, Any]],
    gaps: list[float],
    likelihoods: list[float],
    silence_threshold: float,
    target_seconds: float | None,
    max_line_seconds: float,
    max_line_words: int,
) -> list[list[dict[str, Any]]]:
    """Stage 4: commit breaks left to right with a rolling, adaptive threshold.

    A gap at or over ``silence_threshold`` always breaks, as does a line that
    reaches the hard ceiling or outgrows the rolling target. Otherwise the gap
    must clear a likelihood threshold that drifts to keep the break frequency
    near the song's typical line length.
    """
    lines: list[list[dict[str, Any]]] = []
    recent: list[float] = []
    start = 0
    index = 0
    threshold = BASE_BREAK_THRESHOLD

    while index < len(gaps):
        count = index - start + 1
        seconds = float(words[index]["end"]) - float(words[start]["start"])

        # Hard cap: cut at the best boundary inside the line, then carry on.
        if count >= max_line_words or seconds >= max_line_seconds:
            cut = _best_internal_boundary(words, gaps, start, index) if index > start else index
            lines.append(words[start : cut + 1])
            recent.append(float(words[cut]["end"]) - float(words[start]["start"]))
            if len(recent) > LINE_WINDOW:
                recent.pop(0)
            start = cut + 1
            index = cut + 1
            threshold = _rolling_threshold(recent, target_seconds)
            continue

        # "Nearby lines" = the lines we just finished (they tend to be similar).
        # They may shorten the reference (break sooner), but never below the
        # song's typical line, so a normal line after a run of short ones is not
        # chopped up. Without a learned target (the first pass) nothing is
        # force-broken: that pass only measures where the natural breaks fall.
        rolling_target = statistics.median(recent) if recent else target_seconds
        force_reference = 0.0
        if target_seconds:
            force_reference = max(rolling_target or target_seconds, target_seconds)
        over_long = bool(
            force_reference
            and seconds >= FORCE_LENGTH_MULTIPLIER * force_reference
            and count >= MIN_FORCE_WORDS
        )

        # A phrase can end on a comma or full stop with almost no pause, as long
        # as the line has some words in it already.
        phrase_break = (
            count >= PUNCT_BREAK_MIN_WORDS
            and _punct_weight(str(words[index].get("word", ""))) > 0.0
        )

        if (
            gaps[index] >= silence_threshold
            or phrase_break
            or likelihoods[index] >= threshold
        ):
            # A natural break: a clear pause or a likely-enough boundary.
            lines.append(words[start : index + 1])
            recent.append(seconds)
            if len(recent) > LINE_WINDOW:
                recent.pop(0)
            start = index + 1
            threshold = _rolling_threshold(recent, target_seconds)
        elif over_long:
            # No seam here, but the line has grown too long: break it at its
            # strongest internal boundary so consistent line sizes are kept.
            cut = _best_likely_boundary(words, gaps, likelihoods, start, index)
            lines.append(words[start : cut + 1])
            recent.append(float(words[cut]["end"]) - float(words[start]["start"]))
            if len(recent) > LINE_WINDOW:
                recent.pop(0)
            start = cut + 1
            index = cut + 1
            threshold = _rolling_threshold(recent, target_seconds)
            continue

        index += 1

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
    """Split aligned words into lyric lines with a four-stage break model.

    Every gap first gets a seed break likelihood from the pause size and the
    word's punctuation; a rhyme pass, a line-length pass and a rolling threshold
    pass then adjust and commit those breaks. Lines never exceed
    ``max_line_seconds`` / ``max_line_words``.

    Pure Python and dependency-free so it can be unit tested without a GPU.
    """
    words = list(words)
    if not words:
        return []
    if len(words) == 1:
        return [words]

    gaps = _word_gaps(words)

    # First pass: raw seeds only, to learn how long a typical line is here.
    seeded = _seed_break_probabilities(words, gaps, silence_threshold)
    first_pass = _select_lines(
        words, gaps, seeded, silence_threshold, None, max_line_seconds, max_line_words
    )
    target_seconds = _median_line_seconds(first_pass)

    # Second pass: rhyme + line-length adjustment, then the rolling threshold.
    adjusted = _apply_rhyme_scheme(words, seeded)
    adjusted = _apply_line_length_consistency(words, gaps, adjusted, target_seconds)
    return _select_lines(
        words,
        gaps,
        adjusted,
        silence_threshold,
        target_seconds,
        max_line_seconds,
        max_line_words,
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


def load_vocal_mono(
    path: str, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> tuple["np.ndarray", int]:
    """Load a vocals file as a mono numpy signal (for BPM analysis).

    Raises ``RuntimeError`` (via :func:`_require_dependencies`) when the audio
    stack is missing, so the API can report it instead of crashing.
    """
    import torchaudio

    _require_dependencies()
    wav, sr = _load_audio(path, sample_rate, torchaudio)
    return wav.mean(dim=0).cpu().numpy(), sr


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
    # 3. Group words into lines (four-stage break model)
    # ------------------------------------------------------------------
    grouped_lines = group_words_into_lines(all_words, silence_threshold)

    # ------------------------------------------------------------------
    # 4. BPM: once for the whole take, and once per line (librosa)
    # ------------------------------------------------------------------
    vocal_np = vocals_tensor.mean(dim=0).cpu().numpy()
    take_bpm = estimate_line_bpm(vocal_np, sample_rate, start_bpm)

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
                    {
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"],
                        "length": w["length"],
                    }
                    for w in line_words
                ],
            }
        )

    return {
        "vocals": vocals_m4a,
        "background": background_m4a,
        "bpm": take_bpm,
        "lyrics": lyrics_output,
    }


def _empty_cuda_cache(torch: Any) -> None:
    """Release cached VRAM between stages; a no-op on CPU-only installs."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
