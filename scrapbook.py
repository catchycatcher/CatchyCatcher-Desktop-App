"""Scrapbook rendering: stitch isolated vocal lines into one tempo-matched mix.

A scrapbook is an ordered list of *lines* referenced from processed takes. The
mix has one absolute tempo, always auto-calculated as the median of the clips'
tempos. Each clip is then played at whichever **simple ratio** of that tempo is
closest to its own recorded tempo (1, 3/2, 2/3, 1/2, …), so a half/double-time
clip keeps its own time instead of being stretched to match the mix exactly, and
a clip that is close to the mix tempo is stretched only a little.

The gap between two lines is filled with the audio that actually followed and
preceded them in their own recordings (their ``gap_after`` / ``gap_before``
stretches), cross-faded together.

The module is deliberately free of FastAPI/state: callers pass in the resolved
line data and the output directory, which keeps the maths unit-testable without
a GPU or a real recording.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 44100
CHANNELS = 2
#: Gap used between lines when neither source pause is known.
DEFAULT_GAP_SECONDS = 0.35
MIN_GAP_SECONDS = 0.05
MAX_GAP_SECONDS = 3.0
#: FFmpeg's ``atempo`` accepts roughly 0.5-2.0 per instance; chain for extremes.
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0
#: How far a line may be sped up/slowed down from its recorded tempo.
MIN_SPEED = 0.5
MAX_SPEED = 2.0


def atempo_factors(rate: float) -> list[float]:
    """Break ``rate`` into a product of ``atempo`` factors in ``[0.5, 2.0]``."""
    if rate <= 0:
        return [1.0]
    factors: list[float] = []
    remaining = float(rate)
    while remaining < ATEMPO_MIN:
        factors.append(ATEMPO_MIN)
        remaining /= ATEMPO_MIN
    while remaining > ATEMPO_MAX:
        factors.append(ATEMPO_MAX)
        remaining /= ATEMPO_MAX
    factors.append(round(remaining, 6))
    return factors


def atempo_filter(rate: float) -> str:
    """An ``-af`` filter chain that changes tempo by ``rate`` without pitch shift."""
    return ",".join(f"atempo={factor}" for factor in atempo_factors(rate))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _source_words(entry: dict[str, Any], line_start: float) -> list[tuple[str, float, float]]:
    """Each word's ``(text, start, end)`` in the source timeline."""
    resolved: list[tuple[str, float, float]] = []
    cursor = line_start
    for word in entry.get("words") or []:
        start = word.get("start")
        end = word.get("end")
        start = float(start) if start is not None else cursor
        end = float(end) if end is not None else start + float(word.get("length") or 0.0)
        resolved.append((str(word.get("word", "")), start, end))
        cursor = end
    return resolved


#: Simple ratios a clip's tempo may take relative to the mix's absolute tempo.
#: Choosing the nearest means a half/double-time clip plays in its own time
#: instead of being stretched to hit the mix tempo exactly.
MIX_RATIOS = (0.5, 2.0 / 3.0, 0.75, 0.8, 1.0, 1.25, 4.0 / 3.0, 1.5, 2.0)

#: Fallback tempo when a clip has none.
DEFAULT_MIX_BPM = 120.0


def mix_bpm(entries: list[dict[str, Any]]) -> float:
    """The scrapbook's one absolute tempo: the median of its clips' tempos."""
    tempos = [
        float(entry.get("bpm") or entry.get("take_bpm") or DEFAULT_MIX_BPM) for entry in entries
    ]
    value = _median(tempos) if tempos else DEFAULT_MIX_BPM
    return value if value > 0 else DEFAULT_MIX_BPM


def _line_speeds(entries: list[dict[str, Any]]) -> list[float]:
    """Per-line playback speed so every clip plays at a ratio of the mix tempo.

    The mix has one absolute tempo (the median clip tempo, always auto). Each
    clip is played at whichever **simple ratio** of that tempo is closest to its
    own recorded tempo, so a half/double-time clip stays in its own time (ratio
    1/2 or 2) rather than being stretched to match the mix exactly. The speed is
    ``ratio * mix_bpm / clip_bpm``.
    """
    if not entries:
        return []
    reference = mix_bpm(entries)
    speeds: list[float] = []
    for entry in entries:
        bpm = float(entry.get("bpm") or entry.get("take_bpm") or DEFAULT_MIX_BPM)
        if bpm <= 0:
            speeds.append(1.0)
            continue
        natural = bpm / reference
        ratio = min(MIX_RATIOS, key=lambda value: abs(math.log(value) - math.log(natural)))
        speeds.append((ratio * reference) / bpm)
    return [max(MIN_SPEED, min(MAX_SPEED, speed)) for speed in speeds]


#: If the two candidate gaps differ by this factor (either way), keep the
#: earlier clip's pause instead of blending them.
GAP_DIVERGENCE = 1.5


def _gap_between(
    current: dict[str, Any], following: dict[str, Any], current_speed: float, next_speed: float
) -> float:
    """Choose the silence between two adjacent lines, scaled by their new speeds.

    The earlier line's trailing pause (``gap_after``) and the later line's
    leading pause (``gap_before``) are the candidates. When they are close they
    are averaged; when one is at least ``GAP_DIVERGENCE`` times the other, the
    **earlier line's** pause wins, so a big breath before or after a phrase is not
    diluted by the other side.
    """
    earlier = (
        float(current["gap_after"]) / current_speed
        if current.get("gap_after") is not None
        else None
    )
    later = (
        float(following["gap_before"]) / next_speed
        if following.get("gap_before") is not None
        else None
    )

    if earlier is None and later is None:
        gap = DEFAULT_GAP_SECONDS / current_speed
    elif earlier is None:
        gap = later
    elif later is None:
        gap = earlier
    elif max(earlier, later) >= GAP_DIVERGENCE * min(earlier, later):
        gap = earlier
    else:
        gap = (earlier + later) / 2

    return max(MIN_GAP_SECONDS, min(MAX_GAP_SECONDS, gap))


def _with_ratios(entries: list[dict[str, Any]]) -> None:
    """Fill in each entry's ``before``/``after`` tempo ratios (in place)."""
    for entry in entries:
        bpm = float(entry.get("bpm") or 120.0)
        next_bpm = entry.get("next_bpm")
        prev_bpm = entry.get("prev_bpm")
        entry["after"] = (float(next_bpm) / bpm) if next_bpm else None
        entry["before"] = (bpm / float(prev_bpm)) if prev_bpm else None


def build_timeline(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[float]]:
    """Lay every line out on the mix timeline at its chosen speed.

    Returns ``(timeline, gaps)`` where ``timeline`` is a list of schema-compatible
    lyric lines (``start time``/``end time``/``words``) with a ``speed``, and
    ``gaps`` holds the length of the gap inserted between each consecutive pair
    (filled with the source recordings' own gap audio when rendering).
    """
    _with_ratios(entries)
    speeds = _line_speeds(entries)

    timeline: list[dict[str, Any]] = []
    gaps: list[float] = []
    cursor = 0.0

    for index, entry in enumerate(entries):
        speed = speeds[index]
        source_bpm = float(entry.get("bpm") or 120.0)
        line_start = float(entry["start"])
        line_end = float(entry["end"])
        duration = max(0.0, (line_end - line_start) / speed)

        mix_start = cursor
        mix_end = cursor + duration

        words = [
            {
                "word": text,
                "start": round(mix_start + (word_start - line_start) / speed, 3),
                "end": round(mix_start + (word_end - line_start) / speed, 3),
                "length": round((word_end - word_start) / speed, 3),
            }
            for text, word_start, word_end in _source_words(entry, line_start)
        ]

        entry_line: dict[str, Any] = {
            "bpm": round(source_bpm * speed, 1),
            "start time": round(mix_start, 3),
            "end time": round(mix_end, 3),
            "speed": round(speed, 3),
            "words": words,
        }
        if entry.get("text"):
            entry_line["text"] = entry["text"]
        source = entry.get("source")
        if isinstance(source, dict):
            entry_line["source"] = source
        timeline.append(entry_line)
        cursor = mix_end

        if index < len(entries) - 1:
            gap = _gap_between(entry, entries[index + 1], speed, speeds[index + 1])
            gaps.append(round(gap, 3))
            cursor += gap

    return timeline, gaps


def extract_command(
    vocals: str | Path, start: float, duration: float, rate: float, out: Path
) -> list[str]:
    """FFmpeg command that cuts ``[start, start+duration]`` and applies ``rate``."""
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{max(duration, 0.0):.3f}",
        "-i",
        str(vocals),
        "-af",
        atempo_filter(rate),
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(out),
    ]


def silence_command(seconds: float, out: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
        "-t",
        f"{seconds:.3f}",
        "-c:a",
        "pcm_s16le",
        str(out),
    ]


def concat_command(list_file: Path, out: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out),
    ]


def _read_wav(path: str | Path) -> "np.ndarray":
    """Read a PCM WAV into a ``(frames, channels)`` float32 array."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        return samples.reshape(-1, channels)
    return samples.reshape(-1, 1)


def _write_wav(path: str | Path, samples: "np.ndarray", rate: int) -> None:
    clipped = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(clipped.shape[1] if clipped.ndim > 1 else 1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(clipped.tobytes())


def _pad_after(samples: "np.ndarray", target: int) -> "np.ndarray":
    length = samples.shape[0]
    if length == target:
        return samples
    if length > target:
        return samples[:target]
    pad = np.zeros((target - length, samples.shape[1]), dtype=np.float32)
    return np.concatenate([samples, pad], axis=0)


def _pad_before(samples: "np.ndarray", target: int) -> "np.ndarray":
    length = samples.shape[0]
    if length == target:
        return samples
    if length > target:
        return samples[-target:]
    pad = np.zeros((target - length, samples.shape[1]), dtype=np.float32)
    return np.concatenate([pad, samples], axis=0)


def _blend_gaps(
    earlier: "np.ndarray | None", later: "np.ndarray | None", target: int
) -> "np.ndarray":
    """Blend the tail of the earlier gap into the head of the later one.

    Both gaps are the *silent stretches* of their own recordings, so this keeps
    the breaths/room tone that actually followed and preceded the lines, fading
    them through each other instead of dropping in digital silence.
    """
    if earlier is None and later is None:
        return np.zeros((target, CHANNELS), dtype=np.float32)
    if earlier is None:
        return _pad_before(later, target)
    if later is None:
        return _pad_after(earlier, target)

    channels = earlier.shape[1]
    if earlier.shape[0] + later.shape[0] <= target:
        return _pad_after(np.concatenate([earlier, later], axis=0), target)

    overlap = earlier.shape[0] + later.shape[0] - target
    overlap = max(1, min(overlap, earlier.shape[0], later.shape[0]))
    fade = np.linspace(1.0, 0.0, overlap, dtype=np.float32).reshape(-1, 1)
    head = earlier[: earlier.shape[0] - overlap]
    tail = later[overlap:]
    middle = earlier[earlier.shape[0] - overlap :] * fade + later[:overlap] * (1.0 - fade)
    blended = np.concatenate([head, middle, tail], axis=0)
    if blended.shape[1] != channels:
        blended = blended[:, :CHANNELS]
    return _pad_after(blended, target)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render(
    entries: list[dict[str, Any]], output_dir: str | Path
) -> tuple[list[dict[str, Any]], list[float], Path]:
    """Render the whole scrapbook into ``<output_dir>/mix.m4a``.

    The gap between two lines is filled with the audio that actually followed
    the first line and preceded the second in their own recordings (their
    ``gap_after`` / ``gap_before`` stretches), cross-faded together — never
    running into the neighbouring line of either source take.

    Returns the timeline, the gaps, and the path to the written mix. Raises
    ``ValueError`` when there is nothing to render.
    """
    if not entries:
        raise ValueError("Cannot render a scrapbook with no lines.")

    _with_ratios(entries)
    speeds = _line_speeds(entries)
    timeline, gaps = build_timeline(entries)

    out_dir = Path(output_dir)
    work = out_dir / "render"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    for index, entry in enumerate(entries):
        speed = speeds[index]
        duration = max(0.0, float(entry["end"]) - float(entry["start"]))
        line_path = work / f"line_{index:04d}.wav"
        _run(extract_command(entry["vocals"], float(entry["start"]), duration, speed, line_path))
        parts.append(line_path)

        if index < len(gaps) and gaps[index] > 0.001:
            following = entries[index + 1]
            target = max(1, int(round(gaps[index] * SAMPLE_RATE)))

            earlier: "np.ndarray | None" = None
            gap_after = entry.get("gap_after")
            if gap_after is not None and float(gap_after) > 0.02:
                earlier_path = work / f"gap_{index:04d}_a.wav"
                _run(
                    extract_command(
                        entry["vocals"], float(entry["end"]), float(gap_after), speed, earlier_path
                    )
                )
                read = _read_wav(earlier_path)
                earlier = read if read.shape[0] else None

            later: "np.ndarray | None" = None
            gap_before = following.get("gap_before")
            if gap_before is not None and float(gap_before) > 0.02:
                later_path = work / f"gap_{index:04d}_b.wav"
                _run(
                    extract_command(
                        following["vocals"],
                        float(following["start"]) - float(gap_before),
                        float(gap_before),
                        speeds[index + 1],
                        later_path,
                    )
                )
                read = _read_wav(later_path)
                later = read if read.shape[0] else None

            gap_path = work / f"gap_{index:04d}.wav"
            if earlier is None and later is None:
                _run(silence_command(gaps[index], gap_path))
            else:
                _write_wav(gap_path, _blend_gaps(earlier, later, target), SAMPLE_RATE)
            parts.append(gap_path)

    list_file = work / "concat.txt"
    list_file.write_text("".join(f"file '{part.name}'\n" for part in parts), encoding="utf-8")

    mix = out_dir / "mix.m4a"
    _run(concat_command(list_file, mix))
    return timeline, gaps, mix
