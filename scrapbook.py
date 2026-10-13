"""Scrapbook rendering: stitch isolated vocal lines into one tempo-matched mix.

A scrapbook is an ordered list of *lines* referenced from processed takes. There
is no single BPM: each line keeps its own recorded tempo, and only the *relative*
tempo between neighbouring lines is adjusted.

For each line we know two expected ratios from its own journal — how its tempo
related to the line that followed it (``after``) and to the line that preceded it
(``before``) — and the tempo of the whole take it came from. When line B is
placed after line A, the *baseline* is the ratio of their source takes' tempos,
and the ratio applied to B is whichever of ``after``, ``before`` or **same BPM as
A** sits closest to that baseline. Once every line has a speed, they are
normalised so the median log-speed is zero — so half the lines speed up and half
slow down, with no overall shift.

The module is deliberately free of FastAPI/state: callers pass in the resolved
line data and the output directory, which keeps the maths unit-testable without
a GPU or a real recording.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

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


def _line_speeds(entries: list[dict[str, Any]]) -> list[float]:
    """Per-line playback speed factors (1.0 = played at its recorded tempo).

    The *baseline* for a pair of neighbours is the ratio of the tempos of the
    takes they were sourced from (so two clips from the same-tempo takes are
    expected to line up), rather than a scale of 1. The candidate ratios are the
    previous line's ``after`` expectation, this line's ``before`` expectation,
    and **same BPM as the previous line** (always available); the one closest to
    the baseline wins.

    ``speed > 1`` plays the line faster (shorter); the median log-speed is
    normalised to zero so there is no overall tempo shift.
    """
    speeds: list[float] = []
    played: float | None = None
    previous_take_bpm: float | None = None

    for index, entry in enumerate(entries):
        bpm = float(entry.get("bpm") or 120.0)
        take_bpm = float(entry.get("take_bpm") or bpm)

        if played is None:
            speeds.append(1.0)
            played = bpm
            previous_take_bpm = take_bpm
            continue

        # Baseline: how the two source takes relate in tempo.
        if previous_take_bpm:
            baseline = take_bpm / previous_take_bpm
        else:
            baseline = bpm / played

        # Same-BPM is always an option; the neighbours' expectations add more.
        candidates = [1.0]
        after = entries[index - 1].get("after")
        before = entry.get("before")
        if after:
            candidates.append(float(after))
        if before:
            candidates.append(float(before))

        chosen = min(candidates, key=lambda ratio: abs(math.log(ratio) - math.log(baseline)))
        played = played * chosen
        speeds.append(played / bpm)
        previous_take_bpm = take_bpm

    if speeds:
        logs = [math.log(speed) for speed in speeds]
        shift = _median(logs)
        speeds = [math.exp(value - shift) for value in logs]

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
    ``gaps`` holds the silence inserted between each consecutive pair.
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


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render(
    entries: list[dict[str, Any]], output_dir: str | Path
) -> tuple[list[dict[str, Any]], list[float], Path]:
    """Render the whole scrapbook into ``<output_dir>/mix.m4a``.

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
            gap_path = work / f"gap_{index:04d}.wav"
            _run(silence_command(gaps[index], gap_path))
            parts.append(gap_path)

    list_file = work / "concat.txt"
    list_file.write_text("".join(f"file '{part.name}'\n" for part in parts), encoding="utf-8")

    mix = out_dir / "mix.m4a"
    _run(concat_command(list_file, mix))
    return timeline, gaps, mix
