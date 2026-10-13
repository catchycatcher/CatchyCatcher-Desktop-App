"""Scrapbook rendering: stitch isolated vocal lines into one tempo-matched mix.

A scrapbook is an ordered list of *lines* referenced from processed takes. Every
line is time-stretched to the scrapbook's single BPM with FFmpeg's pitch-
preserving ``atempo`` filter, the gaps between lines are rebuilt from the pauses
that surrounded each line in its own journal (scaled by the tempo change) and
blended, and the whole thing is concatenated into one audio file the client can
play with word highlighting.

The module is deliberately free of FastAPI/state: callers pass in the resolved
line data and the output directory, which keeps the timeline maths unit-testable
without a GPU or a real recording.
"""

from __future__ import annotations

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


def _gap_between(current: dict[str, Any], following: dict[str, Any], target: float) -> float:
    """Blend the pauses around two adjacent lines, scaled by the tempo change.

    The pause after ``current`` (in its own take) is scaled by *its* tempo
    change and the pause before ``following`` by *its* tempo change, then the two
    are averaged, so a gap that came from a slow line is not carried over as-is
    into a fast scrapbook (and vice versa).
    """
    current_scale = float(current.get("bpm") or target) / target
    following_scale = float(following.get("bpm") or target) / target

    gaps: list[float] = []
    if current.get("gap_after") is not None:
        gaps.append(float(current["gap_after"]) * current_scale)
    if following.get("gap_before") is not None:
        gaps.append(float(following["gap_before"]) * following_scale)

    gap = sum(gaps) / len(gaps) if gaps else DEFAULT_GAP_SECONDS * current_scale
    return max(MIN_GAP_SECONDS, min(MAX_GAP_SECONDS, gap))


def build_timeline(
    entries: list[dict[str, Any]], target_bpm: float
) -> tuple[list[dict[str, Any]], list[float]]:
    """Lay every line out on the mix timeline.

    Returns ``(timeline, gaps)`` where ``timeline`` is a list of schema-compatible
    lyric lines (``start time``/``end time``/``words`` with ``start``/``end``)
    placed at their position in the mix, and ``gaps`` holds the silence inserted
    between each consecutive pair.
    """
    target = float(target_bpm) if target_bpm and target_bpm > 0 else 120.0
    timeline: list[dict[str, Any]] = []
    gaps: list[float] = []
    cursor = 0.0

    for index, entry in enumerate(entries):
        source_bpm = float(entry.get("bpm") or target)
        scale = source_bpm / target  # duration multiplier; atempo = 1 / scale
        line_start = float(entry["start"])
        line_end = float(entry["end"])
        duration = max(0.0, (line_end - line_start) * scale)

        mix_start = cursor
        mix_end = cursor + duration

        words = [
            {
                "word": text,
                "start": round(mix_start + (word_start - line_start) * scale, 3),
                "end": round(mix_start + (word_end - line_start) * scale, 3),
                "length": round((word_end - word_start) * scale, 3),
            }
            for text, word_start, word_end in _source_words(entry, line_start)
        ]

        timeline.append(
            {
                "bpm": round(target, 1),
                "start time": round(mix_start, 3),
                "end time": round(mix_end, 3),
                "source_bpm": round(source_bpm, 1),
                "words": words,
            }
        )
        cursor = mix_end

        if index < len(entries) - 1:
            gap = _gap_between(entry, entries[index + 1], target)
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
    entries: list[dict[str, Any]], target_bpm: float, output_dir: str | Path
) -> tuple[list[dict[str, Any]], list[float], Path]:
    """Render the whole scrapbook into ``<output_dir>/mix.m4a``.

    Returns the timeline, the gaps, and the path to the written mix. Raises
    ``ValueError`` when there is nothing to render.
    """
    if not entries:
        raise ValueError("Cannot render a scrapbook with no lines.")

    target = float(target_bpm) if target_bpm and target_bpm > 0 else 120.0
    timeline, gaps = build_timeline(entries, target)

    out_dir = Path(output_dir)
    work = out_dir / "render"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    for index, entry in enumerate(entries):
        source_bpm = float(entry.get("bpm") or target)
        rate = target / source_bpm if source_bpm > 0 else 1.0
        duration = max(0.0, float(entry["end"]) - float(entry["start"]))
        line_path = work / f"line_{index:04d}.wav"
        _run(extract_command(entry["vocals"], float(entry["start"]), duration, rate, line_path))
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
