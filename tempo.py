"""Metrical-grid tempo detection and syllable placement.

Beats are hard to detect on a bare voice, so instead of trusting librosa's
tempo on a short slice we fit a musical grid to the **word onsets**: pick the
beat period and phase that put the words as close as possible to grid lines.

The main beat is free, but landing on a subdivision costs a little — eighths
least, then triplets, then sixteenths — so the fit prefers words on the beat
and only calls something an eighth/sixteenth/triplet when the timing really
says so.

The same grid then places the syllables of a word (so an edited line always has
"a good place" for every syllable) and re-times lyric edits.
"""

from __future__ import annotations

import math

import numpy as np

#: Tempo search range.
MIN_BPM = 50.0
MAX_BPM = 200.0

#: ``(offset within a beat, penalty)``. The main beat is free; subdivisions cost
#: a little, rising with how weak the subdivision is (eighth < triplet <
#: sixteenth), which is the "smaller weight for rational ratios".
SUBDIVISIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.5, 0.10),  # eighth note
    (1.0 / 3.0, 0.16),
    (2.0 / 3.0, 0.16),  # triplets
    (0.25, 0.24),
    (0.75, 0.24),  # sixteenth notes
)

_PERIOD_STEPS = 160
_PHASE_STEPS = 48


def _beat_cost(times: "np.ndarray", period: float, phase: float, subdivisions) -> float:
    """Mean distance (in beats², plus subdivision penalty) to the nearest line."""
    best: "np.ndarray | None" = None
    for offset, penalty in subdivisions:
        index = np.round((times - phase) / period - offset)
        position = phase + (index + offset) * period
        gap = (times - position) / period
        cost = penalty + gap * gap
        best = cost if best is None else np.minimum(best, cost)
    return float(best.mean()) if best is not None else 0.0


def fit_grid(
    onsets,
    *,
    min_bpm: float = MIN_BPM,
    max_bpm: float = MAX_BPM,
    subdivisions=SUBDIVISIONS,
) -> tuple[float, float, float]:
    """Fit a metrical grid to onset times.

    Returns ``(bpm, phase, score)`` where ``phase`` is the offset (seconds) of a
    beat and ``score`` is the mean per-onset cost (lower is a tighter fit). With
    fewer than two onsets there is nothing to fit, so ``bpm`` is ``0.0``.
    """
    times = np.array(sorted(float(t) for t in onsets if t is not None), dtype=np.float64)
    if times.size < 2:
        return (0.0, 0.0, float("inf"))

    lo = 60.0 / max_bpm
    hi = 60.0 / min_bpm

    best_period = lo
    best_phase = 0.0
    best_score = float("inf")

    for step in range(_PERIOD_STEPS):
        period = lo * (hi / lo) ** (step / (_PERIOD_STEPS - 1))
        for phase_step in range(_PHASE_STEPS):
            phase = period * phase_step / _PHASE_STEPS
            score = _beat_cost(times, period, phase, subdivisions)
            if score < best_score:
                best_score, best_period, best_phase = score, period, phase

    # Refine around the winner for a cleaner tempo.
    for _ in range(2):
        span = best_period * 0.03
        for step in range(-20, 21):
            period = best_period + span * step / 20
            if period <= 0:
                continue
            for phase_step in range(-20, 21):
                phase = (best_phase + (best_period / _PHASE_STEPS) * phase_step / 20) % period
                score = _beat_cost(times, period, phase, subdivisions)
                if score < best_score:
                    best_score, best_period, best_phase = score, period, phase

    bpm = 60.0 / best_period if best_period else 0.0
    return (round(bpm, 1), round(best_phase % best_period, 4), round(best_score, 5))


def snap_time(time: float, bpm: float, phase: float, subdivisions=SUBDIVISIONS) -> float:
    """The nearest allowed grid position to ``time`` (unchanged if no grid)."""
    if bpm <= 0:
        return time
    period = 60.0 / bpm
    best = None
    for offset, _penalty in subdivisions:
        index = round((time - phase) / period - offset)
        position = phase + (index + offset) * period
        if best is None or abs(time - position) < abs(time - best):
            best = position
    return best if best is not None else time


def _grid_positions(start: float, end: float, bpm: float, phase: float, subdivisions) -> list[float]:
    if bpm <= 0 or end <= start:
        return []
    period = 60.0 / bpm
    positions: list[float] = []
    index = int(math.floor((start - phase) / period)) - 1
    limit = int(math.ceil((end - phase) / period)) + 1
    while index <= limit:
        base = phase + index * period
        for offset, _penalty in subdivisions:
            position = base + offset * period
            if start - 1e-6 <= position <= end + 1e-6:
                positions.append(position)
        index += 1
    return sorted({round(position, 6) for position in positions})


def predict_syllables(
    start: float,
    end: float,
    count: int,
    bpm: float,
    phase: float,
    subdivisions=SUBDIVISIONS,
) -> list[float]:
    """Onset time for each syllable of a word spanning ``[start, end]``.

    The ideal slots are evenly spread, then each is nudged to the nearest grid
    position that does not collide with its neighbour — so every syllable has a
    sensible place on the grid. Falls back to the even spread when the grid has
    too few lines to hold them all.
    """
    if count <= 0:
        return []
    if count == 1 or bpm <= 0 or end <= start:
        return [start + (end - start) * (i / count) for i in range(count)]

    ideal = [start + (end - start) * (i / count) for i in range(count)]
    positions = _grid_positions(start, end, bpm, phase, subdivisions)
    if len(positions) < count:
        return ideal

    chosen: list[float] = []
    for target in ideal:
        candidate = min(positions, key=lambda position: abs(position - target))
        if chosen and candidate <= chosen[-1]:
            remaining = [p for p in positions if p > chosen[-1]]
            if remaining:
                candidate = min(remaining, key=lambda position: abs(position - target))
            else:
                candidate = chosen[-1] + (end - start) / (count + 1)
        chosen.append(candidate)
    return chosen
