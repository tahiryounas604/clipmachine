"""Candidate segment construction and ranking — pure logic, no ffmpeg.

Candidates are anchored on moments likely to hold a viewer's attention:
scene cuts and audio-energy peaks. Each candidate window is adjusted so
it does not *start* inside a silence (starting mid-pause reads as a
mistake in a short), scored from scene density and mean audio energy,
and the top N non-overlapping windows win.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detect import SILENCE_FLOOR_DB, mean_energy_db

# dBFS range mapped onto [0, 1] for the energy component of the score.
ENERGY_FLOOR_DB = -50.0
ENERGY_CEIL_DB = -10.0
# One scene cut per this many seconds saturates the scene component.
SCENE_SATURATION_SECONDS = 3.0


@dataclass
class Segment:
    start: float
    end: float
    scene_count: int = 0
    mean_energy: float | None = None  # mean RMS dBFS, None when no audio
    score: float = 0.0
    anchors: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _energy_score(mean_db: float | None) -> float:
    if mean_db is None:
        return 0.0
    frac = (mean_db - ENERGY_FLOOR_DB) / (ENERGY_CEIL_DB - ENERGY_FLOOR_DB)
    return min(1.0, max(0.0, frac))


def _scene_score(scene_count: int, duration: float) -> float:
    if duration <= 0:
        return 0.0
    return min(1.0, scene_count / (duration / SCENE_SATURATION_SECONDS))


def score_segment(
    seg: Segment,
    scene_times: list[float],
    energy_profile: list[tuple[float, float]],
    scene_weight: float = 0.5,
    energy_weight: float = 0.5,
) -> Segment:
    """Fill in seg.scene_count / mean_energy / score in place; return seg."""
    seg.scene_count = sum(1 for t in scene_times if seg.start <= t < seg.end)
    seg.mean_energy = mean_energy_db(energy_profile, seg.start, seg.end)
    seg.score = (
        scene_weight * _scene_score(seg.scene_count, seg.duration)
        + energy_weight * _energy_score(seg.mean_energy)
    )
    return seg


def energy_peaks(
    profile: list[tuple[float, float]], window_seconds: float = 0.5
) -> list[float]:
    """Times of windows that are local maxima above the profile mean.

    (Mean, not median: on a bimodal quiet/loud profile the median can
    equal the loud level itself, which would hide every peak.)
    """
    if len(profile) < 3:
        return []
    mean = sum(db for _, db in profile) / len(profile)
    peaks = []
    for i in range(1, len(profile) - 1):
        t, db = profile[i]
        if db > mean and db >= profile[i - 1][1] and db >= profile[i + 1][1]:
            peaks.append(t + window_seconds / 2.0)
    return peaks


def _inside_silence(t: float, silences: list[tuple[float, float]]) -> float | None:
    """End of the silence containing *t*, or None if *t* is not in one."""
    for start, end in silences:
        if start <= t < end:
            return end
    return None


def build_candidates(
    duration: float,
    scene_times: list[float],
    silences: list[tuple[float, float]],
    energy_profile: list[tuple[float, float]],
    clip_length: float = 15.0,
) -> list[Segment]:
    """Anchor-centred windows of ~*clip_length* seconds over the source.

    Anchors: every scene cut ("scene") and every energy peak ("energy").
    A window starts a quarter-clip before its anchor so the moment is
    not the very first frame. Starts landing inside a silence are moved
    to that silence's end. Near-duplicate windows (same start within
    half a clip) collapse, keeping the earliest anchor's kind list.
    If nothing anchors, the source is simply tiled into clip-length
    chunks so the pipeline still produces output.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if clip_length <= 0:
        raise ValueError("clip_length must be positive")

    anchors: list[tuple[float, str]] = [(t, "scene") for t in scene_times]
    anchors += [(t, "energy") for t in energy_peaks(energy_profile)]
    anchors.sort()

    candidates: list[Segment] = []
    for anchor, kind in anchors:
        start = max(0.0, anchor - clip_length * 0.25)
        silence_end = _inside_silence(start, silences)
        if silence_end is not None:
            start = silence_end
        end = min(duration, start + clip_length)
        if end - start < clip_length * 0.5 and start > 0:
            # Too short at the tail (or after a silence shift): pull the
            # window back so it still covers a full clip when possible.
            start = max(0.0, end - clip_length)
            silence_end = _inside_silence(start, silences)
            if silence_end is not None and silence_end + clip_length * 0.5 <= end:
                start = silence_end
        if end - start < 1.0:
            continue
        existing = next(
            (c for c in candidates if abs(c.start - start) < clip_length * 0.5),
            None,
        )
        if existing is not None:
            if kind not in existing.anchors:
                existing.anchors.append(kind)
            continue
        candidates.append(Segment(start=start, end=end, anchors=[kind]))

    if not candidates:
        t = 0.0
        while t < duration:
            end = min(duration, t + clip_length)
            if end - t >= 1.0:
                candidates.append(Segment(start=t, end=end, anchors=["tile"]))
            t += clip_length
    return candidates


def rank_segments(
    candidates: list[Segment],
    scene_times: list[float],
    energy_profile: list[tuple[float, float]],
    top_n: int = 3,
    scene_weight: float = 0.5,
    energy_weight: float = 0.5,
) -> list[Segment]:
    """Score all candidates; return the best *top_n* non-overlapping ones,
    best first. Ties break toward the earlier start (deterministic)."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    scored = [score_segment(c, scene_times, energy_profile,
                            scene_weight, energy_weight) for c in candidates]
    # Scores are compared at manifest precision (4 dp): differences below
    # that are encoder noise, and a full-length clip should beat a
    # tail-truncated one at effectively the same score.
    ordered = sorted(
        scored, key=lambda s: (-round(s.score, 4), -s.duration, s.start))
    chosen: list[Segment] = []
    for seg in ordered:
        if any(seg.start < c.end and c.start < seg.end for c in chosen):
            continue
        chosen.append(seg)
        if len(chosen) >= top_n:
            break
    return chosen


# Re-exported for callers that only import segments.
__all__ = [
    "Segment", "build_candidates", "rank_segments", "score_segment",
    "energy_peaks", "SILENCE_FLOOR_DB",
]
