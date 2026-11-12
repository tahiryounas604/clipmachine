"""Signal detection on a source video, all via ffmpeg.

Three independent signals feed candidate generation:

* **Scene changes** — the ``scdet`` filter; its log lines carry
  ``lavfi.scd.time: <seconds>`` for every detected cut.
* **Silences** — the ``silencedetect`` filter; log lines carry
  ``silence_start`` / ``silence_end`` timestamps.
* **Audio energy** — the audio track is decoded to raw mono PCM with
  ffmpeg and the RMS level (dBFS) is computed per time window in pure
  Python. This avoids fragile parsing of per-frame ``astats`` logs and
  keeps the numbers deterministic and unit-testable.

ffmpeg writes filter logs to stderr even on success, so the detection
helpers read stderr from a successful run instead of using
:func:`clipmachine.ffmpeg_util.run` (which treats output as an error).
"""

from __future__ import annotations

import array
import math
import re
import subprocess

from .ffmpeg_util import check_tool, probe_media

SILENCE_FLOOR_DB = -90.0  # reported for digital silence / missing audio

_SCD_TIME_RE = re.compile(r"lavfi\.scd\.time:\s*([0-9]+(?:\.[0-9]+)?)")
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9]+(?:\.[0-9]+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[0-9]+(?:\.[0-9]+)?)")


def _run_read_stderr(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg detection failed (exit %d):\n%s"
            % (proc.returncode, proc.stderr.strip()[-2000:])
        )
    return proc.stderr


def parse_scene_times(stderr_text: str) -> list[float]:
    """Extract sorted scene-cut timestamps from scdet log output."""
    return sorted(float(m.group(1)) for m in _SCD_TIME_RE.finditer(stderr_text))


def parse_silences(stderr_text: str) -> list[tuple[float, float]]:
    """Pair up silence_start/silence_end lines into (start, end) spans."""
    starts = [float(m.group(1)) for m in _SILENCE_START_RE.finditer(stderr_text)]
    ends = [float(m.group(1)) for m in _SILENCE_END_RE.finditer(stderr_text)]
    spans = []
    for i, start in enumerate(starts):
        end = ends[i] if i < len(ends) else start
        if end > start:
            spans.append((start, end))
    return spans


def detect_scene_changes(path: str, threshold: float = 10.0) -> list[float]:
    """Timestamps (seconds) of scene cuts. *threshold* is scdet's 0-100 scale."""
    check_tool("ffmpeg")
    stderr = _run_read_stderr([
        "ffmpeg", "-hide_banner", "-i", path,
        "-vf", f"scdet=threshold={threshold}",
        "-an", "-f", "null", "-",
    ])
    return parse_scene_times(stderr)


def detect_silences(
    path: str, noise_db: float = -30.0, min_duration: float = 0.5
) -> list[tuple[float, float]]:
    """(start, end) spans quieter than *noise_db* for >= *min_duration* s."""
    check_tool("ffmpeg")
    if not probe_media(path)["has_audio"]:
        return []
    stderr = _run_read_stderr([
        "ffmpeg", "-hide_banner", "-i", path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-vn", "-f", "null", "-",
    ])
    return parse_silences(stderr)


def rms_db(samples: array.array) -> float:
    """RMS level in dBFS of float samples in [-1.0, 1.0]; floor for silence."""
    if not samples:
        return SILENCE_FLOOR_DB
    mean_square = sum(s * s for s in samples) / len(samples)
    if mean_square <= 0.0:
        return SILENCE_FLOOR_DB
    return max(SILENCE_FLOOR_DB, 10.0 * math.log10(mean_square))


def audio_energy_profile(
    path: str, window_seconds: float = 0.5, sample_rate: int = 8000
) -> list[tuple[float, float]]:
    """[(window_start, rms_db)] over the audio track, decoded via ffmpeg.

    Returns an empty list when the file has no audio stream.
    """
    check_tool("ffmpeg")
    if not probe_media(path)["has_audio"]:
        return []
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", path,
         "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg audio decode failed (exit %d): %s"
            % (proc.returncode, proc.stderr.decode("utf-8", "replace").strip()[-500:])
        )
    samples = array.array("f")
    samples.frombytes(proc.stdout)
    per_window = max(1, int(window_seconds * sample_rate))
    profile = []
    for i in range(0, len(samples), per_window):
        chunk = samples[i:i + per_window]
        profile.append((i / sample_rate, rms_db(chunk)))
    return profile


def mean_energy_db(
    profile: list[tuple[float, float]], start: float, end: float,
    window_seconds: float = 0.5,
) -> float | None:
    """Mean RMS dB of profile windows overlapping [start, end); None if none."""
    values = [
        db for (t, db) in profile
        if t + window_seconds > start and t < end
    ]
    if not values:
        return None
    return sum(values) / len(values)
