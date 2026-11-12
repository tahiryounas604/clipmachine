"""clipmachine — turn long videos into ranked vertical short clips.

Pipeline: ffmpeg scene-change detection (scdet) + silence detection
(silencedetect) + per-window audio energy (raw PCM RMS) produce
candidate segments; candidates are scored by scene density and audio
energy; the top N are re-encoded as 9:16 clips with an optional title
overlay and optional loudnorm pass, plus a JSON manifest.
"""

__version__ = "0.1.0"
