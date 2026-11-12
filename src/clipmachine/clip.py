"""Clip extraction: cut, reframe to 9:16, overlay, normalize.

Reframing modes:

* ``crop`` (default) — scale to *cover* the target frame, then
  centre-crop: full-bleed vertical video, edges of wide sources lost.
* ``pad`` — scale to *fit* inside the target frame and pad the empty
  bands with a solid colour: nothing is cut off.

The title overlay follows the sibling youtube-automation pattern:
the text is written to a temp file and referenced with drawtext's
``textfile=`` so arbitrary titles never need filter-graph escaping.
"""

from __future__ import annotations

import os
import tempfile

from .ffmpeg_util import check_tool, probe_media, run

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

TARGET_1080 = (1080, 1920)
TARGET_720 = (720, 1280)


def reframe_filter(width: int, height: int, mode: str = "crop",
                   pad_color: str = "black") -> str:
    """Video filter chain that turns any input into width x height."""
    if mode == "crop":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1"
        )
    if mode == "pad":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={pad_color},setsar=1"
        )
    raise ValueError(f"unknown reframe mode: {mode!r} (use 'crop' or 'pad')")


def extract_clip(
    source: str,
    start: float,
    end: float,
    output_path: str,
    size: tuple[int, int] = TARGET_1080,
    mode: str = "crop",
    title: str | None = None,
    loudnorm: bool = False,
) -> dict:
    """Cut [start, end) from *source* into a vertical MP4 at *output_path*.

    Seeking is done on the input side and the segment is re-encoded, so
    cuts are frame-accurate rather than keyframe-aligned. Returns the
    ffprobe facts of the produced file.
    """
    check_tool("ffmpeg")
    if not os.path.isfile(source):
        raise FileNotFoundError(f"source video not found: {source}")
    if end <= start:
        raise ValueError("end must be after start")
    width, height = size
    has_audio = probe_media(source)["has_audio"]

    vf = reframe_filter(width, height, mode=mode)
    tmpdir = None
    if title:
        tmpdir = tempfile.mkdtemp(prefix="clipmachine_")
        text_path = os.path.join(tmpdir, "title.txt")
        with open(text_path, "w", encoding="utf-8") as fh:
            fh.write(title)
        font = f"fontfile={FONT_BOLD}:" if os.path.isfile(FONT_BOLD) else ""
        vf += (
            f",drawtext={font}textfile='{text_path}':fontsize=64:"
            "fontcolor=white:borderw=3:bordercolor=black:"
            "x=(w-text_w)/2:y=90"
        )
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", source,
            "-t", f"{end - start:.3f}",
            "-vf", vf,
        ]
        if has_audio:
            cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"] if loudnorm else []
            cmd += ["-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd += [
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", output_path,
        ]
        run(cmd)
    finally:
        if tmpdir and os.path.isdir(tmpdir):
            for name in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, name))
            os.rmdir(tmpdir)

    info = probe_media(output_path)
    return {"path": output_path, **info}
