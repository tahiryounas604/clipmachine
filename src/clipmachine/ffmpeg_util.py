"""Thin ffmpeg/ffprobe wrappers.

Adapted from the patterns used in the sibling ``youtube-automation``
product: locate tools with ``shutil.which``, run ffmpeg via
``subprocess`` and raise with the tail of stderr on failure, and read
stream/format facts from ffprobe's JSON output.
"""

from __future__ import annotations

import json
import shutil
import subprocess


def check_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required tool '{name}' not found on PATH. Install ffmpeg.")
    return path


def run(cmd: list[str], capture_text: bool = True) -> subprocess.CompletedProcess:
    """Run *cmd*; raise RuntimeError with the stderr tail on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=capture_text)
    if proc.returncode != 0:
        err = proc.stderr if capture_text else proc.stderr.decode("utf-8", "replace")
        raise RuntimeError(
            "command failed (exit %d): %s\n%s"
            % (proc.returncode, " ".join(cmd[:3]), (err or "").strip()[-2000:])
        )
    return proc


def probe_media(path: str) -> dict:
    """Return duration, video dimensions and stream facts via ffprobe."""
    check_tool("ffprobe")
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("ffprobe failed on %s: %s" % (path, proc.stderr.strip()[-500:]))
    data = json.loads(proc.stdout)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    return {
        "duration_seconds": float(data["format"]["duration"]),
        "width": int(video["width"]) if video else None,
        "height": int(video["height"]) if video else None,
        "has_video": video is not None,
        "has_audio": audio is not None,
    }
