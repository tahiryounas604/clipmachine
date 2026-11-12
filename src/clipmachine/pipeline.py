"""End-to-end pipeline: analyze a long video, write ranked clips + manifest."""

from __future__ import annotations

import json
import os

from . import __version__
from .clip import TARGET_1080, extract_clip
from .detect import (
    audio_energy_profile,
    detect_scene_changes,
    detect_silences,
)
from .ffmpeg_util import probe_media
from .segments import build_candidates, rank_segments


def analyze(source: str, clip_length: float = 15.0,
            scene_threshold: float = 10.0) -> dict:
    """Run all detectors and candidate construction; no clips are cut."""
    if not os.path.isfile(source):
        raise FileNotFoundError(f"source video not found: {source}")
    info = probe_media(source)
    if not info["has_video"]:
        raise ValueError(f"no video stream in {source}")
    duration = info["duration_seconds"]
    scenes = detect_scene_changes(source, threshold=scene_threshold)
    silences = detect_silences(source)
    profile = audio_energy_profile(source)
    candidates = build_candidates(duration, scenes, silences, profile,
                                  clip_length=clip_length)
    return {
        "source": source,
        "duration_seconds": duration,
        "width": info["width"],
        "height": info["height"],
        "has_audio": info["has_audio"],
        "scene_changes": scenes,
        "silences": [list(s) for s in silences],
        "candidates": candidates,
        "energy_profile": profile,
    }


def make_clips(
    source: str,
    outdir: str,
    top_n: int = 3,
    clip_length: float = 15.0,
    size: tuple[int, int] = TARGET_1080,
    mode: str = "crop",
    title: str | None = None,
    loudnorm: bool = False,
    scene_threshold: float = 10.0,
) -> dict:
    """Full pipeline. Writes ``<outdir>/clips/clip_NN.mp4`` files and
    ``<outdir>/manifest.json``; returns the manifest as a dict.

    *title*, when given, is overlaid on every clip; ``{n}`` inside it is
    replaced with the clip's rank (e.g. ``"Clip {n}"``).
    """
    analysis = analyze(source, clip_length=clip_length,
                       scene_threshold=scene_threshold)
    ranked = rank_segments(
        analysis["candidates"],
        analysis["scene_changes"],
        analysis["energy_profile"],
        top_n=top_n,
    )

    clips_dir = os.path.join(outdir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    clip_entries = []
    for rank, seg in enumerate(ranked, start=1):
        filename = f"clip_{rank:02d}.mp4"
        out_path = os.path.join(clips_dir, filename)
        clip_title = title.replace("{n}", str(rank)) if title else None
        result = extract_clip(
            source, seg.start, seg.end, out_path,
            size=size, mode=mode, title=clip_title, loudnorm=loudnorm,
        )
        clip_entries.append({
            "file": os.path.join("clips", filename),
            "rank": rank,
            "source_start_seconds": round(seg.start, 3),
            "source_end_seconds": round(seg.end, 3),
            "duration_seconds": round(result["duration_seconds"], 3),
            "score": round(seg.score, 4),
            "scene_changes": seg.scene_count,
            "mean_energy_db": (
                round(seg.mean_energy, 2) if seg.mean_energy is not None else None
            ),
            "anchors": seg.anchors,
            "width": result["width"],
            "height": result["height"],
        })

    manifest = {
        "tool": "clipmachine",
        "version": __version__,
        "source": os.path.abspath(source),
        "source_duration_seconds": round(analysis["duration_seconds"], 3),
        "source_width": analysis["width"],
        "source_height": analysis["height"],
        "settings": {
            "top_n": top_n,
            "clip_length_seconds": clip_length,
            "size": list(size),
            "mode": mode,
            "title": title,
            "loudnorm": loudnorm,
            "scene_threshold": scene_threshold,
        },
        "scene_changes_seconds": [round(t, 3) for t in analysis["scene_changes"]],
        "silences_seconds": [
            [round(a, 3), round(b, 3)] for a, b in analysis["silences"]
        ],
        "clips": clip_entries,
    }
    manifest_path = os.path.join(outdir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    manifest["manifest_path"] = manifest_path
    manifest["clips_dir"] = clips_dir
    return manifest
