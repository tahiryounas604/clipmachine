"""Command-line interface for clipmachine.

Subcommands: ``make`` (full pipeline) and ``analyze`` (report the
detected signals and ranked candidates without cutting anything).
Run ``python -m clipmachine --help`` for usage.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .clip import TARGET_1080, TARGET_720
from .pipeline import analyze, make_clips
from .segments import rank_segments

SIZES = {"1080x1920": TARGET_1080, "720x1280": TARGET_720}


def _cmd_make(args: argparse.Namespace) -> int:
    manifest = make_clips(
        args.input, args.outdir,
        top_n=args.top, clip_length=args.length,
        size=SIZES[args.size], mode=args.mode,
        title=args.title, loudnorm=args.loudnorm,
        scene_threshold=args.scene_threshold,
    )
    print(f"Source: {manifest['source']} "
          f"({manifest['source_duration_seconds']}s, "
          f"{manifest['source_width']}x{manifest['source_height']})")
    for clip in manifest["clips"]:
        print(f"  #{clip['rank']} {clip['file']} "
              f"[{clip['source_start_seconds']}s -> {clip['source_end_seconds']}s] "
              f"score={clip['score']} scenes={clip['scene_changes']} "
              f"energy={clip['mean_energy_db']}dB "
              f"{clip['width']}x{clip['height']} {clip['duration_seconds']}s")
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    result = analyze(args.input, clip_length=args.length,
                     scene_threshold=args.scene_threshold)
    ranked = rank_segments(result["candidates"], result["scene_changes"],
                           result["energy_profile"], top_n=args.top)
    report = {
        "source": result["source"],
        "duration_seconds": round(result["duration_seconds"], 3),
        "scene_changes_seconds": [round(t, 3) for t in result["scene_changes"]],
        "silences_seconds": [
            [round(a, 3), round(b, 3)] for a, b in result["silences"]
        ],
        "ranked_candidates": [
            {
                "start": round(s.start, 3), "end": round(s.end, 3),
                "score": round(s.score, 4), "scene_changes": s.scene_count,
                "mean_energy_db": (
                    round(s.mean_energy, 2) if s.mean_energy is not None else None
                ),
                "anchors": s.anchors,
            }
            for s in ranked
        ],
    }
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipmachine",
        description="Turn a long video into ranked vertical short clips.",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make", help="Detect, rank and cut the top clips.")
    p.add_argument("input", help="Source video file.")
    p.add_argument("-o", "--outdir", default="output",
                   help="Output directory (clips/ + manifest.json inside).")
    p.add_argument("-n", "--top", type=int, default=3,
                   help="How many clips to cut (default 3).")
    p.add_argument("--length", type=float, default=15.0,
                   help="Target clip length in seconds (default 15).")
    p.add_argument("--size", choices=sorted(SIZES), default="1080x1920",
                   help="Output frame size (default 1080x1920).")
    p.add_argument("--mode", choices=["crop", "pad"], default="crop",
                   help="Reframe mode: crop (full-bleed) or pad (letterbox).")
    p.add_argument("--title", default=None,
                   help="Overlay text; {n} becomes the clip rank.")
    p.add_argument("--loudnorm", action="store_true",
                   help="Apply ffmpeg loudnorm (I=-16) to clip audio.")
    p.add_argument("--scene-threshold", type=float, default=10.0,
                   help="scdet threshold 0-100 (default 10).")
    p.set_defaults(func=_cmd_make)

    p = sub.add_parser("analyze", help="Print signals + ranked candidates as JSON.")
    p.add_argument("input", help="Source video file.")
    p.add_argument("-n", "--top", type=int, default=3)
    p.add_argument("--length", type=float, default=15.0)
    p.add_argument("--scene-threshold", type=float, default=10.0)
    p.set_defaults(func=_cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
