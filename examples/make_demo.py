"""Demo: synthesize a test video and cut vertical clips from it.

    python examples/make_demo.py [outdir]

Generates demo_source.mp4 (three 6s blocks: loud tone + pattern,
silence + solid red, quiet tone + pattern), runs the pipeline, and
prints the manifest summary.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clipmachine.pipeline import make_clips  # noqa: E402
from clipmachine.synth import make_test_video  # noqa: E402


def main() -> int:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "demo_output"
    os.makedirs(outdir, exist_ok=True)
    source = os.path.join(outdir, "demo_source.mp4")
    make_test_video(source)
    manifest = make_clips(source, outdir, top_n=2, clip_length=5.0,
                          title="Clip {n}")
    for clip in manifest["clips"]:
        print(f"#{clip['rank']} {clip['file']} "
              f"[{clip['source_start_seconds']}s -> "
              f"{clip['source_end_seconds']}s] score={clip['score']}")
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
