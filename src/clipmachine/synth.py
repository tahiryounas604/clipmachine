"""Deterministic synthetic test video, shared by tests and examples.

Layout (18 s, 640x360, 30 fps), three 6-second blocks with hard cuts
between them and a distinct audio level in each:

====  ====================  ==========================
time  video                 audio (sine)
====  ====================  ==========================
0-6   testsrc2 pattern      440 Hz, loud (mean ~-4 dB)
6-12  solid red             660 Hz at gain 0 (silence)
12-18 testsrc pattern       880 Hz, quiet (mean ~-15 dB)
====  ====================  ==========================

Note: ffmpeg's ``sine`` source peaks at only ~-18 dBFS on its own, so
the per-block ``volume`` gains below (7.0 / 0.0 / 2.0) are what set the
effective levels; without them even the "loud" block would sit near
the -30 dB silence threshold.

So a correct pipeline must find scene cuts near t=6 and t=12, a
silence spanning roughly [6, 12], and energy ordered
block A > block C > block B.
"""

from __future__ import annotations

import os

from .ffmpeg_util import check_tool, run

BLOCK_SECONDS = 6.0
TOTAL_SECONDS = 18.0


def make_test_video(output_path: str) -> str:
    """Render the synthetic video to *output_path* with one ffmpeg call."""
    check_tool("ffmpeg")
    d = BLOCK_SECONDS
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={d}",
        "-f", "lavfi", "-i", f"color=c=red:size=640x360:rate=30:duration={d}",
        "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=30:duration={d}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={d}",
        "-f", "lavfi", "-i", f"sine=frequency=660:duration={d}",
        "-f", "lavfi", "-i", f"sine=frequency=880:duration={d}",
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v];"
        "[3:a]volume=7.0[a0];[4:a]volume=0.0[a1];[5:a]volume=2.0[a2];"
        "[a0][a1][a2]concat=n=3:v=0:a=1[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", output_path,
    ]
    run(cmd)
    if not os.path.isfile(output_path):
        raise RuntimeError(f"synthetic video was not created: {output_path}")
    return output_path
