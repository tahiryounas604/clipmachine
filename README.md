# clipmachine

Turn a long video into ranked vertical short clips, entirely offline,
with Python + ffmpeg.

clipmachine analyzes a source video with three ffmpeg-based signals —
scene changes, silences, and audio energy — builds candidate segments
around the interesting moments, scores them, and cuts the top N as
9:16 clips (1080x1920 or 720x1280) with an optional title overlay and
optional loudness normalization. Every run writes a `manifest.json`
recording exactly where each clip came from and why it was picked.

## Features

- **Scene-change detection** via ffmpeg's `scdet` filter.
- **Silence detection** via `silencedetect` (default: below -30 dB for
  at least 0.5 s). Candidate windows are shifted so they never *start*
  inside a silence.
- **Audio energy profile**: the audio track is decoded to raw PCM and
  RMS level (dBFS) is computed per 0.5 s window in pure Python;
  energy peaks become candidate anchors alongside scene cuts.
- **Segment ranking**: score = scene density + mean audio energy
  (equal weights by default), saturated and clamped to [0, 1];
  top-N selection enforces non-overlapping clips.
- **9:16 reframe**, two modes: `crop` (scale-to-cover + centre crop,
  full-bleed) or `pad` (scale-to-fit + solid-colour letterbox).
- **Optional title overlay** (`drawtext`, `{n}` in the text expands to
  the clip rank) and **optional loudnorm** (`I=-16:TP=-1.5:LRA=11`).
- **Manifest**: `manifest.json` with source facts, all detected scene
  cuts and silences, and per-clip source timestamps, duration, score,
  scene count, mean energy, and output dimensions.
- **Fallback**: a source with no detectable anchors is tiled into
  clip-length chunks so the pipeline still produces output.
- Zero Python runtime dependencies — stdlib + the `ffmpeg`/`ffprobe`
  binaries only.

## Install

Requires Python 3.11 or 3.12 and ffmpeg (with `libx264`) on PATH.

```bash
sudo apt-get install ffmpeg fonts-dejavu-core   # Debian/Ubuntu
pip install -e .
```

## Usage

```bash
# Cut the top 3 clips (15 s each, 1080x1920) from a long video
clipmachine make podcast.mp4 -o out/

# Top 5 clips, 30 s target, letterboxed, titled, loudness-normalized
clipmachine make stream.mp4 -o out/ -n 5 --length 30 \
    --mode pad --title "Clip {n}" --loudnorm

# Smaller/faster output
clipmachine make video.mp4 -o out/ --size 720x1280

# Just inspect the signals and ranked candidates (JSON, cuts nothing)
clipmachine analyze video.mp4 -n 5
```

Output layout:

```
out/
  clips/clip_01.mp4     # rank 1 (best score)
  clips/clip_02.mp4
  manifest.json
```

Or from Python:

```python
from clipmachine.pipeline import make_clips

manifest = make_clips("podcast.mp4", "out/", top_n=3, clip_length=15.0,
                      title="Clip {n}", loudnorm=True)
```

A self-contained demo (synthesizes its own test video, then clips it):

```bash
python examples/make_demo.py demo_output
```

## Architecture

```
src/clipmachine/
  ffmpeg_util.py  shutil.which tool checks, subprocess runner with
                  stderr-tail errors, ffprobe JSON probing
  detect.py       scdet scene cuts, silencedetect spans, PCM-decode
                  RMS energy profile + log parsers (unit-tested)
  segments.py     pure logic: anchors -> candidate windows (silence-
                  aware), scoring, top-N non-overlapping ranking
  clip.py         frame-accurate cut (input seek + re-encode),
                  crop/pad reframe filter, drawtext title via textfile
                  (no filter-escaping bugs), optional loudnorm
  pipeline.py     analyze() + make_clips(): orchestration + manifest
  synth.py        deterministic 18 s synthetic video (3 blocks: loud
                  tone/pattern, silence/red, quiet tone/pattern) used
                  by the tests and the demo
  cli.py          argparse CLI: `make` and `analyze`
```

The ffmpeg subprocess/probe/drawtext-textfile patterns are adapted
from the sibling Tytechlabs product `youtube-automation`; the code
here is written for this product.

## Tests

```bash
pip install pytest
pytest -q
```

25 tests, all offline. They synthesize the test video with ffmpeg at
runtime (no media files in the repo) and cover: log parsers, RMS math,
ranking/scoring on synthetic profiles, silence-aware candidate
construction, detector integration (cuts found at 6 s/12 s, silence
span 6–12 s, energy ordered loud > quiet > silent), clip extraction
(dimensions, duration, pad mode, title overlay, loudnorm), and the
full pipeline (clips on disk, manifest correctness, ranked order).

Verified end-to-end smoke (this exact flow is what the pipeline test
asserts, run here at full size): an 18 s 640x360 synthetic source with
`clipmachine make -n 2 --length 5 --title "Clip {n}" --loudnorm`
produced two 5.0 s clips, both ffprobe-verified `h264 1080x1920`,
ranked loud-block first (score 0.5, -4.18 dB) over the quiet block
(0.4368, -15.06 dB), with the silent middle block producing no clip;
the manifest recorded scene cuts at exactly [6.0, 12.0] and the
silence span [[6.0, 12.0]].

## Honest limits

- **Centre-frame reframing only.** Crop mode always takes the centre
  of the frame; there is no face/subject tracking, so off-centre
  subjects in wide footage can be cut out. Pad mode avoids losing
  content at the cost of letterbox bands.
- **Heuristic ranking, not understanding.** The score rewards scene
  density and loudness. A loud, cut-heavy intro or advert can outrank
  a quiet but important moment; there is no speech transcription or
  semantic analysis in v0.1.
- **Hard cuts only.** `scdet` finds abrupt scene changes; gradual
  transitions (dissolves, fades) may be missed at the default
  threshold (tunable via `--scene-threshold`).
- **Silence handling is start-only.** Windows never start inside a
  silence, but a clip can still *contain* or *end in* a quiet stretch.
- **Single-pass loudnorm** is ffmpeg's dynamic single-pass mode, not
  the measured two-pass workflow; levels are approximately, not
  exactly, -16 LUFS.
- **Re-encode always.** Cuts are frame-accurate because every clip is
  re-encoded with libx264 (veryfast preset); this is slower than
  stream-copy cutting and always generation-lossy.
- Tested only on Linux with ffmpeg 8.x, Python 3.11/3.12, and
  synthetic test content — not yet on real-world footage.

## Author

Tahir Younas — Tytechlabs — https://www.linkedin.com/in/tahiryounas604

## License

GPL-3.0 — see [LICENSE](LICENSE).
