"""End-to-end tests: synthetic video in, vertical clips + manifest out."""

import json
import os

import pytest

from clipmachine.clip import TARGET_720, extract_clip, reframe_filter
from clipmachine.ffmpeg_util import probe_media
from clipmachine.pipeline import make_clips
from clipmachine.synth import make_test_video


@pytest.fixture(scope="module")
def synth_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("e2e") / "synth.mp4"
    return make_test_video(str(path))


def test_reframe_filter_modes():
    assert "crop=1080:1920" in reframe_filter(1080, 1920, mode="crop")
    assert "pad=720:1280" in reframe_filter(720, 1280, mode="pad")
    with pytest.raises(ValueError):
        reframe_filter(720, 1280, mode="bogus")


def test_extract_clip_dimensions_and_duration(synth_video, tmp_path):
    out = str(tmp_path / "one.mp4")
    result = extract_clip(synth_video, 0.0, 4.0, out, size=TARGET_720)
    assert result["width"] == 720
    assert result["height"] == 1280
    assert result["duration_seconds"] == pytest.approx(4.0, abs=0.6)


def test_extract_clip_pad_mode(synth_video, tmp_path):
    out = str(tmp_path / "padded.mp4")
    result = extract_clip(synth_video, 12.0, 16.0, out,
                          size=TARGET_720, mode="pad")
    assert (result["width"], result["height"]) == (720, 1280)


def test_extract_clip_with_title_and_loudnorm(synth_video, tmp_path):
    out = str(tmp_path / "titled.mp4")
    result = extract_clip(synth_video, 0.0, 3.0, out, size=TARGET_720,
                          title="Hello Clip", loudnorm=True)
    assert (result["width"], result["height"]) == (720, 1280)
    assert probe_media(out)["has_audio"]


def test_extract_clip_rejects_bad_range(synth_video, tmp_path):
    with pytest.raises(ValueError):
        extract_clip(synth_video, 5.0, 5.0, str(tmp_path / "x.mp4"))
    with pytest.raises(FileNotFoundError):
        extract_clip(str(tmp_path / "nope.mp4"), 0.0, 1.0,
                     str(tmp_path / "x.mp4"))


def test_make_clips_end_to_end(synth_video, tmp_path):
    outdir = str(tmp_path / "out")
    manifest = make_clips(
        synth_video, outdir, top_n=2, clip_length=5.0,
        size=TARGET_720, title="Clip {n}",
    )
    assert len(manifest["clips"]) == 2

    manifest_path = os.path.join(outdir, "manifest.json")
    assert os.path.isfile(manifest_path)
    with open(manifest_path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["tool"] == "clipmachine"
    assert on_disk["source_duration_seconds"] == pytest.approx(18.0, abs=0.5)
    assert any(abs(t - 6.0) < 0.7 for t in on_disk["scene_changes_seconds"])
    assert any(abs(t - 12.0) < 0.7 for t in on_disk["scene_changes_seconds"])

    scores = [c["score"] for c in manifest["clips"]]
    assert scores == sorted(scores, reverse=True)  # ranked best first
    for entry in manifest["clips"]:
        path = os.path.join(outdir, entry["file"])
        assert os.path.isfile(path) and os.path.getsize(path) > 0
        info = probe_media(path)
        assert (info["width"], info["height"]) == (720, 1280)
        assert info["duration_seconds"] == pytest.approx(
            entry["duration_seconds"], abs=0.6)
        assert 0.0 <= entry["source_start_seconds"] < entry["source_end_seconds"]
        assert entry["source_end_seconds"] <= 18.5
        assert entry["duration_seconds"] <= 5.6  # target length + tolerance


def test_make_clips_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_clips(str(tmp_path / "nope.mp4"), str(tmp_path / "out"))
