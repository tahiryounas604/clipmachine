"""Parser unit tests + detector integration tests on a synthetic video."""

import array

import pytest

from clipmachine.detect import (
    SILENCE_FLOOR_DB,
    audio_energy_profile,
    detect_scene_changes,
    detect_silences,
    mean_energy_db,
    parse_scene_times,
    parse_silences,
    rms_db,
)
from clipmachine.synth import make_test_video


def test_parse_scene_times():
    log = (
        "[Parsed_scdet_1 @ 0x55] lavfi.scd.mafd: 12.5\n"
        "[Parsed_scdet_1 @ 0x55] lavfi.scd.time: 6.033\n"
        "frame=  100 fps= 50\n"
        "[Parsed_scdet_1 @ 0x55] lavfi.scd.time: 12.000\n"
    )
    assert parse_scene_times(log) == [6.033, 12.0]


def test_parse_silences_pairs_starts_and_ends():
    log = (
        "[silencedetect @ 0x1] silence_start: 5.98\n"
        "[silencedetect @ 0x1] silence_end: 12.01 | silence_duration: 6.03\n"
        "[silencedetect @ 0x1] silence_start: 20.0\n"
        "[silencedetect @ 0x1] silence_end: 21.5 | silence_duration: 1.5\n"
    )
    assert parse_silences(log) == [(5.98, 12.01), (20.0, 21.5)]


def test_rms_db_known_values():
    loud = array.array("f", [0.7071, -0.7071] * 500)   # ~ -3 dBFS
    quiet = array.array("f", [0.01, -0.01] * 500)      # ~ -40 dBFS
    silent = array.array("f", [0.0] * 500)
    assert rms_db(loud) == pytest.approx(-3.0, abs=0.2)
    assert rms_db(quiet) == pytest.approx(-40.0, abs=0.2)
    assert rms_db(silent) == SILENCE_FLOOR_DB
    assert rms_db(array.array("f")) == SILENCE_FLOOR_DB


@pytest.fixture(scope="module")
def synth_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("synth") / "synth.mp4"
    return make_test_video(str(path))


def test_detect_scene_changes_on_synth(synth_video):
    scenes = detect_scene_changes(synth_video)
    assert any(abs(t - 6.0) < 0.7 for t in scenes), scenes
    assert any(abs(t - 12.0) < 0.7 for t in scenes), scenes


def test_detect_silences_on_synth(synth_video):
    silences = detect_silences(synth_video)
    assert silences, "expected the silent middle block to be detected"
    start, end = max(silences, key=lambda s: s[1] - s[0])
    assert start == pytest.approx(6.0, abs=1.0)
    assert end == pytest.approx(12.0, abs=1.0)


def test_energy_profile_ordering_on_synth(synth_video):
    profile = audio_energy_profile(synth_video)
    assert profile
    block_a = mean_energy_db(profile, 0.5, 5.5)
    block_b = mean_energy_db(profile, 6.5, 11.5)
    block_c = mean_energy_db(profile, 12.5, 17.5)
    assert block_a > block_c > block_b
    assert block_b < -60.0  # digital silence
