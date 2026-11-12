"""Unit tests for candidate building and ranking on synthetic scores."""

import pytest

from clipmachine.segments import (
    Segment,
    build_candidates,
    energy_peaks,
    rank_segments,
    score_segment,
)

# Energy profile: 0.5s windows, quiet for 0-5s, loud for 5-10s.
PROFILE = [(i * 0.5, -45.0) for i in range(10)] + \
          [(5.0 + i * 0.5, -12.0) for i in range(10)]


def test_energy_peaks_finds_loud_region():
    peaks = energy_peaks(PROFILE)
    assert peaks, "expected at least one peak in the loud half"
    assert all(5.0 <= t <= 10.0 for t in peaks)


def test_energy_peaks_empty_for_flat_profile():
    flat = [(i * 0.5, -30.0) for i in range(10)]
    assert energy_peaks(flat) == []


def test_score_rewards_energy():
    quiet = Segment(start=0.0, end=5.0)
    loud = Segment(start=5.0, end=10.0)
    score_segment(quiet, [], PROFILE)
    score_segment(loud, [], PROFILE)
    assert loud.score > quiet.score
    assert loud.mean_energy > quiet.mean_energy


def test_score_rewards_scene_density():
    sparse = Segment(start=0.0, end=9.0)
    dense = Segment(start=0.0, end=9.0)
    score_segment(sparse, [1.0], [])
    score_segment(dense, [1.0, 2.0, 3.0, 4.0, 5.0], [])
    assert dense.scene_count == 5
    assert dense.score > sparse.score


def test_score_bounds():
    seg = Segment(start=0.0, end=3.0)
    score_segment(seg, [0.5, 1.0, 1.5, 2.0, 2.5], PROFILE)
    assert 0.0 <= seg.score <= 1.0


def test_build_candidates_anchors_on_scenes():
    candidates = build_candidates(
        duration=60.0, scene_times=[10.0, 30.0, 50.0],
        silences=[], energy_profile=[], clip_length=10.0,
    )
    assert candidates
    for cand in candidates:
        assert 0.0 <= cand.start < cand.end <= 60.0
        assert cand.duration >= 5.0  # never a sliver


def test_build_candidates_avoids_starting_in_silence():
    candidates = build_candidates(
        duration=40.0, scene_times=[10.0],
        silences=[(7.0, 12.0)], energy_profile=[], clip_length=10.0,
    )
    for cand in candidates:
        assert not (7.0 <= cand.start < 12.0), cand


def test_build_candidates_tiles_when_no_anchors():
    candidates = build_candidates(
        duration=25.0, scene_times=[], silences=[],
        energy_profile=[], clip_length=10.0,
    )
    assert [ (c.start, c.end) for c in candidates ] == [
        (0.0, 10.0), (10.0, 20.0), (20.0, 25.0),
    ]
    assert all(c.anchors == ["tile"] for c in candidates)


def test_build_candidates_rejects_bad_args():
    with pytest.raises(ValueError):
        build_candidates(0.0, [], [], [])
    with pytest.raises(ValueError):
        build_candidates(10.0, [], [], [], clip_length=0.0)


def test_rank_returns_top_n_non_overlapping_best_first():
    candidates = [
        Segment(start=0.0, end=5.0),
        Segment(start=5.0, end=10.0),
        Segment(start=2.0, end=7.0),   # overlaps both neighbours
        Segment(start=10.0, end=15.0),
    ]
    ranked = rank_segments(candidates, [], PROFILE, top_n=3)
    assert 1 <= len(ranked) <= 3
    scores = [s.score for s in ranked]
    assert scores == sorted(scores, reverse=True)
    for i, a in enumerate(ranked):
        for b in ranked[i + 1:]:
            assert a.end <= b.start or b.end <= a.start


def test_rank_prefers_loud_segment():
    candidates = [Segment(start=0.0, end=5.0), Segment(start=5.0, end=10.0)]
    ranked = rank_segments(candidates, [], PROFILE, top_n=1)
    assert ranked[0].start == 5.0  # the loud half wins


def test_rank_rejects_nonpositive_top_n():
    with pytest.raises(ValueError):
        rank_segments([Segment(0.0, 5.0)], [], [], top_n=0)
