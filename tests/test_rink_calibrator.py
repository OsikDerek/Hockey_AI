"""Tests for RinkCalibrator (Phase B0/B0.5 similarity + B1 homography)."""

from dataclasses import dataclass

import numpy as np
import pytest

from src.game_analysis.rink_calibrator import (
    CENTER_X_FT, CENTER_Y_FT,
    LEFT_GOAL_X_FT, RIGHT_GOAL_X_FT,
    NHL_FACEOFF_DOTS_FT,
    RinkCalibrator,
)


@dataclass
class _StubObj:
    """Minimal stand-in for TrackedObject — only the fields the calibrator reads."""
    center: tuple
    bbox: tuple
    confidence: float = 0.9


def _square_goal(cx: float, cy: float, size: float = 30.0) -> _StubObj:
    half = size / 2.0
    return _StubObj(
        center=(cx, cy),
        bbox=(cx - half, cy - half, cx + half, cy + half),
        confidence=0.9,
    )


def _dot(cx: float, cy: float) -> _StubObj:
    return _StubObj(center=(cx, cy), bbox=(cx - 5, cy - 5, cx + 5, cy + 5), confidence=0.9)


def _build_known_pixel_view(scale: float, origin_px: tuple) -> dict:
    """Render the canonical NHL landmarks at a known similarity transform.

    Used as ground truth: the calibrator should recover both the
    similarity AND a homography close to identity-relative-to-similarity.
    """
    ox, oy = origin_px

    def at(ice_pt):
        return (ox + scale * ice_pt[0], oy + scale * ice_pt[1])

    landmarks = {
        "goal": [
            _square_goal(*at((LEFT_GOAL_X_FT, CENTER_Y_FT))),
            _square_goal(*at((RIGHT_GOAL_X_FT, CENTER_Y_FT))),
        ],
        "centroid": [_dot(*at((CENTER_X_FT, CENTER_Y_FT)))],
        "faceoff": [_dot(*at(p)) for p in NHL_FACEOFF_DOTS_FT],
    }
    return landmarks, at


def test_similarity_recovers_round_trip_on_clean_view():
    """B0 baseline: with two goals + centroid the similarity transform
    should round-trip a known ice point back to the same pixel."""
    landmarks, at = _build_known_pixel_view(scale=5.0, origin_px=(100.0, 50.0))
    cal = RinkCalibrator()
    cal.update(landmarks, frame_width=1280, frame_height=720)
    assert cal.is_calibrated

    target_ice = (50.0, 30.0)
    expected_px = at(target_ice)
    got_px = cal.ice_to_pixel(target_ice)
    assert got_px is not None
    assert abs(got_px[0] - expected_px[0]) < 5.0  # EMA on first fit = full replace
    assert abs(got_px[1] - expected_px[1]) < 5.0


def test_homography_activates_with_full_landmark_set():
    """B1: with goals + centroid + all 8 faceoffs visible, a homography
    should fit and the calibrator should report has_homography."""
    landmarks, _ = _build_known_pixel_view(scale=5.0, origin_px=(100.0, 50.0))
    cal = RinkCalibrator()
    cal.update(landmarks, frame_width=1280, frame_height=720)
    # Run a few frames so EMA settles on a stable homography
    for _ in range(5):
        cal.update(landmarks, frame_width=1280, frame_height=720)
    assert cal.has_homography, "Homography should fit with full landmark set"


def test_homography_round_trip_within_tolerance():
    """B1: pixel_to_ice(ice_to_pixel(p)) ≈ p when homography is active."""
    landmarks, _ = _build_known_pixel_view(scale=5.0, origin_px=(100.0, 50.0))
    cal = RinkCalibrator()
    for _ in range(5):
        cal.update(landmarks, frame_width=1280, frame_height=720)
    assert cal.has_homography

    for ice_pt in [(20.0, 20.5), (100.0, 42.5), (180.0, 64.5), (50.0, 30.0)]:
        px = cal.ice_to_pixel(ice_pt)
        assert px is not None
        ice_back = cal.pixel_to_ice(px)
        assert ice_back is not None
        assert abs(ice_back[0] - ice_pt[0]) < 0.5
        assert abs(ice_back[1] - ice_pt[1]) < 0.5


def test_homography_clears_on_reset():
    """Camera-cut reset must drop the homography too."""
    landmarks, _ = _build_known_pixel_view(scale=5.0, origin_px=(100.0, 50.0))
    cal = RinkCalibrator()
    for _ in range(5):
        cal.update(landmarks, frame_width=1280, frame_height=720)
    assert cal.has_homography
    cal.reset()
    assert not cal.has_homography
    assert not cal.is_calibrated


def test_falls_back_to_similarity_when_too_few_correspondences():
    """B1 requires ≥4 correspondences. With only goals visible (no
    centroid, no faceoffs) the homography path must NOT engage and the
    similarity transform stays in charge."""
    landmarks, _ = _build_known_pixel_view(scale=5.0, origin_px=(100.0, 50.0))
    landmarks["centroid"] = []
    landmarks["faceoff"] = []
    cal = RinkCalibrator()
    cal.update(landmarks, frame_width=1280, frame_height=720)
    assert cal.is_calibrated
    assert not cal.has_homography
    # Transforms still work via similarity
    assert cal.ice_to_pixel((CENTER_X_FT, CENTER_Y_FT)) is not None


def test_disambiguation_rejects_far_off_dots():
    """A spurious detection that's nowhere near any NHL dot should be
    dropped by the snap-tolerance check rather than hijacking a slot."""
    landmarks, at = _build_known_pixel_view(scale=5.0, origin_px=(100.0, 50.0))
    # Add a junk faceoff detection at the corner of the frame, far from
    # any predicted dot.
    landmarks["faceoff"] = list(landmarks["faceoff"]) + [_dot(1240.0, 700.0)]
    cal = RinkCalibrator()
    for _ in range(5):
        cal.update(landmarks, frame_width=1280, frame_height=720)
    assert cal.has_homography
    # Round-trip should still work — the junk dot must not have corrupted H.
    ice_pt = (CENTER_X_FT, CENTER_Y_FT)
    px = cal.ice_to_pixel(ice_pt)
    expected = at(ice_pt)
    assert abs(px[0] - expected[0]) < 5.0
    assert abs(px[1] - expected[1]) < 5.0


def test_homography_recovers_with_partial_faceoffs():
    """Real broadcast footage rarely shows all 8 dots. Verify B1 still
    fits with goals + centroid + 3 dots (=6 correspondences)."""
    landmarks, _ = _build_known_pixel_view(scale=5.0, origin_px=(100.0, 50.0))
    landmarks["faceoff"] = list(landmarks["faceoff"])[:3]
    cal = RinkCalibrator()
    for _ in range(5):
        cal.update(landmarks, frame_width=1280, frame_height=720)
    assert cal.has_homography
