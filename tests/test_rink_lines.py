"""Tests for rink_lines.py — Hough-based line detection."""

import numpy as np

from src.game_analysis.rink_lines import (
    detect_rink_lines, sample_line_points,
)


def _make_synthetic_rink(w=1280, h=720) -> np.ndarray:
    """Render a simplified rink: white ice with painted blue/red vertical lines.

    NOT pixel-perfect to NHL spec, but each line is in a known x-column so the
    detector can be verified end-to-end (HSV thresholding → Hough → cluster).
    """
    frame = np.full((h, w, 3), 220, dtype=np.uint8)  # off-white ice
    # Vertical bands: left blue, center red, right blue
    blue = (255, 80, 30)   # BGR
    red = (30, 30, 220)
    line_thickness = 16
    cv_x = {"left_blue": w * 0.3, "center_red": w * 0.5, "right_blue": w * 0.7}
    for x_center in cv_x.values():
        x = int(x_center)
        for off in range(-line_thickness // 2, line_thickness // 2):
            color = blue if x_center in (cv_x["left_blue"], cv_x["right_blue"]) else red
            frame[:, x + off] = color
    return frame, cv_x


def test_detects_blue_and_red_lines_on_synthetic_rink():
    frame, expected_x = _make_synthetic_rink()
    result = detect_rink_lines(frame)
    assert len(result["blue"]) >= 2, f"Expected ≥2 blue lines, got {len(result['blue'])}"
    assert len(result["red"]) >= 1, f"Expected ≥1 red line, got {len(result['red'])}"


def test_blue_lines_match_expected_x_positions():
    frame, expected_x = _make_synthetic_rink(w=1280)
    result = detect_rink_lines(frame)
    blue_xs = sorted(L["mid"][0] for L in result["blue"])
    # Two blue lines at 30% and 70% of width = ~384 and ~896
    assert len(blue_xs) >= 2
    assert abs(blue_xs[0] - expected_x["left_blue"]) < 20
    assert abs(blue_xs[-1] - expected_x["right_blue"]) < 20


def test_red_center_line_detected():
    frame, expected_x = _make_synthetic_rink()
    result = detect_rink_lines(frame)
    # The painted center line should yield at least one red detection
    # near the middle of the frame.
    red_xs = [L["mid"][0] for L in result["red"]]
    assert any(abs(x - expected_x["center_red"]) < 25 for x in red_xs), (
        f"No red line near center {expected_x['center_red']}; got {red_xs}"
    )


def test_detects_horizontal_lines_too():
    """Broadcast cams put rink lines near-horizontal in pixel space, so the
    detector must accept any orientation. Caller (RinkCalibrator) is
    responsible for matching to the right ice-line."""
    h, w = 720, 1280
    frame = np.full((h, w, 3), 220, dtype=np.uint8)
    # Horizontal red bar at y=h/2 (could be a center line viewed from a
    # near-side broadcast angle)
    frame[h // 2 - 8: h // 2 + 8, :] = (30, 30, 220)
    result = detect_rink_lines(frame)
    assert len(result["red"]) >= 1, f"Should detect horizontal red, got {result['red']}"


def test_handles_empty_frame_gracefully():
    result = detect_rink_lines(None)
    assert result == {"blue": [], "red": []}
    result = detect_rink_lines(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result == {"blue": [], "red": []}


def test_sample_line_points_returns_correct_count():
    line = {"p1": (10.0, 20.0), "p2": (110.0, 320.0)}
    pts = sample_line_points(line, n_samples=5)
    assert len(pts) == 5
    # Endpoints should match
    assert pts[0] == (10.0, 20.0)
    assert pts[-1] == (110.0, 320.0)
    # Midpoint should be between
    mid = pts[2]
    assert 50 < mid[0] < 70
    assert 150 < mid[1] < 200
