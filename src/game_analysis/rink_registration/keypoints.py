"""Canonical NHL rink keypoint vocabulary for learned rink registration.

The learned registration model (Phase A, 2026-05-21) detects a fixed,
ordered set of named rink keypoints in a broadcast frame. Each keypoint
has a KNOWN ice coordinate, so once the model locates even a subset of
them in pixel space we can solve the pixel->ice homography directly via
cv2.findHomography — no similarity-prior bootstrap, no dot-disambiguation
guesswork.

Ice coordinate convention (matches rink_calibrator.py):
    x in [0, 200] ft  — goal line to goal line (length)
    y in [0, 85]  ft  — board to board (width)
    origin at a corner.

NHL rink spec used here:
    length 200 ft, width 85 ft, corner radius 28 ft
    goal lines 11 ft from each end board    -> x = 11, 189
    blue lines 25 ft from centre            -> x = 75, 125
    centre red line                         -> x = 100
    end-zone faceoff dots 20 ft from goal line, 22 ft apart laterally
    neutral-zone faceoff dots 5 ft from each blue line, 22 ft apart
    faceoff circles 30 ft diameter (15 ft radius)
    goal mouth 6 ft wide

The ORDER of KEYPOINTS is the model's output channel order — never
reorder; only append. Each entry: (name, ice_x, ice_y).
"""

from __future__ import annotations

# Rink geometry (feet)
RINK_LENGTH_FT = 200.0
RINK_WIDTH_FT = 85.0
CENTER_X = 100.0
CENTER_Y = 42.5
GOAL_LINE_X = (11.0, 189.0)
BLUE_LINE_X = (75.0, 125.0)
RED_LINE_X = 100.0
FACEOFF_LATERAL = 22.0          # dots are CENTER_Y +/- 22 = 20.5 / 64.5
DOT_Y = (CENTER_Y - FACEOFF_LATERAL, CENTER_Y + FACEOFF_LATERAL)  # 20.5, 64.5
EZ_DOT_X = (31.0, 169.0)        # 20 ft from each goal line
NZ_DOT_X = (80.0, 120.0)        # 5 ft from each blue line
CIRCLE_RADIUS = 15.0
GOAL_HALF_WIDTH = 3.0           # 6 ft mouth


# ── Keypoint table ──────────────────────────────────────────────────
# (name, ice_x, ice_y). Order is the model output-channel order.
KEYPOINTS: list[tuple[str, float, float]] = [
    # --- 9 faceoff dots ------------------------------------------------
    ("dot_center",        CENTER_X,      CENTER_Y),       # 0
    ("dot_nz_l_lo",       NZ_DOT_X[0],   DOT_Y[0]),       # 1
    ("dot_nz_l_hi",       NZ_DOT_X[0],   DOT_Y[1]),       # 2
    ("dot_nz_r_lo",       NZ_DOT_X[1],   DOT_Y[0]),       # 3
    ("dot_nz_r_hi",       NZ_DOT_X[1],   DOT_Y[1]),       # 4
    ("dot_ez_l_lo",       EZ_DOT_X[0],   DOT_Y[0]),       # 5
    ("dot_ez_l_hi",       EZ_DOT_X[0],   DOT_Y[1]),       # 6
    ("dot_ez_r_lo",       EZ_DOT_X[1],   DOT_Y[0]),       # 7
    ("dot_ez_r_hi",       EZ_DOT_X[1],   DOT_Y[1]),       # 8

    # --- line x board intersections (vertical painted lines x long boards)
    ("blue_l_board_lo",   BLUE_LINE_X[0], 0.0),           # 9
    ("blue_l_board_hi",   BLUE_LINE_X[0], RINK_WIDTH_FT), # 10
    ("red_c_board_lo",    RED_LINE_X,     0.0),           # 11
    ("red_c_board_hi",    RED_LINE_X,     RINK_WIDTH_FT), # 12
    ("blue_r_board_lo",   BLUE_LINE_X[1], 0.0),           # 13
    ("blue_r_board_hi",   BLUE_LINE_X[1], RINK_WIDTH_FT), # 14

    # --- centre circle top/bottom -------------------------------------
    ("center_circle_lo",  CENTER_X,      CENTER_Y - CIRCLE_RADIUS),  # 15
    ("center_circle_hi",  CENTER_X,      CENTER_Y + CIRCLE_RADIUS),  # 16

    # --- end-zone faceoff circle top/bottom (4 circles x 2) -----------
    ("circ_ez_l_lo_top",  EZ_DOT_X[0],   DOT_Y[0] - CIRCLE_RADIUS),  # 17
    ("circ_ez_l_lo_bot",  EZ_DOT_X[0],   DOT_Y[0] + CIRCLE_RADIUS),  # 18
    ("circ_ez_l_hi_top",  EZ_DOT_X[0],   DOT_Y[1] - CIRCLE_RADIUS),  # 19
    ("circ_ez_l_hi_bot",  EZ_DOT_X[0],   DOT_Y[1] + CIRCLE_RADIUS),  # 20
    ("circ_ez_r_lo_top",  EZ_DOT_X[1],   DOT_Y[0] - CIRCLE_RADIUS),  # 21
    ("circ_ez_r_lo_bot",  EZ_DOT_X[1],   DOT_Y[0] + CIRCLE_RADIUS),  # 22
    ("circ_ez_r_hi_top",  EZ_DOT_X[1],   DOT_Y[1] - CIRCLE_RADIUS),  # 23
    ("circ_ez_r_hi_bot",  EZ_DOT_X[1],   DOT_Y[1] + CIRCLE_RADIUS),  # 24

    # --- goal posts (goal mouth on the goal line) ---------------------
    ("goal_l_post_lo",    GOAL_LINE_X[0], CENTER_Y - GOAL_HALF_WIDTH),  # 25
    ("goal_l_post_hi",    GOAL_LINE_X[0], CENTER_Y + GOAL_HALF_WIDTH),  # 26
    ("goal_r_post_lo",    GOAL_LINE_X[1], CENTER_Y - GOAL_HALF_WIDTH),  # 27
    ("goal_r_post_hi",    GOAL_LINE_X[1], CENTER_Y + GOAL_HALF_WIDTH),  # 28
]

NUM_KEYPOINTS = len(KEYPOINTS)
KEYPOINT_NAMES = [k[0] for k in KEYPOINTS]
KEYPOINT_ICE_XY = [(k[1], k[2]) for k in KEYPOINTS]
NAME_TO_INDEX = {name: i for i, (name, _, _) in enumerate(KEYPOINTS)}


def ice_xy(index: int) -> tuple[float, float]:
    """Ice (x, y) in feet for the keypoint at the given channel index."""
    return KEYPOINT_ICE_XY[index]


def as_arrays():
    """Return (names, ice_xy_array) — ice_xy_array shape (NUM_KEYPOINTS, 2)."""
    import numpy as np
    return KEYPOINT_NAMES, np.array(KEYPOINT_ICE_XY, dtype=np.float64)
