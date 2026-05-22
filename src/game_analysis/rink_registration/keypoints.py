"""HockeyRink 56-keypoint rink template - ice coordinates (NHL feet).

Auto-derived by scripts/derive_keypoint_ice_coords.py:
  - 38 keypoints identified by Derek (faceoff dots, line/board
    intersections, trapezoid, nets, creases) -> exact NHL coords.
  - The remaining 18 (hashmarks, circle edges, bench) propagated
    via the reconstructed-template homography.

Ice frame: 200 x 85 ft. x = end board to end board; y = side board
to side board; center (100, 42.5). Keypoint index = HockeyRink
pose-model channel.
"""

NUM_KEYPOINTS = 56

# (ice_x_ft, ice_y_ft) per keypoint channel.
KEYPOINT_ICE_XY = [
    (0.0, 56.5),  #  0 identified
    (0.0, 28.5),  #  1 identified
    (11.0, 85.0),  #  2 identified
    (11.0, 53.5),  #  3 identified
    (11.0, 45.5),  #  4 identified
    (11.0, 39.5),  #  5 identified
    (11.0, 31.5),  #  6 identified
    (11.0, 0.0),  #  7 identified
    (15.5, 46.5),  #  8 identified
    (15.5, 38.5),  #  9 identified
    (28.12, 77.81),  # 10 propagated
    (27.84, 51.2),  # 11 propagated
    (27.83, 34.44),  # 12 propagated
    (27.83, 5.55),  # 13 propagated
    (31.0, 64.5),  # 14 identified
    (31.0, 20.5),  # 15 identified
    (35.01, 76.15),  # 16 propagated
    (33.37, 51.29),  # 17 propagated
    (33.42, 35.29),  # 18 propagated
    (33.39, 5.02),  # 19 propagated
    (75.0, 85.0),  # 20 identified
    (75.0, 0.0),  # 21 identified
    (80.0, 64.5),  # 22 identified
    (80.0, 20.5),  # 23 identified
    (100.0, 85.0),  # 24 identified
    (100.0, 57.5),  # 25 identified
    (100.0, 42.5),  # 26 identified
    (100.0, 27.5),  # 27 identified
    (90.68, -0.84),  # 28 propagated
    (100.0, 10.0),  # 29 identified
    (100.0, 0.0),  # 30 identified
    (108.64, 0.49),  # 31 propagated
    (120.0, 64.5),  # 32 identified
    (120.0, 20.5),  # 33 identified
    (125.0, 85.0),  # 34 identified
    (125.0, 0.0),  # 35 identified
    (201.92, 80.65),  # 36 propagated
    (168.0, 50.86),  # 37 propagated
    (167.15, 34.57),  # 38 propagated
    (165.48, -0.18),  # 39 propagated
    (169.0, 64.5),  # 40 identified
    (169.0, 20.5),  # 41 identified
    (171.05, 75.2),  # 42 propagated
    (172.38, 50.86),  # 43 propagated
    (172.58, 34.54),  # 44 propagated
    (171.53, 0.8),  # 45 propagated
    (184.5, 46.5),  # 46 identified
    (184.5, 38.5),  # 47 identified
    (189.0, 85.0),  # 48 identified
    (189.0, 53.5),  # 49 identified
    (189.0, 45.5),  # 50 identified
    (189.0, 39.5),  # 51 identified
    (189.0, 31.5),  # 52 identified
    (189.0, 0.0),  # 53 identified
    (200.0, 56.5),  # 54 identified
    (200.0, 28.5),  # 55 identified
]

# Indices Derek identified directly (exact coords, highest trust).
IDENTIFIED_KEYPOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 32, 33, 34, 35, 40, 41, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

# Keypoints whose template ice coordinate is unreliable and must NOT feed the
# homography solve. Found by a per-keypoint reprojection-error audit on the
# caufield_trim broadcast (scripts: per-keypoint median error every frame):
#   - 28, 36, 39: template coords land OUTSIDE the rink bounds (y<0 or
#     x>200) -- auto-propagation errors in the original template build.
#   - 29: an "identified" keypoint reprojecting a median 5.4 ft off.
#   - 42: a propagated keypoint reprojecting a median 2.6 ft off.
# RANSAC already rejects most of these per frame; excluding them up front
# keeps the fit clean and frees the RANSAC outlier budget for real noise.
UNRELIABLE_KEYPOINTS = [28, 29, 36, 39, 42]
