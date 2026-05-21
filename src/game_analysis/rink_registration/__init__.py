"""Learned NHL rink registration (Phase A, 2026-05-21).

A keypoint-detection model that locates named rink keypoints in a
broadcast frame; the calibrator then solves the pixel->ice homography
directly from the detected keypoints' known ice coordinates.

Replaces the classical similarity-bootstrap + dot-disambiguation chain,
which plateaued: it compressed the across-rink axis on fixed wide-pano
footage and calibrated only ~3.5% of frames on broadcast footage.

Modules:
    keypoints   — the canonical ordered keypoint vocabulary
    synth_data  — synthetic training-data generator (pinhole camera model)
    train       — training entry point (Phase 2)
    registration_model — inference wrapper: frame -> keypoints -> homography (Phase 3)
"""
