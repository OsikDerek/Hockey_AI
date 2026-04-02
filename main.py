"""Hockey Skating Technique Analyzer — CLI Entry Point.

Processes hockey skating video through pose estimation, biomechanical
angle calculation, and coaching feedback, producing annotated output video.

Usage:
    python main.py --input input_video/skating.mp4 --output output/analyzed.mp4
    python main.py --input input_video/skating.mp4  # auto-names output
"""

import argparse
import os
import sys
import time

from src.video_io import frame_generator, video_writer, get_video_metadata
from src.pose_estimator import create_estimator
from src.angle_calculator import compute_all_angles
from src.smoothing import LandmarkSmoother
from src.mechanics_engine import MechanicsEngine
from src.annotator import SkatingAnnotator
from src.utils import ensure_dir, format_timestamp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hockey Skating Technique Analyzer"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to output annotated video (default: output/<input_name>_analyzed.mp4)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/skating_mechanics.yaml",
        help="Path to skating mechanics YAML config",
    )
    parser.add_argument(
        "--backend", "-b",
        default="mediapipe",
        choices=["mediapipe", "yolo"],
        help="Pose estimation backend: mediapipe (CPU, 33 landmarks) or yolo (GPU, faster, multi-person) (default: mediapipe)",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="MediaPipe model complexity: 0=lite, 1=full, 2=heavy (default: 2)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable Kalman filter smoothing",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable HUD feedback panel overlay",
    )
    parser.add_argument(
        "--no-angles",
        action="store_true",
        help="Disable angle labels at joints",
    )
    parser.add_argument(
        "--skeleton-only",
        action="store_true",
        help="Only draw skeleton (no angles, no HUD)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    if not os.path.isfile(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)

    # Auto-generate output path if not specified
    if args.output is None:
        ensure_dir("output")
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"output/{base}_analyzed.mp4"

    # Ensure output directory exists
    ensure_dir(os.path.dirname(args.output) or ".")

    # Get video metadata
    meta = get_video_metadata(args.input)
    print(f"Input: {args.input}")
    print(
        f"  Resolution: {meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps"
    )
    print(f"  Duration: {meta['duration_sec']:.1f}s ({meta['frame_count']} frames)")
    print(f"Output: {args.output}")
    print(f"Config: {args.config}")
    print()

    # Initialize components
    print(f"Initializing pose estimator (backend: {args.backend})...")
    if args.backend == "mediapipe":
        pose_estimator = create_estimator(
            "mediapipe", model_complexity=args.model_complexity
        )
    else:
        pose_estimator = create_estimator("yolo")

    smoother = None if args.no_smooth else LandmarkSmoother()

    engine = MechanicsEngine(config_path=args.config)

    show_angles = not args.no_angles and not args.skeleton_only
    show_hud = not args.no_hud and not args.skeleton_only

    annotator = SkatingAnnotator(
        show_skeleton=True,
        show_angles=show_angles,
        show_hud=show_hud,
    )

    # Process video
    print("Processing video...")
    start_time = time.time()
    frames_processed = 0
    frames_detected = 0

    with video_writer(
        args.output,
        fps=meta["fps"],
        width=meta["width"],
        height=meta["height"],
    ) as writer:
        for frame_idx, frame in frame_generator(args.input):
            # Pose estimation
            landmarks = pose_estimator.process_frame(frame)

            mechanic_results = None

            if landmarks is not None:
                frames_detected += 1

                # Smooth keypoints
                if smoother is not None:
                    landmarks = smoother.update(landmarks)

                # Calculate angles
                angles = compute_all_angles(landmarks)

                # Evaluate mechanics
                mechanic_results = engine.evaluate(angles)

            # Annotate frame
            annotated = annotator.render(frame, landmarks, mechanic_results)

            # Write output
            writer.write(annotated)
            frames_processed += 1

            # Progress update every 100 frames
            if frames_processed % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = frames_processed / elapsed if elapsed > 0 else 0
                pct = (
                    frames_processed / meta["frame_count"] * 100
                    if meta["frame_count"] > 0
                    else 0
                )
                timestamp = format_timestamp(frame_idx, meta["fps"])
                print(
                    f"  [{timestamp}] {pct:.0f}% complete "
                    f"({fps_actual:.1f} fps processing)"
                )

    # Summary
    elapsed = time.time() - start_time
    fps_actual = frames_processed / elapsed if elapsed > 0 else 0
    detection_rate = (
        frames_detected / frames_processed * 100
        if frames_processed > 0
        else 0
    )

    print()
    print(f"Done! Processed {frames_processed} frames in {elapsed:.1f}s")
    print(f"  Processing speed: {fps_actual:.1f} fps")
    print(f"  Skater detected in {frames_detected}/{frames_processed} frames ({detection_rate:.0f}%)")
    print(f"  Output saved to: {args.output}")

    pose_estimator.close()


if __name__ == "__main__":
    main()
