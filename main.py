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

import cv2
import numpy as np

from src.video_io import frame_generator, video_writer, get_video_metadata
from src.pose_estimator import create_estimator
from src.angle_calculator import compute_all_angles
from src.smoothing import LandmarkSmoother
from src.mechanics_engine import MechanicsEngine
from src.annotator import SkatingAnnotator
from src.stride_detector import StrideDetector
from src.report_generator import ReportGenerator
from src.video_preprocessing import SkaterCropper
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
    parser.add_argument(
        "--auto-crop",
        action="store_true",
        help="Auto-detect and crop to skater (use for distant/wide shots)",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=720,
        help="Target height for auto-crop upscale (default: 720)",
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

    # Auto-crop preprocessor
    cropper = None
    if args.auto_crop:
        print("Initializing auto-crop (YOLO person detection)...")
        cropper = SkaterCropper(target_height=args.crop_height)

    # Stride detector collects angle data across all frames
    stride_detector = StrideDetector()

    # Process video
    # When auto-cropping, output dimensions are determined by the first crop.
    # We do a pre-scan of the first frame to get the output size.
    print("Processing video...")
    start_time = time.time()
    frames_processed = 0
    frames_detected = 0

    if cropper is not None:
        # Pre-scan first frame to determine output dimensions
        for _, first_frame in frame_generator(args.input):
            test_crop, _ = cropper.process_frame(first_frame)
            out_h, out_w = test_crop.shape[:2]
            cropper.reset()  # Reset so first frame is processed fresh
            break
    else:
        out_w, out_h = meta["width"], meta["height"]

    with video_writer(
        args.output,
        fps=meta["fps"],
        width=out_w,
        height=out_h,
    ) as writer:
        for frame_idx, frame in frame_generator(args.input):
            # Auto-crop if enabled
            if cropper is not None:
                frame, crop_info = cropper.process_frame(frame)

            # Pose estimation
            landmarks = pose_estimator.process_frame(frame)

            mechanic_results = None
            angles = {}

            if landmarks is not None:
                frames_detected += 1

                # Smooth keypoints
                if smoother is not None:
                    landmarks = smoother.update(landmarks)

                # Calculate angles
                angles = compute_all_angles(landmarks)

                # Evaluate mechanics
                mechanic_results = engine.evaluate(angles)

            # Feed angles to stride detector (empty dict if no detection)
            stride_detector.add_frame(angles)

            # Annotate frame
            annotated = annotator.render(frame, landmarks, mechanic_results)

            # Ensure consistent output size (auto-crop may vary slightly)
            ah, aw = annotated.shape[:2]
            if aw != out_w or ah != out_h:
                annotated = cv2.resize(annotated, (out_w, out_h))

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

    # Stride analysis
    stride_analysis = stride_detector.analyze(fps=meta["fps"])

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

    # Stride report
    session_results = []
    if stride_analysis.total_strides > 0:
        session_results = engine.evaluate_session(stride_analysis)
        print()
        print("STRIDE ANALYSIS")
        print(f"  Total strides: {stride_analysis.total_strides} "
              f"(L: {len(stride_analysis.left_strides)}, "
              f"R: {len(stride_analysis.right_strides)})")
        if stride_analysis.avg_stride_duration_sec is not None:
            print(f"  Avg stride duration: {stride_analysis.avg_stride_duration_sec:.2f}s")

        sym_results = [r for r in session_results if r.name == "symmetry"]
        if sym_results:
            s = sym_results[0]
            print(f"  L/R symmetry: {s.value:.0%} ({s.rating.upper()})")

        for side_name, strides in [("Left", stride_analysis.left_strides),
                                    ("Right", stride_analysis.right_strides)]:
            if not strides:
                continue
            side_key = side_name.lower()
            print(f"  {side_name} leg:")

            side_results = [r for r in session_results if r.side == side_key]
            metrics = {}
            for r in side_results:
                metrics.setdefault(r.name, []).append(r)

            for metric_name, metric_results in metrics.items():
                values = [r.value for r in metric_results]
                avg_val = np.mean(values)
                ratings = [r.rating for r in metric_results]
                poor_pct = ratings.count("poor") / len(ratings) * 100
                warn_pct = ratings.count("warning") / len(ratings) * 100
                display = metric_results[0].display_name.rsplit(" (", 1)[0]

                status = "GOOD"
                if poor_pct > 30:
                    status = "POOR"
                elif warn_pct + poor_pct > 40:
                    status = "WARNING"

                print(f"    {display}: {avg_val:.0f} avg ({status})")
                if status != "GOOD":
                    print(f"      -> {metric_results[0].feedback}")
    else:
        print("  No strides detected (video may be too short or skater not visible)")

    # Generate reports
    report_gen = ReportGenerator()
    report_base = os.path.splitext(args.output)[0]

    text_report = report_gen.generate(
        video_path=args.input,
        video_meta=meta,
        frames_processed=frames_processed,
        frames_detected=frames_detected,
        stride_analysis=stride_analysis,
        session_results=session_results,
        output_path=f"{report_base}_report.txt",
    )

    report_gen.generate_json(
        video_path=args.input,
        video_meta=meta,
        frames_processed=frames_processed,
        frames_detected=frames_detected,
        stride_analysis=stride_analysis,
        session_results=session_results,
        output_path=f"{report_base}_report.json",
    )

    print(f"\n  Reports saved to:")
    print(f"    {report_base}_report.txt")
    print(f"    {report_base}_report.json")

    pose_estimator.close()


if __name__ == "__main__":
    main()
