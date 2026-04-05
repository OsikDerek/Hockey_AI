"""Hockey Skating Technique Analyzer — CLI Entry Point.

Processes hockey skating video through pose estimation, biomechanical
angle calculation, and coaching feedback, producing annotated output video.

Usage:
    # New technique-based analysis (uses knowledge base):
    python main.py --input video.mp4 --technique crossover
    python main.py --input video.mp4 --technique forward_stride

    # Legacy mode (still works):
    python main.py --input video.mp4 --mode general
    python main.py --input video.mp4 --mode crossover
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
from src.crossover_analyzer import CrossoverDetector as LegacyCrossoverDetector
from src.technique_engine import TechniqueEngine
from src.object_tracker import ObjectTracker
from src.report_generator import ReportGenerator
from src.video_preprocessing import SkaterCropper
from src.utils import ensure_dir, format_timestamp


TECHNIQUE_DIR = "knowledge_base/techniques"


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
        "--game-analysis", "-g",
        action="store_true",
        help="Game film analysis mode: multi-player tracking, zone detection, "
             "decision evaluation (broadcast or practice footage)",
    )
    parser.add_argument(
        "--technique", "-t",
        default=None,
        help="Technique to analyze (e.g., crossover, forward_stride). "
             "Loads from knowledge_base/techniques/<name>.yaml",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/skating_mechanics.yaml",
        help="Path to skating mechanics YAML config (legacy mode only)",
    )
    parser.add_argument(
        "--mode", "-m",
        default=None,
        choices=["general", "crossover"],
        help="Legacy analysis mode (use --technique instead)",
    )
    parser.add_argument(
        "--backend", "-b",
        default="mediapipe",
        choices=["mediapipe", "yolo"],
        help="Pose estimation backend (default: mediapipe)",
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


def run_technique_mode(args, meta):
    """New technique-based pipeline using knowledge base YAML."""
    technique_path = os.path.join(TECHNIQUE_DIR, f"{args.technique}.yaml")
    if not os.path.isfile(technique_path):
        print(f"Error: Technique file not found: {technique_path}")
        available = [f.replace(".yaml", "") for f in os.listdir(TECHNIQUE_DIR)
                     if f.endswith(".yaml")]
        print(f"Available techniques: {', '.join(available)}")
        sys.exit(1)

    print(f"Technique: {args.technique} ({technique_path})")
    print()

    # Initialize components
    print(f"Initializing pose estimator (backend: {args.backend})...")
    if args.backend == "mediapipe":
        pose_estimator = create_estimator("mediapipe", model_complexity=args.model_complexity)
    else:
        pose_estimator = create_estimator("yolo")

    smoother = None if args.no_smooth else LandmarkSmoother()
    technique_engine = TechniqueEngine(technique_path)

    show_angles = not args.no_angles and not args.skeleton_only
    show_hud = not args.no_hud and not args.skeleton_only

    annotator = SkatingAnnotator(
        show_skeleton=True,
        show_angles=show_angles,
        show_hud=show_hud,
        technique_engine=technique_engine,
    )

    # Object tracker (puck detection) — only if technique config requests it
    object_tracker = None
    det_params = technique_engine.config.get("detection", {}).get("params", {})
    if det_params.get("object_tracking", False):
        puck_model = det_params.get("puck_model", "models/puck_yolov8n.pt")
        puck_interval = det_params.get("puck_detection_interval", 2)
        print(f"Initializing object tracker (model: {puck_model})...")
        object_tracker = ObjectTracker(
            model_path=puck_model,
            detection_interval=puck_interval,
        )

    # Auto-crop preprocessor
    cropper = None
    if args.auto_crop:
        print("Initializing auto-crop (YOLO person detection)...")
        cropper = SkaterCropper(target_height=args.crop_height)

    # Process video
    print("Processing video...")
    start_time = time.time()
    frames_processed = 0
    frames_detected = 0

    if cropper is not None:
        for _, first_frame in frame_generator(args.input):
            test_crop, _ = cropper.process_frame(first_frame)
            out_h, out_w = test_crop.shape[:2]
            cropper.reset()
            break
    else:
        out_w, out_h = meta["width"], meta["height"]

    # Pass 1: Pose estimation + data collection
    frame_data = []
    if cropper is not None:
        cropper.reset()

    for frame_idx, frame in frame_generator(args.input):
        if cropper is not None:
            frame, crop_info = cropper.process_frame(frame)

        landmarks = pose_estimator.process_frame(frame)
        angles = {}

        if landmarks is not None:
            frames_detected += 1
            if smoother is not None:
                landmarks = smoother.update(landmarks)
            angles = compute_all_angles(landmarks)

        # Object tracking (puck detection) — merge metrics into angles
        if object_tracker is not None:
            object_metrics = object_tracker.process_frame(frame, landmarks)
            angles.update(object_metrics)

        technique_engine.add_frame(landmarks=landmarks, angles=angles)
        frame_data.append((frame, landmarks, angles))
        frames_processed += 1

        if frames_processed % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frames_processed / elapsed if elapsed > 0 else 0
            pct = frames_processed / meta["frame_count"] * 100 if meta["frame_count"] > 0 else 0
            timestamp = format_timestamp(frame_idx, meta["fps"])
            print(f"  [{timestamp}] {pct:.0f}% complete ({fps_actual:.1f} fps processing)")

    # Analyze
    print(f"  Analyzing {technique_engine.display_name}...")
    analysis = technique_engine.analyze(fps=meta["fps"])

    # Pass 2: Render annotated video
    print("  Rendering annotated video...")
    with video_writer(args.output, fps=meta["fps"], width=out_w, height=out_h) as writer:
        for frame_idx, (frame, landmarks, angles) in enumerate(frame_data):
            active_events = analysis.events_at_frame(frame_idx)

            if technique_engine.is_frame_by_frame:
                # Frame-by-frame: render current frame's event if it exists
                event = active_events[0] if active_events else None
                annotated = annotator.render_technique(frame, landmarks, event)
            else:
                # Temporal: show technique-specific feedback during events only
                annotated = annotator.render_technique(
                    frame, landmarks,
                    events=active_events if active_events else None,
                )
            ah, aw = annotated.shape[:2]
            if aw != out_w or ah != out_h:
                annotated = cv2.resize(annotated, (out_w, out_h))
            writer.write(annotated)

    # Summary
    elapsed = time.time() - start_time
    fps_actual = frames_processed / elapsed if elapsed > 0 else 0
    detection_rate = frames_detected / frames_processed * 100 if frames_processed > 0 else 0

    print()
    print(f"Done! Processed {frames_processed} frames in {elapsed:.1f}s")
    print(f"  Processing speed: {fps_actual:.1f} fps")
    print(f"  Skater detected in {frames_detected}/{frames_processed} frames ({detection_rate:.0f}%)")
    print(f"  Output saved to: {args.output}")

    # Technique-specific report
    _print_technique_report(analysis, meta)

    # Generate reports
    report_base = os.path.splitext(args.output)[0]
    _generate_technique_reports(analysis, args.input, meta, frames_processed, frames_detected, report_base)

    pose_estimator.close()


def _print_technique_report(analysis, meta):
    """Print technique analysis summary to console."""
    print()
    print(f"{analysis.display_name.upper()} ANALYSIS")

    if not analysis.events:
        print(f"  No {analysis.technique_name} events detected")
        return

    # For frame-by-frame: aggregate stats
    if len(analysis.events) > 100:
        # Many events = frame-by-frame, show aggregate
        all_results = []
        for event in analysis.events:
            all_results.extend(event.check_results)

        # Group by check name
        by_check = {}
        for r in all_results:
            by_check.setdefault(r.check_name, []).append(r)

        for check_name, results in by_check.items():
            values = [r.metric_value for r in results if r.metric_value is not None]
            if not values:
                continue
            avg_val = np.mean(values)
            ratings = [r.rating for r in results]
            poor_pct = ratings.count("poor") / len(ratings) * 100
            warn_pct = ratings.count("warning") / len(ratings) * 100

            if poor_pct > 30:
                status = "POOR"
            elif warn_pct + poor_pct > 40:
                status = "WARNING"
            else:
                status = "GOOD"

            display = results[0].display_name
            print(f"  {display}: {avg_val:.0f} avg ({status})")
            if status != "GOOD":
                print(f"    -> {results[0].feedback}")

        # Drill recommendations
        drills_shown = set()
        for event in analysis.events:
            for r in event.check_results:
                for drill in r.drills:
                    key = drill["name"]
                    if key not in drills_shown:
                        drills_shown.add(key)
                        print(f"  Drill: {drill['name']}")
                        print(f"    {drill['description']}")
                        if "cue" in drill:
                            print(f"    Cue: \"{drill['cue']}\"")
    else:
        # Few events = temporal (crossover, shot, etc.), show each
        print(f"  Total events: {len(analysis.events)}")
        for i, event in enumerate(analysis.events):
            t = event.frame_idx / meta["fps"]
            context_str = " | ".join(f"{k}={v}" for k, v in event.context.items()
                                      if isinstance(v, str))
            print(f"  Event #{i+1} at {t:.1f}s ({context_str}):")

            check_strs = []
            for r in event.check_results:
                check_strs.append(f"{r.display_name}: {r.rating.upper()}")
            print(f"    {' | '.join(check_strs)} => {event.overall_rating.upper()}")

            for fb in event.feedback:
                print(f"      -> {fb}")

    # Coaching notes
    if analysis.coaching_notes:
        print()
        print("  COACHING NOTES:")
        for note in analysis.coaching_notes:
            print(f"    - {note}")


def _generate_technique_reports(analysis, video_path, meta, frames_processed, frames_detected, report_base):
    """Generate text and JSON reports for technique analysis."""
    import json
    from datetime import datetime
    from pathlib import Path

    # Text report
    lines = [
        "=" * 60,
        f"  {analysis.display_name.upper()} TECHNIQUE ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Video:      {video_path}",
        f"Technique:  {analysis.display_name}",
        f"Detection:  {frames_detected}/{frames_processed} frames "
        f"({frames_detected / frames_processed * 100:.0f}%)",
        "",
    ]

    for i, event in enumerate(analysis.events[:50]):  # Cap at 50 for readability
        if len(analysis.events) > 100:
            break  # Skip per-event details for frame-by-frame
        t = event.frame_idx / meta["fps"]
        lines.append(f"Event #{i+1} at {t:.1f}s:")
        for r in event.check_results:
            lines.append(f"  {r.display_name}: {r.rating.upper()}")
            if r.rating != "good":
                lines.append(f"    -> {r.feedback}")
                for drill in r.drills:
                    lines.append(f"    Drill: {drill['name']} - {drill['description']}")
        lines.append("")

    if analysis.coaching_notes:
        lines.append("-" * 60)
        lines.append("  COACHING NOTES")
        lines.append("-" * 60)
        for note in analysis.coaching_notes:
            lines.append(f"  - {note}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("  Generated by Hockey Skating AI Analyzer")
    lines.append("=" * 60)

    text_path = f"{report_base}_report.txt"
    Path(text_path).parent.mkdir(parents=True, exist_ok=True)
    with open(text_path, "w") as f:
        f.write("\n".join(lines))

    # JSON report
    json_report = {
        "generated_at": datetime.now().isoformat(),
        "technique": analysis.technique_name,
        "video": {"path": video_path, **meta},
        "detection": {
            "frames_processed": frames_processed,
            "frames_detected": frames_detected,
        },
        "events": [],
        "coaching_notes": analysis.coaching_notes,
    }
    for event in analysis.events[:200]:
        json_report["events"].append({
            "frame_idx": event.frame_idx,
            "overall_rating": event.overall_rating,
            "context": {k: v for k, v in event.context.items() if isinstance(v, str)},
            "checks": [
                {
                    "name": r.check_name,
                    "display_name": r.display_name,
                    "value": round(r.metric_value, 3) if r.metric_value is not None else None,
                    "rating": r.rating,
                    "feedback": r.feedback,
                    "drills": r.drills,
                }
                for r in event.check_results
            ],
        })

    json_path = f"{report_base}_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    print(f"\n  Reports saved to:")
    print(f"    {text_path}")
    print(f"    {json_path}")


def run_legacy_mode(args, meta):
    """Original pipeline using MechanicsEngine (--mode flag)."""
    print(f"Config: {args.config}")
    print()

    print(f"Initializing pose estimator (backend: {args.backend})...")
    if args.backend == "mediapipe":
        pose_estimator = create_estimator("mediapipe", model_complexity=args.model_complexity)
    else:
        pose_estimator = create_estimator("yolo")

    smoother = None if args.no_smooth else LandmarkSmoother()
    engine = MechanicsEngine(config_path=args.config)

    show_angles = not args.no_angles and not args.skeleton_only
    show_hud = not args.no_hud and not args.skeleton_only
    annotator = SkatingAnnotator(show_skeleton=True, show_angles=show_angles, show_hud=show_hud)

    cropper = None
    if args.auto_crop:
        print("Initializing auto-crop (YOLO person detection)...")
        cropper = SkaterCropper(target_height=args.crop_height)

    stride_detector = StrideDetector()
    crossover_detector = LegacyCrossoverDetector()

    print("Processing video...")
    start_time = time.time()
    frames_processed = 0
    frames_detected = 0

    if cropper is not None:
        for _, first_frame in frame_generator(args.input):
            test_crop, _ = cropper.process_frame(first_frame)
            out_h, out_w = test_crop.shape[:2]
            cropper.reset()
            break
    else:
        out_w, out_h = meta["width"], meta["height"]

    frame_data = []
    if cropper is not None:
        cropper.reset()

    for frame_idx, frame in frame_generator(args.input):
        if cropper is not None:
            frame, crop_info = cropper.process_frame(frame)

        landmarks = pose_estimator.process_frame(frame)
        mechanic_results = None
        angles = {}

        if landmarks is not None:
            frames_detected += 1
            if smoother is not None:
                landmarks = smoother.update(landmarks)
            angles = compute_all_angles(landmarks)
            if args.mode == "general":
                mechanic_results = engine.evaluate(angles)

        if args.mode == "general":
            stride_detector.add_frame(angles)
        crossover_detector.add_frame(landmarks)
        frame_data.append((frame, landmarks, mechanic_results))
        frames_processed += 1

        if frames_processed % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frames_processed / elapsed if elapsed > 0 else 0
            pct = frames_processed / meta["frame_count"] * 100 if meta["frame_count"] > 0 else 0
            timestamp = format_timestamp(frame_idx, meta["fps"])
            print(f"  [{timestamp}] {pct:.0f}% complete ({fps_actual:.1f} fps processing)")

    if args.mode == "general":
        print("  Analyzing strides and crossovers...")
        stride_analysis = stride_detector.analyze(fps=meta["fps"])
    else:
        print("  Analyzing crossovers...")
        stride_analysis = stride_detector.analyze(fps=meta["fps"])
    crossover_analysis = crossover_detector.analyze(fps=meta["fps"])

    print("  Rendering annotated video...")
    with video_writer(args.output, fps=meta["fps"], width=out_w, height=out_h) as writer:
        for frame_idx, (frame, landmarks, mechanic_results) in enumerate(frame_data):
            active_crossovers = crossover_analysis.events_at_frame(frame_idx)

            if args.mode == "crossover":
                annotated = annotator.render(
                    frame, landmarks, None,
                    crossover_events=active_crossovers if active_crossovers else [],
                )
            else:
                annotated = annotator.render(
                    frame, landmarks, mechanic_results,
                    crossover_events=active_crossovers if active_crossovers else None,
                )

            ah, aw = annotated.shape[:2]
            if aw != out_w or ah != out_h:
                annotated = cv2.resize(annotated, (out_w, out_h))
            writer.write(annotated)

    elapsed = time.time() - start_time
    fps_actual = frames_processed / elapsed if elapsed > 0 else 0
    detection_rate = frames_detected / frames_processed * 100 if frames_processed > 0 else 0

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

    # Crossover report
    if crossover_analysis.total_crossovers > 0:
        print()
        print("CROSSOVER ANALYSIS")
        print(f"  Total crossovers: {crossover_analysis.total_crossovers} "
              f"(L over R: {len(crossover_analysis.left_over_right)}, "
              f"R over L: {len(crossover_analysis.right_over_left)})")
        avg_kd = crossover_analysis.avg_knee_drive_score()
        if avg_kd is not None:
            kd_label = "GOOD" if avg_kd >= 0.4 else "WARNING" if avg_kd >= 0.15 else "POOR"
            print(f"  Avg knee drive score: {avg_kd:.0%} ({kd_label})")
        for i, evt in enumerate(crossover_analysis.events):
            t = evt.frame_idx / meta["fps"]
            print(f"  Crossover #{i+1} at {t:.1f}s ({evt.crossing_leg} over {evt.stance_leg}):")
            print(f"    Knee drive: {evt.knee_drive_rating.upper()}"
                  f"  |  Rotation: {evt.rotation_rating.upper()}"
                  f"  |  Step-out: {evt.step_out_rating.upper()}"
                  f"  => {evt.overall_rating.upper()}")
            for fb in evt.feedback:
                if evt.overall_rating != "good":
                    print(f"      -> {fb}")
    else:
        print("  No crossovers detected")

    # Generate reports
    report_gen = ReportGenerator()
    report_base = os.path.splitext(args.output)[0]
    report_gen.generate(
        video_path=args.input, video_meta=meta,
        frames_processed=frames_processed, frames_detected=frames_detected,
        stride_analysis=stride_analysis, session_results=session_results,
        output_path=f"{report_base}_report.txt",
    )
    report_gen.generate_json(
        video_path=args.input, video_meta=meta,
        frames_processed=frames_processed, frames_detected=frames_detected,
        stride_analysis=stride_analysis, session_results=session_results,
        output_path=f"{report_base}_report.json",
    )
    print(f"\n  Reports saved to:")
    print(f"    {report_base}_report.txt")
    print(f"    {report_base}_report.json")

    pose_estimator.close()


def main():
    args = parse_args()

    # Validate input
    if not os.path.isfile(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)

    # Auto-generate output path
    if args.output is None:
        ensure_dir("output")
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"output/{base}_analyzed.mp4"

    ensure_dir(os.path.dirname(args.output) or ".")

    meta = get_video_metadata(args.input)
    print(f"Input: {args.input}")
    print(f"  Resolution: {meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps")
    print(f"  Duration: {meta['duration_sec']:.1f}s ({meta['frame_count']} frames)")
    print(f"Output: {args.output}")

    # Route to appropriate pipeline
    if args.game_analysis:
        from src.game_analysis import run_game_analysis_mode
        run_game_analysis_mode(args, meta)
    elif args.technique:
        run_technique_mode(args, meta)
    elif args.mode:
        run_legacy_mode(args, meta)
    else:
        # Default: technique mode with forward_stride
        args.technique = "forward_stride"
        run_technique_mode(args, meta)


if __name__ == "__main__":
    main()
