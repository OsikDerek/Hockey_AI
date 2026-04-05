"""Game analysis module for hockey decision-making optimization.

Tracks all players, puck, and game objects across broadcast footage,
classifies zones, detects decision points, and suggests better plays.

Usage:
    python main.py --game-analysis -i game_clip.mp4 -o output/game_analyzed.mp4
"""

import os
import sys
import time

import cv2
import numpy as np

from .game_context import FrameContext, GameAnalysis, TrackedObject
from .game_tracker import GameTracker, GAME_OBJECT_CLASSES
from .zone_detector import ZoneDetector
from .broadcast_filter import BroadcastFilter
from .possession_detector import PossessionDetector
from .game_annotator import GameAnnotator
from .decisions import DECISION_REGISTRY


def run_game_analysis_mode(args, meta):
    """Main pipeline for game film analysis.

    Two-pass architecture (same as technique mode):
    Pass 1: Track objects, classify zones, detect events
    Pass 2: Render annotated video
    """
    from src.utils import ensure_dir, format_timestamp
    from src.video_io import frame_generator, video_writer

    print("=" * 60)
    print("GAME ANALYSIS MODE")
    print("=" * 60)
    print()

    # Initialize components
    print("Initializing game tracker (HockeyAI model)...")
    tracker = GameTracker(
        model_path=getattr(args, "game_model", "models/HockeyAI_model_weight.pt"),
        conf=0.25,
    )

    broadcast_filter = BroadcastFilter(
        cut_threshold=0.4,
        min_game_objects=3,
    )

    zone_detector = ZoneDetector(smoothing_window=15)
    possession_detector = PossessionDetector(
        proximity_threshold=120,
        hysteresis_window=8,
    )

    # Initialize decision detectors
    active_decisions = getattr(args, "decisions", None) or list(DECISION_REGISTRY.keys())
    decision_detectors = {}
    for name in active_decisions:
        if name in DECISION_REGISTRY:
            decision_detectors[name] = DECISION_REGISTRY[name]()
            print(f"  Decision detector: {name}")
    if not decision_detectors:
        print("  No decision detectors active")

    annotator = GameAnnotator(
        show_boxes=True,
        show_zone=True,
        show_ids=True,
    )

    # Determine output path
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join("output", f"{base}_game_analyzed.mp4")
    ensure_dir(os.path.dirname(args.output))

    print(f"Input: {args.input}")
    print(f"  Resolution: {meta['width']}x{meta['height']} @ {meta['fps']} fps")
    print(f"  Duration: {meta['frame_count'] / meta['fps']:.1f}s ({meta['frame_count']} frames)")
    print(f"Output: {args.output}")
    print()

    # ── Pass 1: Track + Classify ──────────────────────────
    print("Pass 1: Tracking objects and classifying zones...")
    start_time = time.time()
    frames_processed = 0
    analysis = GameAnalysis()
    frame_data = []  # Store frames for pass 2

    for frame_idx, frame in frame_generator(args.input):
        # Count game objects from previous frame for broadcast filter
        prev_game_objects = 0
        if analysis.frame_contexts:
            prev_ctx = analysis.frame_contexts[-1]
            prev_game_objects = len(prev_ctx.players) + len(prev_ctx.goalies) + (1 if prev_ctx.puck else 0)

        # Broadcast filter
        is_gameplay, is_camera_cut = broadcast_filter.process_frame(
            frame, num_game_objects=prev_game_objects
        )

        # Track all objects
        objects = tracker.process_frame(frame, is_camera_cut=is_camera_cut)

        # Classify objects
        puck = tracker.get_puck(objects)
        players = tracker.get_players(objects)
        goalies = tracker.get_goalies(objects)
        referees = tracker.filter_by_class(objects, "referee")
        rink_landmarks = tracker.get_rink_landmarks(objects)

        # Zone detection
        zone = zone_detector.update(objects, puck, meta["width"])

        # Possession detection
        possession_id = possession_detector.update(puck, players)

        # Build frame context
        ctx = FrameContext(
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / meta["fps"],
            objects=objects,
            players=players,
            puck=puck,
            goalies=goalies,
            referees=referees,
            rink_landmarks=rink_landmarks,
            zone=zone,
            possession_player_id=possession_id,
            is_gameplay=is_gameplay,
            is_camera_cut=is_camera_cut,
        )

        # Feed to decision detectors
        if is_gameplay:
            for det in decision_detectors.values():
                det.add_frame(ctx)

        analysis.frame_contexts.append(ctx)
        frame_data.append(frame)
        frames_processed += 1

        # Track stats
        if is_camera_cut:
            analysis.camera_cuts += 1
        if is_gameplay:
            analysis.gameplay_frames += 1
        if zone:
            analysis.zone_time[zone] = analysis.zone_time.get(zone, 0) + (1.0 / meta["fps"])

        # Progress
        if frames_processed % 100 == 0:
            elapsed = time.time() - start_time
            fps_actual = frames_processed / elapsed if elapsed > 0 else 0
            pct = frames_processed / meta["frame_count"] * 100 if meta["frame_count"] > 0 else 0
            ts = format_timestamp(frame_idx, meta["fps"])
            n_players = len(players)
            puck_str = "PUCK" if puck else "no puck"
            zone_str = zone or "?"
            print(f"  [{ts}] {pct:.0f}% ({fps_actual:.1f} fps) | "
                  f"{n_players} players, {puck_str}, zone={zone_str}")

    analysis.total_frames = frames_processed
    pass1_time = time.time() - start_time
    pass1_fps = frames_processed / pass1_time if pass1_time > 0 else 0

    print(f"\n  Pass 1 complete: {frames_processed} frames in {pass1_time:.1f}s ({pass1_fps:.1f} fps)")
    print(f"  Gameplay frames: {analysis.gameplay_frames}/{analysis.total_frames} "
          f"({analysis.gameplay_frames / max(analysis.total_frames, 1) * 100:.0f}%)")
    print(f"  Camera cuts: {analysis.camera_cuts}")

    # ── Decision Analysis ─────────────────────────────────
    print("\nAnalyzing decisions...")
    for name, det in decision_detectors.items():
        events = det.analyze(fps=meta["fps"])
        analysis.events.extend(events)
        if events:
            print(f"  {name}: {len(events)} events detected")
        else:
            print(f"  {name}: no events detected")

    print(f"  Total decision events: {len(analysis.events)}")

    # ── Pass 2: Render ────────────────────────────────────
    print("\nPass 2: Rendering annotated video...")
    render_start = time.time()

    out_h, out_w = meta["height"], meta["width"]

    with video_writer(args.output, fps=meta["fps"], width=out_w, height=out_h) as writer:
        for i, (frame, ctx) in enumerate(zip(frame_data, analysis.frame_contexts)):
            events = analysis.events_at_frame(i)
            annotated = annotator.render(frame, ctx, events if events else None)

            # Ensure correct dimensions
            ah, aw = annotated.shape[:2]
            if aw != out_w or ah != out_h:
                annotated = cv2.resize(annotated, (out_w, out_h))

            writer.write(annotated)

    render_time = time.time() - render_start
    total_time = time.time() - start_time

    print(f"  Render complete in {render_time:.1f}s")

    # ── Report ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("GAME ANALYSIS REPORT")
    print("=" * 60)
    summary = analysis.summarize()

    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Gameplay: {summary['gameplay_frames']} ({summary['gameplay_pct']:.0f}%)")
    print(f"  Camera cuts: {summary['camera_cuts']}")
    print()

    if summary['zone_time']:
        print("  ZONE TIME:")
        for zone, secs in sorted(summary['zone_time'].items()):
            print(f"    {zone}: {secs:.1f}s")
        print()

    if summary['total_events'] > 0:
        print(f"  DECISION EVENTS: {summary['total_events']}")
        for etype, count in summary['events_by_type'].items():
            print(f"    {etype}: {count}")
        print()
        for event in analysis.events:
            ts = f"{event.timestamp_sec:.1f}s"
            print(f"  [{ts}] {event.event_type}: {event.decision_made} "
                  f"(player #{event.player_id})")
            if event.context:
                for k, v in event.context.items():
                    print(f"      {k}: {v}")
    else:
        print("  DECISION EVENTS: None detected in this clip")

    print()
    print(f"  Total processing time: {total_time:.1f}s ({pass1_fps:.1f} fps tracking)")
    print(f"  Output saved to: {args.output}")

    # Save JSON report
    _save_game_report(analysis, args, meta)

    return analysis


def _save_game_report(analysis: GameAnalysis, args, meta):
    """Save game analysis report as JSON."""
    import json

    base = os.path.splitext(args.output)[0]
    json_path = f"{base}_report.json"

    report = {
        "video": {
            "input": args.input,
            "resolution": f"{meta['width']}x{meta['height']}",
            "fps": meta["fps"],
            "duration_sec": meta["frame_count"] / meta["fps"],
        },
        "analysis": analysis.summarize(),
        "events": [
            {
                "type": e.event_type,
                "timestamp": e.timestamp_sec,
                "frame": e.frame_idx,
                "decision": e.decision_made,
                "player_id": e.player_id,
                "context": e.context,
                "evaluation": e.evaluation,
            }
            for e in analysis.events
        ],
    }

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  Report saved to: {json_path}")
