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
from .play_evaluator import PlayEvaluator
from .lane_calculator import LaneCalculator
from .game_annotator import GameAnnotator
from .team_classifier import TeamClassifier
from .game_report import GameReportGenerator
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

    # Play evaluator with style bias
    play_style = getattr(args, "play_style", "balanced")
    evaluator = PlayEvaluator(
        knowledge_base_dir="knowledge_base/game_situations",
        play_style=play_style,
    )
    report_gen = GameReportGenerator(play_style_name=play_style)

    annotator = GameAnnotator(
        show_boxes=True,
        show_zone=True,
        show_ids=True,
    )

    team_classifier = TeamClassifier(warmup_frames=30)

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
    team_data = []   # Store team assignments per frame for pass 2

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

        # Team classification
        if is_gameplay:
            team_assignments = team_classifier.update(frame, players)
        else:
            team_assignments = {}

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
        team_data.append(team_assignments)
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

    # Team classification summary
    team_summary = team_classifier.summary()
    if team_summary["is_ready"]:
        print(f"  Teams: A={team_summary['team_a_count']} players, "
              f"B={team_summary['team_b_count']} players")
    else:
        print("  Teams: not enough data to classify")

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

    # Evaluate all events
    if analysis.events:
        print("  Evaluating decisions...")
        evaluator.evaluate_all(analysis.events)
        ratings = {"good": 0, "warning": 0, "poor": 0, "neutral": 0}
        for e in analysis.events:
            if e.evaluation:
                ratings[e.evaluation.get("rating", "neutral")] += 1
        print(f"  Ratings: {', '.join(f'{k}={v}' for k, v in ratings.items() if v > 0)}")

    # ── Build Event Timeline ─────────────────────────────
    event_timeline = {}  # frame_idx -> phase_info
    for event in analysis.events:
        fs, fe, fi = event.frame_start, event.frame_end, event.frame_idx

        # Approach: event_start to decision-5
        for f in range(fs, max(fs, fi - 5)):
            progress = (f - fs) / max(1, fi - 5 - fs)
            event_timeline.setdefault(f, {
                "phase": "approach", "event": event,
                "alpha": 0.3 + 0.4 * progress,
            })

        # Slowdown: decision-5 to decision
        for f in range(max(fs, fi - 5), fi):
            event_timeline.setdefault(f, {
                "phase": "slowdown", "event": event, "alpha": 0.8,
            })

        # Freeze: decision frame
        event_timeline.setdefault(fi, {
            "phase": "freeze", "event": event, "alpha": 1.0,
        })

        # Result: decision+1 to event_end
        result_len = max(1, fe - fi)
        for f in range(fi + 1, fe + 1):
            fade = (f - fi) / result_len
            event_timeline.setdefault(f, {
                "phase": "result", "event": event,
                "alpha": max(0.1, 0.8 - 0.7 * fade),
            })

    # ── Pass 2: Render with Coaching Overlays ─────────────
    print("\nPass 2: Rendering annotated video with coaching overlays...")
    render_start = time.time()

    out_h, out_w = meta["height"], meta["width"]
    fps = meta["fps"]
    freeze_duration = 2.0  # seconds to hold on decision frame

    with video_writer(args.output, fps=fps, width=out_w, height=out_h) as writer:
        for i, (frame, ctx) in enumerate(zip(frame_data, analysis.frame_contexts)):
            events = analysis.events_at_frame(i)
            annotator.set_team_assignments(team_data[i])

            phase_info = event_timeline.get(i)

            # Split players into teams for lane calculation
            carrier_id = ctx.possession_player_id
            teammates, opponents = _split_teams(
                ctx.players, carrier_id, team_data[i]
            )

            # Ambient connections (always-on when there's a carrier)
            ambient = None
            if carrier_id is not None and teammates:
                carrier_obj = _find_player(ctx.players, carrier_id)
                if carrier_obj:
                    ambient = annotator.lane_calculator.calc_ambient_connections(
                        carrier_obj, teammates, opponents, out_w, out_h
                    )

            if phase_info is not None and annotator.coaching_mode:
                # Coaching mode: calculate full lanes
                event = phase_info["event"]
                eid = event.player_id or carrier_id
                carrier_obj = _find_player(ctx.players, eid)

                lane_data = None
                if carrier_obj:
                    # Re-split using event's player
                    ev_teammates, ev_opponents = _split_teams(
                        ctx.players, eid, team_data[i]
                    )
                    shooting = annotator.lane_calculator.calc_shooting_lane(
                        carrier_obj,
                        ctx.goalies[0] if ctx.goalies else None,
                        ev_opponents, out_w, out_h,
                    )
                    passing = annotator.lane_calculator.calc_passing_lanes(
                        carrier_obj, ev_teammates, ev_opponents, out_w, out_h,
                    )
                    lane_data = {"shooting": shooting, "passing": passing}

                annotated = annotator.render_coaching(
                    frame, ctx, events, phase_info, lane_data,
                )
            else:
                # Standard mode with ambient connections
                annotated = annotator.render(
                    frame, ctx, events if events else None,
                    ambient_connections=ambient,
                )

            # Ensure correct dimensions
            ah, aw = annotated.shape[:2]
            if aw != out_w or ah != out_h:
                annotated = cv2.resize(annotated, (out_w, out_h))

            # Timing control
            if phase_info is not None:
                phase = phase_info["phase"]
                if phase == "freeze":
                    repeat = int(fps * freeze_duration)
                elif phase == "slowdown":
                    repeat = 2
                else:
                    repeat = 1
            else:
                repeat = 1

            for _ in range(repeat):
                writer.write(annotated)

    render_time = time.time() - render_start
    total_time = time.time() - start_time

    print(f"  Render complete in {render_time:.1f}s")

    # ── Report ────────────────────────────────────────────
    print()
    text_report = report_gen.generate_text(analysis, args, meta)
    print(text_report)

    # Print event details to console
    if analysis.events:
        print("\nDECISION DETAILS:")
        for event in analysis.events:
            ts = f"{event.timestamp_sec:.1f}s"
            rating = ""
            if event.evaluation:
                rating = f" => {event.evaluation.get('rating', '?').upper()}"
            print(f"\n  [{ts}] {event.event_type}: {event.decision_made} "
                  f"(player #{event.player_id}){rating}")
            if event.evaluation:
                if event.evaluation.get("reasoning"):
                    print(f"    {event.evaluation['reasoning']}")
                if event.evaluation.get("alternative"):
                    print(f"    >> Suggestion: {event.evaluation['alternative']}")
                if event.evaluation.get("style_note"):
                    print(f"    >> {event.evaluation['style_note']}")

    print()
    print(f"  Total processing time: {total_time:.1f}s ({pass1_fps:.1f} fps tracking)")
    print(f"  Output saved to: {args.output}")

    # Save reports
    print()
    report_gen.save(analysis, args, meta)

    return analysis


def _split_teams(players, carrier_id, team_assignments):
    """Split players into teammates and opponents relative to the carrier."""
    carrier_team = team_assignments.get(carrier_id)
    teammates = []
    opponents = []
    for p in players:
        if p.track_id == carrier_id:
            continue
        p_team = team_assignments.get(p.track_id)
        if carrier_team and p_team == carrier_team:
            teammates.append(p)
        elif carrier_team and p_team and p_team != carrier_team:
            opponents.append(p)
        else:
            # Unknown team — treat as opponent (safer)
            opponents.append(p)
    return teammates, opponents


def _find_player(players, track_id):
    """Find a player by track_id."""
    for p in players:
        if p.track_id == track_id:
            return p
    return None
