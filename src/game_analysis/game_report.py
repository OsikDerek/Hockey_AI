"""Game analysis report generator.

Produces text and JSON reports from game analysis results,
including decision evaluations, zone time, and coaching suggestions.
"""

import json
import os
from datetime import datetime

from .game_context import GameAnalysis


class GameReportGenerator:
    """Generate game analysis reports in text and JSON formats."""

    def __init__(self, play_style_name: str = "balanced"):
        self.play_style_name = play_style_name

    def generate_text(self, analysis: GameAnalysis, args, meta) -> str:
        """Generate a text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("GAME ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Session info
        lines.append("SESSION INFO")
        lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"  Video: {args.input}")
        lines.append(f"  Resolution: {meta['width']}x{meta['height']} @ {meta['fps']} fps")
        lines.append(f"  Play Style: {self.play_style_name}")
        lines.append("")

        # Summary
        summary = analysis.summarize()
        lines.append("GAME OVERVIEW")
        lines.append(f"  Total frames: {summary['total_frames']}")
        lines.append(f"  Gameplay: {summary['gameplay_frames']} ({summary['gameplay_pct']:.0f}%)")
        lines.append(f"  Camera cuts: {summary['camera_cuts']}")
        lines.append("")

        # Zone time
        if summary['zone_time']:
            lines.append("ZONE TIME")
            for zone, secs in sorted(summary['zone_time'].items()):
                pct = secs / max(sum(summary['zone_time'].values()), 0.01) * 100
                lines.append(f"  {zone.capitalize()}: {secs:.1f}s ({pct:.0f}%)")
            lines.append("")

        # Decision events
        if analysis.events:
            lines.append("DECISION EVENTS")
            lines.append(f"  Total: {len(analysis.events)}")

            # Group by type
            by_type = {}
            for e in analysis.events:
                by_type.setdefault(e.event_type, []).append(e)

            for etype, events in by_type.items():
                lines.append(f"\n  --- {etype.upper().replace('_', ' ')} ({len(events)}) ---")

                # Rating distribution
                ratings = {"good": 0, "warning": 0, "poor": 0, "neutral": 0}
                for e in events:
                    if e.evaluation:
                        ratings[e.evaluation.get("rating", "neutral")] += 1

                rating_str = ", ".join(f"{k}: {v}" for k, v in ratings.items() if v > 0)
                lines.append(f"  Ratings: {rating_str}")

                # Individual events
                for e in events:
                    ts = f"{e.timestamp_sec:.1f}s"
                    decision = e.decision_made
                    player = f"player #{e.player_id}" if e.player_id else "unknown"

                    if e.evaluation:
                        rating = e.evaluation.get("rating", "?").upper()
                        lines.append(f"\n  [{ts}] {decision} by {player} => {rating}")
                        if e.evaluation.get("reasoning"):
                            lines.append(f"    {e.evaluation['reasoning']}")
                        if e.evaluation.get("alternative"):
                            lines.append(f"    Suggestion: {e.evaluation['alternative']}")
                        if e.evaluation.get("style_note"):
                            lines.append(f"    Style: {e.evaluation['style_note']}")
                    else:
                        lines.append(f"\n  [{ts}] {decision} by {player}")
        else:
            lines.append("DECISION EVENTS: None detected in this clip")

        lines.append("")

        # Coaching notes from knowledge base
        lines.append("COACHING NOTES")
        # These would come from the YAML files loaded by PlayEvaluator
        lines.append("  (See knowledge_base/game_situations/ for detailed notes)")
        lines.append("")

        return "\n".join(lines)

    def generate_json(self, analysis: GameAnalysis, args, meta) -> dict:
        """Generate a JSON report."""
        return {
            "generated_at": datetime.now().isoformat(),
            "play_style": self.play_style_name,
            "video": {
                "input": args.input,
                "resolution": f"{meta['width']}x{meta['height']}",
                "fps": meta["fps"],
                "duration_sec": meta["frame_count"] / meta["fps"],
            },
            "summary": analysis.summarize(),
            "events": [
                {
                    "type": e.event_type,
                    "timestamp_sec": e.timestamp_sec,
                    "frame": e.frame_idx,
                    "decision": e.decision_made,
                    "player_id": e.player_id,
                    "context": e.context,
                    "evaluation": e.evaluation,
                }
                for e in analysis.events
            ],
        }

    def save(self, analysis: GameAnalysis, args, meta):
        """Save both text and JSON reports."""
        base = os.path.splitext(args.output)[0]

        # Text report
        text_path = f"{base}_report.txt"
        text = self.generate_text(analysis, args, meta)
        with open(text_path, "w") as f:
            f.write(text)
        print(f"  Text report: {text_path}")

        # JSON report
        json_path = f"{base}_report.json"
        data = self.generate_json(analysis, args, meta)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  JSON report: {json_path}")
