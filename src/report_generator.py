"""Post-analysis report generator.

Produces a text and JSON summary of skating analysis including:
- Session overview (video metadata, detection stats)
- Stride analysis (count, symmetry, per-side metrics)
- Mechanic ratings with drill recommendations
"""

import json
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.mechanics_engine import MechanicResult


class ReportGenerator:
    """Generates coaching reports from analysis results."""

    def __init__(self, drill_library_path: str = "config/drill_library.yaml"):
        """Load drill library.

        Args:
            drill_library_path: Path to drill recommendations YAML.
        """
        self.drills = {}
        path = Path(drill_library_path)
        if path.is_file():
            with open(path) as f:
                data = yaml.safe_load(f)
                self.drills = data.get("drills", {})

    def generate(
        self,
        video_path: str,
        video_meta: dict,
        frames_processed: int,
        frames_detected: int,
        stride_analysis,
        session_results: list[MechanicResult],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a text report.

        Args:
            video_path: Input video file path.
            video_meta: Dict with fps, width, height, duration_sec, frame_count.
            frames_processed: Total frames processed.
            frames_detected: Frames with successful pose detection.
            stride_analysis: StrideAnalysis from stride_detector.
            session_results: List of MechanicResult from engine.evaluate_session().
            output_path: If provided, save report to this file.

        Returns:
            Report as a formatted string.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  HOCKEY SKATING TECHNIQUE ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Session info
        lines.append(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Video:      {video_path}")
        lines.append(f"Resolution: {video_meta['width']}x{video_meta['height']} "
                      f"@ {video_meta['fps']:.0f} fps")
        lines.append(f"Duration:   {video_meta['duration_sec']:.1f}s")
        lines.append(f"Detection:  {frames_detected}/{frames_processed} frames "
                      f"({frames_detected / frames_processed * 100:.0f}%)")
        lines.append("")

        # Stride overview
        lines.append("-" * 60)
        lines.append("  STRIDE OVERVIEW")
        lines.append("-" * 60)

        if stride_analysis.total_strides == 0:
            lines.append("  No strides detected.")
            lines.append("")
        else:
            lines.append(f"  Total strides:    {stride_analysis.total_strides} "
                          f"(L: {len(stride_analysis.left_strides)}, "
                          f"R: {len(stride_analysis.right_strides)})")
            if stride_analysis.avg_stride_duration_sec is not None:
                lines.append(f"  Avg duration:     "
                              f"{stride_analysis.avg_stride_duration_sec:.2f}s")

            # Symmetry
            sym_results = [r for r in session_results if r.name == "symmetry"]
            if sym_results:
                s = sym_results[0]
                lines.append(f"  L/R Symmetry:     {s.value:.0%} "
                              f"[{s.rating.upper()}]")
            lines.append("")

            # Per-side breakdown
            for side_name, strides in [("LEFT", stride_analysis.left_strides),
                                        ("RIGHT", stride_analysis.right_strides)]:
                if not strides:
                    continue
                side_key = side_name.lower()
                lines.append(f"  {side_name} LEG:")

                side_results = [r for r in session_results if r.side == side_key]
                metrics = {}
                for r in side_results:
                    metrics.setdefault(r.name, []).append(r)

                for metric_name, metric_results in metrics.items():
                    values = [r.value for r in metric_results]
                    avg_val = np.mean(values)
                    ratings = [r.rating for r in metric_results]
                    poor_count = ratings.count("poor")
                    warn_count = ratings.count("warning")
                    good_count = ratings.count("good")
                    total = len(ratings)
                    display = metric_results[0].display_name.rsplit(" (", 1)[0]

                    # Overall rating for this metric
                    if poor_count / total > 0.3:
                        overall = "POOR"
                    elif (warn_count + poor_count) / total > 0.4:
                        overall = "WARNING"
                    else:
                        overall = "GOOD"

                    line = f"    {display:.<25s} {avg_val:>5.0f} avg  [{overall}]"
                    lines.append(line)

                    if overall != "GOOD":
                        lines.append(f"      {good_count}/{total} good, "
                                      f"{warn_count}/{total} warning, "
                                      f"{poor_count}/{total} poor")
                lines.append("")

        # Drill recommendations
        issues = self._find_issues(session_results)
        if issues:
            lines.append("-" * 60)
            lines.append("  RECOMMENDED DRILLS")
            lines.append("-" * 60)
            lines.append("")

            for mechanic_name, rating in issues:
                drill_config = self.drills.get(mechanic_name, {})
                drill_list = drill_config.get(rating, [])
                if not drill_list:
                    continue

                display = mechanic_name.replace("_", " ").title()
                lines.append(f"  For {display} ({rating.upper()}):")
                for drill in drill_list:
                    lines.append(f"    * {drill['name']}")
                    lines.append(f"      {drill['description']}")
                    if "cue" in drill:
                        lines.append(f"      Cue: \"{drill['cue']}\"")
                lines.append("")

        lines.append("=" * 60)
        lines.append("  Generated by Hockey Skating AI Analyzer")
        lines.append("=" * 60)

        report_text = "\n".join(lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_text)

        return report_text

    def generate_json(
        self,
        video_path: str,
        video_meta: dict,
        frames_processed: int,
        frames_detected: int,
        stride_analysis,
        session_results: list[MechanicResult],
        output_path: Optional[str] = None,
    ) -> dict:
        """Generate a JSON-serializable report dict.

        Same inputs as generate(). Returns a dict; optionally saves to file.
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "video": {
                "path": video_path,
                "width": video_meta["width"],
                "height": video_meta["height"],
                "fps": video_meta["fps"],
                "duration_sec": video_meta["duration_sec"],
            },
            "detection": {
                "frames_processed": frames_processed,
                "frames_detected": frames_detected,
                "detection_rate": frames_detected / frames_processed if frames_processed else 0,
            },
            "strides": {
                "total": stride_analysis.total_strides,
                "left_count": len(stride_analysis.left_strides),
                "right_count": len(stride_analysis.right_strides),
                "avg_duration_sec": stride_analysis.avg_stride_duration_sec,
                "symmetry_ratio": stride_analysis.symmetry_ratio,
            },
            "stride_details": [],
            "session_ratings": [],
            "drill_recommendations": [],
        }

        # Stride details
        for s in stride_analysis.left_strides + stride_analysis.right_strides:
            report["strides"]["total"]  # just accessing for validation
            report["stride_details"].append({
                "side": s.side,
                "frame_start": s.frame_start,
                "frame_end": s.frame_end,
                "push_off_angle": round(s.push_off_angle, 1),
                "glide_angle": round(s.glide_angle, 1),
                "rom": round(s.extension_range, 1),
                "duration_frames": s.duration_frames,
            })

        # Session ratings
        for r in session_results:
            report["session_ratings"].append({
                "name": r.name,
                "display_name": r.display_name,
                "value": round(r.value, 1),
                "rating": r.rating,
                "feedback": r.feedback,
                "side": r.side,
            })

        # Drill recommendations
        issues = self._find_issues(session_results)
        for mechanic_name, rating in issues:
            drill_list = self.drills.get(mechanic_name, {}).get(rating, [])
            for drill in drill_list:
                report["drill_recommendations"].append({
                    "for_mechanic": mechanic_name,
                    "severity": rating,
                    "drill_name": drill["name"],
                    "description": drill["description"],
                    "cue": drill.get("cue", ""),
                })

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

        return report

    def _find_issues(
        self, session_results: list[MechanicResult]
    ) -> list[tuple[str, str]]:
        """Find mechanics with warning or poor ratings.

        Returns deduplicated list of (mechanic_name, worst_rating) tuples.
        """
        issues = {}
        for r in session_results:
            if r.rating in ("warning", "poor"):
                current = issues.get(r.name)
                if current is None or (r.rating == "poor" and current == "warning"):
                    issues[r.name] = r.rating

        return list(issues.items())
