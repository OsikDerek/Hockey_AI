"""Hockey AI — Game-analysis pipeline + 3D viewer feeder (CLI entry point).

Multi-player tracking on broadcast or sports-cam game footage. Emits an
annotated video AND an NHL EDGE-shaped per-frame positions JSON that
drives the Three.js viewer at viewer/.

The body-mechanics half of the original Hockey_AI project was spun out
into a separate repo on 2026-05-11. That code now lives at:
    https://github.com/OsikDerek/Hockey_Vision_AI

Usage:
    python main.py --game-analysis -i data/raw_videos/clip.mp4
    python main.py --game-analysis --play-style possession -i clip.mp4
    python main.py --game-analysis --focus-team a -i my_team_clip.mp4
"""

import argparse
import os
import sys

from src.video_io import get_video_metadata
from src.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hockey AI — Game-analysis + 3D-viewer feeder",
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input game-film video")
    parser.add_argument("--output", "-o", default=None,
                        help="Annotated output video path "
                             "(default: output/<input_name>_analyzed.mp4)")
    parser.add_argument("--game-analysis", "-g", action="store_true",
                        help="(retained for compatibility — this is the only mode)")
    parser.add_argument("--play-style", default="balanced",
                        choices=["balanced", "possession", "physical",
                                 "speed", "defensive"],
                        help="Team play-style preset for decision evaluation "
                             "bias (default: balanced)")
    parser.add_argument("--overlays", default="minimal",
                        choices=["minimal", "full", "off"],
                        help="Overlay preset (default: minimal — only "
                             "reliably-correct features)")
    parser.add_argument("--show", default="",
                        help="Comma-separated overlay features to enable "
                             "(e.g. ambient_connections,goalie_sight_lines)")
    parser.add_argument("--hide", default="",
                        help="Comma-separated overlay features to suppress")
    parser.add_argument("--decision-conf", type=float, default=0.7,
                        help="Minimum decision confidence to render banners "
                             "(default: 0.7). All events stay in the report "
                             "regardless; this only gates video overlays.")
    parser.add_argument("--focus-team", default="both",
                        choices=["a", "b", "both"],
                        help="Show overlays only for events involving the "
                             "chosen team (default: both). Use --focus-team a "
                             "to review your own film and suppress opposition.")
    parser.add_argument("--focus-jersey", default=None,
                        help="Resolve focus team by jersey color description "
                             "(e.g. 'dark', 'white', 'red'). Overrides "
                             "--focus-team when both are passed.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Make --game-analysis optional now since it's the only mode.
    args.game_analysis = True

    if not os.path.isfile(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)

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

    from src.game_analysis import run_game_analysis_mode
    from src.game_analysis.game_annotator import resolve_overlay_config

    show_list = [s for s in (args.show or "").split(",") if s.strip()]
    hide_list = [h for h in (args.hide or "").split(",") if h.strip()]
    args.overlay_config = resolve_overlay_config(
        preset=args.overlays, show=show_list, hide=hide_list,
    )
    run_game_analysis_mode(args, meta)


if __name__ == "__main__":
    main()
