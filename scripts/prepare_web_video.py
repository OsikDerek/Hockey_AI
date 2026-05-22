"""Transcode a raw source video to a browser-friendly H.264 .web.mp4.

The raw videos in data/raw_videos/ often use the mpeg4 / Simple Profile
codec from sports-cam recorders (LiveBarn etc.), which browsers can't
reliably decode. The viewer's source-video panel auto-prefers a
sibling `<basename>.web.mp4` file produced by this script.

Usage:
    .venv/Scripts/python.exe scripts/prepare_web_video.py data/raw_videos/livebarn_60sec_cropped.mp4
    .venv/Scripts/python.exe scripts/prepare_web_video.py --all   # convert every .mp4 missing a sibling .web.mp4

Notes:
- Output goes next to the input as <basename>.web.mp4.
- Audio is dropped (-an) — the side-by-side compare doesn't need it.
- We use -preset fast / -crf 23, which is a good size/quality tradeoff
  for diagnostic playback. Roughly real-time on a modern CPU.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "data" / "raw_videos"


def transcode(src: Path) -> int:
    if not src.exists():
        print(f"  skip (missing): {src}")
        return 1
    dst = src.with_name(src.stem + ".web.mp4")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        print(f"  up to date: {dst.name}")
        return 0
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    print(f"transcoding {src.name} -> {dst.name}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ffmpeg failed:\n{r.stderr[-600:]}")
        return r.returncode
    return 0


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", help="video files to transcode")
    p.add_argument("--all", action="store_true",
                   help=f"transcode every .mp4 in {DEFAULT_DIR} that's missing a sibling .web.mp4")
    args = p.parse_args(argv)

    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not on PATH. Install or add to PATH.")
        return 2

    targets = []
    if args.all:
        for f in sorted(DEFAULT_DIR.glob("*.mp4")):
            if f.name.endswith(".web.mp4"):
                continue
            web = f.with_name(f.stem + ".web.mp4")
            if not web.exists():
                targets.append(f)
    elif args.paths:
        for p in args.paths:
            targets.append(Path(p).resolve())
    else:
        print("Provide paths or --all")
        return 2

    if not targets:
        print("Nothing to transcode.")
        return 0

    rc = 0
    for t in targets:
        rc |= transcode(t)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
