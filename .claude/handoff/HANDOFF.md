# Hockey_AI — Handoff context

This file exists so a Claude Code session on a different machine can pick
up exactly where we left off. Everything important is either in this
directory or in the repo's git history.

## TL;DR for the next Claude session

1. Read `.claude/handoff/memory/` — these are persistent memories about
   the user (Derek Osik, pro hockey player + skating coach), the project,
   and his collaboration preferences.
2. Read `.claude/handoff/plan.md` — the most recent plan (top of the file
   is the most recent batch; older plans retained underneath for history).
3. Run `git log --oneline -20` to see the last ~6 batches of work.
4. The viewer at `viewer/` is the current frontier. Phase B1 (full
   homography) is the most likely next batch.

## Where we are (as of commit `f7a4e4f`)

Phases shipped:
- **Phase A** (commit `1ed104b`): minimal-by-default overlay rendering
  with quality gates + shootout NON-GAMEPLAY hysteresis fix.
- **Phase B0** (commit `ce605a8`): ice-coordinate data layer. New
  `RinkCalibrator` produces a 2D similarity transform; every detection
  gets `ice_xy` (feet); per-frame positions JSON exported alongside the
  output video (NHL EDGE-shaped).
- **Phase B0.5 + C0** (commit `deaca39`): stabilized calibration (EMA
  smoothing, outlier rejection, side disambiguation, camera-cut reset)
  + a Three.js 3D viewer in `viewer/` with top-down / broadcast-side /
  POV cameras + scrubber + auto-loaded JSON via
  `scripts/serve_viewer.py`.
- **Viewer fixes** (commit `f7a4e4f`): persist avatars across uncalibrated
  frames; foot discs for top-down readability; filename + calibration
  quality in status.

## Calibration status (the biggest open issue)

The `RinkCalibrator` uses a 2D similarity transform (translation + uniform
scale, no rotation, no perspective). On clips with a stable side camera
(like `ig_1v1_beating_guys.mp4`) it scores **100% in-rink**. On
broadcast follow-cam clips (like `rush_30sec_clip.mp4`) it scores **~2%**
because the perspective distortion can't be corrected by similarity.

**Phase B1** is the planned upgrade: full 8-DoF homography. The blocker is
that the YOLO model labels all 8 faceoff dots as a single class
(class_name = "faceoff") so we can't pick 4 known-correspondence points
without dot identification logic.

## Canonical test clips

In `data/raw_videos/`:
- `rush_30sec_clip.mp4` — broadcast follow-cam, dense action. 2.4% in-rink.
- `ig_1v1_beating_guys.mp4` — side-cam, 1v1 drill footage. 100% in-rink.
- `shootout_60sec.mp4` — flagged as unsuitable (broadcast fade transitions
  confuse the tracker). User to provide a cleaner shootout clip.
- `rush_30sec_clip.mp4` is the primary smoke-test clip historically
  even though calibration is poor on it.

## How to run things

Project root: `C:\Users\17742\OneDrive\Documents\PersonalProjects\Hockey_AI`
Python: `py` works on the original Windows machine. On the desktop,
verify with `py --version` or use the full path.

Generate positions JSON + annotated video:
```
py main.py --game-analysis -i data/raw_videos/ig_1v1_beating_guys.mp4 -o output/1v1.mp4
```

Launch the 3D viewer:
```
py scripts/serve_viewer.py
```
Auto-opens `http://localhost:8000/viewer/index.html?data=/output/<latest>_positions.json`.

## Restoring memory + plan on the new machine

The next Claude session on the desktop should run something like:
```
mkdir -p ~/.claude/projects/C--Users-<user>-...-Hockey-AI/memory
cp .claude/handoff/memory/* ~/.claude/projects/<...>/memory/

mkdir -p ~/.claude/plans
cp .claude/handoff/plan.md ~/.claude/plans/synthetic-mixing-turtle.md
```
The exact path on Windows is `%USERPROFILE%\.claude\projects\<project-id>\memory\`.
The project-id is derived from the absolute path; Claude will create the
directory automatically the first time it writes to memory, so an easier
approach is just to **paste this HANDOFF.md into the first Claude prompt**
on the new machine and let it write the memory files itself.

## Suggested first prompt on the desktop

> I'm continuing work on this Hockey_AI project from a different machine.
> Read `.claude/handoff/HANDOFF.md` and the contents of
> `.claude/handoff/memory/` and `.claude/handoff/plan.md`. Then save the
> memory files into your persistent memory and orient me on what we
> shipped last and what's next.

## Outstanding work / next likely batch

**Phase B1: full homography.** Need to disambiguate faceoff-dot class
into specific dots using their position relative to other detected
landmarks (goal lines, blue lines, centroid). Once we have 4 known
correspondences we can compute `cv2.findHomography()` instead of the
similarity transform. This should bring rush-clip calibration from 2%
into the 70-90% range.

**Phase C1: viewer polish.** With B1 calibration, the 3D viewer becomes
genuinely useful. Could add: decision overlays in 3D (shot
recommendation arrows pointing at open holes), multi-shift comparison,
playback timeline with event markers.

**Phase D: simulator.** Take a real shift's positions, freeze at decision
points, ask the user "what would you do?", branch on choice. WebXR for VR.

## Things NOT to do

- Don't push to remote without explicit ask — Derek prefers to be in the
  loop on git remote operations.
- Don't burn time perfecting features that are already silenced by the
  overlay quality gates. The goal is honest output, not polished output.
- Don't propose new tracking models or new pose estimators — we use
  HockeyAI YOLOv8 + ByteTrack + MediaPipe. Stick with those.
