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

## Where we are (as of commit `fc64fe4` — May 2026 session)

This session's batches (top = most recent):

- **Position smoothing + Player-POV toggle in Quiz Mode** (`fc64fe4`):
  per-track EMA (alpha=0.30) over chronological ice_xy observations
  removes ±5 ft frame-to-frame jitter (16,675 obs smoothed on
  livebarn_cropped). Avatars now glide instead of twitching — critical
  for the new POV view. **Quiz Mode now has a Top-Down ↔ Player POV
  toggle** (buttons in the choice overlay or V hotkey). Top-down for
  tactical pattern-reading; POV for the end-goal experience: see what
  the carrier saw, decide what you'd have done, then reveal what they
  actually did + the AI's evaluation. Both views share the actor halo.
- **Track-ID stitching** (`cbbe4a9`): post-process JSON merges ByteTrack ghost
  IDs (252 → 79 unique tracks on livebarn_cropped). Three passes: within-frame
  dedup (drop double-detections within 4 ft), temporal stitching (merge tracks
  that "die" + are "born" within 90 frames + 15 ft + same team + same role),
  union-find rewrite of all track_id refs. Integrated into `_write_positions_json`.
  Preserves goalie-vs-player separation.
- **Avatar visible-cap + stale fix** (`37c54ec`): viewer caps visible avatars
  at 14 (5v5 + goalies + buffer); STALE_FRAMES dropped 5s → 1.5s. Defeats the
  "93 ghost avatars" symptom while underlying tracking is fixed.
- **Quiz Mode end-to-end fixes** (`0f6e6c5`, `8fc4188`, `e18a638`): scorecard
  fires when all events answered (was hanging); fixed Three.js crash from
  Quiz button accidentally being matched by `.camera-controls button`
  selector; auto-skip-uncalibrated was leaping over events sitting in
  uncalibrated stretches.
- **Quiz polish** (`2494e7a`, `53650de`, `d6607c7`): top-down camera during
  quiz pauses (was POV — too tactically illegible); end-of-clip scorecard
  with breakdown + replay button + restart; renderable-event filter (now
  detection-count based, was unique-track-count which broke after stitching).
- **Phase D Quiz** (`ecd461d`): pre-decision pause + POV + choice/reveal +
  hotkeys + score widget. Built on top of A2 confidence + C1 events JSON.
- **Self-verification harness** (`992b3d8` + Playwright in this session):
  `scripts/render_play_frames.py` and `scripts/quiz_simulate.py` for static
  diagnostics; `scripts/test_quiz_browser.py` drives a headless Chromium via
  Playwright to verify quiz UI end-to-end. Catches bugs that look fine in
  code review (e.g., the camera-button selector matching too greedily).
- **B2 partial / landmark specialist** (`b918c39`, `2f793a0`, `3497089`,
  `26dd89d`): added `models/landmarks_yolov8n.pt` trained on SHL data
  (mAP50 0.968), a CV-based blue/red line + boards detector, line×board
  intersection homography path. Net effect on rush: limited (model doesn't
  generalize from SHL to NHL broadcast). Net effect on cropped LiveBarn
  junior footage: minimal — main HockeyAI model handles it once the rink
  fills the frame.
- **Phase B1 + safety guards**: 8-DoF homography with degenerate-fit
  rejection (rink-quad area check, pixel-spread check). Honest about its
  ceiling: the YOLO landmark detector is the limiter, not the math.
- **Phase A3** (`4825e29`): `--focus-team {a,b,both}` + `--focus-jersey
  "<color>"` for self-coaching workflows.
- **Phase A2** (`82dd71f`): decision-confidence gating + tightened detector
  triggers. Default `--decision-conf 0.7` filters overlay events.
- **Phase C1** (`8c300ac`): viewer event timeline + 3D shot-recommendation
  arrow.

### Best test bed
**`livebarn_60sec_cropped.mp4`** (data/raw_videos/) — junior hockey shot via
LiveBarn fixed-cam pano lens, cropped to rink-only. 99.9% calibrated, 7/7
quiz events renderable, both goalies present, ~14 visible avatars per
quiz pause. The cropping is what unlocked it — raw uncropped pano view
gave 0% calibration because the rink was a small portion of frame.

### Current frontier
Avatar placement accuracy. The user is comparing rendered avatar positions
vs the actual film by eye. Stitching brought us from 200+ chaotic tracks
to ~79 stable identities, but tracking is still imperfect. Next
investigations: position smoothing per track (Kalman?), better team-
classifier on edge cases, possibly retraining HockeyAI on cropped junior
footage to improve player detection density.

### Headless verification workflow

```
.venv/Scripts/python.exe scripts/test_quiz_browser.py --clip <basename>
```

Drives a real Chromium via Playwright, exercises Quiz Mode, screenshots
each phase. Output to `output/_quiz_browser/`. Server must be running
(`scripts/serve_viewer.py`). The harness is genuinely useful for
verifying viewer/quiz changes without requiring user retest.

---

## Earlier history (before this session)


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

## Outstanding work / next likely batch (May 2026)

**Avatar placement accuracy** is the active frontier. Derek is doing
side-by-side comparison of rendered top-down vs the actual LiveBarn
film. Stitching + smoothing landed (`cbbe4a9`, `fc64fe4`); next
investigations:

1. **Spatial team-classification refinement** — on livebarn_cropped
   the gray-vs-gray clusters are ambiguous (junior teams often wear
   similar colors). Could augment with per-cluster centroid
   stability checks.
2. **Custom HockeyAI retrain on cropped junior footage** — the SHL
   specialist didn't generalize, but a small training set from the
   user's own clips might.
3. **POV camera refinement** — current POV is at avatar.position
   looking along velocity yaw. Could improve: head height tuning,
   stick-on-ice perspective, smoother yaw transitions, "look at the
   puck" alternative aim (vs velocity-direction aim).

**Phase D simulator (full version)**: branching playback ("if you'd
passed instead, here's how the play would have ended"), WebXR/VR view,
multi-shift comparison. Quiz framework + POV view shipped
(`ecd461d` + `fc64fe4`); next layer is alternative-outcome rendering.

## Things NOT to do

- **Push freely** — Derek opted into push-as-we-work routine on
  2026-05-08. Commit + push after each non-trivial change. Don't skip
  hooks or force-push without explicit ask.
- Don't burn time perfecting features silenced by overlay quality gates.
  The goal is honest output, not polished output.
- Don't propose new tracking models / pose estimators — we use
  HockeyAI YOLOv8 + ByteTrack + MediaPipe. The track-ID stitcher is
  the right place to fix tracking weakness, not model swaps.
- Don't write text reports — Derek dropped that as an objective on
  2026-05-08. Focus on the visualizer + simulator path.

## Test-bed summary

| Clip | Best for | Calibrated frames | Quiz events |
|---|---|---|---|
| `livebarn_60sec_cropped.mp4` | **Phase D Quiz, avatar viz** | 99.9% | 7/7 renderable |
| `rush_30sec_clip.mp4` | Detector triggering tests | 21% | 2/16 renderable |
| `shootout_60sec.mp4` | Goalie / shootout context | 6% | 0 quiz-eligible |
| `ig_1v1_beating_guys.mp4` | High event count | 0.25% | 1/30 renderable |
