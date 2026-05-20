# Hockey_AI — queued tasks for `continue-implementation`

This file is queued work for the 9am `continue-implementation` cron routine
(or any session that lands here without a specific user prompt). Pick up
the top item that's autonomous-safe — no permissions or user input — and
make progress. Mark items done by deleting them from this file (then
commit + push, per the push-as-we-work routine).

Active frontier: tracking accuracy. **As of 2026-05-12 the primary
test footage is NHL-broadcast-style clips, NOT junior LiveBarn pano**
— see "Footage strategy" in HANDOFF.md. Use `wpg_pp_60sec.mp4` /
`rush_30sec_clip.mp4` as test beds. `livebarn_60sec_cropped.mp4` is a
deferred hard-case. Server: `py -3.12 scripts/serve_viewer.py` or
`.venv/Scripts/python.exe scripts/serve_viewer.py`.

Recent calibration work (commits through `da128c6`): improved
line/board detection + foot-point projection lifted livebarn_cropped
y-p90 from 27.6 → 64.6 ft. Next: validate on NHL-broadcast footage,
which is the footage type the whole CV stack was built for.

## Queued (in priority order)

### 0. ★ Across-rink y-compression (calibration bug, BIG)
Diagnosed 2026-05-11 on livebarn_cropped: median across-rink player
y-range is **14.8 ft on an 85-ft-wide rink** (p90 = 27.6 ft). Real
hockey would be 50+ ft most frames. The detections aren't overlapping
(min pair distance > 5.9 ft on frame 258) but they're squashed into a
~15-ft band near the center line.

Likely cause: 64% of calibrated frames use the similarity-transform
fallback (no perspective recovery), only 36% use full 8-DoF homography.
The similarity transform can't recover across-rink depth on
broadcast/follow-cam footage. **This is the root cause of the
"avatars bunched in a circle" effect** Derek noticed — smoothing makes
it glide, but the coordinates are still compressed.

Action items (don't start unprompted — Derek may want to pair on this):
- Audit `RinkCalibrator` / homography pipeline in `src/` to understand
  why so many frames fall back to similarity. Likely the YOLO landmark
  detector isn't finding enough labeled points (faceoff dots, lines)
  per frame. b918c39 was a partial step in this direction.
- Per-frame: is there a way to interpolate / extrapolate the last good
  homography into similarity-only frames? Right now we throw away the
  perspective info.
- Worth re-running the pipeline with --debug-calibration to see what
  fraction of frames have what landmark counts.

### 1. Headless puck-placement QA across all 7 events
Extend `scripts/test_quiz_browser.py` (or write a sibling
`scripts/render_play_frames.py`-style script) that loads
livebarn_cropped, walks through each renderable quiz event, screenshots
the paused top-down + POV view at each, and writes the grid to
`output/_puck_qa/`. Goal: visual confirmation that exactly one puck
shows up at every decision moment, on or near the carrier.

### 3. Investigate the 2 uncalibrated frames in livebarn_cropped
1801 total, 1799 calibrated → 2 missing. Find their `frame_idx` and
inspect why calibration dropped. Likely an EMA reset or single-frame
detection gap. Fix may be trivial (gap-filling between calibrated
neighbors).

### 4. POV "look-at-puck" alternative aim (behind a flag)
Current POV camera in `viewer/camera.js` aims along the player's
velocity yaw. Add an alternative that aims at the tracked puck position
when one exists, falling back to velocity. Add a UI toggle in the quiz
overlay (`Aim: motion / puck`). Verify both modes via the headless
harness — screenshot each at the same event for A/B.

### 5. Team-classifier centroid stability on gray-vs-gray
On junior LiveBarn footage the two teams sometimes wear similar grays
and the team_a / team_b clusters can swap mid-clip. Investigate the
classifier in `src/` (grep for `team_a`, `cluster`, or `classify`).
Compute centroid drift per track over time on livebarn_cropped — if
centroids oscillate, propose a stickiness rule (require N consecutive
disagreements before flipping a track's team). Don't ship the rule
without a confirmation step from Derek; just write the analysis to
`output/_team_analysis.md`.

## Off-limits without user input
- **Don't propose new tracking models / pose estimators** — pipeline
  is HockeyAI YOLOv8 + ByteTrack + MediaPipe by design.
- **Don't write text reports** — Derek dropped that as an objective.
- **Don't run the full main.py pipeline on a new clip** without asking
  — it takes ~minutes and ties up the GPU.
- **Don't start the Unity stretch-goal branch** without explicit user
  approval (see "Stretch goals" below).

## Stretch goals (don't start unprompted)

### Unity (or Godot/Unreal) playable simulator
Derek's articulated north-star vision (2026-05-09): **VR-style hockey
decision-training sim loaded from his own shifts**. Take the
tracking-derived world state at a real decision moment, drop the user
into POV with a gamepad / VR controllers, and let them play out
alternative decisions. Repetition rebuilds neural pathways: retraining
bad habits, learning to scan-before-receive, growing hockey IQ in a
safe high-rep environment. "VR hockey, but the world is your actual
shifts."

This is the *answer* to "why are we building this and not just
analyzing film." Every current phase (calibration → tracking →
viewer → Quiz Mode) is a stepping stone toward this.

**When to actually start it:** when the upstream tracking quality is
high enough that loading a real shift into a playable world will feel
trustworthy. We're not there yet — current frontier is still
avatar-placement accuracy on livebarn_cropped. Until Derek gives the
green light, keep iterating on the browser-stack viewer.

## When this file is empty
Pick from the "Outstanding work / next likely batch" section in
`HANDOFF.md`, or just stop and wait for the next user-driven session.
