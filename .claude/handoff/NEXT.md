# Hockey_AI — queued tasks for `continue-implementation`

This file is queued work for the 9am `continue-implementation` cron routine
(or any session that lands here without a specific user prompt). Pick up
the top item that's autonomous-safe — no permissions or user input — and
make progress. Mark items done by deleting them from this file (then
commit + push, per the push-as-we-work routine).

Last user-driven session: commit `1500f7b` (single-puck render fix).
Active frontier: avatar placement accuracy on livebarn_cropped, with
visual comparison in the viewer. Test bed: `livebarn_60sec_cropped.mp4`
→ `output/livebarn_cropped_positions.json` (97.5% in-rink, 7/7
renderable quiz events). Server: `py -3.12 scripts/serve_viewer.py` or
`.venv/Scripts/python.exe scripts/serve_viewer.py`.

## Queued (in priority order)

### 1. Goalie stick rendering parity
After the blade restyle in `viewer/avatar.js`, player sticks read as
sticks but goalies still only have a "pad" block in front of them. Give
goalies a goalie stick (wider paddle, lower angle, distinct color) so
the pose vocabulary is consistent. **Verify** with headless harness on
livebarn_cropped — goalies are visible at most quiz pauses.

### 2. Headless puck-placement QA across all 7 events
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
- **Don't build a Unity branch** — Derek and I discussed it on
  2026-05-09; the answer was to stay in the browser stack. (Notes in
  HANDOFF.md / memory.)

## When this file is empty
Pick from the "Outstanding work / next likely batch" section in
`HANDOFF.md`, or just stop and wait for the next user-driven session.
