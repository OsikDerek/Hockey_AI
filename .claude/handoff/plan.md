# Hockey AI — Phase B0.5 + C0: Stable Calibration + 3D Scene Viewer

## Context

Phase B0 produced per-frame ice-coordinate JSON for every detection (`<basename>_positions.json`, NHL EDGE-shaped). The user reviewed the minimap output and reports that it's not tracking the play effectively — the dots jump around and don't reflect smooth player motion. This must be fixed before Phase C lands, because the 3D viewer will magnify any calibration jitter (avatars teleporting in 3D space is much more jarring than dots jumping on a corner minimap).

Phase C consumes the now-stable JSON to render a Hockey-Verse-style 3D scene the user can interact with — the bridge to the simulator vision (Phase D, including VR via WebXR).

User's choices for this batch (Phase C portion):
- **Browser-based viewer** (not pre-rendered video) — interactive camera, future WebXR path.
- **All three camera modes equally polished**: top-down, broadcast-side, POV.
- **Stick-figure avatars with 3D limbs** — capsule body + stick + simple animated arms/legs. Acknowledged risk: we don't track joints in game-mode, so the limbs use a stylized procedural skating cycle synced to player velocity.
- **Ship working first, polish later**: crude-but-functional viewer with all three cameras + playback controls. No pose accuracy promises; no VR yet.

The shootout clip search came up empty for clean public clips. Testing uses rush_30sec_clip + ig_1v1_beating_guys. No new shootout testing this batch.

## Phase B0.5: Calibration stability fixes (bundled with C0)

The current calibrator re-fits from scratch each frame using whatever landmarks happen to be visible. With three different fit strategies (two goals / goal+centroid / single goal) producing wildly different scales, the transform jumps frame-to-frame. Fix it as follows:

- **Smooth the transform across frames**, not the landmark inputs. Maintain an exponentially-weighted moving average of `(scale_px_per_ft, origin_x, origin_y)`: blend the new fit with the previous transform (e.g., `0.85 * old + 0.15 * new`) instead of replacing outright. Rejects single-frame outliers, gives smooth playback.
- **Reject implausible single-frame fits.** Sanity checks that must pass before a fit updates the transform:
  - `scale_px_per_ft` within 50% of the previous value (camera doesn't zoom 2× in one frame).
  - The new origin pixel within 200 px of the previous (camera doesn't teleport).
  - For single-goal fallback, only accept if the goal's bbox aspect ratio is plausible (width > 0.7 * height — rejects far-zoom misdetections).
- **Disambiguate single-goal sides robustly.** Currently we use frame-half ("if goal x < width/2, it's the left goal"). That fails when the camera has panned to follow play to one end. Better: when we have a previous transform, predict where the goal SHOULD appear in pixel space for both possible sides and pick the side whose prediction is closest to the observed goal pixel.
- **On camera cut, hard-reset.** `BroadcastFilter` already detects cuts; on a cut, call `RinkCalibrator.reset()` so we don't blend a transform from one camera angle into another.
- **Drop the per-frame ice_xy assignment when not calibrated.** Already handled, but verify ghost coords aren't propagating into the JSON when `is_calibrated == False`.

These changes happen in `src/game_analysis/rink_calibrator.py` and the Pass 1 loop in `src/game_analysis/__init__.py` (camera-cut reset hook).

## Scope

### Goals
- A new `viewer/` directory (top-level under project root) containing a self-contained Three.js viewer.
- Load any `*_positions.json` file produced by the Python pipeline.
- Render an NHL rink in 3D (boards, blue lines, red center line, faceoff dots, goal creases, nets).
- Animate stick-figure avatars (capsule + stick + procedural limbs) for every player and goalie, plus a small puck mesh.
- Three camera modes selectable from on-screen UI: **Top-down**, **Broadcast-side**, **POV** (player-selectable).
- Playback controls: play/pause, scrubber, frame counter, current timestamp.
- Dropdown to pick which player's POV to use in POV mode.
- Auto-orient avatars by velocity (the direction they're skating).
- A Python helper to launch a static HTTP server pointed at the viewer directory + auto-open the latest `*_positions.json` (browsers require this for local file loads).

### Non-goals (deferred)
- WebXR / VR mode (Phase D).
- Decision-point overlays in 3D (e.g., shot recommendation arrows). Save for Phase C1.
- Multi-shift comparison / counterfactuals (Phase D).
- Real face/jersey textures. Stick figures only.
- Goalie-specific animations (the goalie is a fatter capsule for now).
- Puck height / arc rendering. Puck stays on ice surface (z=0); we don't track puck height.

## Directory layout (new)

```
viewer/
├── index.html              # main page, includes UI panel
├── viewer.css              # minimal styling
├── viewer.js               # entry point: scene setup, render loop
├── rink.js                 # NHL rink geometry builder
├── avatar.js               # stick-figure avatar factory + skating animation
├── camera.js               # top-down / broadcast / POV camera controllers
├── playback.js             # timeline scrubber, play/pause, frame indexing
├── data.js                 # JSON loader, frame interpolation
├── lib/
│   └── three.module.js    # Three.js (vendored, single file)
└── README.md               # how to run the viewer
```

`scripts/serve_viewer.py` (new): a tiny Python helper that starts `http.server` on port 8000, serves both the `viewer/` directory and `output/`, and prints the URL the user should open. Simpler than asking the user to start an HTTP server themselves.

## Files to add (no Python source modified)

### `viewer/index.html`
- Single-page app: a `<canvas>` for the Three.js renderer, a UI panel with camera-mode buttons, POV-player dropdown, playback controls, and a JSON-file picker. Imports `viewer.js` as a module.

### `viewer/viewer.js`
- Scene + renderer setup (WebGLRenderer, antialiasing, sky-blue background).
- Lights: hemisphere + directional with shadow map enabled (cheap, avatars cast soft shadows on the ice).
- Builds the rink via `rink.js` once on load.
- Maintains a registry of avatars keyed by `track_id` — adds/removes them as they appear/disappear in the JSON frames.
- Render loop: reads current frame from `playback`, interpolates positions toward the next frame for smoothness, updates avatar positions + orientations, updates puck mesh, renders.

### `viewer/rink.js`
- Builds an NHL-regulation rink (200 ft × 85 ft) using Three.js geometries:
  - Ice surface: white plane.
  - Boards: 4 rounded rectangles around the ice perimeter (just thin tan boxes for now).
  - Blue lines: thin blue rectangles on the ice at x=75 and x=125.
  - Center red line: thin red rectangle at x=100.
  - Center ice circle: red ring around the center dot.
  - Faceoff dots: 9 red filled discs at standard positions (4 end-zone, 4 neutral-zone, 1 center).
  - Goal creases: blue half-discs in front of each goal.
  - Goal nets: simple white wireframe boxes 6 ft wide × 4 ft tall × 4 ft deep at x=11 and x=189.
- Returns a `THREE.Group` containing all rink geometry, positioned with origin at ice corner (0, 0).

### `viewer/avatar.js`
- `createAvatar(team_color)` returns a `THREE.Group` containing:
  - Body: a vertical capsule (~5 ft tall, 1 ft radius), colored by team.
  - Head: a small sphere on top.
  - Stick: a thin cylinder extending forward from the body's right side, ~5 ft long.
  - Legs: 2 short cylinders animated in a simple 2-phase skating cycle (alternating forward/back) when the avatar's speed > threshold.
  - Arms: 2 short cylinders, both holding the stick (gripped near the top and middle).
- `updateAvatar(avatar, prev_pos, curr_pos, dt)` orients the avatar to face its velocity vector, advances the skating cycle by `speed * dt`, and updates limb rotations.

### `viewer/camera.js`
- `TopDownCamera`: orthographic camera looking straight down (-Z), framed to fit the rink (200×85 ft) with a small margin. Doesn't move.
- `BroadcastCamera`: perspective camera positioned at (~100, -50, 30) ft, looking at center ice. Approximates a standard broadcast follow-cam vantage.
- `POVCamera`: perspective camera attached to a chosen player avatar. Position = avatar head height (5.5 ft), forward = avatar facing direction. FOV ~75° for a natural human-ish field of view.
- Each camera exposes `apply(scene, currentPlayerAvatar)` that returns the active `THREE.Camera`.

### `viewer/playback.js`
- Reads the parsed JSON. Maintains current frame index, play/pause state, playback speed.
- On each render frame, advances the index by `playback_speed * (real_dt * fps_in_json)` so the viewer plays at "real time" by default.
- Handles play/pause, seek-to-frame, and the scrubber slider in the UI.
- When the JSON contains gaps (`calibrated: false` frames), holds the previous calibrated positions until a calibrated frame is available again.

### `viewer/data.js`
- Loads a JSON file (via a `<input type=file>` picker OR by URL fetch). Validates the structure (must have `rink`, `fps`, `frames`).
- Builds a per-track-id timeline: a sparse map `{track_id: [{frame_idx, ice_x, ice_y, team}]}` for fast lookup during playback.
- Exposes `getPositionsAt(frame_idx)` returning all entities with their positions for that frame, plus a `getPlayer(track_id, frame_idx)` for POV camera following.

### `viewer/viewer.css`
- Black UI panel pinned bottom-center: scrubber, play/pause, frame counter.
- Camera-mode buttons top-right.
- POV player dropdown appears next to the POV button when POV is active.

### `scripts/serve_viewer.py`
- Imports `http.server` + `socketserver`. Serves the project root on port 8000. Prints `http://localhost:8000/viewer/index.html?data=/output/_b0_rush_positions.json` so the user can open it directly. Auto-opens that URL with `webbrowser.open()`.
- Run it: `python scripts/serve_viewer.py`.

### `viewer/README.md`
- 5 lines: how to start the server, what JSON files to point it at, how to use camera buttons / POV picker.

## Critical functions / utilities to reuse

- The existing `_positions.json` shape (already produced by `_write_positions_json` in `src/game_analysis/__init__.py`). No changes to the Python side this batch — Phase C is pure consumer.
- The NHL rink constants from `src/game_analysis/rink_calibrator.py` are duplicated as JS constants in `viewer/rink.js` (small, won't drift).

## Verification

1. **Phase B0 regression**: re-run rush clip via `python main.py --game-analysis ...` and confirm `output/_b0_rush_positions.json` still produced. (No Python changes this batch, so this should be a no-op check.)
2. **Server starts**: `python scripts/serve_viewer.py` launches, prints URL.
3. **Browser load**: open URL, viewer loads `output/_b0_rush_positions.json` automatically. Rink renders. Avatars appear at correct ice positions. Playback runs at ~30 fps.
4. **Camera modes**: clicking each of the three camera buttons changes the view. Top-down shows the rink from above. Broadcast shows it from a side angle. POV picker has player track_ids; selecting one switches the camera to that player's head, and you can watch the play unfold from that player's perspective.
5. **Playback**: scrubber works. Play/pause works. Frame counter increments. Speed control (1×, 0.5×, 2×) works.
6. **Empty-data graceful degrade**: if user loads a file with no calibrated frames, viewer shows the empty rink with a message "No calibrated positional data in this clip" (or similar).

## Risks / watch items

- **Stick-figure limbs without pose data are an approximation.** The skating cycle is procedural; it won't match what the player is actually doing (e.g., a one-timer's wind-up looks identical to a cross-over). Acceptable for this batch — the visual cue of "moving avatar at correct ice position" carries the value, not the exact body pose. If it looks too jarring, we may simplify back to a capsule with no limbs in C1.
- **POV camera height**: 5.5 ft is a reasonable adult skater height; younger users may want it adjustable. For now, fixed.
- **Velocity from differences**: when calibration drops in/out, finite-difference velocity will spike. Smooth with a 5-frame moving average so POV camera doesn't snap.
- **Three.js bundle size**: vendoring `three.module.js` adds ~600 KB to the repo. Acceptable; alternative is loading from CDN at run time, but offline-first is more useful for this project.
- **Browser security on local file://**: that's why `scripts/serve_viewer.py` exists. Without it, `fetch()` on local JSON would CORS-block.
- **Calibration accuracy is the visible weakest link.** Players will appear in roughly correct positions but not pixel-accurate; near-side vs far-side bias from our 2D similarity transform will be visible from POV mode. Phase C1 + Phase B1 (full homography) close this gap.
- **Shootout testing skipped**: rush + 1v1 clips suffice. Once user has a clean shootout clip we'll re-verify.

## Below this line: previous (already-shipped) plans retained for reference

# Hockey AI — Phase B0: Ice-Coordinate Data Layer + Minimap + 3 Render Fixes

## Context

Phase A landed minimal-by-default overlay rendering. The user reviewed the outputs and identified three render-level issues plus is ready to begin Phase B (the bridge to Hockey-Verse-style 3D scene reconstruction):

1. **Player labels are noise.** The "A#146"/"B#5" text above each box is informational clutter. Replace with a colored bounding box outline (green = team_a, red = team_b, gray = unclassified). The team is now visible at-a-glance from the box itself.

2. **Freeze-frame pauses without explanation.** The pipeline duplicates the decision frame 60 times (2s @ 30fps) at every detected event regardless of overlay config. With `decision_freeze` overlay disabled (MINIMAL default), the video pauses with no on-screen reason. Slowdown duplicates frames 2x with the same problem. Fix: tie the duplication count to the corresponding overlay flag.

3. **Shootout clip is unusable** for testing — broadcast fade transitions confuse the tracker. User will provide a cleaner shootout clip in a follow-up; this batch acknowledges the issue and stops final-run testing on it.

Phase B0 (this batch) lays the **ice-coordinate data layer** that's the bridge to Phases C and D:

- A rink calibrator that estimates a pixel→ice transform from detected landmarks (goal, centroid, faceoff dots).
- An `ice_xy` (feet) coordinate computed for every tracked object per frame.
- A top-down minimap rendered as a corner overlay.
- A JSON export of per-frame positional data — same shape as NHL EDGE — so Phase C can consume it for 3D scene reconstruction without re-running CV.

The realistic constraint: full perspective homography from broadcast video is hard because the YOLO model labels all faceoff dots as a single class "faceoff" (we don't know which dot is which). Phase B0 uses a **2D similarity transform** (translation + rotation + uniform scale) anchored on the goal positions and center ice — robust given limited landmarks, not perspective-correct, but good enough for a usable minimap. A full homography is Phase B1 followup work.

## Scope

### Goals
- Replace the player label with team-colored box outlines.
- Suppress freeze/slowdown frame duplication when the corresponding overlay flag is off.
- Build a `RinkCalibrator` producing pixel→ice (feet) transforms.
- Render a top-down minimap as a new overlay flag (default ON in MINIMAL).
- Export per-frame positional data as JSON next to the output video.
- Acknowledge the shootout clip issue; deprioritize it from final canonical-run testing until the user provides a replacement.

### Non-goals (deferred)
- Full 8-DoF homography with perspective correction (Phase B1).
- Faceoff-dot identification (which-dot-is-which). For now treat all faceoffs as supplementary scale checks, not anchors.
- 3D scene reconstruction (Phase C).
- POV camera (Phase C).

## Files to modify

### `src/game_analysis/game_annotator.py`
- In `_draw_objects` for `obj.class_name == "player"`:
  - Determine box color from `team_color` (green for team_a, red for team_b — repurpose `TEAM_COLORS`) when `overlay.team_colors` AND `_can_show_team_colors_for(track_id)`. Otherwise gray (220, 220, 220) as today.
  - Draw the bounding box outline in that color (no separate stripe).
  - **Remove** the team-dot above the box and the "A#146" label entirely.
  - Track-id labels were nice for diagnostics — keep them ONLY when `show_ids` is explicitly True AND the track id is positive AND the user is running with `--overlays full` (gate on a new helper `_should_show_ids()` that returns `show_ids AND overlay.team_colors AND overlay.frame_info_hud`). This pushes IDs into the diagnostic preset, not the production preset.
- Update `TEAM_COLORS` to be vivid colors that read well as box outlines:
  - `team_a`: bright green (40, 220, 40)
  - `team_b`: bright red (40, 40, 220)
- Add new overlay key `minimap` (default True in MINIMAL_OVERLAYS).
- Add `_draw_minimap(frame, context, calibrator)` method that:
  - Draws an NHL rink template (boards as rounded rect, blue lines, red center, faceoff circles, goal creases) on a transparent overlay.
  - Places dots for each player (team-colored), goalies (team color with thicker border), puck (yellow).
  - Sized 30% of frame width, anchored bottom-right with a small inset margin, semi-transparent background.
  - Uses `calibrator.pixel_to_ice(point)` to translate detection centers to ice coords; clips off-rink positions.
- Wire minimap call into `render()` after frame_info HUD: `if ov.get("minimap") and self._calibrator: self._draw_minimap(annotated, context, self._calibrator)`.
- Add `set_calibrator(calibrator)` setter on the annotator (mirrors `set_classifier`).

### New file `src/game_analysis/rink_calibrator.py`
- NHL rink constants module-level: `RINK_LENGTH_FT = 200`, `RINK_WIDTH_FT = 85`, `GOAL_LINE_X = 11`, `BLUE_LINE_LEFT_X = 75`, `BLUE_LINE_RIGHT_X = 125`, `CENTER_X = 100`, `CENTER_Y = 42.5`. Goal positions in ice coords: `(11, 42.5)` and `(189, 42.5)`.
- `class RinkCalibrator`:
  - `update(rink_landmarks_dict: dict, frame_width: int, frame_height: int)` — called per frame. Accumulates goal x-positions (left/right cluster) and centroid x-position over the last ~100 frames (deque, like the existing `ZoneDetector` pattern). When ≥3 frames of goal+center data are accumulated:
    - Median-filter goal x-positions left and right.
    - Median-filter centroid x and y.
    - Compute pixel-per-foot scale from `(right_goal_x - left_goal_x) / (189 - 11)`.
    - Compute ice-origin pixel = position of `(0, 0)` in feet on the image.
    - Compute rotation = angle of the goal-to-goal axis on the image (small unless camera is heavily tilted).
    - Store as a similarity transform: `(origin_x, origin_y, scale_px_per_ft, rotation_rad)`.
  - `pixel_to_ice(pt: tuple) -> tuple` — applies inverse similarity to map pixel `(px, py)` → ice `(ix_ft, iy_ft)`. Returns None if not yet calibrated.
  - `ice_to_pixel(pt: tuple) -> tuple` — forward direction, useful for drawing things at ice coords on the broadcast frame.
  - `is_calibrated` property.
  - Periodic recalibration every 300 frames, matching `ZoneDetector` pattern (camera moves with play).
- Reuse the goal/centroid x-position accumulator pattern from `zone_detector.py::_update_calibration` (lines 83-130). Possibly factor out a small helper `_accumulate_landmark_x` shared by both.

### `src/game_analysis/game_context.py`
- Add `ice_xy: Optional[tuple] = None` to `TrackedObject` (defaulted to None for back-compat).
- Add `ice_calibrated: bool = False` to `FrameContext`.

### `src/game_analysis/__init__.py`
- Pass 1 (frame loop):
  - Instantiate `RinkCalibrator` once before the loop.
  - On each frame, call `calibrator.update(rink_landmarks, width, height)`.
  - For each `TrackedObject` in `objects`, set `obj.ice_xy = calibrator.pixel_to_ice(obj.center)` if calibrated.
  - Set `ctx.ice_calibrated = calibrator.is_calibrated`.
- Pass 2 (render loop):
  - `annotator.set_calibrator(calibrator)` once before the loop.
- New JSON output: after Pass 2, write `output/<basename>_positions.json` with shape:
  ```
  {
    "rink": {"length_ft": 200, "width_ft": 85},
    "fps": 30,
    "frames": [
      {
        "frame_idx": 0,
        "timestamp_sec": 0.0,
        "calibrated": true,
        "puck": {"ice_x": 95.2, "ice_y": 42.1, "confidence": 0.65},
        "players": [
          {"track_id": 5, "team": "team_a", "ice_x": 80.1, "ice_y": 38.0},
          ...
        ],
        "goalies": [...]
      },
      ...
    ]
  }
  ```
- Suppress freeze frame duplication when overlay disabled. In the timing-control block (currently around lines 407-417):
  ```
  if phase_info is not None:
      phase = phase_info["phase"]
      if phase == "freeze" and annotator.overlay.get("decision_freeze"):
          repeat = int(fps * freeze_duration)
      elif phase == "slowdown" and annotator.overlay.get("slowdown_indicator"):
          repeat = 2
      else:
          repeat = 1
  else:
      repeat = 1
  ```

### `main.py`
- No CLI changes needed. The `--overlays minimal` default will turn the minimap ON (since MINIMAL_OVERLAYS[`minimap`] = True). Users can `--hide minimap` to suppress.

## Critical functions / utilities to reuse

- `src/game_analysis/zone_detector.py::_update_calibration` — exact pattern for accumulating landmark x-positions in bounded deques and median-filtering. Mirror this in `RinkCalibrator`.
- `cv2.warpAffine` — for the simpler 2D similarity transform if we want a top-down warp visualization; not strictly needed for the minimap (we draw dots, not warp the image).
- `numpy.median` for landmark consensus.
- The NHL rink constants are documented in the existing `zone_detector.py` (the 0.72 ratio for blue lines was derived from these).

## Verification

After implementation, run all canonical clips except the shootout (which the user said is unusable):

```
python main.py --game-analysis -i data/raw_videos/rush_30sec_clip.mp4 -o output/rush_b0.mp4
python main.py --game-analysis -i data/raw_videos/ig_1v1_beating_guys.mp4 -o output/1v1_b0.mp4
```

Then extract frames and visually verify (using cv2 — ffmpeg unavailable):

For each output:
- **Player boxes**: green for one team, red for the other, gray for unclassified. No "A#146" labels above boxes.
- **Minimap**: visible bottom-right corner, ~30% width. Dots for players colored by team. Puck visible. Rink markings (blue lines, center red, faceoff circles, goal creases) present.
- **Freeze/slowdown**: the video should NOT pause at decision events anymore — playback flows continuously through events because `decision_freeze` and `slowdown_indicator` are off in MINIMAL.
- **JSON**: `output/rush_b0_positions.json` exists, has frames array, each frame has `calibrated`, `puck` if present, `players` list with `team` + `ice_x` + `ice_y`.

Also verify regression: `--overlays full` should still pause on freezes and show the panel. `--show minimap` plus a different preset should still render the minimap.

Acknowledge that `output/shootout_*.mp4` testing is paused pending a cleaner clip from the user.

## Risks / watch items

- **Similarity transform vs. homography**: the camera in broadcast hockey footage has perspective tilt (~5-15° typically). A 2D similarity transform will misplace players near the far boards by 5-15 ft. Acceptable for Phase B0 ("can you see this is roughly right?"), not acceptable for Phase D (simulator). Phase B1 will upgrade to full homography once we solve faceoff-dot identification.
- **Rink template drawing**: easy to over-engineer. Keep it simple — boards, two blue lines, center red line, two goal creases, 9 faceoff dots. Don't draw the trapezoid behind the net or the referee crease.
- **Calibrator drift**: like `ZoneDetector`, recalibrates every 300 frames. If the camera makes a sudden cut (which `BroadcastFilter` should already detect), reset the calibrator state.
- **JSON file size**: at 30 fps × 60s × ~15 objects × 5 fields each = manageable (~hundreds of KB). Don't write per-frame to disk; collect in memory and dump once at end.
- **Box-color change might look ugly with bright green/red**: 40,220,40 and 40,40,220 are a starting point; we'll tune after seeing extracted frames.

## Below this line: previous (already-shipped) plans retained for reference

# Hockey AI — Phase A: Strip Clutter + Quality-Gated Overlays + Shootout Hysteresis

## Context

The project is being recontextualized as a personal-development + future-simulator tool, not a broadcast-overlay tool. Primary users are now (a) young hockey players reviewing their own shifts and (b) future POV/simulator users. Hockey-Verse is the long-term aspiration; ice-coordinate positional data via homography (Phase B) is the bridge to that. Phase A is about making the existing CV output **honest and minimal** — silence underperforming features rather than displaying them at low quality, fix the remaining accuracy bugs, and prepare the architecture for Phases B-D.

The user reports four concrete problems with the current output:
1. **Inconsistent firing** — features fire at the wrong times (e.g., NON-GAMEPLAY stamping shootout frames when there's clearly gameplay)
2. **Inaccurate identification** — passing lanes wrong, team colors wrong, etc.
3. **Visual clutter** — too many overlays at once even when each is correct
4. **No way to silence individual features** — currently the renderer is all-or-nothing

The strategic decision: keep ALL existing features in code, but add **quality gating** + **per-feature on/off flags** + **a sane minimal default** so the output reflects only what we're actually confident in, frame-by-frame.

## Scope

### Goals
- Default output goes from "everything overlaid all the time" to "only confidently-correct features, minimal screen real estate."
- Each overlay feature gets an independent toggle (preserved code, just silenced by default).
- Each overlay feature gets a per-frame quality gate so even when enabled, it only fires when the underlying detection is reliable.
- Fix the shootout NON-GAMEPLAY bug (goalie dropout for one frame breaks `is_shootout_like`).

### Non-goals (deferred to later phases)
- Homography / ice-coordinate positional data (Phase B).
- Top-down minimap rendering (Phase B).
- 3D scene reconstruction or POV camera (Phase C).
- Simulator interaction loop (Phase D).
- Ripping out underperforming detection code (we keep it; we just silence it).

## Files to modify

### `src/game_analysis/game_annotator.py`
- Replace the current 6 constructor flags with a single `overlay_config: dict` parameter (default = MINIMAL preset). Provide module-level constants `MINIMAL_OVERLAYS` and `FULL_OVERLAYS` so callers don't have to spell out 14 keys.
- Per-feature flags (all default to `False` in MINIMAL, all `True` in FULL):
  - `boxes` (player boxes — kept ON in MINIMAL with gray-only color, no team stripe)
  - `team_colors` (the team-color stripe inside boxes)
  - `puck` (kept ON in MINIMAL)
  - `possession_indicator` (the "CARRIER" glow)
  - `zone_banner`
  - `ambient_connections` (carrier→teammate green/red lines)
  - `open_spaces`
  - `passing_lanes` (coaching-mode dashed lines)
  - `shooting_lane`
  - `decision_banner` (event chips top-left)
  - `decision_freeze` (vignette + suggestion panel on freeze frames)
  - `slowdown_indicator`
  - `goalie_sight_lines` (both shootout always-on AND event-driven freeze paths)
  - `frame_info_hud` (kept ON in MINIMAL — useful diagnostic)
  - `non_gameplay_stamp` (kept ON in MINIMAL but the bug fix below makes it fire correctly)
  - `camera_cut_indicator` (kept ON in MINIMAL)
- Wire each draw call to its flag. The current gating conditions (e.g., `if open_spaces and not context.is_shootout_like`) stay; the flag is an additional `and self.overlay.open_spaces` term.
- Add **quality gates** alongside the flags. A quality gate is a per-frame check that returns False when the underlying detection is too unreliable to render even if enabled:
  - `team_colors` quality: require `team_classifier.is_ready` AND `vote_counts[track_id]` ≥ 5 (i.e., the player has been seen and classified at least 5 times). Without this, randomly-assigned colors flicker.
  - `ambient_connections` quality: require team classification to be ready AND ALL teammates passed in have ≥3 votes. Otherwise the lines are connecting to mis-classified players.
  - `goalie_sight_lines` quality: require the chosen goalie to have a team assignment (goalies are now classified, but only after warmup).
  - `puck` quality: render solid only if `confidence ≥ 0.40`; ghost (dashed) for `0 < confidence < 0.40`; skip entirely if `confidence == 0` AND we're past coast frames (already handled, just confirming).
  - `passing_lanes` / `shooting_lane` quality: require ≥3 teammates / a goalie respectively, plus `team_classifier.is_ready`.

These quality gates live as small predicate methods on `GameAnnotator`, e.g. `_can_show_team_colors(context)`. The render path becomes `if self.overlay.team_colors and self._can_show_team_colors(context):`.

### `src/game_analysis/__init__.py`
- Pass the `team_classifier` reference through to the annotator (the annotator currently doesn't see classifier internals, only the assignment dict). The cleanest approach: add a setter `annotator.set_classifier(team_classifier)` called once before Pass 2. This gives the quality gates access to `vote_counts` and `is_ready`.
- Fix the **shootout NON-GAMEPLAY bug** with hysteresis on the Pass 1 detection:
  ```
  was_shootout_like = (analysis.frame_contexts[-1].is_shootout_like
                       if analysis.frame_contexts else False)
  current_match = (
      len(players) <= 2
      and len(goalies) >= 1
      and len(referees) <= 1
  )
  recent_match = (
      was_shootout_like
      and len(players) <= 2
      and len(referees) <= 1
  )  # tolerate goalie dropout for up to N frames if we were just in shootout
  is_shootout_like = current_match or recent_match
  ```
  Use a small streak counter (≤10 frames) to bound the hysteresis. When `current_match` fires we reset; when only `recent_match` fires we increment; once it hits 10 we let `is_shootout_like` go False.
- The instantiation of `GameAnnotator` in `run_game_analysis_mode` needs to switch to passing an `overlay_config`. Default to MINIMAL.

### `main.py`
- Add a single CLI arg pair to control the overlay preset:
  - `--overlays {minimal,full,off}` (default `minimal`)
  - `--show <comma-list>` (e.g. `--show ambient_connections,goalie_sight_lines`) — opt-in additions on top of the chosen preset, for selectively re-enabling features
  - `--hide <comma-list>` — opt-out subtractions, mirror of `--show`
- Translate these into an `overlay_config` dict and pass to `run_game_analysis_mode` (which forwards it to `GameAnnotator`).
- This is intentionally minimal CLI surface — the user said not to over-engineer. 14 individual flags would be parameter sprawl; the preset + show/hide pattern is one mechanism that scales.

## Quality gate logic (per feature)

| Feature              | Default in MINIMAL | Quality gate                                                            |
|----------------------|--------------------|-------------------------------------------------------------------------|
| boxes                | ON                 | (none — always reliable enough)                                          |
| team_colors          | OFF                | classifier.is_ready AND vote_counts[id] ≥ 5                              |
| puck                 | ON                 | (existing — solid for conf≥0.4, dashed otherwise, skip ghost past coast) |
| possession_indicator | OFF                | possession_player_id is not None AND classifier.is_ready                 |
| zone_banner          | OFF                | not is_shootout_like                                                     |
| ambient_connections  | OFF                | classifier.is_ready AND all teammates have ≥3 votes                      |
| open_spaces          | OFF                | not is_shootout_like AND ≥3 players                                      |
| passing_lanes        | OFF                | classifier.is_ready AND ≥1 teammate                                       |
| shooting_lane        | OFF                | classifier.is_ready AND defending goalie identified                       |
| decision_banner      | OFF                | (none — events are confident by detection logic)                          |
| decision_freeze      | OFF                | (none)                                                                    |
| slowdown_indicator   | OFF                | (none)                                                                    |
| goalie_sight_lines   | OFF                | goalie has team assignment OR is_shootout_like (always-on path)           |
| frame_info_hud       | ON                 | (none — diagnostic)                                                       |
| non_gameplay_stamp   | ON                 | actually-not-gameplay (post-hysteresis fix)                                |
| camera_cut_indicator | ON                 | (none)                                                                    |

## Critical functions / utilities to reuse

- `team_classifier.is_ready` and `team_classifier._vote_counts` (in `team_classifier.py`) — already populated, we just need to expose `vote_counts` cleanly. Add a public `get_vote_count(track_id) -> int` helper.
- `_find_player_by_id` static on `GameAnnotator` — already exists from the simplify pass; reuse for any quality gate that needs to look up a specific player.
- The existing `is_shootout_like` calculation in `__init__.py` Pass 1 — we extend it with hysteresis, don't replace it.
- The existing `_draw_*` methods in `game_annotator.py` — all kept verbatim. Only the call sites change.

## Verification

After implementation, run all three canonical clips with the new MINIMAL default:
```
python main.py --game-analysis -i data/raw_videos/rush_30sec_clip.mp4 -o output/rush_minimal.mp4
python main.py --game-analysis -i data/raw_videos/shootout_60sec.mp4 -o output/shootout_minimal.mp4
python main.py --game-analysis -i data/raw_videos/ig_1v1_beating_guys.mp4 -o output/1v1_minimal.mp4
```

Then extract frames from each (cv2-based, ffmpeg unavailable) and visually verify:
- **Rush clip**: gray player boxes, puck (when high-conf), no team colors, no ambient lines, no open-space diamonds, no zone banner, no decision banners. Just clean tracking.
- **Shootout clip**: NO "NON-GAMEPLAY" stamp on McDavid/Kane approach frames (the hysteresis bug fix). Only player boxes + puck visible.
- **1v1 clip**: same as rush — no team colors at all (this clip has shaky team classification).

Then run with `--overlays full` and verify everything still renders correctly (regression check):
```
python main.py --game-analysis --overlays full -i data/raw_videos/rush_30sec_clip.mp4 -o output/rush_full.mp4
```

Then run with the most useful subset for a young player using `--overlays minimal --show ambient_connections,goalie_sight_lines,decision_freeze` and verify those three features render correctly while everything else stays minimal.

Final visual checks for each output frame I extract:
- Are overlays I expect to be off, off?
- When team colors / ambient lines fire, are they on actually-classified players (no random colors)?
- Does the shootout clip cleanly NOT show NON-GAMEPLAY anywhere during gameplay?
- Does the puck still appear correctly (filter still working)?

## Risks / watch items

- **Quality gates may suppress everything**: if `team_classifier` takes too long to converge, none of the team-dependent overlays will ever fire on short clips. Acceptable for Phase A — better to show nothing than to show wrong colors. We'll watch and tune the vote thresholds (5 for team_colors, 3 for ambient) in the smoke test.
- **`team_classifier._vote_counts` exposed via getter is a small API addition** — fine, the classifier is project-internal.
- **Hysteresis bound (10 frames)** for shootout: long enough to handle dropout, short enough that an actual end-of-shootout (player skating away with puck) doesn't keep flagging shootout-like. If 10 is too long the user will see lingering shootout overlays after the play ends; we'll tune.
- **`overlay_config` dict refactor breaks the existing `show_boxes`/`show_zone`/`coaching_mode` constructor args**. We keep them as deprecated kwargs that map into the dict so any external caller doesn't break.

## Below this line: previous (already-shipped) plans retained for reference

# Hockey AI — Always-on Carrier Connections + Shootout Context + Goalie Sight Fix

## Context

After watching the latest output videos and confirming I can extract+view frames directly, four concrete defects are limiting the project's usefulness as a tactical coaching tool:

1. **No always-on tactical readout for the carrier.** Ambient teammate connections already exist in `lane_calculator.py::calc_ambient_connections`, but they only ever rendered during gameplay frames AND have a "partial" yellow tier that dilutes the signal. The user wants binary green/red lines from the puck carrier to every teammate at all times: green if the puck can reach (open lane OR teammate is in open space, accounting for sauce/flip/bank passes), red otherwise. This makes teammate identification accuracy visible at a glance.

2. **Broadcast filter kills shootouts.** `BroadcastFilter._detect_gameplay()` requires `min_game_objects=3` for 15 frames; shootouts only have 2 objects (shooter + goalie). Frames are flagged `NON-GAMEPLAY`, which short-circuits team classification, ambient connections, goalie sight lines, and decision detection. This was confirmed by extracting frames from `output/shootout_final.mp4` — the McDavid approach frames all show the dim "NON-GAMEPLAY" stamp.

3. **Goalie sight lines never appear during freeze frames.** The conditional in `game_annotator.py` is:
   ```
   if is_shot_event and context.goalies and phase in ("slowdown", "freeze"):
   ```
   `context.goalies` is empty in many broadcast frames because the camera zooms on the carrier and the goalie isn't in-frame for that single moment, so the entire goalie analysis block is skipped even when the event type and phase are correct.

4. **Irrelevant overlays clutter shootouts.** The zone banner, generic open-space diamonds, and ambient connections (which need a carrier+teammates pair) all draw during shootout-style 1-on-1s where they convey nothing useful.

## Scope decisions (from user answers)

- **Team lines:** Carrier-to-teammate only, binary green/red. No full mesh.
- **Open space rule:** Primary criterion = nearest opponent farther than threshold (favor this). Secondary boost = teammate position falls inside a nav-mesh open pocket from the existing `space_detector`. Either passes the green test.
- **Broadcast filter fix:** Both (a) lower `min_game_objects` to 2 and (b) detect shootout context via a flag on FrameContext. The shootout flag also drives overlay suppression.
- **Puck dropouts:** Keep coast at 5 frames; lower YOLO confidence threshold for the puck class only via a conditional second YOLO call (only when the primary call returns no puck, to avoid doubling inference cost on every frame).

## Files to modify

### `src/game_analysis/broadcast_filter.py`
- Change `min_game_objects` default from 3 → 2 in `__init__`.

### `src/game_analysis/game_context.py`
- Add `is_shootout_like: bool = False` to `FrameContext`.

### `src/game_analysis/__init__.py` (Pass 1 loop, around line 145)
- After tracker returns `objects`, compute a shootout-like flag:
  ```
  is_shootout_like = (
      len(players) <= 2
      and len(goalies) >= 1
      and len([o for o in objects if o.class_name == "referee"]) <= 1
  )
  ```
- Set `ctx.is_shootout_like = is_shootout_like`.
- When shootout-like, force `is_gameplay = True` even if the broadcast filter rejected (bypass the filter for these frames specifically).

### `src/game_analysis/game_tracker.py`
- After the primary `model.track(...)` call, if no puck class is in the result, run a second `model.predict(frame, classes=[puck_class_id], conf=0.10)` and merge any pucks into the object list. Cache the puck class id at startup (already discovered via `class_names`).
- The existing `PuckFilter` will reject low-quality pucks via on-ice + proximity scoring.
- Lower `PuckFilter.min_conf` default from 0.30 → 0.18 in `puck_filter.py` to let the lower-confidence puck candidates pass when they're on-ice and consistent with the last position.

### `src/game_analysis/lane_calculator.py::calc_ambient_connections`
- Change quality semantics from {"open", "partial", "covered"} → {"reachable", "blocked"}.
- A teammate is "reachable" (green) if EITHER:
  - direct lane to carrier has 0 blockers within 50px perpendicular to the line, OR
  - nearest opponent to teammate is > 110px (favored), OR
  - teammate position falls inside an open-space pocket returned by `space_detector.find_open_spaces` (need to thread the open_spaces list through, OR re-invoke the detector here — prefer threading to avoid duplicate work).
- Otherwise "blocked" (red).
- Always return at least one entry per teammate (no filtering by distance — the user wants every teammate connected at all times).

### `src/game_analysis/__init__.py` (rendering loop)
- Pass `open_spaces` into `calc_ambient_connections` so the nav-mesh pockets can drive the secondary "open space" check without recomputing.

### `src/game_analysis/game_annotator.py`
- `AMBIENT_COLORS` simplified to:
  ```
  AMBIENT_COLORS = {
      "reachable": (0, 220, 0),    # Bright green
      "blocked":   (0, 0, 220),    # Bright red
  }
  ```
- `_draw_ambient_connections` keeps current style (dashed line + endpoint circle + alpha blend).
- `_draw_zone_banner` invocation gated: `if self.show_zone and context.zone and not context.is_shootout_like`.
- `_draw_open_spaces` skipped when `context.is_shootout_like`.
- `render()` and `render_coaching()`: when `context.is_shootout_like` and `context.goalies` AND we have a carrier (the player closest to the puck on the non-goalie team), call `goalie_analyzer.analyze_shootout` (not `analyze_shot`) and render every frame, not just freeze-phase. This is the always-on shootout sight-line rendering.
- Fix goalie-sight-line freeze-frame bug: add a small `_last_goalie_by_team` cache on the annotator (or in FrameContext) so when `context.goalies` is empty at the freeze frame, we fall back to the most recently seen goalie from the same side. This unblocks the existing event-driven goalie sight lines for `missed_opportunity`, `shot_vs_pass`, and `odd_man_rush` events.

## Critical functions / utilities to reuse

- `lane_calculator.py::_count_blockers_along_line` (existing helper that counts opponents along a passing lane segment).
- `space_detector.py::find_open_spaces` (already returns nav-mesh pockets; threading its result into `calc_ambient_connections` avoids recomputation).
- `goalie_analyzer.py::analyze_shootout` (already implemented but never wired up — that's the function that returns the recommend SHOT/DEKE banner with reasoning).
- `puck_filter.py::PuckFilter.filter` (already enforces single-puck + on-ice + coast; just lower min_conf).

## Verification — view frames directly

I can extract frames from output videos and read them as images. Workflow per change:

```python
# Run on canonical clip
python main.py --game-analysis --input data/raw_videos/shootout_60sec.mp4 \
  --play-style balanced --output output/shootout_final.mp4

# Extract frames to inspect (cv2-based, ffmpeg not available on this system)
python -c "
import cv2
cap = cv2.VideoCapture('output/shootout_final.mp4')
for f in [200, 600, 1000, 1500, 2000]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(f'output/_frames/check_f{f:04d}.png', frame)
"
```
Then read the PNGs with the Read tool to visually confirm:
- Every gameplay frame shows green/red dashed lines from the carrier to each teammate.
- No "NON-GAMEPLAY" stamp on McDavid-approach frames in the shootout clip.
- Zone banner is gone during shootout-flagged frames.
- Goalie sight-line rays appear on shootout approaches (always-on) AND on freeze frames for missed-opportunity events.
- No more multi-puck visible (still expected; this is preserved from prior fix).

Run on all three canonical clips: `rush_30sec_clip.mp4`, `shootout_60sec.mp4`, `ig_1v1_beating_guys.mp4`.

## Commit

One commit when all four issues are visibly fixed:
```
Always-on carrier connections + shootout context + goalie sight fix

- FrameContext.is_shootout_like flag (1 player + 1 goalie + few refs);
  drives overlay suppression and bypasses broadcast filter when set
- BroadcastFilter min_game_objects 3->2 so 1v1 / 2-on-1 footage isn't
  flagged non-gameplay anymore
- calc_ambient_connections: binary reachable/blocked per teammate.
  Reachable = open direct lane OR nearest opponent >110px (sauce pass
  range) OR teammate inside a nav-mesh open pocket
- Always-on green/red carrier-to-teammate lines on every gameplay
  frame, not just decision events
- Shootout context: hides zone banner and open-space diamonds; runs
  analyze_shootout every frame for live SHOT/DEKE recommendation
- Goalie sight-line freeze-frame bug: cache last-seen goalie on the
  annotator so freeze frames work even when YOLO drops the goalie
  for that single moment
- GameTracker: conditional second YOLO pass at conf=0.10 for puck-only
  when primary returns no puck; PuckFilter min_conf 0.30->0.18 to
  accept the lower-conf candidates after on-ice+proximity vetting
```

## Risks / watch items

- **Lowering broadcast min_game_objects to 2 may flag more cutaway/replay frames as gameplay.** Watch the rush clip's `Camera cuts` count and gameplay % in the console; should stay roughly similar.
- **Always-on green/red lines could be visually noisy** with 4–5 teammates. If too busy, consider making non-carrier lines thinner OR fading by distance. Monitor with extracted frames.
- **Lower puck conf may reintroduce false positives** despite the on-ice filter. The `_last_pos` proximity bonus should help; if not, raise `PuckFilter.min_conf` back toward 0.20 or 0.22.
- **Always-on goalie sight lines during shootouts** add visual density; the existing `draw_goalie_analysis` style is already calibrated for this so should be fine.
- **Goalie last-seen cache** has the same drift risk as puck coast — if it's stale by many frames the rays will be cast at the wrong defender.

## Below this line: previous (already-shipped) plan retained for reference

## Context

After watching the three final output clips from the previous batch, two recurring issues remain:

1. **Puck false positives.** The YOLO detector is firing on glass-board reflections, tarp logos, and dark spots that look puck-shaped, and ALL of them are drawn in the output. Multiple "pucks" appear simultaneously. There is no temporal continuity, no ice-region constraint, and no single-puck enforcement — `GameTracker._parse_results()` returns every detection, and the renderer iterates all of them.

2. **Team classification fails on dim/low-quality footage.** The current classifier uses median BGR pixels masked against pure-white ice and pure-black shadow. Under dim arena lighting or low-quality streams, jersey colors collapse toward each other in BGR space and the (R−B) dominance test we added doesn't have enough signal to separate teams.

The user's tactical context to encode:
- There is only ever ONE puck in play at a time.
- The puck is always on the ice sheet (not on glass, scoreboard, boards above the rink, etc.).
- Lighting can drift dramatically clip-to-clip; jersey color identity is invariant under lighting.

## Scope decisions (from user answers)

- **Ice region detection:** Color-based ice mask primary; geometric fallback from rink/player landmarks if the color mask covers <10% of the frame.
- **Single-puck enforcement:** Soft. Apply confidence + on-ice filter; let the system warn (not crash) if >1 candidate survives. Renderer should still only draw the best one.
- **Low-light team classification:** Dual-mode. CLAHE-normalized Lab(a,b) clustering primary; if cluster separation is poor, fall back to the existing BGR median method. Decided per-clip during warmup.
- **Puck coast:** Yes, hold last-known position for up to 5 frames when YOLO drops the puck (no velocity prediction — simple position hold).

## Files to modify

### New file
- `src/game_analysis/puck_filter.py` — encapsulates ice-mask building, puck scoring, single-puck selection, and coast logic.

### Modified files

- `src/game_analysis/game_tracker.py`
  - Instantiate a `PuckFilter` in `GameTracker.__init__` (with a `frame` argument added so it can build the ice mask).
  - Modify `process_frame` to call `puck_filter.filter(objects, frame)` after `_parse_results`. The filter:
    - Builds the ice mask for this frame (color → geometry fallback).
    - Drops puck-class objects whose center is not on ice.
    - Drops puck-class objects below `min_conf` (0.30).
    - Scores remaining pucks: `score = conf + proximity_bonus_to_last_known(decay over 5 frames)`.
    - Keeps the top one by default. Logs a warning (rate-limited) if >1 candidate passed all filters.
    - On no surviving puck: if last seen ≤ 5 frames ago, synthesize a ghost `TrackedObject` at last position (mark with `confidence = 0.0` so consumers can detect it; track_id reused).
  - Replaces all puck objects in the returned list with the filtered single survivor (or the ghost, or none).
  - Update `get_puck()` to a no-op trivial: just return the (now single) puck if any.

- `src/game_analysis/team_classifier.py`
  - Refactor `_extract_jersey_color()` to return a dict with both BGR median AND Lab(a,b) median:
    - Apply CLAHE (clipLimit=2, tileGridSize=(8,8)) to the L channel of the jersey crop before computing the Lab median. This neutralizes per-frame brightness drift.
    - Same valid-pixel mask as before (no white-ice / no black-shadow).
  - During warmup, accumulate samples in BOTH BGR and Lab(a,b) feature spaces.
  - At `_run_clustering()`:
    - Run k-means twice (BGR and Lab(a,b)).
    - Compute separation quality for each: normalized inter-center distance / mean intra-cluster spread (a Davies-Bouldin-ish ratio).
    - Pick the higher-quality clustering. Store `self._feature_mode = "lab"` or `"bgr"`.
    - Print which mode won and the separation scores.
  - `_classify()`, `_refine_centers()`, `classify_goalie()` all branch on `_feature_mode` to extract the right feature dimension.
  - Keep the (R−B) ordering convention for BGR mode; for Lab mode, order by `a` channel (higher a = redder).

- `src/game_analysis/game_annotator.py` (line ~241)
  - When drawing pucks, skip ghost pucks (confidence == 0.0) OR draw them with a dashed/translucent style so the user can tell coasted pucks from solid detections. Recommend dashed.

## Critical functions / existing code to reuse

- `src/game_analysis/team_classifier.py::_extract_jersey_color()` (lines 101-155) — reuse the bbox crop and valid-pixel mask logic. Just add the CLAHE+Lab extraction alongside the BGR median.
- `src/game_analysis/team_classifier.py::_numpy_kmeans()` (lines 17-42) — reuse for both BGR and Lab clustering; it's general over feature dimensionality.
- `cv2.createCLAHE` and `cv2.cvtColor(..., cv2.COLOR_BGR2LAB)` — both already in cv2 (no new deps).
- `scipy.ndimage` — available (we use it in space_detector); not strictly needed for this work.

## New module: `puck_filter.py` design

```
class PuckFilter:
    def __init__(self, min_conf=0.30, coast_frames=5, ice_color_thresh=...):
        self._last_pos = None
        self._frames_since_seen = 999
        self._frame_idx = 0
        self._cached_ice_mask = None
        self._cached_ice_mask_frame = -1

    def filter(self, objects, frame, players=None, goalies=None) -> list:
        # 1. Build / reuse ice mask
        # 2. Partition objects into pucks vs non-pucks
        # 3. Drop pucks by conf + on-ice
        # 4. Score remaining: conf + 0.3 * exp(-dist_to_last/200)
        # 5. Keep top one
        # 6. If none survived but coast valid: synthesize ghost
        # 7. Return non_pucks + [chosen_puck]

    def _build_ice_mask(self, frame, players, goalies):
        # HSV: V > 180, S < 60, hue near neutral
        # If coverage > 10%: return that
        # Else: geometric fallback - bbox of (players + goalies + landmarks),
        #   padded by 50px, rendered as a filled polygon mask
```

## Verification

Run after each major change. Working directory: project root. Python: `C:/Users/17742/AppData/Local/Programs/Python/Python312/python.exe`.

**After puck filter implementation:**
```
python main.py --game-analysis --input data/raw_videos/rush_30sec_clip.mp4 --play-style balanced --output output/_puck_smoke.mp4
python main.py --game-analysis --input data/raw_videos/shootout_60sec.mp4 --play-style balanced --output output/_puck_shootout.mp4
```
Spot-check: only one puck visible at any time; on glass-board false positives the puck stays on ice and ghost frames look reasonable; possession events still fire.

**After team classifier dual-mode:**
```
python main.py --game-analysis --input data/raw_videos/ig_1v1_beating_guys.mp4 --play-style balanced --output output/_team_smoke.mp4
```
Console should print which mode (`lab` or `bgr`) was selected and the separation scores. Spot-check: team colors stay consistent across the dim clip.

**Final canonical run (all three clips):**
```
python main.py --game-analysis --input data/raw_videos/rush_30sec_clip.mp4 --play-style balanced --output output/rush_final.mp4
python main.py --game-analysis --input data/raw_videos/shootout_60sec.mp4 --play-style balanced --output output/shootout_final.mp4
python main.py --game-analysis --input data/raw_videos/ig_1v1_beating_guys.mp4 --play-style balanced --output output/game_1v1_final.mp4
```

Visual checks: only one puck on screen at any time; team colors stable through the entire clip even on dim frames; goalies still tagged; no regressions in decision events.

**Commit:**
One commit at the end:
```
Add puck filter (single-puck + on-ice) and dual-mode team classifier

- New PuckFilter: per-frame ice-region mask (HSV color → player-bbox
  geometry fallback), confidence cutoff, single-puck selection by
  conf+proximity, 5-frame coast with ghost-puck synthesis on dropouts
- GameTracker integrates PuckFilter, returns a single puck object per
  frame; renderer marks ghosts with a dashed style so they're visible
  but distinguishable from solid detections
- TeamClassifier dual-mode: extracts both BGR median and CLAHE-Lab(a,b)
  features during warmup, runs clustering on both, picks the mode with
  better separation (intra/inter cluster ratio); robust to dim arena
  lighting where BGR alone collapses
- (R-B) dominance ordering preserved in BGR mode; Lab mode orders by
  a-channel (redder = team_a)
```

## Risks / watch items

- **Ice mask false negatives:** If the color mask is too strict, the puck could be falsely rejected (e.g., on the snow spray near a stop). Geometry fallback should kick in via the <10% threshold; but if both fail, we accept the puck unfiltered (the filter is an optimization, not a hard gate).
- **Lab mode on grayscale jerseys (white vs black):** Lab(a,b) would have low separation here; the dual-mode quality check should correctly fall back to BGR. Watch the warmup log to confirm.
- **Ghost puck and decision detectors:** `shot_vs_pass` and others read `puck.center`; ghosts have valid centers but `confidence=0.0`. Detectors that gate on conf will skip them; detectors that don't will treat them as real (acceptable — coast is short).
- **CLAHE cost:** ~1ms per jersey crop × ~10 players × 30fps × 30s = ~9s extra processing per 30s clip. Acceptable.
