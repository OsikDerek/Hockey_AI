# Hockey AI — Decision-Training Simulator from Real Game Film

A computer-vision + 3D-viewer system that turns broadcast or
sports-cam hockey footage into a tactical decision-training tool.
Built by a professional hockey player and skating coach as a personal
development project — not a generic analytics platform.

## The vision

**A playable VR/gamepad hockey simulator loaded from your own shifts.**
Take the world state at a real decision moment, drop the user into POV
with controllers, and let them play out alternatives. Repetition
rebuilds neural pathways — retraining bad habits, learning to
scan-before-receive, growing hockey IQ through high-rep practice in a
safe environment. The end state is "VR hockey, but the world is your
actual shifts."

Everything currently in this repo — tracking, calibration, the 3D
viewer, Quiz Mode — is a stepping stone toward that.

## What ships today

- **3D top-down / broadcast / POV viewer** (`viewer/`) — Three.js scene
  driven by per-frame positions JSON. Scrub through real game film
  reconstructed as avatars on a regulation rink.
- **Quiz Mode** — at every high-confidence decision moment, playback
  pauses, you pick what you'd do (carry / dump / pass / shoot), and
  the reveal shows the actual decision + the AI's evaluation. Toggle
  between Top-Down (tactical pattern-reading) and Player-POV (the
  end-goal training feel) with the V key.
- **Source video side-by-side** — togglable panel that loads the raw
  clip and scrub-locks to the 3D playback, for verifying what the
  tracker actually saw vs what the system rendered.
- **Two analysis pipelines** through one `main.py`:
  - `--technique <skill>` — single-skater drill / practice analysis
    with 10 YAML-defined techniques (forward_stride, crossover,
    wrist_shot, snap_shot, one_timer, stickhandling, hockey_stop,
    backwards_skating, transitions, edge_work)
  - `--game-analysis` — multi-player game film with zone detection,
    possession tracking, and 7 decision detectors (shot_vs_pass,
    zone_entry, breakout, odd_man_rush, forecheck, defensive_play,
    missed_opportunity)
- **Headless test harness** — Playwright drives a real Chromium against
  the viewer, screenshots each phase of Quiz Mode, catches regressions
  without needing a human in the loop (`scripts/test_quiz_browser.py`).

## Active development focus

**Tracking/calibration accuracy on cropped junior footage.** The
3D-viewer scene is only as good as the positions JSON feeding it.
Recent batches have been:

- Per-track EMA position smoothing
- Track-ID stitching to collapse ByteTrack ghost identities
- Render-tick lerp + outlier rejection (no more 1900 ft/s puck jumps)
- One-and-only-one-puck rendering with carrier snap during decisions
- Diagnostic: across-rink y-compression from similarity-transform
  fallback identified as the biggest remaining accuracy issue

See [`.claude/handoff/NEXT.md`](.claude/handoff/NEXT.md) for the
prioritized work queue.

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get a model (one-time)
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('SimulaMet-HOST/HockeyAI', 'HockeyAI_model_weight.pt', \
  local_dir='models')"

# 3. Run game analysis to produce a positions JSON
python main.py --game-analysis -i data/raw_videos/your_clip.mp4

# 4. Launch the 3D viewer (auto-opens browser to the latest JSON)
python scripts/serve_viewer.py
```

For technique analysis on a single skater:
```bash
python main.py -i drill.mp4 --technique forward_stride --auto-crop
python main.py -i drill.mp4 --technique crossover --auto-crop --mode crossover
python main.py -i shot.mp4 --technique wrist_shot --auto-crop
```

## Test bed

The best demo clip is junior-hockey LiveBarn footage cropped to the
rink:

| Clip | Calibration | Quiz events renderable | Notes |
|---|---|---|---|
| `livebarn_60sec_cropped.mp4` | 97.5% in-rink | 7/7 | Primary test bed |
| `rush_30sec_clip.mp4` | 21% | 2/16 | Broadcast follow-cam — calibration is the limiter |
| `ig_1v1_beating_guys.mp4` | 100% | 1/30 | Side-cam drill footage |

## Architecture

```
main.py              ─→  HockeyAI YOLOv8 + ByteTrack  ─→  positions JSON  ─→  Three.js viewer
                         + RinkCalibrator (homography)                       (Quiz Mode + POV)
                         + 7 decision detectors
```

The two analysis modes share the pose/tracking stack but produce
different output:
- **Technique mode** → annotated video + report.txt + report.json
- **Game-analysis mode** → annotated video + positions JSON (the
  NHL EDGE-shaped per-frame data that drives the 3D viewer)

## Tech stack

- **Detection / tracking:** HockeyAI YOLOv8 (7-class), ByteTrack
- **Pose:** MediaPipe (33-landmark) + YOLOv8-pose backend
- **Calibration:** OpenCV homography + landmark detection (CV-based
  blue/red line detector, faceoff-dot detector via YOLO)
- **3D viewer:** Three.js (vanilla, no framework). Per-render-tick
  lerp + outlier rejection in `viewer/viewer.js`
- **Headless testing:** Playwright on a real Chromium with a
  `window.__hockeyAI.snapshot()` debug hook

## Project structure

```
Hockey_AI/
├── main.py                          # CLI entry point
├── viewer/                          # ★ 3D viewer (centerpiece)
│   ├── index.html
│   ├── viewer.js                    # render loop, smoothing, POV
│   ├── avatar.js                    # skater + puck meshes
│   ├── camera.js                    # top-down / broadcast / POV
│   ├── quiz.js                      # Quiz Mode state machine
│   └── ...
├── scripts/
│   ├── serve_viewer.py              # static HTTP for the viewer
│   ├── test_quiz_browser.py         # headless Quiz Mode test
│   ├── test_motion_smoothness.py    # headless smoothness probe
│   ├── prepare_web_video.py         # transcode raw video → web-playable
│   └── ...
├── src/
│   ├── pose_estimator.py
│   ├── technique_engine.py
│   ├── object_tracker.py
│   └── game_analysis/               # game-mode pipeline
│       ├── game_tracker.py
│       ├── rink_calibrator.py
│       ├── play_evaluator.py
│       └── decisions/               # decision detectors
├── knowledge_base/
│   ├── techniques/                  # 10 technique YAML files
│   └── game_situations/             # decision evaluation criteria
├── models/                          # weights (not in git)
├── data/raw_videos/                 # input clips (not in git)
└── output/                          # generated output (not in git)
```

## Requirements

- Python 3.12 recommended (3.10+ works)
- OpenCV, MediaPipe, Ultralytics, NumPy, SciPy, PyYAML, lapx
- HockeyAI model weights (HuggingFace; see Quick Start)
- Browser-friendly source video for the side-by-side panel
  (`scripts/prepare_web_video.py` transcodes raw mpeg4 sports-cam
  output to H.264 .web.mp4)
- Optional: NVIDIA GPU. Author's desktop (RTX 2080 Super) runs the
  game-analysis pipeline at ~18-19 fps tracking.

## Adding your own knowledge

### New technique
Drop a YAML file in `knowledge_base/techniques/`:
```yaml
technique:
  name: my_technique
  detection:
    detector: FrameByFrame
  checks:
    knee_angle:
      angle_function: knee_angle
      good_range: [95, 125]
      feedback:
        good: "Great knee bend"
        poor: "Bend your knees more"
      drills:
        poor:
          - name: "Wall Sits"
            cue: "Hold for 30 seconds"
```

### New game decision type
1. Add a detector in `src/game_analysis/decisions/`
2. Register it in `DECISION_REGISTRY` in `decisions/__init__.py`
3. Add evaluation YAML in `knowledge_base/game_situations/`

### Custom team play style
Edit `PLAY_STYLES` in `src/game_analysis/play_evaluator.py`. Each
style biases decision ratings per event type — possession favors
carries, physical favors dump-and-chase, etc.

## About

Built by Derek Osik — professional hockey player, skating coach, and
software engineer. The project is a personal-development tool first;
the goal is to use it on my own game film to train my decision-making
the same way pros use video review, but in a richer 3D simulator
environment that VR will eventually unlock.

If you're another hockey player, coach, or CV person poking around:
the issues + roadmap in
[`.claude/handoff/NEXT.md`](.claude/handoff/NEXT.md) are the best
window into what's next.
