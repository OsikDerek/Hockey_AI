# Hockey AI — Decision Simulator (game-analysis + 3D viewer + Quiz Mode)

A computer-vision + 3D-viewer system that turns broadcast or sports-cam
hockey footage into a tactical decision-training tool. Built by a
professional hockey player and skating coach as a personal-development
project — not a generic analytics platform.

> **Split note (2026-05-11):** the body-mechanics / single-skater
> technique analysis half of this project was spun out into a separate
> repo: [Hockey_Vision_AI](https://github.com/OsikDerek/Hockey_Vision_AI).
> Earlier git history of this repo includes the technique pipeline.

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
- **Source-video side-by-side** — togglable panel that loads the raw
  clip and scrub-locks to the 3D playback, for verifying what the
  tracker actually saw vs what the system rendered.
- **Game-analysis pipeline** (`src/game_analysis/`) — multi-player
  tracking via HockeyAI YOLOv8 + ByteTrack, zone detection, possession
  with hysteresis, BGR-median + Lab dual-mode team classifier, 7
  decision detectors (shot_vs_pass, zone_entry, breakout, odd_man_rush,
  forecheck, defensive_play, missed_opportunity).
- **Headless test harness** — Playwright drives a real Chromium against
  the viewer, screenshots each phase of Quiz Mode, catches regressions
  without needing a human in the loop
  (`scripts/test_quiz_browser.py`, `scripts/test_motion_smoothness.py`,
  `scripts/test_source_video_panel.py`).

## Active development focus

**Tracking + calibration accuracy on cropped junior footage.** The
3D-viewer scene is only as good as the positions JSON feeding it.
Recent batches:

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

# 2. Get the HockeyAI model weights (one-time)
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('SimulaMet-HOST/HockeyAI', 'HockeyAI_model_weight.pt', \
  local_dir='models')"

# 3. Run game analysis on a clip → produces output/<name>_positions.json
python main.py --game-analysis -i data/raw_videos/your_clip.mp4

# 4. Launch the 3D viewer (auto-opens browser to the latest JSON)
python scripts/serve_viewer.py
```

For technique / body-mechanics analysis on a single skater, use the
sister repo: [Hockey_Vision_AI](https://github.com/OsikDerek/Hockey_Vision_AI).

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
main.py  ─→  HockeyAI YOLOv8 + ByteTrack  ─→  positions JSON  ─→  Three.js viewer
              + RinkCalibrator (homography)                       (Quiz Mode + POV)
              + 7 decision detectors
```

## Tech stack

- **Detection / tracking:** HockeyAI YOLOv8 (7-class), ByteTrack
- **Calibration:** OpenCV homography + landmark detection (CV-based
  blue/red line detector, faceoff-dot detector via YOLO)
- **3D viewer:** Three.js (vanilla, no framework). Per-render-tick
  lerp + outlier rejection in `viewer/viewer.js`
- **Headless testing:** Playwright on a real Chromium with a
  `window.__hockeyAI.snapshot()` debug hook

## Project structure

```
Hockey_AI/
├── main.py                          # CLI entry point (game-analysis only)
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
│   ├── test_source_video_panel.py   # headless source-video test
│   ├── prepare_web_video.py         # transcode raw video → web-playable
│   └── ...
├── src/
│   ├── video_io.py                  # OpenCV wrappers (shared util)
│   ├── utils.py                     # small helpers (shared util)
│   └── game_analysis/               # game-mode pipeline
│       ├── game_tracker.py          # Multi-player tracking
│       ├── rink_calibrator.py       # Ice-coordinate calibration
│       ├── play_evaluator.py        # Decision scoring + play-style bias
│       ├── game_annotator.py        # Overlay rendering
│       ├── game_report.py
│       ├── broadcast_filter.py
│       └── decisions/               # Decision detectors
├── knowledge_base/
│   └── game_situations/             # Decision evaluation YAML
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

## Adding decision types or play styles

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
software engineer. This project is a personal-development tool first;
the goal is to use it on my own game film to train my decision-making
the same way pros use video review, but in a richer 3D simulator
environment that VR will eventually unlock.

**Companion repo:** [Hockey_Vision_AI](https://github.com/OsikDerek/Hockey_Vision_AI)
(single-skater body-mechanics / technique CV).

If you're another hockey player, coach, or CV person poking around:
the issues + roadmap in
[`.claude/handoff/NEXT.md`](.claude/handoff/NEXT.md) are the best
window into what's next.
