# Hockey AI — Skating Technique Analyzer & Game Decision Optimizer

Computer vision system that analyzes hockey video for individual technique coaching and team-level tactical decision evaluation. Built by a professional hockey player and skating coach.

## Two Modes

### 1. Technique Analysis
Analyzes individual skating and skills technique from practice or drill footage.

```bash
# Forward stride analysis
python main.py -i video.mp4 --technique forward_stride --auto-crop

# Crossover-specific analysis (only flags crossover issues, not everything)
python main.py -i video.mp4 --technique crossover --auto-crop --mode crossover

# Wrist shot mechanics
python main.py -i video.mp4 --technique wrist_shot --auto-crop

# Stickhandling with puck tracking
python main.py -i video.mp4 --technique stickhandling --auto-crop
```

### 2. Game Analysis
Analyzes broadcast game film for tactical decisions — tracks all players, detects zones, evaluates play decisions, and suggests better options.

```bash
# Basic game analysis
python main.py --game-analysis -i game_clip.mp4

# With team play style bias
python main.py --game-analysis --play-style possession -i game_clip.mp4
python main.py --game-analysis --play-style physical -i game_clip.mp4
python main.py --game-analysis --play-style speed -i game_clip.mp4
python main.py --game-analysis --play-style defensive -i game_clip.mp4
```

## Quick Start

```bash
pip install -r requirements.txt

# Download HockeyAI model for game analysis (one-time)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('SimulaMet-HOST/HockeyAI', 'HockeyAI_model_weight.pt', local_dir='models')"

# Run technique analysis
python main.py -i video.mp4 --auto-crop

# Run game analysis
python main.py --game-analysis -i game_clip.mp4
```

## Technique Analysis Features

### Knowledge Base (10 Techniques)
Each technique is defined in YAML with detection rules, thresholds, coaching feedback, and drill recommendations:

| Technique | Key Checks |
|-----------|-----------|
| `forward_stride` | Knee bend, hip hinge, forward lean, ankle dorsiflexion, stride symmetry |
| `crossover` | Knee drive, internal rotation, step-out explosiveness |
| `wrist_shot` | Knee bend, hip rotation, weight transfer, hands out front, follow-through |
| `snap_shot` | Knee bend, hip rotation, hands from body |
| `one_timer` | Knee bend, hip rotation, hand separation |
| `stickhandling` | Athletic stance, head/eyes up, hand spacing, puck control zone |
| `hockey_stop` | Knee bend, hip position, edge engagement |
| `backwards_skating` | Knee bend, hip position, posture |
| `transitions` | Knee bend, hip position, balance |
| `edge_work` | Knee bend, ankle engagement, balance |

### Detection Capabilities
- **Head-up detection**: `head_pitch` metric detects if player is looking down at puck
- **Hand position**: `hand_separation`, `hands_from_body`, `stick_angle_proxy`
- **Puck tracking**: ObjectTracker with HockeyAI YOLO model detects puck proximity
- **Crossover events**: Temporal detection of crossover sequences with knee drive scoring

### Pose Estimation Backends
| Backend | Landmarks | Best For | Speed |
|---------|-----------|----------|-------|
| `mediapipe` (default) | 33 (incl. feet) | Single skater, CPU | ~5 fps |
| `yolo` | 17 (COCO) | Multi-person, GPU | ~15 fps (GPU) |

### Auto-Crop
For distant/wide-angle footage: YOLO person detection finds and tracks the skater, crops and upscales to fill the frame.

## Game Analysis Features

### Multi-Player Tracking
- Tracks all players, puck, goalies, and referees via HockeyAI YOLOv8 + ByteTrack
- Stable track IDs across frames, automatic reset on camera cuts
- Detects 7 object classes: player, puck, goalie, referee, center ice, faceoff dots, goal

### Zone Detection
- Classifies offensive/neutral/defensive zone from rink landmarks
- Falls back to player distribution heuristics when landmarks aren't visible
- Smoothed with rolling majority vote (no flickering)

### Possession Detection
- Puck-to-player proximity with movement direction consistency
- 8-frame hysteresis window prevents flickering on broadcast footage
- Coasts through brief puck-loss frames

### Decision Detectors
| Detector | Trigger | Classifies |
|----------|---------|-----------|
| `shot_vs_pass` | Puck velocity spike in offensive zone | Shot, pass, or dump |
| `zone_entry` | Puck crosses neutral to offensive zone | Carry, dump-in, or pass-in |
| `breakout` | Puck exits defensive zone | Carry, rim, direct pass, or chip |

### Play Evaluation
Each detected decision is rated good/warning/poor based on:
- YAML-defined thresholds (shooting lane quality, open teammates, entry speed, D-zone time)
- Context factors (defenders blocking, support in zone, breakout success)
- Team play style bias (adjustable per team's philosophy)

### Team Play Styles
| Style | Favors | Penalizes |
|-------|--------|-----------|
| `balanced` | No bias — pure outcome evaluation | — |
| `possession` | Carry, pass, controlled play | Dump-ins, chips |
| `physical` | Dump & chase, shots, aggression | Passing, carrying |
| `speed` | Quick transitions, stretch passes | Dumps, slow plays |
| `defensive` | Safe clears, chips, low risk | Carries under pressure |

### Broadcast Film Handling
- Camera cut detection via HSV histogram comparison
- Non-gameplay flagging when too few objects detected
- Tracker automatically resets IDs on camera cuts

## Output

Each run produces:
- `output/<name>_analyzed.mp4` — annotated video with overlays
- `output/<name>_report.txt` — coaching report with evaluations and drill recommendations
- `output/<name>_report.json` — structured data for further analysis

## Project Structure

```
Hockey_AI/
├── main.py                              # CLI entry point
├── config/
│   ├── skating_mechanics.yaml           # Angle thresholds (coach-tunable)
│   └── drill_library.yaml               # Drill recommendations
├── knowledge_base/
│   ├── techniques/                      # 10 YAML technique definitions
│   │   ├── forward_stride.yaml
│   │   ├── crossover.yaml
│   │   ├── wrist_shot.yaml
│   │   └── ...
│   └── game_situations/                 # Game decision evaluation criteria
│       ├── shot_vs_pass.yaml
│       ├── zone_entry.yaml
│       └── breakout.yaml
├── src/
│   ├── pose_estimator.py                # MediaPipe + YOLOv8 backends
│   ├── angle_calculator.py              # Joint angle geometry + head/hand metrics
│   ├── technique_engine.py              # YAML-driven technique evaluation
│   ├── object_tracker.py                # Puck detection via YOLO
│   ├── annotator.py                     # Technique video overlays
│   ├── video_preprocessing.py           # Auto-crop to skater ROI
│   ├── detectors/
│   │   ├── frame_by_frame.py            # Per-frame angle checks
│   │   └── crossover_detector.py        # Temporal crossover detection
│   └── game_analysis/                   # Game film analysis module
│       ├── game_tracker.py              # Multi-player tracking (ByteTrack)
│       ├── zone_detector.py             # Ice zone classification
│       ├── possession_detector.py       # Puck possession detection
│       ├── play_evaluator.py            # Decision scoring + play style bias
│       ├── game_annotator.py            # Game analysis video overlays
│       ├── game_report.py               # Game analysis reports
│       ├── broadcast_filter.py          # Camera cut + replay detection
│       └── decisions/                   # Decision detector registry
│           ├── shot_vs_pass.py
│           ├── zone_entry.py
│           └── breakout.py
├── models/                              # Model weights (not in git)
├── data/raw_videos/                     # Input videos (not in git)
├── demos/                               # Sample output videos + reports
└── output/                              # Generated output (not in git)
```

## Requirements

- Python 3.10+
- OpenCV, MediaPipe, Ultralytics, NumPy, SciPy, PyYAML, lapx
- HockeyAI model weights (download from HuggingFace)
- Optional: NVIDIA GPU for faster inference

## Adding Your Own Knowledge

### New Technique
Create a YAML file in `knowledge_base/techniques/`:
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

### New Game Decision Type
1. Create a detector in `src/game_analysis/decisions/`
2. Add to `DECISION_REGISTRY` in `decisions/__init__.py`
3. Create evaluation YAML in `knowledge_base/game_situations/`

### Custom Play Style
Add to `PLAY_STYLES` dict in `src/game_analysis/play_evaluator.py` with decision biases per event type.
