# Hockey Skating Technique Analyzer

Computer vision tool that analyzes hockey skating video and provides coaching feedback on skating mechanics. Built by a professional hockey player and skating coach.

## What It Does

Processes skating video through pose estimation, biomechanical analysis, and movement-specific evaluation to produce:

- **Annotated video** with visual error highlights on problem joints (not noise on everything)
- **Stride analysis** — detects individual strides, measures push-off extension, glide knee bend, L/R symmetry
- **Crossover analysis** — evaluates knee drive, internal rotation, and step-out explosiveness
- **Coaching report** (text + JSON) with drill recommendations for identified issues

## Quick Start

```bash
pip install -r requirements.txt

# Basic analysis (MediaPipe backend)
python main.py -i video.mp4

# For distant/wide-angle footage (rink cameras, Instagram clips)
python main.py -i video.mp4 --auto-crop

# Use YOLOv8-Pose backend (better detection, needs more compute)
python main.py -i video.mp4 --backend yolo

# All options
python main.py -i video.mp4 -o output/analyzed.mp4 --backend yolo --auto-crop --skeleton-only
```

## Output

Each run produces three files:
- `output/<name>_analyzed.mp4` — annotated video
- `output/<name>_analyzed_report.txt` — coaching report with drill recommendations
- `output/<name>_analyzed_report.json` — structured data for further analysis

## Features

### Pose Estimation Backends
| Backend | Landmarks | Best For | Speed |
|---------|-----------|----------|-------|
| `mediapipe` (default) | 33 (incl. feet) | Single skater, CPU | ~10 fps |
| `yolo` | 17 (COCO) | Multi-person, GPU | ~30 fps (GPU) |

### Biomechanical Analysis
- **Knee bend** — power position depth (95-125 ideal)
- **Hip hinge** — forward lean from hips (70-110 ideal)
- **Forward lean** — torso angle vs vertical (30-50 ideal)
- **Ankle dorsiflexion** — edge engagement (70-95 ideal, MediaPipe only)
- **Trunk alignment** — lateral balance (within 8 degrees)

### Stride Detection
- Detects push-off, glide, and recovery phases from knee angle time series
- Per-stride metrics: extension angle, glide depth, range of motion
- Left/right symmetry ratio

### Crossover Analysis
Evaluates crossovers against coaching criteria:
- **Knee drive** — knee and toe should lead the crossover before the foot crosses
- **Internal rotation** — leg should rotate inward before crossing
- **Step-out** — explosive push after the crossover

### Auto-Crop
For distant/wide-angle footage where the skater is small in frame:
- YOLO person detection finds and tracks the skater
- Crops and upscales to fill the frame
- Smooth tracking with exponential bbox smoothing

### Visual Annotations
- Subtle skeleton that doesn't obscure the skater
- Only problem joints get highlighted (red rings, labels)
- Good mechanics get minimal or no annotation
- Crossover events show movement-specific feedback ("DRIVE KNEE!")
- HUD panel only shows issues, not a wall of green checkmarks

## Project Structure

```
Hockey_AI/
├── main.py                          # CLI entry point
├── config/
│   ├── skating_mechanics.yaml       # Angle thresholds (coach-tunable)
│   └── drill_library.yaml           # Drill recommendations per issue
├── src/
│   ├── pose_estimator.py            # MediaPipe + YOLOv8 backends
│   ├── angle_calculator.py          # Joint angle geometry
│   ├── smoothing.py                 # Kalman filter per landmark
│   ├── mechanics_engine.py          # Threshold evaluation (frame + stride level)
│   ├── stride_detector.py           # Stride phase detection
│   ├── crossover_analyzer.py        # Crossover detection + quality analysis
│   ├── annotator.py                 # Video overlay rendering
│   ├── report_generator.py          # Text + JSON report output
│   ├── video_io.py                  # Frame generator, video writer
│   ├── video_preprocessing.py       # Auto-crop to skater ROI
│   └── utils.py                     # Shared helpers
├── tests/
├── models/                          # Pose model weights (not in git)
├── data/raw_videos/                 # Input videos (not in git)
└── output/                          # Annotated videos + reports (not in git)
```

## Configuration

### Tuning Thresholds
Edit `config/skating_mechanics.yaml` to adjust what counts as good/warning/poor for each mechanic. Ranges are in degrees.

### Adding Drills
Edit `config/drill_library.yaml` to map mechanic deficiencies to specific drills with coaching cues.

## Requirements

- Python 3.10+
- OpenCV, MediaPipe, Ultralytics (YOLOv8), NumPy, SciPy, PyYAML
- MediaPipe model file in `models/` (downloaded separately)
- Optional: NVIDIA GPU for faster YOLOv8 inference

## Roadmap

- [ ] Video comparison mode (side-by-side: you vs. elite technique)
- [ ] Batch processing for multiple videos
- [ ] More movement analyzers (backwards skating, transitions, stops)
- [ ] Real-time webcam mode
- [ ] Web dashboard for session history
