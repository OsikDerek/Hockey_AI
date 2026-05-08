---
name: Hockey_AI project overview
description: Two-mode hockey CV project (technique analysis + broadcast game analysis) with HockeyAI YOLO + MediaPipe on GTX 1070 CUDA 12.4; canonical test clips and environment details
type: project
---

**Project:** Hockey_AI — computer vision for hockey technique + broadcast game film analysis.

**Location:** `C:\Users\17742\OneDrive\Documents\PersonalProjects\Hockey_AI` (git: OsikDerek/Hockey_AI, private)
**Python:** `C:/Users/17742/AppData/Local/Programs/Python/Python312/python.exe`
**GPU:** GTX 1070, CUDA 12.4, PyTorch 2.6 (~13 fps). Power settings set to not sleep.

**Two modes via `main.py`:**
1. **Technique mode** (`--technique <name>`) — 10 YAML-defined skills in `knowledge_base/techniques/` (forward_stride, crossover, wrist_shot, snap_shot, one_timer, stickhandling, hockey_stop, backwards_skating, transitions, edge_work). MediaPipe/YOLO pose + event detection.
2. **Game analysis mode** (`--game-analysis --play-style <style>`) — broadcast film analysis. Styles: balanced, possession, physical, speed, defensive. Pipeline: HockeyAI YOLOv8 + ByteTrack (7 classes) → zone detection → possession with hysteresis → BGR-median team classifier → 7 decision detectors (shot_vs_pass, zone_entry, breakout, odd_man_rush, forecheck, defensive_play, missed_opportunity). Coaching overlays: teammate connections, passing/shooting lanes with success %, open-ice markers (gap-based, not shaded), 19-ray goalie sight-line analysis for all 7 hockey holes. Decision freeze frames: 2s pause with vignette and suggestion overlay.

**Canonical test clips in `data/raw_videos/`:**
- `rush_30sec_clip.mp4` (23MB, ~2min GPU time) — dense action, primary smoke test
- `shootout_60sec.mp4` (43MB) — 1-on-1s, best for goalie analysis
- `ig_1v1_beating_guys.mp4` (14MB, 79s) — broadcast 1v1
- `ig_game_edgework.mp4` (1.4MB, 7s) — fast smoke test

**Why:** Derek wants a coaching tool that reflects real hockey tactics, not just CV-level tracking. Tactical correctness > visual polish.

**How to apply:** When working on this project, default to running on GPU, use `rush_30sec_clip.mp4` for batch smoke tests, and touch all three canonical clips on final verification runs.
