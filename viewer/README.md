# Hockey AI 3D Viewer

Browser-based viewer for the per-frame ice-coordinate positions JSON
produced by the Python pipeline.

## Run it

```
python scripts/serve_viewer.py
```

The script starts a static HTTP server on `localhost:8000`, finds the
most recent `output/*_positions.json` file, and opens the viewer with
that file pre-loaded.

## Camera modes

- **Top-Down** — orthographic view of the rink.
- **Broadcast** — perspective camera at center ice, side angle.
- **POV** — first-person from a chosen player. Pick the player from the
  dropdown next to the POV button.

## Playback

Scrubber, play/pause, and speed dropdown across the bottom. The viewer
holds the most recent valid frame when calibration drops.

## Loading a different file

The file picker in the top-right loads any `*_positions.json` produced
by the Python pipeline.
