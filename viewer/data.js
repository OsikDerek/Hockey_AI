// JSON loader + per-frame query.
//
// JSON shape (from Python pipeline):
// {
//   "rink": {"length_ft": 200, "width_ft": 85},
//   "fps": 30,
//   "calibration_quality": {...},
//   "frames": [{
//     "frame_idx": 0,
//     "timestamp_sec": 0.0,
//     "calibrated": true,
//     "is_gameplay": true,
//     "puck": {ice_x, ice_y, confidence} | null,
//     "players": [{track_id, team, ice_x, ice_y, confidence}],
//     "goalies": [...]
//   }]
// }

export class PositionsData {
  constructor(payload) {
    this.rink = payload.rink;
    this.fps = payload.fps ?? 30;
    this.calibrationQuality = payload.calibration_quality ?? null;
    this.frames = payload.frames || [];
    // Build a mapping from track_id -> the team it most often appears with
    this._dominantTeam = new Map();
    const tally = new Map();
    for (const fr of this.frames) {
      for (const p of fr.players || []) {
        if (!p.team) continue;
        const key = p.track_id;
        if (!tally.has(key)) tally.set(key, { team_a: 0, team_b: 0 });
        tally.get(key)[p.team] = (tally.get(key)[p.team] || 0) + 1;
      }
    }
    for (const [tid, t] of tally) {
      this._dominantTeam.set(tid, t.team_a >= t.team_b ? "team_a" : "team_b");
    }
  }

  get totalFrames() { return this.frames.length; }

  frameAt(idx) {
    return this.frames[Math.max(0, Math.min(this.frames.length - 1, idx))];
  }

  /** Track ids that appear at least once. Useful for the POV picker. */
  knownPlayerIds() {
    const ids = new Set();
    for (const fr of this.frames) {
      for (const p of fr.players || []) {
        if (p.track_id !== undefined && p.track_id >= 0) ids.add(p.track_id);
      }
    }
    return [...ids].sort((a, b) => a - b);
  }

  dominantTeamFor(trackId) {
    return this._dominantTeam.get(trackId) ?? "unknown";
  }
}

export async function loadFromUrl(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
  return new PositionsData(await resp.json());
}

export async function loadFromFile(file) {
  const text = await file.text();
  return new PositionsData(JSON.parse(text));
}
