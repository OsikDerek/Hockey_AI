"""Post-process a positions JSON to stitch ghost track IDs together.

ByteTrack on hockey footage emits a new track ID every time a player
is briefly occluded, leaves a calibrated frame, or simply has a low-conf
detection. Result: ~12 real players × 17 ghost IDs each = 200+ track IDs
per minute of footage. The viewer caps avatars at 14 to keep the scene
readable, but the underlying data still has identity churn — quiz POV
fails because event.player_id often points at a track that no longer
exists by the time playback reaches it.

This script:
  1. Walks all calibrated frames; collects each track's first/last
     frame, first/last ice (x, y), dominant team, total appearances.
  2. Pairs every (A, B) where A ends before B starts within a max
     temporal gap AND B's first ice position is within a max spatial
     distance of A's last ice position AND teams agree.
  3. Greedy-matches lowest score (distance + small time penalty) first,
     each track consumed at most once per pass.
  4. Builds a union-find over the matched pairs, rewrites every
     `track_id` reference in frames, players, goalies, and events to
     the cluster root.
  5. Writes <basename>_stitched_positions.json beside the input.

Tunables (CLI flags):
  --max-gap N          frames of allowable temporal gap (default 90 = 3s)
  --max-distance D     feet of allowable spatial gap (default 15)
  --min-track N        minimum frames a track must have to be kept
                        (default 0 = keep all)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def stitch(payload: dict, max_gap: int, max_dist: float, min_track_frames: int,
           dedup_radius_ft: float = 4.0) -> dict:
    """Mutates payload in place; returns it."""
    frames = payload.get("frames", [])

    # 0. Within-frame dedup: when two players land within dedup_radius_ft
    # of each other in the SAME calibrated frame, they're almost certainly
    # double-detections of one player. Keep the higher-confidence one.
    # Doing this BEFORE temporal stitching cleans up the input data.
    intra_frame_dropped = 0
    for fr in frames:
        if not fr.get("calibrated"):
            continue
        players = fr.get("players") or []
        if len(players) < 2:
            continue
        # Sort by confidence descending so we keep the best-conf detection
        # in any cluster.
        players.sort(key=lambda p: -float(p.get("confidence", 0)))
        keep = []
        for p in players:
            px, py = p.get("ice_x"), p.get("ice_y")
            if px is None or py is None:
                keep.append(p)
                continue
            duplicate = False
            for k in keep:
                kx, ky = k.get("ice_x"), k.get("ice_y")
                if kx is None or ky is None:
                    continue
                if math.hypot(px - kx, py - ky) < dedup_radius_ft:
                    duplicate = True
                    break
            if duplicate:
                intra_frame_dropped += 1
            else:
                keep.append(p)
        fr["players"] = keep

    # 1. Build per-track stats from calibrated frames. Tag goalies vs
    # players separately so we never stitch a goalie's track to a
    # player's track (or vice versa) — they're different roles + the
    # detector classes already separate them.
    trail = defaultdict(list)  # track_id -> [(frame_idx, x, y, team), ...]
    role = {}  # track_id -> "player" | "goalie"
    for i, fr in enumerate(frames):
        if not fr.get("calibrated"):
            continue
        for p in fr.get("players") or []:
            tid = p.get("track_id")
            if tid is None or tid < 0:
                continue
            trail[tid].append((i, p["ice_x"], p["ice_y"], p.get("team")))
            role[tid] = "player"
        for g in fr.get("goalies") or []:
            tid = g.get("track_id")
            if tid is None or tid < 0:
                continue
            trail[tid].append((i, g["ice_x"], g["ice_y"], g.get("team")))
            role[tid] = "goalie"

    info = {}
    for tid, ts in trail.items():
        ts.sort()
        if len(ts) < min_track_frames:
            continue
        teams = Counter(t[3] for t in ts if t[3])
        dominant_team = teams.most_common(1)[0][0] if teams else None
        info[tid] = {
            "first_frame": ts[0][0],
            "last_frame": ts[-1][0],
            "first_pos": (ts[0][1], ts[0][2]),
            "last_pos": (ts[-1][1], ts[-1][2]),
            "team": dominant_team,
            "n": len(ts),
        }

    # 2. Score every plausible (predecessor, successor) pair.
    candidates = []
    track_ids = list(info.keys())
    for a in track_ids:
        for b in track_ids:
            if a == b:
                continue
            ia, ib = info[a], info[b]
            if ia["last_frame"] >= ib["first_frame"]:
                continue
            gap = ib["first_frame"] - ia["last_frame"]
            if gap > max_gap:
                continue
            d = euclid(ia["last_pos"], ib["first_pos"])
            if d > max_dist:
                continue
            # Team disagreement only blocks when both are confidently labeled.
            if ia["team"] and ib["team"] and ia["team"] != ib["team"]:
                continue
            # Never merge across role boundaries (player <-> goalie).
            if role.get(a) != role.get(b):
                continue
            # Score: spatial dominates, small time penalty as tiebreaker
            score = d + 0.05 * gap
            candidates.append((score, a, b))

    candidates.sort()

    # 3. Greedy 1-to-1 matching, then propagate via union-find.
    parent = {tid: tid for tid in info}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Keep the older (lower first_frame) track's id as the canonical
        if info[ra]["first_frame"] <= info[rb]["first_frame"]:
            parent[rb] = ra
        else:
            parent[ra] = rb

    used_pred = set()
    used_succ = set()
    merges = 0
    for score, a, b in candidates:
        # The successor must be unused (one ghost shouldn't match many).
        # Predecessor can chain: A → B, then later C → A (C will end before A).
        if b in used_succ:
            continue
        if a in used_pred:
            continue
        used_pred.add(a)
        used_succ.add(b)
        union(a, b)
        merges += 1

    # 4. Iterate union-find paths to canonical roots.
    canonical = {tid: find(tid) for tid in info}

    def remap(tid):
        if tid is None:
            return tid
        return canonical.get(tid, tid)

    # 5. Rewrite all track_id references.
    for fr in frames:
        for p in fr.get("players") or []:
            if "track_id" in p:
                p["track_id"] = remap(p["track_id"])
        for g in fr.get("goalies") or []:
            if "track_id" in g:
                g["track_id"] = remap(g["track_id"])
    for ev in payload.get("events") or []:
        if ev.get("player_id") is not None:
            ev["player_id"] = remap(ev["player_id"])

    # Stats for the writer
    surviving_tracks = len(set(canonical.values()))
    payload["track_stitching"] = {
        "input_track_count": len(info),
        "merges_applied": merges,
        "surviving_tracks": surviving_tracks,
        "intra_frame_dropped": intra_frame_dropped,
        "max_gap_frames": max_gap,
        "max_distance_ft": max_dist,
        "min_track_frames": min_track_frames,
    }
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("json_path")
    p.add_argument("--max-gap", type=int, default=90)
    p.add_argument("--max-distance", type=float, default=15.0)
    p.add_argument("--min-track", type=int, default=0)
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args()

    src = Path(args.json_path)
    payload = json.loads(src.read_text())
    before = len({tid for fr in payload.get("frames", [])
                  for p in (fr.get("players") or []) for tid in [p.get("track_id")]
                  if tid is not None and tid >= 0})

    stitch(payload, args.max_gap, args.max_distance, args.min_track)

    after_tracks = payload["track_stitching"]["surviving_tracks"]
    merges = payload["track_stitching"]["merges_applied"]
    print(f"Tracks: {before} -> {after_tracks}  (merged {merges} pairs)")

    if args.output:
        out = Path(args.output)
    else:
        out = src.parent / src.name.replace("_positions.json", "_stitched_positions.json")
    out.write_text(json.dumps(payload))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
