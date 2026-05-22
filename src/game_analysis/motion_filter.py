"""Post-pass motion-consistency filter for the puck and player/goalie tracks.

YOLO + ByteTrack produce a *detection* per frame; that detection being
present is not the same as it being *correct*. A false-positive puck or a
mis-projected player jumps frame-to-frame in ways no real object physically
can. This module runs after pass 1, over the full buffered trajectory, and
enforces physical plausibility.

Puck -- a global trajectory selection. Every frame buffers ALL on-ice puck
  candidates (not just the one the streaming PuckFilter picked). A dynamic
  program picks, across the whole clip, the sequence of candidates that
  forms the smoothest physically-plausible path:

    - a frame is either a real candidate, an interpolated bridge frame, or
      "no puck";
    - two real candidates may only be linked (directly or by an interpolated
      bridge) when the speed the link implies is physically possible -- this
      is what rejects a high-confidence false positive that teleports;
    - an isolated candidate that links to nothing earns no reward and is
      dropped in favour of "no puck", so a lone false positive disappears;
    - a linked run is rewarded in proportion to its detections' confidence.

  Because the link motion test spans the whole gap, an interpolated bridge
  can never represent an impossible jump.

Players / goalies -- per-track outlier rejection. Each ByteTrack id is a
  trajectory; an isolated position that jumps away from and back to its
  neighbours (a return-spike) is replaced by the interpolated midpoint.
  Conservative: only isolated single-frame outliers are touched, never
  sustained segments (those are id-continuity errors, a separate problem).

Everything operates in ice coordinates (feet), where "plausible" is a
physical statement, and writes corrected values back into the frame
contexts so both the positions JSON and the rendered video see clean data.
"""

import math

from .game_context import TrackedObject

# Physical ceilings, ft/s. Faster than this between two frames cannot be the
# same real object -- it is a tracking error. Mirrors scripts/verify_motion.py.
PUCK_MAX_FT_S = 165.0     # ~112 mph; the hardest NHL shots sit near 108 mph
PLAYER_MAX_FT_S = 48.0    # ~33 mph
GOALIE_MAX_FT_S = 40.0

# --- Puck trajectory DP costs (smaller = preferred) -------------------------
MISS_COST = 1.0           # cost of leaving a frame with no puck
STATE_COST = 1.15         # cost of placing a real detection in a frame; above
#                           MISS_COST so a candidate that links to nothing
#                           (an isolated false positive) loses to "no puck".
INTERP_COST = 0.5         # cost of an interpolated bridge frame
LINK_W = 6.0              # link reward scale; reward = -LINK_W * min(conf)
COMFORT_FT_S = 75.0       # puck motion below this is unpenalised (a pass)
MOTION_W = 0.05           # ramp penalty per ft/s above COMFORT
MAX_BRIDGE_GAP = 10       # never interpolate a chosen-path gap longer than this
MAX_LOOKBACK = 120        # frames; a longer puck blackout is treated as fresh

# Return-spike thresholds, feet of single-frame displacement. A spike must
# exceed this on both sides AND have its neighbours closer to each other than
# to the suspect point, so a genuine fast stride is not flagged.
PLAYER_SPIKE_FT = 3.5
GOALIE_SPIKE_FT = 2.5

INF = float("inf")


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------- puck DP
def _select_puck_trajectory(contexts, fps: float) -> list:
    """Gap-aware dynamic program over per-frame puck candidates.

    Returns one TrackedObject (real detection or interpolated) or None per
    frame context.
    """
    dt = 1.0 / fps
    n = len(contexts)
    cands = [[c for c in getattr(ctx, "puck_candidates", []) if c.ice_xy is not None]
             for ctx in contexts]
    cuts = [bool(ctx.is_camera_cut) for ctx in contexts]

    def motion_penalty(a_xy, b_xy, gap) -> float:
        """Cost of linking two real pucks `gap` frames apart. INF if the
        implied speed is physically impossible."""
        speed = _dist(a_xy, b_xy) / (gap * dt)
        if speed > PUCK_MAX_FT_S:
            return INF
        if speed <= COMFORT_FT_S:
            return 0.0
        return MOTION_W * (speed - COMFORT_FT_S)

    # dp[f][i] = min cost to cover frames 0..f with the last real puck being
    # candidate i of frame f. pred[f][i] = (f_prev, i_prev, bridged) or None
    # (None = this is the first real puck; frames before it are "no puck").
    #
    # EVERY connection between two real detections is motion-checked over its
    # whole frame gap: an impossible average speed forbids the connection, so
    # neither a direct link nor an interpolated bridge can ever represent a
    # teleport. A connection within MAX_BRIDGE_GAP interpolates the frames
    # between; a longer one leaves them empty (no puck).
    dp = [[INF] * len(cands[f]) for f in range(n)]
    pred = [[None] * len(cands[f]) for f in range(n)]

    for f in range(n):
        for i, c in enumerate(cands[f]):
            best = f * MISS_COST + STATE_COST    # START: 0..f-1 are no-puck
            best_pred = None

            for fp in range(max(0, f - MAX_LOOKBACK), f):
                gap = f - fp
                cut_between = any(cuts[k] for k in range(fp + 1, f + 1))
                bridge = gap <= MAX_BRIDGE_GAP
                fill = (gap - 1) * (INTERP_COST if bridge else MISS_COST)
                for ip, cp in enumerate(cands[fp]):
                    base = dp[fp][ip]
                    if base >= INF:
                        continue
                    if cut_between:
                        # A camera cut breaks puck identity: reconnect with no
                        # motion check and no link reward, frames between
                        # left empty.
                        cost = base + (gap - 1) * MISS_COST + STATE_COST
                        if cost < best:
                            best, best_pred = cost, (fp, ip, False)
                        continue
                    mp = motion_penalty(cp.ice_xy, c.ice_xy, gap)
                    if mp >= INF:
                        continue  # impossible speed -> not the same puck
                    link = -LINK_W * min(cp.confidence, c.confidence)
                    cost = base + fill + STATE_COST + mp + link
                    if cost < best:
                        best, best_pred = cost, (fp, ip, bridge and gap > 1)
            dp[f][i] = best
            pred[f][i] = best_pred

    # Pick the cheapest end state (+ trailing no-puck frames), or all-miss.
    best_end, end = n * MISS_COST, None
    for f in range(n):
        for i in range(len(cands[f])):
            if dp[f][i] >= INF:
                continue
            total = dp[f][i] + (n - 1 - f) * MISS_COST
            if total < best_end:
                best_end, end = total, (f, i)

    chosen = [None] * n
    if end is None:
        return chosen

    # Backtrack, interpolating bridged gaps.
    f, i = end
    while True:
        chosen[f] = cands[f][i]
        p = pred[f][i]
        if p is None:
            break
        fp, ip, bridged = p
        if bridged:
            a, b = cands[fp][ip], cands[f][i]
            for g in range(fp + 1, f):
                t = (g - fp) / (f - fp)
                cx = a.center[0] + t * (b.center[0] - a.center[0])
                cy = a.center[1] + t * (b.center[1] - a.center[1])
                ix = a.ice_xy[0] + t * (b.ice_xy[0] - a.ice_xy[0])
                iy = a.ice_xy[1] + t * (b.ice_xy[1] - a.ice_xy[1])
                chosen[g] = TrackedObject(
                    track_id=-1, class_name="puck", class_id=5,
                    bbox=(cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0),
                    center=(cx, cy), confidence=0.0, ice_xy=(ix, iy),
                )
        f, i = fp, ip
    return chosen


def _write_back_puck(contexts, chosen: list) -> None:
    """Replace each frame's puck (in ctx.puck and ctx.objects)."""
    for ctx, puck in zip(contexts, chosen):
        ctx.objects = [o for o in ctx.objects if o.class_name != "puck"]
        ctx.puck = puck
        if puck is not None:
            ctx.objects.append(puck)


# ------------------------------------------------------ player / goalie
def _filter_tracked(contexts, key: str, spike_ft: float) -> int:
    """Replace isolated return-spike outliers in each ByteTrack id with the
    interpolated midpoint of their neighbours. Operates on ice_xy in place."""
    tracks: dict = {}
    for fi, ctx in enumerate(contexts):
        for o in getattr(ctx, key, []) or []:
            if o.ice_xy is None or o.track_id is None:
                continue
            tracks.setdefault(o.track_id, []).append((fi, o))

    fixed = 0
    for seq in tracks.values():
        seq.sort(key=lambda t: t[0])
        # Two passes so a spike adjacent to a spike still resolves.
        for _ in range(2):
            for i in range(1, len(seq) - 1):
                (fp, op), (fc, oc), (fn, on) = seq[i - 1], seq[i], seq[i + 1]
                if fc - fp != 1 or fn - fc != 1:
                    continue  # need immediate neighbours for a clean test
                d_prev = _dist(oc.ice_xy, op.ice_xy)
                d_next = _dist(oc.ice_xy, on.ice_xy)
                d_skip = _dist(op.ice_xy, on.ice_xy)
                if (d_prev > spike_ft and d_next > spike_ft
                        and d_skip < min(d_prev, d_next)):
                    oc.ice_xy = ((op.ice_xy[0] + on.ice_xy[0]) / 2.0,
                                 (op.ice_xy[1] + on.ice_xy[1]) / 2.0)
                    fixed += 1
    return fixed


# -------------------------------------------------------------- public
def apply_motion_filter(analysis, fps: float, verbose: bool = True) -> dict:
    """Clean the buffered trajectories in `analysis.frame_contexts` in place.

    Returns a stats dict for logging.
    """
    contexts = analysis.frame_contexts
    if not contexts:
        return {}

    puck_before = sum(1 for c in contexts if c.puck is not None)
    chosen = _select_puck_trajectory(contexts, fps)
    puck_after = sum(1 for c in chosen if c is not None)
    interpolated = sum(1 for c in chosen
                       if c is not None and c.confidence == 0.0)
    _write_back_puck(contexts, chosen)

    players_fixed = _filter_tracked(contexts, "players", PLAYER_SPIKE_FT)
    goalies_fixed = _filter_tracked(contexts, "goalies", GOALIE_SPIKE_FT)

    stats = {
        "puck_before": puck_before,
        "puck_after": puck_after,
        "puck_real": puck_after - interpolated,
        "puck_interpolated": interpolated,
        "player_spikes_fixed": players_fixed,
        "goalie_spikes_fixed": goalies_fixed,
    }
    if verbose:
        print(f"  Motion filter: puck {puck_before} -> {puck_after} present "
              f"({stats['puck_real']} real + {interpolated} interpolated); "
              f"player spikes fixed {players_fixed}, "
              f"goalie spikes fixed {goalies_fixed}")
    return stats
