"""Render top-down PNG snapshots at each quiz-eligible event in a positions JSON.

Lets us verify avatar placement at the moments that actually matter for
the Phase D Quiz — without needing browser screenshots. Each PNG shows
the full NHL rink, all players colored by team, the puck, and a halo on
the event's actor (the player making the decision).

Usage:
    python scripts/render_play_frames.py output/<basename>_positions.json [--max 8]

Output: PNGs at output/_quiz_frames/<basename>_evN_<type>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


RINK_LENGTH_FT = 200.0
RINK_WIDTH_FT = 85.0
LEFT_GOAL_X = 11.0
RIGHT_GOAL_X = 189.0
BLUE_LEFT_X = 75.0
BLUE_RIGHT_X = 125.0
CENTER_X = 100.0
CENTER_Y = 42.5

# Event types worth visualizing — quiz-eligible only.
QUIZ_EVENT_TYPES = {
    "shot_vs_pass", "odd_man_rush", "zone_entry", "breakout", "missed_opportunity",
}

TEAM_COLOR = {
    "team_a": "#2ad65a",
    "team_b": "#2a82d6",
    None: "#888888",
}


def draw_rink(ax) -> None:
    """Draw the static NHL rink onto an axes."""
    # Ice background
    ax.add_patch(Rectangle((0, 0), RINK_LENGTH_FT, RINK_WIDTH_FT,
                           facecolor="#f4f7fa", edgecolor="#5a3b1c",
                           linewidth=2.0, zorder=0))
    # Goal lines (red)
    for x in (LEFT_GOAL_X, RIGHT_GOAL_X):
        ax.plot([x, x], [0, RINK_WIDTH_FT], color="#cc1f1f", linewidth=1.2, zorder=1)
    # Blue lines
    for x in (BLUE_LEFT_X, BLUE_RIGHT_X):
        ax.plot([x, x], [0, RINK_WIDTH_FT], color="#2870e0", linewidth=2.0, zorder=1)
    # Red center line
    ax.plot([CENTER_X, CENTER_X], [0, RINK_WIDTH_FT], color="#cc1f1f", linewidth=2.0, zorder=1)
    # Center ice circle
    ax.add_patch(plt.Circle((CENTER_X, CENTER_Y), 15.0, fill=False,
                            edgecolor="#2870e0", linewidth=1.5, zorder=1))
    # Faceoff dots (reduced spec — 8 dots + center)
    dots_ft = [
        (20, 20.5), (20, 64.5), (75, 20.5), (75, 64.5),
        (125, 20.5), (125, 64.5), (180, 20.5), (180, 64.5),
        (CENTER_X, CENTER_Y),
    ]
    for x, y in dots_ft:
        ax.add_patch(plt.Circle((x, y), 1.0, color="#cc1f1f", zorder=2))

    ax.set_xlim(-6, RINK_LENGTH_FT + 6)
    ax.set_ylim(-6, RINK_WIDTH_FT + 6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_held_state(frames: list, target_idx: int) -> tuple:
    """Build a 'frozen tableau' for target_idx using last-known positions.

    Decision events fire on uncalibrated frames (puck-velocity / possession
    signals work without ice coords) but the viewer's avatars persist
    across uncalibrated stretches via STALE_FRAMES hold-last-position. This
    helper does the same for the static diagnostic — for each track_id
    that EVER appeared before target_idx, take its most recent position.

    Returns (held_frame_dict, source_offset). The held_frame_dict has the
    same shape as a payload['frames'] entry but assembled from many frames.
    source_offset is the negative offset of the most-recent calibrated
    frame contributing to the snapshot — useful for the title bar.
    """
    if not frames or target_idx >= len(frames):
        return None, 0
    held_players = {}  # track_id -> player dict
    held_goalies = {}
    held_puck = None
    most_recent_calib = -1
    for i in range(target_idx + 1):  # inclusive
        fr = frames[i]
        if not fr.get("calibrated"):
            continue
        most_recent_calib = i
        for p in fr.get("players") or []:
            held_players[p["track_id"]] = dict(p)
        for g in fr.get("goalies") or []:
            held_goalies[g["track_id"]] = dict(g)
        if fr.get("puck"):
            held_puck = dict(fr["puck"])
    if most_recent_calib < 0:
        return None, 0
    held = {
        "calibrated": True,
        "is_gameplay": frames[target_idx].get("is_gameplay", True),
        "players": list(held_players.values()),
        "goalies": list(held_goalies.values()),
        "puck": held_puck,
    }
    return held, most_recent_calib - target_idx


def render_event(payload: dict, event: dict, out_path: Path) -> bool:
    """Render one event frame to a PNG. Returns True on success."""
    frame_idx = event["frame_idx"]
    if frame_idx >= len(payload["frames"]):
        return False

    # Build a frozen-tableau snapshot using the union of last-known
    # positions across all calibrated frames up to target_idx. Mirrors
    # the viewer's STALE_FRAMES hold-last-position behavior.
    fr, fr_offset = build_held_state(payload["frames"], frame_idx)
    if fr is None:
        # No prior calibration at all — render the metadata-only frame
        # so we at least see the event existed and was uncalibrated.
        fr = payload["frames"][frame_idx]
        fr_offset = 0

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=110)
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    draw_rink(ax)

    actor_id = event.get("player_id")
    actor_pos = None

    # Players
    for p in fr.get("players", []):
        col = TEAM_COLOR.get(p.get("team"))
        x, y = p["ice_x"], p["ice_y"]
        is_actor = actor_id is not None and p["track_id"] == actor_id
        if is_actor:
            actor_pos = (x, y)
            # Halo
            ax.add_patch(plt.Circle((x, y), 4.5, fill=False,
                                    edgecolor="#ffe54a", linewidth=2.5, zorder=4))
        ax.add_patch(plt.Circle((x, y), 2.2, color=col, ec="#000", lw=0.8, zorder=5))
        ax.text(x, y - 0.3, f"#{p['track_id']}", ha="center", va="center",
                fontsize=6, color="#fff", zorder=6)

    # Goalies
    for g in fr.get("goalies", []):
        col = TEAM_COLOR.get(g.get("team"))
        x, y = g["ice_x"], g["ice_y"]
        ax.add_patch(plt.Circle((x, y), 3.0, color=col, ec="#000", lw=1.0, zorder=5))
        ax.text(x, y, "G", ha="center", va="center",
                fontsize=8, color="#fff", weight="bold", zorder=6)

    # Puck
    if fr.get("puck"):
        ax.add_patch(plt.Circle((fr["puck"]["ice_x"], fr["puck"]["ice_y"]),
                                1.2, color="#ff7a18", ec="#000", lw=0.8, zorder=7))

    # Decision-arrow visualization for shot-shaped events
    decision = (event.get("decision_made") or "").lower()
    is_shot = (
        (event["event_type"] == "shot_vs_pass" and decision == "shot")
        or (event["event_type"] == "missed_opportunity" and decision == "missed_shot")
        or (event["event_type"] == "odd_man_rush" and decision == "shot")
    )
    if is_shot and actor_pos is not None:
        ax_x, ax_y = actor_pos
        target_x = RIGHT_GOAL_X if ax_x < CENTER_X else LEFT_GOAL_X
        ax.add_patch(FancyArrowPatch(
            (ax_x, ax_y), (target_x, CENTER_Y),
            color="#ffe54a", linewidth=1.4, arrowstyle="->", mutation_scale=15,
            alpha=0.7, zorder=3,
        ))

    # Title bar with event metadata
    rating = event.get("rating") or "neutral"
    offset_tag = ""
    if fr_offset != 0:
        offset_tag = f"  positions from f{frame_idx + fr_offset:+d}"
    title = (
        f"{event['event_type'].replace('_', ' ').upper()} : "
        f"{decision.upper()}  ({rating.upper()})  "
        f"player #{actor_id}  team {event.get('team') or '?'}  "
        f"conf {event.get('confidence', 0):.2f}  "
        f"frame {frame_idx} ({event.get('timestamp_sec', 0):.1f}s)  "
        f"calibrated={payload['frames'][frame_idx].get('calibrated')}{offset_tag}"
    )
    ax.set_title(title, color="#fff", fontsize=10, pad=10)

    # Legend
    handles = [
        mpatches.Patch(color=TEAM_COLOR["team_a"], label="Team A"),
        mpatches.Patch(color=TEAM_COLOR["team_b"], label="Team B"),
        mpatches.Patch(color="#ff7a18", label="Puck"),
        mpatches.Patch(color="#ffe54a", label="Decision-maker"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              facecolor="#000", edgecolor="#444", labelcolor="#fff", framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("json_path", help="Path to *_positions.json")
    p.add_argument("--max", type=int, default=10, help="Max events to render")
    p.add_argument("--out-dir", default=None, help="Output dir (default: output/_quiz_frames)")
    args = p.parse_args(argv)

    json_path = Path(args.json_path).resolve()
    if not json_path.is_file():
        print(f"Not found: {json_path}")
        return 1

    payload = json.loads(json_path.read_text())
    events = [e for e in payload.get("events", []) if e["event_type"] in QUIZ_EVENT_TYPES]
    if not events:
        print("No quiz-eligible events in this clip.")
        return 0

    project_root = json_path.parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else project_root / "output" / "_quiz_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    basename = json_path.stem.replace("_positions", "")
    rendered = 0
    for i, ev in enumerate(events[: args.max]):
        out_path = out_dir / f"{basename}_ev{i:02d}_{ev['event_type']}.png"
        if render_event(payload, ev, out_path):
            rendered += 1
            print(f"  wrote {out_path.relative_to(project_root)}")

    print(f"\nRendered {rendered}/{len(events)} events.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
