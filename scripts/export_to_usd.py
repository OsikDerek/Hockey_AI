"""Export a positions JSON to a USD scene file (the Omniverse / XR bridge).

USD (Universal Scene Description, openusd.org) is the interchange format
Omniverse, Unreal, Blender, Unity, and Houdini all speak. By emitting our
tracked positions as USD, the same data drives:

  - the existing Three.js viewer (web, fast iteration), AND
  - any USD-aware DCC / engine, including Omniverse XR for the long-term
    VR-decision-sim north star -- just open the USD in Omniverse Composer
    or Kit, reference a hockey rink asset under World/Rink, and skin the
    per-track xforms with character rigs.

What this script produces:
  - World/Rink         -- empty xform placeholder, ready for a rink USD ref.
  - World/Players/p<id> -- one xform per ByteTrack id, with per-frame
                           translate samples in metres (NHL ice in the
                           +X/+Y plane, Z up) and visibility samples toggling
                           the xform off during the frames the track wasn't
                           tracked. Team + confidence go in custom data.
  - World/Goalies/g<id> -- same.
  - World/Puck         -- single xform with the chosen puck trajectory.

Output is .usda (ASCII), self-contained, zero extra Python deps required.
Validate or visualise it in any USD viewer.

Usage:
    .venv/Scripts/python.exe scripts/export_to_usd.py --clip caufield_trim_b3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# NHL ice is laid out in feet; USD scenes use SI metres. 1 ft = 0.3048 m.
FT_TO_M = 0.3048

# A puck on the ice surface; small Z lift so it does not z-fight with the
# rink plane in renderers that draw an ice mesh at Z=0.
PUCK_Z_M = 0.04
PUCK_R_M = 0.038
PUCK_H_M = 0.025

# Player avatar: a vertical capsule standing on the ice (~6 ft skater).
PLAYER_HEIGHT_M = 1.83
PLAYER_RADIUS_M = 0.28
GOALIE_RADIUS_M = 0.40   # wider profile for pads

# Rink dimensions for scene metadata (NHL spec; matches rink.js + calibrator).
RINK_LENGTH_FT = 200.0
RINK_WIDTH_FT = 85.0
RINK_CORNER_RADIUS_FT = 28.0

DEFAULT_TEAM_COLORS = {
    "team_a": (0.16, 0.86, 0.20),
    "team_b": (0.18, 0.40, 0.92),
    "unknown": (0.62, 0.62, 0.62),
}
PUCK_COLOR = (0.05, 0.05, 0.05)
SKIN_COLOR = (0.94, 0.79, 0.69)
STICK_COLOR = (0.42, 0.26, 0.13)
BLADE_COLOR = (0.20, 0.20, 0.22)

# Path (relative to the output USD) of the rink asset to reference under
# World/Rink. Built by scripts/build_rink_usd.py.
RINK_ASSET_RELPATH = "../assets/usd/rink_nhl.usda"


def _runs(present_frames: list) -> list:
    """Group a sorted list of integer frame indices into maximal runs of
    consecutive frames. Returns [(start, end_inclusive), ...]."""
    if not present_frames:
        return []
    runs = []
    s = e = present_frames[0]
    for f in present_frames[1:]:
        if f == e + 1:
            e = f
        else:
            runs.append((s, e))
            s = e = f
    runs.append((s, e))
    return runs


def _capsule_lines(radius: float, height: float, color) -> list:
    """A capsule standing on the ice at z=0 to z=height, axis Z."""
    return [
        '        def Capsule "Avatar"',
        "        {",
        f"            double radius = {radius}",
        f"            double height = {max(0.01, height - 2 * radius)}",
        '            uniform token axis = "Z"',
        f"            double3 xformOp:translate = (0, 0, {height / 2})",
        '            uniform token[] xformOpOrder = ["xformOp:translate"]',
        f"            color3f[] primvars:displayColor = "
        f"[({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})]",
        "        }",
    ]


def _humanoid_lines(team_color, height: float = PLAYER_HEIGHT_M,
                    radius: float = PLAYER_RADIUS_M,
                    is_goalie: bool = False) -> list:
    """Stylised player: body capsule (team) + skin head + stick (skater)
    or wider pads (goalie). All sub-prims in the player's LOCAL space, so
    the parent xform's translate just slides the whole figure on the ice.
    The skater faces local +X (no per-frame yaw yet -- that's the next step
    once we add velocity-based facing)."""
    body_h = height * 0.78           # ~1.4 m torso+legs
    head_r = 0.17 if not is_goalie else 0.15
    head_z = body_h + head_r * 0.6
    L = []
    # Body
    L += [
        '        def Capsule "Body"',
        "        {",
        f"            double radius = {radius}",
        f"            double height = {max(0.01, body_h - 2 * radius)}",
        '            uniform token axis = "Z"',
        f"            double3 xformOp:translate = (0, 0, {body_h / 2})",
        '            uniform token[] xformOpOrder = ["xformOp:translate"]',
        f"            color3f[] primvars:displayColor = "
        f"[({team_color[0]:.3f}, {team_color[1]:.3f}, {team_color[2]:.3f})]",
        "        }",
    ]
    # Head
    L += [
        '        def Sphere "Head"',
        "        {",
        f"            double radius = {head_r}",
        f"            double3 xformOp:translate = (0, 0, {head_z})",
        '            uniform token[] xformOpOrder = ["xformOp:translate"]',
        f"            color3f[] primvars:displayColor = "
        f"[({SKIN_COLOR[0]:.3f}, {SKIN_COLOR[1]:.3f}, {SKIN_COLOR[2]:.3f})]",
        "        }",
    ]
    if not is_goalie:
        # Stick: shaft (long thin cylinder forward + slightly to the side)
        # + blade (small flat box at the end). Local +X is "forward".
        shaft_len = 1.5
        shaft_x = shaft_len / 2 + 0.20  # offset out from the body
        L += [
            '        def Cylinder "StickShaft"',
            "        {",
            "            double radius = 0.025",
            f"            double height = {shaft_len}",
            '            uniform token axis = "X"',
            f"            double3 xformOp:translate = ({shaft_x:.3f}, 0.20, 0.55)",
            '            uniform token[] xformOpOrder = ["xformOp:translate"]',
            f"            color3f[] primvars:displayColor = "
            f"[({STICK_COLOR[0]:.3f}, {STICK_COLOR[1]:.3f}, {STICK_COLOR[2]:.3f})]",
            "        }",
            '        def Cube "StickBlade"',
            "        {",
            "            double size = 1",
            # 0.45 m long blade, ~0.02 thick, 0.06 wide at z~0.04 above ice
            f"            matrix4d xformOp:transform = "
            f"((0.45, 0, 0, 0), (0, 0.06, 0, 0), (0, 0, 0.025, 0),"
            f" ({shaft_x + shaft_len / 2 + 0.18:.3f}, 0.25, 0.04, 1))",
            '            uniform token[] xformOpOrder = ["xformOp:transform"]',
            f"            color3f[] primvars:displayColor = "
            f"[({BLADE_COLOR[0]:.3f}, {BLADE_COLOR[1]:.3f}, {BLADE_COLOR[2]:.3f})]",
            "        }",
        ]
    else:
        # Goalie: wider pad block in front of the body + blocker/glove cubes.
        L += [
            '        def Cube "Pads"',
            "        {",
            "            double size = 1",
            # 0.95 m wide, 0.18 m deep, 0.95 m tall, slightly forward
            f"            matrix4d xformOp:transform = "
            f"((0.95, 0, 0, 0), (0, 0.18, 0, 0), (0, 0, 0.95, 0),"
            f" (0.20, 0, 0.47, 1))",
            '            uniform token[] xformOpOrder = ["xformOp:transform"]',
            f"            color3f[] primvars:displayColor = "
            f"[({team_color[0]:.3f}, {team_color[1]:.3f}, {team_color[2]:.3f})]",
            "        }",
        ]
    return L


def _parse_color(s: str):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3 or not all(0.0 <= p <= 1.0 for p in parts):
        raise ValueError(f"team color must be 'r,g,b' in [0,1], got {s!r}")
    return tuple(parts)


def _compute_yaws(samples_by_frame: dict) -> dict:
    """Per-frame Z-axis yaw (degrees) for a track, from its velocity.

    A central-difference velocity yields a heading per frame; the heading
    is held during near-stationary frames so the avatar doesn't spin when
    standing still. Result drives a rotateZ.timeSamples on the parent
    xform so the stick blade and any POV camera face the velocity dir.
    """
    frames = sorted(samples_by_frame)
    if len(frames) < 2:
        return {}
    yaws = {}
    last_yaw = None
    for i, f in enumerate(frames):
        if 0 < i < len(frames) - 1:
            fp, fn = frames[i - 1], frames[i + 1]
            if fn - fp <= 6:
                x0, y0 = samples_by_frame[fp]
                x1, y1 = samples_by_frame[fn]
                dx, dy = x1 - x0, y1 - y0
                if dx * dx + dy * dy > 0.04:   # > ~0.2 ft, real motion
                    new_yaw = math.degrees(math.atan2(dy, dx))
                    if last_yaw is not None:
                        # blend toward new heading, handling angle wrap
                        d = new_yaw - last_yaw
                        while d > 180: d -= 360
                        while d < -180: d += 360
                        new_yaw = last_yaw + 0.5 * d
                    yaws[f] = new_yaw
                    last_yaw = new_yaw
                    continue
        # held (stationary or endpoint)
        yaws[f] = last_yaw if last_yaw is not None else 0.0
    return yaws


def _pov_camera_lines(name: str = "POV") -> list:
    """A camera inside a player xform; sits at the head, looks down local +X.

    Combined with the parent xform's rotateZ.timeSamples this puts the
    camera at eye height looking where the player is heading.
    """
    return [
        f'        def Camera "{name}"',
        "        {",
        "            float focalLength = 24",
        "            float horizontalAperture = 24",
        "            float2 clippingRange = (0.05, 200)",
        # 0.30 m forward of body centre, 1.55 m up = head height
        "            double3 xformOp:translate = (0.30, 0, 1.55)",
        # default cam looks down local -Z; rotate -90 around Y so it looks down +X
        "            double3 xformOp:rotateXYZ = (0, -90, 0)",
        '            uniform token[] xformOpOrder = '
        '["xformOp:translate", "xformOp:rotateXYZ"]',
        "        }",
    ]


def _broadcast_camera_lines(rink_l_m: float, rink_w_m: float) -> list:
    """A static side-cam at rinkside elevated to mid-stand height, tilted
    down toward centre ice. Pitch = atan2(rink_w/2 + 10, 10) deg around X."""
    cam_x = rink_l_m / 2
    cam_y = -10.0
    cam_z = 10.0
    # delta from cam to rink center (cam_x, rink_w/2, 0):
    dy = rink_w_m / 2 - cam_y
    dz = -cam_z
    pitch = math.degrees(math.atan2(dy, -dz))   # cf. _compute_yaws derivation
    return [
        'def Camera "BroadcastCam"',
        "{",
        "    float focalLength = 35",
        "    float2 clippingRange = (0.1, 300)",
        f"    double3 xformOp:translate = ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})",
        f"    double3 xformOp:rotateXYZ = ({pitch:.3f}, 0, 0)",
        '    uniform token[] xformOpOrder = '
        '["xformOp:translate", "xformOp:rotateXYZ"]',
        "}",
        "",
    ]


def _puck_trail_lines(puck_samples: dict, n_trail: int = 30,
                       w_head: float = 0.18, w_tail: float = 0.02) -> list:
    """A fading line showing the puck's last N positions, time-sampled.

    Single BasisCurves prim with a fixed vertex count (n_trail) and
    per-vertex width interpolating tail->head; the points array is
    re-emitted per frame as the trail slides forward. On the leading
    edge of the clip (fewer than n_trail samples available) the array
    is padded with the oldest available point so the curve always has
    the same vertex count -- a quietly thin trail that grows in over
    the first ~1 s of playback.
    """
    if not puck_samples:
        return []
    frames = sorted(puck_samples)
    # Per-vertex widths: tail (index 0) thin, head (index n_trail-1) thick.
    widths = ", ".join(
        f"{w_tail + (w_head - w_tail) * (i / max(1, n_trail - 1)):.3f}"
        for i in range(n_trail)
    )
    L = [
        'def BasisCurves "PuckTrail"',
        "{",
        '    uniform token type = "linear"',
        f"    int[] curveVertexCounts = [{n_trail}]",
        f"    float[] widths = [{widths}]",
        '    uniform token widthsInterpolation = "vertex"',
        "    color3f[] primvars:displayColor = [(1.00, 0.62, 0.10)]",
        "    point3f[] points.timeSamples = {",
    ]
    trail_z = PUCK_Z_M + 0.06   # just above the puck disc
    for idx, f in enumerate(frames):
        start = max(0, idx - n_trail + 1)
        trail = [puck_samples[frames[j]] for j in range(start, idx + 1)]
        while len(trail) < n_trail:
            trail.insert(0, trail[0])
        pts = ", ".join(
            f"({x * FT_TO_M:.4f}, {y * FT_TO_M:.4f}, {trail_z:.4f})"
            for x, y in trail
        )
        L.append(f"        {f}: [{pts}],")
    L.append("    }")
    L.append("}")
    L.append("")
    return L


def _drone_camera_lines(puck_samples: dict) -> list:
    """A drone hovering above and slightly behind the puck. Static rotation
    (the geometry of being 3 m south + 12 m up of the puck is constant);
    only the translate is time-sampled, matching the puck."""
    if not puck_samples:
        return []
    pitch = math.degrees(math.atan2(3.0, 12.0))  # ~14 deg from straight down
    L = [
        'def Camera "DroneCam"',
        "{",
        "    float focalLength = 35",
        "    float2 clippingRange = (0.1, 300)",
        "    double3 xformOp:translate.timeSamples = {",
    ]
    for f in sorted(puck_samples):
        x_ft, y_ft = puck_samples[f]
        L.append(f"        {f}: ({x_ft * FT_TO_M:.4f},"
                 f" {y_ft * FT_TO_M - 3.0:.4f}, 12.0),")
    L.append("    }")
    L += [
        f"    double3 xformOp:rotateXYZ = ({pitch:.3f}, 0, 0)",
        '    uniform token[] xformOpOrder = '
        '["xformOp:translate", "xformOp:rotateXYZ"]',
        "}",
        "",
    ]
    return L


def _puck_disc_lines() -> list:
    return [
        '        def Cylinder "Disc"',
        "        {",
        f"            double radius = {PUCK_R_M}",
        f"            double height = {PUCK_H_M}",
        '            uniform token axis = "Z"',
        f"            color3f[] primvars:displayColor = "
        f"[({PUCK_COLOR[0]:.3f}, {PUCK_COLOR[1]:.3f}, {PUCK_COLOR[2]:.3f})]",
        "        }",
    ]


def _emit_xform(name: str, samples_by_frame: dict, total_frames: int,
                z_m: float, custom: dict, child_lines=None,
                yaws: dict = None) -> list:
    """Emit one xform prim with translate timeSamples + visibility toggles.

    samples_by_frame: dict[frame_idx] -> (ice_x_ft, ice_y_ft).
    Returns lines of usda text (indented two spaces under the parent prim).
    """
    if not samples_by_frame:
        return []
    present = sorted(samples_by_frame)
    runs = _runs(present)
    # USD customData (team etc.) lives in the prim's METADATA parentheses,
    # not in the body braces -- so build the header with that block when
    # custom data is provided.
    if custom:
        lines = [f'    def Xform "{name}" (', "        customData = {"]
        for k, v in custom.items():
            if isinstance(v, str):
                lines.append(f'            string {k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"            bool {k} = {str(v).lower()}")
            elif isinstance(v, int):
                lines.append(f"            int {k} = {v}")
            elif isinstance(v, float):
                lines.append(f"            double {k} = {v}")
        lines.append("        }")
        lines.append("    )")
        lines.append("    {")
    else:
        lines = [f'    def Xform "{name}"', "    {"]

    # translate time samples
    lines.append("        double3 xformOp:translate.timeSamples = {")
    for f in present:
        x_ft, y_ft = samples_by_frame[f]
        lines.append(
            f"            {f}: ({x_ft * FT_TO_M:.4f}, "
            f"{y_ft * FT_TO_M:.4f}, {z_m:.4f}),"
        )
    lines.append("        }")
    # rotateZ time samples (drives stick + POV camera facing).
    if yaws:
        lines.append("        double xformOp:rotateZ.timeSamples = {")
        for f in sorted(yaws):
            lines.append(f"            {f}: {yaws[f]:.3f},")
        lines.append("        }")
        lines.append('        uniform token[] xformOpOrder = '
                     '["xformOp:translate", "xformOp:rotateZ"]')
    else:
        lines.append('        uniform token[] xformOpOrder = ["xformOp:translate"]')

    # visibility: held-interpolation token. inherited at each run start,
    # invisible at the frame after each run ends.
    if runs[0][0] > 0 or (runs[-1][1] < total_frames - 1) or len(runs) > 1:
        lines.append("        token visibility.timeSamples = {")
        # If the track doesn't start at frame 0, hide from 0 until first run.
        if runs[0][0] > 0:
            lines.append('            0: "invisible",')
        for s, e in runs:
            lines.append(f'            {s}: "inherited",')
            if e + 1 < total_frames:
                lines.append(f'            {e + 1}: "invisible",')
        lines.append("        }")
    # Embedded child geometry (capsule for players, cylinder for puck).
    if child_lines:
        lines.extend(child_lines)
    lines.append("    }")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="caufield_trim_b3")
    ap.add_argument("--positions", default=None,
                    help="positions JSON path (default: output/<clip>_positions.json)")
    ap.add_argument("--out", default=None,
                    help="output USDA path (default: output/<clip>.usda)")
    ap.add_argument("--team-a-color", default=None,
                    help='comma-separated r,g,b in [0,1]; e.g. "0.85,0.12,0.16" '
                         'for red. Override the default team_a colour.')
    ap.add_argument("--team-b-color", default=None,
                    help="same, for team_b.")
    args = ap.parse_args()

    team_colors = dict(DEFAULT_TEAM_COLORS)
    if args.team_a_color:
        team_colors["team_a"] = _parse_color(args.team_a_color)
    if args.team_b_color:
        team_colors["team_b"] = _parse_color(args.team_b_color)

    pos_path = (Path(args.positions) if args.positions
                else PROJECT_ROOT / "output" / f"{args.clip}_positions.json")
    out_path = (Path(args.out) if args.out
                else PROJECT_ROOT / "output" / f"{args.clip}.usda")
    if not pos_path.exists():
        raise SystemExit(f"positions JSON not found: {pos_path}")

    data = json.loads(pos_path.read_text())
    fps = float(data.get("fps", 30.0))
    frames = data["frames"]
    n_frames = len(frames)

    # Gather per-track samples.
    players: dict = {}   # track_id -> {frame: (x, y), '_meta': {...}}
    goalies: dict = {}
    puck_samples: dict = {}
    for fr in frames:
        fi = fr["frame_idx"]
        for p in fr.get("players", []) or []:
            tid = p["track_id"]
            d = players.setdefault(tid, {"_meta": {"team": p.get("team") or "unknown"}})
            d[fi] = (p["ice_x"], p["ice_y"])
        for g in fr.get("goalies", []) or []:
            tid = g["track_id"]
            d = goalies.setdefault(tid, {"_meta": {"team": g.get("team") or "unknown"}})
            d[fi] = (g["ice_x"], g["ice_y"])
        if fr.get("puck"):
            puck_samples[fi] = (fr["puck"]["ice_x"], fr["puck"]["ice_y"])

    # ---- emit USDA ----
    L = []
    L.append("#usda 1.0")
    L.append("(")
    L.append('    defaultPrim = "World"')
    L.append('    upAxis = "Z"')
    L.append("    metersPerUnit = 1")
    L.append(f"    timeCodesPerSecond = {fps}")
    L.append(f"    framesPerSecond = {fps}")
    L.append("    startTimeCode = 0")
    L.append(f"    endTimeCode = {n_frames - 1}")
    L.append("    customLayerData = {")
    L.append(f'        string source = "Hockey_AI pipeline"')
    L.append(f'        string clip = "{args.clip}"')
    L.append(f"        double rink_length_m = {RINK_LENGTH_FT * FT_TO_M:.4f}")
    L.append(f"        double rink_width_m  = {RINK_WIDTH_FT * FT_TO_M:.4f}")
    L.append(f"        double rink_corner_radius_m = {RINK_CORNER_RADIUS_FT * FT_TO_M:.4f}")
    L.append("    }")
    L.append(")")
    L.append("")
    L.append('def Xform "World" (kind = "assembly")')
    L.append("{")

    # Rink — reference the standalone rink asset built by build_rink_usd.py.
    # The asset's defaultPrim ("Rink") provides the ice mesh, lines, dots,
    # nets etc.; swap this reference for a photoreal rink USD when you have
    # one (Omniverse marketplace asset, custom-modeled NHL arena, etc.).
    L.append('    def "Rink" (')
    L.append(f'        prepend references = @{RINK_ASSET_RELPATH}@</Rink>')
    L.append("    )")
    L.append("    {")
    L.append("    }")
    L.append("")

    # Top-3 most-present players get an in-helmet POV camera. Counted on the
    # raw sample dicts before _meta is popped.
    pl_counts = {t: sum(1 for k in players[t] if k != "_meta") for t in players}
    top_pov_ids = sorted(pl_counts, key=lambda t: -pl_counts[t])[:3]

    # Players -- one Xform per ByteTrack id, each containing a humanoid.
    L.append('    def Scope "Players"')
    L.append("    {")
    for tid in sorted(players):
        meta = players[tid].pop("_meta")
        color = team_colors.get(meta.get("team", "unknown"), team_colors["unknown"])
        avatar = _humanoid_lines(color, PLAYER_HEIGHT_M, PLAYER_RADIUS_M,
                                  is_goalie=False)
        if tid in top_pov_ids:
            avatar = avatar + _pov_camera_lines("POV")
        # Per-frame yaw from velocity drives the stick + POV camera facing.
        yaws = _compute_yaws(players[tid])
        for line in _emit_xform(f"p{tid}", players[tid], n_frames,
                                z_m=0.0, custom=meta,
                                child_lines=avatar, yaws=yaws):
            L.append("    " + line)
    L.append("    }")
    L.append("")

    # Goalies -- same but a wider radius to suggest pads.
    if goalies:
        L.append('    def Scope "Goalies"')
        L.append("    {")
        for tid in sorted(goalies):
            meta = goalies[tid].pop("_meta")
            color = team_colors.get(meta.get("team", "unknown"), team_colors["unknown"])
            avatar = _humanoid_lines(color, PLAYER_HEIGHT_M * 0.95,
                                      GOALIE_RADIUS_M, is_goalie=True)
            for line in _emit_xform(f"g{tid}", goalies[tid], n_frames,
                                    z_m=0.0, custom=meta, child_lines=avatar):
                L.append("    " + line)
        L.append("    }")
        L.append("")

    # Puck -- a small black cylinder sitting on the ice.
    for line in _emit_xform("Puck", puck_samples, n_frames,
                            z_m=PUCK_Z_M, custom={},
                            child_lines=_puck_disc_lines()):
        L.append(line)

    # Sun-style overhead light so renderers without a default light have one.
    L.append('    def DistantLight "Sun"')
    L.append("    {")
    L.append("        float inputs:intensity = 2500")
    L.append("        float inputs:angle = 1.5")
    L.append('        double3 xformOp:rotateXYZ = (-55, 0, 35)')
    L.append('        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]')
    L.append("    }")
    L.append("")

    # Default top-down framing camera. Placed above rink centre, looking
    # straight down. A user in Omniverse can switch to a free-fly or VR
    # camera; this one is just a sensible default.
    rink_l_m = RINK_LENGTH_FT * FT_TO_M
    rink_w_m = RINK_WIDTH_FT * FT_TO_M
    L.append('    def Camera "TopDown"')
    L.append("    {")
    L.append("        float focalLength = 35")
    L.append("        float2 clippingRange = (0.1, 200)")
    L.append(f"        double3 xformOp:translate = "
             f"({rink_l_m/2:.3f}, {rink_w_m/2:.3f}, 55.0)")
    L.append("        double3 xformOp:rotateXYZ = (0, 0, 0)")
    L.append('        uniform token[] xformOpOrder = '
             '["xformOp:translate", "xformOp:rotateXYZ"]')
    L.append("    }")
    L.append("")

    # Broadcast side cam: static, rinkside, looking at centre ice.
    for line in _broadcast_camera_lines(rink_l_m, rink_w_m):
        L.append("    " + line if line else "")

    # Drone follow cam: parented to nothing (world-level), translate
    # time-sampled to track the puck.
    for line in _drone_camera_lines(puck_samples):
        L.append("    " + line if line else "")

    # Fading puck trail (last ~1 s of puck history).
    for line in _puck_trail_lines(puck_samples):
        L.append("    " + line if line else "")

    L.append("}")
    L.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L))
    print(f"wrote {out_path}")
    print(f"  {n_frames} frames @ {fps} fps  ({n_frames/fps:.1f} s)")
    print(f"  {len(players)} player tracks, {len(goalies)} goalie tracks, "
          f"puck samples: {len(puck_samples)}")
    print(f"  open in any USD-aware tool (Omniverse Composer / Kit, Blender "
          f"(USD), Houdini, Unreal USD plugin, Unity USD package).")


if __name__ == "__main__":
    main()
