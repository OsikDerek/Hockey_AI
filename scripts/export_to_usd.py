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

TEAM_COLORS = {
    "team_a": (0.16, 0.86, 0.20),
    "team_b": (0.18, 0.40, 0.92),
    "unknown": (0.62, 0.62, 0.62),
}
PUCK_COLOR = (0.05, 0.05, 0.05)

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
                z_m: float, custom: dict, child_lines=None) -> list:
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
    args = ap.parse_args()

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

    # Players -- one Xform per ByteTrack id, each containing a capsule.
    L.append('    def Scope "Players"')
    L.append("    {")
    for tid in sorted(players):
        meta = players[tid].pop("_meta")
        color = TEAM_COLORS.get(meta.get("team", "unknown"), TEAM_COLORS["unknown"])
        cap = _capsule_lines(PLAYER_RADIUS_M, PLAYER_HEIGHT_M, color)
        for line in _emit_xform(f"p{tid}", players[tid], n_frames,
                                z_m=0.0, custom=meta, child_lines=cap):
            L.append("    " + line)
    L.append("    }")
    L.append("")

    # Goalies -- same but a wider radius to suggest pads.
    if goalies:
        L.append('    def Scope "Goalies"')
        L.append("    {")
        for tid in sorted(goalies):
            meta = goalies[tid].pop("_meta")
            color = TEAM_COLORS.get(meta.get("team", "unknown"), TEAM_COLORS["unknown"])
            cap = _capsule_lines(GOALIE_RADIUS_M, PLAYER_HEIGHT_M * 0.95, color)
            for line in _emit_xform(f"g{tid}", goalies[tid], n_frames,
                                    z_m=0.0, custom=meta, child_lines=cap):
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
