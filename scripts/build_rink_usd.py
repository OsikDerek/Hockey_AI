"""Generate a standalone NHL rink USD asset.

Produces assets/usd/rink_nhl.usda -- a self-contained USDA scene with the
ice surface (rounded-rect mesh), boards, painted lines (center red, two
blue lines, two goal lines), the nine faceoff dots, and the two goal
nets. Solid-color via displayColor so it renders in any USD viewer
(Omniverse Composer, Blender USD, Unreal USD, Houdini, omniverse-kit)
without needing UsdShade materials.

Generated once and committed; export_to_usd.py references it from World/Rink.
Re-run only when the rink design itself changes:

    .venv/Scripts/python.exe scripts/build_rink_usd.py
"""

from __future__ import annotations

import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# All dimensions in METRES (NHL spec converted from feet @ 0.3048 m/ft).
FT = 0.3048
RINK_L = 200 * FT          # 60.96 m
RINK_W = 85 * FT           # 25.908 m
CORNER_R = 28 * FT         # 8.5344 m
GOAL_LINE_X_L = 11 * FT    # 3.3528 m
GOAL_LINE_X_R = 189 * FT
BLUE_LINE_L = 75 * FT      # 22.86 m
BLUE_LINE_R = 125 * FT
CENTER_X = 100 * FT        # 30.48 m
CENTER_Y = 42.5 * FT
LINE_W = 0.3048            # 12 in (NHL line width approx)
FACEOFF_DOT_R = 0.1524     # 6 in
GOAL_W = 6 * FT            # 1.83 m wide
GOAL_D = 4 * FT            # 1.22 m deep
GOAL_H = 1.22              # 4 ft tall
ICE_Z = 0.0
PAINT_Z = 0.01             # just above ice to avoid z-fight
BOARD_H = 1.07             # 42 in glass+boards

ICE_COLOR = (0.94, 0.96, 0.98)
RED = (0.85, 0.12, 0.16)
BLUE = (0.13, 0.43, 0.83)
BOARD_COLOR = (0.94, 0.88, 0.65)
NET_COLOR = (0.95, 0.95, 0.95)
NET_POST = (0.80, 0.07, 0.11)


def rounded_rect_outline(length, width, radius, n_arc=12):
    """Return a list of (x, y) points around a centered rounded rectangle."""
    hl, hw = length / 2, width / 2
    pts = []
    # corner centers (cx, cy) and the arc's start angle (radians)
    corners = [
        (hl - radius, -hw + radius, -math.pi / 2),  # bottom-right
        (hl - radius,  hw - radius,  0.0),          # top-right
        (-hl + radius, hw - radius,  math.pi / 2),  # top-left
        (-hl + radius, -hw + radius, math.pi),       # bottom-left
    ]
    # start at (-hl+radius, -hw) -- the bottom-left straight start
    pts.append((-hl + radius, -hw))
    pts.append((hl - radius, -hw))
    for i, (cx, cy, a0) in enumerate(corners):
        # quarter-circle arc
        for k in range(1, n_arc + 1):
            a = a0 + (math.pi / 2) * (k / n_arc)
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        # straight edge after this corner -- already implicitly drawn to next start
    return pts


def fan_mesh_from_outline(outline_pts, z):
    """Triangle-fan mesh: center vertex + outline. Returns (points, face_counts, face_indices)."""
    cx = sum(p[0] for p in outline_pts) / len(outline_pts)
    cy = sum(p[1] for p in outline_pts) / len(outline_pts)
    points = [(cx, cy, z)] + [(x, y, z) for x, y in outline_pts]
    counts = []
    indices = []
    n = len(outline_pts)
    for i in range(n):
        counts.append(3)
        indices.extend([0, 1 + i, 1 + ((i + 1) % n)])
    return points, counts, indices


def rect_mesh(cx, cy, w, h, z, axis_aligned=True):
    """4-vertex axis-aligned rectangle centered at (cx, cy) at height z."""
    hw, hh = w / 2, h / 2
    points = [
        (cx - hw, cy - hh, z),
        (cx + hw, cy - hh, z),
        (cx + hw, cy + hh, z),
        (cx - hw, cy + hh, z),
    ]
    return points, [4], [0, 1, 2, 3]


def emit_mesh(name: str, points, face_counts, face_indices, color, doubleSided=True):
    """Emit USDA def for a Mesh prim with displayColor."""
    p_str = ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in points)
    fc_str = ", ".join(str(c) for c in face_counts)
    fi_str = ", ".join(str(i) for i in face_indices)
    ds = "true" if doubleSided else "false"
    return [
        f'def Mesh "{name}"',
        "{",
        f"    point3f[] points = [{p_str}]",
        f"    int[] faceVertexCounts = [{fc_str}]",
        f"    int[] faceVertexIndices = [{fi_str}]",
        f"    color3f[] primvars:displayColor = [({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})]",
        f"    uniform bool doubleSided = {ds}",
        "}",
        "",
    ]


def emit_goal_net(name: str, x, y, mouth_direction):
    """A simple open-front goal: 3 vertical posts + crossbar + back, no mesh
    geometry -- use the basic prims (Capsule for posts, Cube for back) and
    cast the resulting box-shaped frame."""
    # Mouth at x; goal sits BEHIND the goal line by GOAL_D.
    behind = x + mouth_direction * GOAL_D
    half = GOAL_W / 2
    L = []
    L += [
        f'def Xform "{name}"',
        "{",
        "    def Cube \"Back\"",
        "    {",
        f"        double size = 1",
        f"        matrix4d xformOp:transform = ( ({GOAL_D * 0.06:.4f}, 0, 0, 0),"
        f" (0, {GOAL_W:.4f}, 0, 0), (0, 0, {GOAL_H:.4f}, 0),"
        f" ({behind:.4f}, {y:.4f}, {GOAL_H/2:.4f}, 1) )",
        '        uniform token[] xformOpOrder = ["xformOp:transform"]',
        f'        color3f[] primvars:displayColor = [({NET_COLOR[0]:.3f}, {NET_COLOR[1]:.3f}, {NET_COLOR[2]:.3f})]',
        "    }",
    ]
    # The two posts at the mouth + crossbar
    post_r = 0.04
    for i, py in enumerate((y - half, y + half)):
        L += [
            f"    def Cylinder \"Post{i+1}\"",
            "    {",
            f"        double radius = {post_r}",
            f"        double height = {GOAL_H}",
            "        uniform token axis = \"Z\"",
            f"        double3 xformOp:translate = ({x:.4f}, {py:.4f}, {GOAL_H/2:.4f})",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]',
            f'        color3f[] primvars:displayColor = [({NET_POST[0]:.3f}, {NET_POST[1]:.3f}, {NET_POST[2]:.3f})]',
            "    }",
        ]
    L += [
        "    def Cylinder \"Crossbar\"",
        "    {",
        f"        double radius = {post_r}",
        f"        double height = {GOAL_W}",
        "        uniform token axis = \"Y\"",
        f"        double3 xformOp:translate = ({x:.4f}, {y:.4f}, {GOAL_H:.4f})",
        '        uniform token[] xformOpOrder = ["xformOp:translate"]',
        f'        color3f[] primvars:displayColor = [({NET_POST[0]:.3f}, {NET_POST[1]:.3f}, {NET_POST[2]:.3f})]',
        "    }",
        "}",
        "",
    ]
    return L


def emit_dot(name: str, x, y, color):
    """A small disk via Cylinder of height ~0."""
    return [
        f'def Cylinder "{name}"',
        "{",
        f"    double radius = {FACEOFF_DOT_R}",
        "    double height = 0.005",
        "    uniform token axis = \"Z\"",
        f"    double3 xformOp:translate = ({x:.4f}, {y:.4f}, {PAINT_Z + 0.003:.4f})",
        '    uniform token[] xformOpOrder = ["xformOp:translate"]',
        f"    color3f[] primvars:displayColor = [({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})]",
        "}",
        "",
    ]


def main() -> None:
    out_dir = PROJECT_ROOT / "assets" / "usd"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rink_nhl.usda"

    L = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Rink"',
        '    upAxis = "Z"',
        "    metersPerUnit = 1",
        "    customLayerData = {",
        '        string source = "Hockey_AI build_rink_usd.py"',
        f'        double rink_length_m = {RINK_L:.4f}',
        f'        double rink_width_m  = {RINK_W:.4f}',
        f'        double corner_radius_m = {CORNER_R:.4f}',
        "    }",
        ")",
        "",
        'def Xform "Rink" (kind = "assembly")',
        "{",
    ]

    # Translate the rink so origin (0,0) is the bottom-left corner of the
    # rectangular envelope, matching the positions JSON convention
    # (ice_x in [0, RINK_L], ice_y in [0, RINK_W]). The outline + corners
    # are built centered, so a translate of (RINK_L/2, RINK_W/2, 0) here.
    L.append(f'    double3 xformOp:translate = ({RINK_L/2:.4f}, {RINK_W/2:.4f}, 0)')
    L.append('    uniform token[] xformOpOrder = ["xformOp:translate"]')
    L.append("")

    # --- Ice surface (rounded-rect mesh) ---
    outline = rounded_rect_outline(RINK_L, RINK_W, CORNER_R)
    pts, fc, fi = fan_mesh_from_outline(outline, ICE_Z)
    for line in emit_mesh("Ice", pts, fc, fi, ICE_COLOR):
        L.append("    " + line if line else "")

    # --- Center red line + center red dot ---
    pts, fc, fi = rect_mesh(0, 0, LINE_W, RINK_W, PAINT_Z)
    for line in emit_mesh("CenterLine", pts, fc, fi, RED):
        L.append("    " + line if line else "")

    for line in emit_dot("CenterDot", 0, 0, RED):
        L.append("    " + line if line else "")

    # --- Blue lines (offsets from rink center: -25 ft, +25 ft) ---
    for nm, off in (("BlueLineL", -(CENTER_X - BLUE_LINE_L)),
                     ("BlueLineR", (BLUE_LINE_R - CENTER_X))):
        pts, fc, fi = rect_mesh(off, 0, LINE_W, RINK_W, PAINT_Z)
        for line in emit_mesh(nm, pts, fc, fi, BLUE):
            L.append("    " + line if line else "")

    # --- Goal lines ---
    for nm, off in (("GoalLineL", -(CENTER_X - GOAL_LINE_X_L)),
                     ("GoalLineR", (GOAL_LINE_X_R - CENTER_X))):
        # goal lines span the chord at that x, not the full rink width
        # (the rink narrows inside the corner arcs). Use the rounded-rect
        # half-width at x.
        cx_in_rect = abs(off)  # distance from rink center along x
        dx_from_corner = (RINK_L / 2 - cx_in_rect)
        if dx_from_corner < CORNER_R:
            d = CORNER_R - dx_from_corner
            chord_half = RINK_W / 2 - CORNER_R + math.sqrt(max(0, CORNER_R*CORNER_R - d*d))
        else:
            chord_half = RINK_W / 2
        pts, fc, fi = rect_mesh(off, 0, LINE_W, 2 * chord_half, PAINT_Z)
        for line in emit_mesh(nm, pts, fc, fi, RED):
            L.append("    " + line if line else "")

    # --- Faceoff dots (4 end-zone, 4 neutral-zone). Coords centered. ---
    dot_i = 0
    for x in (-(CENTER_X - (GOAL_LINE_X_L + 20 * FT)),  # left end-zone, 20 ft from goal line
              (GOAL_LINE_X_R - 20 * FT) - CENTER_X):    # right end-zone
        for y in (-22 * FT, 22 * FT):
            dot_i += 1
            for line in emit_dot(f"EndZoneDot_{dot_i}", x, y, RED):
                L.append("    " + line if line else "")
    dot_i = 0
    for x in (-(CENTER_X - (BLUE_LINE_L + 5 * FT)),     # left neutral, 5 ft from blue line
              (BLUE_LINE_R - CENTER_X) + 5 * FT):        # right neutral
        for y in (-22 * FT, 22 * FT):
            dot_i += 1
            for line in emit_dot(f"NeutralDot_{dot_i}", x, y, RED):
                L.append("    " + line if line else "")

    # --- Goal nets behind each goal line ---
    for line in emit_goal_net("NetL", -(CENTER_X - GOAL_LINE_X_L), 0, -1):
        L.append("    " + line if line else "")
    for line in emit_goal_net("NetR", (GOAL_LINE_X_R - CENTER_X), 0, 1):
        L.append("    " + line if line else "")

    L.append("}")
    L.append("")
    out_path.write_text("\n".join(L))
    print(f"wrote {out_path}")
    print(f"  ice {RINK_L:.2f} x {RINK_W:.2f} m, corner radius {CORNER_R:.2f} m")
    print(f"  rink translated to origin at (0,0); ice surface in z=0 plane")


if __name__ == "__main__":
    main()
