"""Convert a downloaded rigged character (FBX / glTF / OBJ / USD) into a
normalized USD player asset for the Hockey_AI scene.

Run through Blender (headless). It imports the mesh, scales it to skater
height, drops the feet onto z=0, centers it on the origin, rotates it to
face +X (local "forward" used by export_to_usd.py), exports a clean USD,
and renders a preview PNG so we can confirm orientation/scale.

    "<blender>" --background --python scripts/convert_character_to_usd.py -- \
        --input assets/usd/source/<file>.fbx \
        --output assets/usd/player_skater.usda \
        --height 1.83 --face-deg 0 --preview

If the preview shows the player facing the wrong way, re-run with a
different --face-deg (0/90/180/270). Everything after `--` is passed to
this script (Blender swallows args before it).
"""
import bpy, sys, math, argparse
from pathlib import Path
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", default=str(ROOT / "assets" / "usd" / "player_skater.usda"))
ap.add_argument("--height", type=float, default=1.83)
ap.add_argument("--face-deg", type=float, default=0.0,
                help="rotate about Z so the character faces +X")
ap.add_argument("--root-name", default="Skater")
ap.add_argument("--preview", action="store_true")
args = ap.parse_args(argv)

inp = Path(args.input)
if not inp.is_absolute():
    inp = ROOT / inp

# --- clean slate ---
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# --- import by extension ---
ext = inp.suffix.lower()
if ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=str(inp))
elif ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=str(inp))
elif ext == ".obj":
    bpy.ops.wm.obj_import(filepath=str(inp))
elif ext in (".usd", ".usda", ".usdc", ".usdz"):
    bpy.ops.wm.usd_import(filepath=str(inp))
else:
    raise SystemExit(f"unsupported input extension: {ext}")

roots = [o for o in bpy.data.objects if o.parent is None]
print("imported top-level objects:", [o.name for o in roots])

# --- world-space bounding box over all mesh objects ---
mins = Vector(( 1e9,  1e9,  1e9))
maxs = Vector((-1e9, -1e9, -1e9))
have_mesh = False
for o in bpy.data.objects:
    if o.type != 'MESH':
        continue
    have_mesh = True
    for corner in o.bound_box:
        w = o.matrix_world @ Vector(corner)
        for i in range(3):
            mins[i] = min(mins[i], w[i])
            maxs[i] = max(maxs[i], w[i])
if not have_mesh:
    raise SystemExit("no mesh objects found in import")

size = maxs - mins
height = max(size.z, 1e-6)
scale = args.height / height
print(f"bbox size {tuple(round(v,3) for v in size)}  scale->height {scale:.4f}")

# --- parent everything under one normalizer empty, then transform it ---
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
root = bpy.context.active_object
root.name = args.root_name
for o in roots:
    o.parent = root
    o.matrix_parent_inverse = root.matrix_world.inverted()

# scale, rotate to face +X, then drop feet to z=0 + center x/y
root.scale = (scale, scale, scale)
root.rotation_euler = (0, 0, math.radians(args.face_deg))
bpy.context.view_layer.update()

# recompute world bbox after scale/rotate to place on ice + centered
mins = Vector(( 1e9,  1e9,  1e9)); maxs = Vector((-1e9, -1e9, -1e9))
for o in bpy.data.objects:
    if o.type != 'MESH':
        continue
    for corner in o.bound_box:
        w = o.matrix_world @ Vector(corner)
        for i in range(3):
            mins[i] = min(mins[i], w[i]); maxs[i] = max(maxs[i], w[i])
cx = (mins.x + maxs.x) / 2
cy = (mins.y + maxs.y) / 2
root.location = (-cx, -cy, -mins.z)
bpy.context.view_layer.update()

# --- export USD ---
outp = Path(args.output)
if not outp.is_absolute():
    outp = ROOT / outp
outp.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.usd_export(filepath=str(outp), export_materials=True,
                      export_animation=False, root_prim_path=f"/{args.root_name}")
print("wrote", outp)

# --- optional preview render (3/4 view, faces +X to the right) ---
if args.preview:
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.30, 0.36, 0.46, 1.0)
        bg.inputs[1].default_value = 0.4
    sun = bpy.data.lights.new("S", 'SUN'); sun.energy = 2.5
    so = bpy.data.objects.new("S", sun); bpy.context.collection.objects.link(so)
    so.rotation_euler = (math.radians(55), 0, math.radians(40))
    cam_d = bpy.data.cameras.new("C"); cam_d.lens = 50
    cam = bpy.data.objects.new("C", cam_d); bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = (2.6, -2.6, 1.4)
    d = Vector((0, 0, 0.9)) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    try:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except Exception:
        bpy.context.scene.render.engine = 'CYCLES'; bpy.context.scene.cycles.samples = 24
    bpy.context.scene.render.resolution_x = 700
    bpy.context.scene.render.resolution_y = 900
    prev = ROOT / "output" / "_character_preview.png"
    bpy.context.scene.render.filepath = str(prev)
    bpy.ops.render.render(write_still=True)
    print("preview", prev)
print("DONE")
