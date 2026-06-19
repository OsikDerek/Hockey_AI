"""Headless Blender render to eyeball the articulated USD avatars.

Run:  blender --background --python scripts/_avatar_check_blender.py
Writes output/_avatar_check.png and output/_avatar_check_wide.png .
"""
import bpy, math, mathutils
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USD = str(ROOT / "output" / "caufield_trim_b3.usda")

# Clean slate.
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)

bpy.ops.wm.usd_import(filepath=USD)
bpy.context.scene.frame_set(0)

# Drop the imported (RTX-tuned) lights so we control exposure here.
for o in list(bpy.data.objects):
    if o.type == 'LIGHT':
        bpy.data.objects.remove(o, do_unlink=True)

# Modest ambient so figures read with contrast (not blown out).
world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[0].default_value = (0.32, 0.38, 0.48, 1.0)
    bg.inputs[1].default_value = 0.35

sun = bpy.data.lights.new("CheckSun", type='SUN')
sun.energy = 2.2
sun_obj = bpy.data.objects.new("CheckSun", sun)
bpy.context.collection.objects.link(sun_obj)
sun_obj.rotation_euler = (math.radians(55), 0, math.radians(40))

scene = bpy.context.scene
try:
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
except Exception:
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 24
scene.render.resolution_x = 1280
scene.render.resolution_y = 800
scene.render.film_transparent = False

cam_data = bpy.data.cameras.new("CheckCam")
cam_data.lens = 50
cam_obj = bpy.data.objects.new("CheckCam", cam_data)
bpy.context.collection.objects.link(cam_obj)
scene.camera = cam_obj


def shoot(loc, target, path):
    cam_obj.location = mathutils.Vector(loc)
    d = mathutils.Vector(target) - cam_obj.location
    cam_obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print("wrote", path)


# Tight 3/4 on a single skater (p4 sits ~(7.6, 20.8) at frame 0).
shoot((10.2, 18.0, 1.9), (7.6, 20.8, 1.0), ROOT / "output" / "_avatar_check.png")
# Wider establishing shot of the whole end-zone action.
shoot((20.0, 2.0, 9.0), (10.0, 18.0, 0.6), ROOT / "output" / "_avatar_check_wide.png")
print("DONE")
