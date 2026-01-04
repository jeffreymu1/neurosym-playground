import bpy
import os, sys, json, math
from mathutils import Vector

r"""
CLI:

$blender42 = "C:\Users\jmu5\Downloads\blender-4.2.16-windows-x64\blender-4.2.16-windows-x64\blender.exe"
$proj = "C:\Users\jmu5\OneDrive - Brown University\Documents\temp school stuff\neurosym-playground"

& $blender42 -b -P "$proj\src\render\render_multi.py" -- `
  --in_dir "$proj\data\partnet_datasets\725" `
  --out_dir "$proj\data\output\725" `
  --views 50 --res 512 --engine CYCLES --radius 2.2 --elev 15 --save_previews 1
"""

# arguments
def get_arg(name, default=None, cast=str):
    argv = sys.argv
    if "--" not in argv:
        return default
    argv = argv[argv.index("--") + 1:]
    if name not in argv:
        return default
    i = argv.index(name)
    if i + 1 >= len(argv):
        return default
    return cast(argv[i + 1])

IN_DIR = get_arg("--in_dir", None, str)
OUT_DIR = get_arg("--out_dir", None, str)
VIEWS = get_arg("--views", 48, int)
RES = get_arg("--res", 512, int)
ENGINE = get_arg("--engine", "CYCLES", str)
RADIUS = get_arg("--radius", 2.2, float)
ELEV_DEG = get_arg("--elev", 15.0, float)
SAVE_PREVIEWS = get_arg("--save_previews", 0, int)

if IN_DIR is None or OUT_DIR is None:
    raise SystemExit("Missing --in_dir or --out_dir")

# outputs
rgb_dir = os.path.join(OUT_DIR, "rgb")
depth_exr_dir = os.path.join(OUT_DIR, "depth_exr")
pid_exr_dir = os.path.join(OUT_DIR, "part_id_exr")
prev_dir = os.path.join(OUT_DIR, "previews")

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_exr_dir, exist_ok=True)
os.makedirs(pid_exr_dir, exist_ok=True)
os.makedirs(prev_dir, exist_ok=True)

# scene set
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = ENGINE
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.use_compositing = True

# cycle engine
if ENGINE == "CYCLES":
    scene.cycles.samples = 64
    scene.cycles.use_adaptive_sampling = True

# enable passes
view_layer = scene.view_layers["ViewLayer"]
view_layer.use_pass_z = True
view_layer.use_pass_object_index = True

# imp obs
obj_files = sorted([f for f in os.listdir(IN_DIR) if f.lower().endswith(".obj")])
if not obj_files: raise SystemExit(f"no obj files found in {IN_DIR}")

# collect all part objects together
part_objs = []

for idx, fn in enumerate(obj_files):

    path = os.path.join(IN_DIR, fn)
    bpy.ops.wm.obj_import(filepath=path)

    imported = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    for o in imported:

        o.name = f"part_{idx:03d}"
        o.pass_index = idx + 1
        part_objs.append(o)

# normalize scale and object

def compute_bbox_world(objs):
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        for v in o.bound_box:
            vw = o.matrix_world @ Vector(v)
            mins.x = min(mins.x, vw.x); mins.y = min(mins.y, vw.y); mins.z = min(mins.z, vw.z)
            maxs.x = max(maxs.x, vw.x); maxs.y = max(maxs.y, vw.y); maxs.z = max(maxs.z, vw.z)

    return mins, maxs

bpy.context.view_layer.update()
mins, maxs = compute_bbox_world(part_objs)
center = (mins + maxs) * 0.5
size = max((maxs - mins).x, (maxs - mins).y, (maxs - mins).z)
scale = 1.0 / size if size > 1e-9 else 1.0

for o in part_objs:
    o.location = (o.location - center)
    o.scale = (o.scale * scale)

bpy.context.view_layer.update()

# lights
def add_area_light(name, loc, energy):
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = loc
    light_data.energy = energy

add_area_light("KeyLight",  (2.0, -2.0, 3.0), 2000)
add_area_light("FillLight", (-2.5, 2.0, 2.0), 1200)

# camera
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

cam_data.lens = 50.0
cam_data.sensor_width = 36.0
cam_data.clip_start = 0.01
cam_data.clip_end = 100.0

def look_at(camera_obj, target):
    direction = (target - camera_obj.location).normalized()
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera_obj.rotation_euler = rot_quat.to_euler()

# compositor writing
scene.use_nodes = True
nt = scene.node_tree
nt.nodes.clear()

rl = nt.nodes.new("CompositorNodeRLayers")
rl.location = (0, 0)

# depth normal
mapr = nt.nodes.new("CompositorNodeMapRange")
mapr.location = (260, -240)
mapr.use_clamp = True
mapr.inputs["From Min"].default_value = cam_data.clip_start
mapr.inputs["From Max"].default_value = max(0.1, RADIUS * 3.0)
mapr.inputs["To Min"].default_value = 0.0
mapr.inputs["To Max"].default_value = 1.0

comp = nt.nodes.new("CompositorNodeComposite")
comp.location = (520, 0)

def get_rl_output(names):
    for n in names:
        if n in rl.outputs:
            return rl.outputs[n]
        
    raise RuntimeError("goon")

sock_image = get_rl_output(["Image"])
sock_depth = get_rl_output(["Depth", "Z"])
sock_index = get_rl_output(["IndexOB"])

# wdp
nt.links.new(sock_depth, mapr.inputs["Value"])

def connect_to_composite(src_socket):
    for link in list(nt.links):
        if link.to_node == comp and link.to_socket == comp.inputs[0]:
            nt.links.remove(link)
    nt.links.new(src_socket, comp.inputs[0])

# rendering settings
def set_png_rgba(scene):
    s = scene.render.image_settings
    s.file_format = "PNG"
    s.color_mode = "RGBA"
    s.color_depth = "8"

def set_exr32(scene):
    s = scene.render.image_settings
    s.file_format = "OPEN_EXR"
    s.color_mode = "RGB"
    s.color_depth = "32"
    s.exr_codec = "ZIP"

def set_png_bw16(scene):
    s = scene.render.image_settings
    s.file_format = "PNG"
    s.color_mode = "BW"
    s.color_depth = "16"

# render
target = Vector((0, 0, 0))
elev = math.radians(ELEV_DEG)
cameras = []

for i in range(VIEWS):
    print(f"rendering view {i+1}/{VIEWS}")

    az = 2.0 * math.pi * (i / VIEWS)
    x = RADIUS * math.cos(az) * math.cos(elev)
    y = RADIUS * math.sin(az) * math.cos(elev)
    z = RADIUS * math.sin(elev)

    cam.location = (x, y, z)
    look_at(cam, target)

    fx = (cam_data.lens / cam_data.sensor_width) * RES
    fy = fx
    cx = RES / 2.0
    cy = RES / 2.0
    c2w = cam.matrix_world.copy()
    w2c = cam.matrix_world.inverted()

    cameras.append({
        "view": i,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": RES, "h": RES},
        "c2w": [list(row) for row in c2w],
        "w2c": [list(row) for row in w2c]})

    # rgb
    connect_to_composite(sock_image)
    set_png_rgba(scene)
    scene.render.filepath = os.path.join(rgb_dir, f"{i:03d}.png")
    bpy.ops.render.render(write_still=True)

    # depth
    connect_to_composite(sock_depth)
    set_exr32(scene)
    scene.render.filepath = os.path.join(depth_exr_dir, f"{i:03d}.exr")
    bpy.ops.render.render(write_still=True)

    # part and object id
    connect_to_composite(sock_index)
    set_exr32(scene)
    scene.render.filepath = os.path.join(pid_exr_dir, f"{i:03d}.exr")
    bpy.ops.render.render(write_still=True)

    # depth preview
    if SAVE_PREVIEWS:
        connect_to_composite(mapr.outputs["Value"])
        set_png_bw16(scene)
        scene.render.filepath = os.path.join(prev_dir, f"depth_preview_{i:03d}.png")
        bpy.ops.render.render(write_still=True)

with open(os.path.join(OUT_DIR, "cameras.json"), "w") as f: json.dump(cameras, f, indent=2)

print("render complete at", OUT_DIR)