import bpy
import os, sys, json, math
from mathutils import Vector

# get args
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

if IN_DIR is None or OUT_DIR is None:
    raise SystemExit("bruh")

# dir
rgb_dir = os.path.join(OUT_DIR, "rgb")
depth_dir = os.path.join(OUT_DIR, "depth")
pid_dir = os.path.join(OUT_DIR, "part_id")
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(pid_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = ENGINE
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.render.resolution_percentage = 100
scene.render.film_transparent = True

if ENGINE == "CYCLES":
    scene.cycles.samples = 64
    scene.cycles.use_adaptive_sampling = True

# enable passes on view layer
view_layer = scene.view_layers["ViewLayer"]
view_layer.use_pass_z = True
view_layer.use_pass_object_index = True

# imp objects
obj_files = sorted([f for f in os.listdir(IN_DIR) if f.lower().endswith(".obj")])
if not obj_files:
    raise SystemExit(f"no .obj files found in {IN_DIR}")

part_objs = []
for idx, fn in enumerate(obj_files):
    path = os.path.join(IN_DIR, fn)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=path)
    else:
        bpy.ops.import_scene.obj(filepath=path)
    imported = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    for o in imported:
        o.name = f"part_{idx:03d}"
        o.pass_index = idx + 1
        part_objs.append(o)

# settings
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

add_area_light("KeyLight", (2.0, -2.0, 3.0), 2000)
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

# RENDERING
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
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.filepath = os.path.join(rgb_dir, f"{i:03d}")
    bpy.ops.render.render(write_still=True)
    
    # extract and save passes
    render_result = bpy.data.images.get('Render Result')
    if render_result:
        render_result.save_render(
            filepath=os.path.join(depth_dir, f"{i:03d}.exr"),
            scene=scene)
        
with open(os.path.join(OUT_DIR, "cameras.json"), "w") as f:
    json.dump(cameras, f, indent=2)

print("render complete @", OUT_DIR)