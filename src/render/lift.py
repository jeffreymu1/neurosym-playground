import bpy
import os, sys, json
import numpy as np

r'''
$blender42 = "C:\Users\jmu5\Downloads\blender-4.2.16-windows-x64\blender-4.2.16-windows-x64\blender.exe"
$out = "data\output\"

& $blender42 -b -P "src\render\sam_seg.py" -- `
  --out_dir "$out\725" `
  --sam2d_dir "$out\725_sam2d" `
  --voxel 0.002

'''

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

OUT_DIR   = get_arg("--out_dir", None, str)      # your render OUT_DIR (contains rgb/, depth_exr/, cameras.json)
SAM2D_DIR = get_arg("--sam2d_dir", None, str)    # output from sam_masks.py (sam2d/)
LIFT_DIR  = get_arg("--lift_dir", None, str)     # where to write lifted/ (default OUT_DIR/lifted)
MIN_D     = get_arg("--min_depth", 1e-6, float)
MAX_D     = get_arg("--max_depth", 1e9, float)
VOXEL     = get_arg("--voxel", 0.0, float)       # optional per-mask voxel downsample

if OUT_DIR is None or SAM2D_DIR is None:
    raise SystemExit("Missing --out_dir or --sam2d_dir")

rgb_dir   = os.path.join(OUT_DIR, "rgb")
depth_dir = os.path.join(OUT_DIR, "depth_exr")
cam_path  = os.path.join(OUT_DIR, "cameras.json")
if LIFT_DIR is None:
    LIFT_DIR = os.path.join(OUT_DIR, "lifted")

os.makedirs(LIFT_DIR, exist_ok=True)

def load_image_rgba(path):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.asarray(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    bpy.data.images.remove(img)
    return px

def load_exr_r(path):
    return load_image_rgba(path)[..., 0].astype(np.float32)

def load_png_rgb_u8(path):
    rgba = load_image_rgba(path)
    rgb = np.clip(rgba[..., :3], 0.0, 1.0)
    return (rgb * 255.0 + 0.5).astype(np.uint8)

def load_mask(path):
    rgba = load_image_rgba(path)
    return rgba[..., 0] > 0.5

def lift_points(depth, fx, fy, cx, cy, valid_mask):
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    v = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)

    d = depth[valid_mask]
    uu = u[valid_mask]
    vv = v[valid_mask]

    x = (uu - cx) * d / fx
    y = (vv - cy) * d / fy
    z = d
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts

def apply_c2w(pts_cam, c2w):
    N = pts_cam.shape[0]
    homo = np.concatenate([pts_cam, np.ones((N,1), np.float32)], axis=1)
    pts_w = (c2w @ homo.T).T[:, :3]
    return pts_w.astype(np.float32)

def voxel_downsample(points, colors, voxel):
    if voxel <= 0: return points, colors
    key = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    return points[idx], colors[idx]

with open(cam_path, "r") as f:
    cams = sorted(json.load(f), key=lambda d: d.get("view", 0))

for caminfo in cams:
    i = int(caminfo["view"])
    stem = f"{i:03d}"
    intr = caminfo["intrinsics"]
    fx, fy, cx, cy = map(float, [intr["fx"], intr["fy"], intr["cx"], intr["cy"]])
    c2w = np.array(caminfo["c2w"], dtype=np.float32)

    rgb_path = os.path.join(rgb_dir, f"{stem}.png")
    d_path   = os.path.join(depth_dir, f"{stem}.exr")
    view_mask_dir = os.path.join(SAM2D_DIR, f"view_{stem}", "masks")
    if not (os.path.exists(rgb_path) and os.path.exists(d_path) and os.path.isdir(view_mask_dir)):
        print("[skip]", stem)
        continue

    rgb = load_png_rgb_u8(rgb_path)
    depth = load_exr_r(d_path)
    depth_valid = (depth > MIN_D) & (depth < MAX_D)

    out_view = os.path.join(LIFT_DIR, f"view_{stem}")
    os.makedirs(out_view, exist_ok=True)

    mask_files = sorted([f for f in os.listdir(view_mask_dir) if f.lower().endswith(".png")])
    for mf in mask_files:
        mp = os.path.join(view_mask_dir, mf)
        m = load_mask(mp)
        valid = depth_valid & m
        if valid.sum() < 200:
            continue

        pts_cam = lift_points(depth, fx, fy, cx, cy, valid)
        pts_w = apply_c2w(pts_cam, c2w)
        cols = rgb[valid].reshape(-1, 3)

        pts_w, cols = voxel_downsample(pts_w, cols, VOXEL)

        out_npz = os.path.join(out_view, mf.replace(".png", ".npz"))
        np.savez_compressed(out_npz, points=pts_w, colors=cols)
    print("[ok] lifted masks for view", stem)

print("[done] lifted masks to", LIFT_DIR)
