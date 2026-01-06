import os, glob, json
import numpy as np
import open3d as o3d

LIFT_DIR = r"C:\Users\jmu5\OneDrive - Brown University\Documents\temp school stuff\neurosym-playground\data\output\725lifted"
OUT_DIR = r"C:\Users\jmu5\OneDrive - Brown University\Documents\temp school stuff\neurosym-playground\data\output\725"
CAM_PATH = os.path.join(OUT_DIR, "cameras.json")

view_dirs = sorted(glob.glob(os.path.join(LIFT_DIR, "view_*")))

with open(CAM_PATH, "r") as f:
    cams = json.load(f)

cam_by_view = {}
for ci in cams:
    v = int(ci["view"])
    cam_by_view[v] = {
        "c2w": np.array(ci["c2w"], dtype=np.float32),
        "w2c": np.array(ci["w2c"], dtype=np.float32)}

rng = np.random.default_rng(0)
geoms = []

def cam_frame(c2w, size=0.1):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(c2w)
    return frame

for vd in view_dirs[:20]:
    view = int(os.path.basename(vd).split("_")[1])

    files = sorted(glob.glob(os.path.join(vd, "*.npz")))
    best = None
    for fpath in files:
        z = np.load(fpath)
        n = int(z["points"].shape[0])
        if n >= 2000 and (best is None or n > best[0]):
            best = (n, fpath)
    if best is None:
        continue

    z = np.load(best[1])
    pts = z["points"].astype(np.float32)

    if pts.shape[0] > 20000:
        idx = rng.choice(pts.shape[0], 20000, replace=False)
        pts = pts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    c = rng.random(3)
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(c[None, :], pts.shape[0], axis=0))
    geoms.append(pcd)

    # add camera frame for this view
    if view in cam_by_view:
        w2c = cam_by_view[view]["w2c"]
        c2w = np.linalg.inv(w2c).astype(np.float32)  # robust
        geoms.append(cam_frame(c2w, size=0.15))

# global axes at origin
geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
print("geoms:", len(geoms))

o3d.visualization.draw_geometries(geoms)
