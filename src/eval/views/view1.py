import os, glob, re, zlib
import numpy as np
import open3d as o3d

LIFT_DIR = r"C:\Users\jmu5\OneDrive - Brown University\Documents\temp school stuff\neurosym-playground\data\output\725lifted\view_000"
files = sorted(glob.glob(os.path.join(LIFT_DIR, "mask_*.npz")))

def stable_color_from_name(name: str) -> np.ndarray:
    h = zlib.crc32(name.encode("utf-8")) & 0xffffffff
    rng = np.random.default_rng(h)
    return rng.random(3).astype(np.float64)

geoms = []
for f in files:
    z = np.load(f)
    pts = z["points"].astype(np.float32)
    # get rid of sparse points
    if pts.shape[0] < 800:
        continue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    c = stable_color_from_name(os.path.basename(f))
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(c[None, :], pts.shape[0], axis=0))

    geoms.append(pcd)

geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
o3d.visualization.draw_geometries(geoms)



