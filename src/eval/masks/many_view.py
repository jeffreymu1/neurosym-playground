import os, glob
import numpy as np
import open3d as o3d

LIFT_DIR = r"C:\Users\jmu5\OneDrive - Brown University\Documents\temp school stuff\neurosym-playground\data\output\725lifted"
files = sorted(glob.glob(os.path.join(LIFT_DIR, "view_*", "*.npz")))
print("found npzs:", len(files))

rng = np.random.default_rng(0)
geoms = []
for f in files[:100]:
    z = np.load(f)
    pts = z["points"].astype(np.float32)
    if pts.shape[0] < 800:
        continue
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    c = rng.random(3)
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(c[None, :], pts.shape[0], axis=0))
    geoms.append(pcd)

o3d.visualization.draw_geometries(geoms)
