import os, glob
import numpy as np
import open3d as o3d

LIFT_DIR = r"C:\Users\jmu5\OneDrive - Brown University\Documents\temp school stuff\neurosym-playground\data\output\725lifted"
files = sorted(glob.glob(os.path.join(LIFT_DIR, "view_*", "*.npz")))

print("files:", len(files))
for k, f in enumerate(files):
    z = np.load(f)
    pts = z["points"].astype(np.float32)
    cols = z["colors"]
    if cols.dtype != np.float32:
        cols = cols.astype(np.float32) / 255.0
        
    if pts.shape[0] < 500:
        continue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    print(f"[{k}/{len(files)}] {pts.shape[0]} pts  {f}")
    o3d.visualization.draw_geometries([pcd])
