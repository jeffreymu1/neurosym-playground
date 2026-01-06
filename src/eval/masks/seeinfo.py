import numpy as np, glob, os

VIEW_DIR = r"C:\Users\jmu5\OneDrive - Brown University\Documents\temp school stuff\neurosym-playground\data\output\725lifted\view_000"
files = sorted(glob.glob(os.path.join(VIEW_DIR, "*.npz")))

for f in files[:5]:
    z = np.load(f)
    print("\n", os.path.basename(f))
    print("keys:", z.files)
    for k in z.files:
        a = z[k]
        print(f"  {k:15s} shape={a.shape} dtype={a.dtype}")
