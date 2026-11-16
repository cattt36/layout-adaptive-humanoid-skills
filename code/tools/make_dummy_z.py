"""Zero z_skill tables (smoke tests only)."""
import glob, os, sys
import numpy as np
ref_root, out, dim = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16
os.makedirs(out, exist_ok=True)
for f in glob.glob(os.path.join(ref_root, "*", "*_motion.npz")):
    T = np.load(f)["joint_pos"].shape[0]
    np.save(os.path.join(out, os.path.basename(f).replace("_motion.npz", "_z.npy")), np.zeros((T, dim), np.float32))
print(len(os.listdir(out)))
