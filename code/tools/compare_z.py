import glob, os, sys
import numpy as np
root = sys.argv[1]
names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(root, "Z0", "*_z.npy")))[:10]
for a, b in (("Z3", "Z4"), ("Z2", "Z3"), ("Z0", "Z1")):
    d = [np.abs(np.load(os.path.join(root, a, n)) - np.load(os.path.join(root, b, n))).max() for n in names]
    print(a, b, "max abs diff", float(max(d)))
