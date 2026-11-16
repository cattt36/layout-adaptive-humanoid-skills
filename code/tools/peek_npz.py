import glob
import sys

import numpy as np

for f in sys.argv[1:]:
    for g in sorted(glob.glob(f))[:1]:
        d = np.load(g)
        print(g)
        for k in d.files:
            print(f"  {k:20s} {d[k].shape} {d[k].dtype}")
