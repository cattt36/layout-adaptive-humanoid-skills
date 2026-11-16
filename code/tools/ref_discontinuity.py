"""Look for reference artefacts in the stake references: per frame, the largest body speed (finite differences of
body_pos_w) and the pelvis height, compared with the original flat reference.  Prints the frames with the largest
speeds (as clip fraction) and the median over clips of where the global maximum occurs.

  python ref_discontinuity.py <stakes_dir/<task>> <ref_v1/<task>>
"""
import glob
import os
import sys

import numpy as np

sd, fd = sys.argv[1:3]
pk_s, pk_f, ex = [], [], []
for f in sorted(glob.glob(os.path.join(sd, "*_motion.npz")))[:12]:
    name = os.path.basename(f)
    a = np.load(f)["body_pos_w"]
    b = np.load(os.path.join(fd, name))["body_pos_w"]
    va = np.linalg.norm(np.diff(a, axis=0), axis=-1).max(1) * 50
    vb = np.linalg.norm(np.diff(b, axis=0), axis=-1).max(1) * 50
    pk_s.append(np.argmax(va) / len(va)); pk_f.append(np.argmax(vb) / len(vb))
    ex.append((name[:-11], va.max(), vb.max(), np.argmax(va) / len(va), a[0, 0, 2], b[0, 0, 2],
               np.abs(a[:, 0, 2] - b[: len(a), 0, 2]).max() if len(b) >= len(a) else np.nan))
for e in ex[:6]:
    print("%s  max body speed stakes %.1f m/s vs flat %.1f m/s at frac %.2f | pelvis z0 %.3f vs %.3f | max |dz| %.3f" % e)
print("median frac of max speed: stakes %.2f flat %.2f" % (np.median(pk_s), np.median(pk_f)))
