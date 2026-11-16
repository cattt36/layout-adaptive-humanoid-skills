"""Scale of the z_skill tables (per-dim std over all frames, frame-to-frame change) for setting the augmentation."""
import glob
import sys

import numpy as np

for P in sys.argv[1:]:
    Z = np.concatenate([np.load(f) for f in sorted(glob.glob(f"/root/gpufree-data/hnu_v2/data/z_tables_v5/{P}/*_z.npy"))])
    d = np.abs(np.diff(Z, axis=0))
    print(f"{P}: frames {len(Z)}, dim {Z.shape[1]}, |z| mean {np.abs(Z).mean():.3f}, per-dim std mean {Z.std(0).mean():.3f} "
          f"(min {Z.std(0).min():.3f}, max {Z.std(0).max():.3f}), mean |dz| per frame {d.mean():.4f}, L2 norm {np.linalg.norm(Z, axis=1).mean():.2f}")
