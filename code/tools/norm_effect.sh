#!/bin/bash
# How much do the fold-1 z tables change when the encoder's own (checkpoint) normalizers are used instead of the
# re-fitted 'subject'-split ones?  Export to a scratch dir and compare with data/z_tables_v5t (no file is replaced).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for P in Z0 Z4; do
  ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES= $PY stage4_latent/export_z_tables.py --profile $P \
    --latent-root $H/runs/latent_v5t/seed42 --motion-root $H/data/ref_v1 --out $H/runs/norm_check/$P 2>&1 | grep -E "normalizers|clips"
done
$PY - << 'EOF'
import glob, os
import numpy as np
for P in ("Z0", "Z4"):
    r = []
    for f in glob.glob(f"/root/gpufree-data/hnu_v2/runs/norm_check/{P}/*_z.npy"):
        a, b = np.load(f), np.load(f"/root/gpufree-data/hnu_v2/data/z_tables_v5t/{P}/{os.path.basename(f)}")
        r.append(np.linalg.norm(a - b, axis=-1).mean() / np.linalg.norm(b, axis=-1).mean())
    print(f"{P}: checkpoint vs re-fitted normalizers, mean relative z difference {np.mean(r):.3f} (max over trials {np.max(r):.3f})")
EOF
