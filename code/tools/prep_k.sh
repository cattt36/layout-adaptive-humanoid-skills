#!/bin/bash
# Preparation of v6g10k (tempo augmentation): stop v6g10j, time-scaled motions 0.95 / 1.05 (0.9 / 1.1 exist),
# encoder z for the time-scaled ThreeStepPoleRun / FiveStepPoleRun trials (+ a 1.0x reproduction check), augmented training set.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
for p in $(pgrep -f "run_v6g10j.sh"); do kill $p; done
for p in $(pgrep -f "tools/monitor_tb.py v6g10j"); do kill $p; done
sleep 2
for p in $(pgrep -f "students_v6g10j|results_v6g10j|profile v6g10j"); do kill $p; done
sleep 10
echo "left over v6g10j processes:"; pgrep -af "v6g10j" | grep -v prep_k || echo none
cd $H/code
$PY tools/make_timescale.py 0.95 1.05 || exit 1
for P in Z0 Z1 Z2 Z3 Z4; do
  ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=0 $PY stage4_latent/export_z_tables_ts.py --profile $P \
    --latent-root $H/runs/latent_v5t/seed42 --motion-root $H/data/ref_v1 --speeds 1.0,0.9,0.95,1.05,1.1 \
    --out-root $H/data/ts > $N/z_enc_$P.log 2>&1 || { tail -20 $N/z_enc_$P.log; exit 1; }
  tail -5 $N/z_enc_$P.log
done
$PY - << 'EOF'
import glob, os
import numpy as np
H = "/root/gpufree-data/hnu_v2/data"
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    d = []
    for f in sorted(glob.glob(f"{H}/ts/ts100/z_enc/{P}/*_z.npy")):
        a, b = np.load(f), np.load(f"{H}/z_tables_v5t/{P}/{os.path.basename(f)}")
        d.append(np.abs(a - b).max() if a.shape == b.shape else np.inf)
    print(f"{P}: 1.0x re-export vs z_tables_v5t, {len(d)} trials, max abs diff {max(d):.2e}")
    # how different is the encoder z from simply interpolating the 1.0x z?
    r = []
    for f in sorted(glob.glob(f"{H}/ts/ts090/z_enc/{P}/*_z.npy")):
        a, b = np.load(f), np.load(f"{H}/ts/ts090/z_tables_v5t/{P}/{os.path.basename(f)}")
        r.append(np.linalg.norm(a - b, axis=-1).mean() / np.linalg.norm(b, axis=-1).mean())
    print(f"{P}: 0.9x encoder z vs interpolated z, mean relative difference {np.mean(r):.3f}")
EOF
$PY tools/make_aug_ts.py
