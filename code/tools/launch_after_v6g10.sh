#!/bin/bash
cd /root/gpufree-data/hnu_v2/code
V=v6g10 setsid nohup bash run_after_v6r.sh > ../runs/night/after_v6g10.out 2>&1 < /dev/null &
cd ../runs
/root/gpufree-data/venvs/mjlab/bin/python - <<'EOF'
import json, glob
for f in sorted(glob.glob('results_v6g10_bank_teacher_dr*/*.json')):
    j = json.load(open(f))['summary']
    for T in ('CrouchedTraverse', 'ThreeStepPoleRun', 'FiveStepPoleRun'):
        s = j[f'{T}/all']
        print(f.split('/')[0][-3:], 'push' if 'push' in f else 'clean', T, 'succ %.1f' % (100 * s['success']),
              'body %.1f cm' % (100 * s['body_err']))
EOF