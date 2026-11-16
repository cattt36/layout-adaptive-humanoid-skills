import json

import numpy as np

D = "/root/gpufree-data/hnu_v2/data"
print(open(f"{D}/ref_stakes_s93_h10/ThreeStepPoleRun/ThreeStepPoleRun_002_stakes.json").read()[:1500])
print("----")
print(open(f"{D}/ref_bridge_video_lift/CrouchedTraverse/CrouchedTraverse_002_bridge.json").read()[:3000])
print("----")
d = np.load(f"{D}/ref_stakes_s93_h10/ThreeStepPoleRun/ThreeStepPoleRun_002_motion.npz")
print({k: d[k].shape for k in d.files})
for k in d.files:
    if d[k].ndim <= 1 and d[k].size < 50:
        print(k, d[k])
print("body_pos_w[0,0]", d["body_pos_w"][0, 0] if "body_pos_w" in d.files else None)
print("body_quat_w[0,0]", d["body_quat_w"][0, 0] if "body_quat_w" in d.files else None)
