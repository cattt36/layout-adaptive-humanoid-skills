"""ONNX vs torch parity of an exported blind tracker (deploy venv): parity_ref.npz from export_tracker.py.
  python tools/tracker_parity.py <export_dir>   -> <export_dir>/parity.txt"""
import os
import sys

import numpy as np
import onnxruntime as ort

ex = sys.argv[1]
r = np.load(os.path.join(ex, "parity_ref.npz"))
sess = ort.InferenceSession(os.path.join(ex, "policy.onnx"), providers=["CPUExecutionProvider"])
a = np.concatenate([sess.run(None, {"obs": r["obs"][i:i + 1]})[0] for i in range(len(r["obs"]))])
err = float(np.abs(a - r["act"]).max())
txt = f"max |onnx - torch| action {err:.2e} ({len(a)} obs)\n{'PARITY OK' if err < 1e-4 else 'PARITY FAILED'}\n"
open(os.path.join(ex, "parity.txt"), "w").write(txt)
print(txt, end="")
