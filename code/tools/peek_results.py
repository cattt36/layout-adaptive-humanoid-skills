"""Print overall clean/push success per student json in a results directory, with file times."""
import glob
import json
import os
import sys
import time

for d in sys.argv[1:]:
    print("==", d)
    for f in sorted(glob.glob(os.path.join(d, "eval_*.json"))):
        j = json.load(open(f))
        s = j.get("summary", j)
        keys = [k for k in s if k.endswith("/all")] if isinstance(s, dict) else []
        vals = {k.split("/")[0]: round(s[k]["success"], 3) for k in keys if isinstance(s[k], dict) and "success" in s[k]}
        print(time.strftime("%H:%M", time.localtime(os.path.getmtime(f))), os.path.basename(f), vals or list(s)[:6])
