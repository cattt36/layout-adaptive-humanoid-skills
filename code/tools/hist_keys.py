"""Print selected loss terms from a latent run's history.json (last epoch, train and val if present)."""
import json
import sys

h = json.load(open(sys.argv[1]))
keys = sys.argv[2:] or ["loss", "future", "human_ctrl", "recon"]
rows = h if isinstance(h, list) else h.get("history", h)
last = rows[-1] if isinstance(rows, list) else rows
print(json.dumps({k: v for k, v in (last.items() if isinstance(last, dict) else []) if any(x in k for x in keys)}, indent=0)[:1500])
