"""Save frames of a video at given times (s) as PNGs: python grab_frames.py video.mp4 out_prefix t1 t2 ..."""
import sys

import imageio.v2 as imageio

vid, pre, ts = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3:]]
r = imageio.get_reader(vid)
fps = r.get_meta_data()["fps"]
want = {int(round(t * fps)): t for t in ts}
for i, fr in enumerate(r):
    if i in want:
        imageio.imwrite(f"{pre}_{want[i]:.1f}s.png", fr)
    if i > max(want):
        break
print("ok", list(want.values()))
