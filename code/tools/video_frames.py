"""Extract evenly spaced frames (and a contact sheet) from a video.  python video_frames.py <video> <out_dir> [n=12]"""
import os
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image

src, out = sys.argv[1:3]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
os.makedirs(out, exist_ok=True)
import imageio.v2 as iio2
rd = iio2.get_reader(src, "ffmpeg")
meta = rd.get_meta_data()
frames = [f for f in rd]
print("frames", len(frames), "shape", frames[0].shape, "fps", meta.get("fps"), "duration", meta.get("duration"))
idx = np.linspace(0, len(frames) - 1, n).astype(int)
imgs = []
for k, i in enumerate(idx):
    im = Image.fromarray(frames[i])
    im.save(os.path.join(out, f"frame_{k:02d}_{i:05d}.jpg"), quality=90)
    imgs.append(im)
w, h = imgs[0].size
sc = 480 / max(w, h)
tw, th = int(w * sc), int(h * sc)
cols = 4
rows = (len(imgs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * th))
for k, im in enumerate(imgs):
    sheet.paste(im.resize((tw, th)), ((k % cols) * tw, (k // cols) * th))
sheet.save(os.path.join(out, "sheet.jpg"), quality=85)
