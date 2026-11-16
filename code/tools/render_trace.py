"""render a sim2sim trace (HNU_S2S_TRACE) of the rope start: phase labels (rope releasing / standing with
the clock held at frame 0 / motion).  No CJK text on the videos (user, ).
  MUJOCO_GL=egl python tools/render_trace.py <export_dir> <trace.npz> <out.mp4> --label "D1 ..." --hold 12 --rope 0.8,10
"""
import argparse
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--label", default="")
    ap.add_argument("--hold", type=float, default=0.0)
    ap.add_argument("--rope", default="")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    a = ap.parse_args()
    m = mujoco.MjModel.from_binary_path(f"{a.export_dir}/scene.mjb")
    W, H = 960, 640
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    tr = np.load(a.trace)
    dd = mujoco.MjData(m)
    dd.mocap_pos[:], dd.mocap_quat[:] = tr["mocap_pos"], tr["mocap_quat"]
    q = tr["qpos"]
    dt = float(tr["control_dt"])
    vis = [p for p in tr["mocap_pos"] if p[2] > -1.0]
    ob = np.array(vis)[:, :2] if vis else q[:, :2]
    u = np.linalg.svd(ob - ob.mean(0))[2][0] if len(ob) > 1 else np.array([1.0, 0.0])
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 4.5, -20.0
    cam.azimuth = np.degrees(np.arctan2(u[1], u[0])) - 90.0
    rnd = mujoco.Renderer(m, H, W)
    font = ImageFont.truetype(FONT, 22)
    rope = tuple(float(x) for x in a.rope.split(",")) if a.rope else None
    Hs = int(round(a.hold / dt))
    ok = bool(tr["success_v2"])
    res = "Success" if ok else ("Fall" if str(tr["cause"]) == "fall" else "Not completed")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], stdin=subprocess.PIPE)
    look = q[0, :3].copy()
    step = a.speed / (25 * dt)
    i, n = 0.0, len(q)
    while True:
        k = min(int(round(i)), n - 1)
        dd.qpos[:] = q[k]
        mujoco.mj_forward(m, dd)
        look = 0.9 * look + 0.1 * np.r_[dd.qpos[0:2], 0.55]
        cam.lookat[:] = look
        rnd.update_scene(dd, cam)
        im = Image.fromarray(rnd.render())
        d = ImageDraw.Draw(im)
        t = k * dt
        if rope and t < rope[1]:
            ph, col = f"1. rope releasing: pull {100 * rope[0] * (1 - t / rope[1]):.0f} % body weight (frame 0 held)", (255, 200, 120)
        elif k < Hs:
            ph, col = "2. rope slack, standing still on its own (frame 0 held)", (255, 240, 140)
        else:
            ph, col = f"3. motion running, frame {k - Hs}", (150, 220, 255)
        for y, s, c in ((12, a.label, (255, 255, 255)), (44, f"t = {t:4.1f} s   (sim2sim, deployed ONNX policy)", (255, 255, 255)),
                        (76, ph, col)):
            d.text((14, y), s, font=font, fill=c, stroke_width=2, stroke_fill=(0, 0, 0))
        if k == n - 1:
            d.text((14, 108), f"Result: {res}", font=font, fill=(120, 255, 120) if ok else (255, 110, 110), stroke_width=2,
                   stroke_fill=(0, 0, 0))
        ff.stdin.write(np.asarray(im).tobytes())
        if k == n - 1:
            for _ in range(40):
                ff.stdin.write(np.asarray(im).tobytes())
            break
        i += step
    ff.stdin.close()
    ff.wait()
    print(a.out, res)


if __name__ == "__main__":
    main()
