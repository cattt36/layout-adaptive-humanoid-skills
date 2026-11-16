"""replacement for HnuPipeline.refresh_obstacles() of the robot team's hnu_pipeline.py (version) --
live obstacle positions are REQUIRED (user), posts AND stakes, rigid bodies first.

Paste this method into the HnuPipeline class (replacing the one) and make the takeover branches use its return
value, e.g.

        if not self.refresh_obstacles():
            print("[障碍物实测] 失败，不起步（检查刚体 / 遮挡后重按 Start）", flush=True)
            return                                   # stay in the current mode, nothing is started
        self.policy.reset()
        ...

Config (config.yaml, mocap):
  obstacle_bodies: {posts: [post1, post2, post3, post4], stakes: [stake1, stake2, stake3, stake4, stake5]}
  obstacles_live: {enabled: true, required: true, seconds: 1.0, max_jitter_m: 0.01, marker_fallback: true}
required: true -> a missing / moving obstacle or a layout outside the training coverage stops the start (the run
15:57 silently fell back to an old file while post 1 had been knocked 11 cm forward).
"""


def refresh_obstacles(self):  # noqa: C901  (method of HnuPipeline)
    import os
    import shutil

    import numpy as np
    cfg = self.hnu_cfg["mocap"].get("obstacles_live") or {}
    if not (self.real and cfg.get("enabled", True)):
        return True
    required = bool(cfg.get("required", True))
    try:
        from hnu_deploy import ClipContext
        from obstacles_live import Layout, apply_to_dir, check, fill_missing
        from obstacles_rigid import measure_obstacles
        from robojudo.hnu.dof import META
        ep = self.env.episode
        lay = Layout(ep.layout_dir)
        if lay.kind is None:
            return True
        names = (self.hnu_cfg["mocap"].get("obstacle_bodies") or {}).get(lay.kind, [])[:len(lay.plan)]
        shift = self.hnu_cfg["mocap"].get("aligned_shift_xy") or [0.0, 0.0]
        meas, lines = measure_obstacles(self.env.mocap, lay.kind, names, shift, lay.plan, cfg.get("seconds", 1.0),
                                        max_jitter=cfg.get("max_jitter_m", 0.01), marker_fallback=cfg.get("marker_fallback", True))
        print("[障碍物实测] " + "\n  ".join(lines), flush=True)
        missing = [i + 1 for i, m in enumerate(meas) if m is None]
        if missing:
            print(f"[障碍物实测] 没看到 {lay.kind} {missing}", flush=True)
            if required:
                return False
        full, _ = fill_missing(lay.plan, meas)
        rep, ok = check(lay.kind, lay.plan, full)
        print("[障碍物实测] " + "\n  ".join(rep), flush=True)
        if not ok and required:
            print("[障碍物实测] 超出训练范围：把障碍物挪近规划位置后重试", flush=True)
            return False
        # the plan copy may hold an earlier measurement: start from the sheet version of the layout file
        f = ep.file("bridge.json") if lay.kind == "posts" else ep.file("stakes.json")
        sheet = (lay.bridge_f if lay.kind == "posts" else lay.stakes_f).replace(".json", "_sheet.json")
        if os.path.exists(sheet):
            shutil.copy(sheet, f)
        print("[障碍物实测] " + "\n  ".join(apply_to_dir(ep.layout_dir, {lay.kind: meas}, ep.plan_dir)), flush=True)
        sj = ep.file("stakes.json") if os.path.exists(ep.file("stakes.json")) else None
        bj = ep.file("bridge.json") if os.path.exists(ep.file("bridge.json")) else None
        ctx = ClipContext(META, np.load(ep.file("z.npy")), sj, bj)
        self.env.ctx = self.policy.policy.ctx = ctx
        self.result["obstacles_measured"] = dict(kind=lay.kind, positions=meas)  # logged with the run
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[障碍物实测] 出错（{e!r}）", flush=True)
        return not required
