"""DuMuQiao bridge as per-env mocap geoms (tools/make_bridge.py: plank / ramp / leg boxes; tools/make_bridge_usd.py:
DuMuQiao.usd parts incl. support posts).

Geom sizes are fixed at compile time (median over the clips' scaled sizes); every env places the parts of the clip it
tracks from <clip>_bridge.json at each resample.  Envs whose clip has no bridge park the parts below the floor.
"""
from __future__ import annotations

import glob
import json
import os

import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg

HIDDEN_Z = -10.0

# Support-post bases: every post of the real bridge stands on a floor bar (across the beam, centred on the
# post) with two diagonal braces in the bar's vertical plane, from the bar top brace_d off the post axis up to the post
# surface at brace_h (user measurement brace 50 cm long, foot 10 cm from the bar end -> brace_d 0.40, brace_h
# ~0.375 with the post radius ~4 cm).  They were there in the motion capture too but missing in the sim  (the robot
# tripped over a bar).  HNU_POST_BASE=1 adds them as parts base<i> / bracea<i> / braceb<i>; dimensions in m from
# HNU_BASE_DIMS = bar_len,bar_w,bar_h,brace_d,brace_h,brace_t (real size, not scaled with the clip).  Off by default.
BASE_KEYS = ("bar_len", "bar_w", "bar_h", "brace_d", "brace_h", "brace_t")
BASE_DIMS = dict(zip(BASE_KEYS, map(float, os.environ.get("HNU_BASE_DIMS", "1.0,0.10,0.03,0.40,0.375,0.04").split(",")))) \
    if os.environ.get("HNU_POST_BASE", "0") == "1" else None
BAR_HALF_Z = 0.02  # bar box half height: centred at bar_h - 0.02, i.e. reaching below the floor (height jitter = z shift)
# , bars cut on the real bridge: HNU_BASE_MID / HNU_BASE_END = json overrides (bar_len, brace_d,
# brace_h) for the inner posts / the two posts at the ends of the beam, as deploy/sim2sim.py and the export meta
# (bridge.post_base.mid / .end).  Post order along the beam is the same in every clip (post0 .. post3).
if BASE_DIMS is not None:
    for _ev, _key in (("HNU_BASE_MID", "mid"), ("HNU_BASE_END", "end")):
        if os.environ.get(_ev):
            BASE_DIMS[_key] = json.loads(os.environ[_ev])


def add_post_bases(parts: dict, dims: dict) -> dict:
    """parts + base<i> (floor bar) and bracea<i> / braceb<i> (diagonal braces) for every post<i> (deploy/hnu_deploy.py
    has a copy).  dims["mid"] / dims["end"]: overrides for the inner / end posts."""
    out = dict(parts)
    yaw = float(parts["plank"]["yaw"])
    nrm = np.array([-np.sin(yaw), np.cos(yaw)])  # horizontal normal of the beam
    ax = np.array([np.cos(yaw), np.sin(yaw)])
    order = sorted((k for k in parts if k.startswith("post")), key=lambda k: float(np.array(parts[k]["center"][:2]) @ ax))
    over = {k: dims.get("mid") for k in order[1:-1]}
    over.update({k: dims.get("end") for k in (order[0], order[-1])} if order else {})
    base_dims = dims
    for k, p in parts.items():
        if not k.startswith("post"):
            continue
        dims = {**base_dims, **over[k]} if over.get(k) else base_dims
        i, c = k[4:], np.array(p["center"][:2], float)
        out[f"base{i}"] = dict(type="box", center=[float(c[0]), float(c[1]), dims["bar_h"] - BAR_HALF_Z],
                               half=[dims["bar_w"] / 2, dims["bar_len"] / 2, BAR_HALF_Z], yaw=yaw, pitch=0.0)
        for s, tag in ((1.0, "a"), (-1.0, "b")):
            a = np.r_[c + s * nrm * dims["brace_d"], dims["bar_h"]]
            b = np.r_[c + s * nrm * float(p["radius"]), dims["brace_h"]]
            v = b - a
            out[f"brace{tag}{i}"] = dict(type="box", center=(0.5 * (a + b)).tolist(),
                                         half=[0.5 * float(np.linalg.norm(v)), dims["brace_t"] / 2, dims["brace_t"] / 2],
                                         yaw=float(np.arctan2(v[1], v[0])), pitch=float(-np.arctan2(v[2], np.hypot(v[0], v[1]))))
    return out


def is_post_part(name: str) -> bool:
    """Parts standing on the floor that move with their post (the post itself and its base)."""
    return name.startswith(("post", "base", "brace"))


def post_owner(name: str) -> str:
    """post<i> of a post / base part."""
    for pre in ("post", "base", "bracea", "braceb"):
        if name.startswith(pre) and name[len(pre):].isdigit():
            return "post" + name[len(pre):]
    return name


def _parts(meta: dict) -> dict:
    p = meta["parts"] if "parts" in meta else {k: dict(type="box", **v) for k, v in meta["boxes"].items()}
    return add_post_bases(p, BASE_DIMS) if BASE_DIMS is not None and "plank" in p else p


POST_SHAPE = os.environ.get("HNU_POST_SHAPE", "cylinder")  # cylinder | mesh<N> (convex N-gon prism, same radius)


def _geom_fn(kind: str, size: list):
    """Bridge part collision geom.  MuJoCo-Warp loses contacts with cylinders once a sphere / capsule is
    deep inside (CPU MuJoCo keeps them); HNU_POST_SHAPE=mesh<N> replaces the cylindrical posts by convex prisms."""
    def fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="bridge_part")
        g = body.add_geom()
        g.name = "bridge_collision"
        if kind == "box":
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = list(size)
        elif POST_SHAPE.startswith("mesh"):
            nside = int(POST_SHAPE[4:] or 12)
            r, hh = float(size[0]), float(size[1])
            ang = np.arange(nside) * 2 * np.pi / nside
            ring = np.stack([r * np.cos(ang), r * np.sin(ang)], -1)
            verts = np.concatenate([np.c_[ring, np.full(nside, -hh)], np.c_[ring, np.full(nside, hh)]])
            m = spec.add_mesh()
            m.name = "post_prism"
            m.uservert = verts.astype(np.float64).flatten().tolist()
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = "post_prism"
        else:
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = list(size)
        g.rgba = [0.50, 0.38, 0.26, 1.0]
        g.friction = [1.0, 0.005, 0.0001]
        g.condim = 3
        g.contype = 1
        g.conaffinity = 1
        return spec
    return fn


def _metas(bridge_dir: str) -> list[dict]:
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(bridge_dir, "*_bridge.json")))]


def part_names(bridge_dir: str) -> list[str]:
    return list(_parts(_metas(bridge_dir)[0]))


def bridge_entity_cfgs(bridge_dir: str) -> dict[str, EntityCfg]:
    metas = _metas(bridge_dir)
    cfgs = {}
    for p, v in _parts(metas[0]).items():
        if v["type"] == "box":
            size = np.median([_parts(m)[p]["half"] for m in metas], axis=0).tolist()
        else:
            size = [float(np.median([_parts(m)[p]["radius"] for m in metas])),
                    float(np.median([_parts(m)[p]["half_height"] for m in metas])), 0.0]
        c = EntityCfg(spec_fn=_geom_fn(v["type"], size))
        c.init_state = EntityCfg.InitialStateCfg(pos=(0.0, 0.0, HIDDEN_Z))
        cfgs[f"bridge_{p}"] = c
    return cfgs


def _quat(yaw: float, pitch: float) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([yaw, pitch, 0.0]), "zyx")
    return q.tolist()


def load_bridge_table(files: list[str], device: str, names: list[str]) -> torch.Tensor:
    """[num_clips, len(names), 7] part poses (pos, quat wxyz) in motion coordinates; hidden when no bridge json."""
    tab = np.zeros((len(files), len(names), 7), np.float32)
    tab[:, :, 2] = HIDDEN_Z
    tab[:, :, 3] = 1.0
    for c, f in enumerate(files):
        j = f.replace("_motion.npz", "_bridge.json")
        if os.path.exists(j):
            parts = _parts(json.load(open(j)))
            for i, p in enumerate(names):
                b = parts[p]
                tab[c, i, :3] = b["center"]
                tab[c, i, 3:] = _quat(b["yaw"], b["pitch"])
    return torch.tensor(tab, device=device)


def bridge_part_offset(offset: torch.Tensor, names: list[str]) -> torch.Tensor:
    """Per-env bridge placement error [B, 3] (dx, dy, dz) -> per-part offsets [B, P, 3]: the whole bridge shifts
    horizontally; the height error moves the beam / ramp / leg, the support posts (and their bases) stand on the ground
    (xy only)."""
    out = offset[:, None, :].repeat(1, len(names), 1)
    for i, p in enumerate(names):
        if is_post_part(p):
            out[:, i, 2] = 0.0
    return out


def place_bridge(env, clip_ids: torch.Tensor, env_ids: torch.Tensor, table: torch.Tensor, names: list[str],
                 offset: torch.Tensor | None = None, part_off: torch.Tensor | None = None) -> None:
    """offset: whole-bridge placement error [n, 3]; part_off: explicit per-part offsets [n, P, 3] (takes priority)."""
    origins = env.scene.env_origins[env_ids]
    po = part_off if part_off is not None else (bridge_part_offset(offset, names) if offset is not None else None)
    for i, p in enumerate(names):
        pose = table[clip_ids, i].clone()
        pose[:, :3] += origins
        if po is not None:  # placement randomisation (HNU_BRIDGE_JITTER_XY / _Z)
            pose[:, :3] += po[:, i]
        env.scene[f"bridge_{p}"].write_mocap_pose_to_sim(pose, env_ids=env_ids)
