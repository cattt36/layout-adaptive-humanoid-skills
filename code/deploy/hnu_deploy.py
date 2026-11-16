"""Deployment runtime for the distilled G1 students: numpy + onnxruntime only (no torch / mjlab).

Robot-side usage (50 Hz):
    pol = StudentPolicy(export_dir)                              # policy.onnx + policy_meta.json
    ctx = ClipContext(pol.meta, z_table, stakes_json, bridge_json)  # z_skill table from the deployment chain
    pol.reset(ctx, state0)
    every 20 ms: q_target = pol.step(state, frame)                # state: RobotState from the robot / simulator
    PD on the joints: tau = kp * (q_target - q) - kd * dq, |tau| <= effort_limit (gains in meta)

RobotState (all in the policy joint order of meta["joint_names"]):
    ang_vel  [3]  IMU gyro (pelvis IMU, body frame)
    lin_vel  [3]  estimated base linear velocity (body frame, state estimator)
    quat     [4]  pelvis orientation wxyz (world)
    pos      [3]  pelvis position (world; only used by the obstacle features: localisation / mocap)
    (meta "perception_pose" = "odometry", "lin_vel_input" = "zero"): the obstacle features use perc_pos /
    perc_quat from deploy/odometry.py (leg odometry + IMU from the planned start pose), lin_vel is ignored -- only the
    robot's own sensors reach the policy.  Waist roll / pitch: action_scale 0 -> constant target 0 (locked).
    q, dq    [29] joint positions / velocities
Sensor latency is NOT simulated here; the simulator (deploy/sim2sim.py) delays the RobotState it hands over.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class RobotState:
    ang_vel: np.ndarray
    lin_vel: np.ndarray
    quat: np.ndarray
    pos: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    # optional pose used ONLY for the obstacle features (e.g. from localisation); default: pos / quat
    perc_pos: np.ndarray | None = None
    perc_quat: np.ndarray | None = None


def quat_rotate_inv(q, v):
    """Rotate world vector v into the frame of quaternion q (wxyz)."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ v


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_yaw_frame(v_xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    v = np.asarray(v_xy, dtype=np.float64)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def euler_zyx_quat(yaw, pitch):
    """Quaternion (wxyz) of R = Rz(yaw) Ry(pitch) (same as mju_euler2Quat(..., 'zyx') with roll 0)."""
    cy, sy, cp, sp = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


def add_post_bases(parts: dict, dims: dict) -> dict:
    """Support-post bases of the real bridge (copy of hnu_tasks/bridge.py add_post_bases, ): per post<i> a
    floor bar base<i> across the beam and two diagonal braces bracea<i> / braceb<i>; only placed in the simulator."""
    out = dict(parts)
    yaw = float(parts["plank"]["yaw"])
    nrm = np.array([-np.sin(yaw), np.cos(yaw)])
    # optional dims["mid"] / dims["end"] (shortened bars): overrides (bar_len, brace_d, brace_h) for the
    # inner posts / the two posts at the ends of the beam; {"none": true} = no base under them
    ax = np.array([np.cos(yaw), np.sin(yaw)])
    order = sorted((k for k in parts if k.startswith("post")), key=lambda k: float(np.array(parts[k]["center"][:2]) @ ax))
    over = {k: dims.get("mid") for k in order[1:-1]}
    over.update({k: dims.get("end") for k in (order[0], order[-1])} if order else {})
    for k, p in parts.items():
        if not k.startswith("post"):
            continue
        ov = over.get(k)
        if ov and ov.get("none"):
            continue
        dims_k = {**dims, **ov} if ov else dims
        i, c = k[4:], np.array(p["center"][:2], float)
        out[f"base{i}"] = dict(type="box", center=[float(c[0]), float(c[1]), dims_k["bar_h"] - 0.02],
                               half=[dims_k["bar_w"] / 2, dims_k["bar_len"] / 2, 0.02], yaw=yaw, pitch=0.0)
        for s, tag in ((1.0, "a"), (-1.0, "b")):
            a = np.r_[c + s * nrm * dims_k["brace_d"], dims_k["bar_h"]]
            b = np.r_[c + s * nrm * float(p["radius"]), dims_k["brace_h"]]
            v = b - a
            out[f"brace{tag}{i}"] = dict(type="box", center=(0.5 * (a + b)).tolist(),
                                         half=[0.5 * float(np.linalg.norm(v)), dims_k["brace_t"] / 2, dims_k["brace_t"] / 2],
                                         yaw=float(np.arctan2(v[1], v[0])), pitch=float(-np.arctan2(v[2], np.hypot(v[0], v[1]))))
    return out


class ClipContext:
    """Everything that depends on the chosen motion: z_skill table and obstacle layout (stakes / bridge)."""

    def __init__(self, meta: dict, z_table: np.ndarray, stakes_json: str | None = None, bridge_json: str | None = None):
        self.z = np.asarray(z_table, np.float32)
        self.T = self.z.shape[0]
        sm = meta["stakes"]
        self.stakes = []  # per stake: centre xyz (z = height / 2), foot (0 L / 1 R), core window f0, f1, xy
        if stakes_json and os.path.exists(stakes_json):
            j = json.load(open(stakes_json))
            for p, s in zip(j["stake_xyz_50hz"][: sm["n"]], j["stakes"][: sm["n"]]):
                a, b = s["stance"][0] * 50.0 / j["fps"], s["stance"][1] * 50.0 / j["fps"]
                self.stakes.append(dict(c=np.array(p, float), foot=0 if s["foot"] == "L" else 1,
                                        f0=int(round(a + 0.25 * (b - a))), f1=int(round(b - 0.25 * (b - a))),
                                        xy=np.array([s["x"], s["y"]], float)))
        self.bridge = None  # part name -> (centre, quat wxyz)
        if bridge_json and os.path.exists(bridge_json):
            j = json.load(open(bridge_json))
            parts = j["parts"] if "parts" in j else {k: dict(type="box", **v) for k, v in j["boxes"].items()}
            if meta["bridge"].get("post_base"):
                parts = add_post_bases(parts, meta["bridge"]["post_base"])
            self.bridge = {p: (np.array(parts[p]["center"], float), euler_zyx_quat(parts[p]["yaw"], parts[p]["pitch"]))
                           for p in meta["bridge"]["parts"] if p in parts}  # (a base left out by dims["mid"] stays parked)
            self.base_half = {p: parts[p]["half"] for p in parts if p.startswith(("base", "brace"))}  # box half sizes

    def set_reference(self, motion_npz: str):
        """(meta obstacle_input "reference"): planned pelvis trajectory of the layout (motion.npz of the
        planned layout: body 0 = pelvis), used instead of the measured pose for the obstacle features."""
        r = np.load(motion_npz)
        self.ref_pos, self.ref_quat = r["body_pos_w"][:, 0].astype(float), r["body_quat_w"][:, 0].astype(float)

    # -- obstacle features (training: hnu_tasks/perception.py)
    def stake_features(self, pos, quat, frame, n_ahead=2):
        out = np.zeros((n_ahead, 5))
        up = [s for s in self.stakes if s["f1"] >= frame]
        yaw = yaw_of(quat)
        for k, s in enumerate(up[:n_ahead]):
            rel = to_yaw_frame(s["c"][:2] - pos[:2], yaw)
            dt = np.clip((s["f0"] - frame) / 50.0, 0.0, 2.0)
            out[k] = [rel[0], rel[1], s["c"][2] * 2.0 - pos[2], dt, 1.0]
        return out.reshape(-1)

    def bridge_features(self, pos, quat, beam_half):
        if self.bridge is None:
            return np.zeros(7)
        yaw = yaw_of(quat)
        bc, q = self.bridge["plank"]
        w, x, y, z = q
        ax = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
        nh = max(np.linalg.norm(ax[:2]), 1e-6)
        t = np.dot(pos[:2] - bc[:2], ax[:2] / nh)
        closest = bc + ax * (t / nh)
        rel = to_yaw_frame(closest[:2] - pos[:2], yaw)
        beam = [rel[0], rel[1], closest[2] - beam_half - pos[2]]
        posts = np.array([c[:2] for n, (c, _) in self.bridge.items() if n.startswith("post")])
        if len(posts) == 0:
            return np.array(beam + [0.0] * 4)
        near = posts[np.argsort(np.linalg.norm(posts - pos[:2], axis=1), kind="stable")[:2]]
        relp = to_yaw_frame(near - pos[:2], yaw).reshape(-1)
        relp = np.concatenate([relp, np.zeros(4 - len(relp))])
        return np.concatenate([beam, relp])


class StudentPolicy:
    def __init__(self, export_dir: str, providers=("CPUExecutionProvider",), threads: int = 1):
        import onnxruntime as ort
        self.meta = json.load(open(os.path.join(export_dir, "policy_meta.json")))
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(os.path.join(export_dir, "policy.onnx"), so, providers=list(providers))
        m = self.meta
        self.default_q = np.array(m["default_joint_pos"])
        self.scale = np.array(m["action_scale"])
        self.offset = np.array(m["action_offset"])
        self.layout = m["obs_layout"]
        self.ctx = None
        self.hist = {}
        self.obst_noise = 0.0  # evaluation only (sim2sim perception-error test)
        self.rng = np.random.default_rng(0)
        self.last_action = np.zeros(m["act_dim"])

    # -- per-frame (un-delayed) term values
    def _terms(self, s: RobotState, frame: int):
        m, ctx = self.meta, self.ctx
        v = {"base_ang_vel": s.ang_vel, "projected_gravity": quat_rotate_inv(s.quat, np.array([0.0, 0.0, -1.0])),
             "joint_pos": s.q - self.default_q, "joint_vel": s.dq, "actions": self.last_action,
             # meta lin_vel_input "zero" -> the input is constant 0 (onboard sensors only, no estimate needed)
             "base_lin_vel": np.zeros(3) if m.get("lin_vel_input") == "zero" else s.lin_vel}
        for off in m["z_offsets"]:
            v[f"z_{off}"] = ctx.z[min(frame + off, ctx.T - 1)]
        if any(t["name"] == "obst_stakes" for t in self.layout):
            pp = s.pos if s.perc_pos is None else s.perc_pos
            pq = s.quat if s.perc_quat is None else s.perc_quat
            if m.get("obstacle_input") == "reference":  # plan-based features (no localisation, no sensors)
                k_ = min(frame, len(ctx.ref_pos) - 1)
                pp, pq = ctx.ref_pos[k_], ctx.ref_quat[k_]
            v["obst_stakes"] = ctx.stake_features(pp, pq, frame, m["stakes"]["n_ahead"])
            v["obst_bridge"] = ctx.bridge_features(pp, pq, m["bridge"]["beam_half"])
            if m.get("obstacle_input") == "zero":  # proprioception only -> no localisation needed
                v["obst_stakes"], v["obst_bridge"] = np.zeros_like(v["obst_stakes"]), np.zeros_like(v["obst_bridge"])
            if self.obst_noise > 0 and m.get("obstacle_input") != "zero":  # evaluation only: per-step detection noise
                st = v["obst_stakes"].reshape(-1, 5)
                st[:, :3] += (st[:, 4:5] > 0) * self.rng.uniform(-self.obst_noise, self.obst_noise, (len(st), 3))
                v["obst_stakes"] = st.reshape(-1)
                if ctx.bridge is not None:
                    v["obst_bridge"] = v["obst_bridge"] + self.rng.uniform(-self.obst_noise, self.obst_noise, 7)
        return v

    def reset(self, ctx: ClipContext, s: RobotState, frame: int = 0):
        self.ctx = ctx
        self.last_action = np.zeros(self.meta["act_dim"])
        v = self._terms(s, frame)
        # history buffers start filled with the first frame (as the training env after a reset)
        self.hist = {t["name"]: [np.asarray(v[t["name"]], float)] * t["history"] for t in self.layout if t["history"] > 0}

    def observation(self, s: RobotState, frame: int) -> np.ndarray:
        v = self._terms(s, frame)
        parts = []
        for t in self.layout:
            n = t["name"]
            x = np.asarray(v[n], float).reshape(-1)
            if t["history"] > 0:
                self.hist[n] = self.hist[n][1:] + [x]
                x = np.concatenate(self.hist[n])  # oldest -> newest
            parts.append(x)
        return np.concatenate(parts).astype(np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        a = self.sess.run(None, {"obs": obs[None]})[0][0].astype(np.float64)
        self.last_action = a
        return a

    def step(self, s: RobotState, frame: int) -> np.ndarray:
        """Observation -> action -> joint position targets."""
        a = self.act(self.observation(s, frame))
        return self.offset + self.scale * a


# ------------------------------------------------------------------------------------------------ blind tracker
def quat_mul_np(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class TrackerPolicy:
    """BeyondMimic-style BLIND tracker (one per task, tools/export_tracker.py).  The robot plays the
    reference motion of the planned layout (motion.npz) and needs only its own sensors:
      obs = [ref joint pos (29), ref joint vel (29), 0 (3, anchor position), anchor orientation (6), 0 (3, linear
             velocity), IMU gyro (3), joint pos - default (29), joint vel (29), last action (29), 0 (10 / 7, obstacles)]
      anchor orientation = first two columns of R(q_torso)^T R(q_torso_ref), torso = anchor body (torso_link):
             q_torso from the pelvis attitude (IMU) and the joint angles by forward kinematics; heading = planned
             start heading + (IMU yaw - IMU yaw at the takeover) (set_heading_origin)
    While the clock is held (frame 0 before the start): reference joint velocity 0 (as in training).
      pol = TrackerPolicy(export_dir); pol.reset(motion_npz, state0, start_yaw)
      every 20 ms: q_target = pol.step(state, frame, holding)"""

    def __init__(self, export_dir: str, providers=("CPUExecutionProvider",), threads: int = 1):
        import mujoco
        self.meta = m = json.load(open(os.path.join(export_dir, "policy_meta.json")))
        self._ort = (os.path.join(export_dir, "policy.onnx"), providers, threads)  # session created at the first step()
        self.sess = None
        self.default_q, self.scale, self.offset = (np.array(m[k]) for k in ("default_joint_pos", "action_scale", "action_offset"))
        self.last_action = np.zeros(m["act_dim"])
        self.mj = mujoco
        # scene (robot kinematics for the torso FK): <export_dir>/scene.mjb, else one shared ../scene.mjb (demo bundle)
        sc = os.path.join(export_dir, "scene.mjb")
        sc = sc if os.path.exists(sc) else os.path.join(os.path.dirname(os.path.abspath(export_dir)), "scene.mjb")
        self.model = mujoco.MjModel.from_binary_path(sc)
        self.data = mujoco.MjData(self.model)
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + j) for j in m["joint_names"]]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in jid])
        self.anchor_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot/" + m["tracker"]["anchor_body"])
        self.ref_anchor = m["body_names"].index(m["tracker"]["anchor_body"])
        self.n_perc = int(m["tracker"]["perception_dim"])

    def torso_quat(self, pelvis_quat, q):
        d = self.data
        d.qpos[0:3] = 0.0
        d.qpos[3:7] = pelvis_quat
        d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.model, d)
        return d.xquat[self.anchor_bid].copy()

    def reset(self, motion_npz: str, s: RobotState, start_yaw: float | None = None):
        r = np.load(motion_npz)
        self.ref_q, self.ref_dq = r["joint_pos"].astype(float), r["joint_vel"].astype(float)
        self.ref_aq = r["body_quat_w"][:, self.ref_anchor].astype(float)
        self.T = len(self.ref_q)
        self.last_action = np.zeros(self.meta["act_dim"])
        # heading origin: the IMU yaw at the takeover is mapped to the planned start heading (None: use s.quat as is)
        self.yaw_off = 0.0 if start_yaw is None else float(start_yaw) - yaw_of(s.quat)

    def observation(self, s: RobotState, frame: int, holding: bool = False) -> np.ndarray:
        k = min(max(int(frame), 0), self.T - 1)
        qp = quat_mul_np(np.array([np.cos(self.yaw_off / 2), 0, 0, np.sin(self.yaw_off / 2)]), s.quat)
        qa = self.torso_quat(qp, s.q)
        rel = quat_mat(qa).T @ quat_mat(self.ref_aq[k])
        ref_dq = np.zeros_like(self.ref_dq[k]) if holding else self.ref_dq[k]
        return np.concatenate([self.ref_q[k], ref_dq, np.zeros(3), rel[:, :2].reshape(-1), np.zeros(3), s.ang_vel,
                               s.q - self.default_q, s.dq, self.last_action, np.zeros(self.n_perc)]).astype(np.float32)

    def step(self, s: RobotState, frame: int, holding: bool = False) -> np.ndarray:
        if self.sess is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = so.inter_op_num_threads = self._ort[2]
            self.sess = ort.InferenceSession(self._ort[0], so, providers=list(self._ort[1]))
        a = self.sess.run(None, {"obs": self.observation(s, frame, holding)[None]})[0][0].astype(np.float64)
        self.last_action = a
        return self.offset + self.scale * a
