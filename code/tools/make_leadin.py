"""(real G1: the robot has to stand for ~12 s in the frame-0 pose of the clip, which is a frozen moment of a
human in motion -- D1 frame 0: right knee nearly straight, left knee bent, turning at 104 deg/s): a LEAD-IN segment
before frame 0 of every clip.

  neutral stance (both feet flat on the floor at the frame-0 foot positions / headings, pelvis upright above the
  middle of the feet, knees symmetric, arms / waist at the default pose)  --N frames-->  frame 0 of the clip

Every lead-in frame is solved by inverse kinematics with the feet fixed on the floor (a foot that is lifted at frame 0
rises only in the last 30 % of the lead-in); pelvis pose and upper-body joints are blended with a smoothstep.  Output =
the motion file with the lead-in prepended (same keys; velocities by finite differences; foot_contact True during the
lead-in) + `leadin_frames` = N.  The clock / z tables / stake windows of the clip must be shifted by N (shift_layout()).

  python make_leadin.py preview <scene.mjb> <policy_meta.json> <motion.npz> <out.png>
  python make_leadin.py make <scene.mjb> <policy_meta.json> <motion.npz> <out_motion.npz> [N=75]
"""
import json
import sys

import mujoco
import numpy as np

STANCE_Z = None  # pelvis height of the neutral stance (None: frame-0 pelvis height - 0.02 m)


def quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def slerp(a, b, s):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + s * (b - a)
        return r / np.linalg.norm(r)
    th = np.arccos(d)
    return (np.sin((1 - s) * th) * a + np.sin(s * th) * b) / np.sin(th)


def smooth(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


class Robot:
    def __init__(self, scene, meta):
        self.m = mujoco.MjModel.from_binary_path(scene)
        self.d = mujoco.MjData(self.m)
        self.meta = json.load(open(meta))
        jn = self.meta["joint_names"]
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + j) for j in jn]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in jid])
        self.jn = jn
        self.leg = [i for i, j in enumerate(jn) if any(k in j for k in ("hip", "knee", "ankle"))]
        self.body = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "robot/" + n) for n in self.meta["body_names"]}
        self.default = np.array(self.meta["default_joint_pos"], float)

    def set(self, base_p, base_q, q):
        self.d.qpos[:] = 0
        self.d.qpos[0:3], self.d.qpos[3:7] = base_p, base_q
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)  # cdof for mj_jacBody

    def bodies(self):
        n = self.meta["body_names"]
        return np.array([self.d.xpos[self.body[b]] for b in n]), np.array([self.d.xquat[self.body[b]] for b in n])

    def ik(self, base_p, base_q, q, feet, pelvis_p, pelvis_q, q_ref, iters=150):
        """legs + floating base; targets: ankle_roll_link positions / orientations (feet), pelvis pose (soft)."""
        m, d = self.m, self.d
        base_p, base_q, q = np.array(base_p, float), np.array(base_q, float), np.array(q, float)
        dofs = list(range(6)) + [int(self.vadr[i]) for i in self.leg]
        for _ in range(iters):
            self.set(base_p, base_q, q)
            rows, J = [], []
            for name, (tp, tq, w) in feet.items():
                b = self.body[name]
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jacBody(m, d, jp, jr, b)
                ep = tp - d.xpos[b]
                er = np.zeros(3)
                qe = np.zeros(4)
                mujoco.mju_mulQuat(qe, tq, np.array([d.xquat[b][0], -d.xquat[b][1], -d.xquat[b][2], -d.xquat[b][3]]))
                mujoco.mju_quat2Vel(er, qe, 1.0)
                rows += [w * ep, w * er]
                J += [w * jp[:, dofs], w * jr[:, dofs]]
            b0 = self.body["pelvis"]
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jr, b0)
            wp = np.array([0.3, 0.3, 1.0])
            rows.append(wp * (pelvis_p - d.xpos[b0]))
            J.append(wp[:, None] * jp[:, dofs])
            er, qe = np.zeros(3), np.zeros(4)
            mujoco.mju_mulQuat(qe, pelvis_q, np.array([d.xquat[b0][0], -d.xquat[b0][1], -d.xquat[b0][2], -d.xquat[b0][3]]))
            mujoco.mju_quat2Vel(er, qe, 1.0)
            rows.append(0.5 * er)
            J.append(0.5 * jr[:, dofs])
            # posture regularisation (legs towards q_ref)
            wr = 0.05
            rr = wr * (q_ref[self.leg] - q[self.leg])
            Jr = np.zeros((len(self.leg), len(dofs)))
            Jr[:, 6:] = wr * np.eye(len(self.leg))
            rows.append(rr)
            J.append(Jr)
            r, A = np.concatenate(rows), np.vstack(J)
            dx = np.linalg.solve(A.T @ A + 1e-4 * np.eye(A.shape[1]), A.T @ r)
            dq = np.zeros(m.nv)
            dq[dofs] = dx
            qpos = d.qpos.copy()
            mujoco.mj_integratePos(m, qpos, dq, 1.0)
            base_p, base_q = qpos[0:3].copy(), qpos[3:7].copy()
            q = qpos[self.qadr].copy()
            lo, hi = self.m.jnt_range[[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + j) for j in self.jn]].T
            q = np.clip(q, lo, hi)
            if np.linalg.norm(r[:12]) < 1e-4:
                break
        self.set(base_p, base_q, q)
        return base_p, base_q, q, float(np.linalg.norm(np.concatenate(rows[:4])))


def neutral_targets(R, mo, zfloor):
    """feet flat on the floor at the frame-0 xy / heading; pelvis above the middle, heading = mean foot heading"""
    bn = R.meta["body_names"]
    P, Q = mo["body_pos_w"][0], mo["body_quat_w"][0]
    feet = {}
    yaws = []
    for n in ("left_ankle_roll_link", "right_ankle_roll_link"):
        i = bn.index(n)
        y = quat_yaw(Q[i])
        yaws.append(y)
        feet[n] = (np.array([P[i, 0], P[i, 1], zfloor]), yaw_quat(y), 1.0)
    mid = 0.5 * (feet["left_ankle_roll_link"][0] + feet["right_ankle_roll_link"][0])
    ym = np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))
    pz = STANCE_Z if STANCE_Z else float(P[0, 2]) - 0.02
    return feet, np.array([mid[0], mid[1], pz]), yaw_quat(ym)


def stand_floor(mo, R):
    bn = R.meta["body_names"]
    z = mo["body_pos_w"][:, [bn.index("left_ankle_roll_link"), bn.index("right_ankle_roll_link")], 2]
    return float(np.percentile(z, 5))  # stance ankle height above the floor


def make(R, mo, N=75):
    P0, Q0 = mo["body_pos_w"][0], mo["body_quat_w"][0]
    q0 = mo["joint_pos"][0].astype(float)
    zf = stand_floor(mo, R)
    feet_n, pel_p, pel_q = neutral_targets(R, mo, zf)
    q_neu = R.default.copy()
    # neutral stance by IK from the default pose
    bp, bq, qn, err = R.ik(pel_p, pel_q, q_neu, feet_n, pel_p, pel_q, R.default, iters=300)
    bn = R.meta["body_names"]
    fi = {n: bn.index(n) for n in feet_n}
    frames_q, frames_p, frames_r, errs = [], [], [], []
    up = [i for i in range(len(R.jn)) if i not in R.leg]  # waist / arms: blended joint angles
    for t in range(N):
        s = smooth(t / (N - 1))
        feet = {}
        for n, (tp, tq, w) in feet_n.items():
            # a foot lifted at frame 0 rises in the last 30 % of the lead-in
            sf = smooth((t / (N - 1) - 0.7) / 0.3)
            p = tp + sf * (P0[fi[n]] - tp)
            qf = slerp(tq, Q0[fi[n]], sf)
            feet[n] = (p, qf, 1.0)
        pp = pel_p + s * (P0[0] - pel_p)
        pq = slerp(pel_q, Q0[0], s)
        qr = (1 - s) * qn + s * q0
        bp, bq, q, e = R.ik(bp, bq, (qr if t == 0 else q), feet, pp, pq, qr, iters=60)
        q[up] = qr[up]
        R.set(bp, bq, q)
        bpos, bquat = R.bodies()
        frames_q.append(q.copy())
        frames_p.append(bpos)
        frames_r.append(bquat)
        errs.append(e)
    lead_q, lead_p, lead_r = np.array(frames_q), np.array(frames_p), np.array(frames_r)
    # the last lead-in frame -> frame 0: keep the clip's frame 0 exactly, drop the solved last frame if it differs
    jq = np.concatenate([lead_q[:-1], mo["joint_pos"]], 0)
    bpw = np.concatenate([lead_p[:-1], mo["body_pos_w"]], 0)
    bqw = np.concatenate([lead_r[:-1], mo["body_quat_w"]], 0)
    dt = 1.0 / float(mo["fps"])
    jv = np.gradient(jq, dt, axis=0)
    lv = np.gradient(bpw, dt, axis=0)
    av = np.zeros_like(lv)
    for i in range(1, len(bqw)):  # angular velocity from consecutive quaternions (world frame)
        for b in range(bqw.shape[1]):
            qe = np.zeros(4)
            mujoco.mju_mulQuat(qe, bqw[i, b], np.array([bqw[i - 1, b, 0], -bqw[i - 1, b, 1], -bqw[i - 1, b, 2], -bqw[i - 1, b, 3]]))
            w = np.zeros(3)
            mujoco.mju_quat2Vel(w, qe, dt)
            av[i, b] = w
    n = N - 1
    jv[n:], lv[n:], av[n:] = mo["joint_vel"], mo["body_lin_vel_w"], mo["body_ang_vel_w"]  # the clip part as recorded
    fc = np.concatenate([np.ones((n, 2), bool), mo["foot_contact"]], 0) if "foot_contact" in mo else None
    out = dict(fps=mo["fps"], joint_pos=jq.astype(np.float32), joint_vel=jv.astype(np.float32),
               body_pos_w=bpw.astype(np.float32), body_quat_w=bqw.astype(np.float32),
               body_lin_vel_w=lv.astype(np.float32), body_ang_vel_w=av.astype(np.float32), leadin_frames=np.int64(n))
    if fc is not None:
        out["foot_contact"] = fc
    return out, dict(stance_err=err, max_ik_err=float(max(errs)), neutral_q=qn, knees=(qn[R.jn.index("left_knee_joint")],
                                                                                      qn[R.jn.index("right_knee_joint")]))


def render(R, poses, out, labels):
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image, ImageDraw
    W, H = 480, 560
    R.m.vis.global_.offwidth, R.m.vis.global_.offheight = W, H
    rnd = mujoco.Renderer(R.m, H, W)
    ims = []
    for (bp, bq, q), lab in zip(poses, labels):
        R.set(bp, bq, q)
        R.d.mocap_pos[:, 2] = -10.0  # hide the obstacles
        mujoco.mj_forward(R.m, R.d)
        cam = mujoco.MjvCamera()
        cam.type, cam.distance, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 2.2, -10.0
        cam.azimuth = np.degrees(quat_yaw(bq)) + 200.0
        cam.lookat[:] = [bp[0], bp[1], 0.55]
        rnd.update_scene(R.d, cam)
        im = Image.fromarray(rnd.render())
        ImageDraw.Draw(im).text((10, 10), lab, fill=(255, 255, 255))
        ims.append(im)
    canvas = Image.new("RGB", (W * len(ims), H))
    for i, im in enumerate(ims):
        canvas.paste(im, (i * W, 0))
    canvas.save(out)


def main():
    cmd = sys.argv[1]
    R = Robot(sys.argv[2], sys.argv[3])
    mo = dict(np.load(sys.argv[4]))
    if cmd == "make":
        N = int(sys.argv[6]) if len(sys.argv) > 6 else 75
        out, info = make(R, mo, N)
        np.savez(sys.argv[5], **out)
        print(sys.argv[4].split("/")[-1], "lead-in", out["leadin_frames"], "stance IK err %.4f, max IK err %.4f, knees %.2f / %.2f"
              % (info["stance_err"], info["max_ik_err"], *info["knees"]))
    elif cmd == "preview":
        out, info = make(R, mo, 75)
        n = int(out["leadin_frames"])
        # neutral stance = first lead-in frame; frame 0 = first clip frame
        poses = []
        for k in (0, n // 2, n):
            R.set(out["body_pos_w"][k, 0], out["body_quat_w"][k, 0], out["joint_pos"][k])
            poses.append((out["body_pos_w"][k, 0].copy(), out["body_quat_w"][k, 0].copy(), out["joint_pos"][k].copy()))
        render(R, poses, sys.argv[5], ["neutral stance (new hold pose)", "lead-in 0.75 s", "clip frame 0 (old hold pose)"])
        print("knees neutral %.2f / %.2f, IK err %.4f / %.4f" % (*info["knees"], info["stance_err"], info["max_ik_err"]))


if __name__ == "__main__":
    main()
