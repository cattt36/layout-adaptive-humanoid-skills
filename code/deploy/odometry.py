"""Onboard pose estimate for the obstacle features (user: the actor may only use the robot's own sensors).

Dead reckoning from the PLANNED start pose (layout_aligned.json start = reference frame 0: the robot is placed there):
  heading  = planned heading + (IMU yaw - IMU yaw at the takeover)
  position = leg odometry: the lower foot (sole spheres, forward kinematics of the robot model with the measured joint
             angles and the IMU attitude) is the stance foot and stays fixed in the world; the pelvis follows from it;
             at a stance change the new foot is anchored where it is.  Height included (stepping onto a stake raises
             the anchor).
Only joint encoders and the IMU are used.  The motion capture is NOT needed while the policy runs (it may still be used
before the run to check the placement).

  odo = LegOdometry(export_dir)                                  # model/: scene.mjb + policy_meta.json
  odo.reset(q, imu_quat, start_pos, start_yaw)                    # at the takeover (command 1), start = planned start
  every 20 ms: pos, quat = odo.update(q, imu_quat)               -> RobotState(perc_pos=pos, perc_quat=quat)
q = joint angles in the policy joint order (meta joint_names), imu_quat = wxyz.
"""
from __future__ import annotations

import json
import os
import re

import mujoco
import numpy as np


def _yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _qz(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class LegOdometry:
    def __init__(self, export_dir: str, switch_margin: float = 0.01):
        meta = json.load(open(os.path.join(export_dir, "policy_meta.json")))
        self.m = mujoco.MjModel.from_binary_path(os.path.join(export_dir, "scene.mjb"))
        self.d = mujoco.MjData(self.m)
        jid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + j) for j in meta["joint_names"]]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in jid])
        nm = lambda i: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""  # noqa: E731
        self.soles = [[i for i in range(self.m.ngeom) if re.fullmatch(rf"robot/{s}_foot[1-7]_collision", nm(i))]
                      for s in ("left", "right")]
        self.margin = switch_margin
        self.pos, self.quat = np.zeros(3), np.array([1.0, 0, 0, 0])

    def _feet(self, q, quat):
        """sole sphere bottom points of both feet relative to the pelvis origin, world axes (pelvis orientation = quat):
        [2][7, 3]."""
        d = self.d
        d.qpos[0:3] = 0.0
        d.qpos[3:7] = quat
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        return [np.array([d.geom_xpos[i] - [0, 0, self.m.geom_size[i][0]] for i in g]) for g in self.soles]

    def _attitude(self, imu_quat):
        return _qmul(_qz(self.yaw0 + _yaw(imu_quat) - self.imu_yaw0 - _yaw(imu_quat)), imu_quat)

    def _anchor(self, pts, side):
        """new stance foot: its lowest sole point at this moment, kept (same sphere) for the whole stance."""
        self.side, self.sph = side, int(np.argmin(pts[side][:, 2]))
        self.anchor = self.pos + pts[side][self.sph]

    def reset(self, q, imu_quat, start_pos, start_yaw):
        imu_quat = np.asarray(imu_quat, float) / np.linalg.norm(imu_quat)
        self.yaw0, self.imu_yaw0 = float(start_yaw), _yaw(imu_quat)
        self.quat = self._attitude(imu_quat)
        pts = self._feet(np.asarray(q, float), self.quat)
        self.pos = np.asarray(start_pos, float).copy()
        self._anchor(pts, int(np.argmin([p[:, 2].min() for p in pts])))
        return self.pos.copy(), self.quat.copy()

    def update(self, q, imu_quat):
        imu_quat = np.asarray(imu_quat, float) / np.linalg.norm(imu_quat)
        self.quat = self._attitude(imu_quat)
        pts = self._feet(np.asarray(q, float), self.quat)
        self.pos = self.anchor - pts[self.side][self.sph]
        other = 1 - self.side
        if pts[other][:, 2].min() < pts[self.side][:, 2].min() - self.margin:  # other foot clearly lower: new stance
            self._anchor(pts, other)
        else:
            low = int(np.argmin(pts[self.side][:, 2]))
            if low != self.sph and pts[self.side][low, 2] < pts[self.side][self.sph, 2] - 0.003:
                self._anchor(pts, self.side)  # rolling over the sole (heel -> toe): the pivot moves, no jump
        return self.pos.copy(), self.quat.copy()
