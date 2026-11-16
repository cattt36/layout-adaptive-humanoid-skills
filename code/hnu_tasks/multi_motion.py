"""Multi-clip motion command for mjlab tracking.

`motion_file` may be a single .npz, a directory (all *_motion.npz inside) or a comma-separated list.
All clips are concatenated into one frame table; every environment tracks one clip at a time and is
re-sampled (new clip + start frame) when it reaches the end of its clip.  Start frames are drawn with
BeyondMimic-style adaptive sampling over (clip, 1 s bin) cells, so that hard parts of hard clips are
visited more often.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform


TASK_NAMES = ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse")


def resolve_motion_files(spec: str) -> list[str]:
    """A file, a directory of *_motion.npz, or a comma-separated list of files and/or directories."""
    files = []
    for part in [s for s in spec.split(",") if s]:
        if os.path.isdir(part):
            files += sorted(glob.glob(os.path.join(part, "*_motion.npz")))
        else:
            files.append(part)
    # held-out protocol: HNU_CLIPS_ALLOW = file(s) with allowed clip names (one per line, comma-separated paths);
    # set for student TRAINING only, so test-subject clips are never seen in DAgger
    allow = os.environ.get("HNU_CLIPS_ALLOW", "")
    if allow:
        names = {ln.strip() for p in allow.split(",") if p for ln in open(p) if ln.strip()}
        files = [f for f in files if os.path.basename(f).replace("_motion.npz", "") in names]
    assert files, f"no motion files in {spec}"
    return files


def task_of_clip(name: str) -> int:
    for i, t in enumerate(TASK_NAMES):
        if name.startswith(t + "_"):
            return i
    return -1


def clip_weights(motion, device):
    """Per-clip sampling weight; None when neither knob is set.
    HNU_TASK_WEIGHT="CrouchedTraverse:1.5,..."  per-action factor (default 1).
    HNU_TS_SHARE=0.5  time-scaled copies (clip names containing "_ts", tools/make_timescale.py) share this fraction of
    their action's sampling mass; the action's total mass stays (factor x number of original clips), so adding
    copies does not change the balance between the actions."""
    spec = os.environ.get("HNU_TASK_WEIGHT", "")
    share = float(os.environ.get("HNU_TS_SHARE", "0"))
    if not spec and share <= 0:
        return None
    wt = dict((k, float(v)) for k, v in (s.split(":") for s in spec.split(",") if s))
    is_ts = torch.tensor(["_ts" in n for n in motion.names], device=device)
    w = torch.tensor([wt.get(TASK_NAMES[t], 1.0) if t >= 0 else 1.0 for t in motion.clip_task.tolist()], device=device)
    if share > 0:
        for t in range(len(TASK_NAMES)):
            sel = motion.clip_task == t
            n_o, n_c = int((sel & ~is_ts).sum()), int((sel & is_ts).sum())
            if n_o and n_c:
                w[sel & ~is_ts] *= 1 - share
                w[sel & is_ts] *= share * n_o / n_c
    return w


def task_sampling_weights(motion, clip_ids, device):
    """Per-sample weight (clip_weights) of every bin / candidate (clip_ids); None when unset."""
    if not hasattr(motion, "_clip_w"):
        motion._clip_w = clip_weights(motion, device)
    return None if motion._clip_w is None else motion._clip_w[clip_ids]


class MultiMotionLoader:
    def __init__(self, files: list[str], body_indexes: torch.Tensor, device: str, fps: float, latent_dir: str = ""):
        keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        parts = {k: [] for k in keys}
        contacts = []
        self.clip_start, self.clip_len, self.names = [], [], []
        n = 0
        for f in files:
            d = np.load(f)
            assert abs(float(d["fps"]) - fps) < 1e-3, f"{f}: fps {float(d['fps'])} != env {fps}"
            T = d["joint_pos"].shape[0]
            for k in keys:
                parts[k].append(d[k])
            contacts.append(d["foot_contact"] if "foot_contact" in d else np.zeros((T, 2), bool))
            self.clip_start.append(n)
            self.clip_len.append(T)
            self.names.append(os.path.basename(f).replace("_motion.npz", ""))
            n += T
        cat = {k: torch.tensor(np.concatenate(v), dtype=torch.float32, device=device) for k, v in parts.items()}
        self.joint_pos, self.joint_vel = cat["joint_pos"], cat["joint_vel"]
        self._body_indexes = body_indexes
        self.body_pos_w = cat["body_pos_w"][:, body_indexes]
        self.body_quat_w = cat["body_quat_w"][:, body_indexes]
        self.body_lin_vel_w = cat["body_lin_vel_w"][:, body_indexes]
        self.body_ang_vel_w = cat["body_ang_vel_w"][:, body_indexes]
        self.foot_contact = torch.tensor(np.concatenate(contacts), dtype=torch.bool, device=device)
        self.time_step_total = n
        self.num_clips = len(files)
        self.clip_start = torch.tensor(self.clip_start, dtype=torch.long, device=device)
        self.clip_len = torch.tensor(self.clip_len, dtype=torch.long, device=device)
        self.clip_end = self.clip_start + self.clip_len
        self.frame_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=device), self.clip_len)
        self.clip_task = torch.tensor([task_of_clip(n) for n in self.names], dtype=torch.long, device=device)
        # optional per-frame skill latents: <latent_dir>/<clip>_z.npy with shape [clip_len, D]
        self.z = None
        if latent_dir:
            zs = []
            for name, T in zip(self.names, self.clip_len.tolist()):
                z = np.load(os.path.join(latent_dir, f"{name}_z.npy"))
                assert z.shape[0] == T, f"{name}: latent length {z.shape[0]} != clip {T}"
                zs.append(z)
            self.z = torch.tensor(np.concatenate(zs), dtype=torch.float32, device=device)


class MultiMotionCommand(MotionCommand):
    cfg: "MultiMotionCommandCfg"

    def __init__(self, cfg: "MultiMotionCommandCfg", env):
        files = resolve_motion_files(cfg.motion_file)
        # Parent __init__ loads a single file; give it the first clip, then replace the loader.
        first = cfg.motion_file
        cfg.motion_file = files[0]
        super().__init__(cfg, env)
        cfg.motion_file = first
        self.motion = MultiMotionLoader(files, self.body_indexes, self.device, 1.0 / env.step_dt, cfg.latent_dir)
        m = self.motion
        # (clip, bin) cells of ~1 s
        bins_per_clip = torch.clamp((m.clip_len.float() * env.step_dt).ceil().long(), min=1)
        self.bin_clip = torch.repeat_interleave(torch.arange(m.num_clips, device=self.device), bins_per_clip)
        first_bin = torch.cumsum(bins_per_clip, 0) - bins_per_clip
        self.bin_index_in_clip = torch.arange(len(self.bin_clip), device=self.device) - first_bin[self.bin_clip]
        self.bins_per_clip = bins_per_clip
        self.first_bin = first_bin
        self.bin_count = int(bins_per_clip.sum())
        self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, device=self.device)
        self.clip_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["clip_id"] = torch.zeros(self.num_envs, device=self.device)
        self.stake_table = None
        self.stake_windows = None
        self.stake_offset = None
        if cfg.use_stakes:
            from .stakes import N_STAKES, load_stake_table, load_stake_windows
            self.stake_table = load_stake_table(files, self.device)
            self.stake_windows = load_stake_windows(files, self.device)
            # per-env horizontal stake offsets (layout randomisation, HNU_STAKE_JITTER): applied consistently to the
            # placed stakes, the perception features (teacher and student) and the missed-stake check
            self.stake_offset = torch.zeros(self.num_envs, N_STAKES, 2, device=self.device)
        self.bridge_table = None
        if cfg.use_bridge:
            from .bridge import load_bridge_table, part_names
            # (demo flat CT): no loaded clip has a bridge -> the parts come from the motion dir's template json
            bdir = next((os.path.dirname(f) for f in files if os.path.exists(f.replace("_motion.npz", "_bridge.json"))),
                        os.path.dirname(files[0]))
            self.bridge_parts = part_names(bdir)
            self.bridge_table = load_bridge_table(files, self.device, self.bridge_parts)
            self.bridge_offset = torch.zeros(self.num_envs, 3, device=self.device)  # placement error (dx, dy, dz)
            self.bridge_part_off = torch.zeros(self.num_envs, len(self.bridge_parts), 3, device=self.device)  # per part

    # -- helpers
    def _course_dirs(self) -> torch.Tensor:
        """[C, 2] unit direction of each clip's stake course (first -> last visible stake); (1, 0) without stakes."""
        if getattr(self, "_course_dir", None) is None:
            tab = self.stake_table
            out = torch.zeros(tab.shape[0], 2, device=self.device)
            out[:, 0] = 1.0
            for c in range(tab.shape[0]):
                vis = torch.nonzero(tab[c, :, 2] > -5.0).flatten()
                if len(vis) >= 2:
                    d = tab[c, vis[-1], :2] - tab[c, vis[0], :2]
                    out[c] = d / d.norm().clamp(min=1e-6)
            self._course_dir = out
        return self._course_dir

    def _straight_offsets(self) -> torch.Tensor:
        """[C, N_STAKES, 2] offsets that move a clip's stakes onto the best-fit straight line through them
        (perpendicular projection; spacing along the line unchanged).  Only ThreeStepPoleRun (3 stakes, laid out in a line
        on the real course); FiveStepPoleRun is a zig-zag by design and stays as it is.  Hidden stakes / other clips: 0."""
        if getattr(self, "_straight_off", None) is None:
            tab = self.stake_table
            out = torch.zeros(tab.shape[0], tab.shape[1], 2, device=self.device)
            for c in range(tab.shape[0]):
                vis = tab[c, :, 2] > -5.0
                if not self.motion.names[c].startswith("ThreeStepPoleRun") or int(vis.sum()) != 3:
                    continue
                xy = tab[c, vis, :2].cpu()  # once per clip, on the CPU (no CUDA linalg library in this install)
                ctr = xy.mean(0)
                u = torch.linalg.svd(xy - ctr)[2][0]
                proj = ctr + ((xy - ctr) @ u)[:, None] * u
                out[c, vis] = (proj - xy).to(self.device)
            self._straight_off = out
        return self._straight_off

    def _frame_bin(self, frames: torch.Tensor) -> torch.Tensor:
        m = self.motion
        clip = m.frame_clip[frames]
        local = frames - m.clip_start[clip]
        b = (local * self.bins_per_clip[clip]) // m.clip_len[clip]
        return self.first_bin[clip] + torch.clamp(b, max=self.bins_per_clip[clip] - 1)

    def _bins_to_frames(self, bins: torch.Tensor) -> torch.Tensor:
        m = self.motion
        clip = self.bin_clip[bins]
        u = (self.bin_index_in_clip[bins].float() + sample_uniform(0.0, 1.0, (len(bins),), device=self.device))
        local = (u / self.bins_per_clip[clip].float() * (m.clip_len[clip] - 1).float()).long()
        return m.clip_start[clip] + local

    # -- sampling
    def _adaptive_sampling(self, env_ids: torch.Tensor):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            fail_bins = self._frame_bin(self.time_steps[env_ids][episode_failed])
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count).float()
        p = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        w = task_sampling_weights(self.motion, self.bin_clip, self.device)  # HNU_TASK_WEIGHT (e.g. CrouchedTraverse:1.5)
        if w is not None:
            p = p * w
        p = p / p.sum()
        bins = torch.multinomial(p, len(env_ids), replacement=True)
        self.time_steps[env_ids] = self._bins_to_frames(bins)
        H = -(p * (p + 1e-12).log()).sum()
        self.metrics["sampling_entropy"][:] = H / np.log(self.bin_count) if self.bin_count > 1 else 1.0
        self.metrics["sampling_top1_prob"][:] = p.max()
        self.metrics["sampling_top1_bin"][:] = p.argmax().float() / self.bin_count

    def _uniform_sampling(self, env_ids: torch.Tensor):
        bins = torch.randint(0, self.bin_count, (len(env_ids),), device=self.device)
        self.time_steps[env_ids] = self._bins_to_frames(bins)

    def _resample_command(self, env_ids: torch.Tensor):
        if self.cfg.sampling_mode == "start":
            if self.cfg.fixed_clip >= 0:
                clips = torch.full((len(env_ids),), self.cfg.fixed_clip, device=self.device, dtype=torch.long)
            elif self.cfg.start_clip_mode == "env_mod":
                clips = env_ids.long() % self.motion.num_clips
            else:
                clips = torch.randint(0, self.motion.num_clips, (len(env_ids),), device=self.device)
            self.time_steps[env_ids] = self.motion.clip_start[clips]
        elif self.cfg.sampling_mode == "uniform":
            self._uniform_sampling(env_ids)
        else:
            self._adaptive_sampling(env_ids)
        # fine-tuning guard: a share of the resets starts at the first frame of a random clip, so that the start of
        # the motion (standing start, run-up) keeps being practised and is not forgotten (HNU_START_FRAC)
        sf = float(os.environ.get("HNU_START_FRAC", "0"))
        if sf > 0 and self.cfg.sampling_mode != "start":
            pick = torch.rand(len(env_ids), device=self.device) < sf
            if pick.any():
                wc = task_sampling_weights(self.motion, torch.arange(self.motion.num_clips, device=self.device), self.device)
                if wc is None:
                    clips = torch.randint(0, self.motion.num_clips, (int(pick.sum()),), device=self.device)
                else:
                    clips = torch.multinomial(wc / wc.sum(), int(pick.sum()), replacement=True)
                self.time_steps[env_ids[pick]] = self.motion.clip_start[clips]
        self.clip_id[env_ids] = self.motion.frame_clip[self.time_steps[env_ids]]
        self.metrics["clip_id"][env_ids] = self.clip_id[env_ids].float()

        n = len(env_ids)
        rest, step_st = self._pick_rest_starts(env_ids)  # sets self.hold[env_ids] (before the velocities are read)
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_ori = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()
        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        r = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        s = sample_uniform(r[:, 0], r[:, 1], (n, 6), device=self.device)
        root_pos += s[:, 0:3]
        root_ori = quat_mul(quat_from_euler_xyz(s[:, 3], s[:, 4], s[:, 5]), root_ori)
        r = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        s = sample_uniform(r[:, 0], r[:, 1], (n, 6), device=self.device)
        root_lin_vel += s[:, :3]
        root_ang_vel += s[:, 3:]
        joint_pos = self.joint_pos[env_ids].clone()
        joint_pos += sample_uniform(self.cfg.joint_position_range[0], self.cfg.joint_position_range[1],
                                    joint_pos.shape, device=self.device)
        joint_vel = self.joint_vel[env_ids].clone()
        if rest.any():
            self._rest_state(env_ids, rest, step_st, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel)
        if not hasattr(self, "rest_flag"):  # rest starts (odometry error model: the start pose is the planned one)
            self.rest_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rest_flag[env_ids] = rest
        if os.environ.get("HNU_WAIST_LOCK", "0") == "1":  # locked waist roll / pitch start at 0
            if not hasattr(self, "_waist_ids"):
                jn = self._env.scene[self.cfg.entity_name].joint_names
                self._waist_ids = [jn.index(n) for n in ("waist_roll_joint", "waist_pitch_joint")]
            wo = getattr(self._env, "_hnu_waist_off", None)  # HNU_WAIST_OFF: locked around a per-episode offset
            joint_pos[:, self._waist_ids] = wo[env_ids] if wo is not None else 0.0
            joint_vel[:, self._waist_ids] = 0.0
        self._write_reference_state_to_sim(env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel,
                                           joint_pos, joint_vel)
        if self.stake_table is not None:
            from .stakes import place_stakes
            jit = float(os.environ.get("HNU_STAKE_JITTER", "0"))  # per stake, uniform +-jit (m)
            shift = float(os.environ.get("HNU_STAKE_SHIFT", "0"))  # whole layout, uniform +-shift (m)
            straight = float(os.environ.get("HNU_STAKE_STRAIGHT", "0"))  # share of resets with a straight-line layout
            # anisotropic jitter: HNU_STAKE_JITTER_ALONG / _LAT along / across the course direction
            # (along-course jitter changes the jump length; long W1 strides are close to the G1 jumping limit)
            j_al = float(os.environ.get("HNU_STAKE_JITTER_ALONG", "0"))
            j_lat = float(os.environ.get("HNU_STAKE_JITTER_LAT", "0"))
            if jit > 0 or shift > 0 or straight > 0 or j_al > 0 or j_lat > 0:
                off = sample_uniform(-jit, jit, (n, self.stake_offset.shape[1], 2), device=self.device) if jit > 0 else \
                    torch.zeros(n, self.stake_offset.shape[1], 2, device=self.device)
                if j_al > 0 or j_lat > 0:
                    u = self._course_dirs()[self.clip_id[env_ids]][:, None, :]  # [n, 1, 2]
                    nrm = torch.stack([-u[..., 1], u[..., 0]], -1)
                    S = self.stake_offset.shape[1]
                    off = off + sample_uniform(-j_al, j_al, (n, S, 1), device=self.device) * u \
                        + sample_uniform(-j_lat, j_lat, (n, S, 1), device=self.device) * nrm
                if shift > 0:
                    off = off + sample_uniform(-shift, shift, (n, 1, 2), device=self.device)
                if straight > 0:  # real courses are laid out in a straight line: project the stakes onto their line
                    pick = (torch.rand(n, device=self.device) < straight)[:, None, None]
                    off = off + pick.float() * self._straight_offsets()[self.clip_id[env_ids]]
                self.stake_offset[env_ids] = off
            place_stakes(self._env, self.clip_id[env_ids], env_ids, self.stake_table, self.stake_offset[env_ids])
            from .stakes import reset_stake_state  # contact-based stake objective: new episode / clip
            reset_stake_state(self._env, env_ids, self.time_steps[env_ids] - self.motion.clip_start[self.clip_id[env_ids]])
        if self.bridge_table is not None:
            from .bridge import place_bridge
            # bridge placement randomisation: uniform +-XY horizontal shift of the whole bridge and +-Z
            # height error of the beam (posts xy only); applied consistently to the placed parts and the perception
            bxy = float(os.environ.get("HNU_BRIDGE_JITTER_XY", "0"))
            bz = float(os.environ.get("HNU_BRIDGE_JITTER_Z", "0"))
            bdz = float(os.environ.get("HNU_BRIDGE_DZ", "0"))  # fixed beam height error (evaluation: + higher, - lower)
            # asymmetric height range: the bridge teacher keeps 94-96 % for a beam up to 8 cm HIGHER but only
            # down to 3 cm LOWER (-5 cm: 79 %, -8 cm: 44 %) -> train within [ZLO, ZHI], e.g. [-0.03, +0.06]
            zlo = float(os.environ.get("HNU_BRIDGE_JITTER_ZLO", "0"))
            zhi = float(os.environ.get("HNU_BRIDGE_JITTER_ZHI", "0"))
            if bxy > 0 or bz > 0 or bdz != 0 or zlo != 0 or zhi != 0:
                off = torch.zeros(n, 3, device=self.device)
                if bxy > 0:
                    off[:, :2] = sample_uniform(-bxy, bxy, (n, 2), device=self.device)
                if bz > 0:
                    off[:, 2] = sample_uniform(-bz, bz, (n,), device=self.device)
                if zlo != 0 or zhi != 0:
                    off[:, 2] += sample_uniform(zlo, zhi, (n,), device=self.device)
                off[:, 2] += bdz
                self.bridge_offset[env_ids] = off
            from .bridge import bridge_part_offset
            po = bridge_part_offset(self.bridge_offset[env_ids], self.bridge_parts)  # [n, P, 3]
            # post spacing: every support post moves independently ALONG the beam axis by uniform
            # +-HNU_BRIDGE_POST_JITTER, or by a fixed spread HNU_BRIDGE_POST_SPREAD (evaluation: posts move away from
            # (+) / towards (-) the beam centre)
            pj = float(os.environ.get("HNU_BRIDGE_POST_JITTER", "0"))
            ps = float(os.environ.get("HNU_BRIDGE_POST_SPREAD", "0"))
            if pj > 0 or ps != 0:
                pl = self.bridge_parts.index("plank")
                tab = self.bridge_table[self.clip_id[env_ids]]  # [n, P, 7]
                q = tab[:, pl, 3:]
                ax = torch.stack([1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2), 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])], -1)
                ax = ax / ax.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [n, 2] beam direction
                from .bridge import is_post_part, post_owner
                is_post = torch.tensor([is_post_part(p) for p in self.bridge_parts], device=self.device)
                along = sample_uniform(-pj, pj, (n, len(self.bridge_parts)), device=self.device) if pj > 0 else \
                    torch.zeros(n, len(self.bridge_parts), device=self.device)
                if ps != 0:  # away from the beam centre along the axis
                    rel = ((tab[:, :, :2] - tab[:, pl:pl + 1, :2]) * ax[:, None]).sum(-1)
                    along = along + ps * torch.sign(rel)
                # a post's base (HNU_POST_BASE) moves with its post
                own = [self.bridge_parts.index(post_owner(p)) if post_owner(p) in self.bridge_parts else k
                       for k, p in enumerate(self.bridge_parts)]
                along = along[:, own] * is_post.float()
                po[..., :2] += along[..., None] * ax[:, None, :]
            # base height (HNU_POST_BASE): bars and braces of a bridge raised by U(HNU_BASE_DZ_LO, HNU_BASE_DZ_HI), e.g.
            # -0.005 .. 0.01 -> bar 2.5-4 cm high
            blo = float(os.environ.get("HNU_BASE_DZ_LO", "0"))
            bhi = float(os.environ.get("HNU_BASE_DZ_HI", "0"))
            if blo != 0 or bhi != 0:
                is_base = torch.tensor([p.startswith(("base", "brace")) for p in self.bridge_parts], device=self.device)
                po[..., 2] += sample_uniform(blo, bhi, (n, 1), device=self.device) * is_base.float()
            self.bridge_part_off[env_ids] = po
            place_bridge(self._env, self.clip_id[env_ids], env_ids, self.bridge_table, self.bridge_parts, part_off=po)
        self._pending_forward = True

    # -- start from rest (real robot): the robot does not start with the reference's frame-0 velocities
    def _pick_rest_starts(self, env_ids: torch.Tensor):
        """Envs (re)starting at frame 0 of their clip start from rest with probability HNU_REST_FRAC: either standing
        still in the frame-0 pose or (share HNU_STEP_FRAC) in a stepping-in-place state of a locomotion controller
        (HNU_STEP_STATES npz, tools/make_step_states.py); the reference clock is then held at frame 0 for
        U(HNU_HOLD) s (probability HNU_HOLD_P; reference velocities zero while held) before the motion starts.
        Returns the boolean masks (rest, stepping state) over env_ids."""
        n = len(env_ids)
        if not hasattr(self, "hold"):
            self.hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.hold0 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.hold[env_ids] = 0
        self.hold0[env_ids] = 0
        none = torch.zeros(n, dtype=torch.bool, device=self.device)
        rf = float(os.environ.get("HNU_REST_FRAC", "0"))
        if rf <= 0:
            return none, none
        at0 = self.time_steps[env_ids] == self.motion.clip_start[self.clip_id[env_ids]]
        rest = at0 & (torch.rand(n, device=self.device) < rf)
        step_st = rest & (torch.rand(n, device=self.device) < float(os.environ.get("HNU_STEP_FRAC", "0")))
        lo, hi = (float(x) for x in os.environ.get("HNU_HOLD", "0,0").split(","))
        dt = self._env.step_dt
        h = (sample_uniform(lo, hi, (n,), device=self.device) / dt).round().long()
        h = torch.where(rest & (torch.rand(n, device=self.device) < float(os.environ.get("HNU_HOLD_P", "1"))), h, 0)
        # rope release (the user's start procedure: hang on the rope in the frame-0 pose, set down, switch the
        # policy on, release the rope, stand, start): HNU_ROPE="frac_max,t_lo,t_hi,p" -> for a share p of the rest starts
        # in the frame-0 pose an upward pull on the torso of U(0, frac_max) x body weight (+-5 % horizontal), released
        # linearly over U(t_lo, t_hi) s; the clock stays held until 0.3-2 s after the release
        rope = os.environ.get("HNU_ROPE", "")
        if rope:
            if not hasattr(self, "rope_left"):
                self.rope_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                self.rope_T = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
                self.rope_f = torch.zeros(self.num_envs, 3, device=self.device)
            fmax, tlo, thi, p = (float(x) for x in rope.split(","))
            on = rest & ~step_st & (torch.rand(n, device=self.device) < p)
            T = (sample_uniform(tlo, thi, (n,), device=self.device) / dt).round().long().clamp(min=1)
            w = 33.3 * 9.81
            f = torch.zeros(n, 3, device=self.device)
            f[:, 2] = sample_uniform(0.0, fmax, (n,), device=self.device) * w
            f[:, :2] = sample_uniform(-0.05, 0.05, (n, 2), device=self.device) * w
            self.rope_left[env_ids] = torch.where(on, T, 0)
            self.rope_T[env_ids] = T
            self.rope_f[env_ids] = f * on[:, None].float()
            after = (sample_uniform(0.3, 2.0, (n,), device=self.device) / dt).round().long()
            h = torch.where(on, torch.maximum(h, T + after), h)
            self._rope_apply()
        self.hold[env_ids] = h
        self.hold0[env_ids] = h
        return rest, step_st

    def _rope_apply(self):
        """Write the current rope pull (linear release) to the torso of every env (zero when no rope)."""
        robot = self._env.scene[self.cfg.entity_name]
        if not hasattr(self, "_torso_id"):
            self._torso_id = robot.body_names.index("torso_link")
        frac = (self.rope_left.float() / self.rope_T.float()).clamp(0, 1)[:, None]
        force = self.rope_f * frac
        L = float(os.environ.get("HNU_ROPE_LEN", "0"))
        if L > 0:
            # (real G1: the rope holds the robot like a pendulum, the simulated rope pulled only vertically --
            # sim robots drifted 0.4 m while standing, the real one could not): horizontal restoring force of a rope of
            # length L hanging from above the start, F_xy = -F_z / L * (torso_xy - anchor_xy)
            if not hasattr(self, "rope_anchor"):
                self.rope_anchor = torch.zeros(self.num_envs, 2, device=self.device)
            p = robot.data.body_link_pos_w[:, self._torso_id, :2]
            fresh = self._env.episode_length_buf <= 1
            self.rope_anchor = torch.where(fresh[:, None], p, self.rope_anchor)
            force = force.clone()
            force[:, :2] = force[:, :2] - (force[:, 2:3] / L) * (p - self.rope_anchor)
        force = force[:, None, :]
        robot.write_external_wrench_to_sim(force, torch.zeros_like(force), body_ids=[self._torso_id])

    def _step_states(self):
        if getattr(self, "_steps", None) is None:
            d = np.load(os.environ["HNU_STEP_STATES"])
            names = [str(x) for x in d["joint_names"]]
            robot = self._env.scene[self.cfg.entity_name]
            idx = [names.index(j) for j in robot.joint_names]
            t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
            self._steps = dict(q=t(d["q"][:, idx]), dq=t(d["dq"][:, idx]), z=t(d["z"]), rp=t(d["rp_quat"]),
                               v=t(d["v_yaw"]), w=t(d["w_body"]),
                               upper=torch.tensor([not j.endswith(("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint",
                                                                   "knee_joint", "ankle_pitch_joint", "ankle_roll_joint"))
                                                   for j in robot.joint_names], device=self.device))
        return self._steps

    def _rest_state(self, env_ids, rest, step_st, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel):
        """Overwrite the reset state of the rest starts (in place): start offset HNU_REST_XY (m, per axis) and heading
        error HNU_REST_YAW (deg) w.r.t. the reference start; frame-0 pose or stepping state; small residual velocities."""
        from mjlab.utils.lab_api.math import quat_apply
        dev = self.device
        n = len(env_ids)
        xy = float(os.environ.get("HNU_REST_XY", "0.10"))
        dyaw = np.radians(float(os.environ.get("HNU_REST_YAW", "15")))
        ref_q = self.body_quat_w[env_ids, 0]
        yaw_ref = torch.atan2(2 * (ref_q[:, 0] * ref_q[:, 3] + ref_q[:, 1] * ref_q[:, 2]),
                              1 - 2 * (ref_q[:, 2] ** 2 + ref_q[:, 3] ** 2))
        yaw = yaw_ref + sample_uniform(-dyaw, dyaw, (n,), device=dev)
        zero = torch.zeros(n, device=dev)
        qyaw = quat_from_euler_xyz(zero, zero, yaw)
        ref_pos = self.body_pos_w[env_ids, 0]
        new_pos = ref_pos.clone()
        new_pos[:, :2] += sample_uniform(-xy, xy, (n, 2), device=dev)
        new_pos[:, 2] += 0.01
        # frame-0 pose, re-oriented to the perturbed heading (roll / pitch of the reference kept)
        new_ori = quat_mul(quat_from_euler_xyz(zero, zero, yaw - yaw_ref), ref_q)
        rp = np.radians(float(os.environ.get("HNU_REST_RP", "0")))  # set down slightly tilted (rope release), deg
        if rp > 0:
            new_ori = quat_mul(quat_from_euler_xyz(sample_uniform(-rp, rp, (n,), device=dev),
                                                   sample_uniform(-rp, rp, (n,), device=dev), zero), new_ori)
        lin = sample_uniform(-0.05, 0.05, (n, 3), device=dev)
        ang = sample_uniform(-0.1, 0.1, (n, 3), device=dev)
        jp = self.joint_pos[env_ids] + sample_uniform(-0.05, 0.05, joint_pos.shape, device=dev)
        jv = torch.zeros_like(joint_vel)
        if step_st.any():
            S = self._step_states()
            k = torch.randint(0, S["q"].shape[0], (n,), device=dev)
            sel = step_st[:, None]
            # legs (and waist) from the stepping state; arms: the state's arms or the frame-0 arms (half / half), + noise
            ref_arms = torch.rand(n, 1, device=dev) < 0.5
            up = S["upper"][None]
            q_st = torch.where(up & ref_arms, self.joint_pos[env_ids], S["q"][k])
            q_st = q_st + torch.where(up, sample_uniform(-0.15, 0.15, joint_pos.shape, device=dev), 0.0)
            jp = torch.where(sel, q_st, jp)
            jv = torch.where(sel, S["dq"][k], jv)
            o_st = quat_mul(qyaw, S["rp"][k])
            new_ori = torch.where(sel, o_st, new_ori)
            new_pos[:, 2] = torch.where(step_st, self._env.scene.env_origins[env_ids, 2] + S["z"][k] + 0.005, new_pos[:, 2])
            lin = torch.where(sel, quat_apply(qyaw, S["v"][k]), lin)
            ang = torch.where(sel, quat_apply(o_st, S["w"][k]), ang)
        m = rest[:, None]
        root_pos[:] = torch.where(m, new_pos, root_pos)
        root_ori[:] = torch.where(m, new_ori, root_ori)
        root_lin_vel[:] = torch.where(m, lin, root_lin_vel)
        root_ang_vel[:] = torch.where(m, ang, root_ang_vel)
        joint_pos[:] = torch.where(m, jp, joint_pos)
        joint_vel[:] = torch.where(m, jv, joint_vel)

    def _moving(self) -> torch.Tensor | None:
        """[num_envs, 1] 0 while the reference clock is held (reference velocities zero), else 1; None without holds."""
        h = getattr(self, "hold", None)
        if h is None:
            return None
        return (h == 0).float()[:, None]

    @property
    def joint_vel(self) -> torch.Tensor:
        v = self.motion.joint_vel[self.time_steps]
        mv = self._moving()
        return v if mv is None else v * mv

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        v = self.motion.body_lin_vel_w[self.time_steps]
        mv = self._moving()
        return v if mv is None else v * mv[:, :, None]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        v = self.motion.body_ang_vel_w[self.time_steps]
        mv = self._moving()
        return v if mv is None else v * mv[:, :, None]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        v = self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]
        mv = self._moving()
        return v if mv is None else v * mv

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        v = self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]
        mv = self._moving()
        return v if mv is None else v * mv

    def _update_command(self, env_ids: torch.Tensor | None = None):
        if getattr(self, "rope_left", None) is not None and env_ids is None:  # rope release (HNU_ROPE)
            if bool((self.rope_left > 0).any()) or getattr(self, "_rope_on", True):
                self.rope_left = (self.rope_left - 1).clamp(min=0)
                self._rope_apply()
                self._rope_on = bool((self.rope_left > 0).any())
        h = getattr(self, "hold", None)
        if h is not None:  # held clocks (start from rest) do not advance
            ids = slice(None) if env_ids is None else env_ids
            adv = (h[ids] == 0).long()
            h[ids] = (h[ids] - 1).clamp(min=0)
            self.time_steps[ids] += adv
        elif env_ids is None:
            self.time_steps += 1
        else:
            self.time_steps[env_ids] += 1
        end = self.motion.clip_end[self.clip_id]
        wrap_ids = torch.where(self.time_steps >= end)[0]
        if wrap_ids.numel() > 0:
            self.time_steps[wrap_ids] = end[wrap_ids] - 1  # stay valid until resampled
            self._resample_command(wrap_ids)
        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()
        self.update_relative_body_poses()
        if env_ids is None and self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (self.cfg.adaptive_alpha * self._current_bin_failed
                                     + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count)
            self._current_bin_failed.zero_()

    # reference contact labels for the current frame (used by optional rewards / observations)
    @property
    def ref_foot_contact(self) -> torch.Tensor:
        return self.motion.foot_contact[self.time_steps]

    @property
    def task_id(self) -> torch.Tensor:
        return self.motion.clip_task[self.clip_id]

    def z_skill(self, offset: int = 0) -> torch.Tensor:
        m = self.motion
        f = torch.minimum(self.time_steps + offset, m.clip_end[self.clip_id] - 1)
        return m.z[f]

    @property
    def clip_progress(self) -> torch.Tensor:
        m = self.motion
        return ((self.time_steps - m.clip_start[self.clip_id]).float() / m.clip_len[self.clip_id].float()).unsqueeze(-1)


@dataclass(kw_only=True)
class MultiMotionCommandCfg(MotionCommandCfg):
    fixed_clip: int = -1
    start_clip_mode: str = "random"  # "random" | "env_mod" (env i tracks clip i % num_clips)
    latent_dir: str = ""  # optional per-frame z_skill tables for students
    use_stakes: bool = False  # place per-clip stakes (mocap entities stake0..4) at every resample
    use_bridge: bool = False  # place the per-clip DuMuQiao bridge boxes (mocap entities bridge_*) at every resample

    def build(self, env) -> MultiMotionCommand:
        return MultiMotionCommand(self, env)
