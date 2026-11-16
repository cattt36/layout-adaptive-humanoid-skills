"""export a BLIND tracker (teacher fine-tuned with HNU_ACTOR_BLIND=1, tools/rtG_bm.sh) for the robot:
policy.onnx (observation normaliser folded in) + policy_meta.json (runtime/hnu_deploy.py TrackerPolicy) + scene.mjb.
  python tools/export_tracker.py <model_*.pt> <task> <base_export_dir> <out_dir>
base_export_dir: a student export of the same robot (joint order, PD gains, action scale / offset, scene); the waist
roll / pitch action scale is set to 0 (locked, as in training) unless HNU_EXPORT_WAIST_FREE=1.  Also writes parity.txt: ONNX vs torch on 2000
observations sampled around the normaliser statistics."""
import json
import os
import shutil
import sys

import numpy as np
import torch

OBS_BASE = 58 + 3 + 6 + 3 + 3 + 29 + 29 + 29  # command, anchor pos, anchor ori, lin vel, ang vel, joints, joint vel, action


def main():
    ck_path, task, base, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    sd = torch.load(ck_path, map_location="cpu", weights_only=False)["actor_state_dict"]
    mean, std = sd["obs_normalizer._mean"].flatten().float(), sd["obs_normalizer._std"].flatten().float()
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    layers = []
    for n, i in enumerate(idx):
        w, b = sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if n < len(idx) - 1:
            layers.append(torch.nn.ELU())
    mlp = torch.nn.Sequential(*layers).eval()

    class Net(torch.nn.Module):
        def forward(self, obs):
            return mlp((obs - mean) / (std + 1e-2))
    net = Net().eval()
    D = mean.numel()
    n_perc = D - OBS_BASE
    assert n_perc in (0, 7, 10), f"unexpected observation size {D}"
    torch.onnx.export(net, torch.zeros(1, D), os.path.join(out, "policy.onnx"), input_names=["obs"],
                      output_names=["action"], opset_version=17)
    meta = json.load(open(os.path.join(base, "policy_meta.json")))
    # HNU_EXPORT_WAIST_FREE=1 -> waist roll / pitch keep their action scale (tracker trained waist free)
    free = os.environ.get("HNU_EXPORT_WAIST_FREE", "0") == "1"
    sc = list(meta["action_scale"])
    for i, j in enumerate(meta["joint_names"]):
        if j in ("waist_roll_joint", "waist_pitch_joint") and not free:
            sc[i] = 0.0
    meta.update(action_scale=sc, act_dim=len(sc), obs_dim=D, policy_type="tracker", waist_lock=[] if free else ["waist_roll_joint", "waist_pitch_joint"],
                tracker=dict(task=task, anchor_body="torso_link", perception_dim=n_perc, checkpoint=ck_path,
                             obs="ref joint pos 29, ref joint vel 29 (0 while held), 0 x3, anchor ori 6, 0 x3, gyro 3, "
                                 "joint pos - default 29, joint vel 29, last action 29, 0 x perception"),
                obstacle_input="none", lin_vel_input="zero")
    meta.pop("obs_layout", None)
    json.dump(meta, open(os.path.join(out, "policy_meta.json"), "w"), indent=1)
    shutil.copy(os.path.join(base, "scene.mjb"), os.path.join(out, "scene.mjb"))
    # parity reference for the deploy venv (tools/tracker_parity.py; no onnxruntime in the training venv)
    x = (mean + std * torch.randn(2000, D)).float()
    with torch.no_grad():
        np.savez(os.path.join(out, "parity_ref.npz"), obs=x.numpy(), act=net(x).numpy())
    print(out, D, n_perc)


if __name__ == "__main__":
    main()
