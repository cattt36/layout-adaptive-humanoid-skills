# Deployment_mocap (new G1, waist free, standard start, obstacles measured by motion capture)

**This bundle is for the newly purchased G1**: the waist works normally and is not locked. The robot with the faulty
waist needs the separate faulty-waist bundle.

All changes concern deployment only. The models and results of the paper are unchanged.

## 0. Changes compared with the previous deployment bundle

| # | Change | Details |
|---|---|---|
| 1 | **Standard start** | Every layout uses the same start: the robot stands at the origin in a normal stance (feet parallel, 24 cm apart) and **faces +X**. Placing the robot at a skill-specific position and heading is no longer needed. After "Start", the robot takes one step onto the start foothold of the skill and then blends into the skill over about 2 s. The small shuffle of D1 and the open-arm phase of S2 are removed |
| 2 | **New models (waist free)** | Retrained on the standard-start data: bridge `model_bridge` = rtJb (no body velocity input), stakes `model_stakes` = rtJs (body velocity measured by motion capture). Waist roll / pitch are controlled by the policy as usual |
| 3 | **Standard-start planner** | Every candidate skill keeps its own "stand + step-in" code, and only the motion part is aligned with the current layout. With the old method the robot walked 45 cm forward on D1 by itself after the rope release, with the new one 13 cm (Sec. 7). The plans of all candidates are precomputed in `planned_layouts/<layout>/cands/` |
| 4 | **All point coordinates updated** | Frame: origin = robot start, +X = robot heading. See Sec. 2. Every layout comes with a check code and a top view |
| 5 | **Live obstacle measurement by motion capture: required** | At the takeover the real positions of the posts / stakes are measured with rigid bodies. If one is not visible, moving or outside the trained range, the start is **refused** (Sec. 4) |
| 6 | **Placement check with motion capture** | `place_live.py` reads the motion-capture coordinates, compares them one by one with the reference coordinates of Sec. 2 and shows the check code on screen for comparison with this README (Sec. 2.2) |
| 7 | **Preview** | Uses real-robot conditions: pendulum rope, placement errors, 0–40 ms latency, at least 12 rollouts per candidate. Candidates whose standing drift is too large are not selected (bridge > 20 cm, stakes > 10 cm) |
| 8 | **Health check tool** | `robot_health.py`: per joint it checks motor output, torque saturation, joint stops, temperature and left-right symmetry. Run it before testing and after every fall (Sec. 5) |
| 9 | **Start thresholds** | Back to **5 cm / 10°** (the trained range). They were once relaxed to 20 cm / 20°. Do not use that again |

**Do not use your `chain/planner.py`**: it plans on the original clock and does not match the standard start. For the
retrieval chain use `code/chain/pipeline.py` of this bundle (Sec. 7.2).

---

## 1. Overview

1. **Health check** (Sec. 5, about 10 min): hang the robot on the rope, let the policy take over and hold frame 0 for 5–10 s without releasing the rope, then run `robot_health.py`. Stop if there are FLAGS.
2. **Standing only** (3 times): takeover → release the rope over about 10 s → **no hand support**, stand for 2–4 s → tighten the rope again. The robot must stand steadily all 3 times and drift less than 5 cm.
3. **Bridge D2 → N3 → D1**, 3 runs per layout (Sec. 6, with the pass and stop rules).
4. **Three-step stakes S2 → S1 → N1**, then **five-step stakes T1 → T2 → N2**.
5. Keep the trace / data / obs logs and the video of every run and note the times.

---

## 2. Point coordinates (important)

### 2.1 Frame

- **Layout frame = motion-capture frame + `aligned_shift_xy`** (`config.yaml`), a pure translation without rotation. The +X of the layout is therefore the +X of the motion capture.
- **Origin = robot start**: the floor projection of the pelvis in a normal stance. **+X = initial robot heading**, Y positive to the left, units in metres.
- **Robot**: stands at (0, 0) in every layout and faces motion-capture +X. Left foot at about (0, +0.118), right foot at about (0, −0.118). Tolerance **±5 cm, ±10°** (trained range), measured with the motion-capture pelvis rigid body.
- **Post coordinates = post axis** (centre of the top face). Posts are numbered 1–4 in the order the robot passes them.
- **Stake coordinates = centre of the stake top face**. Stakes are 40 cm in diameter and 10 cm high. They are numbered in stepping order, and the "Foot" column is the foot that steps on them.
- Post base bars use the shortened dimensions: the two inner posts 15 cm per side (30 cm in total), the two end posts 20 cm per side (40 cm in total). The lower edge of the beam is about 111 cm above the floor.
- **D1 and D2 / N3 have different post spacings** (D1: 1.156 / 1.442 / 0.871 m, D2 / N3: 1.222 / 1.524 / 0.920 m). Rebuild the bridge when switching layouts.
- The bridge is no longer parallel to the X axis. For example, the line through the posts of D1 makes +20.4° with +X, so the bridge stands at an angle while the robot always faces +X.

All coordinates follow `planned_layouts/<layout>/layout_aligned.json`. The tables below are generated from the same
files. In addition:
- `point_coordinates.csv`: all points, for printing or import.
- `figures/<layout>.png`: top views with the robot, the posts / stakes and their coordinates.

### 2.2 Placement check with motion capture (for every layout)

```
python code/runtime/place_live.py planned_layouts/<layout> --config <config.yaml of the hardware layer>
```

- The screen refreshes about every 0.25 s, **reads the measured motion-capture coordinates and compares them one by one with the reference coordinates**. For every post / stake it shows the reference (x, y), the measured (x, y), the errors dx / dy, and how many centimetres it still has to move toward +X / −X / +Y / −Y. Colours: green ≤ 3 cm, yellow ≤ 5 cm, red beyond.
- **The third line on the screen shows the check code of the reference coordinates**. It must match the check code of the layout in the tables below. If it does not, the program is not reading the files of this bundle or the files were modified. Find out why before placing anything.
- Below that it shows the mean overall offset, the neighbour spacings (reference vs measured) and whether the placement lies within the trained range:
  - posts: overall ±5 cm, single post along the beam ±10 cm, off the beam line ≤ 3 cm.
  - stakes: overall ±5 cm, single stake along the walking direction ±3 cm, sideways ±6 cm.
- Ctrl+C ends it with a verdict. The record is saved to `planned_layouts/<layout>/placement_records/` (with the check code). Exit code 0 means the run can start.
- **Goal**: all green. Yellow may start but should preferably be adjusted. Red must be placed again.
- After placing, **do not** add `--write-plan`: the takeover measures the obstacles automatically (Sec. 4). `--write-plan` is only an emergency fallback when the live measurement is not connected.
- The first time, add `--seconds 3` to confirm that all targets are visible and the readings are plausible. Without motion capture, `--source fake` shows the interface.
- The rigid bodies are named `post1`–`post4` and `stake1`–`stake5` by default. They can also be set in `mocap.obstacle_bodies` of `config.yaml`.

**How to attach the rigid bodies** (also needed for Sec. 4): make one rigid body per post / stake and place markers
**evenly around** the post (or the stake edge), at least 3, preferably 4. The centre of the rigid body then lies on the
axis / centre, and the pivot of the rigid body is set to the centre of the top face. Markers on one side only shift the
centre by a radius (about 4 cm for a post).

**Robot start**: use your start check (thresholds 5 cm position / 10° heading) or check by hand:
`python code/runtime/check_placement.py planned_layouts/<layout> --start <measured x>,<measured y>,<measured heading in deg>`.

### 2.3 Points per layout

#### D1 (bridge, `D1_CrouchedTraverse_024`, check code **44ae45**)

Robot: pelvis projection at (0, 0), facing +X. Left foot (+0.000, +0.119), right foot (-0.001, -0.119). Figure: `figures/D1.png`

| Post | x (m) | y (m) | Top height (m) | To next (m) |
|---|---|---|---|---|
| 1 | 1.074 | 1.359 | 1.159 | 1.156 |
| 2 | 2.158 | 1.762 | 1.129 | 1.442 |
| 3 | 3.510 | 2.264 | 1.091 | 0.871 |
| 4 | 4.326 | 2.568 | 1.068 | - |

Beam: centre (2.624, 1.935), length 3.74 m, centre height 1.174 m. Line through posts 1→4 at **+20.4°** to +X. Start to post 1: 1.73 m.

#### D2 (bridge, `D2_CrouchedTraverse_008`, check code **ba4cf8**)

Robot: pelvis projection at (0, 0), facing +X. Left foot (-0.000, +0.118), right foot (-0.000, -0.118). Figure: `figures/D2.png`

| Post | x (m) | y (m) | Top height (m) | To next (m) |
|---|---|---|---|---|
| 1 | 1.425 | 1.372 | 1.178 | 1.222 |
| 2 | 2.646 | 1.337 | 1.146 | 1.524 |
| 3 | 4.169 | 1.294 | 1.106 | 0.920 |
| 4 | 5.089 | 1.268 | 1.082 | - |

Beam: centre (3.171, 1.322), length 3.95 m, centre height 1.193 m. Line through posts 1→4 at **-1.6°** to +X. Start to post 1: 1.98 m.

#### N3 (bridge, `N3_CrouchedTraverse_008_shift_left0.10`, check code **267aa6**)

Robot: pelvis projection at (0, 0), facing +X. Left foot (-0.000, +0.118), right foot (-0.000, -0.118). Figure: `figures/N3.png`

| Post | x (m) | y (m) | Top height (m) | To next (m) |
|---|---|---|---|---|
| 1 | 1.415 | 1.272 | 1.178 | 1.221 |
| 2 | 2.636 | 1.238 | 1.146 | 1.524 |
| 3 | 4.159 | 1.194 | 1.106 | 0.920 |
| 4 | 5.079 | 1.168 | 1.082 | - |

Beam: centre (3.161, 1.223), length 3.95 m, centre height 1.193 m. Line through posts 1→4 at **-1.6°** to +X. Start to post 1: 1.90 m.

#### S1 (three-step stakes, `S1_ThreeStepPoleRun_011`, check code **72a6cd**)

Robot: pelvis projection at (0, 0), facing +X. Left foot (-0.001, +0.118), right foot (+0.000, -0.118). Figure: `figures/S1.png`

| Stake | Foot | x (m) | y (m) | To next (m) |
|---|---|---|---|---|
| 1 | L | 3.014 | 0.211 | 1.402 |
| 2 | R | 4.416 | 0.199 | 1.447 |
| 3 | L | 5.849 | 0.402 | - |

Stakes: diameter 0.40 m, height 0.10 m, coordinates at the centre of the top face. Start to stake 1: 3.02 m.

#### S2 (three-step stakes, `S2_ThreeStepPoleRun_012`, check code **83454b**)

Robot: pelvis projection at (0, 0), facing +X. Left foot (-0.000, +0.118), right foot (-0.000, -0.119). Figure: `figures/S2.png`

| Stake | Foot | x (m) | y (m) | To next (m) |
|---|---|---|---|---|
| 1 | R | 3.410 | 0.120 | 1.334 |
| 2 | L | 4.735 | 0.277 | 1.435 |
| 3 | R | 6.161 | 0.119 | - |

Stakes: diameter 0.40 m, height 0.10 m, coordinates at the centre of the top face. Start to stake 1: 3.41 m.

#### N1 (three-step stakes, `N1_ThreeStepPoleRun_011_spacing1.05`, check code **0cc244**)

Robot: pelvis projection at (0, 0), facing +X. Left foot (-0.001, +0.118), right foot (+0.000, -0.118). Figure: `figures/N1.png`

| Stake | Foot | x (m) | y (m) | To next (m) |
|---|---|---|---|---|
| 1 | L | 3.014 | 0.211 | 1.472 |
| 2 | R | 4.486 | 0.198 | 1.519 |
| 3 | L | 5.990 | 0.411 | - |

Stakes: diameter 0.40 m, height 0.10 m, coordinates at the centre of the top face. Start to stake 1: 3.02 m.

#### T1 (five-step stakes, `T1_FiveStepPoleRun_013`, check code **46fe56**)

Robot: pelvis projection at (0, 0), facing +X. Left foot (-0.000, +0.118), right foot (+0.000, -0.119). Figure: `figures/T1.png`

| Stake | Foot | x (m) | y (m) | To next (m) |
|---|---|---|---|---|
| 1 | L | 2.217 | 0.516 | 1.018 |
| 2 | R | 2.983 | -0.154 | 1.121 |
| 3 | L | 3.478 | 0.852 | 1.016 |
| 4 | R | 4.287 | 0.237 | 1.073 |
| 5 | L | 4.702 | 1.227 | - |

Stakes: diameter 0.40 m, height 0.10 m, coordinates at the centre of the top face. Start to stake 1: 2.28 m.

#### T2 (five-step stakes, `T2_FiveStepPoleRun_019`, check code **8aaa2d**)

Robot: pelvis projection at (0, 0), facing +X. Left foot (-0.001, +0.118), right foot (+0.000, -0.118). Figure: `figures/T2.png`

| Stake | Foot | x (m) | y (m) | To next (m) |
|---|---|---|---|---|
| 1 | L | 2.666 | 1.519 | 1.037 |
| 2 | R | 3.556 | 0.987 | 1.091 |
| 3 | L | 3.763 | 2.058 | 1.086 |
| 4 | R | 4.741 | 1.587 | 1.000 |
| 5 | L | 4.867 | 2.579 | - |

Stakes: diameter 0.40 m, height 0.10 m, coordinates at the centre of the top face. Start to stake 1: 3.07 m.

#### N2 (five-step stakes, `N2_FiveStepPoleRun_019_spacing0.95`, check code **f55fc3**)

Robot: pelvis projection at (0, 0), facing +X. Left foot (-0.001, +0.118), right foot (+0.000, -0.118). Figure: `figures/N2.png`

| Stake | Foot | x (m) | y (m) | To next (m) |
|---|---|---|---|---|
| 1 | L | 2.666 | 1.519 | 0.986 |
| 2 | R | 3.512 | 1.013 | 1.037 |
| 3 | L | 3.708 | 2.031 | 1.032 |
| 4 | R | 4.638 | 1.583 | 0.950 |
| 5 | L | 4.757 | 2.526 | - |

Stakes: diameter 0.40 m, height 0.10 m, coordinates at the centre of the top face. Start to stake 1: 3.07 m.


---

## 3. Start procedure (standard start)

| Step | Action | How | Note |
|---|---|---|---|
| 1 | **Rope** | The robot hangs on the rope, feet off the floor or just touching it | Rope only, **no hand support** |
| 2 | **Posture** | PD interpolates the joints to frame 0 within 2–3 s, i.e. `joint_pos[0]` of `planned_layouts/<layout>/*_motion.npz`. **Frame 0 of every layout is the normal stance** with the arms hanging naturally | Done while hanging |
| 3 | **Position** | Pelvis projection on the origin, **facing motion-capture +X** | Within ±5 cm and ±10°, measured with the motion-capture pelvis |
| 4 | **Touchdown** | Lower slowly until both feet are flat and carry weight | Do not take over while the feet are in the air |
| 5 | **Takeover** (command ①) | Load the model of the task (Sec. 7.1), switch to the policy, frame index fixed at 0 | Do not go back to PD afterwards |
| 6 | **Rope release** | Release evenly over about **10 s** until the rope carries no load and only protects | Trained range: at most 90 % of the body weight held, released within 1–12 s, with a 2 m pendulum pull-back. No sideways pull, no twisting |
| 7 | **Stand** | Wait another **1–2 s** after the release, **no hand support** | In training the skill starts 0.3–2 s after the release. Do not stand too long |
| 8 | **Start** (command ②) | Frame index starts at 1 and increases by 1 every 20 ms | The robot does the rest by itself: weight shift to the stance foot 0.5 s → one step 0.7 s → transition 0.6 s → slow blend 2 s → skill |
| 9 | End | After the last frame switch back to standing or protection mode | |

**Why no hand support**: while a person held the robot, the policy kept fighting the hand (the left ankle produced
about 24 Nm all the time). When the person let go, this force was released and the robot lunged forward. Training only
simulates the rope, not a hand.
**If the robot cannot stand without support after the rope release**, something is wrong with the robot. Go back to the
health check of Sec. 5.

**In simulation the robot never fell in the standing phase**: in 2304 acceptance rollouts of the new models there were
0 falls during the 12 s standing / rope-release phase.

---

## 4. Live obstacle measurement by motion capture (required)

At the takeover the real position of every post / stake is measured with motion-capture rigid bodies, replaces the
position of the plan file and is passed to the policy. Placement errors and motion-capture frame errors therefore do not
reach the policy.

**Integration** (robot side, `hnu_pipeline.py`):
1. Replace the method of the same name with `refresh_obstacles()` from `code/robot_side/refresh_obstacles.py`. It depends on `code/robot_side/obstacles_rigid.py` and `code/runtime/obstacles_live.py`.
2. In the takeover branch, **decide on the start from its return value**:
   ```python
   if not self.refresh_obstacles():
       print("[obstacle measurement] failed, no start (check the rigid bodies / occlusion, then press Start again)", flush=True)
       return
   ```
3. `config.yaml`:
   ```yaml
   mocap:
     obstacle_bodies: {posts: [post1, post2, post3, post4], stakes: [stake1, stake2, stake3, stake4, stake5]}
     obstacles_live: {enabled: true, required: true, seconds: 1.0, max_jitter_m: 0.01, marker_fallback: true}
   ```

**What it does**:
- Every rigid body is averaged over the last 1 s. A jitter > 1 cm counts as moving or badly tracked. Posts without a rigid body fall back to the old scattered-marker method (markers on one side, axis estimated 4 cm behind them, poor accuracy).
- Checks the trained range: posts overall 5 cm, along the beam 10 cm, sideways 3 cm. Stakes overall 5 cm, along the walking direction 3 cm, sideways 6 cm.
- With `required: true`, **the start is refused if any obstacle is not visible, moving or out of range**. In one earlier run an old file was silently reused while post 1 had already been knocked 11 cm forward.
- Every time it rewrites the layout from the planned version (`*_sheet.json`, else the original file), so measurements of the previous run do not accumulate.
- The measured positions are stored in `obstacles_measured` of the run result.

**After the robot hits or falls onto a post / stake**: first check whether it moved, then check again with
`place_live.py`. The next takeover measures again automatically.

---

## 5. Health check (for every robot, before testing each day and after every fall)

```
python code/robot_side/robot_health.py <data_*.csv> [more csv]
python code/robot_side/waist_check.py <data_*.csv>          # waist only
```

It reads the data_*.csv that the robot saves on every run (columns q_des / q / dq / tau_est / kp / kd / tau_ff / temp
are needed, earlier logs already have them). It separates the "rope phase" from the "policy phase" and reports per
joint:

| Column | Meaning | Normal |
|---|---|---|
| ratio | measured torque / commanded torque | 0.8–1.25 |
| sat% | fraction of time the commanded torque is at the motor limit | ≈ 0 |
| stop% | fraction of time the joint is on its mechanical stop | ≈ 0 |
| err / errmax | target minus measured angle (median / maximum) | < 0.05 rad in the rope phase |
| temp | maximum motor temperature | < 70 °C |

FLAGS are listed at the end. **How to decide**:
- **No FLAGS**: continue.
- **Waist sagging or on its stop**: do not use this bundle, use the faulty-waist bundle.
- **Abnormal leg motor output, saturated torque, left-right asymmetry**: do not test. Send the logs back or contact the manufacturer.
- **Temperature > 70 °C**: stop and let it cool down.

Procedure: hang on the rope → take over and hold frame 0, stand for 5–10 s without releasing the rope and without hand
support → exit and run the commands above on the data CSV of this run.
Tested on an earlier log, it reported "waist pitch on its stop 10 % of the time, mean sag 0.29 rad".

---

## 6. Test order, pass rules, stop rules

| Order | Layout | Default skill | Per layout |
|---|---|---|---|
| 0 | Health check + standing only | - | Stable all 3 times, drift < 5 cm |
| 1 | **D2** | CrouchedTraverse_005 | 3 runs |
| 2 | **N3** (whole bridge shifted 10 cm to the left) | old plan (re-timed) | 3 runs |
| 3 | **D1** | CrouchedTraverse_028 | 3 runs. The robot drifts about 13 cm forward while standing (simulation), this is expected |
| 4 | Three-step stakes **S2 → S1 → N1** | see Sec. 8 | 3 runs each |
| 5 | Five-step stakes **T1 → T2 → N2** | see Sec. 8 | 3 runs each (the five-step stakes are the weakest in simulation) |

- **Move to the next layout**: at least 1 success in 3 runs, or all failed but every run passed the second post / the second stake.
- **Stop**: two failures in the same way at the same place. Send the logs and videos back and do not continue.
- **After every failure**: check whether the obstacles moved, check the robot (joint noises, ankles) and run `robot_health.py` once.

---

## 7. Models and skills

### 7.1 Models (`models.json`)

| Task | Folder | Training | Body linear velocity input | Waist |
|---|---|---|---|---|
| Bridge (D1 / D2 / N3) | `model_bridge/` | rtJb iteration 2000 | none (`lin_vel_input` = "zero") | controlled normally |
| Three-step and five-step stakes | `model_stakes/` | rtJs iteration 2000 | **motion-capture estimate** (`lin_vel_input` = "mocap", 60 ms fit, same as the previous version) | controlled normally |

- The interface is the same as in the previous version (model chosen per task, `pol.reset(ctx, state, 0)` at command ①, then `pol.step(state, k)` every 20 ms).
- **The waist is no longer locked**: waist roll / pitch have an `action_scale` of 0.439 and the policy outputs the waist targets. Do not force the waist targets to 0 in the deployment software.
- Both models use the same `scene.mjb`. The ONNX vs training-model consistency checks (parity) passed for both.
- File checksums (md5): `model_bridge/policy.onnx` a993904710d96435f9a90600eeee2d58, `model_stakes/policy.onnx` 9e04f18c976675d055bd76a42b698831, `scene.mjb` 96fbf4aaab1beccbb7d8d3e106d4ff8a.

### 7.2 Which skill to use

**Use the z.npy / stakes.json in the root of `planned_layouts/<layout>/` directly**: these are the default skills of
the acceptance in Sec. 8 and the safest choice.

For the full "instruction → router → retrieval → preview" chain:
- Use `code/chain/pipeline.py` (replacing your `chain/pipeline.py`), **do not use `chain/planner.py`**. The new flow reads the precomputed standard-start plans in `planned_layouts/<layout>/cands/<candidate>/`.
- Candidates without a standard-start plan are skipped and printed. These clips have no suitable double-support frame at their beginning, e.g. CrouchedTraverse_017–023 and 026.
- **Selection rule**: among the candidates with a median standing drift ≤ 20 cm (bridge) / 10 cm (stakes), choose the one with the most preview successes. Refuse to execute if it has fewer than 2.
- Preview conditions: start from rest + pendulum rope (80 % of the body weight, released over 10 s, rope length 2 m) + 12 s hold + start ±5 cm / ±10° + 0–40 ms latency + real-robot errors, for the stakes also placement errors, ≥ 12 rollouts per candidate. Waist locking follows the model (not locked in this bundle). Wrong conditions raise an error.
- Tested on the server (D1, 5 candidates, 12 rollouts each, about 19 s with 100 processes): 017 skipped. 009 succeeded 10/12 but drifted 123 cm while standing and was excluded. 029 succeeded 12/12 with 25 cm drift and was excluded. **028 succeeded 12/12 with 17 cm drift and was selected**. 005 succeeded 5/12. The robot computer has fewer cores and will take longer (an estimated 1–3 min).

---

## 8. Simulation acceptance (CPU MuJoCo, success criterion v2, 64 rollouts per cell)

Two condition sets, both with: rope holding 80 % of the body weight released over 10 s (2 m pendulum), frame 0 held for
12 s, start ±5 cm / ±10°, body tilt ±3°, 0–40 ms latency, contact / motor / mass randomisation, motion-capture heading
offset ±3° and height offset ±2 cm. The velocity input of the stakes model is estimated by differencing the motion
capture.
- **RP10X**: the conditions above.
- **RP10XS**: additionally an overall obstacle offset of 3 cm, perception noise of 2 cm and ±3 cm per stake.

"Standing drift" is the horizontal pelvis displacement during the rope-release and standing phase (frame fixed at 0).

| Layout | Default skill | RP10X | RP10XS | Standing drift (median / worst 10 %) | Falls while standing |
|---|---|---|---|---|---|
| D1 | CrouchedTraverse_028 | 97% | 95% | 13 / 22 cm | 0 |
| D2 | CrouchedTraverse_005 | 88% | 84% | 7 / 9 cm | 0 |
| N3 | old plan (re-timed) | 94% | 88% | 4 cm | 0 |
| S1 | ThreeStepPoleRun_014 | 94% | 95% | 3 / 6 cm | 0 |
| S2 | old plan (017, re-timed) | 94% | 95% | 4 cm | 0 |
| N1 | old plan (020, re-timed) | 88% | 91% | 3 cm | 0 |
| T1 | FiveStepPoleRun_018 | 86% | 89% | 4 / 7 cm | 0 |
| T2 | old plan (020, re-timed) | 81% | 83% | 4 cm | 0 |
| N2 | FiveStepPoleRun_020 | 77% | 86% | 4 / 7 cm | 0 |

- D2 / N3 were also run 192 times (RP10XS conditions without the stake errors): D2 default skill 95 %, N3 default 94 %.
- With the old plans (the method before this bundle) D1 reached 89 % with 46 cm standing drift and T1 67 %. These two layouts now use the skills selected by the new planner.
- The statistical error of 64 rollouts per cell is about ±5–8 points.
- **Simulation is an upper bound**, the real robot will be lower. On the first day with a new robot I estimate the single-attempt success rate at only 20–40 %.

---

## 9. Folders

| Path | Content |
|---|---|
| `models.json` | task → model folder |
| `model_bridge/`, `model_stakes/` | `policy.onnx`, `policy_meta.json`, `scene.mjb`, `parity.txt` |
| `planned_layouts/<layout>/` | `layout_aligned.json` (coordinates, with `default_plan` and `standard_start`), `*_motion.npz` (frame 0 = normal stance), `*_z.npy` (default skill code), `*_bridge.json` / `*_stakes.json`, `cands/<candidate>/` precomputed standard-start plans |
| `figures/` | top views of the 9 layouts |
| `point_coordinates.csv` | all points and check codes |
| `code/runtime/` | `hnu_deploy.py`, `obstacles_live.py`, `place_live.py` (**now shows the check code**), `sim2sim.py` (new: pendulum rope `HNU_S2S_ROPE_LEN`, standing drift `hold_drift`, waist gains `HNU_S2S_WAIST_KP`), `check_placement.py`, `update_bridge.py`, `parity_check.py`, `export_student.py`, `layout_sheet.py`, `odometry.py` |
| `code/chain/pipeline.py` | preview and selection of the retrieval chain (standard-start plans, standing-drift filter) |
| `code/robot_side/` | `refresh_obstacles.py`, `obstacles_rigid.py` (live obstacle measurement), `robot_health.py`, `waist_check.py` (health check) |
| `code/tools/` | `make_start.py` (builds the standard start), `std_plan.py` (standard-start planner), `coords.py` (builds the coordinate tables and figures of Sec. 2) |
| `router/`, `text_encoder/`, `skill_library/`, `code/planner`, `code/router`, `code/demo` | same as in the previous version, only needed for the retrieval chain |

---

## 10. Unverified or uncertain items (read before going on the robot)

| Item | Why uncertain | How to check |
|---|---|---|
| Standard start on the real robot | Only verified in simulation (stand → one step → blend into the skill), never run on the real robot | First do "standing only" of Sec. 1, then start with D2 |
| State of the new robot | Every robot differs. Motors, soles and battery cannot be confirmed in advance | Health check of Sec. 5 |
| Integration of the live obstacle measurement | `refresh_obstacles.py` was only tested on the server with synthetic data, never inside your `hnu_pipeline` | At the first takeover compare the printed measurement report with the `place_live` readings |
| Rigid-body markers | One ring of markers per post / stake was never tried on site. Whether the pivot lies on the axis has to be confirmed by hand | After placing, measure the neighbour spacings with a tape and compare them with the spacings shown by `place_live`. The difference should be < 1 cm |
| `code/chain/pipeline.py` | Only tested with stand-in planner / router. Depends on your `router.py` and on `layout_files` of `chain/planner.py`. Run time on the robot computer not measured | After replacing, run one instruction end to end. If it fails, use the default skills in the root of `planned_layouts` |
| D1 drifts 13 cm forward while standing | Success above 95 % in simulation, but whether the real robot drifts as little is unknown | Test D1 last among the bridge layouts. Stop if the drift exceeds 30 cm |
| How long to stand after the rope release | Training covers 0.3–2 s after the release, longer is not guaranteed (the simulation acceptance uses 2 s) | Start within 1–2 s after the release |
| Waist | This bundle is trained with a working waist. If the waist sags like on the faulty-waist robot, this bundle will fail | If the health check reports a waist problem, use the faulty-waist bundle |
| Velocity input of the stakes model | Training uses the motion-capture velocity from a 60 ms fit. Your estimator should match, but it was not checked item by item | While standing the velocity should be close to 0 without spikes |
| Beam height | Only the horizontal positions are measured. The beam height was not checked for every layout | After building the bridge, measure the height of the lower beam edge once (about 111 cm) |
| Real-robot success rate | Simulation is an upper bound | Run 3 times per layout first and record logs and videos |
