"""(user: live obstacle positions from the motion capture are PART of the deployment, not optional).

Measure the bridge posts / stakes at the takeover from their RIGID BODIES (XINGYING; pivot = post axis / stake centre at
the top), averaged over the last `seconds`; fall back to the unlabeled-marker method of the code
(obstacles_mocap.measure_posts) for posts without a rigid body.  Positions in the aligned frame (= mocap +
aligned_shift_xy), numbered as in layout_aligned.json.

  from obstacles_rigid import measure_obstacles
  meas, lines = measure_obstacles(src, kind, names, shift_xy, plan_xy, seconds=1.0)
    src   : real/mocap.py MocapSource (get / window by rigid-body name)
    kind  : "posts" | "stakes";  names: rigid-body names in layout order (config mocap.obstacle_bodies[kind])
    meas  : list of [x, y] or None per obstacle;  lines: report for the screen / log
"""
import numpy as np


def measure_obstacles(src, kind, names, shift_xy, plan_xy, seconds=1.0, min_poses=5, max_jitter=0.01, marker_fallback=True):
    shift = np.asarray(shift_xy or [0.0, 0.0], float)
    meas, lines = [], []
    for i, n in enumerate(names or []):
        ps = src.window(n, seconds) if n else []
        if len(ps) >= min_poses:
            xy = np.array([p.pos[:2] for p in ps], float)
            jit = float(np.max(np.std(xy, 0)))
            m = xy.mean(0) + shift
            if jit <= max_jitter:
                meas.append(m.tolist())
                d = np.asarray(plan_xy[i], float) - m if i < len(plan_xy) else np.zeros(2)
                lines.append(f"{kind[:-1]} {i + 1} ({n}): ({m[0]:.3f}, {m[1]:.3f}), {len(ps)} poses, jitter {1000 * jit:.1f} mm, "
                             f"vs plan dx {100 * -d[0]:+.1f} dy {100 * -d[1]:+.1f} cm")
                continue
            lines.append(f"{kind[:-1]} {i + 1} ({n}): jitter {1000 * jit:.1f} mm > {1000 * max_jitter:.0f} mm (moving / bad tracking)")
        else:
            lines.append(f"{kind[:-1]} {i + 1} ({n}): rigid body not seen ({len(ps)} poses in {seconds:.1f} s)")
        meas.append(None)
    if kind == "posts" and marker_fallback and any(m is None for m in meas):
        try:
            from obstacles_mocap import measure_posts  # the marker method (near-side markers, axis offset)
            mk, ml = measure_posts(src, plan_xy, shift.tolist())
            if mk is not None:
                for i, v in enumerate(mk):
                    if i < len(meas) and meas[i] is None and v is not None:
                        meas[i] = v
                        lines.append(f"post {i + 1}: from markers {ml[i]} (rigid body missing; axis offset assumption +-4 cm)")
        except Exception as e:  # noqa: BLE001
            lines.append(f"marker fallback failed: {e!r}")
    return meas, lines
