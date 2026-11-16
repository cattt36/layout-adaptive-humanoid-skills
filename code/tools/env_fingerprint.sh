#!/bin/bash
# Environment fingerprint for the migration check (run on both servers, compare).
B=/opt/conda/envs/isaaclab/bin/python3.11
echo "base python: $(ls -la $B 2>&1 | cut -c1-120)"
[ -x $B ] && $B -m pip freeze 2>/dev/null | sort | md5sum | sed 's/^/base pip freeze md5: /'
[ -x $B ] && $B -m pip freeze 2>/dev/null | grep -iE "^(torch|numpy|warp|mujoco|jax|scipy)" | tr '\n' ' '; echo
V=/root/gpufree-data/venvs/mjlab/bin/python
if [ -x $V ]; then
  $V -c "import mjlab, mujoco, warp, torch, numpy; print('venv:', 'mjlab', getattr(mjlab,'__version__','?'), mjlab.__file__, '| mujoco', mujoco.__version__, '| warp', warp.__version__, '| torch', torch.__version__, '| numpy', numpy.__version__)" 2>&1 | tail -2
  $V -c "import mujoco_warp; print('mujoco_warp', mujoco_warp.__file__)" 2>&1 | tail -1
fi
ls /root/gpufree-data/third_party 2>/dev/null
