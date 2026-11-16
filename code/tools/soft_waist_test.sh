#!/bin/bash
# evening: does the REAL soft waist (PD kp 28.5 to target 0, not locked) explain the falls?
#   A lock = rigid (training assumption), B soft = PD with the policy gains (real ), C/D = stiffer waist PD gains
H=/root/gpufree-data/hnu_v2; C=$H/code; RS=/root/gpufree-data/robot_sim_check; PYD=/root/gpufree-data/venvs/deploy/bin/python
O=$RS/out_softwaist; mkdir -p $O; cd $C
B="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.05 --start-yaw 5 --dr-x --head-bias 3 --height-bias 0.02"
for n in D1 D2 S1 T1; do
  L=$(ls -d $RS/v6/in/planned_layouts/${n}_*); t=$(basename $L | cut -d_ -f2)
  if [ $t = CrouchedTraverse ]; then M=rtHwb_s5_c1500; X="$B"; else M=rtHws_s5_c1500; X="$B --vel-est --stake-jitter 0.03 --obst-shift 0.03 --perc-noise 0.02"; fi
  for v in lock soft kp150 kp300; do
    f=$O/${n}_${M}_$v.json; [ -f $f ] && continue
    case $v in lock) E=""; W="--waist-lock";; soft) E=""; W="";; kp150) E="150,4"; W="";; kp300) E="300,6"; W="";; esac
    if [ -n "$E" ]; then export HNU_S2S_WAIST_KP=$E; else unset HNU_S2S_WAIST_KP; fi
    $PYD deploy/sim2sim.py $H/deploy_exports/$M/L3 --crit task --trials 64 --workers 100 --tasks $t $W \
      --motion-dirs $L --latent-dir $L $X --out $f > $O/${n}_$v.log 2>&1
    echo "$(date +%T) $n $v done" >> $O/progress.txt
  done
done
$PYD - <<'PY' > $O/summary.txt
import json,glob,numpy as np,os
for f in sorted(glob.glob("/root/gpufree-data/robot_sim_check/out_softwaist/*.json")):
    r=json.load(open(f))["rows"]; wp=[x.get("waist_pitch_max",0) for x in r]
    print(os.path.basename(f), "success %.0f%%" % (100*np.mean([x["success_v2"] for x in r])), "waist pitch max median %.2f" % np.median(wp))
PY
echo "softwaist done" >> $O/progress.txt
