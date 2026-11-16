#!/bin/bash
# Where do the current teachers come from?  Checkpoints, run dirs, saved env / agent params, launch scripts.
H=/root/gpufree-data/hnu_v2
cat $H/runs/night/v6g10t_teachers.txt | tr ' ' '\n'
echo
for f in $(cat $H/runs/night/v6g10t_teachers.txt | tr ' ' '\n' | grep '/'); do
  d=$(dirname $f)
  echo "== $d"; ls $d | tail -4; ls $d/params 2>/dev/null
  [ -f $d/params/env.yaml ] && grep -nE "stake|STAKE|push|bridge_contact|weight:" $d/params/env.yaml | head -30
done
echo "== launch scripts mentioning the run names"
for f in $(cat $H/runs/night/v6g10t_teachers.txt | tr ' ' '\n' | grep '/'); do
  r=$(basename $(dirname $f)); grep -l "$r\|${r#*_*_}" $H/code/*.sh $H/code/tools/*.sh 2>/dev/null | head -3
done
ls $H/code/*.sh | head -40
