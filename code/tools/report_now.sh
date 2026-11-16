#!/bin/bash
# Snapshot: v5 pipeline, bridge students (v4d baseline), stakes perception h10.
H=/root/gpufree-data/hnu_v2
PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
date '+%m-%d %H:%M'
echo "== v5 log"; cat $H/runs/night/v5.log
echo "== v5 latents"; ls $H/runs/latent_v5/seed*/ 2>/dev/null | tr '\n' ' '; echo
for p in Z0 Z1 Z2 Z3 Z4; do echo "v5 $p $(tail -n 1 $H/runs/students_v5/$p/train_log.jsonl 2>/dev/null | cut -c1-70)"; done
echo "== bridge students (v4d)"; cat $H/runs/night/bridge_students.log
for p in Z0 Z1 Z2 Z3 Z4; do echo "v4d $p $(tail -n 1 $H/runs/students_bridge/$p/train_log.jsonl 2>/dev/null | cut -c1-70)"; done
ls $H/runs/results_bridge 2>/dev/null | tr '\n' ' '; echo
echo "== stakes h10"
timeout 120 $PY tools/stakes_summary.py "$H/runs/stakes_eval/perc_h10/*/*.json"
grep -h 'curriculum finished' $H/runs/logs/stakes_perc_*.log
