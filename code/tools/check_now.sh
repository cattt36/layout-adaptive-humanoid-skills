#!/bin/bash
# Stakes eval results + v4d tables.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
echo "== stakes3"; ls $H/runs/stakes_eval/stakes3
for f in $H/runs/stakes_eval/stakes3/*.log; do echo "-- $f"; grep -aE 'success|hit|anchor' $f | tail -n 8; done
echo "== tables_v4d"; cat $N/tables_v4d.md
tail -n 2 $N/v4d.log
