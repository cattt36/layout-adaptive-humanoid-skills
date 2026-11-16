#!/bin/bash
# Stop the du_ft_auto.sh driver (its gate parser failed on the push file name); the round-2 training (du3_B2) is
# left running.  Then score the round-1 gates with the fixed parser.
H=/root/gpufree-data/hnu_v2
for p in $(pgrep -f "bash tools/du_ft_auto.sh"); do kill $p; done
sleep 1
pgrep -af "du_ft_auto" | grep -v stop_du_auto || echo "auto script stopped"
pgrep -af "run-name du3_B2" | grep -v stop_du_auto | cut -c1-90
cd $H/code
/root/gpufree-data/venvs/mjlab/bin/python tools/du_gate_score.py A B
