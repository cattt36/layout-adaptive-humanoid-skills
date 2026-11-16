#!/bin/bash
# List running training jobs (task + run name) and GPU memory.
ps -eo args | grep 'train_hn[u]' | sed -E 's/.*train_hnu.py (HNU-Track-[A-Za-z0-9-]+).*run-name ([a-z0-9_]+).*/\1 \2/'
ps -eo args | grep 'distill_trai[n]' | sed -E 's/.*--profile ([a-z0-9_]+).*/distill \1/'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
