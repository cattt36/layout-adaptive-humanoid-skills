#!/bin/bash
# GPU processes with their command (profile / run name) and memory.
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read pid mem; do
  cmd=$(ps -o args= -p $pid 2>/dev/null | grep -o -e '--profile [a-z0-9_]*' -e 'run-name [a-z0-9_]*' -e 'eval_[a-z]*.py' | head -n 2 | tr '\n' ' ')
  echo "$pid $mem $cmd"
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
