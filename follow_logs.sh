#!/bin/bash
# follow_logs.sh [output dir]: a tmux window per experiment, following its run.log
OUT=${1:-./figures}
KEEP='#####|multistart:|reused|refitting|fitting|successful|failed|WARNING|Error|Traceback|worker stopped'

tmux kill-session -t logs 2>/dev/null
first=1
for exp in $(cat BehaviorScreen/screen/analysis/experiments.txt); do
  cmd="tail -F -n 50 $OUT/$exp/run.log 2>/dev/null | grep --line-buffered -E '$KEEP'"
  if [ $first = 1 ]; then tmux new-session -d -s logs -n "$exp" "$cmd"; first=0
  else tmux new-window -t logs -n "$exp" "$cmd"; fi
done
tmux select-window -t logs:0
if [ -n "$TMUX" ]; then tmux switch-client -t logs; else tmux attach -t logs; fi