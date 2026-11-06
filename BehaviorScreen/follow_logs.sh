#!/bin/bash
# follow_logs.sh [output dir]: a tmux pane per experiment, following its run.log
OUT=${1:-./figures}
EXPS=$(cat BehaviorScreen/point_process/analysis/experiments.txt)
KEEP='#####|multistart:|reused|refitting|fitting|successful|failed|WARNING|Error|Traceback|worker stopped'

tmux new-session -d -s logs -x "$(tput cols)" -y "$(tput lines)"
tmux set -t logs pane-border-status top
first=1
for exp in $EXPS; do
  cmd="tail -F -n 30 $OUT/$exp/run.log 2>/dev/null | grep --line-buffered -E '$KEEP'"
  if [ $first = 1 ]; then tmux send-keys -t logs "$cmd" C-m; first=0
  else tmux split-window -t logs "$cmd"; tmux select-layout -t logs tiled; fi
  tmux select-pane -t logs -T "$exp"
done
tmux attach -t logs