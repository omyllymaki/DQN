#!/bin/bash

tmux new-session -d -s dqn_session

tmux split-window -t dqn_session -v
tmux split-window -t dqn_session -v
tmux split-window -t dqn_session -v
tmux split-window -t dqn_session -v
tmux select-layout -t dqn_session even-vertical

for ((i=0; i<=3; i++)); do
    tmux select-pane -t $i
    tmux send-keys -t dqn_session "source venv/bin/activate" C-m
    tmux send-keys -t dqn_session "export PYTHONPATH=$PWD" C-m
done

tmux select-pane -t 0
tmux send-keys -t dqn_session "python src/samples/sample_multi_run.py --target_network_update_rate 0.05 --device cpu" C-m
tmux select-pane -t 1
tmux send-keys -t dqn_session "python src/samples/sample_multi_run.py --target_network_update_rate 0.01 --device cpu " C-m
tmux select-pane -t 2
tmux send-keys -t dqn_session "python src/samples/sample_multi_run.py --target_network_update_rate 0.1 --device cpu" C-m

tmux rename-window 'runs'

tmux new-window -t dqn_session
tmux rename-window 'monitoring'
tmux split-window -t monitoring -h
tmux select-layout -t monitoring even-horizontal
tmux select-pane -t 0
tmux send-keys -t monitoring "watch -n 0.5 nvidia-smi" C-m
tmux select-pane -t 1
tmux send-keys -t monitoring "htop" C-m

tmux select-window -t 0


tmux attach-session -t dqn_session