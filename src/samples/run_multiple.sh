#!/bin/bash

# Function to split panes
make_grid() {
    local h="$1"
    local v="$2"

    # Horizontal split
    if [ "$h" -gt 1 ]; then
        for (( i=1; i<h; i++ )); do
            tmux split-window -h
        done
        tmux select-layout even-horizontal
    fi

    # Vertical split
    if [ "$v" -gt 1 ]; then
        tmux select-pane -t 0
        v_total=$(tmux display-message -p '#{pane_height}')
        v_size=$(( v_total /  v ))
        for (( i=0; i<h; i++ )); do
            pane=$(( i * v ))
            tmux select-pane -t $pane
            for (( j=1; j<v; j++ )); do
                tmux split-window -l $v_size -v
                tmux select-pane -t $pane
            done
        done
        tmux select-pane -t 0
    fi
}

tmux new-session -d -s dqn_session

h=2
v=4
make_grid "$h" "$v"

for ((i=0; i<=7; i++)); do
    tmux select-pane -t $i
    tmux send-keys -t dqn_session "source venv/bin/activate" C-m
    tmux send-keys -t dqn_session "export PYTHONPATH=$PWD" C-m
done

command_prefix="python src/samples/sample_test_bench.py --device cuda --method priority_sampling --n_episodes 1000 --n_runs 20"
update_rates=("0.005" "0.01" "0.1")
for i in "${!update_rates[@]}"; do
    tmux select-pane -t "$i"
    tmux send-keys -t dqn_session "$command_prefix --target_network_update_rate ${update_rates[$i]}" C-m
done

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