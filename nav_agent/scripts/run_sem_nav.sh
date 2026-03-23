#!/bin/bash

SESSION_NAME="robot_nav"

# run on hostmachine
# Kill existing tmux session if it exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux kill-session -t $SESSION_NAME
    echo "Session '$SESSION_NAME' has been deleted."
fi

# Create a new tmux session
tmux new-session -d -s $SESSION_NAME -n nav

# -------------------
# Pane 0: Voice interaction
# -------------------
tmux send-keys -t $SESSION_NAME:0 "1" C-m
tmux send-keys -t $SESSION_NAME:0 "source /mnt/disk1/mapvln/FSR-VLN/nav_agent/sem_nav_ctr/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0 "unset ASAN_OPTIONS" C-m
tmux send-keys -t $SESSION_NAME:0 "ros2 run chat_loc_python topic_chat_loc_pub" C-m

# -------------------
# Pane 1: Semantic goal localization
# -------------------
tmux split-window -h -t $SESSION_NAME:0
tmux send-keys -t $SESSION_NAME:0.1 "1" C-m
tmux send-keys -t $SESSION_NAME:0.1 "source /mnt/disk1/mapvln/FSR-VLN/nav_agent/sem_nav_ctr/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0.1 "unset ASAN_OPTIONS" C-m
tmux send-keys -t $SESSION_NAME:0.1 "ros2 run goal_publisher goal_pose_publisher" C-m

# -------------------
# Pane 2: Pipe writer (g1_getvel_node) — reads /cmd_vel from Nav2 and writes to named pipe
# -------------------
tmux split-window -v -t $SESSION_NAME:0
tmux send-keys -t $SESSION_NAME:0.2 "1" C-m
tmux send-keys -t $SESSION_NAME:0.2 "[ -p /tmp/vel_fifo ] && rm /tmp/vel_fifo" C-m
tmux send-keys -t $SESSION_NAME:0.2 "mkfifo /tmp/vel_fifo" C-m
tmux send-keys -t $SESSION_NAME:0.2 "source /mnt/disk1/mapvln/FSR-VLN/nav_agent/sem_nav_ctr/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0.2 "unset ASAN_OPTIONS" C-m
tmux send-keys -t $SESSION_NAME:0.2 "ros2 run g1_move g1_getvel_node" C-m

# -------------------
# Pane 3: Pipe reader (g1_pubvel_node) — sends velocity commands to motor driver
# -------------------
tmux split-window -v -t $SESSION_NAME:0
tmux send-keys -t $SESSION_NAME:0.3 "1" C-m
tmux send-keys -t $SESSION_NAME:0.3 "source /mnt/disk1/mapvln/FSR-VLN/nav_agent/sem_nav_ctr/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0.3 "unset ASAN_OPTIONS" C-m
tmux send-keys -t $SESSION_NAME:0.3 "ros2 run g1_move g1_pubvel_node" C-m

# Arrange all four panes to be visible
tmux select-layout -t $SESSION_NAME:0 tiled

# Attach to the tmux session
tmux attach-session -t $SESSION_NAME
