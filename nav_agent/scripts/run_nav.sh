#!/bin/bash

# run inside docker
SESSION_NAME="robot_nav_ros2"

# Kill existing tmux session if it exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux kill-session -t $SESSION_NAME
    echo "Session '$SESSION_NAME' has been deleted."
fi

# Create a new tmux session
tmux new-session -d -s $SESSION_NAME -n nav

# -------------------
# Pane 0: fast_livo online_reloc
# -------------------
tmux send-keys -t $SESSION_NAME:0 "source /agentic_robot/G1_Nav_Bringup/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0 "source /workspace/fastlivo_new_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0 "ros2 launch fast_livo online_reloc.launch.py use_rviz:=True" C-m

# -------------------
# Pane 1: fast_livo online_livo
# -------------------
tmux split-window -h -t $SESSION_NAME:0
tmux send-keys -t $SESSION_NAME:0.1 "source /agentic_robot/G1_Nav_Bringup/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0.1 "source /workspace/fastlivo_new_ws/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0.1 "ros2 launch fast_livo online_livo.launch.py" C-m

# -------------------
# Pane 2: g1_navigation2
# -------------------
tmux split-window -v -t $SESSION_NAME:0
tmux send-keys -t $SESSION_NAME:0.2 "source /agentic_robot/G1_Nav_Bringup/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0.2 "ros2 launch g1_navigation2 navigation2.launch.py" C-m

# -------------------
# Pane 3: pubpose
# -------------------
tmux split-window -v -t $SESSION_NAME:0
tmux send-keys -t $SESSION_NAME:0.3 "source /agentic_robot/G1_Nav_Bringup/install/setup.bash" C-m
tmux send-keys -t $SESSION_NAME:0.3 "ros2 run pubpose pubpose" C-m

# Arrange all four panes to be visible
tmux select-layout -t $SESSION_NAME:0 tiled

# Attach to the tmux session
tmux attach-session -t $SESSION_NAME

