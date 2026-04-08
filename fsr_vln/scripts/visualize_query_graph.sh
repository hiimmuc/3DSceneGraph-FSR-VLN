#!/usr/bin/env bash
# 3D Scene Graph - Visualize Query Graph (2x2 tmux layout)
set -e

SESSION="sg_viz_query"
FSR_VLN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# fsr_vln root — all python commands run from here
# FSR_VLN="$DIR/dependencies/HoloAgent/fsr_vln"
CONDA="conda activate fsrvln"
ROS="source /opt/ros/humble/setup.bash"
RVIZ_CFG="config/layout.rviz"
ZENOH_CFG="config/zenoh.json5"

SLEEP="sleep 1" # delay to ensure ROS2 nodes start in order

command -v tmux &>/dev/null || { echo "Install tmux: sudo apt install tmux"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION" -n main -x 220 -y 55
tmux set-option -t "$SESSION" -g mouse on

# Build the 2x2 grid:
#   pane 0 (top-left) | pane 1 (top-right)
#   pane 2 (bot-left) | pane 3 (bot-right)
tmux split-window -h -t "$SESSION:main"
tmux select-pane -t "$SESSION:main.0" && tmux split-window -v -t "$SESSION:main.0"
tmux select-pane -t "$SESSION:main.1" && tmux split-window -v -t "$SESSION:main.1"
# Pane order after splits: 0=top-left, 1=bot-left, 2=top-right, 3=bot-right

# --- Panel 0 (top-left): ROS2 map visual publisher ---
tmux send-keys -t "$SESSION:main.0" "cd $FSR_VLN && $ROS" C-m
tmux send-keys -t "$SESSION:main.0" "python application/visualize_query_graph/ros2_map_visual.py" C-m

# --- Panel 1 (bot-left): Visualize query graph ---
tmux send-keys -t "$SESSION:main.1" "$CONDA" C-m
tmux send-keys -t "$SESSION:main.1" "cd $FSR_VLN && $ROS" C-m
tmux send-keys -t "$SESSION:main.1" "$SLEEP && python application/visualize_query_graph/visualize_query_graph.py" C-m

# --- Panel 2 (top-right): RViz2 ---
tmux send-keys -t "$SESSION:main.2" "$ROS" C-m
tmux send-keys -t "$SESSION:main.2" "ros2 run rviz2 rviz2 -d $RVIZ_CFG" C-m

# --- Panel 3 (bot-right): Visualize graph ---
tmux send-keys -t "$SESSION:main.3" "cd $FSR_VLN && $ROS" C-m
tmux send-keys -t "$SESSION:main.3" "zenoh-bridge-ros2dds -c $ZENOH_CFG" C-m

tmux select-pane -t "$SESSION:main.0"
tmux select-layout -t "$SESSION:main" tiled
echo "Session: $SESSION | Attach: tmux attach -t $SESSION"
tmux attach -t "$SESSION"
