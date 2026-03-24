<div align="center">
<img src="docs/assets/holoagent_logo_text.png" alt="HoloAgent Logo" width="500"/>

---

# HoloAgent: Unified Robot Agent Framework

</div>

A unified, agentic system for general-purpose robots, enabling multi-modal perception, mapping and localization, and autonomous mobility and manipulation, with intelligent interaction with users.

## 🤖 FSR-VLN

[![Projcet](https://img.shields.io/badge/📖-Project-blue)](https://horizonrobotics.github.io/robot_lab/fsr-vln)
[![📄 arXiv](https://img.shields.io/badge/📄-arXiv-b31b1b)](https://arxiv.org/abs/2509.13733)
[![中文介绍](https://img.shields.io/badge/中文介绍-07C160?logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/HqnBlTNqOL3Z4Kg8tLHCSw)
> ***FSR-VLN*** is a core component of the HoloAgent framework. It provides natural language guided navigation and intelligent interaction for general-purpose robots, and is built on core agent components such as mapping and localization, multimodal perception, decision-making and planning, and memory management. At its core, FSR-VLN is a vision–language navigation system that integrates a Hierarchical Multi-modal Scene Graph (HMSG) for coarse-to-fine environment representation with Fast-to-Slow Navigation Reasoning (FSR), leveraging VLM-driven refinement to enable efficient, real-time, long-range spatial reasoning.

<img src="docs/assets/FSR_VLN_framework.png" alt="Overall Framework" width="700"/>

## Checklist

- [x] Release the code of FSR-VLN.

---

## 🗺️ System Overview

The full HoloAgent pipeline has three sequential phases:

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — DATA COLLECTION (ROS2 bag)                                        │
│  Drive robot through environment → record /rgb, /depth, /tf into a .db3 bag  │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │ scripts/rosbag_2_dataset.py
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — OFFLINE SEMANTIC MAPPING (fsr_vln)                                │
│  Horizon RGB-D dataset → SAM masks + CLIP embeddings → HMSG graph on disk    │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │ copy graph.json to robot
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — ONLINE NAVIGATION (nav_agent / ROS2)                              │
│  Voice/text query → HMSG lookup → PoseStamped goal → Nav2 → robot motion     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Step-by-step Workflow

### Prerequisites

| Requirement | Version |
|---|---|
| Ubuntu | 22.04 |
| ROS 2 | Humble |
| Python | 3.9 |
| CUDA | 11.8+ |
| Hardware | Livox MID360 LiDAR, RealSense D435i, IMU |

---

### Phase 1 — Data Collection with ROS2

> **Goal:** Record a synchronized RGB-D + odometry bag while manually driving the robot through the target environment.

#### 1.1 Start sensors

```bash
bash nav_agent/scripts/run_sensors.sh
# Launches (via tmux):
#   - RealSense D435i:   ros2 launch realsense2_camera rs_new.py
#   - IMU publisher:     ros2 run imu_publisher imu_extractor
#   - Livox MID360:      ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

#### 1.2 Start LiDAR odometry (for poses)

Inside the Docker container:

```bash
# Build map while driving (writes /tf: map → camera_color_optical_frame)
ros2 launch fast_livo online_livo.launch.py
```

#### 1.3 Record the bag

```bash
ros2 bag record \
    /camera/color/image_raw \
    /camera/depth/image_rect_raw \
    /camera/color/camera_info \
    /tf \
    /tf_static \
    -o /data/my_scene_bag
```

Drive the robot slowly through all rooms (≈ 0.3 m/s). Aim for overlapping viewpoints.

#### 1.4 Convert bag to dataset format

Install the pure-Python rosbag reader (no ROS2 install required):

```bash
pip install rosbags opencv-python scipy tqdm
```

Run the conversion script:

```bash
python scripts/rosbag_2_dataset.py \
    --bag    /data/my_scene_bag/ \
    --output /mnt/holoagent/fsrvln/rgbd_datasets/my_scene/ \
    --rgb-topic   /camera/color/image_raw \
    --depth-topic /camera/depth/image_rect_raw \
    --tf-parent   map \
    --tf-child    camera_color_optical_frame \
    --skip 7          # save every 8th frame (matches pipeline skip_frames=8)
```

**Output dataset layout:**

```text
my_scene/
├── images/          # RGB frames:  {timestamp:.4f}.png
├── depth/           # Depth frames: {timestamp:.4f}.png  (16-bit mm)
├── poses.txt        # TUM format: timestamp tx ty tz qx qy qz qw
└── camera_info.yaml       # Camera intrinsics
```

> **Tip:** Check `poses.txt` — if the robot was stationary the translation should change each line. All-zero translations indicate TF was not published; ensure the odometry node was running during recording.

---

### Phase 2 — Offline Semantic Mapping (fsr_vln)

> **Goal:** Build the Hierarchical Multi-modal Scene Graph (HMSG) from the collected RGB-D dataset.

#### 2.1 Environment setup

```bash
cd HoloAgent/fsr_vln/
conda env create -f environment.yaml
conda activate fsrvln
pip install -e .
```

#### 2.2 Download model checkpoints

```bash
mkdir -p checkpoints

# Open CLIP (ViT-L/14 — 800 MB)
wget "https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_pytorch_model.bin?download=true" \
     -O checkpoints/open_clip_pytorch_model.bin

# SAM ViT-H (2.5 GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
     -O checkpoints/sam_vit_h_4b8939.pth
```

#### 2.3 Create a config for your scene

Copy an existing config and edit it:

```bash
cp config/semantic_scene_reconstruction_ic4f.yaml \
   config/semantic_scene_reconstruction_custom.yaml
```

Edit `config/semantic_scene_reconstruction_custom.yaml`:

```yaml
main:
  device: cuda
  use_gpt: false          # set true only if Azure OpenAI is configured
  dataset: horizon        # use the Horizon/custom RGB-D dataloader
  scene_id: my_scene      # must match the folder name inside dataset_path
  dataset_path: /mnt/holoagent/fsrvln/rgbd_datasets/
  depth_cut: 10.0
  save_path: /mnt/holoagent/fsrvln/scene_graphs/

models:
  clip:
    type: ViT-L/14
    checkpoint: checkpoints/open_clip_pytorch_model.bin
  sam:
    checkpoint: checkpoints/sam_vit_h_4b8939.pth
    type: vit_h
    points_per_side: 12
    pred_iou_thresh: 0.88

pipeline:
  voxel_size: 0.05
  skip_frames: 8          # must match --skip+1 used in rosbag_2_dataset.py
  obj_labels: SCANNET200
  merge_type: sequential
```

#### 2.4 Run the mapping pipeline

```bash
cd fsr_vln/
python application/semantic_scene_reconstrucion_offline/semantic_scene_reconstruction.py \
    --config-name=semantic_scene_reconstruction_custom
```

Expected console output:

```text
horizon dataset init success!
Processing frame 0/N ...
[SAM] generating masks ...
[CLIP] extracting features ...
Building scene graph ...
Saving graph → /mnt/holoagent/fsrvln/scene_graphs/my_scene/graph.json
Done.
```

**Output files:**

```text
scene_graphs/my_scene/
├── graph.json            # full HMSG (floors → rooms → objects)
├── pcd/                  # per-floor / per-room point clouds (.ply)
├── objects/              # per-object metadata + point cloud
├── rooms/
└── features/
    ├── object_features.pkl
    └── room_features.pkl
```

#### 2.5 Visualize and test queries (optional)

Set up Azure OpenAI (needed for room-name generation via LLM):

```bash
export AZURE_OPENAI_API_KEY="<your-key>"
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
```

Edit `config/visualize_query_graph_icra_ic4f.yaml` — set `main.graph_path` to your output, then:

```bash
python application/visualize_query_graph/visualize_query_graph_ic4f.py
```

An interactive 3D viewer opens. Type a query (e.g. *"chair"*, *"office desk"*) and the best-matching object is highlighted.

---

### Phase 3 — Online Navigation with ROS2 (nav_agent)

> **Goal:** Use the pre-built HMSG to navigate the robot to natural-language goals in real time.

#### 3.1 Copy the scene graph to the robot

```bash
scp -r /mnt/holoagent/fsrvln/scene_graphs/my_scene/ robot:/mnt/graph/
```

#### 3.2 Configure the goal_publisher node

Edit `nav_agent/sem_nav_ctr/src/goal_publisher/config/visualize_query_graph_demo.yaml`:

```yaml
main:
  graph_path: /mnt/graph/my_scene/
  use_gpt: false
```

#### 3.3 Build the ROS2 workspace (host — sem_nav_ctr)

```bash
cd nav_agent/sem_nav_ctr/
colcon build --symlink-install
source install/setup.bash
```

#### 3.4 Build the Docker image (nav2 + FastLIVO2)

Use the prebuilt image or build from scratch:

```bash
# Option A: use prebuilt image
docker pull ghcr.io/zhaoyu1992101/fsrvln:v1.0

# Option B: build locally (ubuntu22.04 + ros2-humble base)
cd nav_agent/humble_localization_nav2/
colcon build --symlink-install
```

#### 3.5 Launch the full navigation stack

Open **3 terminals**:

**Terminal 1 — Sensors (host):**

```bash
bash nav_agent/scripts/run_sensors.sh
```

**Terminal 2 — Nav2 + FastLIVO2 (Docker):**

```bash
# Starts (tmux, 4 panes):
#   Pane 0: FastLIVO2 relocalization  → ros2 launch fast_livo online_reloc.launch.py
#   Pane 1: FastLIVO2 odometry        → ros2 launch fast_livo online_livo.launch.py
#   Pane 2: ROS2 Navigation2          → ros2 launch g1_navigation2 navigation2.launch.py
#   Pane 3: Goal pose relay           → ros2 run pubpose pubpose
bash nav_agent/scripts/run_nav.sh
```

**Terminal 3 — Semantic navigation (host):**

```bash
# Starts (tmux, 4 panes):
#   Pane 0: Voice/text input          → ros2 run chat_loc_python topic_chat_loc_pub
#   Pane 1: HMSG query → /object_pose → ros2 run goal_publisher goal_pose_publisher
#   Pane 2: Velocity reader           → ros2 run g1_move g1_getvel_node
#   Pane 3: Motor driver              → ros2 run g1_move g1_pubvel_node
bash nav_agent/scripts/run_sem_nav.sh
```

#### 3.6 Send navigation goals

**Via ROS2 topic (programmatic):**

```bash
ros2 topic pub --once /chat_loc_pub std_msgs/msg/String \
    'data: "loc::office chair::1"'
```

**Via voice:** Speak a command to the DRobotC ASR service; it publishes `loc::<object>::1` automatically.

#### 3.7 Data flow diagram

```
  User: "Take me to the chair"
         │
         ▼
  /chat_loc_pub  (std_msgs/String — "loc::chair::1")
         │
         ▼  [goal_publisher node]
  HMSG graph.load_graph()
  → CLIP embed "chair" → cosine similarity search
  → best-match object → get median point of object PCD
         │
         ▼
  /object_pose  (geometry_msgs/PoseStamped)
         │
         ▼  [pubpose node — inside Docker]
  Nav2 /follow_waypoints  (action)
         │
  ┌──────┴───────┐
  │ Global path  │  Dijkstra planner
  │ Local path   │  DWB controller → /cmd_vel
  └──────┬───────┘
         │
         ▼  [g1_getvel_node → /tmp/vel_fifo → g1_pubvel_node]
  Motor commands → Robot moves to goal
         │
         ▼
  Nav2: goal reached ✓
```

---

### Quick-start Checklist

| Step | Command / Action | Done? |
|---|---|---|
| 1 | Record ROS2 bag during manual exploration | ☐ |
| 2 | `python scripts/rosbag_2_dataset.py --bag ... --output ...` | ☐ |
| 3 | `conda activate fsrvln && pip install -e fsr_vln/` | ☐ |
| 4 | Download CLIP + SAM checkpoints | ☐ |
| 5 | Edit `config/semantic_scene_reconstruction_custom.yaml` | ☐ |
| 6 | `python application/.../semantic_scene_reconstruction.py --config-name=...` | ☐ |
| 7 | Copy `graph.json` + feature files to robot | ☐ |
| 8 | Edit `goal_publisher/config/visualize_query_graph_demo.yaml` | ☐ |
| 9 | `bash nav_agent/scripts/run_sensors.sh` | ☐ |
| 10 | `bash nav_agent/scripts/run_nav.sh` (Docker) | ☐ |
| 11 | `bash nav_agent/scripts/run_sem_nav.sh` (host) | ☐ |
| 12 | `ros2 topic pub /chat_loc_pub ... "loc::chair::1"` | ☐ |

---

## 🏗 Original Pipeline Reference

### 1. Semantic Mapping and Retrieval Pipeline

- **Task:** Implement the semantic mapping and retrieval system based on the instructions in `fsr_vln/README.md`.
- **Steps:**
    1.  Download the necessary pre-trained model checkpoints.
    2.  Download and configure the required datasets.
    3.  Set up the environment and dependencies as specified.
    4.  Run the complete pipeline to verify its functionality for semantic mapping and visual place retrieval.

### 2. Navigation Agent Setup and Execution

- **Task:** Set up and test the navigation agent according to `nav_agent/README.md`.
- **Steps:**
    1.  Install all required dependencies for the navigation environment.
    2.  Configure the necessary parameters and environment settings.
    3.  Execute the navigation agent to ensure it runs successfully and performs its intended tasks.

## 📚 Publications & Citation

If you find our project useful, please consider citing it:

```bibtex
@misc{zhou2025fsrvlnfastslowreasoning,
      title={FSR-VLN: Fast and Slow Reasoning for Vision-Language Navigation with Hierarchical Multi-modal Scene Graph}, 
      author={Xiaolin Zhou and Tingyang Xiao and Liu Liu and Yucheng Wang and Maiyue Chen and Xinrui Meng and Xinjie Wang and Wei Feng and Wei Sui and Zhizhong Su},
      year={2025},
      eprint={2509.13733},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.13733}, 
}
```

---

## ⚖️ License

This project is licensed under the [Apache License 2.0](LICENSE). See the `LICENSE` file for details.
