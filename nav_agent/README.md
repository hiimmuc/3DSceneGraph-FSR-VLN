# NavAgent

This project contains a complete **embodied intelligence navigation system** (NavAgent), including modules for localization, mapping, navigation, semantic goal localization, voice interaction, and motor control. It supports a hybrid Docker + host machine deployment.

---

## 📁 Project Structure

```text
nav_agent/
├── humble_localization_nav2/        # Localization / navigation & obstacle avoidance modules (run inside Docker)
│   ├── g1_nav_bringup/              # One-click launcher for all launch files
│   ├── g1_navigation2/              # ROS 2 Navigation2 parameter interface
│   ├── lio_mapping_loc/             # FastLIVO2 + map relocalization module
│   ├── navigation2-humble/          # ROS 2 Navigation2 core package
│   ├── pubpose/                      # Receives goal pose from goal_publisher and forwards it to Nav2 for global navigation & obstacle avoidance
│   └── rpg_vikit-ros2/              # Third-party dependency for FastLIVO2
├── scripts/                         # Launch scripts (Docker / host machine)
│   ├── run_nav.sh                    # One-click launch of all algorithm modules inside Docker
│   ├── run_sem_nav.sh                # One-click launch of semantic navigation modules on host
│   └── run_sensors.sh                # One-click launch of sensors on host
└── sem_nav_ctr/                      # Voice / motor control / semantic goal localization modules (run on host)
    ├── chat_loc_python/             # Voice interaction client
    ├── g1_move/                     # G1 motor control interface
    └── goal_publisher/              # Semantic localization of target instances/regions; internally calls the fsr-vln HMSG to query goal poses
```

## 🚀 Feature Overview

- ✅ **Navigation and Obstacle Avoidance (Nav2)**  
  Based on the ROS 2 Navigation2 framework; supports global path planning and local obstacle avoidance.

- ✅ **FastLIVO2 Odometry + Relocalization**  
  Real-time LiDAR-inertial-visual odometry with map-based relocalization support.

- ✅ **Voice Interaction Control**  
  Implements voice-guided navigation tasks via a local voice client paired with a remote server. The current code only includes the client-side data acquisition component; it is recommended to implement your own voice interaction module, or wait for the next open-source release.

- ✅ **Semantic Goal Localization**  
  Resolves target names (e.g., "sofa", "exhibition hall") into specific 3D spatial goal poses.

- ✅ **One-click Launch Scripts**  
  Provides one-click startup solutions for both Docker and host machine environments.

---

## 🏃 Getting Started

### Docker Build and Run

Ensure Docker and the NVIDIA Container Toolkit are installed.

**Base image configuration:**

- Build a `ubuntu22.04 + ros2-humble` base image yourself
- Run `colcon build` inside the base image for all submodules in `humble_localization_nav2`

**Using the prebuilt image:**

- Use the image we provide directly: `ghcr.io/zhaoyu1992101/fsrvln:v1.0`

### Launch Commands

**Start navigation modules inside Docker:**

```bash
bash scripts/run_nav.sh
```

**Start semantic navigation modules on host:**

```bash
bash scripts/run_sem_nav.sh
```

**Start sensors on host:**

```bash
bash scripts/run_sensors.sh
