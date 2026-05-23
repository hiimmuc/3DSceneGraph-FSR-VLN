"""
LICENSE.

This project as a whole is licensed under the Apache License, Version 2.0.

THIRD-PARTY LICENSES

Third-party software already included in HoloAgent is governed by the separate
Open Source license terms under which the third-party software has been
distributed.

NOTICE ON LICENSE COMPATIBILITY FOR DISTRIBUTORS

Notably, this project depends on the third-party software FAST-LIVO2 and HOVSG.
Their default licenses restrict commercial use—separate permission from their
original authors is required for commercial integration/redistribution.

The third-party software FAST-LIVO2 dependency (licensed under GPL-2.0-only)
utilizes rpg_vikit-ros2 which contains components under the GPL-3.0. Please be
aware of license compatibility when distributing a combined work.

DISCLAIMER

Users are solely responsible for ensuring compliance with all applicable
license terms when using, modifying, or distributing the project. Project
maintainers accept no liability for any license violations arising from such
use.
"""

import json
import logging
import os
import re
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig, OmegaConf

# ROS 2 imports
try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from application.visualize_query_graph.map_visual import MapVisual
from benchmark_queries import get_queries
from memory.hmsg.graph.graph_runtime import GraphRuntime
from memory.hmsg.utils.label_feats import (
    CSV_LABEL_REGISTRY,
    LABEL_REGISTRY,
    _flatten_classes,
    _load_csv_classes,
)
from memory.hmsg.utils.llm_utils import check_llm_connection

# Coordinate transform: scene graph → LiDAR map
_T_SWITCH_AXIS = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)
_T_TO_MAP = np.linalg.inv(_T_SWITCH_AXIS)

# Distinct colors for each result object (cycles if more than 8)
_OBJ_COLORS = [
    [1.00, 0.15, 0.15],  # red
    [0.15, 0.85, 0.15],  # green
    [0.15, 0.45, 1.00],  # blue
    [1.00, 0.60, 0.10],  # orange
    [0.80, 0.10, 0.80],  # purple
    [0.10, 0.85, 0.85],  # cyan
    [1.00, 1.00, 0.10],  # yellow
    [1.00, 0.40, 0.65],  # pink
]


# ---------------------------------------------------------------------------
# ROS 2 publisher
# ---------------------------------------------------------------------------


class QueryResultPublisher(Node if _ROS2_AVAILABLE else object):
    """ROS 2 node that publishes query results as PoseStamped + MarkerArray."""

    def __init__(self):
        super().__init__("hmsg_query_result_publisher")
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_position", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/goal_marker", 10)
        self.get_logger().info("QueryResultPublisher ready.")

    def publish_results(self, result_entries: List[dict], query: str) -> None:
        """Publish top result as PoseStamped goal and all results as Markers."""
        if not result_entries:
            return

        now = self.get_clock().now().to_msg()

        # --- Goal: top-ranked object ---
        top = result_entries[0]
        pos = top["lidar_map_position"]

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = now
        pose_msg.pose.position.x = float(pos[0])
        pose_msg.pose.position.y = float(pos[1])
        pose_msg.pose.position.z = float(pos[2])
        pose_msg.pose.orientation.w = 1.0
        self.goal_pub.publish(pose_msg)

        # --- Markers: one sphere per result object ---
        marker_array = MarkerArray()

        # Clear previous markers
        for old_id in range(100):
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = now
            delete_marker.ns = "hmsg_query_goals"
            delete_marker.id = old_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

        for i, entry in enumerate(result_entries):
            p = entry["lidar_map_position"]
            color = _OBJ_COLORS[i % len(_OBJ_COLORS)]

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = now
            marker.ns = "hmsg_query_goals"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(p[0])
            marker.pose.position.y = float(p[1])
            marker.pose.position.z = float(p[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color = ColorRGBA(
                r=float(color[0]), g=float(color[1]), b=float(color[2]), a=1.0
            )
            marker.lifetime.sec = 0
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)
        self.get_logger().info(
            f"Published goal for '{query}' → "
            f"[{', '.join(f'{v:.3f}' for v in pos)}], "
            f"{len(result_entries)} marker(s)."
        )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_and_save(
    room_pcd: o3d.geometry.PointCloud,
    obj_pcds: List[o3d.geometry.PointCloud],
    spheres: List[o3d.geometry.TriangleMesh],
    save_path: str = "scene.png",
) -> None:
    """Render room + all result objects (each a different color) in one scene.

    Saves a screenshot to *save_path*, then keeps the window open until the
    user presses Q or Esc.
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        visible=True,
        window_name="Scene Query Result — Press Q or Esc to close",
    )
    vis.add_geometry(room_pcd)
    for obj_pcd in obj_pcds:
        vis.add_geometry(obj_pcd)
    for sphere in spheres:
        vis.add_geometry(sphere)

    vis.poll_events()
    vis.update_renderer()

    # Focus camera on the first object (if any)
    if obj_pcds:
        obj_center = np.array(obj_pcds[0].get_center())
        room_center = np.array(room_pcd.get_center())
        cam_pos = room_center + np.array([0, 5.0, 0.0])
        ctr = vis.get_view_control()
        ctr.set_lookat(obj_center)
        diff = cam_pos - obj_center
        norm = np.linalg.norm(diff)
        if norm > 1e-6:
            ctr.set_front(diff / norm)
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.7)
        vis.poll_events()
        vis.update_renderer()

    vis.capture_screen_image(save_path)
    print(f"Saved visualization to {save_path}")

    # Keep window open until Q or Esc
    close_flag = [False]

    def _close(v):
        close_flag[0] = True
        return False

    vis.register_key_callback(ord("Q"), _close)  # GLFW key Q (covers q too)
    vis.register_key_callback(256, _close)  # GLFW_KEY_ESCAPE

    while not close_flag[0]:
        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()


def _print_result_tree(
    hmsg: GraphRuntime,
    objects: list,
    rooms: list,
    scores: list,
) -> None:
    """Print query results as a floor → room → object tree."""
    # Build ordered groups: floor_label -> room_label -> [(obj_name, score)]
    from collections import OrderedDict

    tree: "OrderedDict[str, OrderedDict[str, list]]" = OrderedDict()

    for i, (obj, room) in enumerate(zip(objects, rooms)):
        score = float(scores[i]) if i < len(scores) else None

        # Floor label
        obj_id_parts = str(obj.object_id).split("_")
        floor_idx = int(obj_id_parts[0]) if obj_id_parts else -1
        if 0 <= floor_idx < len(hmsg.floors):
            fl = hmsg.floors[floor_idx]
            floor_label = fl.name if fl.name else f"Floor {floor_idx}"
        else:
            floor_label = f"Floor {floor_idx}"

        # Room label
        room_label = room.name if room.name else room.room_id

        tree.setdefault(floor_label, OrderedDict()).setdefault(room_label, []).append(
            (obj.name, score)
        )

    print("\nQuery Graph Result:")
    for floor_label, room_map in tree.items():
        print(f"[{floor_label}]")
        room_items = list(room_map.items())
        for r_idx, (room_label, obj_list) in enumerate(room_items):
            is_last_room = r_idx == len(room_items) - 1
            room_prefix = "└── " if is_last_room else "├── "
            print(f"  {room_prefix}[{room_label}]")
            obj_indent = "      " if is_last_room else "  │   "
            for o_idx, (obj_name, score) in enumerate(obj_list):
                is_last_obj = o_idx == len(obj_list) - 1
                obj_prefix = "└── " if is_last_obj else "├── "
                score_str = f" ({score:.4f})" if score is not None else ""
                print(f"{obj_indent}{obj_prefix}{obj_name}{score_str}")


def _make_safe_filename(name: str) -> str:
    """Convert a label into a filename-safe string."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    name = re.sub(r"_+", "_", name)
    return name.strip("_.-") or "unnamed"


# ---------------------------------------------------------------------------
# Query loop
# ---------------------------------------------------------------------------


def run_query_loop(
    hmsg: GraphRuntime,
    queries: List[str],
    slow_reasoning: bool,
    run_dir: str,
    visualize: bool = True,
    ros_publisher: "QueryResultPublisher | None" = None,
) -> Tuple[List[dict], Dict[str, float]]:
    """Run the full query loop over a list of instructions.

    Returns:
        (all_results, metric_sums) where metric_sums contains accumulated timing values.
    """
    all_results: List[dict] = []
    metric_sums: Dict[str, float] = {
        "Total_Time": 0.0,
        "FastMatching": 0.0,
        "ObjectInImageCheck": 0.0,
        "VLM_Rethinking": 0.0,
        "Re_Matching": 0.0,
        "LLM_Parse_Time": 0.0,
    }

    for query in queries:
        print("=" * 80)
        print(f"[Query] {query}")

        query_save_dir = os.path.join(run_dir, "query_session_logs")
        os.makedirs(query_save_dir, exist_ok=True)
        hmsg.curr_query_save_dir = query_save_dir

        t0 = time.time()
        floor, rooms, objects, res_dict = hmsg.query_hierarchy(
            query, top_k=5, slow_reasoning=slow_reasoning
        )
        query_time = time.time() - t0

        print(f"Elapsed: {query_time:.4f}s")
        floor_id_val = floor.floor_id if floor is not None else -1

        scores = res_dict.get("object_scores", [])
        _print_result_tree(hmsg, objects, rooms, scores)

        found = len(objects) > 0
        obj_name = objects[0].name if found else "unknown"

        # --- Collect per-object geometry and build result entries ---
        result_entries: List[dict] = []
        obj_pcds: List[o3d.geometry.PointCloud] = []
        spheres: List[o3d.geometry.TriangleMesh] = []

        # Union of unique room PCDs as background
        seen_room_ids: set = set()
        room_pcd_combined: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        for room in rooms:
            if room.room_id not in seen_room_ids:
                seen_room_ids.add(room.room_id)
                room_pcd_combined = room_pcd_combined + deepcopy(room.pcd)

        for i, (obj, room) in enumerate(zip(objects, rooms)):
            color = _OBJ_COLORS[i % len(_OBJ_COLORS)]

            obj_pcd = deepcopy(obj.pcd)
            obj_pcd.paint_uniform_color(color)

            obj_center = np.array(obj.pcd.get_center())
            obj_center_in_map = (_T_TO_MAP @ np.hstack((obj_center, 1.0)))[:3]
            print(f"Object {i} - '{obj.name}' position:")
            print(f"  - scene graph: {obj_center}")
            print(f"  - lidar map:   {obj_center_in_map}")

            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            sphere.translate(obj_center)
            sphere.paint_uniform_color(color)

            obj_pcds.append(obj_pcd)
            spheres.append(sphere)

            score_val = float(scores[i]) if i < len(scores) else None
            obj_id_parts = str(obj.object_id).split("_")
            actual_floor = int(obj_id_parts[0]) if obj_id_parts else floor_id_val
            result_entries.append(
                {
                    "floor": actual_floor,  # Get floor from object_id prefix if floor not provided (floor=-1)
                    "room_id": room.room_id,
                    "object_id": obj.object_id,
                    "object_score": score_val,
                    "scene_graph_position": obj_center.tolist(),
                    "lidar_map_position": obj_center_in_map.tolist(),
                }
            )

        # --- Publish ROS 2 goal + markers ---
        if ros_publisher is not None:
            ros_publisher.publish_results(result_entries, query)

        # --- Save combined scene (one PLY + one PNG) ---
        combined = room_pcd_combined
        for obj_pcd in obj_pcds:
            combined = combined + obj_pcd
        for sphere in spheres:
            combined = combined + sphere.sample_points_uniformly(number_of_points=500)

        if result_entries:
            primary = result_entries[0]
            base_name = _make_safe_filename(
                f"{obj_name}_{primary['room_id']}_floor{primary['floor']}"
            )
        else:
            base_name = _make_safe_filename(f"unknown_{floor_id_val}")

        pcd_path = os.path.join(query_save_dir, f"{base_name}.ply")
        png_path = os.path.join(query_save_dir, f"{base_name}.png")
        o3d.io.write_point_cloud(pcd_path, combined)
        print(f"  Saved {pcd_path}")

        if visualize:
            visualize_and_save(room_pcd_combined, obj_pcds, spheres, save_path=png_path)

        # --- Write per-query results.json ---
        query_result = {
            "query": query,
            "object": obj_name,
            "found": found,
            "results": result_entries,
            "elapsed": f"{query_time:.4f}s",
        }
        results_json_path = os.path.join(query_save_dir, f"{base_name}.json")
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(query_result, f, ensure_ascii=False, indent=2)
        print(f"  Results saved to {results_json_path}")

        all_results.append(query_result)
        for key in metric_sums:
            metric_sums[key] += res_dict.get(key, 0.0)

    return all_results, metric_sums


def _print_and_build_metrics(metric_sums: Dict[str, float], n: int) -> Dict[str, float]:
    """Compute averages, print them, and return the summary dict."""
    avgs = {k: v / n for k, v in metric_sums.items()}
    print(f"\naverage_total_time          : {avgs['Total_Time']:.4f}s")
    print(f"average_llm_parse_time      : {avgs['LLM_Parse_Time']:.4f}s")
    print(f"average_fast_matching_time  : {avgs['FastMatching']:.4f}s")
    print(f"average_obj_in_image_check  : {avgs['ObjectInImageCheck']:.4f}s")
    print(f"average_vlm_rethinking_time : {avgs['VLM_Rethinking']:.4f}s")
    print(f"average_re_matching_time    : {avgs['Re_Matching']:.4f}s")
    return avgs


def _interactive_queries() -> Generator[str, None, None]:
    """Yield queries from stdin until the user types 'q'."""
    print("Interactive mode — type a query and press Enter. Type 'q' to quit.")
    while True:
        try:
            query = input("Query: ").strip()
        except EOFError:
            break
        if query.lower() == "q":
            break
        if query:
            yield query


def _load_object_labels_from_config(obj_labels_key: str, label_feat_path: str = None) -> List[str]:
    """Load object labels from the label registry by key.

    Args:
        obj_labels_key: Label registry key (e.g., 'FINALLABEL', 'SCANNET200').
        label_feat_path: Optional path where CSV files are stored. If None, tries to infer from module location.

    Returns:
        List of object class names.

    Raises:
        ValueError: If obj_labels_key not found in registry.
    """
    if obj_labels_key in LABEL_REGISTRY:
        classes_source, _ = LABEL_REGISTRY[obj_labels_key]
        if obj_labels_key == "OPENVOCAB_MATTERPORT_LABELS":
            # Special handling: flatten nested dict
            classes = set()
            from memory.hmsg.utils.constants import OPENVOCAB_MATTERPORT_LABELS

            for key, val in OPENVOCAB_MATTERPORT_LABELS.items():
                classes.add(key)
                classes.update(val)
            return list(classes)
        else:
            return _flatten_classes(classes_source)

    if obj_labels_key in CSV_LABEL_REGISTRY:
        csv_filename, _ = CSV_LABEL_REGISTRY[obj_labels_key]

        if label_feat_path is None:
            # Try to infer from module location: navigate to fsr_vln root then to memory/hmsg/labels
            current_dir = os.path.dirname(os.path.abspath(__file__))
            label_feat_path = os.path.normpath(
                os.path.join(current_dir, "../../memory/hmsg/labels")
            )

        csv_path = os.path.join(label_feat_path, csv_filename)
        if os.path.exists(csv_path):
            return _load_csv_classes(csv_path)
        else:
            import warnings

            warnings.warn(
                f"CSV file not found at {csv_path}, using empty labels list", stacklevel=2
            )
            return []

    available = list(LABEL_REGISTRY.keys()) + list(CSV_LABEL_REGISTRY.keys())
    raise ValueError(f"Unknown obj_labels key: {obj_labels_key}. Available: {available}")


def run_conversation_loop(
    hmsg: GraphRuntime,
    llm_enable: bool,
    slow_reasoning: bool,
    run_dir: str,
    ros_publisher: "QueryResultPublisher | None" = None,
    room_types: Optional[List[str]] = None,
    object_labels: Optional[List[str]] = None,
) -> List[dict]:
    """Run an interactive multi-turn navigation conversation using MotionAgent.

    Stage 1 (Conversation): The agent understands user intent, confirms inferred
    objects, and disambiguates between candidate locations across multiple turns.
    Controlled by ``llm_enable`` — when False the agent falls back to CLIP-only
    keyword matching.

    Stage 2 (Slow reasoning): VLM re-ranking applied to scene-graph candidates.
    Controlled independently by ``slow_reasoning`` — can be enabled without LLM and
    vice-versa for independent development of each module.

    Args:
        hmsg: Loaded scene graph instance.
        llm_enable: Enable LLM for query understanding and goal retrieval selection.
        slow_reasoning: Enable VLM slow-reasoning re-ranking (independent of llm_enable).
        run_dir: Directory to write per-session JSON logs.
        ros_publisher: Optional ROS 2 goal/marker publisher.

    Returns:
        List of result dicts (one per resolved navigation goal).
    """
    from memory.hmsg.tools.agents import NavigationAgent
    from memory.hmsg.utils.llm_utils import publish_navigation_goal

    agent = NavigationAgent(
        scene_graph=hmsg, room_types=room_types or [], object_labels=object_labels or []
    )
    history: List[Dict[str, str]] = []
    nav_state: Dict[str, Any] = {}
    all_results: List[dict] = []
    query_save_dir = os.path.join(run_dir, "query_session_logs")
    os.makedirs(query_save_dir, exist_ok=True)
    hmsg.curr_query_save_dir = query_save_dir

    llm_label = "on" if llm_enable else "off"
    vlm_label = "on" if slow_reasoning else "off"
    print(f"\nConversation mode — LLM: {llm_label}  |  VLM slow-reasoning: {vlm_label}")
    print("Talk to Motion. Type 'q' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if user_input.lower() == "q":
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        response, action = agent.process_message(
            user_input, history, nav_state, llm_enable=llm_enable, slow_reasoning=slow_reasoning
        )
        print(f"Motion: {response}\n")
        history.append({"role": "assistant", "content": response})

        if action is not None:
            print(f"  [NAVIGATE] {action['name']} → ({action['x']:.3f}, {action['y']:.3f})")
            publish_navigation_goal(action)

            if ros_publisher is not None:
                result_entries = [{"lidar_map_position": [action["x"], action["y"], action["z"]]}]
                ros_publisher.publish_results(result_entries, action["name"])

            session_log = {
                "query": user_input,
                "object": action["name"],
                "found": True,
                "action": action,
                "history": history[-10:],
            }
            log_path = os.path.join(
                run_dir,
                "query_session_logs",
                f"nav_{len(all_results):04d}.json",
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(session_log, f, ensure_ascii=False, indent=2)
            all_results.append(session_log)

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="../../config/visualize_graph",
    config_name="visualize_query_graph",
)
def main(params: DictConfig) -> None:
    """Run graph query visualization for a given profile.

    Profile selection (via Hydra override):
        python visualize_query_graph.py profiles=ic3f
        python visualize_query_graph.py profiles=ic7f main.spatial_reasoning_method=human_assign
        python visualize_query_graph.py profiles=custom main.graph_path=/path/to/graph

    LLM / VLM control (independent flags):
        python visualize_query_graph.py main.llm_enable=true   # LLM query understanding + goal selection (default)
        python visualize_query_graph.py main.llm_enable=false  # CLIP-only retrieval, no LLM
        python visualize_query_graph.py main.slow_reasoning=true   # VLM slow-reasoning re-ranking
        python visualize_query_graph.py main.slow_reasoning=false  # fast CLIP-only matching (default)
    """
    verbose: bool = bool(params.main.get("verbose", False))
    if not verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("memory").setLevel(logging.INFO)

    scene_id = params.main.scene_id
    spatial_reasoning_method = params.main.spatial_reasoning_method

    # Independent flags — either can be enabled without the other.
    llm_enable: bool = bool(params.main.get("llm_enable", False))
    slow_reasoning: bool = bool(params.main.get("slow_reasoning", False))

    if llm_enable:
        print("Checking LLM connection...")
        ok, reason = check_llm_connection(timeout=5.0)
        if not ok:
            print(f"[WARNING] LLM unreachable ({reason}). Falling back to CLIP-only mode.")
            llm_enable = False
        else:
            print("LLM connection OK.")

    print(f"LLM enable     : {llm_enable}")
    print(f"VLM slow-reason: {slow_reasoning}")
    visualize = not params.main.query_only

    # Build per-run output directory: <save_path>/<dataset>/<YYYYMMDD_HHMMSS>/
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(params.main.save_path, params.main.dataset, datetime_str)
    os.makedirs(run_dir, exist_ok=True)
    print(f"run_dir      : {run_dir}")

    # Write config.yaml — copy the profile yaml and append runtime fields
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if scene_id == "custom":
        profile_name = "custom"
    else:
        profile_name = scene_id.replace("icra_", "") if scene_id.startswith("icra_") else scene_id
    profile_yaml_src = os.path.normpath(
        os.path.join(script_dir, "../../config/visualize_graph/profiles", f"{profile_name}.yaml")
    )
    config_dst = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(profile_yaml_src):
        shutil.copy2(profile_yaml_src, config_dst)
        with open(config_dst, "a", encoding="utf-8") as _f:
            _f.write(f"spatial_reasoning_method: {spatial_reasoning_method}\n")
            _f.write(f"llm_enable: {llm_enable}\n")
            _f.write(f"slow_reasoning: {slow_reasoning}\n")
    else:
        # Fallback: serialize resolved params
        with open(config_dst, "w", encoding="utf-8") as _f:
            _f.write(OmegaConf.to_yaml(params.main))
    print(f"config saved : {config_dst}")

    # Build and load graph
    params.main.dataset_path = os.path.join(
        params.main.dataset_path, params.main.scene_id
    )  # Ensure scene_id is appended to dataset_path for loading
    hmsg = GraphRuntime(params)
    hmsg.load(params.main.graph_path)
    hmsg.curr_query_save_dir = run_dir

    # Connect to map visualizer and publish the map (skipped in query_only mode)
    map_visual = None
    if visualize:
        print("\nPublishing map to Rviz visualizer...")
        map_visual = MapVisual(hmsg)
        map_visual.publish_map()

    # Initialise ROS 2 goal/marker publisher (optional)
    ros_publisher = None
    if _ROS2_AVAILABLE:
        if not rclpy.ok():
            rclpy.init()
        ros_publisher = QueryResultPublisher()
    else:
        print("ROS 2 not available — skipping ROS publishers (install rclpy to enable)")

    # Room naming
    room_types = list(params.main.room_types) if params.main.room_types else []
    hmsg.generate_room_names(
        generate_method=params.main.room_generation_method,
        default_room_types=room_types,
    )
    if spatial_reasoning_method == "human_assign":
        hmsg.set_room_names(room_names=list(params.main.room_names_human_assign))

    # Load object labels from config for LLM query understanding
    object_labels = []
    if hasattr(params, "pipeline") and params.pipeline.get("obj_labels"):
        try:
            object_labels = _load_object_labels_from_config(params.pipeline.obj_labels)
            print(f"Loaded {len(object_labels)} object labels from '{params.pipeline.obj_labels}'")
        except Exception as exc:
            print(
                f"[WARNING] Failed to load object labels: {exc}. LLM will not use label constraints."
            )

    # Determine query list
    is_custom = params.main.dataset == "custom"
    if is_custom:
        # Interactive mode: multi-turn conversation via MotionAgent
        all_results = run_conversation_loop(
            hmsg,
            llm_enable,
            slow_reasoning,
            run_dir,
            ros_publisher=ros_publisher,
            room_types=room_types,
            object_labels=object_labels,
        )

        if not all_results:
            print("No navigation goals were resolved. Exiting.")
            # Tear down ROS 2 node and return without writing a summary
            if ros_publisher is not None:
                ros_publisher.destroy_node()
            if _ROS2_AVAILABLE and rclpy.ok():
                rclpy.shutdown()
            return

        summary_json = {
            "mode": "conversation",
            "llm_enable": llm_enable,
            "slow_reasoning": slow_reasoning,
            "resolved_goals": len(all_results),
            "goals": [r["object"] for r in all_results],
        }
    else:
        # Batch mode: run fixed query list with the traditional query loop
        profile = scene_id.replace("icra_", "") if scene_id.startswith("icra_") else scene_id
        queries = get_queries(profile, spatial_reasoning_method)

        if not queries:
            print("No queries to run. Exiting.")
            if ros_publisher is not None:
                ros_publisher.destroy_node()
            if _ROS2_AVAILABLE and rclpy.ok():
                rclpy.shutdown()
            return

        all_results, metric_sums = run_query_loop(
            hmsg,
            queries,
            slow_reasoning,
            run_dir,
            visualize=visualize,
            ros_publisher=ros_publisher,
        )

        avgs = _print_and_build_metrics(metric_sums, len(queries))
        summary_json = {
            "mode": "batch",
            "llm_enable": llm_enable,
            "slow_reasoning": slow_reasoning,
            "average_total_time": avgs["Total_Time"],
            "average_llm_parse_time": avgs["LLM_Parse_Time"],
            "average_fast_matching_time": avgs["FastMatching"],
            "average_obj_in_image_check_time": avgs["ObjectInImageCheck"],
            "average_vlm_rethinking_time": avgs["VLM_Rethinking"],
            "average_re_matching_time": avgs["Re_Matching"],
            "queries": [r["query"] for r in all_results],
        }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Run directory : {run_dir}")

    # Tear down ROS 2 node
    if ros_publisher is not None:
        ros_publisher.destroy_node()
    if _ROS2_AVAILABLE and rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
