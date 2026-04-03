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
import os
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, Generator, List, Tuple

import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmark_queries import get_queries
from memory.hmsg.graph.graph import Graph

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


# ---------------------------------------------------------------------------
# Query loop
# ---------------------------------------------------------------------------


def run_query_loop(
    hmsg: Graph,
    queries: List[str],
    use_vlm: bool,
    run_dir: str,
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
        print(f"\n[Query] {query}")

        query_save_dir = os.path.join(run_dir)
        os.makedirs(query_save_dir, exist_ok=True)
        hmsg.curr_query_save_dir = query_save_dir

        t0 = time.time()
        floor, rooms, objects, res_dict = hmsg.query_hierarchy(query, top_k=5, use_vlm=use_vlm)
        query_time = time.time() - t0

        print(f"Elapsed: {query_time:.4f}s")
        floor_id_val = floor.floor_id if floor is not None else -1
        print(
            floor_id_val,
            [(r.room_id, r.name) for r in rooms],
            [o.object_id for o in objects],
        )

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
            print(f"  obj[{i}] '{obj.name}' in scene graph: {obj_center}")
            print(f"  obj[{i}] in lidar map:   {obj_center_in_map}")

            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            sphere.translate(obj_center)
            sphere.paint_uniform_color(color)

            obj_pcds.append(obj_pcd)
            spheres.append(sphere)

            scores = res_dict.get("object_scores", [])
            score_val = float(scores[i]) if i < len(scores) else None
            # Derive the actual floor from the object ID (format: "<floor>_<room>_<obj>")
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

        # --- Save combined scene (one PLY + one PNG) ---
        combined = room_pcd_combined
        for obj_pcd in obj_pcds:
            combined = combined + obj_pcd
        for sphere in spheres:
            combined = combined + sphere.sample_points_uniformly(number_of_points=500)

        pcd_path = os.path.join(query_save_dir, "scene.ply")
        png_path = os.path.join(query_save_dir, "scene.png")
        o3d.io.write_point_cloud(pcd_path, combined)
        print(f"  Saved {pcd_path}")

        visualize_and_save(room_pcd_combined, obj_pcds, spheres, save_path=png_path)

        # --- Write per-query results.json ---
        query_result = {
            "query": query,
            "object": obj_name,
            "found": found,
            "results": result_entries,
            "elapsed": f"{query_time:.4f}s",
        }
        results_json_path = os.path.join(query_save_dir, "results.json")
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
    """
    scene_id = params.main.scene_id
    spatial_reasoning_method = params.main.spatial_reasoning_method
    fast_slow_method = params.main.fast_slow_method
    use_vlm = params.main.use_vlm and fast_slow_method != "fast_match"

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
            _f.write(f"fast_slow_method: {fast_slow_method}\n")
            _f.write(f"use_vlm: {use_vlm}\n")
    else:
        # Fallback: serialize resolved params
        with open(config_dst, "w", encoding="utf-8") as _f:
            _f.write(OmegaConf.to_yaml(params.main))
    print(f"config saved : {config_dst}")

    # Build and load graph
    hmsg = Graph(params)
    hmsg.load_hmsg_graph(params.main.graph_path)
    hmsg.vln_result_dir = run_dir

    # Room naming
    room_types = list(params.main.room_types) if params.main.room_types else []
    hmsg.generate_room_names(
        generate_method=params.main.room_generation_method,
        default_room_types=room_types,
    )
    if spatial_reasoning_method == "human_assign":
        hmsg.set_room_names(room_names=list(params.main.room_names_human_assign))

    # Determine query list
    is_custom = scene_id == "custom"
    if is_custom:
        # In interactive mode, process each query immediately as it's entered
        all_results: List[dict] = []
        metric_sums: Dict[str, float] = {
            "Total_Time": 0.0,
            "FastMatching": 0.0,
            "ObjectInImageCheck": 0.0,
            "VLM_Rethinking": 0.0,
            "Re_Matching": 0.0,
            "LLM_Parse_Time": 0.0,
        }
        for query in _interactive_queries():
            results, sums = run_query_loop(hmsg, [query], use_vlm, run_dir)
            all_results.extend(results)
            for key in metric_sums:
                metric_sums[key] += sums.get(key, 0.0)

        if not all_results:
            print("No queries were run. Exiting.")
            return

        queries = [r["query"] for r in all_results]
    else:
        # Derive profile name from scene_id (e.g. "icra_ic3f" → "ic3f")
        profile = scene_id.replace("icra_", "") if scene_id.startswith("icra_") else scene_id
        queries = get_queries(profile, spatial_reasoning_method)

        if not queries:
            print("No queries to run. Exiting.")
            return

        all_results, metric_sums = run_query_loop(hmsg, queries, use_vlm, run_dir)

    # Compute and save metrics summary
    avgs = _print_and_build_metrics(metric_sums, len(queries))
    summary_json = {
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


if __name__ == "__main__":
    main()
