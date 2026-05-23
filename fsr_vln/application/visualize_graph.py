"""Scene graph visualization with configurable parameters and CLI argument support."""

import argparse
import glob
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple

import hydra
import numpy as np
import open3d as o3d
import pyvista as pv
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""

    graph_path: str
    point_size: int = 5
    sphere_radius_floor: float = 0.5
    sphere_radius_room: float = 0.15
    ceiling_filter_margin: float = 0.4
    color_brightness_room: float = 1.2
    excluded_object_keywords: Tuple[str, ...] = (
        "wall",
        "floor",
        "ceiling",
        "paneling",
        "banner",
        "overhang",
    )
    excluded_object_names: Tuple[str, ...] = (
        "divider",
        "ledge",
        "pillar",
        "tape",
        "stairs",
        "door",
        "doors",
        "stair",
        "window",
        "glass",
        "railing",
        "glass doors",
        "whiteboard",
        "sliding door",
        "carpet",
        "picture",
    )
    initial_offset: Tuple[float, float, float] = (7.0, 2.5, 4.0)
    min_object_points: int = 10
    room_height_offset: float = 3.5


def _load_json(filepath: str) -> Dict:
    """Load JSON file safely."""
    with open(filepath, "r") as fp:
        return json.load(fp)


def _load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load point cloud file."""
    return o3d.io.read_point_cloud(filepath)


def _extract_metadata(data: Dict, keys: Tuple[str, ...]) -> Dict:
    """Extract specific keys from metadata dictionary."""
    return {k: v for k, v in data.items() if k in keys}


def _load_floors(graph_path: str, config: VisualizationConfig) -> Tuple[Dict, Dict, Dict]:
    """Load and process floor data."""
    floors_ply_paths = sorted(glob.glob(os.path.join(graph_path, "floors", "*.ply")))
    floors_info_paths = sorted(glob.glob(os.path.join(graph_path, "floors", "*.json")))

    floor_pcds = {}
    floor_infos = {}
    hier_topo = defaultdict(dict)
    init_offset = np.array(config.initial_offset)

    for counter, (ply_path, info_path) in enumerate(zip(floors_ply_paths, floors_info_paths)):
        floor_info = _load_json(info_path)
        floor_id = floor_info["floor_id"]

        floor_infos[floor_id] = _extract_metadata(
            floor_info,
            ("floor_id", "name", "rooms", "floor_height", "floor_zero_level", "vertices"),
        )
        floor_infos[floor_id]["viz_offset"] = init_offset * counter

        for r_id in floor_info["rooms"]:
            hier_topo[floor_id][r_id] = []

        floor_pcds[floor_id] = _load_point_cloud(ply_path)

    return floor_pcds, floor_infos, hier_topo


def _load_rooms(
    graph_path: str, floor_infos: Dict, config: VisualizationConfig
) -> Tuple[Dict, Dict]:
    """Load and process room data."""
    rooms_ply_paths = sorted(glob.glob(os.path.join(graph_path, "rooms", "*.ply")))
    rooms_info_paths = sorted(glob.glob(os.path.join(graph_path, "rooms", "*.json")))

    room_pcds = {}
    room_infos = {}

    for ply_path, info_path in zip(rooms_ply_paths, rooms_info_paths):
        room_info = _load_json(info_path)
        room_id = room_info["room_id"]
        floor_id = room_info["floor_id"]

        room_infos[room_id] = _extract_metadata(
            room_info,
            ("room_id", "name", "floor_id", "room_height", "room_zero_level", "vertices"),
        )

        # Load and filter point cloud
        orig_cloud = _load_point_cloud(ply_path)
        orig_cloud_xyz = np.asarray(orig_cloud.points)
        ceiling_level = (
            room_infos[room_id]["room_zero_level"]
            + room_infos[room_id]["room_height"]
            - config.ceiling_filter_margin
        )
        below_ceiling = orig_cloud_xyz[:, 1] < ceiling_level
        room_pcds[room_id] = orig_cloud.select_by_index(np.where(below_ceiling)[0])

        # Apply visualization offset
        cloud_xyz = np.asarray(room_pcds[room_id].points)
        cloud_xyz += floor_infos[floor_id]["viz_offset"]

        # Enhance colors
        room_pcds[room_id].colors = o3d.utility.Vector3dVector(
            np.clip(np.array(room_pcds[room_id].colors) * config.color_brightness_room, 0.0, 1.0)
        )

    return room_pcds, room_infos


def _load_objects(graph_path: str, floor_infos: Dict, room_infos: Dict) -> Tuple[Dict, Dict, Dict]:
    """Load and process object data."""
    objects_ply_paths = sorted(glob.glob(os.path.join(graph_path, "objects", "*.ply")))
    objects_info_paths = sorted(glob.glob(os.path.join(graph_path, "objects", "*.json")))

    object_pcds = {}
    object_infos = {}
    object_feats = {}

    for ply_path, info_path in zip(objects_ply_paths, objects_info_paths):
        obj_info = _load_json(info_path)
        obj_id = obj_info["object_id"]
        room_id = obj_info["room_id"]
        floor_id = room_infos[room_id]["floor_id"]

        object_infos[obj_id] = _extract_metadata(
            obj_info, ("object_id", "name", "room_id", "object_height", "object_zero_level")
        )
        object_feats[obj_id] = np.asarray(obj_info["embedding"])

        # Load point cloud with offset
        object_pcds[obj_id] = _load_point_cloud(ply_path)
        cloud_xyz = np.asarray(object_pcds[obj_id].points)
        cloud_xyz += floor_infos[floor_id]["viz_offset"]

    return object_pcds, object_infos, object_feats


def _should_include_object(
    obj_info: Dict, obj_pcd: o3d.geometry.PointCloud, config: VisualizationConfig
) -> bool:
    """Check if object should be included in visualization."""
    name_lower = obj_info["name"].lower()

    # Exclude by keyword
    if any(kw in name_lower for kw in config.excluded_object_keywords):
        return False

    # Exclude by specific name
    if obj_info["name"] in config.excluded_object_names:
        return False

    # Exclude by point count
    if len(obj_pcd.points) < config.min_object_points:
        return False

    return True


def _visualize_hierarchy(
    plotter: pv.Plotter,
    floor_pcds: Dict,
    floor_infos: Dict,
    room_pcds: Dict,
    room_infos: Dict,
    object_pcds: Dict,
    object_infos: Dict,
    hier_topo: Dict,
    config: VisualizationConfig,
) -> None:
    """Add hierarchy visualization to plotter."""
    # Visualize floor centroids
    floor_centroids = {
        fid: np.mean(np.asarray(floor_pcds[fid].points), axis=0) for fid in hier_topo.keys()
    }
    floor_centroids_viz = {
        fid: floor_centroids[fid] + floor_infos[fid]["viz_offset"] + np.array([0.0, 4.0, 0.0])
        for fid in hier_topo.keys()
    }
    for fid, centroid in floor_centroids_viz.items():
        plotter.add_mesh(
            pv.Sphere(center=tuple(centroid), radius=config.sphere_radius_floor), color="orange"
        )

    # Visualize room centroids
    room_centroids = {
        rid: np.mean(np.asarray(room_pcds[rid].points), axis=0) for rid in room_infos.keys()
    }
    room_centroids_viz = {
        rid: room_centroids[rid] + np.array([0.0, config.room_height_offset, 0.0])
        for rid in room_infos.keys()
    }
    for rid, centroid in room_centroids_viz.items():
        plotter.add_mesh(
            pv.Sphere(center=tuple(centroid), radius=config.sphere_radius_room), color="blue"
        )

    # Visualize objects and connections
    for obj_id, obj_info in object_infos.items():
        if not _should_include_object(obj_info, object_pcds[obj_id], config):
            continue

        logger.info(f"Including object: {obj_info['name']}")

        room_id = obj_info["room_id"]
        obj_centroid = np.mean(np.asarray(object_pcds[obj_id].points), axis=0)

        # Draw connection from room to object
        plotter.add_mesh(
            pv.Line(tuple(room_centroids_viz[room_id]), tuple(obj_centroid)),
            line_width=1.5,
            opacity=0.5,
        )

        # Add object point cloud
        object_pcds[obj_id].paint_uniform_color(np.random.rand(3))
        cloud_xyz = np.asarray(object_pcds[obj_id].points)
        cloud = pv.PolyData(cloud_xyz)

        plotter.add_mesh(
            cloud,
            scalars=np.asarray(object_pcds[obj_id].colors),
            rgb=True,
            point_size=config.point_size,
            show_vertices=True,
        )

        plotter.add_point_labels(
            [obj_centroid], [obj_info["name"]], font_size=8, point_color="blue", text_color="black"
        )


def _merge_configs(config_dict: DictConfig, cli_args: Dict) -> VisualizationConfig:
    """Merge config file with CLI arguments."""
    # merged = OmegaConf.to_container(config_dict)

    # # Override with CLI arguments if provided
    # for key, value in cli_args.items():
    #     if value is not None:
    #         merged[key] = value

    # return VisualizationConfig(**merged)

    # Chỉ lấy các field VisualizationConfig cần

    main_cfg = config_dict.main

    merged = {
        "graph_path": main_cfg.graph_path,
        "point_size": 5,
        "min_object_points": 10,
        "sphere_radius_floor": 0.5,
        "sphere_radius_room": 0.15,
        "ceiling_filter_margin": 0.4,
        "color_brightness_room": 1.2,
        "room_height_offset": 3.5,
    }

    # CLI override
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value

    return VisualizationConfig(**merged)

@hydra.main(
    version_base=None, config_path="../config/visualize_graph", config_name="visualize_query_graph"
)
def main(params: DictConfig) -> None:
    """Main visualization function with CLI argument support."""
    # Parse CLI arguments for config override
    # parser = argparse.ArgumentParser(description="Visualize scene graph")
    # parser.add_argument("--graph-path", type=str, default=None, help="Path to graph directory")
    # parser.add_argument("--point-size", type=int, default=None, help="Point size in visualization")
    # parser.add_argument(
    #     "--min-object-points", type=int, default=None, help="Minimum points for object inclusion"
    # )
    # args = parser.parse_args()

    # # Merge config with CLI arguments
    # cli_overrides = {
    #     "graph_path": args.graph_path,
    #     "point_size": args.point_size,
    #     "min_object_points": args.min_object_points,
    # }\


    cli_overrides = {}
    config = _merge_configs(params, cli_overrides)

    logger.info(f"Using config: {config}")

    # Initialize plotter
    plotter = pv.Plotter()

    # Load data
    logger.info("Loading floors...")
    floor_pcds, floor_infos, hier_topo = _load_floors(config.graph_path, config)

    logger.info("Loading rooms...")
    room_pcds, room_infos = _load_rooms(config.graph_path, floor_infos, config)

    logger.info("Loading objects...")
    object_pcds, object_infos, object_feats = _load_objects(
        config.graph_path, floor_infos, room_infos
    )

    # Update hierarchy
    for obj_id, obj_info in object_infos.items():
        room_id = obj_info["room_id"]
        floor_id = room_infos[room_id]["floor_id"]
        hier_topo[floor_id][room_id].append(obj_id)

    # Visualize
    logger.info("Building visualization...")
    _visualize_hierarchy(
        plotter,
        floor_pcds,
        floor_infos,
        room_pcds,
        room_infos,
        object_pcds,
        object_infos,
        hier_topo,
        config,
    )

    logger.info("Showing visualization...")
    plotter.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
