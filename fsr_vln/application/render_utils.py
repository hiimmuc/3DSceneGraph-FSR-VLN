"""Off-screen rendering utilities for scene graph and query result visualization.

Both functions return a numpy (H, W, 3) RGB array suitable for st.image().
"""

import glob
import json
import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Full scene graph — PyVista off-screen
# ---------------------------------------------------------------------------

def render_full_scene_graph(
    graph_path: str,
    point_size: int = 5,
    sphere_radius_floor: float = 0.5,
    sphere_radius_room: float = 0.15,
    ceiling_filter_margin: float = 0.4,
    color_brightness_room: float = 1.2,
    initial_offset: Tuple[float, float, float] = (7.0, 2.5, 4.0),
    min_object_points: int = 10,
    room_height_offset: float = 3.5,
    window_size: Tuple[int, int] = (960, 720),
) -> Optional[np.ndarray]:
    """Render the full pre-scanned scene graph to an RGB numpy array.

    Adapts visualize_graph.py for off-screen (headless) rendering so the result
    can be embedded directly in Streamlit via ``st.image()``.

    Args:
        graph_path: Path to the scene graph directory (contains floors/, rooms/, objects/).
        window_size: Output image size in pixels (width, height).

    Returns:
        RGB numpy array of shape (H, W, 3), or None if rendering fails.
    """
    try:
        import open3d as o3d
        import pyvista as pv
        from collections import defaultdict
    except ImportError as e:
        logger.warning("render_full_scene_graph: missing dependency — %s", e)
        return None

    excluded_keywords = ("wall", "floor", "ceiling", "paneling", "banner", "overhang")
    excluded_names = (
        "divider", "ledge", "pillar", "tape", "stairs", "door", "doors", "stair",
        "window", "glass", "railing", "glass doors", "whiteboard", "sliding door",
        "carpet", "picture",
    )

    def _load_json(path):
        with open(path) as f:
            return json.load(f)

    def _load_pcd(path):
        return o3d.io.read_point_cloud(path)

    init_offset = np.array(initial_offset)
    floor_pcds, floor_infos, hier_topo = {}, {}, defaultdict(dict)

    floors_ply = sorted(glob.glob(os.path.join(graph_path, "floors", "*.ply")))
    floors_json = sorted(glob.glob(os.path.join(graph_path, "floors", "*.json")))
    for counter, (ply, info) in enumerate(zip(floors_ply, floors_json)):
        fi = _load_json(info)
        fid = fi["floor_id"]
        floor_infos[fid] = {k: fi[k] for k in fi if k in (
            "floor_id", "name", "rooms", "floor_height", "floor_zero_level", "vertices"
        )}
        floor_infos[fid]["viz_offset"] = init_offset * counter
        for r_id in fi["rooms"]:
            hier_topo[fid][r_id] = []
        floor_pcds[fid] = _load_pcd(ply)

    room_pcds, room_infos = {}, {}
    rooms_ply = sorted(glob.glob(os.path.join(graph_path, "rooms", "*.ply")))
    rooms_json = sorted(glob.glob(os.path.join(graph_path, "rooms", "*.json")))
    for ply, info in zip(rooms_ply, rooms_json):
        ri = _load_json(info)
        rid = ri["room_id"]
        fid = ri["floor_id"]
        room_infos[rid] = {k: ri[k] for k in ri if k in (
            "room_id", "name", "floor_id", "room_height", "room_zero_level", "vertices"
        )}
        orig = _load_pcd(ply)
        xyz = np.asarray(orig.points)
        ceiling = ri["room_zero_level"] + ri["room_height"] - ceiling_filter_margin
        orig = orig.select_by_index(np.where(xyz[:, 1] < ceiling)[0])
        np.asarray(orig.points)[:] += floor_infos[fid]["viz_offset"]
        orig.colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(orig.colors) * color_brightness_room, 0.0, 1.0)
        )
        room_pcds[rid] = orig

    object_pcds, object_infos = {}, {}
    objs_ply = sorted(glob.glob(os.path.join(graph_path, "objects", "*.ply")))
    objs_json = sorted(glob.glob(os.path.join(graph_path, "objects", "*.json")))
    for ply, info in zip(objs_ply, objs_json):
        oi = _load_json(info)
        oid = oi["object_id"]
        rid = oi["room_id"]
        fid = room_infos[rid]["floor_id"]
        object_infos[oid] = {k: oi[k] for k in oi if k in (
            "object_id", "name", "room_id", "object_height", "object_zero_level"
        )}
        pcd = _load_pcd(ply)
        np.asarray(pcd.points)[:] += floor_infos[fid]["viz_offset"]
        object_pcds[oid] = pcd

    for oid, oi in object_infos.items():
        rid = oi["room_id"]
        fid = room_infos[rid]["floor_id"]
        hier_topo[fid][rid].append(oid)

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))

    # Floor centroids
    floor_centroids_viz = {}
    for fid in hier_topo:
        c = np.mean(np.asarray(floor_pcds[fid].points), axis=0)
        floor_centroids_viz[fid] = c + floor_infos[fid]["viz_offset"] + np.array([0.0, 4.0, 0.0])
        plotter.add_mesh(
            pv.Sphere(center=tuple(floor_centroids_viz[fid]), radius=sphere_radius_floor),
            color="orange",
        )

    # Room centroids
    room_centroids_viz = {}
    for rid in room_infos:
        c = np.mean(np.asarray(room_pcds[rid].points), axis=0)
        room_centroids_viz[rid] = c + np.array([0.0, room_height_offset, 0.0])
        plotter.add_mesh(
            pv.Sphere(center=tuple(room_centroids_viz[rid]), radius=sphere_radius_room),
            color="blue",
        )

    # Objects
    for oid, oi in object_infos.items():
        name_lower = oi["name"].lower()
        if any(kw in name_lower for kw in excluded_keywords):
            continue
        if oi["name"] in excluded_names:
            continue
        if len(object_pcds[oid].points) < min_object_points:
            continue

        rid = oi["room_id"]
        centroid = np.mean(np.asarray(object_pcds[oid].points), axis=0)

        plotter.add_mesh(
            pv.Line(tuple(room_centroids_viz[rid]), tuple(centroid)),
            line_width=1.5,
            opacity=0.5,
        )
        object_pcds[oid].paint_uniform_color(np.random.rand(3))
        cloud = pv.PolyData(np.asarray(object_pcds[oid].points))
        plotter.add_mesh(
            cloud,
            scalars=np.asarray(object_pcds[oid].colors),
            rgb=True,
            point_size=point_size,
            show_vertices=True,
        )
        plotter.add_point_labels(
            [centroid], [oi["name"]], font_size=8, point_color="blue", text_color="black"
        )

    screenshot: np.ndarray = plotter.screenshot(return_img=True)
    plotter.close()
    return screenshot


# ---------------------------------------------------------------------------
# Query result — Open3D off-screen
# ---------------------------------------------------------------------------

def render_query_result(
    room_pcd,
    obj_pcds: List,
    spheres: List,
    window_size: Tuple[int, int] = (960, 720),
) -> Optional[np.ndarray]:
    """Render query result (room + colored objects + spheres) to RGB numpy array.

    Adapts visualize_query_graph.py for headless rendering.

    Args:
        room_pcd: Open3D PointCloud — background room point cloud.
        obj_pcds: List of Open3D PointClouds — one per matched object (pre-colored).
        spheres: List of Open3D TriangleMesh spheres marking object centers.
        window_size: Output image size (width, height).

    Returns:
        RGB numpy array of shape (H, W, 3), or None if rendering fails.
    """
    try:
        import open3d as o3d
        import open3d.visualization.rendering as rendering
    except ImportError as e:
        logger.warning("render_query_result: missing open3d — %s", e)
        return None

    W, H = window_size
    renderer = rendering.OffscreenRenderer(W, H)
    scene = renderer.scene
    scene.set_background([0.1, 0.1, 0.1, 1.0])

    mat_room = rendering.MaterialRecord()
    mat_room.shader = "defaultUnlit"
    mat_room.point_size = 3.0

    mat_obj = rendering.MaterialRecord()
    mat_obj.shader = "defaultUnlit"
    mat_obj.point_size = 6.0

    mat_sphere = rendering.MaterialRecord()
    mat_sphere.shader = "defaultLit"

    try:
        scene.add_geometry("room_pcd", room_pcd, mat_room)
    except Exception as e:
        logger.debug("Could not add room_pcd: %s", e)

    for i, pcd in enumerate(obj_pcds):
        try:
            scene.add_geometry(f"obj_{i}", pcd, mat_obj)
        except Exception as e:
            logger.debug("Could not add obj_%d: %s", i, e)

    for i, mesh in enumerate(spheres):
        try:
            scene.add_geometry(f"sphere_{i}", mesh, mat_sphere)
        except Exception as e:
            logger.debug("Could not add sphere_%d: %s", i, e)

    # Auto-fit camera to visible geometry
    bounds = scene.bounding_box
    center = bounds.get_center()
    extent = np.linalg.norm(bounds.get_max_bound() - bounds.get_min_bound())
    renderer.setup_camera(60.0, center, center + np.array([0, extent * 0.8, extent * 0.4]), [0, 1, 0])

    img = renderer.render_to_image()
    rgb = np.asarray(img)  # (H, W, 3) uint8 RGB
    return rgb
