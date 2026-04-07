"""Rendering utilities for scene graph and query result visualization.

Off-screen functions return a numpy (H, W, 3) RGB array suitable for st.image().
Plotly functions return a plotly.graph_objects.Figure for interactive 3-D viewing.
"""

import glob
import json
import logging
import os
from typing import Any, List, Optional, Tuple

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
        from collections import defaultdict

        import open3d as o3d
        import pyvista as pv
    except ImportError as e:
        logger.warning("render_full_scene_graph: missing dependency — %s", e)
        return None

    excluded_keywords = ("wall", "floor", "ceiling", "paneling", "banner", "overhang")
    excluded_names = (
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
        floor_infos[fid] = {
            k: fi[k]
            for k in fi
            if k in ("floor_id", "name", "rooms", "floor_height", "floor_zero_level", "vertices")
        }
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
        room_infos[rid] = {
            k: ri[k]
            for k in ri
            if k in ("room_id", "name", "floor_id", "room_height", "room_zero_level", "vertices")
        }
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
        object_infos[oid] = {
            k: oi[k]
            for k in oi
            if k in ("object_id", "name", "room_id", "object_height", "object_zero_level")
        }
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
    renderer.setup_camera(
        60.0, center, center + np.array([0, extent * 0.8, extent * 0.4]), [0, 1, 0]
    )

    img = renderer.render_to_image()
    rgb = np.asarray(img)  # (H, W, 3) uint8 RGB
    return rgb


# ---------------------------------------------------------------------------
# helpers shared by both Plotly functions
# ---------------------------------------------------------------------------


def _subsample_pcd(pts: np.ndarray, cols: np.ndarray, max_n: int):
    if len(pts) > max_n:
        idx = np.random.default_rng(42).choice(len(pts), max_n, replace=False)
        return pts[idx], cols[idx]
    return pts, cols


def _float_to_plotly_rgb(rgb_float: np.ndarray) -> List[str]:
    r = (np.clip(rgb_float[:, 0], 0.0, 1.0) * 255).astype(int)
    g = (np.clip(rgb_float[:, 1], 0.0, 1.0) * 255).astype(int)
    b = (np.clip(rgb_float[:, 2], 0.0, 1.0) * 255).astype(int)
    return [f"rgb({ri},{gi},{bi})" for ri, gi, bi in zip(r, g, b)]


_SCENE_LAYOUT = dict(
    xaxis=dict(showgrid=False, showticklabels=False, title="", zeroline=False),
    yaxis=dict(showgrid=False, showticklabels=False, title="", zeroline=False),
    zaxis=dict(showgrid=False, showticklabels=False, title="", zeroline=False),
    bgcolor="#0e1117",
)
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(font=dict(size=9, color="white"), bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
)


# ---------------------------------------------------------------------------
# Interactive full scene graph — Plotly
# ---------------------------------------------------------------------------


def render_full_scene_graph_plotly(
    graph_path: str,
    total_room_pts: int = 2000,
    max_pts_per_obj: int = 150,
    ceiling_filter_margin: float = 0.4,
    color_brightness_room: float = 1.2,
    initial_offset: Tuple[float, float, float] = (7.0, 2.5, 4.0),
    min_object_points: int = 10,
    room_height_offset: float = 3.5,
) -> Optional[Any]:
    """Return an interactive Plotly Figure for the full pre-scanned scene graph.

    Uses batched traces (one per category) to keep the JSON payload small and
    the browser responsive.  ``total_room_pts`` is shared across ALL rooms;
    ``max_pts_per_obj`` caps each individual object cloud.
    """
    try:
        from collections import defaultdict

        import open3d as o3d
        import plotly.graph_objects as go
    except ImportError as e:
        logger.warning("render_full_scene_graph_plotly: missing dependency — %s", e)
        return None

    excluded_keywords = ("wall", "floor", "ceiling", "paneling", "banner", "overhang")
    excluded_names = (
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
        floor_infos[fid] = {
            k: fi[k]
            for k in fi
            if k in ("floor_id", "name", "rooms", "floor_height", "floor_zero_level", "vertices")
        }
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
        room_infos[rid] = {
            k: ri[k]
            for k in ri
            if k in ("room_id", "name", "floor_id", "room_height", "room_zero_level", "vertices")
        }
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
        object_infos[oid] = {
            k: oi[k]
            for k in oi
            if k in ("object_id", "name", "room_id", "object_height", "object_zero_level")
        }
        pcd = _load_pcd(ply)
        np.asarray(pcd.points)[:] += floor_infos[fid]["viz_offset"]
        object_pcds[oid] = pcd

    for oid, oi in object_infos.items():
        rid = oi["room_id"]
        fid = room_infos[rid]["floor_id"]
        hier_topo[fid][rid].append(oid)

    rng = np.random.default_rng(42)
    traces = []

    # ------------------------------------------------------------------
    # Floor centroids — single batched trace
    # ------------------------------------------------------------------
    floor_centroids_viz = {}
    fx, fy, fz, ftxt = [], [], [], []
    for fid in hier_topo:
        c = np.mean(np.asarray(floor_pcds[fid].points), axis=0)
        floor_centroids_viz[fid] = c + floor_infos[fid]["viz_offset"] + np.array([0.0, 4.0, 0.0])
        fx.append(float(floor_centroids_viz[fid][0]))
        fy.append(float(floor_centroids_viz[fid][1]))
        fz.append(float(floor_centroids_viz[fid][2]))
        ftxt.append(floor_infos[fid].get("name", str(fid)))

    if fx:
        traces.append(
            go.Scatter3d(
                x=fx,
                y=fy,
                z=fz,
                mode="markers+text",
                marker=dict(size=12, color="orange"),
                text=ftxt,
                textposition="top center",
                textfont=dict(color="orange", size=11),
                name="Floors",
                showlegend=False,
            )
        )

    # ------------------------------------------------------------------
    # Room centroids — single batched trace
    # ------------------------------------------------------------------
    room_centroids_viz = {}
    rx, ry, rz, rtxt = [], [], [], []
    for rid, ri in room_infos.items():
        pts = np.asarray(room_pcds[rid].points)
        centroid = pts.mean(axis=0)
        room_centroids_viz[rid] = centroid + np.array([0.0, room_height_offset, 0.0])
        rx.append(float(room_centroids_viz[rid][0]))
        ry.append(float(room_centroids_viz[rid][1]))
        rz.append(float(room_centroids_viz[rid][2]))
        rtxt.append(ri.get("name", str(rid)))

    if rx:
        traces.append(
            go.Scatter3d(
                x=rx,
                y=ry,
                z=rz,
                mode="markers+text",
                marker=dict(size=7, color="cornflowerblue"),
                text=rtxt,
                textposition="top center",
                textfont=dict(color="cornflowerblue", size=9),
                name="Rooms",
                showlegend=False,
            )
        )

    # ------------------------------------------------------------------
    # Room point clouds — ALL rooms merged into ONE trace
    # ------------------------------------------------------------------
    all_room_pts = np.empty((0, 3), dtype=np.float32)
    all_room_cols = np.empty((0, 3), dtype=np.float32)
    for rid in room_infos:
        p = np.asarray(room_pcds[rid].points, dtype=np.float32)
        c = np.asarray(room_pcds[rid].colors, dtype=np.float32)
        all_room_pts = np.vstack([all_room_pts, p])
        all_room_cols = np.vstack([all_room_cols, c])

    if len(all_room_pts):
        sp, _ = _subsample_pcd(all_room_pts, all_room_cols, total_room_pts)
        traces.append(
            go.Scatter3d(
                x=sp[:, 0].tolist(),
                y=sp[:, 1].tolist(),
                z=sp[:, 2].tolist(),
                mode="markers",
                marker=dict(size=1.5, color="rgba(160,160,160,0.45)"),
                hoverinfo="none",
                showlegend=False,
                name="Room cloud",
            )
        )

    # ------------------------------------------------------------------
    # Object clouds & edge lines — all edges in ONE trace, objects batched
    # ------------------------------------------------------------------
    edge_x, edge_y, edge_z = [], [], []  # None-separated segments
    obj_cx, obj_cy, obj_cz, obj_txt = [], [], [], []  # centroid labels

    for oid, oi in object_infos.items():
        name_lower = oi["name"].lower()
        if any(kw in name_lower for kw in excluded_keywords):
            continue
        if oi["name"] in excluded_names:
            continue
        pcd = object_pcds[oid]
        if len(pcd.points) < min_object_points:
            continue

        rid = oi["room_id"]
        pts = np.asarray(pcd.points)
        centroid = pts.mean(axis=0)

        # Edge from room centroid to object centroid
        rc = room_centroids_viz[rid]
        edge_x += [float(rc[0]), float(centroid[0]), None]
        edge_y += [float(rc[1]), float(centroid[1]), None]
        edge_z += [float(rc[2]), float(centroid[2]), None]

        # Object centroid label
        obj_cx.append(float(centroid[0]))
        obj_cy.append(float(centroid[1]))
        obj_cz.append(float(centroid[2]))
        obj_txt.append(oi["name"])

        # Object point cloud (small budget per object)
        sub_pts, _ = _subsample_pcd(pts, pts, max_pts_per_obj)
        color_rgb = rng.random(3)
        rgb_str = f"rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})"
        traces.append(
            go.Scatter3d(
                x=sub_pts[:, 0].tolist(),
                y=sub_pts[:, 1].tolist(),
                z=sub_pts[:, 2].tolist(),
                mode="markers",
                marker=dict(size=2.5, color=rgb_str, opacity=0.9),
                name=oi["name"],
                hovertemplate=f"<b>{oi['name']}</b><extra></extra>",
                showlegend=True,
            )
        )

    if edge_x:
        traces.insert(
            2,
            go.Scatter3d(  # behind objects, after centroids
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color="rgba(180,180,180,0.3)", width=1),
                hoverinfo="none",
                showlegend=False,
                name="Edges",
            ),
        )

    if obj_cx:
        traces.append(
            go.Scatter3d(
                x=obj_cx,
                y=obj_cy,
                z=obj_cz,
                mode="text",
                text=obj_txt,
                textfont=dict(color="white", size=7),
                hoverinfo="none",
                showlegend=False,
                name="Labels",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=_SCENE_LAYOUT,
        uirevision="scene_graph",
        **_PLOTLY_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Interactive query result — Plotly
# ---------------------------------------------------------------------------


def render_query_result_plotly(
    room_pcd,
    obj_pcds: List,
    spheres: List,
    obj_names: Optional[List[str]] = None,
    max_pts_room: int = 1500,
    max_pts_obj: int = 500,
) -> Optional[Any]:
    """Return an interactive Plotly Figure for a query result.

    Kept compact: room background cap is ``max_pts_room`` total; each object
    cloud is capped at ``max_pts_obj``.  All sphere markers are a single
    batched trace.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        logger.warning("render_query_result_plotly: missing plotly — %s", e)
        return None

    _OBJ_COLORS = [
        "rgb(255,38,38)",
        "rgb(38,217,38)",
        "rgb(38,115,255)",
        "rgb(255,153,26)",
        "rgb(204,26,204)",
        "rgb(26,217,217)",
    ]

    traces = []

    # Background room cloud — single trace
    if room_pcd is not None and len(room_pcd.points) > 0:
        pts = np.asarray(room_pcd.points)
        sub_pts, _ = _subsample_pcd(pts, pts, max_pts_room)
        traces.append(
            go.Scatter3d(
                x=sub_pts[:, 0].tolist(),
                y=sub_pts[:, 1].tolist(),
                z=sub_pts[:, 2].tolist(),
                mode="markers",
                marker=dict(size=1.5, color="rgba(140,140,140,0.4)"),
                hoverinfo="none",
                name="Room",
                showlegend=False,
            )
        )

    # Object clouds — one trace per object (needed for legend / hover identity)
    for i, pcd in enumerate(obj_pcds):
        if len(pcd.points) == 0:
            continue
        pts = np.asarray(pcd.points)
        sub_pts, _ = _subsample_pcd(pts, pts, max_pts_obj)
        name = obj_names[i] if obj_names and i < len(obj_names) else f"Object {i + 1}"
        traces.append(
            go.Scatter3d(
                x=sub_pts[:, 0].tolist(),
                y=sub_pts[:, 1].tolist(),
                z=sub_pts[:, 2].tolist(),
                mode="markers",
                marker=dict(size=3, color=_OBJ_COLORS[i % len(_OBJ_COLORS)]),
                name=name,
                hovertemplate=f"<b>{name}</b><extra></extra>",
            )
        )

    # Sphere markers — single batched trace with text
    sx, sy, sz, stxt, scolors = [], [], [], [], []
    for i, sphere in enumerate(spheres):
        center = np.asarray(sphere.get_center())
        name = obj_names[i] if obj_names and i < len(obj_names) else f"Object {i + 1}"
        sx.append(float(center[0]))
        sy.append(float(center[1]))
        sz.append(float(center[2]))
        stxt.append(name)
        scolors.append(_OBJ_COLORS[i % len(_OBJ_COLORS)])

    if sx:
        traces.append(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="markers+text",
                marker=dict(size=10, color=scolors, symbol="circle"),
                text=stxt,
                textposition="top center",
                textfont=dict(color="white", size=10),
                name="Targets",
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=_SCENE_LAYOUT,
        uirevision="query_result",
        **_PLOTLY_LAYOUT,
    )
    return fig
