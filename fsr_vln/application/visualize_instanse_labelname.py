"""
Point cloud visualization with instance-level label annotations.

LICENSE:
This project as a whole is licensed under the Apache License, Version 2.0.

THIRD-PARTY LICENSES:
See project LICENSE file for details on third-party dependencies and their licenses.
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for point cloud clustering."""

    pcd_path: str
    eps: float = 0.05
    min_points: int = 50
    cmap_name: str = "tab20"
    cluster_names: Optional[Dict[int, str]] = None


@dataclass
class VisualizationConfig:
    """Configuration for GUI visualization."""

    window_title: str = "Point Cloud with Instance Labels"
    window_width: int = 1024
    window_height: int = 768
    bg_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    font_size: int = 8
    camera_fov: float = 60.0


def _load_point_cloud(filepath: str) -> o3d.geometry.PointCloud:
    """Load point cloud from file."""
    logger.info(f"Loading point cloud from {filepath}")
    return o3d.io.read_point_cloud(filepath)


def _cluster_point_cloud(pcd: o3d.geometry.PointCloud, config: ClusteringConfig) -> np.ndarray:
    """Perform DBSCAN clustering on point cloud."""
    logger.info(f"Clustering with eps={config.eps}, min_points={config.min_points}")
    labels = np.array(
        pcd.cluster_dbscan(
            eps=config.eps,
            min_points=config.min_points,
            print_progress=True,
        )
    )
    return labels


def _colorize_clusters(
    pcd: o3d.geometry.PointCloud, labels: np.ndarray, cmap_name: str = "tab20"
) -> None:
    """Apply colormap to clusters."""
    max_label = labels.max()
    logger.info(f"Found {max_label + 1} clusters")

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = 0  # Noise points -> black

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])


def _get_cluster_names(
    labels: np.ndarray, provided_names: Optional[Dict[int, str]] = None
) -> Dict[int, str]:
    """Generate cluster names from labels and provided mapping."""
    cluster_names = {}
    max_label = labels.max()

    for i in range(max_label + 1):
        if provided_names and i in provided_names:
            cluster_names[i] = provided_names[i]
        else:
            cluster_names[i] = f"cluster_{i}"

    return cluster_names


def _setup_scene(
    window: gui.Window,
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    cluster_names: Dict[int, str],
    config: VisualizationConfig,
) -> gui.SceneWidget:
    """Setup the visualization scene."""
    # Create scene widget
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    # Set background and add point cloud
    scene.scene.set_background(config.bg_color)
    scene.scene.add_geometry("pcd", pcd, rendering.MaterialRecord())

    # Add 3D labels for each cluster
    logger.info("Adding cluster labels to scene")
    for i in range(labels.max() + 1):
        cluster_idx = np.where(labels == i)[0]
        if len(cluster_idx) == 0:
            continue

        cluster_points = np.asarray(pcd.points)[cluster_idx]
        center = cluster_points.mean(axis=0)
        label = cluster_names.get(i, f"cluster_{i}")

        scene.add_3d_label(center, label)

    # Configure camera
    bounding_box = pcd.get_axis_aligned_bounding_box()
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1)
    scene.setup_camera(config.camera_fov, bounding_box, center)

    return scene


def _run_visualization(window: gui.Window, config: VisualizationConfig) -> None:
    """Launch the visualization GUI."""
    app = gui.Application.instance
    app.run()


def visualize_point_cloud_with_labels(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    cluster_names: Optional[Dict[int, str]] = None,
    vis_config: Optional[VisualizationConfig] = None,
) -> None:
    """
    Display point cloud with cluster labels in GUI.

    Args:
        pcd: Open3D point cloud object
        labels: Cluster assignments for each point
        cluster_names: Mapping from cluster ID to name
        vis_config: Visualization configuration
    """
    if vis_config is None:
        vis_config = VisualizationConfig()

    if cluster_names is None:
        cluster_names = {}

    # Initialize GUI
    app = gui.Application.instance
    app.initialize()

    # Create window
    window = app.create_window(
        vis_config.window_title, vis_config.window_width, vis_config.window_height
    )

    # Setup scene
    _setup_scene(window, pcd, labels, cluster_names, vis_config)

    # Run visualization
    _run_visualization(window, vis_config)


def _merge_cli_args(
    clustering_config: ClusteringConfig,
    vis_config: VisualizationConfig,
    cli_args: argparse.Namespace,
) -> Tuple[ClusteringConfig, VisualizationConfig]:
    """Merge CLI arguments into configs."""
    # Update clustering config
    if cli_args.pcd_path:
        clustering_config.pcd_path = cli_args.pcd_path
    if cli_args.eps is not None:
        clustering_config.eps = cli_args.eps
    if cli_args.min_points is not None:
        clustering_config.min_points = cli_args.min_points
    if cli_args.cmap:
        clustering_config.cmap_name = cli_args.cmap

    # Update visualization config
    if cli_args.title:
        vis_config.window_title = cli_args.title
    if cli_args.width is not None:
        vis_config.window_width = cli_args.width
    if cli_args.height is not None:
        vis_config.window_height = cli_args.height

    return clustering_config, vis_config


def main() -> None:
    """Main entry point with CLI argument support."""
    parser = argparse.ArgumentParser(
        description="Visualize point cloud with instance-level labels"
    )

    # Clustering arguments
    parser.add_argument(
        "--pcd-path",
        type=str,
        default="outputs/built_graphs/horizon/LivingRoomDataset_20260330/full_pcd.ply",
        help="Path to point cloud file",
    )
    parser.add_argument(
        "--eps", type=float, default=None, help="DBSCAN epsilon (clustering radius)"
    )
    parser.add_argument(
        "--min-points", type=int, default=None, help="Minimum points for cluster (DBSCAN)"
    )
    parser.add_argument(
        "--cmap", type=str, default=None, help="Matplotlib colormap name (default: tab20)"
    )

    # Visualization arguments
    parser.add_argument("--title", type=str, default=None, help="Window title")
    parser.add_argument("--width", type=int, default=None, help="Window width in pixels")
    parser.add_argument("--height", type=int, default=None, help="Window height in pixels")

    args = parser.parse_args()

    # Create configs
    clustering_config = ClusteringConfig(pcd_path=args.pcd_path)
    vis_config = VisualizationConfig()

    # Merge with CLI arguments
    clustering_config, vis_config = _merge_cli_args(clustering_config, vis_config, args)

    logger.info(f"Clustering config: {clustering_config}")
    logger.info(f"Visualization config: {vis_config}")

    # Load and process point cloud
    pcd = _load_point_cloud(clustering_config.pcd_path)
    labels = _cluster_point_cloud(pcd, clustering_config)
    _colorize_clusters(pcd, labels, clustering_config.cmap_name)

    # Generate and display cluster names
    cluster_names = _get_cluster_names(labels)
    logger.info(f"Cluster names: {cluster_names}")

    # visualize
    visualize_point_cloud_with_labels(pcd, labels, cluster_names, vis_config)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
