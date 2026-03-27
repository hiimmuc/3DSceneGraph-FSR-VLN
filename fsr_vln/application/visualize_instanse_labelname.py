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

# pylint: disable=missing-docstring
import time

import matplotlib.pyplot as plt
import numpy as np

# Load point cloud
import open3d as o3d

# For creating the GUI and rendering the scene
import open3d.visualization.gui as gui  # type: ignore
import open3d.visualization.rendering as rendering  # type: ignore


def exit_after_delay():
    time.sleep(5)  # Wait 5 seconds
    gui.Application.instance.quit()  # Quit the application


# Function: display point cloud with category labels


def show_point_cloud_with_labels(pcd, point_labels, cluster_labels):
    """
    Display point cloud with category label annotations.

    Args:
        pcd (open3d.geometry.PointCloud): input point cloud data.
        labels (numpy.ndarray): cluster labels for the point cloud; each point corresponds to one cluster index.
        cluster_names (dict): mapping from cluster index to category name, used for displaying labels on the point cloud.

    Returns:
        None
    """
    # Initialize GUI application
    app = gui.Application.instance
    app.initialize()

    # Create window and scene
    window = app.create_window("mapvln raw existing object instances", 1024, 768)
    # Create a SceneWidget and add it to the window
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    # Set scene background and lighting
    scene.scene.set_background([1, 1, 1, 1])  # White background
    scene.scene.add_geometry("pcd", pcd, rendering.MaterialRecord())  # Add point cloud to scene

    # Iterate over each cluster label and add the corresponding text label to the point cloud
    for i in range(max(point_labels) + 1):
        cluster_idx = np.where(point_labels == i)[
            0
        ]  # Find indices belonging to the current cluster
        if len(cluster_idx) == 0:
            continue
        cluster_points = np.asarray(pcd.points)[cluster_idx]
        center = cluster_points.mean(axis=0)  # Compute centroid of the current cluster
        # Get cluster name, or fall back to cluster_{i}
        label = cluster_labels.get(i, f"cluster_{i}")
        # Add 3D text label
        scene.add_3d_label(center, label)

    # Set the camera center to [0, 0, 0]
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(3, 1)
    # Get the axis-aligned bounding box of the point cloud
    bounding_box = pcd.get_axis_aligned_bounding_box()
    # Set camera FOV to 60 degrees and align it to the point cloud bounding box
    scene.setup_camera(60.0, bounding_box, center)

    # Launch the GUI application and display the window
    app.run()


# Load point cloud and run clustering
pcd = o3d.io.read_point_cloud(
    "/mnt/disk2/hovsg/HOV-SG/data/scannet/scene_graph/scannet/scene0378_00/full_pcd.ply"
)
labels = np.array(
    pcd.cluster_dbscan(
        eps=0.05,  # Clustering radius
        min_points=50,  # Minimum number of points
        print_progress=True,
    )
)  # Show progress

# Colors
max_label = labels.max()
print("max_label: ", max_label)
colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
print(colors)
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Cluster label-to-name mapping
cluster_names = {0: "chair", 1: "table", 2: "sofa"}  # Extend as needed based on clustering results

# Display
# Start a thread to quit the application after 5 seconds
# threading.Thread(target=exit_after_delay).start()
show_point_cloud_with_labels(
    pcd, labels, cluster_names
)  # Keep variable names consistent when calling
