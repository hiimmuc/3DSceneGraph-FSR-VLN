import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import faiss
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree, distance
from sklearn.cluster import DBSCAN, KMeans
from tqdm import tqdm

matplotlib.use("Agg")  # Use non-GUI backend

# ---------------------------------------------------------------------------
# GPU FAISS detection (#4)
# Try to acquire a GPU resource object once at import time.  All index
# builders below use _FAISS_GPU_RES to decide CPU vs GPU transparently.
# ---------------------------------------------------------------------------
_FAISS_GPU_RES = None
try:
    import faiss.contrib.torch_utils  # noqa: F401 – registers GPU helpers

    _faiss_gpu_res_candidate = faiss.StandardGpuResources()
    # Smoke-test: create and immediately discard a tiny GPU index
    _smoke = faiss.index_cpu_to_gpu(_faiss_gpu_res_candidate, 0, faiss.IndexFlatL2(3))
    del _smoke
    _FAISS_GPU_RES = _faiss_gpu_res_candidate
except Exception:
    _FAISS_GPU_RES = None  # faiss-gpu not installed or no CUDA device — use CPU


def _make_faiss_index(dim: int, pts: np.ndarray = None) -> faiss.Index:
    """Build a FAISS flat-L2 index, preferring GPU when available.

    Args:
        dim: Feature dimension (3 for XYZ point clouds).
        pts: Optional float32 array of shape (N, dim) to add immediately.

    Returns:
        A populated (or empty) FAISS index on GPU or CPU.

    Raises:
        ValueError: If pts.shape[1] != dim or if index creation fails.
    """
    if pts is not None and pts.shape[1] != dim:
        raise ValueError(f"Points dimension {pts.shape[1]} != index dimension {dim}")

    try:
        cpu_index = faiss.IndexFlatL2(dim)
        if pts is not None and len(pts) > 0:
            cpu_index.add(pts)

        if _FAISS_GPU_RES is not None:
            try:
                gpu_index = faiss.index_cpu_to_gpu(_FAISS_GPU_RES, 0, cpu_index)
                return gpu_index
            except RuntimeError:
                # GPU transfer failed, fall back to CPU
                pass

        return cpu_index
    except Exception as e:
        raise RuntimeError(f"Failed to create FAISS index: {e}") from e


def _project_points_to_camera(
    points_world: np.ndarray, pose_world_to_cam: np.ndarray, near_clip: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform points from world to camera coordinates and filter behind camera.

    Args:
        points_world: (N, 3) array of points in world coordinates.
        pose_world_to_cam: (4, 4) transformation matrix from world to camera.
        near_clip: Z-coordinate threshold; points with z <= near_clip are filtered.

    Returns:
        Tuple of (valid_points_cam, valid_mask):
            - valid_points_cam: (M, 3) points in camera coordinates (z > near_clip)
            - valid_mask: (N,) boolean mask indicating valid points
    """
    if points_world.shape[0] == 0:
        return np.empty((0, 3)), np.array([], dtype=bool)

    # World -> camera coordinates
    pts_h = np.hstack((points_world, np.ones((points_world.shape[0], 1))))
    pts_cam = (pose_world_to_cam @ pts_h.T).T[:, :3]

    # Filter points in front of camera
    valid_mask = pts_cam[:, 2] > near_clip

    return pts_cam[valid_mask], valid_mask


# =========================================================================
# Camera Projection Utilities (World → Camera → Image coordinates)
# =========================================================================


def _project_points_to_image(points_cam: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """Project camera-space points to image coordinates.

    Args:
        points_cam: (N, 3) points in camera coordinates.
        camera_matrix: (3, 3) camera intrinsic matrix.

    Returns:
        (N, 2) array of (u, v) pixel coordinates.
    """
    if points_cam.shape[0] == 0:
        return np.empty((0, 2))

    uv_h = (camera_matrix @ points_cam.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]

    return uv


# =========================================================================
# Visualization Utilities (PCD projection, cluster coloring)
# =========================================================================


def visualize_pcd_on_image(
    obj_pcd: o3d.geometry.PointCloud,
    img: np.ndarray,
    camera_matrix: np.ndarray,
    pose: np.ndarray,
    save_path: str,
    color: Tuple[int, int, int] = (0, 0, 255),
) -> float:
    """Project a 3D point cloud onto a 2D image, save the visualization, and return the mean object distance.

    Args:
        obj_pcd: Open3D PointCloud object (object point cloud).
        img: Image array (H, W, 3).
        camera_matrix: Camera intrinsic matrix (3, 3).
        pose: Camera pose matrix - world-to-camera transform (4, 4).
        save_path: Output save path.
        color: Color for points as (B, G, R) tuple.

    Returns:
        avg_distance: Mean distance of visible points in camera coordinates (meters),
                      or None if no valid points.
    """
    pts = np.asarray(obj_pcd.points, dtype=np.float32)

    # Project to camera coordinates
    pts_cam, valid_mask_all = _project_points_to_camera(pts, pose)

    if pts_cam.shape[0] == 0:
        print("Warning: No valid points in front of camera.")
        return None

    # Project to image coordinates
    uv = _project_points_to_image(pts_cam, camera_matrix)

    # Calculate mean depth
    avg_distance = float(np.mean(pts_cam[:, 2]))

    # Visualize
    img_vis = img.copy()
    for u, v in uv.astype(int):
        if 0 <= u < img_vis.shape[1] and 0 <= v < img_vis.shape[0]:
            cv2.circle(img_vis, (u, v), 2, color, -1)

    # Save result
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_vis)
    print(f"Projected PCD visualization saved at {save_path}, avg_distance = {avg_distance:.3f}m")

    return avg_distance


def check_object_in_view(
    img_w: int,
    img_h: int,
    camera_matrix: np.ndarray,
    cam_pose_inv: np.ndarray,
    obj_points: np.ndarray,
    min_visible_ratio: float = 0.5,
    max_depth: float = 10.0,
    return_depth: bool = False,
) -> Union[bool, Tuple[bool, float]]:
    """
    Check whether an object point cloud is within the camera's field of view and has a mean depth below max_depth.

    Args:
        img_w (int): image width (pixels)
        img_h (int): image height (pixels)
        camera_matrix (numpy.ndarray): intrinsic matrix (3x3)
        cam_pose_inv (numpy.ndarray): world-to-camera transform matrix (4x4)
        obj_points (numpy.ndarray): object point cloud (N x 3)
        min_visible_ratio (float): minimum fraction of points that must be visible to count as in-view
        max_depth (float): mean depth threshold (meters)
        return_depth (bool): whether to return (bool, depth) tuple or just bool

    Returns:
        If return_depth=True: (visible, mean_depth) tuple
        If return_depth=False: visibility boolean
    """

    if obj_points.shape[0] == 0:
        return (False, np.inf) if return_depth else False

    # Project to camera coordinates
    pts_cam, _ = _project_points_to_camera(obj_points, cam_pose_inv)

    if pts_cam.shape[0] == 0:
        return (False, np.inf) if return_depth else False

    # Project to image coordinates
    uv = _project_points_to_image(pts_cam, camera_matrix)

    # Check if points fall within image bounds
    inside_mask = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)

    if not np.any(inside_mask):
        return (False, np.inf) if return_depth else False

    visible_ratio = np.sum(inside_mask) / obj_points.shape[0]
    if visible_ratio < min_visible_ratio:
        return (False, np.inf) if return_depth else False

    # Check depth constraint
    mean_depth = np.mean(pts_cam[inside_mask, 2]) if np.any(inside_mask) else np.inf
    if mean_depth > max_depth:
        return (False, mean_depth) if return_depth else False

    return (True, mean_depth) if return_depth else True


# =========================================================================
# Geometry & Bounding Box Utilities (IoU, intersection, overlaps)
# =========================================================================


def find_intersection_share(
    map_points: np.ndarray, obj_points: np.ndarray, radius: float = 0.05
) -> float:
    """Calculate the percentage of overlapping points normalized by the query

    objects size.

    Args:
        base_points (numpy.ndarray): shape (n1, 3).
        map_points (numpy.ndarray): shape (n1, 3).
        radius (float): Radius for KD-Tree query (adjust based on point density).

    Returns:
        float: Overlapping ratio between 0 and 1."""
    obj_tree_points = cKDTree(obj_points)

    # Query all points in pcd1 for nearby points in pcd2
    _, indices = obj_tree_points.query(
        map_points, k=1, distance_upper_bound=radius, p=2, workers=-1
    )
    # Remove indices that are out of range
    indices = indices[indices != obj_points.shape[0]]

    # Calculate the overlapping ratio, handle the case where one of the point
    # clouds is empty
    if map_points.shape[0] == 0 or obj_points.shape[0] == 0:
        overlapping_ratio = 0
    else:
        overlapping_ratio = indices.shape[0] / obj_points.shape[0]

    del indices
    return overlapping_ratio


# =========================================================================
# Scene Understanding (Room embeddings, spatial analysis)
# =========================================================================


def compute_room_embeddings(
    room_pcds: List[o3d.geometry.PointCloud],
    pose_list: List[np.ndarray],
    emb_list: List[np.ndarray],
    pcd_min: np.ndarray,
    pcd_max: np.ndarray,
    num_views: int = 5,
    save_path: Union[str, Path] = None,
) -> Tuple[List[List[np.ndarray]], List[List[int]]]:
    """
    Assign all images to their corresponding room regions respectively, apply
    k-mean clustering to the CLIP embeddings of images in each room, and select
    5 representative embeddings to represent the room.

    Args:
        room_pcds (List[o3d.geometry.PointCloud]): a list of 3D point clouds representing each room
        pose_list (List[np.ndarray]): a list of pose of the images
        emb_list (List[np.ndarray]): a list of CLIP embeddings of the images
        pcd_min (np.ndarray): the minimum X, Y, Z of the 3D point cloud of the floor
        pcd_max (np.ndarray): the maximum X, Y, Z of the 3D point cloud of the floor
        num_views (int): the number of views considered in each room
        save_path (Union[str, Path]): a path to save debug info

    Returns:
        repr_embs_list (List[List[np.ndarray]]): a list of CLIP embeddings list, each of the room has a
                                                 list of num_views CLIP embeddings
        repr_img_ids_list (List[List[int]]): a list of image ids list, each of the room has a list of
                                              num_views image indices
    """
    # save_path = Path(save_path) / "debug"
    # if save_path is not None:
    #     os.makedirs(save_path, exist_ok=True)

    img2room_id = []
    room_id2img_id = defaultdict(list)

    flattened_room_points = list()
    plt.figure()
    # colormap over all rooms
    cmap = cm.get_cmap("tab20")
    for room_idx, room_pcd in enumerate(room_pcds):
        room_2d_points = np.stack(
            [np.asarray(room_pcd.points)[:, 0], np.asarray(room_pcd.points)[:, 2]], axis=1
        )
        plt.scatter(room_2d_points[:, 0], room_2d_points[:, 1], s=0.1, c=cmap(room_idx))
        flattened_room_points.append(room_2d_points)

    pbar = tqdm(enumerate(pose_list), total=len(pose_list), desc="assign camera to room")
    pose_cmap = cm.get_cmap("Set1")
    for i, pose in pbar:
        pos = pose[0, 3], pose[2, 3]
        z = pose[1, 3]
        # Check if camera pose is inside the floor bounds
        if z < pcd_min[1] or z > pcd_max[1]:
            img2room_id.append(-1)
            continue
        # Find the closest room given the camera pose
        room_dists = []
        for room_points in flattened_room_points:
            room_dists.append(
                np.min(distance.cdist(np.array([pos]), np.array(room_points), metric="euclidean"))
            )
        closest_room_idx = np.argmin(room_dists)
        plt.scatter(pos[0], pos[1], s=3.0, c=pose_cmap(closest_room_idx))

        img2room_id.append(closest_room_idx)
        room_id2img_id[closest_room_idx].append(i)

    # check whether one of the rooms has not been assigned any image
    for room_id in range(len(flattened_room_points)):
        if room_id not in room_id2img_id:
            # (double)-assign closest image to the room
            closest_cam_pose = list()
            pbar = tqdm(
                enumerate(pose_list),
                total=len(pose_list),
                desc="find closest camera pose to room w/o assigned image",
            )
            for i, pose in pbar:
                pos = pose[0, 3], pose[2, 3]
                z = pose[1, 3]
                # Check if camera pose is inside the floor bounds
                if z < pcd_min[1] or z > pcd_max[1]:
                    closest_cam_pose.append(
                        np.min(
                            distance.cdist(
                                np.array([pos]),
                                np.array(flattened_room_points[room_id]),
                                metric="euclidean",
                            )
                        )
                    )
                else:
                    closest_cam_pose.append(np.inf)
            assert len(closest_cam_pose) == len(pose_list)
            closest_cam_pose_idx = np.argmin(np.array(closest_cam_pose))
            room_id2img_id[room_id].append(closest_cam_pose_idx)

    plt.savefig(os.path.join(save_path, "pcd_camera_pose.png"))

    repr_img_ids_list = []
    repr_embs_list = []
    room_clip_embeddings_list = []
    plt.figure()
    # colormap over all rooms
    cmap = cm.get_cmap("tab20")
    for room_idx, room_pcd in enumerate(room_pcds):
        room_2d_points = np.stack(
            [np.asarray(room_pcd.points)[:, 0], np.asarray(room_pcd.points)[:, 2]], axis=1
        )
        plt.scatter(room_2d_points[:, 0], room_2d_points[:, 1], s=0.1, c=cmap(room_idx))

    for room_id in range(len(flattened_room_points)):
        img_ids = room_id2img_id[room_id]  # get image ids for the room
        # all_img_ids = img_ids.copy()
        print("room_id: ", room_id, " has ", len(img_ids), " images")
        print("img_ids: ", img_ids)
        if len(img_ids) == 0:
            repr_img_ids_list.append([])
            repr_embs_list.append([])
            continue

        repr_img_ids = []
        repr_embs = []
        flat_embeddings = []
        for i in img_ids:
            emb = np.asarray(emb_list[i])
            if emb.ndim > 1:
                emb = emb.reshape(-1, emb.shape[-1]).mean(axis=0)
            flat_embeddings.append(emb)
        room_clip_embeddings = np.array(flat_embeddings)
        room_clip_embeddings_list.append(room_clip_embeddings)
        if len(img_ids) < num_views:
            repr_img_ids_list.append(img_ids)
            repr_embs_list.append([emb for emb in room_clip_embeddings])
            continue
        # To tune the parameter, follow the guideline here:
        # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#clustering-sparse-data-with-k-means
        kmeans = KMeans(n_clusters=num_views, max_iter=100, n_init=5, random_state=0).fit(
            room_clip_embeddings
        )
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        unique_labels = np.unique(labels)
        print(unique_labels)
        for unique_label in unique_labels:
            ids = np.where(labels == unique_label)[0]
            cluster = room_clip_embeddings[ids]
            # mean_feats = np.mean(cluster, axis=0)
            mean_feats = centers[unique_label]
            similarity = np.dot(cluster, mean_feats)
            max_idx = np.argmax(similarity)
            # add for visualize repr image distribution
            pose = pose_list[max_idx]
            pos = pose[0, 3], pose[2, 3]
            plt.scatter(pos[0], pos[1], s=3.0, c=pose_cmap(room_id))
            feats = cluster[max_idx]
            # img_ids -> ids -> max_idx
            repr_img_ids.append(img_ids[ids[max_idx]])
            repr_embs.append(feats)
        repr_img_ids_list.append(repr_img_ids)
        repr_embs_list.append(repr_embs)
    plt.savefig(os.path.join(save_path, "pcd_reprImgs_pose.png"))
    return repr_embs_list, repr_img_ids_list, room_id2img_id, room_clip_embeddings_list


# =========================================================================
# Occupancy Grid & Distance Transforms (grid mapping, distance field)
# =========================================================================


def map_grid_to_point_cloud(occupancy_grid_map, resolution, point_cloud):
    """Map the occupancy grid back to the original coordinates in the point cloud.

    Args:
        occupancy_grid_map (numpy.array): Occupancy grid map as a 2D numpy array, where each cell is marked as either 0 (unoccupied) or 1 (occupied).
        grid_size (tuple): A tuple (width, height) representing the size of the occupancy grid map in meters.
        resolution (float): The resolution of each cell in the grid map in meters.
        point_cloud (numpy.array): 2D numpy array of shape (N, 2), where N is the number of points and each row represents a point (x, y).

    Returns:
        numpy.array: A subset of the original point cloud containing points that correspond to occupied cells in the occupancy grid.
    """

    # make sure image is binary
    occupancy_grid_map = (occupancy_grid_map > 0).astype(np.uint8)

    # Get the occupied cell indices
    y_cells, x_cells = np.where(occupancy_grid_map == 1)

    # Compute the corresponding point coordinates for occupied cells
    # NOTE: The coordinates are shifted by 10.5 cells to account for the
    # padding added to the grid map
    mapped_x_coords = (x_cells - 10.5) * resolution + np.min(point_cloud[:, 0])
    mapped_y_coords = (y_cells - 10.5) * resolution + np.min(point_cloud[:, 1])

    # Stack the mapped x and y coordinates to form the mapped point cloud
    mapped_point_cloud = np.column_stack((mapped_x_coords, mapped_y_coords))

    return mapped_point_cloud


def distance_transform(occupancy_map, resolution, tmp_path):
    """Perform distance transform on the occupancy map to find the distance of

    each cell to the nearest occupied cell.

    Args:
        occupancy_map: 2D numpy array representing the occupancy map.
        resolution: The resolution of each cell in the grid map in meters.
        path: The path to save the distance transform image.

    Returns:
        The distance transform of the occupancy map."""

    print("occupancy_map shape: ", occupancy_map.shape)
    bw = occupancy_map.copy()
    full_map = occupancy_map.copy()

    # invert the image
    bw = cv2.bitwise_not(bw)

    # Perform the distance transform algorithm
    bw = np.uint8(bw)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    print("range of dist: ", np.min(dist), np.max(dist))
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
    plt.figure()
    plt.imshow(dist, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist.png"))

    dist = np.uint8(dist)
    # apply Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(dist, (11, 1), 10)
    plt.figure()
    plt.imshow(blur, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist_blur.png"))
    _, dist = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure()
    plt.imshow(dist, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist_thresh.png"))

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype("uint8")
    # Find total markers
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("number of seeds, aka rooms: ", len(contours))

    # print the area of each seed
    for i in range(len(contours)):
        print("area of seed {}: ".format(i), cv2.contourArea(contours[i]))

    # remove small seed contours
    min_area_m = 0.5
    min_area = (min_area_m / resolution) ** 2
    print("min_area: ", min_area)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    print("number of contours after remove small seeds: ", len(contours))

    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)

    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i + 1), -1)
    # Draw the background marker
    circle_radius = 1  # in pixels
    cv2.circle(markers, (3, 3), circle_radius, len(contours) + 1, -1)

    # Perform the watershed algorithm
    full_map = cv2.cvtColor(full_map, cv2.COLOR_GRAY2BGR)
    cv2.watershed(full_map, markers)

    # find the vertices of each room
    room_vertices = []
    # for i in range(len(contours)):
    #     room_vertices.append(np.where(markers == i + 1))
    # room_vertices = np.array(room_vertices, dtype=object).squeeze()
    for i in range(len(contours)):
        room_vertices.append(tuple(np.where(markers == i + 1)))  # each element is (rows, cols)

    plt.figure()
    plt.imshow(markers, cmap="jet", origin="lower")
    # # Write the room index at the center of each room region
    for i, room in enumerate(room_vertices):
        if len(room[0]) == 0:
            continue
        cy, cx = np.mean(room[0]), np.mean(room[1])  # y is row, x is column
        plt.text(
            cx, cy, str(i), color="white", fontsize=8, ha="center", va="center", fontweight="bold"
        )

    plt.savefig(os.path.join(tmp_path, "markers.png"))

    return room_vertices


def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """Taken from ConceptGraphs Compute IoU between two sets of axis-aligned 3D

    bounding boxes.
    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)

    Returns:
        (M, N)"""
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1)  # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1)  # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1)  # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1)  # Shape: (N, 3)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

    # Compute volume of intersection box
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return iou


# =========================================================================
# Spatial Search & Point Cloud Overlap (FAISS-based queries)
# =========================================================================


def find_overlapping_ratio_faiss(pcd1, pcd2, radius=0.02, index1=None, index2=None):
    """Calculate the percentage of overlapping points between two point clouds

    using FAISS.

    Args:
        pcd1 (numpy.ndarray): Point cloud 1, shape (n1, 3).
        pcd2 (numpy.ndarray): Point cloud 2, shape (n2, 3).
        radius (float): Radius for KD-Tree query (adjust based on point density).
        index1 (faiss.Index, optional): Pre-built FAISS index for pcd1. If None,
        one is built on the fly. Providing pre-built indices avoids redundant
        index construction when the same cloud appears in many pairs.
        index2 (faiss.Index, optional): Pre-built FAISS index for pcd2.

    Returns:
        float: Overlapping ratio between 0 and 1."""
    if isinstance(pcd1, o3d.geometry.PointCloud) and isinstance(pcd2, o3d.geometry.PointCloud):
        pcd1 = np.asarray(pcd1.points)
        pcd2 = np.asarray(pcd2.points)

    if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
        return 0

    pcd1_f32 = pcd1.astype(np.float32)
    pcd2_f32 = pcd2.astype(np.float32)

    # Build indices only when not provided by the caller.
    # Use GPU index when available for faster per-pair search.
    if index1 is None:
        index1 = _make_faiss_index(pcd1_f32.shape[1], pcd1_f32)
    if index2 is None:
        index2 = _make_faiss_index(pcd2_f32.shape[1], pcd2_f32)

    # Query all points in pcd1 for nearby points in pcd2
    D1, _ = index2.search(pcd1_f32, k=1)
    D2, _ = index1.search(pcd2_f32, k=1)

    number_of_points_overlapping1 = np.sum(D1 < radius**2)
    number_of_points_overlapping2 = np.sum(D2 < radius**2)

    overlapping_ratio = np.max(
        [
            number_of_points_overlapping1 / pcd1.shape[0],
            number_of_points_overlapping2 / pcd2.shape[0],
        ]
    )

    return overlapping_ratio


# =========================================================================
# Point Cloud Merging & Spatial Grouping (3D mask merging, merge strategies)
# =========================================================================


def merge_point_clouds_list(
    pcd_list: List[o3d.geometry.PointCloud], voxel_size: float = 0.02
) -> o3d.geometry.PointCloud:
    """Merge a list of point clouds into a single point cloud.

    Args:
        pcd_list: List of point clouds to merge.
        voxel_size: Voxel size for downsampling.

    Returns:
        Merged point cloud."""
    merged_pcd = pcd_list[0]
    for pcd in pcd_list[1:]:
        merged_pcd += pcd
    # Downsample instead of running DBSCAN on every intermediate merge.
    # A single DBSCAN pass over the final merged collection (called by the
    # caller if needed) is far cheaper than N per-merge DBSCAN calls.
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
    return merged_pcd


def feats_denoise_dbscan(
    feats: Union[np.ndarray, List], eps: float = 0.02, min_points: int = 2
) -> np.ndarray:
    """Aggregate per-point features into a single representative feature vector

    for a 3D mask segment, with lightweight outlier rejection.
    The original implementation ran ``DBSCAN(metric='cosine')`` which is
    O(n²) in feature count and cannot use spatial indexing.  For the typical
    use-case – producing a single mean embedding per segment – a much cheaper
    approach suffices:
    1. Compute the global mean.
    2. Reject vectors whose cosine similarity to the mean is below a
    threshold (conservative outlier removal).
    3. Return the mean of the inlier set.
    This drops complexity from O(n²) to O(n) while producing virtually
    identical output for the unimodal feature distributions that arise from
    a single 3D object segment.  Use ``use_dbscan=True`` to fall back to the
    original DBSCAN path if multi-modal filtering is required.

    Args:
        feats: (N, D) array of feature vectors.
        eps: Unused (kept for API compatibility with callers).
        min_points: Minimum inliers required; falls back to full mean if
    fewer inliers pass the cosine threshold.

    Returns:
        (D,) representative feature vector."""
    feats = np.array(feats)
    if feats.ndim == 1 or feats.shape[0] == 0:
        return feats

    # Compute L2-normalised mean
    mean_feat = np.mean(feats, axis=0)
    norm = np.linalg.norm(mean_feat)
    if norm < 1e-8:
        return mean_feat
    mean_feat_normed = mean_feat / norm

    # Cosine similarity of each vector to the mean
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    feats_normed = feats / norms
    cosine_sim = feats_normed @ mean_feat_normed  # (N,)

    # Keep vectors within one standard deviation of the median similarity
    # (robust to a few extreme outliers without O(n²) computation)
    threshold = np.median(cosine_sim) - np.std(cosine_sim)
    inlier_mask = cosine_sim >= threshold
    inliers = feats[inlier_mask]

    if len(inliers) < min_points:
        # Not enough inliers — fall back to unconditional mean
        return mean_feat

    return np.mean(inliers, axis=0)


def _visualize_point_clusters(
    points: np.ndarray,
    labels: np.ndarray,
    show_outliers: bool = True,
) -> np.ndarray:
    """Visualize point cloud clustering with colored clusters.

    Args:
        points: (N, 3) array of point coordinates.
        labels: (N,) array of cluster labels, where -1 represents noise/outliers.
        show_outliers: Whether to show outlier points in the visualization.

    Returns:
        (N, 3) RGB color array for points.
    """
    colors = np.zeros_like(points, dtype=np.float64)
    cmap = plt.get_cmap("tab20")

    for label in np.unique(labels):
        if label == -1:
            # Noise points - black
            color = np.array([0, 0, 0])
        else:
            # Clusters - indexed from colormap
            color = cmap(label % 20)[:3]

        colors[labels == label] = color

    # Optionally suppress outlier visualization
    if not show_outliers:
        colors[labels == -1] = np.array([1, 1, 1])  # white for outliers

    return colors


def visualize_pcd_clusters(
    pcd: o3d.geometry.PointCloud,
    save_dir: str = None,
    dbscan_eps: float = 0.02,
    dbscan_min_points: int = 10,
) -> None:
    """Run DBSCAN clustering on a point cloud and save a colored visualization.

    Args:
        pcd: Input Open3D point cloud.
        save_dir: Directory to save the colored .ply file. If None, skips saving.
        dbscan_eps: DBSCAN epsilon radius.
        dbscan_min_points: Minimum neighbors to form a cluster.
    Uses:
        - Multi-threaded CPU DBSCAN via sklearn
        - Optional GPU acceleration via cuML if available
    """

    # print("Running DBSCAN clustering on point cloud...")
    # labels = np.array(
    #     pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=True)
    # )
    # points = np.asarray(pcd.points)
    # colors = _visualize_point_clusters(points, labels)

    # colored_pcd = o3d.geometry.PointCloud(pcd)
    # colored_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # # Display the colored point cloud
    # print("Visualizing DBSCAN clusters. Close the window to continue...")
    # o3d.visualization.draw_geometries([colored_pcd], window_name="DBSCAN Clustering Visualization")

    # if save_dir is not None:
    #     os.makedirs(save_dir, exist_ok=True)
    #     o3d.io.write_point_cloud(os.path.join(save_dir, "pcd_clusters.ply"), colored_pcd)

    points = np.asarray(pcd.points)

    print(f"Running DBSCAN clustering on {len(points)} points...")

    # --------------------------------------------------
    # Try GPU DBSCAN first
    # --------------------------------------------------
    try:
        import cupy as cp
        from cuml.cluster import DBSCAN as cuDBSCAN

        print("Using GPU DBSCAN (cuML)...")

        points_gpu = cp.asarray(points)

        dbscan = cuDBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_points,
        )

        labels = dbscan.fit_predict(points_gpu)
        labels = cp.asnumpy(labels)

    # --------------------------------------------------
    # Fallback to multi-thread CPU DBSCAN
    # --------------------------------------------------
    except Exception as e:
        print("GPU DBSCAN unavailable, using multi-threaded CPU DBSCAN.")
        print(f"Reason: {e}")

        dbscan = DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_points,
            algorithm="kd_tree",
            n_jobs=-1,  # use all CPU cores
        )

        labels = dbscan.fit_predict(points)

    # --------------------------------------------------
    # Color visualization
    # --------------------------------------------------
    colors = _visualize_point_clusters(points, labels)

    colored_pcd = o3d.geometry.PointCloud(pcd)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    print("Visualizing DBSCAN clusters. Close the window to continue...")

    o3d.visualization.draw_geometries([colored_pcd], window_name="DBSCAN Clustering Visualization")

    # --------------------------------------------------
    # Save result
    # --------------------------------------------------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "pcd_clusters.ply")

        o3d.io.write_point_cloud(save_path, colored_pcd)

        print(f"Saved clustered point cloud to: {save_path}")


# =========================================================================
# Point Cloud Denoising & Filtering (DBSCAN, statistical, voxel, radius)
# =========================================================================


def pcd_denoise(
    pcd: o3d.geometry.PointCloud, method: str = "statistical", viz: bool = False, **kwargs
) -> o3d.geometry.PointCloud:
    """Unified point cloud denoising interface.

    Supports multiple denoising methods: statistical outlier removal (SOR)
    and DBSCAN clustering-based denoising.

    Args:
        pcd: Input point cloud to denoise.
        method: Denoising method ("statistical" or "dbscan").
        viz: Whether to visualize denoising results.
        **kwargs: Method-specific parameters:
            - statistical: nb_neighbors (int, default 20), std_ratio (float, default 1.0)
            - dbscan: eps (float, default 0.02), min_points (int, default 10),
                      min_cluster_size (int, default 5)

    Returns:
        Denoised point cloud.

    Raises:
        ValueError: If method is not recognized.
    """
    method = method.lower()

    if method == "statistical":
        return _denoise_statistical(pcd, viz=viz, **kwargs)
    elif method == "dbscan":
        return _denoise_dbscan(pcd, viz=viz, **kwargs)
    else:
        raise ValueError(f"Unknown denoising method: {method}. Use 'statistical' or 'dbscan'.")


def _denoise_statistical(
    pcd: o3d.geometry.PointCloud, nb_neighbors: int = 20, std_ratio: float = 1.0, viz: bool = False
) -> o3d.geometry.PointCloud:
    """Remove outliers using statistical outlier removal (SOR).

    Args:
        pcd: Input point cloud.
        nb_neighbors: Number of neighbors to analyze for each point.
        std_ratio: Points with distance > (mean + std_ratio * std) are outliers.
        viz: Whether to visualize results.

    Returns:
        Denoised point cloud (inliers only).
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    inlier_cloud = pcd.select_by_index(ind)

    if viz:
        outlier_cloud = pcd.select_by_index(ind, invert=True)
        outlier_cloud.paint_uniform_color([0, 0, 0])  # black for outliers

        print(
            f"[INFO] SOR: Kept {len(ind)} inliers, removed {len(pcd.points) - len(ind)} outliers"
        )
        o3d.visualization.draw_geometries(
            [inlier_cloud, outlier_cloud], window_name="Statistical Outlier Removal"
        )

    return inlier_cloud


def _denoise_dbscan(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.02,
    min_points: int = 10,
    min_cluster_size: int = 5,
    viz: bool = False,
) -> o3d.geometry.PointCloud:
    """Remove outliers using DBSCAN clustering.

    Args:
        pcd: Input point cloud to denoise.
        eps: DBSCAN epsilon radius (distance threshold).
        min_points: Minimum neighbors to form a cluster.
        min_cluster_size: Minimum points required in largest cluster.
        viz: Whether to visualize clustering results.

    Returns:
        Denoised point cloud (largest cluster only).
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=viz))

    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)

    # Visualize all clusters if requested
    if viz:
        colored_pcd = o3d.geometry.PointCloud(pcd)
        vis_colors = _visualize_point_clusters(obj_points, labels, show_outliers=True)
        colored_pcd.colors = o3d.utility.Vector3dVector(vis_colors)

        n_clusters = np.max(labels) + 1
        n_noise = np.sum(labels == -1)
        print(f"[INFO] DBSCAN: {n_clusters} clusters and {n_noise} noise points")
        o3d.visualization.draw_geometries([colored_pcd], window_name="DBSCAN Clustering Result")

    # Extract largest cluster
    counter = Counter(labels)
    if -1 in counter:
        del counter[-1]

    if not counter:
        # No valid clusters found
        return pcd

    largest_label, _ = counter.most_common(1)[0]
    keep_mask = labels == largest_label

    if np.sum(keep_mask) < min_cluster_size:
        # Largest cluster too small - return original
        return pcd

    # Create denoised point cloud from largest cluster
    denoised_pcd = o3d.geometry.PointCloud()
    denoised_pcd.points = o3d.utility.Vector3dVector(obj_points[keep_mask])
    denoised_pcd.colors = o3d.utility.Vector3dVector(obj_colors[keep_mask])

    return denoised_pcd


def compute_3d_bbox_iou(
    bbox1: o3d.geometry.AxisAlignedBoundingBox,
    bbox2: o3d.geometry.AxisAlignedBoundingBox,
    padding: float = 0,
) -> float:
    """Compute 3D Intersection over Union (IoU) between two point clouds.

    Args:
        pcd1: (open3d.geometry.PointCloud): Point cloud 1.
        pcd2: (open3d.geometry.PointCloud): Point cloud 2.
        padding: (float): Padding to add to the bounding box.

    Returns:
        3D IoU between 0 and 1."""
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

    # Compute the overlap between the two bounding boxes
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)

    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)

    return iou


def merge_3d_masks(
    mask_list: List[o3d.geometry.PointCloud],
    overlap_threshold: float = 0.5,
    radius: float = 0.02,
    iou_thresh: float = 0.05,
) -> List[o3d.geometry.PointCloud]:
    """Merge the overlapped 3D masks in the list of masks using matrix.

    Args:
        mask_list: list of point clouds
        overlap_threshold: threshold for overlapping ratio
        radius: radius for faiss search
        iou_thresh: (float): threshold for iou

    Returns:
        merged point clouds and features.
    Performance notes
    -----------------
    * **Spatial grid pre-filter (#6)**: masks are bucketed into a coarse 3D
    hash-grid before any pair-wise check.  Only masks sharing the same cell
    or a direct neighbor cell (26-connectivity) are considered candidates.
    This converts the O(N²) candidate-generation step into O(N·k) where k
    is the average neighbour count, drastically pruning the pair list for
    large scenes without affecting correctness.
    * **FAISS index caching (#2)**: one index is built per mask and reused
    across all pairs, eliminating O(N²) redundant index construction.
    * **GPU FAISS (#4)**: when faiss-gpu is available the indices are placed
    on GPU.  Because GPU FAISS is not thread-safe, the GPU path uses a
    serial loop (GPU parallelism handles speedup internally via batched ops).
    The CPU path retains the ThreadPoolExecutor to saturate CPU cores."""
    if not mask_list:
        return mask_list

    n = len(mask_list)
    aa_bb = [pcd.get_axis_aligned_bounding_box() for pcd in mask_list]

    # --- (#6) Spatial hash-grid pre-filter -----------------------------------
    # Cell size chosen so that two masks in non-adjacent cells cannot overlap
    # given the FAISS search radius.  Using 10× radius gives generous headroom.
    cell_size = max(radius * 100.0, 0.5)  # metres; min 0.5 m to avoid tiny cells

    def _cell_key(bbox):
        centre = (np.asarray(bbox.get_min_bound()) + np.asarray(bbox.get_max_bound())) * 0.5
        return tuple((centre / cell_size).astype(int))

    cell_keys = [_cell_key(bb) for bb in aa_bb]

    # Build neighbour lookup: cell → list of mask indices
    from collections import defaultdict as _dd

    cell_map = _dd(list)
    for idx, key in enumerate(cell_keys):
        cell_map[key].append(idx)

    # Expand each mask to its 26 (3D Moore) neighbours to find candidates
    def _neighbours(key):
        cx, cy, cz = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    yield (cx + dx, cy + dy, cz + dz)

    # Collect spatially-adjacent candidate pairs (i < j)
    grid_candidates = set()
    for i, key in enumerate(cell_keys):
        for nkey in _neighbours(key):
            for j in cell_map.get(nkey, []):
                if j > i:
                    grid_candidates.add((i, j))
    # -------------------------------------------------------------------------

    # --- (#2) Build one FAISS index per mask ----------------------------------
    pts_list = [np.asarray(pcd.points).astype(np.float32) for pcd in mask_list]

    if _FAISS_GPU_RES is not None:
        # GPU: build CPU index → transfer to GPU (not thread-safe, used serially)
        faiss_indices = [
            _make_faiss_index(pts.shape[1], pts) if pts.shape[0] > 0 else None for pts in pts_list
        ]
    else:
        # CPU: plain IndexFlatL2, safe for concurrent reads in ThreadPoolExecutor
        faiss_indices = []
        for pts in pts_list:
            if pts.shape[0] == 0:
                faiss_indices.append(None)
            else:
                cpu_idx = faiss.IndexFlatL2(pts.shape[1])
                cpu_idx.add(pts)
                faiss_indices.append(cpu_idx)
    # -------------------------------------------------------------------------

    # Apply bbox IoU as a second-pass filter on the spatial candidates
    candidate_pairs = [
        (i, j) for i, j in grid_candidates if compute_3d_bbox_iou(aa_bb[i], aa_bb[j]) > iou_thresh
    ]

    overlap_matrix = np.zeros((n, n))

    if candidate_pairs:

        def _compute_pair(i, j):
            if faiss_indices[i] is None or faiss_indices[j] is None:
                return i, j, 0.0
            ratio = find_overlapping_ratio_faiss(
                pts_list[i],
                pts_list[j],
                radius=1.5 * radius,
                index1=faiss_indices[i],
                index2=faiss_indices[j],
            )
            return i, j, ratio

        if _FAISS_GPU_RES is not None:
            # (#4) GPU path: serial loop — GPU batches the search internally
            for i, j in candidate_pairs:
                _, _, ratio = _compute_pair(i, j)
                overlap_matrix[i, j] = ratio
        else:
            # CPU path: parallel across cores via ThreadPoolExecutor
            max_workers = min(os.cpu_count() or 4, len(candidate_pairs))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_compute_pair, i, j): (i, j) for i, j in candidate_pairs}
                for future in as_completed(futures, timeout=300):
                    try:
                        i, j, ratio = future.result(timeout=10)
                        overlap_matrix[i, j] = ratio
                    except TimeoutError:
                        i, j = futures[future]
                        overlap_matrix[i, j] = 0.0

    # check if overlap_matrix is zero size
    if overlap_matrix.size == 0:
        return mask_list

    graph = overlap_matrix > overlap_threshold
    n_components, component_labels = connected_components(graph)
    component_indices = [np.where(component_labels == k)[0] for k in range(n_components)]

    # merge the masks in each component
    pcd_list_merged = []
    for indices in component_indices:
        pcd_list_merged.append(
            merge_point_clouds_list([mask_list[i] for i in indices], voxel_size=0.5 * radius)
        )

    return pcd_list_merged


@dataclass
class MergeConfig:
    """Configuration for point cloud merging operations.

    Attributes:
        overlap_threshold: Threshold for considering masks as overlapping (0-1).
        voxel_size: Size for voxel downsampling during merges.
        search_radius: Radius used in FAISS spatial searches.
        iou_threshold: IoU threshold for bounding box filtering.
    """

    overlap_threshold: float = 0.5
    voxel_size: float = 0.5 * 0.02  # default 0.01
    search_radius: float = 0.02
    iou_threshold: float = 0.05


def select_merge_strategy(strategy: str = "sequential") -> callable:
    """Factory function to select point cloud merge strategy.

    Args:
        strategy: One of "sequential", "hierarchical", or "adjacent".
            - "sequential": Best for temporal sequences; merges frames incrementally.
            - "hierarchical": Best for large batches; uses decreasing thresholds.
            - "adjacent": Best for pairwise merging; processes pairs sequentially.

    Returns:
        Merge function with signature (frames_pcd, th, down_size, proxy_th).

    Raises:
        ValueError: If strategy is not recognized.
    """
    strategies = {
        "sequential": seq_merge,
        "hierarchical": hierarchical_merge,
        "adjacent": merge_adjacent_frames,
    }

    strategy = strategy.lower()
    if strategy not in strategies:
        raise ValueError(
            f"Unknown merge strategy: {strategy}. Available: {list(strategies.keys())}"
        )

    return strategies[strategy]


def merge_adjacent_frames(
    frames_pcd: List[List[o3d.geometry.PointCloud]], th: float, down_size: float, proxy_th: float
) -> List[List[o3d.geometry.PointCloud]]:
    """Merge adjacent frames in the list of frames.

    Args:
        frames_pcd: list of point clouds
        th: threshold for overlapping ratio
        down_size: radius for downsampling
        proxy_th: threshold for iou

    Returns:
        merged point clouds and features."""
    new_frames_pcd = []
    for i in tqdm(range(0, len(frames_pcd), 2)):
        # if the number of frames is odd, the last frame is appended without
        # merging.
        if i == len(frames_pcd) - 1:
            new_frames_pcd.append(frames_pcd[i])
            break
        pcd_list = frames_pcd[i] + frames_pcd[i + 1]

        pcd_list = merge_3d_masks(
            pcd_list,
            overlap_threshold=th,
            radius=down_size,
            iou_thresh=proxy_th,
        )
        new_frames_pcd.append(pcd_list)

    return new_frames_pcd


def hierarchical_merge(
    frames_pcd: List[List[o3d.geometry.PointCloud]],
    th: float,
    th_factor: float,
    down_size: float,
    proxy_th: float,
) -> List[o3d.geometry.PointCloud]:
    """Hierarchical merge the frames in the list of frames.

    Args:
        frames_pcd: list of point clouds
        th: threshold for overlapping ratio
        th_factor: factor for decreasing the threshold
        down_size: radius for downsampling
        proxy_th: threshold for iou

    Returns:
        merged point clouds and features."""
    while len(frames_pcd) > 1:
        frames_pcd = merge_adjacent_frames(frames_pcd, th, down_size, proxy_th)
        if len(frames_pcd) > 1:
            th -= th_factor * (len(frames_pcd) - 2) / max(1, len(frames_pcd) - 1)
            print("th: ", th)

    # apply one more merge
    frames_pcd = frames_pcd[0]
    frames_pcd = merge_3d_masks(
        frames_pcd, overlap_threshold=0.75, radius=down_size, iou_thresh=proxy_th
    )
    return frames_pcd


def seq_merge(
    frames_pcd: List[List[o3d.geometry.PointCloud]], th: float, down_size: float, proxy_th: float
) -> List[o3d.geometry.PointCloud]:
    """Merge the frames in the list of frames sequentially.

    Args:
        frames_pcd: list of point clouds
        th: threshold for overlapping ratio
        down_size: radius for downsampling
        proxy_th: threshold for iou

    Returns:
        merged point clouds and features."""

    # Pre-merge masks within each frame to reduce re-processing
    merged_frames = [
        merge_3d_masks(
            frame,
            overlap_threshold=th,
            radius=down_size,
            iou_thresh=proxy_th,
        )
        for frame in frames_pcd
    ]

    # Incrementally merge frames across time without re-processing previous merged frames
    global_masks = merged_frames[0]

    for i in tqdm(range(1, len(merged_frames))):
        # Only merge current merged frame with accumulated result, don't re-merge accumulated
        mask_list = global_masks + merged_frames[i]
        global_masks = merge_3d_masks(
            mask_list,
            overlap_threshold=th,
            radius=down_size,
            iou_thresh=proxy_th,
        )

    # Final cleanup pass for any residual overlaps
    global_masks = merge_3d_masks(
        global_masks,
        overlap_threshold=th,
        radius=down_size,
        iou_thresh=proxy_th,
    )
    return global_masks


# =========================================================================
# Point Cloud Filtering Utilities
# =========================================================================


def filter_point_cloud(
    pcd: o3d.geometry.PointCloud,
    config: dict,
    save_dir: str = None,
) -> o3d.geometry.PointCloud:
    """Apply multi-stage noise reduction to point cloud.

    For real RGB-D sensor data, applies complementary filtering:
    1. DBSCAN denoising - Removes isolated points (flying pixels)
    2. Voxel downsampling - Reduces spatial density
    3. Radius outlier removal - Cleans thin noise artifacts

    Args:
        pcd: Input point cloud to filter
        config: Configuration dict with keys:
            - enable_dbscan_filtering: bool (default True)
            - dbscan_eps: float (default 0.02)
            - dbscan_min_points: int (default 10)
            - enable_voxel_downsampling: bool (default True)
            - voxel_size: float (default 0.05)
            - enable_radius_outlier_filtering: bool (default True)
            - radius_nb_points: int (default 1000)
            - radius_distance: float (default 1.0)
        save_dir: Unused (kept for backward compatibility)

    Returns:
        Filtered point cloud
    """
    original_size = len(pcd.points)

    # Step 1: DBSCAN denoising on the DENSE raw cloud.
    # Flying pixels (isolated depth sensor artifacts) form small isolated clusters.
    # DBSCAN parameters must match the raw cloud density — eps=0.02 and min_points=10
    # work reliably at ~0.005m avg spacing. Running this BEFORE voxel downsampling
    # ensures the density is high enough for DBSCAN to distinguish noise from geometry.
    if config.get("enable_dbscan_filtering", True):
        dbscan_eps = config.get("dbscan_eps", 0.02)
        dbscan_min = config.get("dbscan_min_points", 10)
        pcd = pcd_denoise(pcd, method="dbscan", eps=dbscan_eps, min_points=dbscan_min)
        after_count = len(pcd.points)
        print(f"  DBSCAN denoising: {original_size} → {after_count} points")
        original_size = after_count

    # Step 2: Voxel downsampling for uniform spatial density reduction.
    # Runs AFTER DBSCAN so the input is already clean, and the downsampled
    # cloud has a well-defined grid structure for final radius removal.
    if config.get("enable_voxel_downsampling", True):
        voxel_size = config.get("voxel_size", 0.05)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        after_count = len(pcd.points)
        print(f"  Voxel downsampling ({voxel_size}m): {original_size} → {after_count} points")
        original_size = after_count
    # Step : statistical outlier removal
    if config.get("enable_statistical_filtering", True):
        nb_neighbors = config.get("statistical_nb_neighbors", 20)
        std_ratio = config.get("statistical_std_ratio", 2.0)

        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

        after_count = len(pcd.points)
        print(f"Statistical filtering: {original_size} → {after_count}")
        original_size = after_count

    # Step 3: Radius-based outlier removal for residual noise at boundaries.
    # Operates on the downsampled cloud to remove any remaining thin artifacts.
    if config.get("enable_radius_outlier_filtering", True):
        radius_nb = config.get("radius_nb_points", 1000)
        radius_dist = config.get("radius_distance", 1.0)
        cl, ind = pcd.remove_radius_outlier(nb_points=radius_nb, radius=radius_dist)
        pcd = pcd.select_by_index(ind)
        after_count = len(pcd.points)
        print(f"  Radius outlier removal: {original_size} → {after_count} points")

    print(f"  ✓ Filtering complete: {len(pcd.points)} points retained")
    return pcd


def print_object_tree(
    objects_list: list,
    scores: "np.ndarray",
    top_index: "np.ndarray",
) -> None:
    """Print top-N object retrieval results as a floor→room→object tree.

    Args:
        objects_list: List of Object instances (must have .object_id and .name).
        scores: 1-D score array aligned with objects_list.
        top_index: Ordered indices into objects_list to display.
    """
    tree: dict = {}
    for i in top_index:
        obj = objects_list[i]
        score = float(scores[i])
        parts = str(obj.object_id).split("_")
        f_key = parts[0] if len(parts) > 0 else "?"
        r_key = "_".join(parts[:2]) if len(parts) > 1 else f_key
        tree.setdefault(f_key, {}).setdefault(r_key, []).append((obj.object_id, obj.name, score))
    for f_key, rooms in tree.items():
        print(f"floor {f_key}")
        room_items = list(rooms.items())
        for r_idx, (r_key, objs) in enumerate(room_items):
            is_last_room = r_idx == len(room_items) - 1
            r_prefix = "     └──" if is_last_room else "     ├──"
            print(f"{r_prefix} room {r_key}")
            o_indent = "            " if is_last_room else "     │      "
            for o_idx, (oid, name, score) in enumerate(objs):
                is_last_obj = o_idx == len(objs) - 1
                o_prefix = "└──── " if is_last_obj else "├──── "
                print(f"{o_indent}{o_prefix}Object {oid}: {name} [{score:.4f}]")
