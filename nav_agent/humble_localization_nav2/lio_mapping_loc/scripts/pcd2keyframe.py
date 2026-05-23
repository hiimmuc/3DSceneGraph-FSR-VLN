import os
import numpy as np
import open3d as o3d
from tqdm import tqdm


def quat_to_rotmat_wxyz(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    # normalize to be safe
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n <= 0.0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def load_keyframe_poses(pose_txt_path: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    pose format:
    index timestamp x y z qw qx qy qz
    returns: dict[keyframe_id] = (t(3,), R(3,3)) for T_map_lidar.
    """
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if not os.path.exists(pose_txt_path):
        raise FileNotFoundError(f"pose file not found: {pose_txt_path}")

    with open(pose_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            k = int(parts[0])
            # parts[1] timestamp (unused)
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            qw, qx, qy, qz = float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])
            t = np.array([x, y, z], dtype=np.float64)
            R = quat_to_rotmat_wxyz(qw, qx, qy, qz)
            poses[k] = (t, R)
    return poses


def map_to_lidar(points_map: np.ndarray, t_map_lidar: np.ndarray, R_map_lidar: np.ndarray) -> np.ndarray:
    """
    points_map: (N,3) in map frame
    T_map_lidar = [R_map_lidar, t_map_lidar]
    return points_lidar = R^T (p_map - t)
    """
    pm = points_map.astype(np.float64, copy=False)
    out = (pm - t_map_lidar.reshape(1, 3)) @ R_map_lidar  # row-vector form == R^T on column vectors
    return out


def _normalize_colors_if_needed(colors: np.ndarray) -> np.ndarray:
    if colors is None:
        return None
    c = colors.astype(np.float32, copy=False)
    if c.size == 0:
        return c
    # if rgb is 0..255
    if np.nanmax(c) > 1.0:
        c = c / 255.0
    return c


def pcd2keyframe(input_pcd_path: str, output_dir: str, pose_txt_path: str | None = None):
    if not os.path.exists(input_pcd_path):
        print(f"Error: Input file '{input_pcd_path}' not found.")
        return

    if pose_txt_path is None:
        pose_txt_path = os.path.join(os.path.dirname(input_pcd_path), "keyframe_pose.txt")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    print(f"Loading point cloud from {input_pcd_path} ...")
    try:
        pcd = o3d.t.io.read_point_cloud(input_pcd_path)
    except Exception as e:
        print(f"Failed to read PCD with Open3D Tensor IO: {e}")
        return

    # find label field
    label_attr_name = None
    for name in ["label", "Label"]:
        if name in pcd.point:
            label_attr_name = name
            break
    if label_attr_name is None:
        print("Error: Could not find 'label' attribute in the point cloud.")
        print(f"pcd.point = {pcd.point}")
        return

    print(f"Using attribute '{label_attr_name}' as keyframe ID.")

    # load poses
    try:
        poses = load_keyframe_poses(pose_txt_path)
    except Exception as e:
        print(f"Error: failed to load pose file: {e}")
        return
    print(f"Loaded {len(poses)} poses from {pose_txt_path}")

    # data arrays
    points = pcd.point.positions.numpy()  # (N,3)
    labels = pcd.point[label_attr_name].numpy().reshape(-1).astype(np.int64)

    colors = None
    intensities = None
    rgbs = None

    if "colors" in pcd.point:
        colors = pcd.point.colors.numpy()  # (N,3)
    if "rgb" in pcd.point:
        rgbs = pcd.point.rgb.numpy()  # (N,3) possibly uint8
    if "intensity" in pcd.point:
        intensities = pcd.point.intensity.numpy().reshape(-1)  # (N,)

    if colors is None and rgbs is not None:
        colors = rgbs
    colors = _normalize_colors_if_needed(colors)

    unique_labels = np.unique(labels)
    print(f"Found {len(unique_labels)} unique keyframes.")

    for kf_id in tqdm(unique_labels, desc="Saving keyframes (map->lidar)", unit="frame"):
        idx = np.where(labels == kf_id)[0]
        if idx.size == 0:
            continue

        if int(kf_id) not in poses:
            # pose missing -> skip (or save raw). Here: skip to avoid wrong frame.
            # If you prefer saving raw, replace `continue` with saving without transform.
            continue

        t, R = poses[int(kf_id)]
        pts_map = points[idx]
        pts_lidar = map_to_lidar(pts_map, t, R).astype(np.float32)

        sub_pcd_t = o3d.t.geometry.PointCloud(o3d.core.Tensor(pts_lidar))
        if colors is not None:
            sub_pcd_t.point.colors = o3d.core.Tensor(colors[idx].astype(np.float32))
        if intensities is not None:
            sub_pcd_t.point.intensity = o3d.core.Tensor(intensities[idx].reshape(-1, 1).astype(np.float32))

        file_name = f"{int(kf_id):06d}.pcd"
        save_path = os.path.join(output_dir, file_name)
        o3d.t.io.write_point_cloud(save_path, sub_pcd_t)

    print("Done.")


if __name__ == "__main__":
    input_file = "/home/users/tingyang.xiao/VLN/IC-7F/cloudGlobal.pcd"
    output_folder = "/home/users/tingyang.xiao/VLN/IC-7F/keyframe_cloud"
    pose_txt_path = "/home/users/tingyang.xiao/VLN/IC-7F/keyframe_pose.txt"
    # 默认读取 input_file 同目录下的 keyframe_pose.txt
    pcd2keyframe(input_file, output_folder,pose_txt_path)