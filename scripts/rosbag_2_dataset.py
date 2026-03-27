#!/usr/bin/env python3
"""
rosbag_2_dataset.py
-------------------
Convert a ROS2 bag file (.db3 or .mcap) into the Horizon RGB-D dataset
format consumed by HorizonDataset / semantic_scene_reconstruction.py.

Output layout
─────────────
<output_dir>/
├── images/          # RGB frames:   {timestamp_sec:.4f}.png
├── depth/           # Depth frames: {timestamp_sec:.4f}.png  (16-bit, mm)
├── poses.txt        # Camera trajectory, TUM format:
│                    #   timestamp tx ty tz qx qy qz qw
└── camera_info.yaml       # Camera intrinsics (fx, fy, cx, cy, width, height)

TUM pose format
───────────────
Each line: timestamp tx ty tz qx qy qz qw
  - timestamp : seconds (float, same as image filename)
  - tx ty tz  : translation (metres)
  - qx qy qz qw : rotation quaternion

The poses are written as world-to-camera (w2c) transforms, matching the
convention of HorizonDataset.load_tum_pose_w2c().

Usage
─────
  python scripts/rosbag_2_dataset.py \\
      --bag   /path/to/rosbag2_xxx/ \\
      --output /mnt/holoagent/fsrvln/rgbd_datasets/my_scene/ \\
      --rgb-topic   /camera/color/image_raw \\
      --depth-topic /camera/depth/image_rect_raw \\
      --tf-parent   map \\
      --tf-child    camera_color_optical_frame \\
      [--fx 615.0 --fy 615.0 --cx 320.0 --cy 240.0 --width 640 --height 480] \\
      [--skip 1] \\
      [--max-depth-mm 10000]

Dependencies
────────────
  pip install rosbags opencv-python numpy scipy tqdm
  # rosbags: pure-Python ROS2 bag reader (no ROS2 installation required)
  #   https://github.com/rpng/rosbags
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# rosbags: pure-Python ROS2 bag parser (works with both .db3 and .mcap)
# ---------------------------------------------------------------------------
try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    sys.exit("ERROR: 'rosbags' package not found.\n" "Install it with:  pip install rosbags")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def stamp_to_sec(stamp) -> float:
    """Convert ROS2 stamp (sec + nanosec) or raw nanoseconds int to seconds."""
    if isinstance(stamp, int):
        return stamp * 1e-9
    return stamp.sec + stamp.nanosec * 1e-9


def msg_to_bgr(msg, typestore) -> np.ndarray:
    """Decode sensor_msgs/Image to a BGR uint8 numpy array."""
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    h, w, step = msg.height, msg.width, msg.step

    if enc in ("rgb8", "bgr8", "mono8"):
        img = data.reshape(h, w, -1)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    elif enc == "bgra8":
        img = data.reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif enc == "rgba8":
        img = data.reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError(f"Unsupported RGB encoding: {enc}")


def msg_to_depth_mm(msg, max_depth_mm: int = 10000) -> np.ndarray:
    """
    Decode sensor_msgs/Image depth frame to 16-bit uint16 in millimetres.
    Supports: 16UC1 (mm already), 32FC1 (metres).
    """
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width

    if enc == "16uc1":
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    elif enc == "32fc1":
        depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        depth = (depth_m * 1000.0).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported depth encoding: {enc}")

    depth = np.clip(depth, 0, max_depth_mm).astype(np.uint16)
    return depth


def extract_tf(bag_reader, typestore, parent_frame: str, child_frame: str):
    """
    Read all /tf messages and build an interpolable trajectory for a given
    parent→child frame pair.

    Returns
    -------
    timestamps : np.ndarray  shape (N,)   seconds
    translations : np.ndarray  shape (N, 3)
    quaternions  : np.ndarray  shape (N, 4)  (x, y, z, w)
    """
    timestamps, translations, quaternions = [], [], []

    tf_topics = ["/tf", "/tf_static"]
    available = {c.topic for c in bag_reader.connections}
    tf_topics = [t for t in tf_topics if t in available]

    if not tf_topics:
        return np.array([]), np.zeros((0, 3)), np.zeros((0, 4))

    for connection, ts_ns, raw in bag_reader.messages(
        connections=[c for c in bag_reader.connections if c.topic in tf_topics]
    ):
        msg = typestore.deserialize_cdr(raw, connection.msgtype)
        for transform in msg.transforms:
            if (
                transform.header.frame_id == parent_frame
                and transform.child_frame_id == child_frame
            ):
                t = transform.transform.translation
                r = transform.transform.rotation
                timestamps.append(stamp_to_sec(transform.header.stamp))
                translations.append([t.x, t.y, t.z])
                quaternions.append([r.x, r.y, r.z, r.w])

    if not timestamps:
        return np.array([]), np.zeros((0, 3)), np.zeros((0, 4))

    idx = np.argsort(timestamps)
    return (np.array(timestamps)[idx], np.array(translations)[idx], np.array(quaternions)[idx])


def interpolate_pose(query_ts: float, ts_arr, trans_arr, quat_arr, max_dt: float = 0.1):
    """
    Linearly interpolate translation, SLERP quaternion at query_ts.
    Returns (tx, ty, tz, qx, qy, qz, qw) or None if gap > max_dt.
    """
    if len(ts_arr) == 0:
        return None

    idx = np.searchsorted(ts_arr, query_ts, side="left")
    if idx == 0:
        idx = 1
    if idx >= len(ts_arr):
        idx = len(ts_arr) - 1

    t0, t1 = ts_arr[idx - 1], ts_arr[idx]
    if max(query_ts - t0, t1 - query_ts) > max_dt:
        return None

    alpha = (query_ts - t0) / max(t1 - t0, 1e-9)
    trans = (1 - alpha) * trans_arr[idx - 1] + alpha * trans_arr[idx]

    # SLERP for rotation
    rots = Rotation.from_quat(quat_arr[[idx - 1, idx]])
    q_interp = Slerp([0.0, 1.0], rots)(alpha).as_quat()  # (x,y,z,w)

    return (*trans.tolist(), *q_interp.tolist())


def write_camera_yaml(
    path: str, fx: float, fy: float, cx: float, cy: float, width: int, height: int
):
    """Write camera_info.yaml in the format expected by HorizonDataset."""
    content = {
        "Camera1.fx": float(fx),
        "Camera1.fy": float(fy),
        "Camera1.cx": float(cx),
        "Camera1.cy": float(cy),
        "Camera.width": int(width),
        "Camera.height": int(height),
    }
    with open(path, "w") as f:
        yaml.dump(content, f, default_flow_style=False)
    print(f"  Wrote camera intrinsics → {path}")


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_camera_info(bag_reader, typestore, camera_info_topic: str):
    """
    Try to read fx/fy/cx/cy/width/height from sensor_msgs/CameraInfo.
    Returns a dict or None.
    """
    available = {c.topic for c in bag_reader.connections}
    if camera_info_topic not in available:
        return None
    for conn, ts_ns, raw in bag_reader.messages(
        connections=[c for c in bag_reader.connections if c.topic == camera_info_topic]
    ):
        msg = typestore.deserialize_cdr(raw, conn.msgtype)
        K = msg.k  # 3x3 row-major (rosbags uses lowercase field names)
        return {
            "fx": K[0],
            "fy": K[4],
            "cx": K[2],
            "cy": K[5],
            "width": msg.width,
            "height": msg.height,
        }
    return None


def convert(
    bag_path: str,
    output_dir: str,
    rgb_topic: str,
    depth_topic: str,
    tf_parent: str,
    tf_child: str,
    camera_info_topic: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    skip: int,
    max_depth_mm: int,
    max_interp_gap: float,
    extrinsics=None,
):
    bag_path = Path(bag_path)
    output_dir = Path(output_dir)

    # Build 4×4 extrinsics matrix (axes transformation applied to each pose)
    if extrinsics is not None:
        ext_matrix = np.array(extrinsics, dtype=np.float64)
        if ext_matrix.shape != (4, 4):
            sys.exit(f"ERROR: 'extrinsics' must be a 4×4 matrix, got shape {ext_matrix.shape}")
        print(f"  Extrinsics (axes transform):\n{ext_matrix}")
    else:
        ext_matrix = np.eye(4, dtype=np.float64)
        print("  Extrinsics: not defined, using identity matrix")

    img_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    img_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Pick typestore based on bag format
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    print(f"Opening bag: {bag_path}")
    with Reader(str(bag_path)) as reader:
        available_topics = {c.topic for c in reader.connections}
        print(f"  Available topics: {sorted(available_topics)}")

        # Camera intrinsics
        if camera_info_topic and camera_info_topic in available_topics:
            print(f"  Reading camera info from {camera_info_topic} …")
            info = extract_camera_info(reader, typestore, camera_info_topic)
            if info:
                fx, fy, cx, cy = info["fx"], info["fy"], info["cx"], info["cy"]
                width, height = info["width"], info["height"]
                print(
                    f"  Intrinsics from bag: fx={fx:.2f} fy={fy:.2f} "
                    f"cx={cx:.2f} cy={cy:.2f} {width}×{height}"
                )

        write_camera_yaml(
            str(output_dir / "camera_info.yaml"),
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        )

        # TF trajectory
        print(f"  Extracting TF: '{tf_parent}' → '{tf_child}' …")
        ts_tf, trans_tf, quat_tf = extract_tf(reader, typestore, tf_parent, tf_child)

        if len(ts_tf) == 0:
            print(
                f"  WARNING: No TF transforms found for "
                f"'{tf_parent}' → '{tf_child}'.\n"
                "  Poses will be written as identity (0 0 0 0 0 0 1).\n"
                "  Supply correct --tf-parent / --tf-child, or edit "
                "poses.txt manually."
            )

        # RGB + depth messages
        rgb_conns = [c for c in reader.connections if c.topic == rgb_topic]
        depth_conns = [c for c in reader.connections if c.topic == depth_topic]

        if not rgb_conns:
            sys.exit(f"ERROR: RGB topic '{rgb_topic}' not found in bag.")
        if not depth_conns:
            sys.exit(f"ERROR: Depth topic '{depth_topic}' not found in bag.")

        # Collect depth timestamps → frames as a dict for sync
        print("  Indexing depth frames …")
        depth_cache: dict[float, np.ndarray] = {}
        for conn, ts_ns, raw in reader.messages(connections=depth_conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            ts = stamp_to_sec(msg.header.stamp)
            depth_cache[ts] = msg_to_depth_mm(msg, max_depth_mm)

        depth_ts_arr = np.array(sorted(depth_cache.keys()))

        # Process RGB frames
        pose_lines: list[str] = []
        saved = 0
        skipped = 0

        print("  Processing RGB frames …")
        rgb_iter = reader.messages(connections=rgb_conns)
        total_rgb = sum(c.msgcount for c in rgb_conns)

        for frame_idx, (conn, ts_ns, raw) in enumerate(
            tqdm(rgb_iter, total=total_rgb, unit="frame")
        ):

            if frame_idx % (skip + 1) != 0:
                skipped += 1
                continue

            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            ts = stamp_to_sec(msg.header.stamp)
            ts_str = f"{ts:.4f}"

            # RGB image
            try:
                bgr = msg_to_bgr(msg, typestore)
            except ValueError as e:
                print(f"  [WARN] frame {frame_idx}: {e} — skipping")
                continue

            # Sync depth: find nearest depth frame
            if len(depth_ts_arr) > 0:
                nearest_idx = np.argmin(np.abs(depth_ts_arr - ts))
                nearest_ts = depth_ts_arr[nearest_idx]
                if abs(nearest_ts - ts) > 0.05:  # 50 ms tolerance
                    continue  # no depth match
                depth_frame = depth_cache[nearest_ts]
            else:
                depth_frame = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.uint16)

            # Camera pose from TF
            if len(ts_tf) > 0:
                pose = interpolate_pose(ts, ts_tf, trans_tf, quat_tf, max_interp_gap)
            else:
                pose = None

            if pose is None:
                # Write identity pose
                pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

            tx, ty, tz, qx, qy, qz, qw = pose

            # Apply axes transformation (extrinsics)
            rot_mat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            pose_mat = np.eye(4)
            pose_mat[:3, :3] = rot_mat
            pose_mat[:3, 3] = [tx, ty, tz]
            pose_mat = ext_matrix @ pose_mat
            tx, ty, tz = pose_mat[:3, 3].tolist()
            qx, qy, qz, qw = Rotation.from_matrix(pose_mat[:3, :3]).as_quat().tolist()

            pose_lines.append(
                f"{ts_str} {tx:.6f} {ty:.6f} {tz:.6f} " f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
            )

            # Save frames
            cv2.imwrite(str(img_dir / f"{ts_str}.png"), bgr)
            cv2.imwrite(str(depth_dir / f"{ts_str}.png"), depth_frame)
            saved += 1

    # Write to poses.txt
    poses_path = output_dir / "poses.txt"
    with open(poses_path, "w") as f:
        f.write("\n".join(pose_lines) + "\n")

    print(f"\nDone. Saved {saved} frames ({skipped} skipped by --skip).")
    print(f"Output dataset → {output_dir}")
    print(f"  images/ : {len(list(img_dir.glob('*.png')))} files")
    print(f"  depth/  : {len(list(depth_dir.glob('*.png')))} files")
    print(f"  poses.txt: {len(pose_lines)} entries")
    print("  camera_info.yaml: written")

    print("\n-- Next step --")
    print("  Edit fsr_vln/config/semantic_scene_reconstruction_custom.yaml:")
    print(f"    main.dataset_path: {output_dir.parent}/")
    print(f"    main.scene_id:     {output_dir.name}")
    print("  Then run:")
    print("    cd fsr_vln/")
    print(
        "    python application/semantic_scene_reconstruction_offline/"
        "semantic_scene_reconstruction.py "
        "--config-name=semantic_scene_reconstruction_custom"
    )


# ---------------------------------------------------------------------------
# Config file loader
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "rgb_topic": "/camera/color/image_raw",
    "depth_topic": "/camera/depth/image_rect_raw",
    "camera_info_topic": "/camera/color/camera_info",
    "tf_parent": "map",
    "tf_child": "camera_color_optical_frame",
    "fx": 615.0,
    "fy": 615.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640,
    "height": 480,
    "skip": 0,
    "max_depth_mm": 10000,
    "max_interp_gap": 0.1,
    "extrinsics": None,  # 4×4 axes-transformation matrix; None → identity
}


def load_config(config_path: str) -> dict:
    """Load convert.yaml and merge with defaults."""
    config_path = Path(config_path)
    if not config_path.exists():
        sys.exit(f"ERROR: Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    # Merge defaults for any missing keys
    for key, val in _DEFAULTS.items():
        cfg.setdefault(key, val)

    for required in ("bag", "output"):
        if not cfg.get(required):
            sys.exit(f"ERROR: '{required}' is required in {config_path}")

    return cfg


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Convert ROS2 bag to Horizon RGB-D dataset format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(Path(__file__).parent / "convert.yaml"),
        help="Path to the YAML config file (default: scripts/convert.yaml)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"Using config: {args.config}")

    convert(
        bag_path=cfg["bag"],
        output_dir=cfg["output"],
        rgb_topic=cfg["rgb_topic"],
        depth_topic=cfg["depth_topic"],
        tf_parent=cfg["tf_parent"],
        tf_child=cfg["tf_child"],
        camera_info_topic=cfg["camera_info_topic"],
        fx=float(cfg["fx"]),
        fy=float(cfg["fy"]),
        cx=float(cfg["cx"]),
        cy=float(cfg["cy"]),
        width=int(cfg["width"]),
        height=int(cfg["height"]),
        skip=int(cfg["skip"]),
        max_depth_mm=int(cfg["max_depth_mm"]),
        max_interp_gap=float(cfg["max_interp_gap"]),
        extrinsics=cfg.get("extrinsics"),
    )
