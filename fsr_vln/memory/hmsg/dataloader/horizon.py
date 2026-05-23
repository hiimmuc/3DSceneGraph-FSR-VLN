import os

import numpy as np
import open3d as o3d
import yaml
from memory.hmsg.dataloader.generic import RGBDDataset
from PIL import Image
from scipy.spatial.transform import Rotation as R


class HorizonDataset(RGBDDataset):
    """Dataset class for the Horizon dataset.

    This class provides an interface to load RGB-D data samples from the
    Horizon dataset."""

    def __init__(self, cfg):
        """Args:

        root_dir: Path to the root directory containing the dataset.
        mode: "train", "val", or "test" depending on the data split.
        transforms: Optional transformations to apply to the data."""
        super(HorizonDataset, self).__init__(cfg)
        self.root_dir = cfg["root_dir"]
        self.transforms = cfg["transforms"]
        self.depth_cut = float(cfg["depth_cut"])
        pose_name = "poses"
        camera_config_path = "camera_info.yaml"
        self.rgb_intrinsics, self.depth_intrinsics = self.load_camera_params(
            os.path.join(self.root_dir, camera_config_path)
        )
        self.scale = 1000.0
        if pose_name is not None and os.path.exists(
            os.path.join(self.root_dir, f"{pose_name}.txt")
        ):
            camtoworlds, ts_list = self.load_tum_pose_w2c(
                os.path.join(self.root_dir, f"{pose_name}.txt")
            )
        elif os.path.exists(os.path.join(self.root_dir, "CameraTrajectory.txt")):
            camtoworlds, ts_list = self.load_tum_pose(
                os.path.join(self.root_dir, "CameraTrajectory.txt")
            )
        elif os.path.exists(os.path.join(self.root_dir, "cam_1_poses_updated.txt")):
            camtoworlds, ts_list = self.load_tum_pose(
                os.path.join(self.root_dir, "cam_1_poses_updated.txt")
            )
        else:
            assert False, "No pose file found in the directory"

        if int(ts_list[0]) != ts_list[0]:
            self.ts_list = ["{:.4f}".format(ts) for ts in ts_list]
        else:
            self.ts_list = [str(int(ts)) for ts in ts_list]

        self.camtoworlds = camtoworlds
        self.indices = np.arange(len(self.ts_list))

        if int(ts_list[0]) != ts_list[0]:
            self.image_paths = [
                os.path.join(self.root_dir, "images", f"{float(ts):.4f}.png")
                for ts in self.ts_list
            ]
            self.depth_paths = [
                os.path.join(self.root_dir, "depth", f"{float(ts):.4f}.png") for ts in self.ts_list
            ]
        else:
            self.image_paths = [
                os.path.join(self.root_dir, "color", f"{int(ts):05d}.png") for ts in self.ts_list
            ]
            self.depth_paths = [
                os.path.join(self.root_dir, "depth", f"{int(ts):05d}.png") for ts in self.ts_list
            ]
        self.frameId2imgPath = self.image_paths

    def _get_data_list(self):
        # dummy function to satisfy the abstract method requirement in the base class
        # rgb_data_list = []
        # depth_data_list = []
        # pose_data_list = []
        pass

    def __len__(self):
        return len(self.indices)

    def get_camera_intrinsics(self):
        return self.depth_intrinsics

    def load_camera_params(self, config_path: str, camera_name: str = None) -> np.ndarray:

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        K = np.eye(3)
        if "Camera.fx" in config.keys() and isinstance(config["Camera.fx"], set):
            K[0, 0] = next(iter(config["Camera.fx"]))
            K[1, 1] = next(iter(config["Camera.fy"]))
            K[0, 2] = next(iter(config["Camera.cx"]))
            K[1, 2] = next(iter(config["Camera.cy"]))
        else:
            K[0, 0] = config["Camera1.fx"]
            K[1, 1] = config["Camera1.fy"]
            K[0, 2] = config["Camera1.cx"]
            K[1, 2] = config["Camera1.cy"]

        depth_K = K.copy()
        return K, depth_K

    def get_frame_pose(self, idx: int) -> np.ndarray:
        return self.camtoworlds[idx, :]

    def load_tum_pose_w2c(self, path: str) -> np.ndarray:
        """
        Load ego pose from file.

        Args:
            path (str): Path to ego pose file.

        Returns:
            np.ndarray: Ego pose, tum format, ts tx ty tz qx qy qz qw
        """
        tum_pose_raw = np.loadtxt(path)
        tum_pose_raw = tum_pose_raw[tum_pose_raw[:, 0].argsort()]
        ts_list = []
        T_list = []
        for pose in tum_pose_raw:
            ts, tx, ty, tz, qx, qy, qz, qw = pose
            quat = [qx, qy, qz, qw]
            rot_matrix = R.from_quat(quat).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rot_matrix
            T[:3, 3] = [tx, ty, tz]
            c2w = np.linalg.inv(T)
            T_list.append(c2w)
            ts_list.append(ts)

        camtoworlds = np.array(T_list)
        return camtoworlds, ts_list

    def load_tum_pose(self, path: str) -> np.ndarray:
        """
        Load ego pose from file.

        Args:
            path (str): Path to ego pose file.

        Returns:
            np.ndarray: Ego pose, tum format, ts tx ty tz qx qy qz qw
        """
        tum_pose_raw = np.loadtxt(path)
        tum_pose_raw = tum_pose_raw[tum_pose_raw[:, 0].argsort()]
        ts_list = []
        T_list = []
        for pose in tum_pose_raw:
            ts, tx, ty, tz, qw, qx, qy, qz = pose
            quat = [qx, qy, qz, qw]
            rot_matrix = R.from_quat(quat).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rot_matrix
            T[:3, 3] = [tx, ty, tz]
            T_list.append(T)
            ts_list.append(ts)

        camtoworlds = np.array(T_list)
        return camtoworlds, ts_list

    def __getitem__(self, idx):
        """
        Get a data sample based on the given index.

        Args:
            idx: Index of the data sample.

        Returns:
            RGB image, depth image as PIL images, pose, and camera intrinsics.
        """
        rgb_path = self.image_paths[idx]
        depth_path = self.depth_paths[idx]
        pose = self.get_frame_pose(idx)

        T_switch_axis = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        pose = T_switch_axis @ pose
        rgb_image = self._load_image(rgb_path)
        depth_image = self._load_depth(depth_path)

        depth = np.array(depth_image)
        clip_depth_mask = depth > self.depth_cut * 1000
        depth[clip_depth_mask] = 0
        depth_image = Image.fromarray(depth)

        if self.transforms is not None:
            rgb_image = self.transforms(rgb_image)
            depth_image = self.transforms(depth_image)

        return rgb_image, depth_image, pose, self.rgb_intrinsics, self.depth_intrinsics

    def _load_image(self, path):
        """Load RGB image from the given path."""
        return Image.open(path)

    def _load_depth(self, path):
        """Load depth image from the given path."""
        return Image.open(path)

    def _create_pcd(self, rgb, depth, camera_pose=None):
        """Create a point cloud from RGB-D images."""
        rgb = np.array(rgb)
        depth = np.array(depth)
        rgb = np.array(Image.fromarray(rgb).resize((depth.shape[1], depth.shape[0])))
        camera_matrix = self.depth_intrinsics
        depth_img = depth.astype(np.float32) / 1000.0
        x, y = np.meshgrid(np.arange(depth_img.shape[1]), np.arange(depth_img.shape[0]))
        mask = depth_img > 0
        x = x[mask]
        y = y[mask]
        depth_img = depth_img[mask]
        X = (x - camera_matrix[0, 2]) * depth_img / camera_matrix[0, 0]
        Y = (y - camera_matrix[1, 2]) * depth_img / camera_matrix[1, 1]
        Z = depth_img
        pcd = np.hstack(
            ([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), np.ones((X.shape[0], 1))])
        )

        if camera_pose is not None:
            pcd = np.dot(camera_pose, pcd.T).T
            pcd = pcd[:, :3] / pcd[:, 3:]
        colors = rgb.reshape(-1, 3) / 255
        colors = colors[mask.reshape(-1)]
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcd)
        pcd1.colors = o3d.utility.Vector3dVector(colors)
        return pcd1
