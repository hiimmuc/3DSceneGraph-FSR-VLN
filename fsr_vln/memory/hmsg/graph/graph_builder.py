"""GraphBuilder — constructs an HMSG Graph from raw RGB-D data.

Owns model loading (CLIP, SAM), dataset I/O, and the full build pipeline.
Call ``GraphBuilder(cfg).build(save_path)`` to create a fully populated
``Graph`` instance ready for querying.
"""

import os
import shutil
from copy import deepcopy  # noqa: F401 (kept for consistency with original)
from datetime import datetime
from typing import List, Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.utility as utility
import open_clip
import torch
from application.download_checkpoints import ensure_checkpoints
from memory.hmsg.dataloader.custom_dataset import CustomDataset
from memory.hmsg.graph.floor import Floor
from memory.hmsg.graph.graph import Graph
from memory.hmsg.graph.navigation_graph import NavigationGraph
from memory.hmsg.graph.object import Object
from memory.hmsg.graph.room import Room
from memory.hmsg.graph.view import View
from memory.hmsg.utils.clip_utils import get_img_feats
from memory.hmsg.utils.constants import CLIP_DIM
from memory.hmsg.utils.graph_utils import (
    check_object_in_view,
    compute_room_embeddings,
    distance_transform,
    feats_denoise_dbscan,
    filter_point_cloud,
    find_intersection_share,
    hierarchical_merge,
    map_grid_to_point_cloud,
    pcd_denoise,
    seq_merge,
    visualize_pcd_clusters,
)
from memory.hmsg.utils.label_feats import CSV_LABEL_REGISTRY, get_label_feats
from omegaconf import DictConfig
from perception.models.sam_clip_feats_extractor import extract_feats_per_pixel
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sklearn.cluster import DBSCAN
from tqdm import tqdm

utility.set_verbosity_level(utility.VerbosityLevel.Error)
matplotlib.use("Agg")

# pylint: disable=all


class GraphBuilder:
    """Builds a populated ``Graph`` from raw RGB-D data.

    Args:
        cfg: Hydra/OmegaConf configuration.  Must contain a ``pipeline``
             block (build mode).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_clip_model()
        self._init_directories()
        self._load_dataset()

        if cfg.main.slow_reasoning:
            # Slow-reasoning mode uses VLM; SAM is not needed.
            pass
        else:
            self._load_sam_model()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_clip_model(self) -> None:
        clip_type = self.cfg.models.clip.type
        checkpoint = str(self.cfg.models.clip.checkpoint)
        ensure_checkpoints([checkpoint])
        _MODEL_MAP = {
            "ViT-L/14": ("ViT-L-14", {}),
            "ViT-H-14": ("ViT-H-14", {}),
            "ViT-B-32": ("ViT-B-32", {"precision": "fp16"}),
            "MobileCLIP2-S4": ("MobileCLIP2-S4", {}),
        }
        if clip_type not in _MODEL_MAP:
            raise ValueError(f"Unsupported CLIP model type: {clip_type}")
        model_name, extra_kwargs = _MODEL_MAP[clip_type]
        print(f"[BUILD] Loading CLIP '{model_name}' from {checkpoint}")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=checkpoint, device=self.device, **extra_kwargs
        )
        self.clip_feat_dim = CLIP_DIM[model_name]
        self.clip_model.eval()

    def _init_directories(self) -> None:
        self.graph_tmp_folder = os.path.join(self.cfg.main.save_path, "tmp")
        os.makedirs(self.graph_tmp_folder, exist_ok=True)
        self._get_label_csv()

    def _get_label_csv(self) -> None:
        """Copy label CSV to save_path if needed."""
        obj_labels = self.cfg.pipeline.obj_labels
        if obj_labels in CSV_LABEL_REGISTRY:
            csv_filename, _ = CSV_LABEL_REGISTRY[obj_labels]
            labels_src_dir = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labels")
            )
            src = os.path.join(labels_src_dir, csv_filename)
            dst = os.path.join(self.cfg.main.save_path, csv_filename)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    def _load_dataset(self) -> None:
        dataset_cfg = {
            "root_dir": self.cfg.main.dataset_path,
            "transforms": None,
            "depth_cut": self.cfg.main.depth_cut,
        }
        self.dataset = CustomDataset(dataset_cfg)

    def _load_sam_model(self) -> None:
        model_type = self.cfg.models.sam.type
        checkpoint = str(self.cfg.models.sam.checkpoint)
        ensure_checkpoints([checkpoint])
        print(f"[BUILD] Loading SAM '{model_type}' from {checkpoint}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.cfg.models.sam.points_per_side,
            pred_iou_thresh=self.cfg.models.sam.pred_iou_thresh,
            points_per_batch=self.cfg.models.sam.points_per_batch,
            stability_score_thresh=self.cfg.models.sam.stability_score_thresh,
            crop_n_layers=self.cfg.models.sam.crop_n_layers,
            min_mask_region_area=self.cfg.models.sam.min_mask_region_area,
        )
        self.sam.eval()

    # ------------------------------------------------------------------
    # Public build entry point
    # ------------------------------------------------------------------

    def build(self, save_path: str) -> Graph:
        """Run the full build pipeline and return a populated Graph.

        Steps:
            1. create_feature_map  — RGB-D → per-point CLIP features
            2. save_masked_pcds / save_full_pcd / save_full_pcd_feats
            3. segment_floors_manually
            4. segment_hmsg_room  (per floor)
            5. segment_hmsg_objects
            6. (optional) merge_objects
            7. create_graph / create_nav_graph
            8. save_hmsg_graph

        Args:
            save_path: Root directory for all output artifacts.

        Returns:
            Fully populated Graph.
        """
        graph = Graph()

        print("[BUILD] ── Stage 1/7: building feature map …")
        self.create_feature_map(graph)

        print("[BUILD] ── Stage 2/7: saving intermediate artifacts …")
        self.save_masked_pcds(graph, path=save_path, state="both")
        self.save_full_pcd(graph, path=save_path, visualize_clusters=False)
        self.save_full_pcd_feats(graph, path=save_path)

        print("[BUILD] ── Stage 3/7: segmenting floors …")
        self.segment_floors_manually(graph, save_path)

        print("[BUILD] ── Stage 4/7: segmenting rooms …")
        for floor in graph.floors:
            self.segment_hmsg_room(graph, floor, save_path)

        print("[BUILD] ── Stage 5/7: segmenting / identifying objects …")
        self.segment_hmsg_objects(graph, save_path)
        print(f"  ├── total objects: {len(graph.objects)}")

        if self.cfg.pipeline.merge_objects_graph:
            print("[BUILD] ── Stage 5b: merging nearby same-name objects …")
            for room in tqdm(graph.rooms):
                before = len(room.objects)
                room.merge_objects()
                after = len(room.objects)
                if before != after:
                    tqdm.write(f"  room {room.room_id}: {before} → {after} objects")

        print("[BUILD] ── Stage 6/7: building topology graph + nav graph …")
        self.create_graph(graph)
        self.create_nav_graph(graph)

        print("[BUILD] ── Stage 7/7: persisting graph …")
        now_str = datetime.now().strftime("%Y%m%d%H%M%S")
        graph_save_dir = os.path.join(save_path, f"graph_{now_str}")
        graph.save_hmsg_graph(graph_save_dir)

        print(
            f"[BUILD] Done — floors={len(graph.floors)}, rooms={len(graph.rooms)}, "
            f"views={len(graph.views)}, objects={len(graph.objects)}"
        )
        return graph

    # ------------------------------------------------------------------
    # Feature-map construction (stages 1–3)
    # ------------------------------------------------------------------

    def create_feature_map(self, graph: Graph) -> None:
        """Accumulate RGB-D PCD and extract per-point CLIP features into *graph*."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded.")

        # === Stage 1: accumulate RGB-D point cloud ===
        print("[BUILD]   ├── Stage 1/3: accumulating RGB-D point cloud …")
        for i in tqdm(
            range(0, len(self.dataset), self.cfg.pipeline.skip_frames),
            desc="  Building full PCD",
        ):
            rgb_image, depth_image, pose, _, _ = self.dataset[i]
            graph.full_pcd += self.dataset.create_pcd(rgb_image, depth_image, pose, idx=i)
        print(f"[BUILD]   │   raw PCD points: {len(graph.full_pcd.points)}")

        if self.cfg.pipeline.get("enable_pcd_filtering", True):
            print("[BUILD]   │   filtering point cloud …")
            filter_config = {
                "enable_voxel_downsampling": self.cfg.pipeline.get(
                    "enable_voxel_downsampling", True
                ),
                "voxel_size": self.cfg.pipeline.voxel_size,
                "enable_dbscan_filtering": self.cfg.pipeline.get("enable_dbscan_filtering", True),
                "dbscan_eps": self.cfg.pipeline.get("dbscan_eps", 0.01),
                "dbscan_min_points": self.cfg.pipeline.get("dbscan_min_points", 100),
                "enable_radius_outlier_filtering": self.cfg.pipeline.get(
                    "enable_radius_outlier_filtering", True
                ),
                "radius_nb_points": self.cfg.pipeline.get("radius_nb_points", 1000),
                "radius_distance": self.cfg.pipeline.get("radius_distance", 1.0),
            }
            graph.full_pcd = filter_point_cloud(graph.full_pcd, filter_config)

        self.save_full_pcd(graph, path=self.cfg.main.save_path, visualize_clusters=False)

        # === Stage 2: per-point CLIP feature extraction ===
        print("[BUILD]   ├── Stage 2/3: extracting per-point CLIP features …")
        locs_in = np.array(graph.full_pcd.points)
        tree_pcd = cKDTree(locs_in)
        n_points = locs_in.shape[0]
        counter = torch.zeros((n_points, 1), device="cpu")
        sum_features = torch.zeros((n_points, self.clip_feat_dim), device="cpu")

        frames_pcd = []
        for i in tqdm(
            range(0, len(self.dataset), self.cfg.pipeline.skip_frames),
            desc="  Computing per-point features",
        ):
            rgb_image, depth_image, pose, _, _ = self.dataset[i]
            if rgb_image.size != depth_image.size:
                rgb_image = rgb_image.resize(depth_image.size)
            F_2D, F_masks, masks, F_g = extract_feats_per_pixel(
                np.array(rgb_image),
                self.mask_generator,
                self.clip_model,
                self.preprocess,
                clip_feat_dim=self.clip_feat_dim,
                bbox_margin=self.cfg.pipeline.clip_bbox_margin,
                masked_weight=self.cfg.pipeline.clip_masked_weight,
            )
            F_2D = F_2D.cpu()
            pcd = self.dataset.create_pcd(rgb_image, depth_image, pose, idx=i)
            masks_3d = self.dataset.create_3d_masks(
                masks,
                depth_image,
                graph.full_pcd,
                tree_pcd,
                pose,
                i,
                down_size=self.cfg.pipeline.voxel_size,
                filter_distance=self.cfg.pipeline.max_mask_distance,
            )
            frames_pcd.append(masks_3d)
            depth_mask = torch.from_numpy(np.array(depth_image) > 0)
            F_2D_masked = F_2D[depth_mask]
            _, idx = tree_pcd.query(np.asarray(pcd.points), k=1, workers=-1)
            sum_features[idx] += F_2D_masked
            counter[idx] += 1

        counter[counter == 0] = 1e-5
        sum_features = sum_features / counter
        graph.full_feats_array = sum_features.cpu().numpy()
        print(f"[BUILD]   │   feature array: {graph.full_feats_array.shape}")
        del sum_features, counter
        torch.cuda.empty_cache()

        # === Stage 3: 3D mask merging and feature aggregation ===
        print("[BUILD]   └── Stage 3/3: merging masks and aggregating features …")
        self._merge_masks(graph, frames_pcd)
        self._aggregate_mask_features(graph, tree_pcd)

    def _merge_masks(self, graph: Graph, frames_pcd) -> None:
        merge_type = self.cfg.pipeline.merge_type.lower()
        if merge_type == "hierarchical":
            tqdm.write("  [BUILD] merging hierarchically …")
            graph.mask_pcds = hierarchical_merge(
                frames_pcd,
                self.cfg.pipeline.init_overlap_thresh,
                self.cfg.pipeline.overlap_thresh_factor,
                self.cfg.pipeline.voxel_size,
                self.cfg.pipeline.iou_thresh,
            )
        elif merge_type == "sequential":
            tqdm.write("  [BUILD] merging sequentially …")
            graph.mask_pcds = seq_merge(
                frames_pcd,
                self.cfg.pipeline.init_overlap_thresh,
                self.cfg.pipeline.voxel_size,
                self.cfg.pipeline.iou_thresh,
            )
        else:
            raise ValueError(
                f"Invalid merge_type: '{merge_type}'. Must be 'hierarchical' or 'sequential'."
            )
        original_count = len(graph.mask_pcds)
        min_pts = self.cfg.pipeline.get("min_mask_points", 10)
        graph.mask_pcds = [
            p for p in graph.mask_pcds if not p.is_empty() and len(p.points) >= min_pts
        ]
        removed = original_count - len(graph.mask_pcds)
        if removed > 0:
            print(f"[BUILD]   filtered {removed} masks below {min_pts} pts")

    def _aggregate_mask_features(self, graph: Graph, tree_pcd: cKDTree) -> None:
        masks_feats = []
        voxel_size = self.cfg.pipeline.voxel_size
        dist_threshold = self.cfg.pipeline.get("mask_feature_dist_threshold", 0.8)
        downsampled = [m.voxel_down_sample(voxel_size) for m in graph.mask_pcds]
        pts_per_mask = [np.asarray(m.points) for m in downsampled]
        mask_lengths = [len(p) for p in pts_per_mask]
        if sum(mask_lengths) > 0:
            all_pts = np.vstack([p for p in pts_per_mask if len(p) > 0])
            all_dist, all_idx = tree_pcd.query(all_pts, k=1, workers=-1)
        else:
            all_dist, all_idx = np.array([]), np.array([])
        offset = 0
        for i, pts in enumerate(pts_per_mask):
            n = mask_lengths[i]
            if n == 0:
                masks_feats.append(
                    np.zeros((1, self.clip_feat_dim), dtype=graph.full_feats_array.dtype)
                )
                continue
            dist = all_dist[offset : offset + n]
            idx = all_idx[offset : offset + n]
            offset += n
            valid = dist <= dist_threshold
            n_valid = int(valid.sum())
            if n - n_valid > 0 and n_valid > 0:
                tqdm.write(f"    mask[{i}]: kept {n_valid}/{n} pts (dist ≤ {dist_threshold:.2f}m)")
            if n_valid == 0:
                masks_feats.append(
                    np.zeros((1, self.clip_feat_dim), dtype=graph.full_feats_array.dtype)
                )
                continue
            feats = graph.full_feats_array[idx[valid]]
            feats = np.nan_to_num(feats)
            feats = feats_denoise_dbscan(feats, eps=0.01, min_points=100)
            masks_feats.append(feats)
        graph.mask_feats = masks_feats
        print(
            f"[BUILD]   mask features: {len(graph.mask_feats)} from {len(graph.mask_pcds)} masks"
        )
        assert len(graph.mask_pcds) == len(graph.mask_feats), "Mask-feature mismatch!"

    # ------------------------------------------------------------------
    # Floor / room / object segmentation
    # ------------------------------------------------------------------

    def segment_floors_manually(
        self, graph: Graph, path: str, flip_zy: bool = False, mid_points: List = []
    ):
        """Detect floor levels from PCD histogram and populate graph.floors."""
        downpcd = graph.full_pcd.voxel_down_sample(voxel_size=0.05)
        if flip_zy:
            downpcd.points = o3d.utility.Vector3dVector(np.array(downpcd.points)[:, [0, 2, 1]])
            downpcd.transform(np.eye(4) * np.array([1, 1, -1, 1]))
        downpcd_pts = np.asarray(downpcd.points)
        print(f"[BUILD] floor detection: PCD shape {downpcd_pts.shape}")

        resolution = 0.01
        bins = int(np.abs(np.max(downpcd_pts[:, 1]) - np.min(downpcd_pts[:, 1])) / resolution)
        z_hist = np.histogram(downpcd_pts[:, 1], bins=bins)
        z_hist_smooth = gaussian_filter1d(z_hist[0], sigma=2)
        distance = 0.2 / resolution
        min_peak_height = np.percentile(z_hist_smooth, 90)
        peaks, _ = find_peaks(z_hist_smooth, distance=distance, height=min_peak_height)

        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r")
            plt.savefig(os.path.join(self.graph_tmp_folder, "floor_histogram.png"))

        peaks_locations = z_hist[1][peaks]
        clustering = DBSCAN(eps=1, min_samples=1).fit(peaks_locations.reshape(-1, 1))
        labels = clustering.labels_

        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r")
            for i in range(len(np.unique(labels))):
                plt.plot(z_hist[1][peaks[labels == i]], z_hist_smooth[peaks[labels == i]], "o")
            plt.savefig(os.path.join(self.graph_tmp_folder, "floor_histogram_cluster.png"))

        clustered_peaks = []
        for i in range(len(np.unique(labels))):
            p = peaks[labels == i]
            n_top = 1 if (i == 0 or i == len(np.unique(labels)) - 1) else 2
            top_p = p[np.argsort(z_hist_smooth[p])[-n_top:]].tolist()
            top_p = [z_hist[1][pp] for pp in top_p]
            clustered_peaks.extend(top_p)
        clustered_peaks = np.sort(clustered_peaks)

        adjusted_peaks = []
        for i in range(len(clustered_peaks) - 1):
            adjusted_peaks.append(clustered_peaks[i])
            if clustered_peaks[i + 1] - clustered_peaks[i] >= 2.5:
                adjusted_peaks.append(clustered_peaks[i + 1] - 0.2)
        adjusted_peaks.append(clustered_peaks[-1])
        clustered_peaks = np.array(adjusted_peaks)

        floors = [
            [clustered_peaks[i], clustered_peaks[i + 1]] for i in range(len(clustered_peaks) - 1)
        ]
        if not floors:
            floors.append([z_hist[1].min().item(), z_hist[1].max().item()])
        floors[0][0] = (floors[0][0] + np.min(downpcd_pts[:, 1])) / 2
        floors[-1][1] = np.max(downpcd_pts[:, 1])
        print(f"[BUILD]   ├── detected {len(floors)} floor(s): {floors}")

        for i, floor_range in enumerate(floors):
            floor_obj = Floor(str(i), name=f"floor_{i}")
            floor_pcd = graph.full_pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(-np.inf, floor_range[0], -np.inf),
                    max_bound=(np.inf, floor_range[1], np.inf),
                )
            )
            bbox = floor_pcd.get_axis_aligned_bounding_box()
            floor_obj.vertices = np.asarray(bbox.get_box_points())
            floor_obj.pcd = floor_pcd
            floor_obj.floor_zero_level = float(np.min(np.array(floor_pcd.points)[:, 1]))
            floor_obj.floor_height = floor_range[1] - floor_obj.floor_zero_level
            graph.floors.append(floor_obj)
        return floors

    def segment_hmsg_room(self, graph: Graph, floor: Floor, path: str) -> None:
        """Segment a floor into rooms and populate graph.rooms."""
        tmp_floor_path = os.path.join(self.graph_tmp_folder, floor.floor_id)
        os.makedirs(tmp_floor_path, exist_ok=True)

        floor_pcd = floor.pcd
        xyz = np.asarray(floor_pcd.points)
        xyz_full = xyz.copy()
        floor_zero_level = floor.floor_zero_level
        floor_height = floor.floor_height
        print(
            f"[BUILD] segmenting rooms on floor {floor.floor_id}: "
            f"zero_level={floor_zero_level:.2f}, height={floor_height:.2f}"
        )

        xyz = xyz[
            (xyz[:, 1] < floor_zero_level + floor_height - 0.3)
            & (xyz[:, 1] >= floor_zero_level + 0.3)
        ]
        xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + floor_height - 0.2]
        pcd_2d = xyz[:, [0, 2]]
        xyz_full = xyz_full[:, [0, 2]]

        grid_size = (
            int(np.max(pcd_2d[:, 0]) - np.min(pcd_2d[:, 0])) + 1,
            int(np.max(pcd_2d[:, 1]) - np.min(pcd_2d[:, 1])) + 1,
        )
        resolution = self.cfg.pipeline.grid_resolution
        num_bins = (int(grid_size[1] // resolution) + 1, int(grid_size[0] // resolution) + 1)

        hist, _, _ = np.histogram2d(pcd_2d[:, 1], pcd_2d[:, 0], bins=num_bins)
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.imshow(hist, cmap="jet", origin="lower")
            plt.colorbar()
            plt.savefig(os.path.join(tmp_floor_path, "2D_histogram.png"))

        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist = cv2.GaussianBlur(hist, (5, 5), 1)
        _, walls_skeleton = cv2.threshold(hist, 0.25 * np.max(hist), 255, cv2.THRESH_BINARY)
        walls_skeleton = cv2.copyMakeBorder(
            walls_skeleton, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        walls_skeleton = cv2.morphologyEx(walls_skeleton, cv2.MORPH_CLOSE, kernel, iterations=1)

        hist_full, _, _ = np.histogram2d(xyz_full[:, 1], xyz_full[:, 0], bins=num_bins)
        hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist_full = cv2.GaussianBlur(hist_full, (21, 21), 2)
        _, outside_boundary = cv2.threshold(hist_full, 0, 255, cv2.THRESH_BINARY)
        outside_boundary = cv2.copyMakeBorder(
            outside_boundary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        outside_boundary = cv2.morphologyEx(
            outside_boundary, cv2.MORPH_CLOSE, kernel, iterations=3
        )
        contours, _ = cv2.findContours(
            outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outside_boundary = np.zeros_like(outside_boundary)
        cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1)

        if self.cfg.pipeline.save_intermediate_results:
            for name, img in [
                ("walls_skeleton", walls_skeleton),
                ("outside_boundary", outside_boundary),
            ]:
                plt.figure()
                plt.imshow(img, cmap="gray", origin="lower")
                plt.savefig(os.path.join(tmp_floor_path, f"{name}.png"))

        full_map = cv2.bitwise_or(
            walls_skeleton, cv2.bitwise_not(outside_boundary.astype(np.uint8))
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, kernel, iterations=2)
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.imshow(full_map, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "full_map.png"))

        room_vertices = distance_transform(full_map, resolution, tmp_floor_path)
        room_pcds, room_masks, room_2d_points = [], [], []
        floor_tree = cKDTree(np.array(floor_pcd.points))

        for i in tqdm(range(len(room_vertices)), desc=f"  Assign floor {floor.floor_id} → rooms"):
            room_mask = np.zeros_like(full_map)
            room_mask[room_vertices[i][0], room_vertices[i][1]] = 255
            room_masks.append(room_mask)
            room_m = map_grid_to_point_cloud(room_mask, resolution, pcd_2d)
            room_2d_points.append(room_m)
            z_levels = (
                np.arange(floor_zero_level, floor_zero_level + floor_height, 0.05).reshape(-1, 1)
                * -1
            )
            room_m3d = np.concatenate(
                [np.hstack((room_m, np.ones((room_m.shape[0], 1)) * z)) for z in z_levels]
            )
            pcd_tmp = o3d.geometry.PointCloud()
            pcd_tmp.points = o3d.utility.Vector3dVector(room_m3d)
            T1 = np.eye(4)
            T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
            pcd_tmp.transform(T1)
            _, idx = floor_tree.query(np.array(pcd_tmp.points), k=1, workers=-1)
            room_pcds.append(floor_pcd.select_by_index(idx))
        graph.room_masks[floor.floor_id] = room_masks

        # Room CLIP embeddings
        pose_list, F_g_list, all_global_clip_feats = [], [], {}
        for i, img_id in tqdm(
            enumerate(range(0, len(self.dataset), self.cfg.pipeline.skip_frames)),
            desc=f"  CLIP embeddings floor {floor.floor_id}",
        ):
            rgb_image, _, pose, _, _ = self.dataset[img_id]
            F_g = get_img_feats(np.array(rgb_image), self.preprocess, self.clip_model)
            all_global_clip_feats[str(img_id)] = F_g
            pose_list.append(pose)
            F_g_list.append(F_g)
        np.savez(os.path.join(self.graph_tmp_folder, "room_views.npz"), **all_global_clip_feats)

        pcd_min = np.min(np.array(floor_pcd.points), axis=0)
        pcd_max = np.max(np.array(floor_pcd.points), axis=0)
        repr_embs_list, repr_img_ids_list, room_id2img_id, room_clip_embeddings_list = (
            compute_room_embeddings(
                room_pcds, pose_list, F_g_list, pcd_min, pcd_max, 24, tmp_floor_path
            )
        )

        room_index = 0
        for i in range(len(room_2d_points)):
            room = Room(
                f"{floor.floor_id}_{room_index}",
                floor.floor_id,
                name=f"room_{room_index}",
            )
            room.pcd = room_pcds[i]
            room.vertices = room_2d_points[i]
            graph.floors[int(floor.floor_id)].add_room(room)
            room.room_height = floor_height
            room.room_zero_level = floor_zero_level
            room.embeddings = repr_embs_list[i]
            room.represent_images = [
                int(k * self.cfg.pipeline.skip_frames) for k in repr_img_ids_list[i]
            ]
            room.sample_images = [
                int(k * self.cfg.pipeline.skip_frames) for k in room_id2img_id[i]
            ]
            room.clip_embeddings = room_clip_embeddings_list[i]
            graph.rooms.append(room)
            room_index += 1
        print(
            f"[BUILD]   floor {floor.floor_id}: {len(graph.floors[int(floor.floor_id)].rooms)} rooms"
        )

        view_index = 0
        for room_id in range(len(room_id2img_id)):
            for img_id in room_id2img_id[room_id]:
                retarget = img_id * self.cfg.pipeline.skip_frames
                view = View(f"{floor.floor_id}_{room_id}_{view_index}", room_id, retarget)
                view.img_path = self.dataset.frameId2imgPath[retarget]
                graph.views.append(view)
                view_index += 1
                floor.rooms[room_id].views.append(view)

    def identify_object(
        self, object_feat: np.ndarray, text_feats: np.ndarray, classes: List[str]
    ) -> str:
        """CLIP-classify a single mask feature vector."""
        similarity = np.dot(object_feat.reshape(1, -1), text_feats.T)
        return classes[int(np.argmax(similarity))]

    def segment_hmsg_objects(self, graph: Graph, save_dir: Optional[str] = None) -> None:
        """Classify masks into objects and assign them to rooms."""
        for i, pcd in enumerate(graph.mask_pcds):
            graph.mask_pcds[i] = pcd_denoise(
                pcd, method="dbscan", viz=False, eps=0.05, min_points=10
            )

        text_feats, classes = get_label_feats(
            self.clip_model,
            self.clip_feat_dim,
            self.cfg.pipeline.obj_labels,
            self.cfg.main.save_path,
        )

        pbar = tqdm(enumerate(graph.floors), total=len(graph.floors), desc="[BUILD] Floor")
        margin = 0.2
        for f_idx, floor in pbar:
            pbar.set_description(f"[BUILD] Floor {f_idx}")
            objects_inside_floor = [
                i
                for i, pcd in enumerate(graph.mask_pcds)
                if len(pcd.points) >= 10
                and np.min(np.asarray(pcd.points)[:, 1]) > floor.floor_zero_level - margin
                and np.max(np.asarray(pcd.points)[:, 1])
                < floor.floor_zero_level + floor.floor_height + margin
            ]
            print(f"\n[BUILD]   floor {f_idx}: {len(objects_inside_floor)} candidate masks")

            obj_pbar = tqdm(
                enumerate(objects_inside_floor),
                total=len(objects_inside_floor),
                desc="  Object",
                leave=False,
            )
            for _, mask_idx in obj_pbar:
                room_assoc = [
                    find_intersection_share(
                        room.vertices, np.array(graph.mask_pcds[mask_idx].points)[:, [0, 2]], 0.2
                    )
                    for room in floor.rooms
                ]
                if np.sum(room_assoc) == 0:
                    mask_center = np.mean(
                        np.array(graph.mask_pcds[mask_idx].points)[:, [0, 2]], axis=0
                    )
                    room_assoc = [
                        -np.linalg.norm(np.mean(room.vertices, axis=0) - mask_center)
                        for room in floor.rooms
                    ]
                    if self.cfg.pipeline.save_intermediate_results:
                        closest_idx = int(np.argmax(room_assoc))
                        plt.clf()
                        fig, ax = plt.subplots()
                        plt.scatter(
                            floor.rooms[closest_idx].vertices[:, 0],
                            floor.rooms[closest_idx].vertices[:, 1],
                            color="red",
                        )
                        pts2d = np.array(graph.mask_pcds[mask_idx].points)[:, [0, 2]]
                        plt.scatter(pts2d[:, 0], pts2d[:, 1], s=0.05, alpha=0.5, color="green")
                        ax.set_aspect("equal")
                        debug_dir = os.path.join(self.graph_tmp_folder, "objects")
                        os.makedirs(debug_dir, exist_ok=True)
                        plt.savefig(
                            os.path.join(
                                debug_dir,
                                f"{floor.rooms[closest_idx].room_id}_{floor.rooms[closest_idx].object_counter}.png",
                            )
                        )

                closest_room_idx = int(np.argmax(room_assoc))
                name = self.identify_object(graph.mask_feats[mask_idx], text_feats, classes)
                parent_room = floor.rooms[closest_room_idx]
                obj = Object(
                    f"{parent_room.room_id}_{parent_room.object_counter}", parent_room.room_id
                )
                parent_room.object_counter += 1
                obj.name = name
                obj_pbar.set_description(f"  {obj.name} ({obj.object_id})")
                obj.pcd = graph.mask_pcds[mask_idx]
                obj.vertices = np.array(graph.mask_pcds[mask_idx].points)[:, [0, 2]]
                obj.embedding = graph.mask_feats[mask_idx]

                camera_matrix = self.dataset.get_camera_intrinsics()
                best_view_id, best_depth = None, float("inf")
                for view in parent_room.views:
                    img, _, pose, _, _ = self.dataset[view.img_id]
                    obj_in_view, mean_depth = check_object_in_view(
                        np.array(img).shape[1],
                        np.array(img).shape[0],
                        camera_matrix,
                        np.linalg.inv(pose),
                        np.array(graph.mask_pcds[mask_idx].points),
                        return_depth=True,
                    )
                    if obj_in_view:
                        obj.view_ids.append(view.view_id)
                        view.object_ids.append(obj.object_id)
                        view.text_descriptions.append(obj.name)
                        if mean_depth < best_depth:
                            best_depth = mean_depth
                            best_view_id = view.view_id
                obj.best_view_id = best_view_id
                floor.rooms[closest_room_idx].add_object(obj)
                graph.objects.append(obj)

    # ------------------------------------------------------------------
    # Graph / nav-graph construction
    # ------------------------------------------------------------------

    def create_graph(self, graph: Graph) -> None:
        """Wire up the NetworkX topology graph inside *graph*."""
        for floor in graph.floors:
            graph.graph.add_node(floor, name="floor", type="floor")
            graph.graph.add_edge(0, floor)
            for room in floor.rooms:
                graph.graph.add_node(room, name="room", type="room")
                graph.graph.add_edge(floor, room)
                for obj in room.objects:
                    graph.graph.add_node(obj, name=obj.name, type="object")
                    graph.graph.add_edge(room, obj)
        for view in graph.views:
            graph.graph.add_node(view, name="view", type="view")
            for floor in graph.floors:
                for room in floor.rooms:
                    if room.room_id == view.room_id:
                        graph.graph.add_edge(room, view)
                        break
            for obj in graph.objects:
                if obj.object_id in view.object_ids:
                    graph.graph.add_edge(view, obj)

    def create_nav_graph(self, graph: Graph) -> None:
        """Create navigation Voronoi graph for each floor."""
        nav_dir = os.path.join(self.cfg.main.save_path, "graph", "nav_graph")
        os.makedirs(nav_dir, exist_ok=True)
        poses_list = []
        for i in range(0, len(self.dataset), self.cfg.pipeline.skip_frames):
            _, _, pose, _, _ = self.dataset[i]
            poses_list.append(pose)
        last_nav_graph = None
        global_voronoi = None
        for floor_id, floor in enumerate(graph.floors):
            nav_graph = NavigationGraph(floor.pcd, cell_size=0.03)
            upperbound = (
                graph.floors[floor_id + 1].floor_zero_level
                if floor_id + 1 < len(graph.floors)
                else None
            )
            floor_poses_list = nav_graph.get_floor_poses(floor, poses_list, upperbound)
            sparse_stairs_voronoi = nav_graph.get_stairs_graph_with_poses_v2(
                floor, floor_id, poses_list, nav_dir
            )
            sparse_floor_voronoi = nav_graph.get_floor_graph(floor, floor_poses_list, nav_dir)
            if sparse_stairs_voronoi is not None:
                print(f"[BUILD]   connecting stairs ↔ floor {floor_id}")
                sparse_floor_voronoi = nav_graph.connect_stairs_and_floor_graphs(
                    sparse_stairs_voronoi, sparse_floor_voronoi, nav_dir
                )
            NavigationGraph.save_voronoi_graph(sparse_floor_voronoi, nav_dir, "sparse_voronoi")
            if last_nav_graph is not None and last_nav_graph.has_stairs:
                print(f"[BUILD]   connecting floor {floor_id - 1} ↔ {floor_id}")
                global_voronoi = nav_graph.connect_voronoi_graphs(
                    last_nav_graph.sparse_floor_voronoi, nav_graph.sparse_floor_voronoi
                )
            last_nav_graph = nav_graph
        if global_voronoi is None:
            global_voronoi = last_nav_graph.sparse_floor_voronoi
        NavigationGraph.save_voronoi_graph(global_voronoi, nav_dir, "global_nav_graph")

    # ------------------------------------------------------------------
    # I/O helpers (write-side)
    # ------------------------------------------------------------------

    def save_full_pcd(self, graph: Graph, path: str, visualize_clusters: bool = True) -> None:
        """Save full_pcd to *path*/full_pcd.ply."""
        os.makedirs(path, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(path, "full_pcd.ply"), graph.full_pcd)
        print(f"[SAVE] full_pcd → {path}")
        if visualize_clusters:
            dbscan_eps = (
                self.cfg.pipeline.get("dbscan_eps", 0.02)
                if hasattr(self.cfg, "pipeline")
                else 0.02
            )
            dbscan_min = (
                self.cfg.pipeline.get("dbscan_min_points", 10)
                if hasattr(self.cfg, "pipeline")
                else 10
            )
            visualize_pcd_clusters(
                graph.full_pcd, save_dir=path, dbscan_eps=dbscan_eps, dbscan_min_points=dbscan_min
            )

    def save_full_pcd_feats(self, graph: Graph, path: str) -> None:
        """Save mask_feats and full_feats_array to *path*."""
        os.makedirs(path, exist_ok=True)
        valid_pcds, valid_feats = (
            zip(*[(p, f) for p, f in zip(graph.mask_pcds, graph.mask_feats) if len(p.points) > 0])
            if graph.mask_pcds
            else ([], [])
        )
        graph.mask_pcds = list(valid_pcds)
        graph.mask_feats = list(valid_feats)
        if graph.mask_feats:
            arr = np.array(graph.mask_feats)
            torch.save(torch.from_numpy(arr), os.path.join(path, "mask_feats.pt"))
        if len(graph.full_feats_array) > 0:
            torch.save(
                torch.from_numpy(graph.full_feats_array), os.path.join(path, "full_feats.pt")
            )
        print(f"[SAVE] full_pcd_feats → {path}")

    def save_masked_pcds(self, graph: Graph, path: str, state: str = "both") -> None:
        """Save mask PCDs to *path*."""
        tqdm.write("[SAVE] removing small/empty masks …")
        for i in reversed(range(len(graph.mask_pcds))):
            if graph.mask_pcds[i].is_empty() or len(graph.mask_pcds[i].points) < 10:
                graph.mask_pcds.pop(i)
                graph.mask_feats.pop(i)
        os.makedirs(path, exist_ok=True)
        objects_path = os.path.join(path, "objects")
        os.makedirs(objects_path, exist_ok=True)
        print(f"[SAVE]   mask PCDs: {len(graph.mask_pcds)}, mask_feats: {len(graph.mask_feats)}")
        if state in ("both", "objects"):
            for i, pcd in enumerate(graph.mask_pcds):
                o3d.io.write_point_cloud(os.path.join(objects_path, f"pcd_{i}.ply"), pcd)
        if state in ("both", "full"):
            masked_pcd = o3d.geometry.PointCloud()
            for pcd in graph.mask_pcds:
                pcd.paint_uniform_color(np.random.rand(3))
                masked_pcd += pcd
            o3d.io.write_point_cloud(os.path.join(path, "masked_pcd.ply"), masked_pcd)
        print(f"[SAVE] masked_pcds → {path}")
