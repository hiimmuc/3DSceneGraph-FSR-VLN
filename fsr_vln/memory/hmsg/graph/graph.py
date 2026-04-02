"""Class to represent the HMSG graph."""

import json
import os
import re
import shutil
import time
from copy import deepcopy
from datetime import datetime
from typing import List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d
import open3d.utility as utility
import open_clip
import torch
from application.download_checkpoints import ensure_checkpoints
from memory.hmsg.dataloader.horizon import HorizonDataset
from memory.hmsg.graph.floor import Floor
from memory.hmsg.graph.navigation_graph import NavigationGraph
from memory.hmsg.graph.object import Object
from memory.hmsg.graph.room import Room
from memory.hmsg.graph.view import View
from memory.hmsg.utils.clip_utils import (
    get_img_feats,
    get_text_feats_multiple_templates,
)
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
    visualize_pcd_on_image,
)
from memory.hmsg.utils.label_feats import CSV_LABEL_REGISTRY, get_label_feats
from memory.hmsg.utils.llm_utils import (
    create_llm_client,
    infer_floor_id_from_query,
    parse_hier_query_use_prompt_insentence_parse_icra,
)
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


matplotlib.use("Agg")  # Use non-GUI backend


# pylint: disable=all


class Graph:
    """Hierarchical Multi-modal Scene Graph representation and querying.

    Manages RGB-D data processing, CLIP feature extraction, object/room segmentation,
    and spatial reasoning for navigation tasks.

    Attributes:
        cfg: Configuration object from Hydra/OmegaConf
        device: Compute device ('cuda' or 'cpu')
        full_pcd: Complete point cloud from all frames
        objects: List of detected objects
        rooms: List of segmented rooms
        floors: List of floor levels
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the Graph and load models.

        Args:
            cfg: Hydra configuration object containing model, dataset, and pipeline settings.
        """
        self.cfg = cfg
        self._init_state()
        self._load_clip_model()
        self._init_directories()

        if not hasattr(self.cfg, "pipeline"):
            print("-- entering querying and evaluation mode")
            return

        self._load_dataset()

        if self.cfg.main.use_vlm:
            self._init_vlm_client()
        else:
            self._load_sam_model()

    def _init_state(self) -> None:
        """Initialize all graph data containers and device."""
        self.full_pcd = o3d.geometry.PointCloud()
        self.mask_feats = []
        self.mask_pcds = []
        self.objects = []
        self.rooms = []
        self.floors = []
        self.views = []
        self.full_feats_array = []
        self.graph = nx.Graph()
        self.graph.add_node(0, name="building", type="building")
        self.room_masks = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_clip_model(self) -> None:
        """Load and configure the CLIP model specified in config."""
        clip_type = self.cfg.models.clip.type
        checkpoint = str(self.cfg.models.clip.checkpoint)
        ensure_checkpoints([checkpoint])
        _MODEL_MAP = {
            "ViT-L/14": ("ViT-L-14", {}),
            "ViT-H-14": ("ViT-H-14", {}),
            "ViT-B-32": ("ViT-B-32", {"precision": "fp16"}),
        }
        if clip_type not in _MODEL_MAP:
            raise ValueError(f"Unsupported CLIP model type: {clip_type}")
        model_name, extra_kwargs = _MODEL_MAP[clip_type]
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=checkpoint, device=self.device, **extra_kwargs
        )
        self.clip_feat_dim = CLIP_DIM[model_name]
        self.clip_model.eval()

    def _init_directories(self) -> None:
        """Create working directories and set save paths."""
        self.graph_tmp_folder = os.path.join(self.cfg.main.save_path, "tmp")
        os.makedirs(self.graph_tmp_folder, exist_ok=True)
        self.vln_result_dir = os.path.join(self.cfg.main.save_path, "vln_result_presentation")
        os.makedirs(self.vln_result_dir, exist_ok=True)
        self.curr_query_save_dir = self.vln_result_dir

        self._get_label_csv()

    def _get_label_csv(self):
        # Copy the label CSV file for the configured obj_labels to save_path
        if hasattr(self.cfg, "pipeline"):
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
        """Load RGB-D dataset from configuration.

        Creates HorizonDataset instance with paths and settings from config.
        """
        dataset_cfg = {
            "root_dir": self.cfg.main.dataset_path,
            "transforms": None,
            "depth_cut": self.cfg.main.depth_cut,
        }
        self.dataset = HorizonDataset(dataset_cfg)

    def _init_vlm_client(self) -> None:
        """Initialize VLM client for semantic reasoning.

        Connects to Azure OpenAI or Qwen based on environment configuration.
        """
        self.graph_path = self.cfg.main.graph_path
        self.client, self.vlm_model = create_llm_client()

    def _load_sam_model(self) -> None:
        """Load Segment Anything Model for instance segmentation.

        Initializes SAM and creates automatic mask generator with configured thresholds.
        """
        model_type = self.cfg.models.sam.type
        checkpoint = str(self.cfg.models.sam.checkpoint)
        ensure_checkpoints([checkpoint])
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

    def generate_object_queries(self, instruction: str) -> List[str]:
        """Extract object queries from navigation instruction using VLM.

        Uses LLM to parse instruction and generate diverse search phrases for CLIP retrieval.

        Args:
            instruction: Navigation instruction string.

        Returns:
            List of search phrases for object retrieval.
        """

        prompt = f"""
You are an AI assistant for visual navigation, and your name is **Motion**. Please ignore all occurrences of the word Motion in the input instructions, as they do not represent navigation targets.

Given a navigation instruction, extract the main target object(s) mentioned or implied.
If the instruction does not explicitly mention an object, infer the most likely target object(s) based on common sense and the user's intent.
Generate a diverse bullet list of English phrases for CLIP-based image retrieval, including synonyms and descriptive variants.
If no clear object is mentioned, output an empty list.
Instruction: {instruction}"""

        response_flag = False
        while not response_flag:
            try:
                print("Sending request stage 1 ...")
                response = self.client.chat.completions.create(
                    model=self.vlm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    seed=123,
                )
                response_flag = True
            except Exception as e:
                print(e)
                time.sleep(1)
                print("Retrying ...")
        response = response.choices[0].message.content
        text_probes = re.findall(r"-(.*?)\n", response)
        text_probes = [item.strip(' "-') for item in text_probes]
        text_probes = [item for item in text_probes if len(item) > 0]
        return text_probes

    def create_feature_map(self) -> None:
        """Build hierarchical multi-modal scene graph with per-point CLIP features.

        3-stage pipeline:
            1. Accumulate RGB-D point cloud from dataset frames
            2. Extract per-pixel CLIP features and project to 3D space
            3. Merge overlapping masks and aggregate mask-level features

        Raises:
            ValueError: If dataset not loaded.
        """

        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")

        # === STAGE 1: RGB-D Point Cloud Accumulation ===
        print("[Stage 1/3] Accumulating RGB-D point cloud from frames...")
        for i in tqdm(
            range(0, len(self.dataset), self.cfg.pipeline.skip_frames),
            desc="Building full point cloud",
        ):
            rgb_image, depth_image, pose, _, _ = self.dataset[i]
            self.full_pcd += self.dataset.create_pcd(rgb_image, depth_image, pose, idx=i)

        print(f"   Full PCD points (raw): {len(self.full_pcd.points)}")

        # === Optional: Multi-Stage Point Cloud Filtering ===
        # For real RGB-D sensor data, apply noise reduction pipeline
        if self.cfg.pipeline.get("enable_pcd_filtering", True):
            # Build config dict for filter_point_cloud utility
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
            self.full_pcd = filter_point_cloud(self.full_pcd, filter_config)

        self.save_full_pcd(path=self.cfg.main.save_path)

        # === STAGE 2: Per-Point Feature Extraction ===
        print("[Stage 2/3] Extracting and aggregating per-point CLIP features...")
        locs_in = np.array(self.full_pcd.points)
        tree_pcd = cKDTree(locs_in)
        n_points = locs_in.shape[0]
        counter = torch.zeros((n_points, 1), device="cpu")
        sum_features = torch.zeros((n_points, self.clip_feat_dim), device="cpu")

        frames_pcd = []
        for i in tqdm(
            range(0, len(self.dataset), self.cfg.pipeline.skip_frames),
            desc="Computing per-point features",
        ):
            rgb_image, depth_image, pose, _, _ = self.dataset[i]
            if rgb_image.size != depth_image.size:
                rgb_image = rgb_image.resize(depth_image.size)

            # Extract per-pixel features and masks
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

            # Project frame to 3D and create mask point clouds
            pcd = self.dataset.create_pcd(rgb_image, depth_image, pose, idx=i)
            masks_3d = self.dataset.create_3d_masks(
                masks,
                depth_image,
                self.full_pcd,
                tree_pcd,
                pose,
                i,
                down_size=self.cfg.pipeline.voxel_size,
                filter_distance=self.cfg.pipeline.max_mask_distance,
            )
            frames_pcd.append(masks_3d)

            # Aggregate features: find nearest full_pcd point for each frame point
            depth_mask = np.array(depth_image) > 0
            depth_mask = torch.from_numpy(depth_mask)
            F_2D_masked = F_2D[depth_mask]
            _, idx = tree_pcd.query(np.asarray(pcd.points), k=1, workers=-1)
            sum_features[idx] += F_2D_masked
            counter[idx] += 1

        # Compute per-point average features
        counter[counter == 0] = 1e-5
        sum_features = sum_features / counter
        self.full_feats_array = sum_features.cpu().numpy()
        print(f"   Full feature array shape: {self.full_feats_array.shape}")

        # Free intermediate tensors
        del sum_features, counter
        torch.cuda.empty_cache()

        # === STAGE 3: 3D Mask Merging and Feature Aggregation ===
        print("[Stage 3/3] Merging overlapping masks and aggregating mask features...")
        self._merge_masks(frames_pcd)
        self._aggregate_mask_features(tree_pcd)

    def _merge_masks(self, frames_pcd: List[List[o3d.geometry.PointCloud]]) -> None:
        """Merge overlapping 3D masks across frames.

        Applies hierarchical or sequential merging strategy based on configuration.

        Args:
            frames_pcd: List of point clouds for each frame's detected masks.
        """

        merge_type = self.cfg.pipeline.merge_type.lower()

        if merge_type == "hierarchical":
            tqdm.write("  Merging masks hierarchically...")
            self.mask_pcds = hierarchical_merge(
                frames_pcd,
                self.cfg.pipeline.init_overlap_thresh,
                self.cfg.pipeline.overlap_thresh_factor,
                self.cfg.pipeline.voxel_size,
                self.cfg.pipeline.iou_thresh,
            )
        elif merge_type == "sequential":
            tqdm.write("  Merging masks sequentially...")
            self.mask_pcds = seq_merge(
                frames_pcd,
                self.cfg.pipeline.init_overlap_thresh,
                self.cfg.pipeline.voxel_size,
                self.cfg.pipeline.iou_thresh,
            )
        else:
            raise ValueError(
                f"Invalid merge_type: '{merge_type}'. Must be 'hierarchical' or 'sequential'. "
                f"Check config.pipeline.merge_type"
            )

        # Filter small point clouds (noise/artifacts)
        original_count = len(self.mask_pcds)
        min_points_threshold = self.cfg.pipeline.get("min_mask_points", 10)
        self.mask_pcds = [
            pcd
            for pcd in self.mask_pcds
            if not pcd.is_empty() and len(pcd.points) >= min_points_threshold
        ]
        removed_count = original_count - len(self.mask_pcds)
        if removed_count > 0:
            print(f"   Filtered {removed_count} masks below {min_points_threshold} points")

    def _aggregate_mask_features(self, tree_pcd: cKDTree) -> None:
        """Aggregate CLIP features for each merged mask.

        Uses batched KDTree queries for efficiency. Removes points with
        poor reconstruction quality (distance > threshold).

        Args:
            tree_pcd: KD-tree of full point cloud for spatial queries.
        """

        masks_feats = []
        voxel_size = self.cfg.pipeline.voxel_size
        dist_threshold = self.cfg.pipeline.get("mask_feature_dist_threshold", 0.8)

        # Batch gather all mask points for single KDTree query
        downsampled_masks = [m.voxel_down_sample(voxel_size) for m in self.mask_pcds]
        pts_per_mask = [np.asarray(m.points) for m in downsampled_masks]
        mask_lengths = [len(p) for p in pts_per_mask]

        # Query all mask points at once (scipy parallelizes internally with workers=-1)
        if sum(mask_lengths) > 0:
            all_points = np.vstack([p for p in pts_per_mask if len(p) > 0])
            all_dist, all_idx = tree_pcd.query(all_points, k=1, workers=-1)
        else:
            all_dist, all_idx = np.array([]), np.array([])

        # Extract features for each mask
        offset = 0
        for i, pts in enumerate(pts_per_mask):
            n = mask_lengths[i]
            if n == 0:
                masks_feats.append(
                    np.zeros((1, self.clip_feat_dim), dtype=self.full_feats_array.dtype)
                )
                continue

            dist = all_dist[offset : offset + n]
            idx = all_idx[offset : offset + n]
            offset += n

            # Filter points biased toward camera (good reconstruction)
            valid_mask = dist <= dist_threshold
            n_valid = int(valid_mask.sum())
            n_removed = n - n_valid
            if n_removed > 0 and n_valid > 0:
                tqdm.write(
                    f"    mask[{i}]: kept {n_valid}/{n} points (dist <= {dist_threshold:.2f}m)"
                )

            if n_valid == 0:
                masks_feats.append(
                    np.zeros((1, self.clip_feat_dim), dtype=self.full_feats_array.dtype)
                )
                continue

            # Aggregate features from valid points
            valid_idx = idx[valid_mask]
            feats = self.full_feats_array[valid_idx]
            feats = np.nan_to_num(feats)
            feats = feats_denoise_dbscan(feats, eps=0.01, min_points=100)
            masks_feats.append(feats)

        self.mask_feats = masks_feats
        print(f"   Created {len(self.mask_feats)} mask features from {len(self.mask_pcds)} masks")
        assert len(self.mask_pcds) == len(self.mask_feats), "Mask-feature mismatch!"

    def segment_floors_manually(
        self, path: str, flip_zy: bool = False, mid_points: List = []
    ) -> None:
        """Manually segment point cloud into floors.

        Args:
            path: Path to save segmented floor point clouds.
            flip_zy: Whether to flip Z-Y axes (coordinate system conversion).
            mid_points: Y-axis values marking floor boundaries.
        """

        # downsample the point cloud
        downpcd = self.full_pcd.voxel_down_sample(voxel_size=0.05)
        # flip the z and y axis
        if flip_zy:
            downpcd.points = o3d.utility.Vector3dVector(np.array(downpcd.points)[:, [0, 2, 1]])
            downpcd.transform(np.eye(4) * np.array([1, 1, -1, 1]))
        # rotate the point cloud to align floor with the y axis
        downpcd = np.asarray(downpcd.points)
        print("downpcd", downpcd.shape)

        # divide z axis range into 0.01m bin
        reselotion = 0.01
        bins = np.abs(np.max(downpcd[:, 1]) - np.min(downpcd[:, 1])) / reselotion
        print("min, max", np.min(downpcd[:, 1]), np.max(downpcd[:, 1]))
        print("bins", bins)
        z_hist = np.histogram(downpcd[:, 1], bins=int(bins))
        # smooth the histogram
        z_hist_smooth = gaussian_filter1d(z_hist[0], sigma=2)
        # Find the peaks in this histogram.
        distance = 0.2 / reselotion
        print("distance", distance)
        # set the min peak height based on the histogram
        print(np.mean(z_hist_smooth))
        min_peak_height = np.percentile(z_hist_smooth, 90)
        print("min_peak_height", min_peak_height)
        peaks, _ = find_peaks(z_hist_smooth, distance=distance, height=min_peak_height)

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r")
            plt.savefig(os.path.join(self.graph_tmp_folder, "floor_histogram.png"))

        # cluster the peaks using DBSCAN
        peaks_locations = z_hist[1][peaks]
        clustering = DBSCAN(eps=1, min_samples=1).fit(peaks_locations.reshape(-1, 1))
        labels = clustering.labels_

        # plot the histogram
        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.plot(z_hist[1][:-1], z_hist_smooth)
            plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
            plt.hlines(min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r")
            # plot the clusters
            for i in range(len(np.unique(labels))):
                plt.plot(
                    z_hist[1][peaks[labels == i]],
                    z_hist_smooth[peaks[labels == i]],
                    "o",
                )
            plt.savefig(os.path.join(self.graph_tmp_folder, "floor_histogram_cluster.png"))

        # for each cluster find the top 2 peaks
        clustered_peaks = []
        for i in range(len(np.unique(labels))):
            # for first and last cluster, find the top 1 peak
            if i == 0 or i == len(np.unique(labels)) - 1:
                p = peaks[labels == i]
                top_p = p[np.argsort(z_hist_smooth[p])[-1:]].tolist()
                top_p = [z_hist[1][p] for p in top_p]
                clustered_peaks.append(top_p)
                continue
            p = peaks[labels == i]
            top_p = p[np.argsort(z_hist_smooth[p])[-2:]].tolist()
            top_p = [z_hist[1][p] for p in top_p]
            clustered_peaks.append(top_p)
        clustered_peaks = [item for sublist in clustered_peaks for item in sublist]
        clustered_peaks = np.sort(clustered_peaks)
        print("clustered_peaks", clustered_peaks)

        # Check if the distance between adjacent peaks exceeds or equals 2.5m
        adjusted_peaks = []
        for i in range(len(clustered_peaks) - 1):
            adjusted_peaks.append(clustered_peaks[i])
            if clustered_peaks[i + 1] - clustered_peaks[i] >= 2.5:
                # Insert a virtual boundary between the two peaks
                mid_point = clustered_peaks[i + 1] - 0.2
                adjusted_peaks.append(mid_point)
        adjusted_peaks.append(clustered_peaks[-1])

        clustered_peaks = np.array(adjusted_peaks)
        print("adjusted_peaks", clustered_peaks)
        floors = []
        # Generate floor ranges based on the adjusted peaks
        for i in range(len(clustered_peaks) - 1):
            floors.append([clustered_peaks[i], clustered_peaks[i + 1]])
        print("computed floors: ", floors)

        if not floors:
            floors.append([z_hist[1].min().item(), z_hist[1].max().item()])
            print("priors floors", floors)

        # Extend the first and last floor ranges
        floors[0][0] = (floors[0][0] + np.min(downpcd[:, 1])) / 2
        floors[-1][1] = np.max(downpcd[:, 1])

        print("Original clustered_peaks:", clustered_peaks)
        print("Adjusted clustered_peaks after inserting virtual boundaries:", adjusted_peaks)
        print("Generated floor ranges:", floors)
        print("Total number of floors detected:", len(floors))

        for i, floor in enumerate(floors):
            floor_obj = Floor(str(i), name="floor_" + str(i))
            floor_pcd = self.full_pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=(-np.inf, floor[0], -np.inf),
                    max_bound=(np.inf, floor[1], np.inf),
                )
            )
            bbox = floor_pcd.get_axis_aligned_bounding_box()
            floor_obj.vertices = np.asarray(bbox.get_box_points())
            floor_obj.pcd = floor_pcd
            floor_obj.floor_zero_level = np.min(np.array(floor_pcd.points)[:, 1])
            floor_obj.floor_height = floor[1] - floor_obj.floor_zero_level
            self.floors.append(floor_obj)
        print("final floors: ", floors)
        return floors

    def segment_hmsg_room(self, floor: Floor, path: str) -> None:
        """Segment a floor into rooms with semantic embeddings.

        Uses 2D projection, connected components analysis, and CLIP embeddings.

        Args:
            floor: Floor object containing point cloud and frames.
            path: Directory to save room segmentation results.
        """

        tmp_floor_path = os.path.join(self.graph_tmp_folder, floor.floor_id)
        if not os.path.exists(tmp_floor_path):
            os.makedirs(tmp_floor_path, exist_ok=True)

        floor_pcd = floor.pcd
        xyz = np.asarray(floor_pcd.points)
        xyz_full = xyz.copy()
        floor_zero_level = floor.floor_zero_level
        floor_height = floor.floor_height
        print("floor_zero_level, floor_height = ", floor_zero_level, floor_height)
        ## Slice below the ceiling ##
        xyz = xyz[xyz[:, 1] < floor_zero_level + floor_height - 0.3]
        xyz = xyz[xyz[:, 1] >= floor_zero_level + 0.3]
        xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + floor_height - 0.2]

        # project the point cloud to 2d
        pcd_2d = xyz[:, [0, 2]]
        xyz_full = xyz_full[:, [0, 2]]

        # define the grid size and resolution based on the 2d point cloud
        grid_size = (
            int(np.max(pcd_2d[:, 0]) - np.min(pcd_2d[:, 0])),
            int(np.max(pcd_2d[:, 1]) - np.min(pcd_2d[:, 1])),
        )
        grid_size = (grid_size[0] + 1, grid_size[1] + 1)
        resolution = self.cfg.pipeline.grid_resolution
        print("grid_size: ", grid_size)

        # calc 2d histogram of the floor using the xyz point cloud to extract
        # the walls skeleton
        num_bins = (int(grid_size[0] // resolution), int(grid_size[1] // resolution))
        num_bins = (num_bins[1] + 1, num_bins[0] + 1)
        hist, _, _ = np.histogram2d(pcd_2d[:, 1], pcd_2d[:, 0], bins=num_bins)
        if self.cfg.pipeline.save_intermediate_results:
            # plot the histogram
            plt.figure()
            plt.imshow(hist, interpolation="nearest", cmap="jet", origin="lower")
            plt.colorbar()
            plt.savefig(os.path.join(tmp_floor_path, "2D_histogram.png"))

        # apply threshold
        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist = cv2.GaussianBlur(hist, (5, 5), 1)
        hist_threshold = 0.25 * np.max(hist)
        _, walls_skeleton = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY)

        # create a bigger image to avoid losing the walls
        walls_skeleton = cv2.copyMakeBorder(
            walls_skeleton, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )

        # apply closing to the walls skeleton
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        walls_skeleton = cv2.morphologyEx(walls_skeleton, cv2.MORPH_CLOSE, kernel, iterations=1)

        # extract outside boundary from histogram of xyz_full
        hist_full, _, _ = np.histogram2d(xyz_full[:, 1], xyz_full[:, 0], bins=num_bins)
        hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist_full = cv2.GaussianBlur(hist_full, (21, 21), 2)
        _, outside_boundary = cv2.threshold(hist_full, 0, 255, cv2.THRESH_BINARY)

        # create a bigger image to avoid losing the walls
        outside_boundary = cv2.copyMakeBorder(
            outside_boundary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
        )

        # apply closing to the outside boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        outside_boundary = cv2.morphologyEx(
            outside_boundary, cv2.MORPH_CLOSE, kernel, iterations=3
        )

        # extract the outside contour from the outside boundary
        contours, _ = cv2.findContours(
            outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outside_boundary = np.zeros_like(outside_boundary)
        cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1)
        outside_boundary = outside_boundary.astype(np.uint8)

        if self.cfg.pipeline.save_intermediate_results:
            plt.figure()
            plt.imshow(walls_skeleton, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "walls_skeleton.png"))

            plt.figure()
            plt.imshow(outside_boundary, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "outside_boundary.png"))

        # combine the walls skelton and outside boundary
        full_map = cv2.bitwise_or(walls_skeleton, cv2.bitwise_not(outside_boundary))

        # apply closing to the full map
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, kernel, iterations=2)

        if self.cfg.pipeline.save_intermediate_results:
            # plot the full map
            plt.figure()
            plt.imshow(full_map, cmap="gray", origin="lower")
            plt.savefig(os.path.join(tmp_floor_path, "full_map.png"))
        # apply distance transform to the full map
        room_vertices = distance_transform(full_map, resolution, tmp_floor_path)

        # using the 2D room vertices, map the room back to the original point
        # cloud using KDTree
        room_pcds = []
        room_masks = []
        room_2d_points = []
        floor_tree = cKDTree(np.array(floor_pcd.points))
        for i in tqdm(range(len(room_vertices)), desc="Assign floor points to rooms"):
            print("idx = ", i)
            room = np.zeros_like(full_map)
            room[room_vertices[i][0], room_vertices[i][1]] = 255
            room_masks.append(room)
            room_m = map_grid_to_point_cloud(room, resolution, pcd_2d)
            room_2d_points.append(room_m)
            # extrude the 2D room to 3D room by adding z value from floor zero
            # level to floor zero level + floor height, step by 0.1m
            z_levels = np.arange(floor_zero_level, floor_zero_level + floor_height, 0.05)
            z_levels = z_levels.reshape(-1, 1)
            z_levels *= -1
            room_m3dd = []
            for z in z_levels:
                room_m3d = np.hstack((room_m, np.ones((room_m.shape[0], 1)) * z))
                room_m3dd.append(room_m3d)
            room_m3d = np.concatenate(room_m3dd, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(room_m3d)
            # rotate floor pcd to align with the original point cloud
            T1 = np.eye(4)
            T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
            pcd.transform(T1)
            # find the nearest point in the original point cloud  # slow
            _, idx = floor_tree.query(np.array(pcd.points), k=1, workers=-1)
            pcd = floor_pcd.select_by_index(idx)
            room_pcds.append(pcd)
        self.room_masks[floor.floor_id] = room_masks

        # compute the features of room: input a list of poses and images,
        # output a list of embeddings list
        pose_list = []
        F_g_list = []

        all_global_clip_feats = dict()
        for i, img_id in tqdm(
            enumerate(range(0, len(self.dataset), self.cfg.pipeline.skip_frames)),
            desc="Computing room features",
        ):
            rgb_image, _, pose, _, _ = self.dataset[img_id]
            F_g = get_img_feats(np.array(rgb_image), self.preprocess, self.clip_model)
            all_global_clip_feats[str(img_id)] = F_g
            pose_list.append(pose)
            F_g_list.append(F_g)
        np.savez(
            os.path.join(self.graph_tmp_folder, "room_views.npz"),
            **all_global_clip_feats,
        )

        pcd_min = np.min(np.array(floor_pcd.points), axis=0)
        pcd_max = np.max(np.array(floor_pcd.points), axis=0)
        assert pcd_min.shape[0] == 3

        repr_embs_list, repr_img_ids_list, room_id2img_id, room_clip_embeddings_list = (
            compute_room_embeddings(
                room_pcds, pose_list, F_g_list, pcd_min, pcd_max, 24, tmp_floor_path
            )
        )
        assert len(repr_embs_list) == len(room_2d_points)
        assert len(repr_img_ids_list) == len(room_2d_points)
        assert len(room_id2img_id) == len(room_2d_points)

        room_index = 0
        for i in range(len(room_2d_points)):
            room = Room(
                str(floor.floor_id) + "_" + str(room_index),
                floor.floor_id,
                name="room_" + str(room_index),
            )
            room.pcd = room_pcds[i]
            room.vertices = room_2d_points[i]
            self.floors[int(floor.floor_id)].add_room(room)
            room.room_height = floor_height
            room.room_zero_level = floor.floor_zero_level
            room.embeddings = repr_embs_list[i]
            room.represent_images = [
                int(k * self.cfg.pipeline.skip_frames) for k in repr_img_ids_list[i]
            ]
            room.sample_images = [
                int(k * self.cfg.pipeline.skip_frames) for k in room_id2img_id[i]
            ]
            room.clip_embeddings = room_clip_embeddings_list[i]
            self.rooms.append(room)
            room_index += 1
        print(
            "number of rooms in floor {} is {}".format(
                floor.floor_id, len(self.floors[int(floor.floor_id)].rooms)
            )
        )
        # Build view hierarchy
        view_index = 0
        for room_id in range(len(room_id2img_id)):
            for i, img_id in enumerate(room_id2img_id[room_id]):
                retarget_img_id = img_id * self.cfg.pipeline.skip_frames
                img_path = self.dataset.frameId2imgPath[retarget_img_id]
                view = View(
                    str(floor.floor_id) + "_" + str(room_id) + "_" + str(view_index),
                    room_id,
                    retarget_img_id,
                )
                view.img_path = img_path
                self.views.append(view)
                view_index += 1
                floor.rooms[room_id].views.append(view)

    def identify_object(
        self, object_feat: np.ndarray, text_feats: np.ndarray, classes: List[str]
    ) -> str:
        """Classify object using CLIP similarity to text embeddings.

        Args:
            object_feat: CLIP feature vector of object.
            text_feats: CLIP feature vectors of class labels.
            classes: List of class names.

        Returns:
            Best matching class name.
        """

        similarity = np.dot(object_feat.reshape(1, -1), text_feats.T)
        # find the class with the highest similarity
        return classes[np.argmax(similarity)]

    def segment_hmsg_objects(self, save_dir: str = None) -> None:
        """Detect and segment objects in all rooms.

        Processes masks through VLM ranking and stores Object instances.

        Args:
            save_dir: Directory to save object segmentation results.
        """

        for i, pcd in enumerate(self.mask_pcds):
            self.mask_pcds[i] = pcd_denoise(
                pcd, method="dbscan", viz=False, eps=0.05, min_points=10
            )
        text_feats, classes = get_label_feats(
            self.clip_model,
            self.clip_feat_dim,
            self.cfg.pipeline.obj_labels,
            self.cfg.main.save_path,  # should be "./memory/hmsg/labels/",
        )

        pbar = tqdm(enumerate(self.floors), total=len(self.floors), desc="Floor: ")
        margin = 0.2
        for f_idx, floor in pbar:
            pbar.set_description(f"Floor: {f_idx}")
            objects_inside_floor = list()
            # assign objects to rooms
            for i, pcd in enumerate(self.mask_pcds):
                if len(pcd.points) < 10:
                    continue
                min_z = np.min(np.asarray(pcd.points)[:, 1])
                max_z = np.max(np.asarray(pcd.points)[:, 1])
                if min_z > floor.floor_zero_level - margin and max_z < (
                    floor.floor_zero_level + floor.floor_height + margin
                ):
                    objects_inside_floor.append(i)

            print("number of objects inside floor {}: {}".format(f_idx, len(objects_inside_floor)))

            # show the second layer of pbar with tqdm
            obj_pbar = tqdm(
                enumerate(objects_inside_floor),
                total=len(objects_inside_floor),
                desc="Object: ",
                leave=False,
            )
            for obj_floor_idx, mask_idx in obj_pbar:
                room_assoc = list()
                for r_idx, room in enumerate(floor.rooms):
                    room_assoc.append(
                        find_intersection_share(
                            room.vertices,
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]],
                            0.2,
                        )
                    )
                # for outlier objects, utilize Euclidean distance between room
                # centers and mask centers
                if np.sum(room_assoc) == 0:
                    for r_idx, room in enumerate(floor.rooms):
                        # use negative distance to align with the similarity
                        # metric
                        room_assoc[r_idx] = -1 * np.linalg.norm(
                            np.mean(room.vertices, axis=0)
                            - np.mean(
                                np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]],
                                axis=0,
                            )
                        )
                    if self.cfg.pipeline.save_intermediate_results:
                        plt.clf()
                        fig, ax = plt.subplots()
                        for r_idx, room in enumerate(floor.rooms):
                            if np.argmax(room_assoc) == r_idx:
                                plt.scatter(
                                    room.vertices[:, 0],
                                    room.vertices[:, 1],
                                    color="red",
                                )
                            else:
                                continue
                                # plt.scatter(room.vertices[:, 0], room.vertices[:, 1])
                        plt.scatter(
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]][:, 0],
                            np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]][:, 1],
                            s=0.05,
                            alpha=0.5,
                            color="green",
                        )
                        ax.set_aspect("equal")

                        debug_objects_dir = os.path.join(self.graph_tmp_folder, "objects")
                        os.makedirs(debug_objects_dir, exist_ok=True)
                        plt.savefig(
                            os.path.join(
                                debug_objects_dir,
                                f"{floor.rooms[np.argmax(room_assoc)].room_id}_{floor.rooms[np.argmax(room_assoc)].object_counter}.png",
                            )
                        )

                closest_room_idx = np.argmax(room_assoc)

                name = self.identify_object(self.mask_feats[mask_idx], text_feats, classes)
                parent_room = floor.rooms[closest_room_idx]
                object = Object(
                    parent_room.room_id + "_" + str(parent_room.object_counter),
                    parent_room.room_id,
                )
                parent_room.object_counter += 1
                object.name = name
                obj_pbar.set_description(f"object name: {object.name}, {object.object_id}")
                object.pcd = self.mask_pcds[mask_idx]
                object.vertices = np.array(self.mask_pcds[mask_idx].points)[:, [0, 2]]
                object.embedding = self.mask_feats[mask_idx]
                # build view-object topology graph
                best_view_id = None
                best_depth = float("inf")
                all_views_in_room = parent_room.views
                camera_matrix = self.dataset.get_camera_intrinsics()
                for view in all_views_in_room:
                    img, _, pose, _, _ = self.dataset[view.img_id]
                    obj_in_view, mean_depth = check_object_in_view(
                        np.array(img).shape[1],
                        np.array(img).shape[0],
                        camera_matrix,
                        np.linalg.inv(pose),
                        np.array(self.mask_pcds[mask_idx].points),
                        return_depth=True,  # Modified check_object_in_view to support returning depth
                    )
                    if obj_in_view:
                        object.view_ids.append(view.view_id)
                        view.object_ids.append(object.object_id)
                        view.text_descriptions.append(object.name)
                        # Find the best viewpoint (minimum average depth)
                        if mean_depth < best_depth:
                            best_depth = mean_depth
                            best_view_id = view.view_id
                object.best_view_id = best_view_id
                floor.rooms[closest_room_idx].add_object(object)
                self.objects.append(object)

    def create_graph(self) -> None:
        """Build graph structure connecting building, floors, rooms, and objects."""

        # add nodes to the graph
        for floor in self.floors:
            self.graph.add_node(floor, name="floor", type="floor")
            self.graph.add_edge(0, floor)
            for room in floor.rooms:
                self.graph.add_node(room, name="room", type="room")
                self.graph.add_edge(floor, room)
                for object in room.objects:
                    self.graph.add_node(object, name=object.name, type="object")
                    self.graph.add_edge(room, object)

        for view in self.views:
            self.graph.add_node(view, name="view", type="view")
            for floor in self.floors:
                for room in floor.rooms:
                    if room.room_id == view.room_id:
                        self.graph.add_edge(room, view)
                        break
            for obj in self.objects:
                if obj.object_id in view.object_ids:
                    self.graph.add_edge(view, obj)

    def save_hmsg_graph(self, path: str) -> None:
        """Serialize graph to disk.

        Args:
            path: Directory to save graph files.
        """

        # create a folder for the graph
        if not os.path.exists(path):
            os.makedirs(path)
        # create a folder for floors, rooms and objects
        if not os.path.exists(os.path.join(path, "floors")):
            os.makedirs(os.path.join(path, "floors"))
        if not os.path.exists(os.path.join(path, "rooms")):
            os.makedirs(os.path.join(path, "rooms"))
        if not os.path.exists(os.path.join(path, "objects")):
            os.makedirs(os.path.join(path, "objects"))
        if not os.path.exists(os.path.join(path, "views")):
            os.makedirs(os.path.join(path, "views"))  # save the graph
        for i, node in enumerate(self.graph.nodes(data=True)):
            topo_obj, node_dict = node
            if isinstance(topo_obj, Floor):
                topo_obj.save(os.path.join(path, "floors"))
            elif isinstance(topo_obj, Room):
                topo_obj.save(os.path.join(path, "rooms"))
            elif isinstance(topo_obj, Object):
                topo_obj.save(os.path.join(path, "objects"))
            elif isinstance(topo_obj, View):
                topo_obj.save(os.path.join(path, "views"))

    def load_hmsg_graph(self, path: str) -> None:
        """Load graph from disk.

        Args:
            path: Directory containing saved graph files.
        """

        print("... loading predicted graph")
        self.graph_path = path
        # load floors
        floor_files = sorted(os.listdir(os.path.join(path, "floors")))
        floor_files = sorted([f for f in floor_files if f.endswith(".ply")])
        for floor_file in floor_files:
            floor_file = floor_file.split(".")[0]
            floor = Floor(str(floor_file), name="floor_" + str(floor_file))
            floor.load(os.path.join(path, "floors"))
            self.floors.append(floor)
            self.graph.add_node(floor, name="floor_" + str(floor_file), type="floor")
            self.graph.add_edge(0, floor)
        print("# pred floors: ", len(self.floors))
        # load rooms
        room_files = sorted(os.listdir(os.path.join(path, "rooms")))
        room_files = [f for f in room_files if f.endswith(".ply")]
        for room_file in room_files:
            room_file = room_file.split(".")[0]
            room = Room(str(room_file), room_file.split("_")[0])
            room.load(os.path.join(path, "rooms"))
            self.rooms.append(room)
            self.graph.add_node(room, name="room_" + str(room_file), type="room")
            self.graph.add_edge(self.floors[int(room_file.split("_")[0])], room)
            self.floors[int(room.floor_id)].rooms.append(room)
        print("# pred rooms: ", len(self.rooms))
        # load objects
        object_files = sorted(os.listdir(os.path.join(path, "objects")))
        object_files = [f for f in object_files if f.endswith(".ply")]
        for object_file in object_files:
            object_file = object_file.split(".")[0]
            room_id = "_".join(object_file.split("_")[:2])
            parent_room = None
            for room in self.rooms:
                if room.room_id == room_id:
                    parent_room = room
                    break
            assert parent_room is not None, f"Couldn't find the room with room id {room_id}"
            objectt = Object(str(object_file), room_id, name="object_" + str(object_file))
            objectt.load(os.path.join(path, "objects"))
            self.objects.append(objectt)
            self.graph.add_node(objectt, name="object_" + str(object_file), type="object")
            self.graph.add_edge(parent_room, objectt)
            # add object to the room
            parent_room.add_object(objectt)
        print("# pred objects: ", len(self.objects))

        # load views
        view_files = sorted(os.listdir(os.path.join(path, "views")))
        for view_file in view_files:
            view_file = view_file.split(".")[0]
            room_id = "_".join(view_file.split("_")[:2])
            parent_room = None
            for room in self.rooms:
                if room.room_id == room_id:
                    parent_room = room
                    break
            assert parent_room is not None, f"Couldn't find the room with room id {room_id}"
            vieww = View(str(view_file), room_id, img_id=None, name="view_" + str(view_file))
            vieww.load(os.path.join(path, "views"))
            self.views.append(vieww)
            self.graph.add_node(vieww, name="view_" + str(view_file), type="view")
            self.graph.add_edge(parent_room, vieww)

        print("-------------------")

    def build_hier_multimodal_scene_graph(self, save_path: str = None) -> None:
        """Build complete hierarchical scene graph from raw data.

        Args:
            save_path: Path to save intermediate and final results.
        """

        print("segmenting floors...")
        self.segment_floors_manually(save_path)

        print("segmenting rooms...")
        for floor in self.floors:
            self.segment_hmsg_room(floor, save_path)

        print("segmenting/identifying objects...")
        self.segment_hmsg_objects(save_path)

        print("number of objects: ", len(self.objects))
        if self.cfg.pipeline.merge_objects_graph:
            # merge objects that close to each other with same name
            for room in tqdm(self.rooms):
                print("room: ", room.room_id)
                print(" number of objects before merging: ", len(room.objects))
                room.merge_objects()
                print(" number of objects after merging: ", len(room.objects))

        print("creating graph...")
        self.create_graph()

        # create navigation graph for each floor
        print("createing nav_graph...")
        self.create_nav_graph()

        # save the graph
        now_str = datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_hmsg_graph(os.path.join(save_path, "graph_" + now_str))

        print("# floors: ", len(self.floors))
        print("# rooms: ", len(self.rooms))
        print("# views: ", len(self.views))
        print("# objects: ", len(self.objects))
        print("--> HMSG representation successfully built")

    def create_nav_graph(self) -> None:
        """Create navigation graph from room topology."""

        last_nav_graph = None
        global_voronoi = None

        # create a folder for the resulting navigation graph
        nav_dir = os.path.join(self.cfg.main.save_path, "graph", "nav_graph")
        os.makedirs(nav_dir, exist_ok=True)

        # get pose list
        poses_list = []
        for i in range(0, len(self.dataset), self.cfg.pipeline.skip_frames):
            _, _, pose, _, _ = self.dataset[i]
            poses_list.append(pose)

        for floor_id, floor in enumerate(self.floors):
            nav_graph = NavigationGraph(floor.pcd, cell_size=0.03)
            upperbound = None
            if floor_id + 1 < len(self.floors):
                upperbound = self.floors[floor_id + 1].floor_zero_level
            floor_poses_list = nav_graph.get_floor_poses(floor, poses_list, upperbound)
            sparse_stairs_voronoi = nav_graph.get_stairs_graph_with_poses_v2(
                floor, floor_id, poses_list, nav_dir
            )
            sparse_floor_voronoi = nav_graph.get_floor_graph(floor, floor_poses_list, nav_dir)
            if sparse_stairs_voronoi is not None:
                print(f"connecting stairs and floor {floor_id}")
                sparse_floor_voronoi = nav_graph.connect_stairs_and_floor_graphs(
                    sparse_stairs_voronoi, sparse_floor_voronoi, nav_dir
                )
            NavigationGraph.save_voronoi_graph(sparse_floor_voronoi, nav_dir, "sparse_voronoi")

            if last_nav_graph is not None and last_nav_graph.has_stairs:
                print(f"connecting two floors {floor_id}")
                global_voronoi = nav_graph.connect_voronoi_graphs(
                    last_nav_graph.sparse_floor_voronoi, nav_graph.sparse_floor_voronoi
                )
            last_nav_graph = nav_graph

        if global_voronoi is None:
            global_voronoi = last_nav_graph.sparse_floor_voronoi

        NavigationGraph.save_voronoi_graph(global_voronoi, nav_dir, "global_nav_graph")

    def set_room_names(self, room_names: List[str]) -> None:
        """Set semantic names for rooms.

        Args:
            room_names: List of room type names (bedroom, kitchen, etc.).
        """

        assert len(room_names) == len(
            self.rooms
        ), "The length of room_names should be the same as the number of rooms in the graph"
        for i in range(len(self.rooms)):
            self.rooms[i].name = room_names[i]
            vertices = self.rooms[i].vertices
            center = np.mean(vertices, axis=0)
            self.rooms[i].room_center_pos = center

    def generate_room_names(
        self,
        generate_method: str = "label",
        default_room_types: List[str] = None,
    ) -> None:
        """Generate semantic names for rooms.

        Args:
            generate_method: Method to generate names ('label' or 'llm').
            default_room_types: List of candidate room types.
        """

        for i in range(len(self.rooms)):
            if generate_method in ["obj_embedding", "view_embedding"]:
                assert (
                    default_room_types is not None
                ), "You should provide a list of default room types"
                assert self.clip_model is not None, "You should provide a clip model"
                assert (
                    self.clip_feat_dim is not None
                ), "You should provide the clip features dimension"
            self.rooms: List[Room]
            if generate_method in ["obj_embedding", "label"]:
                self.rooms[i].infer_room_type_from_objects(
                    infer_method=generate_method,
                    default_room_types=default_room_types,
                    clip_model=self.clip_model,
                    clip_feat_dim=self.clip_feat_dim,
                )
            elif generate_method in ["view_embedding"]:
                self.rooms[i].infer_room_type_from_view_embedding(
                    default_room_types, self.clip_model, self.clip_feat_dim
                )
            else:
                return NotImplementedError

    def query_graph(self, query: str) -> Tuple[Floor, Room, List[Object]]:
        """Query graph for objects matching description.

        Args:
            query: Natural language query string.

        Returns:
            Tuple of (best_floor, best_room, list_of_objects).
        """

        text_feats = get_text_feats_multiple_templates(
            [query], self.clip_model, self.clip_feat_dim
        )
        # compute similarity between the text query and the objects embeddings
        # in the graph
        similarity = np.dot(text_feats, np.array([o.embedding for o in self.objects]).T)
        top_index = np.argsort(similarity[0])[::-1][:5]
        # print the top 5 similar objects
        for i in top_index:
            print(self.objects[i].name, similarity[0])
            print("room: ", self.objects[i].room_id)
            obj_pcd = self.objects[i].pcd.paint_uniform_color([1, 0, 0])
            # find the room with a room id that matches the object's room id
            for room in self.rooms:
                if room.room_id == self.objects[i].room_id:
                    room_pcd = room.pcd
                    break
            o3d.visualization.draw_geometries([room_pcd, obj_pcd])

        # return the object with the highest similarity
        return self.objects[top_index[0]]

    def query_floor(self, query: str, query_method: str = "clip") -> int:
        """Find best matching floor for query.

        Args:
            query: Natural language query.
            query_method: Retrieval method ('clip' or 'llm').

        Returns:
            Best floor ID.
        """

        # TODO: assume that the self.floors are ordered according to the floor level in ascending order. Check again.
        zero_levels_list = [x.floor_zero_level for x in self.floors]
        zero_level_order_ids = np.argsort(zero_levels_list)

        # check whether the query is a number that is an integer
        try:
            return zero_level_order_ids[int(query) - 1]
        except BaseException:
            if query_method == "clip":
                text_feats = get_text_feats_multiple_templates(
                    [query], self.clip_model, self.clip_feat_dim
                )
                floor_names = ["floor " + str(i) for i in range(len(self.floors))]
                floor_embs = get_text_feats_multiple_templates(
                    floor_names, self.clip_model, self.clip_feat_dim
                )
                sim_mat = np.dot(text_feats, floor_embs.T)
                top_index = np.argsort(sim_mat[0])[::-1][0]
                return zero_level_order_ids[top_index]

            elif query_method == "vlm":
                floor_ids_list = [i + 1 for i in range(len(self.floors))]
                floor_id = infer_floor_id_from_query(floor_ids_list, query)
                return zero_level_order_ids[floor_id - 1]

    def vlm_choose(self, video_image_local_paths: List[str], instruction: str) -> int:
        """Select best frame using VLM for given instruction.

        Args:
            video_image_local_paths: List of frame image paths.
            instruction: Navigation instruction.

        Returns:
            Index of best matching frame.
        """

        system_prompt = """You are a robot operating in an indoor environment and your task is to respond to the user command about going to a specific location by finding the closest frame in the provided locations to navigate to."""

        # self.upload2oss(video_image_local_paths)

        video_prompt = []
        for i, img_path in enumerate(video_image_local_paths):
            video_prompt.append({"type": "text", "text": f"Frame:{i}"})
            video_prompt.append({"type": "image_url", "image_url": {"url": img_path}})
        instruction_prompt = f"User says: {instruction}. Can you find the closet frame in the provided locations to navigate to?"
        rules_prompt = """Rules to follow:
1. Output the frame id (integer) wrapped in the <frame_id> tag.
2. Carefully compare the candidate locations with the user instruction and select the closest one.  Describe the image you choose in detail to justify your choice after you respond the frame_id."""

        messages = [
            {
                "role": "user",
                "content": [
                    *video_prompt,
                    {
                        "type": "text",
                        "text": system_prompt,
                    },
                    {
                        "type": "text",
                        "text": instruction_prompt,
                    },
                    {
                        "type": "text",
                        "text": rules_prompt,
                    },
                ],
            }
        ]
        response_flag = False
        while not response_flag:
            try:
                print("Sending request stage 2 ...")
                response = self.client.chat.completions.create(
                    model=self.vlm_model,
                    messages=messages,
                    seed=123,
                )
                response_flag = True
            except Exception as e:
                print(e)
                time.sleep(1)
                print("Retrying ...")
        response = response.choices[0].message.content
        return response

    def detect_and_select_best_vlm(
        self, imglist: List[str], query: str, score_threshold: float = 0.5
    ) -> int:
        """Detect and rank objects in images using VLM.

        Args:
            imglist: List of image file paths.
            query: Object query string.
            score_threshold: Minimum confidence threshold.

        Returns:
            Index of image with highest confidence detection.
        """

        # self.upload2oss(imglist)

        results, scores = [], []

        for img in imglist:
            # Step 1: yes/no detection
            prompt_yesno = (
                f"Does this image contain a '{query}'? Answer strictly with 'yes' or 'no'."
            )
            messages_yesno = [
                {
                    "role": "system",
                    "content": "You are an object detector. Answer only 'yes' or 'no', no explanation.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_yesno},
                        {"type": "image_url", "image_url": {"url": img}},
                    ],
                },
            ]
            resp_yesno = self.client.chat.completions.create(
                model=self.vlm_model,
                messages=messages_yesno,
            )
            ans_raw = resp_yesno.choices[0].message.content.strip().lower()
            has_object = ans_raw == "yes"  # Strict match

            score = 0.0
            if has_object:
                # Step 2: Scoring
                prompt_score = f"On a scale from 0 to 1, how strongly does this image contain a '{query}'? Respond only with a single number (e.g., 0.73)."
                messages_score = [
                    {
                        "role": "system",
                        "content": "You are an object detector. Answer only with a single number between 0 and 1, no text.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_score},
                            {"type": "image_url", "image_url": {"url": img}},
                        ],
                    },
                ]
                resp_score = self.client.chat.completions.create(
                    model=self.vlm_model,
                    messages=messages_score,
                )
                ans_score_raw = resp_score.choices[0].message.content.strip()
                try:
                    score = float(ans_score_raw)
                    if not (0.0 <= score <= 1.0):
                        score = 0.0
                except Exception:
                    score = 0.0

                # Reject if score is below threshold
                if score < score_threshold:
                    has_object = False

            results.append(has_object)
            scores.append(score)
            print(
                f"[VLM] Image: {img} → raw_yesno='{ans_raw}', score={score:.3f}, has_object={has_object}, query={query}"
            )

        # Step 3: Select best (return None if none found)
        if any(results):
            best_idx = int(np.argmax(scores))
            best_image = imglist[best_idx]
        else:
            best_image = None

        return results, best_image

    def detect_object_in_image(
        self, img_path: str, query: str, score_threshold: float = 0.3
    ) -> bool:
        """Check if object matching query is visible in image.

        Args:
            img_path: Path to image file.
            query: Object query string.
            score_threshold: Minimum confidence threshold.

        Returns:
            True if object detected with sufficient confidence.
        """

        # self.upload2oss([img_path])
        img_url = img_path
        prompt = (
            f"On a scale from 0 to 1, does this image contain a '{query}'? "
            "Respond only with a single number between 0 and 1."
        )
        messages = [
            {
                "role": "system",
                "content": "You are an object detector. Answer only with a single number, no text.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.vlm_model,
            messages=messages,
        )
        ans_raw = resp.choices[0].message.content.strip()

        try:
            score = float(ans_raw)
            if not (0.0 <= score <= 1.0):
                score = 0.0
        except Exception:
            score = 0.0

        has_object = score >= score_threshold
        print(f"[VLM] Image: {img_url} → score={score:.3f}, has_object={has_object}")
        return has_object

    def visualize_goal_images(
        self,
        mean_depth: np.ndarray,
        goal_image_path_online: str,
        goal_image_path_by_clip: str,
        goal_image_path_by_vlm: str,
        save_name: str = "goal_compare.png",
    ) -> None:
        """Visualize goal image candidates from different retrieval methods.

        Args:
            mean_depth: Mean depth image.
            goal_image_path_online: Image path from online retrieval.
            goal_image_path_by_clip: Image path from CLIP retrieval.
            goal_image_path_by_vlm: Image path from VLM retrieval.
            save_name: Output visualization filename.
        """
        # Read the images
        img_online = cv2.imread(goal_image_path_online)
        img_vlm_best = cv2.imread(goal_image_path_by_clip)
        img_vlm = cv2.imread(goal_image_path_by_vlm)

        if img_vlm_best is None or img_vlm is None or img_online is None:
            raise FileNotFoundError("One of the image paths is invalid, please verify the paths")

        # Ensure both images have the same size (scaled to 640x480)
        img_vlm_best = cv2.resize(img_vlm_best, (640, 480))
        img_vlm = cv2.resize(img_vlm, (640, 480))
        img_online = cv2.resize(img_online, (640, 480))

        # Add labels in the top-left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 255, 0)  # Green

        cv2.putText(
            img_vlm_best, "BEST", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA
        )
        cv2.putText(img_vlm, "VLM", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(
            img_online, "ObjBestView", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA
        )
        cv2.putText(
            img_online,
            f"{mean_depth:.2f}",
            (10, 300),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        # Horizontal concatenation
        combined = np.hstack((img_online, img_vlm_best, img_vlm))
        # Save result
        save_path = os.path.join(self.curr_query_save_dir, save_name)
        cv2.imwrite(save_path, combined)
        # Visualization
        cv2.imshow("Goal Image Comparison", combined)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    def get_object_info(self, target_obj_id: int) -> Object:
        """Get object by ID.

        Args:
            target_obj_id: Object ID.

        Returns:
            Object instance.
        """
        target_object = self.objects[target_obj_id]
        target_object_id = target_object.object_id
        target_object_best_view_id = target_object.best_view_id
        best_view = None
        for view in self.views:
            if view.view_id == target_object_best_view_id:
                best_view = view
                break
        assert best_view is not None, "best view is None"
        best_view_image_path = best_view.img_path
        best_view_img_id = best_view.img_id
        return best_view_image_path, best_view_img_id, target_object_id

    def get_object_best_view(self, target_object: Object) -> str:
        """Get image path with best view of object.

        Args:
            target_object: Object instance.

        Returns:
            Path to image with best object visibility.
        """
        # target_object_id = target_object.object_id
        target_object_best_view_id = target_object.best_view_id
        best_view = None
        for view in self.views:
            if view.view_id == target_object_best_view_id:
                best_view = view
                break
        # assert best_view is not None, "best view is None"
        if best_view is None:
            return ""
        best_view_image_path = best_view.img_path
        return best_view_image_path

    def find_view_by_imgpath(self, img_path: str) -> View:
        """Get View object by image path.

        Args:
            img_path: Image file path.

        Returns:
            View instance.
        """
        for view in self.views:
            if view.img_path == img_path:
                return view, view.img_id
        return None, None

    def find_object_by_object_id(self, object_id: int) -> Object:
        """Find object by ID.

        Args:
            object_id: Object ID.

        Returns:
            Object instance.
        """
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def query_room_obj_slow_reasoning(
        self,
        instruction: str,
        room_query: str,
        object_query: str,
        negative_prompt: str,
        floor_id: int = -1,
        room_query_method: str = "label",
        object_query_method: str = "clip",
        update_flag: bool = True,
    ) -> Tuple[Room, List[Object]]:
        """Query rooms and objects with multi-modal reasoning.

        Args:
            instruction: Original navigation instruction.
            room_query: Room type query.
            object_query: Object query.
            negative_prompt: Negative instances to exclude.
            floor_id: Floor to search (-1 for all).
            room_query_method: Room retrieval method.
            object_query_method: Object retrieval method.
            update_flag: Whether to update visualization.

        Returns:
            Tuple of (best_room, list_of_objects).
        """

        print("process object query use vlm....")
        query_time_consumer = dict()
        query_time_consumer["room_query"] = room_query
        query_time_consumer["object_query"] = object_query
        query_time_consumer["negative_prompt"] = negative_prompt
        is_dectect_room = "unknown" not in room_query.lower()
        if room_query is None or room_query == "":
            is_dectect_room = False

        is_dectect_obj = "unknown" not in object_query.lower()
        if object_query is None or object_query == "":
            is_dectect_obj = False

        print("is_dectect_room: ", is_dectect_room, "is_dectect_obj: ", is_dectect_obj)

        # query room
        rooms_list = self.rooms if floor_id == -1 else self.floors[floor_id].rooms
        start_time = time.time()
        if room_query_method == "label" and is_dectect_room:
            print("query room use label")
            for room in rooms_list:
                assert (
                    room.name is not None
                ), "The name attribute for the room has not been generated"
            room_names_list = [room.name for room in rooms_list]
            room_embs = get_text_feats_multiple_templates(
                room_names_list, self.clip_model, self.clip_feat_dim
            )
            query_room_text_feats = get_text_feats_multiple_templates(
                [room_query], self.clip_model, self.clip_feat_dim
            )
            similarity = np.dot(query_room_text_feats, room_embs.T)
            top_index = np.argsort(similarity[0])[::-1]
            for i in top_index[:3]:
                print("room: ", rooms_list[i].room_id, rooms_list[i].name, similarity[0][i])
            same_sim_indices = []
            tar_sim = similarity[0, top_index[0]]
            same_sim_indices.append(top_index[0])
            for i in top_index[1:]:
                if np.abs(similarity[0, i] - tar_sim) < 1e-3:
                    same_sim_indices.append(i)

            target_rooms = [rooms_list[i] for i in same_sim_indices]
            target_room_ids = [target_room.room_id for target_room in target_rooms]
            target_ids = [i for i, x in enumerate(rooms_list) if x.room_id in target_room_ids]

        else:
            query_room_text_feats = get_text_feats_multiple_templates(
                [room_query], self.clip_model, self.clip_feat_dim
            )
            room2query_sim = dict()
            for room in rooms_list:
                embeddings = np.stack(room.embeddings)  # [view_num, 768]
                # [1, view_num], similarity between query and each view
                sims = np.dot(query_room_text_feats, embeddings.T)
                max_idx = np.argmax(sims)  # Find the position of maximum similarity
                max_sim = sims[0, max_idx]  # Maximum similarity value

                room2query_sim[room.room_id] = max_sim

            room2query_sim_sorted = {
                int(k.split("_")[-1]): v
                for k, v in sorted(room2query_sim.items(), key=lambda item: item[1], reverse=True)
            }
            target_ids = list(room2query_sim_sorted.keys())[
                0 : min(len(room2query_sim_sorted), 10)
            ]
        room_retrival_time = time.time() - start_time

        # print query room result
        print("target_room_ids: ", target_ids)
        query_time_consumer["room_retrieval_by_clip"] = room_retrival_time

        # query object
        if not is_dectect_obj:
            print("not found object, use llm to find intention object")
            object_query = self.generate_object_queries(instruction)

        if object_query in negative_prompt:
            query_id = negative_prompt.index(object_query)
        else:
            query_id = None

        if query_id is None:
            object_query = [object_query, *negative_prompt]
            query_id = 0
        else:
            object_query = negative_prompt

        query_object_text_feats = get_text_feats_multiple_templates(
            object_query, self.clip_model, self.clip_feat_dim
        )  # (len(categories), feat_dim)

        room_ids_list = []
        for obj in self.objects:
            for i, room in enumerate(rooms_list):
                if obj.room_id == room.room_id:
                    room_ids_list.append(i)
                    break

        if object_query_method == "clip":
            if len(target_ids) != 0:
                objects_list = []
                room_ids_list = []
                for i in target_ids:
                    objects_list.extend(rooms_list[i].objects)
                    room_ids_list.extend([i] * len(rooms_list[i].objects))
            objects_list: List[Object]
            object_embs = np.array([obj.embedding for obj in objects_list])
            sim_mat = np.dot(query_object_text_feats, object_embs.T)
            top_index = np.argsort(sim_mat[query_id])[::-1][:10]  # top-10
            for i in top_index:
                print("object name, score: ", objects_list[i].name, sim_mat[0][i])
                print("object id: ", objects_list[i].object_id)

            top_k = 5
            top_index = np.argsort(sim_mat[query_id])[::-1][:top_k]
            if len(negative_prompt) > 0:
                # category id for each object
                cls_ids = np.argmax(sim_mat, axis=0)
                print(f"cls_ids: {cls_ids}")
                # max scores for each object
                max_scores = np.max(sim_mat, axis=0)
                # find the obj ids that assign max score to the target category
                obj_ids = np.where(cls_ids == query_id)[0]
                if len(obj_ids) > 0:
                    obj_scores = max_scores[obj_ids]
                    resort_ids = np.argsort(
                        -obj_scores
                    )  # sort the obj ids based on max score (descending)
                    top_index = obj_ids[resort_ids]  # get the top index
                    top_index = top_index[:top_k]
            target_object_id = [objects_list[i].object_id for i in top_index]
            target_room_id = [room_ids_list[i] for i in top_index]
            target_id = []
            for ti in target_object_id:
                target_id.append([i for i, x in enumerate(self.objects) if x.object_id == ti][0])
            FastMatching_time = time.time() - start_time
            query_time_consumer["FastMatching_time"] = FastMatching_time

        save_json_path = os.path.join(self.curr_query_save_dir, "query_time_consumer.json")
        best_object = self.objects[target_id[0]]
        best_object_best_view_id = best_object.best_view_id
        best_view = None
        for view in self.views:
            if view.view_id == best_object_best_view_id:
                best_view = view
                break

        if best_view is None:
            total_online_query_time = FastMatching_time
            query_time_consumer["total_query_time"] = f"{total_online_query_time:.4f} seconds"
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(query_time_consumer, f, ensure_ascii=False, indent=4)
            res_dict = dict()
            res_dict["FastMatching"] = FastMatching_time
            res_dict["ObjectInImageCheck"] = 0.0
            res_dict["VLM_Rethinking"] = 0.0
            res_dict["Re_Matching"] = 0.0
            res_dict["Total_Time"] = total_online_query_time
            return res_dict, target_id, target_room_id

        best_view_image_path = best_view.img_path
        best_view_img_id = best_view.img_id
        print("online_best_view_image_path: ", best_view_image_path)
        goal_image_path_online = best_view_image_path
        query_time_consumer["top1_image_path_online_object_best_view"] = goal_image_path_online
        start_time = time.time()
        Object_in_goal_view_check = self.detect_object_in_image(
            best_view_image_path, object_query[query_id]
        )
        Object_in_goal_view_check_time = time.time() - start_time
        query_time_consumer["Object_in_goal_view_check_time"] = (
            f"{Object_in_goal_view_check_time:.4f} seconds"
        )
        query_time_consumer["Object_in_goal_view_check_res"] = Object_in_goal_view_check
        if Object_in_goal_view_check:
            total_online_query_time = FastMatching_time + Object_in_goal_view_check_time
            query_time_consumer["total_query_time"] = f"{total_online_query_time:.4f} seconds"
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(query_time_consumer, f, ensure_ascii=False, indent=4)
            res_dict = dict()
            res_dict["FastMatching"] = FastMatching_time
            res_dict["ObjectInImageCheck"] = Object_in_goal_view_check_time
            res_dict["VLM_Rethinking"] = 0.0
            res_dict["Re_Matching"] = 0.0
            res_dict["Total_Time"] = total_online_query_time
            return res_dict, target_id, target_room_id

        else:  # run vlm-refine
            total_online_query_time = FastMatching_time + Object_in_goal_view_check_time
            all_image_incides = []
            all_image_embedding = []
            for room in rooms_list:
                img_ids = room.sample_images  # list of images
                embs = room.clip_embeddings  # shape [view, 768]
                # Ensure lengths are aligned
                assert len(img_ids) == len(
                    embs
                ), f"Number of images ({len(img_ids)}) != embeddings ({len(embs)})"
                all_image_incides.extend(img_ids)
                all_image_embedding.extend(embs)  # Each embedding corresponds to one image
            print("all_image_incides: ", len(all_image_incides))
            print("all_image_embedding: ", len(all_image_embedding))
            room_ids = target_ids
            for room_id in room_ids[:1]:
                print(f"find goal image in room {room_id}")
                start_time = time.time()
                # find goal image by clip
                gloal_embedding = np.stack(all_image_embedding)  # [total_view_num, 768]
                sims = np.dot(
                    query_object_text_feats[0], gloal_embedding.T
                )  # [1, view_num], similarity between query and each view
                clip_max_idx = np.argmax(sims)  # Find the position of maximum similarity

                # Compute top_k, ensuring it does not exceed the length of sims
                top_k = min(24, sims.shape[0])
                top_idx = np.argsort(sims)[-top_k:][::-1]  # indices of top_k in descending order

                # find goal image by clip
                goal_image_path_by_clip = self.dataset.frameId2imgPath[
                    all_image_incides[clip_max_idx]
                ]
                print(f"goal_image_path_by_clip: {goal_image_path_by_clip}")
                end_time = time.time()
                query_time_consumer[f"goal_image_reterival_by_clip_{room_id}"] = (
                    end_time - start_time
                )
                print(f"find goal image by clip elapsed time: {end_time - start_time:.4f} seconds")
                query_time_consumer["goal_image_path_by_clip"] = goal_image_path_by_clip
                start_time = time.time()

                # find goal image by vlm
                room_clip_refined_topk_image_local_paths = [
                    self.dataset.frameId2imgPath[all_image_incides[idx]] for idx in top_idx
                ]
                room_image_local_paths = room_clip_refined_topk_image_local_paths
                print("room_image_local_paths: ", room_image_local_paths)

                response = self.vlm_choose(room_image_local_paths, instruction)
                print(response)
                match = re.findall(r"\d+", response)
                if match:
                    frame_id = match[0]
                    goal_img_path = room_image_local_paths[int(frame_id)]
                else:
                    print("No frame id found in response text.")
                    goal_img_path = None
                goal_image_path_by_vlm = goal_img_path
                end_time = time.time()
                query_time_consumer[f"goal_image_reterival_by_vlm_{room_id}"] = (
                    end_time - start_time
                )
                query_time_consumer["goal_image_path_by_vlm"] = goal_image_path_by_vlm
                print(f"find goal image by vlm elapsed time: {end_time - start_time:.4f} seconds")
                print("goal_image_path_by_vlm: ", goal_image_path_by_vlm)

                # judge whether object in goal image
                select_imgs = [
                    goal_image_path_online,
                    goal_image_path_by_clip,
                    goal_image_path_by_vlm,
                ]

                vlm_check_start_time = time.time()
                vlm_check_result, best_image_path = self.detect_and_select_best_vlm(
                    select_imgs, object_query[query_id]
                )
                vlm_check_time = time.time() - vlm_check_start_time
                query_time_consumer["vlm_check_time"] = vlm_check_time
                print("Detection results:", vlm_check_result)  # [True, False]
                print("Best image:", best_image_path)
                query_time_consumer["detection_results"] = vlm_check_result
                query_time_consumer["best_image"] = best_image_path
                print(best_image_path != goal_image_path_online)
                print("update_flatg ", update_flag)
                avg_distance_in_vlmview = -1.0
                vlm_refine_time_start = time.time()
                if (
                    vlm_check_result[0] is False
                    and update_flag
                    and best_image_path != goal_image_path_online
                    and best_image_path is not None
                ):
                    print("performing vlm refineing..............................")
                    objs_embedding_in_view = []
                    vlm_refine_best_view, vlm_refine_best_view_img_id = self.find_view_by_imgpath(
                        best_image_path
                    )
                    assert vlm_refine_best_view is not None
                    object_ids_in_view = vlm_refine_best_view.object_ids
                    for object_id in object_ids_in_view:
                        object_target = self.find_object_by_object_id(object_id)
                        assert object_target is not None
                        objs_embedding_in_view.append(object_target.embedding)
                    if len(objs_embedding_in_view) > 0:
                        objs_embedding_in_view = np.stack(objs_embedding_in_view)
                        obj_sims = np.dot(
                            query_object_text_feats[0], objs_embedding_in_view.T
                        )  # [1, obj_num], similarity between query and each object under this view
                        max_obj_idx = np.argmax(
                            obj_sims
                        )  # Find the position of maximum similarity
                        max_obj_sim = obj_sims[max_obj_idx]  # Maximum similarity value
                        print(f"max_obj_sim: {max_obj_sim}")
                        max_sim_object_id = object_ids_in_view[max_obj_idx]
                        final_object = self.find_object_by_object_id(max_sim_object_id)
                        final_obj_pcd = final_object.pcd
                        camera_matrix = self.dataset.get_camera_intrinsics()
                        img, _, pose, _, _ = self.dataset[vlm_refine_best_view_img_id]
                        if not isinstance(img, np.ndarray):
                            img = np.array(img)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        avg_distance_in_vlmview = visualize_pcd_on_image(
                            final_obj_pcd,
                            img,
                            camera_matrix,
                            np.linalg.inv(pose),
                            save_path=os.path.join(
                                self.curr_query_save_dir,
                                f"vlm_refine_object_id_{final_object.object_id}.png",
                            ),
                        )
                        new_objects_path = os.path.join(self.graph_path, "objects_update")
                        if not os.path.exists(new_objects_path):
                            os.makedirs(new_objects_path)
                        final_object.save(os.path.join(self.graph_path, "objects_update"))
                vlm_refine_time = time.time() - vlm_refine_time_start
                query_time_consumer["vlm_refine_time"] = vlm_refine_time

                # Calculate distance
                obj_pcd = deepcopy(best_object.pcd)
                camera_matrix = self.dataset.get_camera_intrinsics()
                img, _, pose, _, _ = self.dataset[best_view_img_id]
                _, mean_depth_online = check_object_in_view(
                    np.array(img).shape[1],
                    np.array(img).shape[0],
                    camera_matrix,
                    np.linalg.inv(pose),
                    np.array(obj_pcd.points),
                    return_depth=True,
                )
                if best_image_path is not None:
                    self.visualize_goal_images(
                        mean_depth_online,
                        goal_image_path_online,
                        goal_image_path_by_clip,
                        goal_image_path_by_vlm,
                        save_name=f"goal_compare_room_{room_id}.png",
                    )
                else:
                    best_image_path = goal_image_path_by_vlm
                    self.visualize_goal_images(
                        mean_depth_online,
                        goal_image_path_online,
                        goal_image_path_by_clip,
                        goal_image_path_by_vlm,
                        save_name=f"goal_compare_room_{room_id}.png",
                    )

                # Save as JSON file
            total_query_time_offline = total_online_query_time + vlm_check_time + vlm_refine_time
            query_time_consumer["total_query_time"] = f"{total_query_time_offline:.4f} seconds"
            query_time_consumer["online_object_distance_in_online_view"] = mean_depth_online
            query_time_consumer["vlmref_object_distance_in_ofline_view"] = avg_distance_in_vlmview
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(query_time_consumer, f, ensure_ascii=False, indent=4)
            res_dict = dict()
            res_dict["FastMatching"] = FastMatching_time
            res_dict["ObjectInImageCheck"] = Object_in_goal_view_check_time
            res_dict["VLM_Rethinking"] = vlm_check_time
            res_dict["Re_Matching"] = vlm_refine_time
            res_dict["Total_Time"] = total_query_time_offline
            return res_dict, target_id, target_room_id

    def query_object(
        self,
        query: str,
        floor_id: int = -1,
        room_ids: List[int] = [],
        query_method: str = "clip",
        top_k: int = 1,
        negative_prompt: List[str] = [],
        return_scores: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """Retrieve objects matching query within spatial constraints.

        Args:
            query: Object query string.
            floor_id: Restrict to floor (-1 for all).
            room_ids: Restrict to rooms ([] for all).
            query_method: Retrieval method ('clip' or 'llm').
            top_k: Number of results to return.
            negative_prompt: Objects to exclude.
            return_scores: Return similarity scores.

        Returns:
            Tuple of (object_ids, scores_if_requested).
        """

        if query in negative_prompt:
            query_id = negative_prompt.index(query)
        else:
            query_id = None

        if query_id is None:
            query = [query, *negative_prompt]
            query_id = 0
        else:
            query = negative_prompt

        print(f"query_id: {query_id}")
        print(f"categories list: {query}")

        query_text_feats = get_text_feats_multiple_templates(
            query, self.clip_model, self.clip_feat_dim
        )  # (len(categories), feat_dim)

        # Build default room→object mapping from all objects
        room_id_by_obj_idx = []
        for obj in self.objects:
            for i, room in enumerate(self.rooms):
                if obj.room_id == room.room_id:
                    room_id_by_obj_idx.append(i)
                    break

        objects_list: List[Object] = self.objects
        room_ids_list: List[int] = room_id_by_obj_idx

        if len(room_ids) != 0:
            objects_list = []
            room_ids_list = []
            for i in room_ids:
                src_rooms = self.floors[floor_id].rooms if floor_id != -1 else self.rooms
                objects_list.extend(src_rooms[i].objects)
                room_ids_list.extend([i] * len(src_rooms[i].objects))

        if query_method == "clip":
            object_embs = np.array([obj.embedding for obj in objects_list])
            sim_mat = np.dot(query_text_feats, object_embs.T)

            # Log top-10 matches for debugging
            debug_top = np.argsort(sim_mat[query_id])[::-1][:10]
            for i in debug_top:
                print("object name, score: ", objects_list[i].name, sim_mat[0][i])
                print("object id: ", objects_list[i].object_id)

            top_index = np.argsort(sim_mat[query_id])[::-1][:top_k]
            if len(negative_prompt) > 0:
                cls_ids = np.argmax(sim_mat, axis=0)
                print(f"cls_ids: {cls_ids}")
                max_scores = np.max(sim_mat, axis=0)
                obj_ids = np.where(cls_ids == query_id)[0]
                if len(obj_ids) > 0:
                    resort_ids = np.argsort(-max_scores[obj_ids])
                    top_index = obj_ids[resort_ids][:top_k]

            target_object_id = [objects_list[i].object_id for i in top_index]
            object_id_map = {obj.object_id: idx for idx, obj in enumerate(self.objects)}
            target_id = [object_id_map[oid] for oid in target_object_id]
            target_room_id = [room_ids_list[i] for i in top_index]
            target_scores = [sim_mat[query_id][i] for i in top_index]

            if return_scores:
                return target_id, target_room_id, target_scores
            return target_id, target_room_id
        raise NotImplementedError(f"Unsupported query_method: {query_method}")

    def query_room(
        self,
        query: str,
        floor_id: int = -1,
        query_method: str = "view_embedding",
        top_k: int = 3,
    ) -> List[int]:
        """Retrieve rooms matching query.

        Args:
            query: Room query string.
            floor_id: Restrict to floor (-1 for all).
            query_method: Retrieval method ('view_embedding' or 'llm').
            top_k: Number of results to return.

        Returns:
            List of room IDs.
        """

        is_room_text_valid = query is not None and query != "" and "unknown" not in query.lower()
        query_text_feats = get_text_feats_multiple_templates(
            [query], self.clip_model, self.clip_feat_dim
        )

        rooms_list: List[Room] = self.floors[floor_id].rooms if floor_id != -1 else self.rooms

        if query_method == "label" and is_room_text_valid:
            print("query room use label")
            for room in rooms_list:
                assert room.name is not None, "Room name has not been generated"
            room_names_list = [room.name for room in rooms_list]
            room_embs = get_text_feats_multiple_templates(
                room_names_list, self.clip_model, self.clip_feat_dim
            )
            similarity = np.dot(query_text_feats, room_embs.T)
            top_index = np.argsort(similarity[0])[::-1]
            for i in top_index[:3]:
                print("room: ", rooms_list[i].room_id, rooms_list[i].name, similarity[0][i])

            # Collect all indices with similarity equal to the best match
            tar_sim = similarity[0, top_index[0]]
            same_sim_indices = [i for i in top_index if np.abs(similarity[0, i] - tar_sim) < 1e-3]
            room_id_set = {rooms_list[i].room_id for i in same_sim_indices}
            return [i for i, r in enumerate(rooms_list) if r.room_id in room_id_set]
        else:
            print("query room use view embedding")
            room2query_sim = {
                room.room_id: float(np.max(np.dot(query_text_feats, np.stack(room.embeddings).T)))
                for room in rooms_list
            }
            sorted_ids = [
                int(k.split("_")[-1])
                for k, _ in sorted(room2query_sim.items(), key=lambda x: x[1], reverse=True)
            ]
            limit = top_k if is_room_text_valid else top_k * 2
            return sorted_ids[: min(len(sorted_ids), limit)]

    def query_hierarchy_protected_icra(
        self, query_instruction: str, top_k: int = 1, use_vlm: bool = False
    ) -> Tuple[Floor, Room, List[Object]]:
        """Query graph with hierarchical reasoning (ICRA method).

        Args:
            query_instruction: Natural language instruction.
            top_k: Number of object results to return.
            use_vlm: Whether to use VLM for ranking.

        Returns:
            Tuple of (floor, room, objects).
        """

        negative_labels = ["background"]
        start_time = time.time()
        floor_query, room_query, object_query = parse_hier_query_use_prompt_insentence_parse_icra(
            self.cfg, query_instruction
        )
        llm_parse_time = time.time() - start_time
        print("llm_parse_time: ", llm_parse_time)

        if "Exhibition" in room_query:
            negative_labels = ["wall"]

        floor_id = self.query_floor(floor_query) if floor_query is not None else -1
        print(f"floor id: {floor_id}")

        is_dectect_room = "unknown" not in room_query.lower()
        if room_query is None or room_query == "":
            is_dectect_room = False

        is_dectect_obj = "unknown" not in object_query.lower()
        if object_query is None or object_query == "":
            is_dectect_obj = False

        print("is_dectect_room: ", is_dectect_room, "is_dectect_obj: ", is_dectect_obj)

        # ## offline use vlm to check and update object-retrieval
        if use_vlm:
            res_dict, object_ids, room_ids = self.query_room_obj_slow_reasoning(
                query_instruction,
                room_query,
                object_query,
                negative_prompt=negative_labels,
                floor_id=floor_id,
                room_query_method="label",
                object_query_method="clip",
                update_flag=True,
            )
            res_dict["LLM_Parse_Time"] = llm_parse_time
            res_dict["room_query"] = room_query
            res_dict["object_query"] = object_query
            res_dict["negative_labels"] = negative_labels
            return (
                self.floors[floor_id] if floor_id != -1 else None,
                (
                    [self.floors[floor_id].rooms[k] for k in room_ids]
                    if floor_id != -1
                    else [self.rooms[k] for k in room_ids]
                ),
                [self.objects[i] for i in object_ids],
                res_dict,
            )
        room_ids = (
            self.query_room(room_query, floor_id=floor_id, query_method="label")
            if room_query is not None
            else []
        )
        print("room_ids: ", room_ids)

        print(f"room ids: {room_ids}")
        object_ids, room_ids, object_scores = (
            self.query_object(
                object_query,
                floor_id=floor_id,
                room_ids=room_ids,
                top_k=top_k,
                negative_prompt=negative_labels,
                return_scores=True,
            )
            if object_query is not None
            else ([], [], [])
        )

        res_dict = dict()
        res_dict["room_query"] = room_query
        res_dict["object_query"] = object_query
        res_dict["negative_labels"] = negative_labels
        res_dict["LLM_Parse_Time"] = llm_parse_time
        res_dict["FastMatching"] = 0.0
        res_dict["ObjectInImageCheck"] = 0.0
        res_dict["VLM_Rethinking"] = 0.0
        res_dict["Re_Matching"] = 0.0
        res_dict["Total_Time"] = 0.0

        print("query_hierarchy_protected_icra fun cost: ", time.time() - start_time)
        return (
            self.floors[floor_id] if floor_id != -1 else None,
            (
                [self.floors[floor_id].rooms[k] for k in room_ids]
                if floor_id != -1
                else [self.rooms[k] for k in room_ids]
            ),
            [self.objects[i] for i in object_ids],
            res_dict,
        )

    def save_full_pcd(self, path: str, visualize_clusters: bool = True) -> None:
        """Save point cloud to disk with optional visualization.

        Args:
            path: Directory to save PCD files.
            visualize_clusters: Whether to visualize point clusters.
        """

        if not os.path.exists(path):
            os.makedirs(path)
        o3d.io.write_point_cloud(os.path.join(path, "full_pcd.ply"), self.full_pcd)
        print("full pcd saved to disk in {}".format(path))

        if visualize_clusters:
            # Visualize the final full point cloud with cluster coloring
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
                self.full_pcd,
                save_dir=path,
                dbscan_eps=dbscan_eps,
                dbscan_min_points=dbscan_min,
            )

        return None

    def load_full_pcd(self, path: str) -> None:
        """Load point cloud from disk.

        Args:
            path: Directory containing PCD files.
        """

        if not os.path.exists(path):
            print("full pcd not found in {}".format(path))
            return None
        self.full_pcd = o3d.io.read_point_cloud(os.path.join(path, "full_pcd.ply"))
        print(
            "full pcd loaded from disk with shape {}".format(
                np.asarray(self.full_pcd.points).shape
            )
        )
        return self.full_pcd

    def save_full_pcd_feats(self, path: str) -> None:
        """Save per-point CLIP feature vectors.

        Args:
            path: Directory to save feature files.
        """

        if not os.path.exists(path):
            os.makedirs(path)

        valid_mask_pcds = []
        valid_mask_feats = []

        for pcd, feat in zip(self.mask_pcds, self.mask_feats):
            if len(pcd.points) > 0:
                valid_mask_pcds.append(pcd)
                valid_mask_feats.append(feat)

        self.mask_pcds = valid_mask_pcds
        self.mask_feats = valid_mask_feats

        # check if the full pcd feats is empty list
        if len(self.mask_feats) != 0:
            self.mask_feats = np.array(self.mask_feats)
            torch.save(torch.from_numpy(self.mask_feats), os.path.join(path, "mask_feats.pt"))
        if len(self.full_feats_array) != 0:
            torch.save(
                torch.from_numpy(self.full_feats_array),
                os.path.join(path, "full_feats.pt"),
            )
        print("full pcd feats saved to disk in {}".format(path))
        return None

    def load_full_pcd_feats(
        self, path: str, full_feats: bool = False, normalize: bool = True
    ) -> None:
        """Load per-point CLIP feature vectors.

        Args:
            path: Directory containing feature files.
            full_feats: Whether to load all features.
            normalize: Whether to normalize features.
        """

        if not os.path.exists(path):
            print("full pcd feats not found in {}".format(path))
            return None

        if full_feats:
            self.full_feats_array = torch.load(os.path.join(path, "full_feats.pt")).float()
            if normalize:
                self.full_feats_array = (
                    torch.nn.functional.normalize(self.full_feats_array, p=2, dim=-1).cpu().numpy()
                )
            else:
                self.full_feats_array = self.full_feats_array.cpu().numpy()
            print(
                "full pcd feats loaded from disk with shape {}".format(self.full_feats_array.shape)
            )
            return self.full_feats_array
        else:
            self.mask_feats = torch.load(os.path.join(path, "mask_feats.pt")).float()
            if normalize:
                self.mask_feats = (
                    torch.nn.functional.normalize(self.mask_feats, p=2, dim=-1).cpu().numpy()
                )
            else:
                self.mask_feats = self.mask_feats.cpu().numpy()
            print("full pcd feats loaded from disk with shape {}".format(self.mask_feats.shape))
            return self.mask_feats

    def print_details(self) -> None:
        print("number of floors: ", len(self.floors))
        print("number of rooms: ", len(self.rooms))
        print("number of objects: ", len(self.objects))
        return None

    def save_masked_pcds(self, path: str, state: str = "both") -> None:
        """Save segmented object/room point clouds.

        Args:
            path: Directory to save PCD files.
            state: Which masks to save ('objects', 'rooms', or 'both').
        """

        tqdm.write("-- removing small and empty masks --")
        for i, pcd in reversed(list(enumerate(self.mask_pcds))):
            if len(pcd.points) < 10:
                self.mask_pcds.pop(i)
                self.mask_feats.pop(i)

        for i, pcd in reversed(list(enumerate(self.mask_pcds))):
            if pcd.is_empty():
                self.mask_pcds.pop(i)
                self.mask_feats.pop(i)

        if state == "both":
            if not os.path.exists(path):
                os.makedirs(path)
            objects_path = os.path.join(path, "objects")
            if not os.path.exists(objects_path):
                os.makedirs(objects_path)
            print("number of masked pcds: ", len(self.mask_pcds))
            print("number of mask_feats: ", len(self.mask_feats))
            for i, pcd in enumerate(self.mask_pcds):
                o3d.io.write_point_cloud(os.path.join(objects_path, "pcd_{}.ply".format(i)), pcd)

            masked_pcd = o3d.geometry.PointCloud()
            for pcd in self.mask_pcds:
                pcd.paint_uniform_color(np.random.rand(3))
                masked_pcd += pcd
            o3d.io.write_point_cloud(os.path.join(path, "masked_pcd.ply"), masked_pcd)
            print("masked pcds saved to disk in {}".format(path))

        elif state == "objects":
            if not os.path.exists(path):
                os.makedirs(path)
            for i, pcd in enumerate(self.mask_pcds):
                o3d.io.write_point_cloud(os.path.join(objects_path, "pcd_{}.ply".format(i)), pcd)
            print("masked pcds saved to disk in {}".format(path))

        elif state == "full":
            if not os.path.exists(path):
                os.makedirs(path)
            masked_pcd = o3d.geometry.PointCloud()
            for pcd in self.mask_pcds:
                pcd.paint_uniform_color(np.random.rand(3))
                masked_pcd += pcd
            o3d.io.write_point_cloud(os.path.join(path, "masked_pcd.ply"), masked_pcd)
            print("masked pcds saved to disk in {}".format(path))

    def load_masked_pcds(self, path: str) -> None:
        """Load segmented object/room point clouds.

        Args:
            path: Directory containing PCD files.
        """

        # make sure that self.mask_feats is already loaded
        if len(self.mask_feats) == 0:
            print("load full pcd feats first")
            return None
        if os.path.exists(os.path.join(path, "objects")):
            self.mask_pcds = []
            number_of_pcds = len(os.listdir(os.path.join(path, "objects")))
            not_found = []
            for i in range(number_of_pcds):
                if os.path.exists(os.path.join(path, "objects", "pcd_{}.ply".format(i))):
                    self.mask_pcds.append(
                        o3d.io.read_point_cloud(
                            os.path.join(path, "objects", "pcd_{}.ply".format(i))
                        )
                    )
                else:
                    print("masked pcd {} not found in {}".format(i, path))
                    not_found.append(i)
            print("number of masked pcds loaded from disk {}".format(len(self.mask_pcds)))
            # remove mask_feats that are not found, guarding against out-of-bounds indices
            not_found = [i for i in not_found if i < len(self.mask_feats)]
            self.mask_feats = np.delete(self.mask_feats, not_found, axis=0)
            print("number of mask_feats loaded from disk {}".format(len(self.mask_feats)))
            return self.mask_pcds
        else:
            print("masked pcds for objects not found in {}".format(path))
            return None

    def transform(self, transform: np.ndarray) -> None:
        """Apply rigid transformation to entire graph.

        Args:
            transform: 4x4 transformation matrix.
        """

        self.full_pcd.transform(transform)
        for i, pcd in enumerate(self.mask_pcds):
            self.mask_pcds[i].transform(transform)
        return None

    def visualize_instances(self) -> None:
        """Visualize all segmented objects and rooms."""

        all_objects_pcd = o3d.geometry.PointCloud()
        number_of_objects = 0
        for i, node in enumerate(self.graph.nodes):
            if isinstance(node, Object):
                print("object name: ", node.name, node.object_id)
                print("number of points: ", len(node.pcd.points))
                all_objects_pcd += node.pcd
                number_of_objects += 1
        print("number of objects: ", number_of_objects)
        o3d.visualization.draw_geometries([all_objects_pcd])
        return None

    # def upload2oss(self, retrieved_img_list: list):
    #     import oss2
    #     from oss2.credentials import EnvironmentVariableCredentialsProvider

    #     self.upload_flag = True
    #     self.force_reupload = False
    #     self.src_img_root = os.path.dirname(retrieved_img_list[0])
    #     img_dir_prefix = f"{os.path.basename(self.src_img_root)}/images"
    #     img_list = retrieved_img_list  # Do not sort
    #     self.downsampeld_img_list = img_list
    #     auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
    #     bucket = oss2.Bucket(auth, "oss-cn-beijing.aliyuncs.com", "mapvln")
    #     self.oss_img_list = []
    #     for file in tqdm(self.downsampeld_img_list):
    #         file_name = os.path.basename(file)
    #         self.oss_img_list.append(
    #             f"https://mapvln.oss-cn-beijing.aliyuncs.com/{img_dir_prefix}/{file_name}"
    #         )
    #         oss_url = f"{img_dir_prefix}/{file_name}"
    #         if self.upload_flag:
    #             if not bucket.object_exists(oss_url) or self.force_reupload:
    #                 bucket.put_object_from_file(
    #                     oss_url,
    #                     file,
    #                 )
    #             else:
    #                 print(f"{oss_url} already exists in Aliyun OSS, skipping.")
    #     print(f"Uploaded {len(self.downsampeld_img_list)} images to Aliyun OSS.")
