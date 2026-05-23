"""Label feature computation and caching utilities."""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from memory.hmsg.utils.clip_utils import get_text_feats_multiple_templates
from memory.hmsg.utils.constants import (
    COCO_STUFF_CLASSES,
    MATTERPORT_GT_LABELS,
    MATTERPORT_LABELS_40,
    MATTERPORT_LABELS_160,
    OPENVOCAB_MATTERPORT_LABELS,
)


def _flatten_classes(classes_dict_or_list) -> List[str]:
    """Flatten class definitions into a single list."""
    if isinstance(classes_dict_or_list, dict):
        return list(classes_dict_or_list.values())
    return classes_dict_or_list


def _load_csv_classes(csv_path: str) -> List[str]:
    """Load classes from CSV file."""
    df = pd.read_csv(csv_path, header=0, sep=";")
    return list(df.iloc[:, 0].values)


# Mapping: label_name -> (classes_source, cache_filename)
LABEL_REGISTRY = {
    "COCO_STUFF_CLASSES": (COCO_STUFF_CLASSES, "text_feats_COCO_STUFF_CLASSES.npy"),
    "MATTERPORT_LABELS_160": (MATTERPORT_LABELS_160, "text_feats_MATTERPORT_LABELS_160.npy"),
    "MATTERPORT_LABELS_40": (MATTERPORT_LABELS_40, "text_feats_MATTERPORT_LABELS_40.npy"),
    "MATTERPORT_GT_LABELS": (MATTERPORT_GT_LABELS, "text_feats_MATTERPORT_GT_LABELS.npy"),
    "OPENVOCAB_MATTERPORT_LABELS": (None, "text_feats_OPENVOCAB_MATTERPORT_LABELS.npy"),  # Special handling
}

CSV_LABEL_REGISTRY = {
    "HM3DSEM_LABELS": ("HM3D_CountsOfObjectTypes.csv", "text_feats_HM3DSEM_LABELS.npy"),
    "IMAGENET21K_LABELS": ("imagenet21k.csv", "text_feats_IMAGENET21K_LABELS.npy"),
    "SCANNET200": ("scannet200.csv", "text_feats_SCANNET200_LABELS.npy"),
    "SCANNET20": ("scannet20.csv", "text_feats_SCANNET20_LABELS.npy"),
    "FINALLABEL": ("final_label.csv", "text_feats_FINALLABEL_LABELS.npy"),
}


def compute_label_feats(
    clip_model, clip_feat_dim: int, label_feat_path: str, classes: List[str], cache_filename: str
) -> Tuple[np.ndarray, List[str]]:
    """Load precomputed or compute new label features.

    Args:
        clip_model: CLIP model.
        clip_feat_dim: Feature dimension.
        label_feat_path: Directory to store/load features.
        classes: List of class names.
        cache_filename: Filename for cached features.

    Returns:
        Tuple of (text_features, classes).
    """
    os.makedirs(label_feat_path, exist_ok=True)
    cache_path = os.path.join(label_feat_path, cache_filename)

    if os.path.exists(cache_path) and cache_path.endswith(".npy"):
        return np.load(cache_path, allow_pickle=True), classes

    text_feats = get_text_feats_multiple_templates(classes, clip_model, clip_feat_dim)
    np.save(cache_path, text_feats)
    return text_feats, classes


def get_label_feats(clip_model, clip_feat_dim: int, obj_labels: str, label_feat_path: str = None) -> Tuple[np.ndarray, List[str]]:
    """Get label features from registry by name.

    Args:
        clip_model: CLIP model.
        clip_feat_dim: Feature dimension.
        obj_labels: Label registry key (e.g., 'MATTERPORT_LABELS_160').
        label_feat_path: Optional override for feature storage path.

    Returns:
        Tuple of (text_features, classes).

    Raises:
        ValueError: If obj_labels not found in registry.
    """
    if label_feat_path is None:
        label_feat_path = os.path.dirname(os.path.abspath(__file__))

    # Handle standard label registries
    if obj_labels in LABEL_REGISTRY:
        classes_source, cache_filename = LABEL_REGISTRY[obj_labels]

        if obj_labels == "OPENVOCAB_MATTERPORT_LABELS":
            # Special handling: flatten nested dict
            classes = set()
            for key, val in OPENVOCAB_MATTERPORT_LABELS.items():
                classes.add(key)
                classes.update(val)
            classes = list(classes)
        else:
            classes = _flatten_classes(classes_source)

        return compute_label_feats(clip_model, clip_feat_dim, label_feat_path, classes, cache_filename)

    # Handle CSV-based registries
    if obj_labels in CSV_LABEL_REGISTRY:
        csv_filename, cache_filename = CSV_LABEL_REGISTRY[obj_labels]
        csv_path = os.path.join(label_feat_path, csv_filename)
        classes = _load_csv_classes(csv_path)
        return compute_label_feats(clip_model, clip_feat_dim, label_feat_path, classes, cache_filename)

    # Unknown label type
    available = list(LABEL_REGISTRY.keys()) + list(CSV_LABEL_REGISTRY.keys())
    raise ValueError(f"Unknown obj_labels: {obj_labels}. Available: {available}")
