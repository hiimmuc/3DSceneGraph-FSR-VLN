"""Persistence utilities for graph entities (SRP: Single Responsibility for I/O)."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d as o3d


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class PersistenceHandler(ABC):
    """Abstract base class for entity persistence (SRP: I/O Responsibility)."""

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Convert entity to serializable dictionary."""
        pass

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Load entity from serialized dictionary."""
        pass


class EntityPersistence:
    """Helper class for consistent entity save/load operations (DRY: Don't Repeat Yourself)."""

    @staticmethod
    def save_entity_with_pcd(
        entity_id: str, metadata: Dict[str, Any], pcd: o3d.geometry.PointCloud, path: str
    ) -> None:
        """Save entity point cloud and metadata to disk.

        Args:
            entity_id: Identifier for the entity
            metadata: Dictionary with entity metadata
            pcd: Open3D point cloud object
            path: Directory path to save files
        """
        os.makedirs(path, exist_ok=True)

        # Save point cloud
        if pcd is not None:
            o3d.io.write_point_cloud(os.path.join(path, f"{entity_id}.ply"), pcd)

        # Save metadata
        with open(os.path.join(path, f"{entity_id}.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, cls=_NumpyEncoder)

    @staticmethod
    def load_entity_with_pcd(
        entity_id: str, path: str, load_pcd: bool = True
    ) -> Tuple[Optional[o3d.geometry.PointCloud], Dict[str, Any]]:
        """Load entity point cloud and metadata from disk.

        Args:
            entity_id: Identifier for the entity
            path: Directory path to load files from
            load_pcd: Whether to load the point cloud

        Returns:
            Tuple of (point_cloud, metadata_dict)
        """
        pcd = None
        if load_pcd:
            pcd_path = os.path.join(path, f"{entity_id}.ply")
            if os.path.exists(pcd_path):
                pcd = o3d.io.read_point_cloud(pcd_path)

        metadata_path = os.path.join(path, f"{entity_id}.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return pcd, metadata

    @staticmethod
    def save_entity_metadata_only(entity_id: str, metadata: Dict[str, Any], path: str) -> None:
        """Save only entity metadata to disk (no point cloud).

        Args:
            entity_id: Identifier for the entity
            metadata: Dictionary with entity metadata
            path: Directory path to save files
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{entity_id}.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, cls=_NumpyEncoder)

    @staticmethod
    def load_entity_metadata_only(entity_id: str, path: str) -> Dict[str, Any]:
        """Load only entity metadata from disk.

        Args:
            entity_id: Identifier for the entity
            path: Directory path to load files from

        Returns:
            Metadata dictionary
        """
        with open(os.path.join(path, f"{entity_id}.json"), "r", encoding="utf-8") as f:
            return json.load(f)
