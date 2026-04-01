"""This file contains the class definition for the Object in HMSG."""

import numpy as np
from memory.hmsg.graph.persistence import EntityPersistence, PersistenceHandler


class Object(PersistenceHandler):
    """Class to represent an object in a room.

    Args:
        object_id: Unique identifier for the object
        room_id: Identifier of the room this object belongs to
        name: Name of the object (e.g., "Chair", "Table")
    """

    def __init__(self, object_id: str | int, room_id: str | int, name: str = None):
        """Initialize an Object entity.

        Args:
            object_id: Unique identifier for the object
            room_id: Identifier of the room this object belongs to
            name: Name of the object
        """
        self._object_id = object_id
        self._room_id = room_id
        self._name = name
        self._vertices = None
        self._embedding = None
        self._pcd = None
        self._gt_name = None
        self._best_view_id = None
        self._view_ids: list = []

    # Properties for better encapsulation (OCP: Open/Closed Principle)
    @property
    def object_id(self):
        return self._object_id

    @property
    def room_id(self):
        return self._room_id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = value

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        self._embedding = value

    @property
    def pcd(self):
        return self._pcd

    @pcd.setter
    def pcd(self, value):
        self._pcd = value

    @property
    def gt_name(self):
        return self._gt_name

    @gt_name.setter
    def gt_name(self, value: str):
        self._gt_name = value

    @property
    def best_view_id(self):
        return self._best_view_id

    @best_view_id.setter
    def best_view_id(self, value):
        self._best_view_id = value

    @property
    def view_ids(self):
        return self._view_ids

    def add_view(self, view_id: int | str) -> None:
        """Add a view ID to the object's list of views.

        Args:
            view_id: View identifier to add
        """
        if view_id not in self._view_ids:
            self._view_ids.append(view_id)

    def serialize(self) -> dict:
        """Serialize the object to a dictionary (SRP: Persistence handling).

        Returns:
            Dictionary with all object data
        """
        return {
            "object_id": self._object_id,
            "vertices": np.array(self._vertices).tolist() if self._vertices is not None else None,
            "room_id": self._room_id,
            "name": self._name,
            "embedding": self._embedding.tolist() if self._embedding is not None else None,
            "view_ids": self._view_ids,
            "best_view_id": self._best_view_id,
            "gt_name": self._gt_name,
        }

    def deserialize(self, data: dict) -> None:
        """Deserialize the object from a dictionary.

        Args:
            data: Dictionary with object data
        """
        self._vertices = np.asarray(data["vertices"]) if data.get("vertices") is not None else None
        self._name = data.get("name")
        self._embedding = (
            np.asarray(data["embedding"]) if data.get("embedding") is not None else None
        )
        self._view_ids = data.get("view_ids", [])
        self._best_view_id = data.get("best_view_id")
        self._gt_name = data.get("gt_name")

    def save(self, path: str) -> None:
        """Save object to disk with point cloud and metadata.

        Args:
            path: Directory path to save the object
        """
        EntityPersistence.save_entity_with_pcd(
            str(self._object_id), self.serialize(), self._pcd, path
        )

    def load(self, path: str, load_pcd: bool = True) -> None:
        """Load object from disk (unified method replaces load and load_new).

        Args:
            path: Directory path to load the object from
            load_pcd: Whether to load the point cloud (useful for metadata-only loading)
        """
        pcd, data = EntityPersistence.load_entity_with_pcd(str(self._object_id), path, load_pcd)
        if pcd is not None:
            self._pcd = pcd
        self.deserialize(data)

    def merge(self, other: "Object") -> "Object":
        """Merge this object with another object (combine point clouds and embeddings).

        Args:
            other: Object to merge with

        Returns:
            Self for chaining operations
        """
        if self._pcd is None or self._pcd.is_empty():
            self._pcd = other._pcd
        elif other._pcd is not None and not other._pcd.is_empty():
            self._pcd += other._pcd
            self._vertices = self._pcd.get_axis_aligned_bounding_box().get_box_points()

        # Merge embeddings by averaging
        if self._embedding is not None and other._embedding is not None:
            self._embedding = np.mean([self._embedding, other._embedding], axis=0)
        elif other._embedding is not None:
            self._embedding = other._embedding

        # Merge view IDs
        self._view_ids = list(set(self._view_ids + other._view_ids))

        return self

    def __add__(self, other: "Object") -> "Object":
        """Operator overload for merge using + operator.

        Args:
            other: Object to merge with

        Returns:
            Self after merging
        """
        return self.merge(other)

    def __str__(self) -> str:
        """String representation of the object."""
        return f"{self.__class__.__name__}(id={self._object_id}, name={self._name}, room={self._room_id})"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return self.__str__()
