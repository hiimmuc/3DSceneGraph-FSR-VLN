"""Floor class to represent a floor in a HMSG (Hierarchical Multi-Floor Scene Graph)."""

import numpy as np
import open3d as o3d
from memory.hmsg.graph.persistence import EntityPersistence, PersistenceHandler


class Floor(PersistenceHandler):
    """Class to represent a floor in a building (Single Responsibility: Floor entity data).

    Args:
        floor_id: Unique identifier for the floor
        name: Name of the floor (e.g., "First", "Second")
    """

    def __init__(self, floor_id: str | int, name: str = None):
        """Initialize a Floor entity.

        Args:
            floor_id: Unique identifier for the floor
            name: Name of the floor
        """
        self._floor_id = floor_id
        self._name = name
        self._rooms: list = []
        self._txt_embeddings: list = []
        self._pcd: o3d.geometry.PointCloud = None
        self._vertices = np.array([])
        self._floor_height: float = None
        self._floor_zero_level: float = None

    # Properties for encapsulation (OCP: Open/Closed Principle)
    @property
    def floor_id(self):
        return self._floor_id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def rooms(self):
        return self._rooms

    @property
    def txt_embeddings(self):
        return self._txt_embeddings

    @property
    def pcd(self):
        return self._pcd

    @pcd.setter
    def pcd(self, value):
        self._pcd = value

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = value

    @property
    def floor_height(self):
        return self._floor_height

    @floor_height.setter
    def floor_height(self, value: float):
        self._floor_height = value

    @property
    def floor_zero_level(self):
        return self._floor_zero_level

    @floor_zero_level.setter
    def floor_zero_level(self, value: float):
        self._floor_zero_level = value

    def add_room(self, room) -> None:
        """Add a room to the floor.

        Args:
            room: Room to add
        """
        self._rooms.append(room)

    def serialize(self) -> dict:
        """Serialize the floor to a dictionary (SRP: Persistence handling).

        Returns:
            Dictionary with all floor data
        """
        return {
            "floor_id": self._floor_id,
            "name": self._name,
            "rooms": [room.room_id for room in self._rooms],
            "vertices": (
                self._vertices.tolist()
                if isinstance(self._vertices, np.ndarray)
                else self._vertices
            ),
            "floor_height": self._floor_height,
            "floor_zero_level": self._floor_zero_level,
            "txt_embeddings": [e.tolist() for e in self._txt_embeddings],
        }

    def deserialize(self, data: dict) -> None:
        """Deserialize the floor from a dictionary.

        Args:
            data: Dictionary with floor data
        """
        self._name = data.get("name")
        self._vertices = np.asarray(data.get("vertices", []))
        self._floor_height = data.get("floor_height")
        self._floor_zero_level = data.get("floor_zero_level")
        self._txt_embeddings = [np.asarray(e) for e in data.get("txt_embeddings", [])]

    def save(self, path: str) -> None:
        """Save the floor to disk with point cloud and metadata.

        Args:
            path: Directory path to save the floor
        """
        EntityPersistence.save_entity_with_pcd(
            str(self._floor_id), self.serialize(), self._pcd, path
        )

    def load(self, path: str, load_pcd: bool = True) -> None:
        """Load floor from disk (unified method replaces duplicate load methods).

        Args:
            path: Directory path to load the floor from
            load_pcd: Whether to load the point cloud
        """
        pcd, data = EntityPersistence.load_entity_with_pcd(str(self._floor_id), path, load_pcd)
        if pcd is not None:
            self._pcd = pcd
        self.deserialize(data)

    def __str__(self) -> str:
        """String representation of the floor."""
        return (
            f"{self.__class__.__name__}("
            f"id={self._floor_id}, name={self._name}, "
            f"rooms={len(self._rooms)})"
        )

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return self.__str__()
