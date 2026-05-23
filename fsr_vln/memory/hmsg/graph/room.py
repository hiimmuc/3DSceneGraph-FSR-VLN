"""Room class to represent a room in a HMSG (Hierarchical Multi-Floor Scene Graph)."""

from collections import defaultdict
from typing import Any, List, Union

import numpy as np
import open3d as o3d
from memory.hmsg.graph.persistence import EntityPersistence, PersistenceHandler
from memory.hmsg.utils.clip_utils import get_text_feats_multiple_templates
from memory.hmsg.utils.graph_utils import (
    feats_denoise_dbscan,
    find_overlapping_ratio_faiss,
)


class Room(PersistenceHandler):
    """Class to represent a room in a building (Single Responsibility: Room entity data).

    Args:
        room_id: Unique identifier for the room
        floor_id: Identifier of the floor this room belongs to
        name: Name of the room (e.g., "Living Room", "Bedroom")
    """

    def __init__(self, room_id: Union[str, int], floor_id: Union[str, int], name: str = None):
        """Initialize a Room entity.

        Args:
            room_id: Unique identifier for the room
            floor_id: Identifier of the floor this room belongs to
            name: Name of the room
        """
        self._room_id = room_id
        self._floor_id = floor_id
        self._name = name
        self._category = None
        self._objects: list = []
        self._vertices = np.array([])
        self._embeddings: list = []
        self._pcd: o3d.geometry.PointCloud = None
        self._room_height: float = None
        self._room_zero_level: float = None
        self._represent_images: list = []
        self._sample_images: list = []
        self._clip_embeddings: list = []
        self._views: list = []
        self._room_center_pos: tuple = (0, 0, 0)
        self.object_counter: int = 0

    # Properties for encapsulation (OCP: Open/Closed Principle)
    @property
    def room_id(self):
        return self._room_id

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
    def category(self):
        return self._category

    @category.setter
    def category(self, value: str):
        self._category = value

    @property
    def objects(self):
        return self._objects

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = value

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value

    @property
    def pcd(self):
        return self._pcd

    @pcd.setter
    def pcd(self, value):
        self._pcd = value

    @property
    def room_height(self):
        return self._room_height

    @room_height.setter
    def room_height(self, value: float):
        self._room_height = value

    @property
    def room_zero_level(self):
        return self._room_zero_level

    @room_zero_level.setter
    def room_zero_level(self, value: float):
        self._room_zero_level = value

    @property
    def represent_images(self):
        return self._represent_images

    @represent_images.setter
    def represent_images(self, value):
        self._represent_images = value

    @property
    def sample_images(self):
        return self._sample_images

    @sample_images.setter
    def sample_images(self, value):
        self._sample_images = value

    @property
    def clip_embeddings(self):
        return self._clip_embeddings

    @clip_embeddings.setter
    def clip_embeddings(self, value):
        self._clip_embeddings = value

    @property
    def views(self):
        return self._views

    @property
    def room_center_pos(self):
        return self._room_center_pos

    @room_center_pos.setter
    def room_center_pos(self, value: Union[tuple, list]):
        self._room_center_pos = tuple(value)

    def add_object(self, obj) -> None:
        """Add an object to the room.

        Args:
            obj: Object to add
        """
        self._objects.append(obj)

    def add_view(self, view) -> None:
        """Add a view to the room.

        Args:
            view: View to add
        """
        self._views.append(view)

    def set_text_embeddings(self, text: str, clip_model: Any, clip_feat_dim: int) -> None:
        """Set text embeddings for the room using CLIP model.

        Args:
            text: Text describing the room
            clip_model: CLIP model instance
            clip_feat_dim: CLIP feature dimension
        """
        self._embeddings.append(
            get_text_feats_multiple_templates(
                text, clip_model=clip_model, clip_feat_dim=clip_feat_dim
            )
        )

    def merge_objects(self, overlap_threshold: float = 0.01, radius: float = 0.1) -> None:
        """Merge objects that overlap and have the same name (SRP: Extracted from monolithic logic).

        Args:
            overlap_threshold: Minimum overlap ratio to consider objects for merging
            radius: Radius for overlap calculation
        """
        merger = ObjectMerger(self._objects, self._room_id)
        self._objects = merger.merge(overlap_threshold, radius)

    def infer_room_type_from_view_embedding(
        self,
        default_room_types: List[str],
        clip_model: Any,
        clip_feat_dim: int,
    ) -> str:
        """
        Use the embeddings stored inside the room to infer room type. We should
        already save k views CLIP embeddings for each room. We match the k
        embeddings with room types' textual CLIP embeddings to get a room label
        for each of the k views. Then we count which room type has the most
        votes and return that.

        Args:
            default_room_types (List[str]): the output room type should only be a room type from the list.
            clip_model (Any): when the generate_method is set to "embedding", a clip model needs to be
                              provided to the method.
            clip_feat_dim (int): when the generate_method is set to "embedding", the clip features dimension
                                 needs to be provided to this method

        Returns:
            str: a room type from the default_room_types list
        """
        if len(self.embeddings) == 0:
            print("empty embeddings")
            return "unknown room type"
        text_feats = get_text_feats_multiple_templates(
            default_room_types, clip_model, clip_feat_dim
        )
        embeddings = np.array(self.embeddings)
        sim_mat = np.dot(embeddings, text_feats.T)
        col_ids = np.argmax(sim_mat, axis=1)
        unique, counts = np.unique(col_ids, return_counts=True)
        unique_id = np.argmax(counts)
        type_id = unique[unique_id]
        self.name = default_room_types[type_id]
        print(f"The room type is {default_room_types[type_id]}")
        return default_room_types[type_id]

    def infer_room_type_from_room_name(
        self,
        infer_method: str = "name",
        default_room_types: List[str] = None,
        clip_model: Any = None,
        clip_feat_dim: int = None,
    ) -> str:
        """Use the room.name to infer a room type.
        Args:
            infer_method (str): "llm" if we want to directly use the pre-computed object names in the children nodes.
                                "name" if we want to use the name of the room node's children to infer the
                                room type.
            default_room_types (List[str] = None): the output room type should only be a room type from the list.
            clip_model (Any): when the generate_method is set to "embedding", a clip model needs to be
                              provided to the method.
            clip_feat_dim (int): when the generate_method is set to "embedding", the clip features dimension
                                 needs to be provided to this method

        Returns:
            str: room type name
        Func Descriptions:
            "name" : infer room type from the name of objects in the room
            "llm" : use LLM to compare room_text with default_room_types text similarity to infer room type
        """
        from memory.hmsg.utils.llm_utils import infer_room_type_from_objects

        # use similarity of object text feature and room text feature
        if infer_method == "llm":
            objects_list = []
            for obj_i, obj in enumerate(self.objects):
                if not any(
                    substring in obj.name.lower()
                    for substring in [
                        "wall",
                        "floor",
                        "ceiling",
                        "railing",
                        "roof",
                        "void",
                        "unlabeled",
                        "misc",
                    ]
                ):
                    objects_list.append(obj.name)
            room_type = infer_room_type_from_objects(
                objects_list, candidate_room_types=default_room_types
            )
            self.name = room_type

        # use similarity of object feature embedding and room text feature
        if infer_method == "name":
            assert (
                default_room_types
            ), "default_room_types can not be None if infer_method is 'embedding'"
            represent_feat = get_text_feats_multiple_templates(
                self.name, clip_model, clip_feat_dim
            )
            text_feats = get_text_feats_multiple_templates(
                default_room_types, clip_model, clip_feat_dim
            )
            sim_mat = np.dot(represent_feat, text_feats.T)
            col_id = np.argmax(sim_mat)
            self.name = default_room_types[col_id]
        print("room_id, name: ", self.room_id, self.name)

    def infer_room_type_from_objects(
        self,
        infer_method: str = "label",
        default_room_types: List[str] = None,
        clip_model: Any = None,
        clip_feat_dim: int = None,
    ) -> str:
        """
        Use the objects contained in the room to infer a room type. We want to
        ask GPT what kind of room it is from the names for the objects
        contained in the room.

        Args:
            infer_method (str): "label" if we want to directly use the pre-computed object names in the children nodes.
                                "obj_embedding" if we want to use the embedding of the room node's children to infer the
                                room type. default_room_types can not be None if infer_method is "embedding".
            default_room_types (List[str] = None): the output room type should only be a room type from the list.
            clip_model (Any): when the generate_method is set to "embedding", a clip model needs to be
                              provided to the method.
            clip_feat_dim (int): when the generate_method is set to "embedding", the clip features dimension
                                 needs to be provided to this method

        Returns:
            str: room type name
        Func Descriptions:
            "label" : infer room type from the name of objects in the room
            "obj_embedding" : infer room type from the embeddings of objects in the room
        """
        from memory.hmsg.utils.llm_utils import infer_room_type_from_objects

        # use similarity of object text feature and room text feature
        if infer_method == "label":
            objects_list = []
            for obj_i, obj in enumerate(self.objects):
                if not any(
                    substring in obj.name.lower()
                    for substring in [
                        "wall",
                        "floor",
                        "ceiling",
                        "railing",
                        "roof",
                        "void",
                        "unlabeled",
                        "misc",
                    ]
                ):
                    objects_list.append(obj.name)
            room_type = infer_room_type_from_objects(
                objects_list, candidate_room_types=default_room_types
            )
            self.name = room_type

        # use similarity of object feature embedding and room text feature
        if infer_method == "obj_embedding":
            assert (
                default_room_types
            ), "default_room_types can not be None if infer_method is 'embedding'"
            object_embs = []
            for obj_i, obj in enumerate(self.objects):
                object_embs.append(obj.embedding)

            represent_feat = feats_denoise_dbscan(object_embs).reshape((1, -1))
            text_feats = get_text_feats_multiple_templates(
                default_room_types, clip_model, clip_feat_dim
            )
            sim_mat = np.dot(represent_feat, text_feats.T)
            col_id = np.argmax(sim_mat)
            self.name = default_room_types[col_id]
        print("room_id, name: ", self.room_id, self.name)

    def save(self, path: str) -> None:
        """Save the room to disk with point cloud and metadata.

        Args:
            path: Directory path to save the room
        """
        EntityPersistence.save_entity_with_pcd(
            str(self._room_id), self.serialize(), self._pcd, path
        )

    def load(self, path: str, load_pcd: bool = True) -> None:
        """Load room from disk (unified method replaces load and load_new).

        Args:
            path: Directory path to load the room from
            load_pcd: Whether to load the point cloud
        """
        pcd, data = EntityPersistence.load_entity_with_pcd(str(self._room_id), path, load_pcd)
        if pcd is not None:
            self._pcd = pcd
        self.deserialize(data)

    def serialize(self) -> dict:
        """Serialize the room to a dictionary (SRP: Persistence handling).

        Returns:
            Dictionary with all room data
        """
        return {
            "room_id": self._room_id,
            "name": self._name,
            "floor_id": self._floor_id,
            "category": self._category,
            "objects": [obj.room_id for obj in self._objects],
            "views": [v.view_id for v in self._views],
            "vertices": (
                self._vertices.tolist()
                if isinstance(self._vertices, np.ndarray)
                else self._vertices
            ),
            "room_height": self._room_height,
            "room_zero_level": self._room_zero_level,
            "embeddings": [i.tolist() for i in self._embeddings],
            "represent_images": self._represent_images,
            "sample_images": self._sample_images,
            "clip_embeddings": [i.tolist() for i in self._clip_embeddings],
            "room_center_pos": self._room_center_pos,
        }

    def deserialize(self, data: dict) -> None:
        """Deserialize the room from a dictionary.

        Args:
            data: Dictionary with room data
        """
        self._name = data.get("name")
        self._floor_id = data.get("floor_id", self._floor_id)
        self._category = data.get("category")
        self._vertices = np.asarray(data.get("vertices", []))
        self._room_height = data.get("room_height")
        self._room_zero_level = data.get("room_zero_level")
        self._embeddings = [np.asarray(i) for i in data.get("embeddings", [])]
        self._represent_images = data.get("represent_images", [])
        self._sample_images = data.get("sample_images", [])
        self._clip_embeddings = [np.asarray(i) for i in data.get("clip_embeddings", [])]
        self._room_center_pos = tuple(data.get("room_center_pos", (0, 0, 0)))

    def __str__(self) -> str:
        """String representation of the room."""
        return (
            f"{self.__class__.__name__}("
            f"id={self._room_id}, name={self._name}, floor={self._floor_id}, "
            f"objects={len(self._objects)}, views={len(self._views)})"
        )

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return self.__str__()


class ObjectMerger:
    """Helper class to merge objects based on overlap criteria (SRP: Merging logic responsibility)."""

    def __init__(self, objects: List, room_id: Union[str, int]):
        """Initialize ObjectMerger.

        Args:
            objects: List of objects to merge
            room_id: Room ID for renumbering merged objects
        """
        self.objects = objects
        self.room_id = room_id

    def merge(self, overlap_threshold: float = 0.01, radius: float = 0.1) -> List:
        """Merge overlapping objects with the same name.

        Args:
            overlap_threshold: Minimum overlap ratio to trigger merge
            radius: Radius for overlap calculation

        Returns:
            List of merged objects
        """
        # Calculate overlap scores between similar objects
        overlap_scores = self._calculate_overlap_scores(overlap_threshold, radius)

        # Group objects by overlap relationships
        merge_groups = self._build_merge_groups(overlap_scores)

        # Perform actual merging and re-indexing
        merged_objects = self._perform_merging(merge_groups)

        return merged_objects

    def _calculate_overlap_scores(self, overlap_threshold: float, radius: float) -> np.ndarray:
        """Calculate overlap scores between all object pairs.

        Args:
            overlap_threshold: Minimum overlap ratio
            radius: Radius for overlap calculation

        Returns:
            Matrix of overlap scores
        """
        overlap_scores = np.zeros((len(self.objects), len(self.objects)))

        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i >= j or obj1.name != obj2.name:
                    continue

                overlap = find_overlapping_ratio_faiss(obj1.pcd, obj2.pcd, radius)
                if overlap > overlap_threshold:
                    overlap_scores[i, j] = overlap
                    overlap_scores[j, i] = overlap

        return overlap_scores

    def _build_merge_groups(self, overlap_scores: np.ndarray) -> dict:
        """Build groups of objects that should be merged together.

        Args:
            overlap_scores: Matrix of overlap scores

        Returns:
            Dictionary mapping object indices to merge groups
        """
        merge_groups = defaultdict(list)
        merge_indices = set()

        i_indices, j_indices = np.where(overlap_scores > 0)

        for i, j in zip(i_indices, j_indices):
            merge_indices.add(i)
            merge_indices.add(j)

            if i not in merge_groups and j not in merge_groups:
                merge_groups[i].append(j)
            elif i in merge_groups:
                merge_groups[i].append(j)
            elif j in merge_groups:
                merge_groups[j].append(i)

        # Add non-merging objects
        for idx in range(len(self.objects)):
            if idx not in merge_indices:
                merge_groups[idx].append(idx)

        return merge_groups

    def _perform_merging(self, merge_groups: dict) -> List:
        """Perform the actual object merging and re-indexing.

        Args:
            merge_groups: Dictionary of merge groups

        Returns:
            List of merged objects with updated IDs
        """
        merged_objects = []
        object_index = 0

        for i, indices_to_merge in merge_groups.items():
            unique_indices = list(set(indices_to_merge))

            # Merge all objects in the group
            merged_obj = self.objects[i]
            for idx in unique_indices:
                if idx != i:
                    merged_obj = merged_obj + self.objects[idx]

            # Update object ID
            merged_obj._object_id = f"{self.room_id}_{object_index}"
            merged_objects.append(merged_obj)
            object_index += 1

        return merged_objects
