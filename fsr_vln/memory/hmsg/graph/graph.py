"""Data container for the HMSG graph.
"""

import os

# ---------------------------------------------------------------------------
# Re-exported enums (kept here so existing imports don't break)
# ---------------------------------------------------------------------------
from enum import Enum
from typing import List, Optional, Tuple

import faiss
import networkx as nx
import numpy as np
import open3d as o3d
import torch
from memory.hmsg.graph.floor import Floor
from memory.hmsg.graph.object import Object
from memory.hmsg.graph.room import Room
from memory.hmsg.graph.view import View


class ObjectQueryMethod(str, Enum):
    CLIP = "clip"


class RoomQueryMethod(str, Enum):
    LABEL = "label"
    VIEW_EMBEDDING = "view_embedding"


class RoomNameMethod(str, Enum):
    LABEL = "label"
    OBJ_EMBEDDING = "obj_embedding"
    VIEW_EMBEDDING = "view_embedding"


# pylint: disable=all


class Graph:
    """Pure data container for the Hierarchical Multi-modal Scene Graph (HMSG).

    Holds the topology (floors / rooms / objects / views / graph) and
    persistence helpers only.  All construction logic lives in
    ``GraphBuilder`` and all query/reasoning logic lives in
    ``GraphRetriever``.  Use ``GraphRuntime`` as the single external entry
    point.
    """

    def __init__(self) -> None:
        self._init_state()

    def _init_state(self) -> None:
        """Reset all graph data containers."""
        self.full_pcd = o3d.geometry.PointCloud()
        self.mask_feats: list = []
        self.mask_pcds: list = []
        self.objects: List[Object] = []
        self.rooms: List[Room] = []
        self.floors: List[Floor] = []
        self.views: List[View] = []
        self.full_feats_array: list = []
        self.graph = nx.Graph()
        self.graph.add_node(0, name="building", type="building")
        self.room_masks: dict = {}
        self._object_emb_matrix: Optional[np.ndarray] = None
        self._faiss_index = None

    # ------------------------------------------------------------------
    # Basic lookups
    # ------------------------------------------------------------------

    def _find_view_by_id(self, view_id) -> Optional[View]:
        """Return the first View whose view_id matches, or None."""
        return next((v for v in self.views if v.view_id == view_id), None)

    def find_view_by_imgpath(self, img_path: str) -> Tuple[Optional[View], Optional[int]]:
        """Return (View, img_id) for the given image path, or (None, None)."""
        for view in self.views:
            if view.img_path == img_path:
                return view, view.img_id
        return None, None

    def find_object_by_object_id(self, object_id: int) -> Optional[Object]:
        """Return the Object with the given object_id, or None."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_hmsg_graph(self, path: str) -> None:
        """Serialize graph to *path* (creates sub-dirs as needed)."""
        for sub in ("", "floors", "rooms", "objects", "views"):
            os.makedirs(os.path.join(path, sub), exist_ok=True)
        for topo_obj, _ in self.graph.nodes(data=True):
            if isinstance(topo_obj, Floor):
                topo_obj.save(os.path.join(path, "floors"))
            elif isinstance(topo_obj, Room):
                topo_obj.save(os.path.join(path, "rooms"))
            elif isinstance(topo_obj, Object):
                topo_obj.save(os.path.join(path, "objects"))
            elif isinstance(topo_obj, View):
                topo_obj.save(os.path.join(path, "views"))

    def load_hmsg_graph(self, path: str) -> None:
        """Deserialize graph from *path* and rebuild FAISS index."""
        print(f"[LOAD] Loading graph from {path} …")
        self.graph_path = path

        # floors
        floor_files = sorted(
            f.split(".")[0]
            for f in os.listdir(os.path.join(path, "floors"))
            if f.endswith(".ply")
        )
        for ff in floor_files:
            floor = Floor(str(ff), name="floor_" + str(ff))
            floor.load(os.path.join(path, "floors"))
            self.floors.append(floor)
            self.graph.add_node(floor, name="floor_" + str(ff), type="floor")
            self.graph.add_edge(0, floor)
        print(f"  ├── floors: {len(self.floors)}")

        # rooms
        room_files = sorted(
            f.split(".")[0]
            for f in os.listdir(os.path.join(path, "rooms"))
            if f.endswith(".ply")
        )
        for rf in room_files:
            room = Room(str(rf), rf.split("_")[0])
            room.load(os.path.join(path, "rooms"))
            self.rooms.append(room)
            self.graph.add_node(room, name="room_" + str(rf), type="room")
            self.graph.add_edge(self.floors[int(rf.split("_")[0])], room)
            self.floors[int(room.floor_id)].rooms.append(room)
        print(f"  ├── rooms: {len(self.rooms)}")

        # objects
        object_files = sorted(
            f.split(".")[0]
            for f in os.listdir(os.path.join(path, "objects"))
            if f.endswith(".ply")
        )
        for of in object_files:
            room_id = "_".join(of.split("_")[:2])
            parent_room = next((r for r in self.rooms if r.room_id == room_id), None)
            assert parent_room is not None, f"Couldn't find room {room_id}"
            obj = Object(str(of), room_id, name="object_" + str(of))
            obj.load(os.path.join(path, "objects"))
            self.objects.append(obj)
            self.graph.add_node(obj, name="object_" + str(of), type="object")
            self.graph.add_edge(parent_room, obj)
            parent_room.add_object(obj)
        print(f"  ├── objects: {len(self.objects)}")

        # views
        for vf in sorted(os.listdir(os.path.join(path, "views"))):
            vf = vf.split(".")[0]
            room_id = "_".join(vf.split("_")[:2])
            parent_room = next((r for r in self.rooms if r.room_id == room_id), None)
            assert parent_room is not None, f"Couldn't find room {room_id}"
            view_node = View(str(vf), room_id, img_id=None, name="view_" + str(vf))
            view_node.load(os.path.join(path, "views"))
            self.views.append(view_node)
            self.graph.add_node(view_node, name="view_" + str(vf), type="view")
            self.graph.add_edge(parent_room, view_node)
        print(f"  └── views: {len(self.views)}")

        self._rebuild_object_index()

    def _rebuild_object_index(self) -> None:
        """Pre-stack object embeddings and build a FAISS inner-product index.

        Infers the feature dimension from the stored embeddings so no external
        ``clip_feat_dim`` parameter is required.
        """
        if not self.objects:
            self._object_emb_matrix = np.empty((0, 0), dtype=np.float32)
            self._faiss_index = None
            return

        raw = [obj.embedding for obj in self.objects]
        embs = np.array(raw, dtype=np.float32)
        if embs.ndim == 3:
            embs = embs.mean(axis=1)
        faiss.normalize_L2(embs)
        self._object_emb_matrix = embs

        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        self._faiss_index = index
        print(f"[LOAD] FAISS index built: {index.ntotal} objects, dim={embs.shape[1]}")

    # ------------------------------------------------------------------
    # Debug / maintenance I/O helpers
    # ------------------------------------------------------------------

    def load_full_pcd(self, path: str):
        """Load full point cloud from *path* (debug helper)."""
        pcd_file = os.path.join(path, "full_pcd.ply")
        if not os.path.exists(pcd_file):
            print(f"[LOAD] full_pcd.ply not found in {path}")
            return None
        self.full_pcd = o3d.io.read_point_cloud(pcd_file)
        print(
            f"[LOAD] Full PCD loaded: {np.asarray(self.full_pcd.points).shape}"
        )
        return self.full_pcd

    def load_full_pcd_feats(
        self, path: str, full_feats: bool = False, normalize: bool = True
    ):
        """Load per-point CLIP feature vectors (debug helper)."""
        if not os.path.exists(path):
            print(f"[LOAD] feature dir not found: {path}")
            return None
        if full_feats:
            arr = torch.load(os.path.join(path, "full_feats.pt")).float()
            if normalize:
                arr = torch.nn.functional.normalize(arr, p=2, dim=-1)
            self.full_feats_array = arr.cpu().numpy()
            print(f"[LOAD] full_feats loaded: {self.full_feats_array.shape}")
            return self.full_feats_array
        else:
            arr = torch.load(os.path.join(path, "mask_feats.pt")).float()
            if normalize:
                arr = torch.nn.functional.normalize(arr, p=2, dim=-1)
            self.mask_feats = arr.cpu().numpy()
            print(f"[LOAD] mask_feats loaded: {self.mask_feats.shape}")
            return self.mask_feats

    def load_masked_pcds(self, path: str):
        """Load segmented object point clouds (debug helper)."""
        if len(self.mask_feats) == 0:
            print("[LOAD] load mask_feats first (load_full_pcd_feats)")
            return None
        obj_dir = os.path.join(path, "objects")
        if not os.path.exists(obj_dir):
            print(f"[LOAD] masked PCDs not found in {path}")
            return None
        self.mask_pcds = []
        n = len(os.listdir(obj_dir))
        not_found = []
        for i in range(n):
            p = os.path.join(obj_dir, f"pcd_{i}.ply")
            if os.path.exists(p):
                self.mask_pcds.append(o3d.io.read_point_cloud(p))
            else:
                print(f"[LOAD] pcd_{i}.ply missing")
                not_found.append(i)
        # align mask_feats length
        not_found = [i for i in not_found if i < len(self.mask_feats)]
        if not_found:
            self.mask_feats = np.delete(self.mask_feats, not_found, axis=0)
        print(
            f"[LOAD] mask PCDs: {len(self.mask_pcds)}, mask_feats: {len(self.mask_feats)}"
        )
        return self.mask_pcds
