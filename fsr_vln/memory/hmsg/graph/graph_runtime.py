"""GraphRuntime — facade that owns Graph + GraphBuilder + GraphRetriever.

This is the single external entry point.  Callers import and instantiate
``GraphRuntime``; they do *not* interact with the inner classes directly.

Backward-compatible public surface:
    • ``build(save_path)``  —  equivalent to old ``build_hier_multimodal_scene_graph``
    • ``load(path)``        —  creates Graph + Retriever from a saved graph
    • ``load_hmsg_graph(path)``  —  backward-compat alias for ``load``
    • ``query_hierarchy(...)``   —  delegates to retriever
    • ``generate_room_names(...)``
    • ``set_room_names(...)``
    • ``query_floor(...)``, ``query_room(...)``, ``query_object(...)``
    • Properties: ``floors``, ``rooms``, ``objects``, ``views``
    • ``curr_query_save_dir``  (get/set on retriever)
    • ``graph_path``  (from graph)
"""

import os
import shutil

import open_clip
import torch
from application.download_checkpoints import ensure_checkpoints
from memory.hmsg.dataloader.custom_dataset import CustomDataset
from memory.hmsg.graph.graph import Graph
from memory.hmsg.graph.graph_builder import GraphBuilder
from memory.hmsg.graph.graph_retriever import GraphRetriever
from memory.hmsg.utils.constants import CLIP_DIM
from memory.hmsg.utils.label_feats import CSV_LABEL_REGISTRY
from omegaconf import DictConfig

# pylint: disable=all

_MODEL_MAP = {
    "ViT-L/14": ("ViT-L-14", {}),
    "ViT-H-14": ("ViT-H-14", {}),
    "ViT-B-32": ("ViT-B-32", {"precision": "fp16"}),
    "MobileCLIP2-S4": ("MobileCLIP2-S4", {}),
}


class GraphRuntime:
    """Facade: owns Graph data, GraphBuilder, and GraphRetriever.

    Usage::

        # build mode
        service = GraphRuntime(cfg)
        service.build(save_dir)

        # query mode
        service = GraphRuntime(cfg)
        service.load(graph_dir)
        floor, rooms, objects, timing = service.query_hierarchy("Find the bed")
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._graph: Graph = Graph()
        self._builder: GraphBuilder = None
        self._engine: GraphRetriever = None

        self._load_clip_model()
        self._init_directories()
        self._load_dataset()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_clip_model(self) -> None:
        clip_type = self.cfg.models.clip.type
        checkpoint = str(self.cfg.models.clip.checkpoint)
        ensure_checkpoints([checkpoint])
        if clip_type not in _MODEL_MAP:
            raise ValueError(f"Unsupported CLIP model type: {clip_type}")
        model_name, extra_kwargs = _MODEL_MAP[clip_type]
        print(f"[SERVICE] Loading CLIP '{model_name}' …")
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
        dataset_cfg = {
            "root_dir": self.cfg.main.dataset_path,
            "transforms": None,
            "depth_cut": self.cfg.main.depth_cut,
        }
        self.dataset = CustomDataset(dataset_cfg)

    def _make_engine(self, graph_path: str = None) -> None:
        self._engine = GraphRetriever(
            graph=self._graph,
            cfg=self.cfg,
            clip_model=self.clip_model,
            preprocess=self.preprocess,
            clip_feat_dim=self.clip_feat_dim,
            device=self.device,
            dataset=self.dataset,
            graph_path=graph_path,
        )

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    def build(self, save_path: str = None) -> None:
        """Run the full build pipeline and save the resulting graph.

        Args:
            save_path: Root directory for all output artifacts.
                       Defaults to ``cfg.main.save_path``.
        """
        save_path = save_path or self.cfg.main.save_path
        self._builder = GraphBuilder(self.cfg)
        # GraphBuilder loads its own models; here we only need to run build()
        self._graph = self._builder.build(save_path)
        self._make_engine()

    def load(self, path: str) -> None:
        """Load a saved graph and prepare the query engine.

        Args:
            path: Path to the saved graph directory (contains graph.pkl etc.).
        """
        print(f"[SERVICE] Loading graph from {path} …")
        self._graph = Graph()
        self._graph.load_hmsg_graph(path)
        self._make_engine(graph_path=path)
        print(
            f"[SERVICE] Graph loaded — "
            f"floors={len(self._graph.floors)}, rooms={len(self._graph.rooms)}, "
            f"objects={len(self._graph.objects)}"
        )

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    def load_hmsg_graph(self, path: str) -> None:
        """Backward-compatible alias for ``load``."""
        self.load(path)

    def build_hier_multimodal_scene_graph(self, save_path: str = None) -> None:
        """Backward-compatible alias for ``build``."""
        self.build(save_path)

    def _init_vlm_client(self) -> None:
        """Delegate to engine (backward-compat)."""
        if self._engine is not None:
            self._engine._init_vlm_client()

    # ------------------------------------------------------------------
    # Query delegation
    # ------------------------------------------------------------------

    def query_hierarchy(self, *args, **kwargs):
        return self._engine.query_hierarchy(*args, **kwargs)

    def generate_room_names(self, *args, **kwargs):
        return self._engine.generate_room_names(*args, **kwargs)

    def set_room_names(self, *args, **kwargs):
        return self._engine.set_room_names(*args, **kwargs)

    def query_floor(self, *args, **kwargs):
        return self._engine.query_floor(*args, **kwargs)

    def query_room(self, *args, **kwargs):
        return self._engine.query_room(*args, **kwargs)

    def query_object(self, *args, **kwargs):
        return self._engine.query_object(*args, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def floors(self):
        return self._graph.floors

    @property
    def rooms(self):
        return self._graph.rooms

    @property
    def objects(self):
        return self._graph.objects

    @property
    def views(self):
        return self._graph.views

    @property
    def curr_query_save_dir(self) -> str:
        return self._engine.curr_query_save_dir if self._engine else ""

    @curr_query_save_dir.setter
    def curr_query_save_dir(self, value: str) -> None:
        if self._engine is not None:
            self._engine.curr_query_save_dir = value

    @property
    def graph_path(self) -> str:
        return getattr(self._graph, "graph_path", None)
