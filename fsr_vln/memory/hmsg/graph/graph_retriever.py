"""GraphRetriever — query/reasoning layer over an HMSG Graph.

Wraps a populated ``Graph`` and all CLIP/LLM/VLM resources needed for
hierarchy queries, room-name generation, and VLM-guided refinement."""

import json
import os
import re
import time
from copy import deepcopy
from typing import List, Optional, Tuple

import cv2
import faiss
import numpy as np
from memory.hmsg.graph.graph import Graph
from memory.hmsg.graph.object import Object
from memory.hmsg.graph.room import Room
from memory.hmsg.tools.llm import (
    QueryParser,
    infer_floor_id_from_query,
    parse_hierarchy_query,
)
from memory.hmsg.tools.vlm import VLMClient
from memory.hmsg.utils.clip_utils import get_text_feats_multiple_templates
from memory.hmsg.utils.graph_utils import check_object_in_view, visualize_pcd_on_image
from omegaconf import DictConfig

# pylint: disable=all


class GraphRetriever:
    """All query and reasoning operations over a pre-built Graph.

    Args:
        graph:         A fully loaded/built ``Graph`` data container.
        cfg:           Hydra/OmegaConf config.
        clip_model:    Pre-loaded CLIP model (eval mode).
        preprocess:    CLIP preprocessing transform.
        clip_feat_dim: CLIP embedding dimensionality.
        device:        Torch device string.
        dataset:       Loaded ``CustomDataset`` (may be ``None`` in query-only mode).
        graph_path:    Path where the graph was loaded/saved from.
    """

    def __init__(
        self,
        graph: Graph,
        cfg: DictConfig,
        clip_model,
        preprocess,
        clip_feat_dim: int,
        device: str,
        dataset,
        graph_path: Optional[str] = None,
    ) -> None:
        self._graph = graph
        self.cfg = cfg
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.clip_feat_dim = clip_feat_dim
        self.device = device
        self.dataset = dataset
        self.graph_path = graph_path or getattr(cfg.main, "graph_path", None)

        self.vln_result_dir = os.path.join(cfg.main.save_path, "vln_result_presentation")
        os.makedirs(self.vln_result_dir, exist_ok=True)
        self.curr_query_save_dir = self.vln_result_dir

        self.vlm: Optional[VLMClient] = None
        self._query_parser: Optional[QueryParser] = None

        # Lazy-initialise the VLM client if slow reasoning is enabled in the config
        if getattr(cfg.main, "slow_reasoning", False):
            self._init_vlm_client()

        # Pre-create the QueryParser singleton so the LLM client is
        # initialized once and reused across all queries.
        self._query_parser = QueryParser()

    # ------------------------------------------------------------------
    # Lazy / optional initialization
    # ------------------------------------------------------------------

    def _init_vlm_client(self) -> None:
        """Lazy-initialise the VLM client (OWLv2-based local model)."""
        device = getattr(self.cfg.main, "vlm_device", "auto")
        model_id = getattr(self.cfg.main, "vlm_model_id", None)
        self.vlm = VLMClient(**({} if model_id is None else {"model_id": model_id}), device=device)

    # ------------------------------------------------------------------
    # Private query helpers
    # ------------------------------------------------------------------

    def _build_query_list(self, query: str, negative_prompt: List[str]) -> Tuple[List[str], int]:
        """Return (category_list, query_index) with negatives appended after the target."""
        if query in negative_prompt:
            return negative_prompt, negative_prompt.index(query)
        return [query, *negative_prompt], 0

    def _find_room_indices_by_label(self, rooms_list: List[Room], room_query: str) -> List[int]:
        """Return indices into *rooms_list* whose label best matches *room_query*."""
        for room in rooms_list:
            assert (
                room.name is not None
            ), "Room name not generated — call generate_room_names() first"
        room_names = [room.name for room in rooms_list]
        room_embs = get_text_feats_multiple_templates(
            room_names, self.clip_model, self.clip_feat_dim
        )
        query_feats = get_text_feats_multiple_templates(
            [room_query], self.clip_model, self.clip_feat_dim
        )
        similarity = np.dot(query_feats, room_embs.T)[0]
        top_index = np.argsort(similarity)[::-1]
        for i in top_index[:3]:
            mark = " <--" if i == top_index[0] else ""
            print(
                f"            {rooms_list[i].room_id:<14} '{rooms_list[i].name}'  sim={similarity[i]:.4f}{mark}"
            )
        tar_sim = similarity[top_index[0]]
        same = {rooms_list[i].room_id for i in top_index if abs(similarity[i] - tar_sim) < 1e-3}
        return [i for i, r in enumerate(rooms_list) if r.room_id in same]

    def _save_timing_json(self, query_time_consumer: dict) -> None:
        save_json_path = os.path.join(self.curr_query_save_dir, "query_time_consumer.json")
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(query_time_consumer, f, ensure_ascii=False, indent=4)

    # ------------------------------------------------------------------
    # Room / object name generation
    # ------------------------------------------------------------------

    def set_room_names(self, room_names: List[str]) -> None:
        """Set semantic names for rooms."""
        assert len(room_names) == len(
            self._graph.rooms
        ), "The length of room_names should be the same as the number of rooms in the graph"
        for i in range(len(self._graph.rooms)):
            self._graph.rooms[i].name = room_names[i]
            vertices = self._graph.rooms[i].vertices
            center = np.mean(vertices, axis=0)
            self._graph.rooms[i].room_center_pos = center

    def generate_room_names(
        self,
        generate_method: str = "label",
        default_room_types: List[str] = None,
    ) -> None:
        """Generate semantic names for rooms using CLIP embedding."""
        for i in range(len(self._graph.rooms)):
            if generate_method in ["obj_embedding", "view_embedding"]:
                assert (
                    default_room_types is not None
                ), "You should provide a list of default room types"
            rooms: List[Room] = self._graph.rooms
            if generate_method in ["obj_embedding", "label"]:
                rooms[i].infer_room_type_from_objects(
                    infer_method=generate_method,
                    default_room_types=default_room_types,
                    clip_model=self.clip_model,
                    clip_feat_dim=self.clip_feat_dim,
                )
            elif generate_method in ["view_embedding"]:
                rooms[i].infer_room_type_from_view_embedding(
                    default_room_types, self.clip_model, self.clip_feat_dim
                )
            else:
                raise NotImplementedError(f"Unknown generate_method: {generate_method}")

    # ------------------------------------------------------------------
    # Core query methods
    # ------------------------------------------------------------------

    def query_floor(self, query: str, query_method: str = "clip") -> int:
        """Find best matching floor for *query*.

        Returns:
            Best floor index.
        """
        zero_levels_list = [x.floor_zero_level for x in self._graph.floors]
        zero_level_order_ids = np.argsort(zero_levels_list)

        try:
            return zero_level_order_ids[int(query) - 1]
        except Exception:
            if query_method == "clip":
                text_feats = get_text_feats_multiple_templates(
                    [query], self.clip_model, self.clip_feat_dim
                )
                floor_names = ["floor " + str(i) for i in range(len(self._graph.floors))]
                floor_embs = get_text_feats_multiple_templates(
                    floor_names, self.clip_model, self.clip_feat_dim
                )
                sim_mat = np.dot(text_feats, floor_embs.T)
                top_index = np.argsort(sim_mat[0])[::-1][0]
                return zero_level_order_ids[top_index]
            elif query_method == "vlm":
                floor_ids_list = [i + 1 for i in range(len(self._graph.floors))]
                floor_id = infer_floor_id_from_query(floor_ids_list, query)
                return zero_level_order_ids[floor_id - 1]

    def query_room(
        self,
        query: str,
        floor_id: int = -1,
        query_method: str = "view_embedding",
        top_k: int = 3,
    ) -> List[int]:
        """Retrieve rooms matching *query*.

        Returns:
            List of room indices.
        """
        is_room_text_valid = query is not None and query != "" and "unknown" not in query.lower()
        rooms_list: List[Room] = (
            self._graph.floors[floor_id].rooms if floor_id != -1 else self._graph.rooms
        )

        if query_method == "label" and is_room_text_valid:
            return self._find_room_indices_by_label(rooms_list, query)

        print(f"            method=view_embedding  query={query!r}")
        query_text_feats = get_text_feats_multiple_templates(
            [query], self.clip_model, self.clip_feat_dim
        )
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

    def query_object(
        self,
        query: str,
        floor_id: int = -1,
        room_ids: List[int] = [],
        query_method: str = "clip",
        top_k: int = 1,
        negative_prompt: List[str] = [],
        return_scores: bool = False,
    ) -> Tuple[List[int], List[int]]:
        """Retrieve objects matching *query* within spatial constraints.

        Returns:
            (object_ids, room_ids[, scores_if_requested])
        """
        query, query_id = self._build_query_list(query, list(negative_prompt))

        query_text_feats = get_text_feats_multiple_templates(
            query, self.clip_model, self.clip_feat_dim
        )

        # Build default room→object mapping
        room_id_by_obj_idx = []
        for obj in self._graph.objects:
            for i, room in enumerate(self._graph.rooms):
                if obj.room_id == room.room_id:
                    room_id_by_obj_idx.append(i)
                    break

        objects_list: List[Object] = self._graph.objects
        room_ids_list: List[int] = room_id_by_obj_idx

        if len(room_ids) != 0:
            objects_list = []
            room_ids_list = []
            for i in room_ids:
                src_rooms = (
                    self._graph.floors[floor_id].rooms if floor_id != -1 else self._graph.rooms
                )
                objects_list.extend(src_rooms[i].objects)
                room_ids_list.extend([i] * len(src_rooms[i].objects))

        if query_method == "clip":
            is_filtered = len(room_ids) != 0
            n_pool = len(objects_list)
            _path = (
                "FAISS"
                if not is_filtered and getattr(self._graph, "_faiss_index", None) is not None
                else "brute-force"
            )
            print(f"  [4] Object   query={query}  query_id={query_id}")
            print(f"            path={_path}  pool={n_pool} objects")

            if not is_filtered and getattr(self._graph, "_faiss_index", None) is not None:
                # Fast path: FAISS index over all objects.
                # Always fetch a wide candidate set so cls filter is meaningful.
                q = query_text_feats.astype(np.float32)
                faiss.normalize_L2(q)
                fetch_k = min(max(top_k * 20, 128), self._graph._faiss_index.ntotal)
                all_scores_mat, all_indices_mat = self._graph._faiss_index.search(q, fetch_k)
                sim_mat_full = np.full(
                    (len(query_text_feats), len(self._graph.objects)), -1.0, dtype=np.float32
                )
                for qi in range(len(query_text_feats)):
                    sim_mat_full[qi, all_indices_mat[qi]] = all_scores_mat[qi]
                cls_ids = np.argmax(sim_mat_full, axis=0)
                max_scores = np.max(sim_mat_full, axis=0)

                sampled = cls_ids[all_indices_mat[query_id]]
                _cls_str = " ".join(str(x) for x in sampled[:20]) + (
                    "..." if len(sampled) > 20 else ""
                )
                obj_ids = np.where(cls_ids == query_id)[0]
                print(f"            cls_ids  : [{_cls_str}]")
                print(f"            matched  : {len(obj_ids)}/{len(self._graph.objects)}")

                if len(obj_ids) == 0:
                    print(f"            -> '{query[query_id]}' not found in graph")
                    if return_scores:
                        return [], [], []
                    return [], []

                resort_ids = np.argsort(-max_scores[obj_ids])
                top_index = obj_ids[resort_ids][:top_k]
                top_scores = max_scores[top_index]

                for i in top_index:
                    score = float(self._graph._object_emb_matrix[i] @ query_text_feats[query_id])
                    print(
                        f"            -> {self._graph.objects[i].object_id:<12} "
                        f" {self._graph.objects[i].name:<20}  score={score:.4f}"
                    )

                target_object_id = [self._graph.objects[i].object_id for i in top_index]
                object_id_map = {obj.object_id: idx for idx, obj in enumerate(self._graph.objects)}
                target_id = [object_id_map[oid] for oid in target_object_id]
                target_room_id = [room_id_by_obj_idx[i] for i in top_index]
                target_scores = list(top_scores)
            else:
                # Filtered / brute-force path
                object_embs = (
                    self._graph._object_emb_matrix[
                        [self._graph.objects.index(o) for o in objects_list], :
                    ]
                    if (
                        not is_filtered
                        and getattr(self._graph, "_object_emb_matrix", None) is not None
                    )
                    else np.array([obj.embedding for obj in objects_list], dtype=np.float32)
                )
                sim_mat = np.dot(query_text_feats, object_embs.T)
                cls_ids = np.argmax(sim_mat, axis=0)
                max_scores_arr = np.max(sim_mat, axis=0)
                obj_ids = np.where(cls_ids == query_id)[0]

                _cls_str = " ".join(str(x) for x in cls_ids[:20]) + (
                    "..." if len(cls_ids) > 20 else ""
                )
                print(f"            cls_ids  : [{_cls_str}]")
                print(f"            matched  : {len(obj_ids)}/{n_pool}")

                if len(obj_ids) == 0:
                    print(f"            -> '{query[query_id]}' not found in graph")
                    if return_scores:
                        return [], [], []
                    return [], []

                resort_ids = np.argsort(-max_scores_arr[obj_ids])
                top_index = obj_ids[resort_ids][:top_k]

                for i in top_index:
                    print(
                        f"            -> {objects_list[i].object_id:<12} "
                        f" {objects_list[i].name:<20}  score={sim_mat[query_id][i]:.4f}"
                    )

                target_object_id = [objects_list[i].object_id for i in top_index]
                object_id_map = {obj.object_id: idx for idx, obj in enumerate(self._graph.objects)}
                target_id = [object_id_map[oid] for oid in target_object_id]
                target_room_id = [room_ids_list[i] for i in top_index]
                target_scores = [float(sim_mat[query_id][i]) for i in top_index]

            if return_scores:
                return target_id, target_room_id, target_scores
            return target_id, target_room_id
        raise NotImplementedError(f"Unsupported query_method: {query_method}")

    def query_room_obj_slow_reasoning(
        self,
        instruction: str,
        room_query: str,
        object_query: str,
        negative_prompt,
        floor_id: int = -1,
        room_query_method: str = "label",
        object_query_method: str = "clip",
        update_flag: bool = True,
        top_k: int = 5,
        llm_enable: bool = True,
    ) -> Tuple[dict, List[int], List[int]]:
        """Query rooms and objects with VLM-guided reasoning.

        Returns:
            (timing_dict, object_ids, room_ids)
        """
        # Ensure VLM client is available
        if self.vlm is None:
            raise RuntimeError("VLM client not initialized — cannot perform slow reasoning query")

        query_time_consumer: dict = {
            "room_query": room_query,
            "object_query": object_query,
            "negative_prompt": negative_prompt,
        }
        is_detect_room = bool(room_query) and "unknown" not in room_query.lower()

        rooms_list = self._graph.rooms if floor_id == -1 else self._graph.floors[floor_id].rooms
        start_time = time.time()
        if room_query_method == "label" and is_detect_room:
            target_ids = self._find_room_indices_by_label(rooms_list, room_query)
        else:
            room_text = room_query if room_query else instruction
            query_room_text_feats = get_text_feats_multiple_templates(
                [room_text], self.clip_model, self.clip_feat_dim
            )
            room2query_sim = {}
            for room in rooms_list:
                embeddings = np.stack(room.embeddings)
                sims = np.dot(query_room_text_feats, embeddings.T)
                room2query_sim[room.room_id] = float(sims[0, np.argmax(sims)])
            room2query_sim_sorted = {
                int(k.split("_")[-1]): v
                for k, v in sorted(room2query_sim.items(), key=lambda x: x[1], reverse=True)
            }
            target_ids = list(room2query_sim_sorted.keys())[: min(len(room2query_sim_sorted), 10)]

        room_retrieval_time = time.time() - start_time
        print(f"  [3] Rooms    {target_ids if target_ids else '─ (no filter)'}")
        query_time_consumer["room_retrieval_by_clip"] = room_retrieval_time

        if object_query in negative_prompt:
            query_id = negative_prompt.index(object_query)
            object_query = list(negative_prompt)
        else:
            object_query = [object_query, *negative_prompt]
            query_id = 0

        query_object_text_feats = get_text_feats_multiple_templates(
            object_query, self.clip_model, self.clip_feat_dim
        )

        room_ids_list = []
        for obj in self._graph.objects:
            for i, room in enumerate(rooms_list):
                if obj.room_id == room.room_id:
                    room_ids_list.append(i)
                    break

        if object_query_method == "clip":
            if len(target_ids) != 0:
                objects_list: List[Object] = []
                room_ids_list = []
                for i in target_ids:
                    objects_list.extend(rooms_list[i].objects)
                    room_ids_list.extend([i] * len(rooms_list[i].objects))
            else:
                objects_list = self._graph.objects

            n_pool = len(objects_list)
            print(f"  [4] Object   query={object_query}  query_id={query_id}")
            print(f"            path=brute-force  pool={n_pool} objects")
            if len(target_ids) == 0 and self._graph._object_emb_matrix is not None:
                object_embs = self._graph._object_emb_matrix
            else:
                object_embs = np.array([obj.embedding for obj in objects_list], dtype=np.float32)
            sim_mat = np.dot(query_object_text_feats, object_embs.T)
            cls_ids = np.argmax(sim_mat, axis=0)
            max_scores = np.max(sim_mat, axis=0)
            obj_ids = np.where(cls_ids == query_id)[0]
            _cls_str = " ".join(str(x) for x in cls_ids[:20]) + (
                "..." if len(cls_ids) > 20 else ""
            )
            print(f"            cls_ids  : [{_cls_str}]")
            print(f"            matched  : {len(obj_ids)}/{n_pool}")

            top_index = np.argsort(sim_mat[query_id])[::-1][:top_k]
            if len(obj_ids) > 0:
                resort_ids = np.argsort(-max_scores[obj_ids])
                top_index = obj_ids[resort_ids][:top_k]

            if len(obj_ids) == 0:
                fast_matching_time = time.time() - start_time
                query_time_consumer["fast_matching_time"] = fast_matching_time
                self._save_timing_json(query_time_consumer)
                return (
                    {
                        "FastMatching": fast_matching_time,
                        "ObjectInImageCheck": 0.0,
                        "VLM_Rethinking": 0.0,
                        "Re_Matching": 0.0,
                        "Total_Time": fast_matching_time,
                        "object_scores": [],
                    },
                    [],
                    [],
                )

            for i in top_index:
                print(
                    f"            -> {objects_list[i].object_id:<12} "
                    f" {objects_list[i].name:<20}  score={sim_mat[query_id][i]:.4f}"
                )

            target_object_id = [objects_list[i].object_id for i in top_index]
            target_room_id = [room_ids_list[i] for i in top_index]
            target_scores = [float(sim_mat[query_id][i]) for i in top_index]
            target_id = []
            for ti in target_object_id:
                target_id.append(
                    [i for i, x in enumerate(self._graph.objects) if x.object_id == ti][0]
                )
            fast_matching_time = time.time() - start_time
            query_time_consumer["fast_matching_time"] = fast_matching_time

        best_object = self._graph.objects[target_id[0]]
        best_view = self._graph._find_view_by_id(best_object.best_view_id)

        if best_view is None or not llm_enable:
            total_online_query_time = fast_matching_time
            query_time_consumer["total_query_time"] = f"{total_online_query_time:.4f} seconds"
            self._save_timing_json(query_time_consumer)
            res_dict = {
                "FastMatching": fast_matching_time,
                "ObjectInImageCheck": 0.0,
                "VLM_Rethinking": 0.0,
                "Re_Matching": 0.0,
                "Total_Time": total_online_query_time,
                "object_scores": target_scores,
            }
            return res_dict, target_id, target_room_id

        best_view_image_path = best_view.img_path
        best_view_img_id = best_view.img_id
        print("[QUERY] online_best_view_image_path:", best_view_image_path)
        goal_image_path_online = best_view_image_path
        query_time_consumer["top1_image_path_online_object_best_view"] = goal_image_path_online

        start_time = time.time()
        object_detected = self.vlm.detect_in_image(best_view_image_path, object_query[query_id])
        object_in_goal_view_check_time = time.time() - start_time
        query_time_consumer["Object_in_goal_view_check_time"] = (
            f"{object_in_goal_view_check_time:.4f} seconds"
        )
        query_time_consumer["Object_in_goal_view_check_res"] = object_detected

        if object_detected:
            total_online_query_time = fast_matching_time + object_in_goal_view_check_time
            query_time_consumer["total_query_time"] = f"{total_online_query_time:.4f} seconds"
            self._save_timing_json(query_time_consumer)
            return (
                {
                    "FastMatching": fast_matching_time,
                    "ObjectInImageCheck": object_in_goal_view_check_time,
                    "VLM_Rethinking": 0.0,
                    "Re_Matching": 0.0,
                    "Total_Time": total_online_query_time,
                    "object_scores": target_scores,
                },
                target_id,
                target_room_id,
            )

        # === VLM refinement path ===
        total_online_query_time = fast_matching_time + object_in_goal_view_check_time
        if self.dataset is None:
            res_dict = {
                "FastMatching": fast_matching_time,
                "ObjectInImageCheck": object_in_goal_view_check_time,
                "VLM_Rethinking": 0.0,
                "Re_Matching": 0.0,
                "Total_Time": total_online_query_time,
                "note": "dataset not loaded; skipping VLM refinement",
            }
            return res_dict, target_id, target_room_id

        all_image_indices = []
        all_image_embedding = []
        for room in rooms_list:
            img_ids = room.sample_images
            embs = room.clip_embeddings
            assert len(img_ids) == len(embs)
            all_image_indices.extend(img_ids)
            all_image_embedding.extend(embs)
        print("[QUERY] all_image_indices:", len(all_image_indices))

        vlm_check_time = 0.0
        vlm_refine_time = 0.0
        avg_distance_in_vlmview = -1.0
        goal_image_path_by_clip = goal_image_path_online
        goal_image_path_by_vlm = goal_image_path_online

        for room_id in target_ids[:1]:
            print(f"[QUERY] ├── searching goal image in room {room_id}")
            start_time = time.time()
            global_embedding = np.stack(all_image_embedding)
            sims = np.dot(query_object_text_feats[0], global_embedding.T)
            clip_max_idx = np.argmax(sims)
            top_k_view = min(24, sims.shape[0])
            top_idx = np.argsort(sims)[-top_k_view:][::-1]

            old_path = self.dataset.frameId2imgPath[all_image_indices[clip_max_idx]]
            filename = os.path.basename(old_path)
            goal_image_path_by_clip = os.path.join(self.dataset.root_dir, "images", filename)
            print(f"[QUERY] goal_image_path_by_clip: {goal_image_path_by_clip}")
            query_time_consumer[f"goal_image_retrieval_by_clip_{room_id}"] = (
                time.time() - start_time
            )
            query_time_consumer["goal_image_path_by_clip"] = goal_image_path_by_clip

            start_time = time.time()
            room_clip_refined_topk_image_local_paths = [
                self.dataset.frameId2imgPath[all_image_indices[idx]] for idx in top_idx
            ]
            response = self.vlm.choose_best_frame(
                room_clip_refined_topk_image_local_paths, instruction
            )
            print(response)
            match = re.findall(r"\d+", response)
            if match:
                goal_img_path = room_clip_refined_topk_image_local_paths[int(match[0])]
            else:
                print("[QUERY] No frame id found in VLM response.")
                goal_img_path = None
            goal_image_path_by_vlm = goal_img_path
            query_time_consumer[f"goal_image_retrieval_by_vlm_{room_id}"] = (
                time.time() - start_time
            )
            query_time_consumer["goal_image_path_by_vlm"] = goal_image_path_by_vlm

            select_imgs = [goal_image_path_online, goal_image_path_by_clip, goal_image_path_by_vlm]
            vlm_check_start_time = time.time()
            vlm_check_result, best_image_path = self.vlm.detect_and_select(
                select_imgs, object_query[query_id]
            )
            if best_image_path is None:
                print("[QUERY] Warning: VLM could not find a best image. Skipping refinement.")
                res_dict = {
                    "FastMatching": fast_matching_time,
                    "ObjectInImageCheck": object_in_goal_view_check_time,
                    "VLM_Rethinking": 0.0,
                    "Re_Matching": 0.0,
                    "Total_Time": total_online_query_time,
                    "error": "VLM failed or no best image found",
                }
                return res_dict, target_id, target_room_id

            vlm_check_time = time.time() - vlm_check_start_time
            query_time_consumer["vlm_check_time"] = vlm_check_time
            print("[QUERY] Detection results:", vlm_check_result)
            print("[QUERY] Best image:", best_image_path)
            query_time_consumer["detection_results"] = vlm_check_result
            query_time_consumer["best_image"] = best_image_path

            vlm_refine_time_start = time.time()
            if (
                vlm_check_result[0] is False
                and update_flag
                and best_image_path != goal_image_path_online
                and best_image_path is not None
            ):
                print("[QUERY] └── performing VLM refinement …")
                objs_embedding_in_view = []
                vlm_refine_best_view, vlm_refine_best_view_img_id = (
                    self._graph.find_view_by_imgpath(best_image_path)
                )
                if vlm_refine_best_view_img_id is None:
                    res_dict = {
                        "error": "vlm_refine_best_view_img_id is None",
                        "FastMatching": fast_matching_time,
                        "ObjectInImageCheck": object_in_goal_view_check_time,
                        "VLM_Rethinking": 0.0,
                        "Re_Matching": 0.0,
                        "Total_Time": total_online_query_time,
                    }
                    return res_dict, target_id, target_room_id

                object_ids_in_view = vlm_refine_best_view.object_ids
                for object_id in object_ids_in_view:
                    object_target = self._graph.find_object_by_object_id(object_id)
                    assert object_target is not None
                    objs_embedding_in_view.append(object_target.embedding)
                if len(objs_embedding_in_view) > 0:
                    objs_embedding_in_view = np.stack(objs_embedding_in_view)
                    obj_sims = np.dot(query_object_text_feats[0], objs_embedding_in_view.T)
                    max_obj_idx = int(np.argmax(obj_sims))
                    max_sim_object_id = object_ids_in_view[max_obj_idx]
                    final_object = self._graph.find_object_by_object_id(max_sim_object_id)
                    final_obj_pcd = final_object.pcd
                    camera_matrix = self.dataset.get_camera_intrinsics()
                    img, _, pose, _, _ = self.dataset[vlm_refine_best_view_img_id]
                    if not isinstance(img, np.ndarray):
                        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
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
                    new_objects_path = os.path.join(self._graph.graph_path, "objects_update")
                    os.makedirs(new_objects_path, exist_ok=True)
                    final_object.save(new_objects_path)
                    self._graph._rebuild_object_index()
            vlm_refine_time = time.time() - vlm_refine_time_start
            query_time_consumer["vlm_refine_time"] = vlm_refine_time

            # Depth for online best view
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
            self.visualize_goal_images(
                mean_depth_online,
                goal_image_path_online,
                goal_image_path_by_clip,
                goal_image_path_by_vlm,
                save_name=f"goal_compare_room_{room_id}.png",
            )

        total_query_time_offline = total_online_query_time + vlm_check_time + vlm_refine_time
        query_time_consumer["total_query_time"] = f"{total_query_time_offline:.4f} seconds"
        query_time_consumer["online_object_distance_in_online_view"] = mean_depth_online
        query_time_consumer["vlmref_object_distance_in_offline_view"] = avg_distance_in_vlmview
        self._save_timing_json(query_time_consumer)
        return (
            {
                "FastMatching": fast_matching_time,
                "ObjectInImageCheck": object_in_goal_view_check_time,
                "VLM_Rethinking": vlm_check_time,
                "Re_Matching": vlm_refine_time,
                "Total_Time": total_query_time_offline,
                "object_scores": target_scores,
            },
            target_id,
            target_room_id,
        )

    def visualize_goal_images(
        self,
        mean_depth: np.ndarray,
        goal_image_path_online: str,
        goal_image_path_by_clip: str,
        goal_image_path_by_vlm: str,
        save_name: str = "goal_compare.png",
    ) -> None:
        """Composite online / CLIP / VLM goal-image candidates side-by-side."""
        img_online = cv2.imread(goal_image_path_online)
        img_vlm_best = cv2.imread(goal_image_path_by_clip)
        img_vlm = cv2.imread(goal_image_path_by_vlm)
        if img_vlm_best is None or img_vlm is None or img_online is None:
            raise FileNotFoundError("One of the image paths is invalid")
        for img in (img_vlm_best, img_vlm, img_online):
            img = cv2.resize(img, (640, 480))
        img_vlm_best = cv2.resize(img_vlm_best, (640, 480))
        img_vlm = cv2.resize(img_vlm, (640, 480))
        img_online = cv2.resize(img_online, (640, 480))
        font, fs, th, color = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2, (0, 255, 0)
        cv2.putText(img_vlm_best, "BEST", (10, 30), font, fs, color, th, cv2.LINE_AA)
        cv2.putText(img_vlm, "VLM", (10, 30), font, fs, color, th, cv2.LINE_AA)
        cv2.putText(img_online, "ObjBestView", (10, 30), font, fs, color, th, cv2.LINE_AA)
        cv2.putText(img_online, f"{mean_depth:.2f}", (10, 300), font, fs, color, th, cv2.LINE_AA)
        combined = np.hstack((img_online, img_vlm_best, img_vlm))
        save_path = os.path.join(self.curr_query_save_dir, save_name)
        cv2.imwrite(save_path, combined)
        cv2.imshow("Goal Image Comparison", combined)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Result display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_object_tree(
        objects_list: List["Object"],
        scores: List[float],
        top_index: List[int],
    ) -> None:
        """Print query results as a floor → room → object tree.

        Args:
            objects_list: Full list of graph objects (indexed by top_index).
            scores:       Score for every entry in objects_list.
            top_index:    Indices into objects_list to display.
        """
        from collections import defaultdict

        if not top_index:
            print("[RESULT] No matching objects found.")
            return

        # Group selected objects by floor then room.
        tree: dict = defaultdict(lambda: defaultdict(list))
        for enum_i, idx in enumerate(top_index):
            obj = objects_list[idx]
            parts = str(obj.object_id).split("_")
            floor_key = f"floor_{parts[0]}" if len(parts) >= 1 else "floor_?"
            room_key = str(obj.room_id)
            tree[floor_key][room_key].append(
                (obj.object_id, obj.name, scores[enum_i] if enum_i < len(scores) else 0.0)
            )

        for floor_key in sorted(tree):
            print(floor_key)
            rooms = sorted(tree[floor_key].items())
            for r_idx, (room_key, objs) in enumerate(rooms):
                is_last_room = r_idx == len(rooms) - 1
                room_connector = "└────" if is_last_room else "├────"
                obj_indent = "       " if is_last_room else "│      "
                print(f"     {room_connector} room {room_key}")
                for o_idx, (obj_id, obj_name, score) in enumerate(objs):
                    is_last_obj = o_idx == len(objs) - 1
                    obj_connector = "└──────" if is_last_obj else "├──────"
                    print(
                        f"     {obj_indent} {obj_connector} Object {obj_id}: {obj_name} [{score:.4f}]"
                    )

    # ------------------------------------------------------------------
    # Top-level hierarchical query
    # ------------------------------------------------------------------

    def query_hierarchy(
        self,
        query_instruction: str,
        top_k: int = 1,
        slow_reasoning: bool = False,
        llm_enable: bool = True,
    ) -> Tuple:
        """Hierarchical query: floor → room → objects.

        Returns:
            (floor, rooms, objects, timing_dict)
        """
        negative_labels = ["background"]
        start_time = time.time()
        llm_parse_time = 0.0

        _W = 62
        mode_str = "slow reasoning (VLM)" if slow_reasoning else "fast (CLIP-only)"
        print(f"\n{'─'*_W}")
        print(f"  HIERARCHY QUERY  |  {query_instruction!r}  |  {mode_str}")
        print(f"{'─'*_W}")

        if llm_enable:
            floor_query, room_query, object_query = parse_hierarchy_query(
                self.cfg,
                query_instruction,
                parser=getattr(self, "_query_parser", None),
            )
            llm_parse_time = time.time() - start_time
        else:
            floor_query, room_query, object_query = None, None, query_instruction

        print(
            f"  [1] Parse    floor={floor_query or 'N/A'}  "
            f"room={room_query or 'N/A'}  object={object_query or 'N/A'}  "
            f"({llm_parse_time*1000:.1f} ms)"
        )

        if room_query and "Exhibition" in room_query:
            negative_labels = ["wall"]

        floor_id = self.query_floor(floor_query) if floor_query is not None else -1
        floor_str = f"floor_{floor_id}" if floor_id != -1 else "all"  # for display only

        is_detect_room = bool(room_query) and "unknown" not in room_query.lower()
        is_detect_obj = bool(object_query) and "unknown" not in object_query.lower()
        print(
            f"  [2] Scope    floor={floor_str}  "
            f"detect_room={is_detect_room}  detect_obj={is_detect_obj}"
        )

        if slow_reasoning:
            res_dict, object_ids, room_ids = self.query_room_obj_slow_reasoning(
                query_instruction,
                room_query,
                object_query,
                negative_prompt=negative_labels,
                floor_id=floor_id,
                room_query_method="label",
                object_query_method="clip",
                update_flag=True,
                top_k=top_k,
                llm_enable=llm_enable,
            )
            res_dict["LLM_Parse_Time"] = llm_parse_time
            res_dict["room_query"] = room_query
            res_dict["object_query"] = object_query
            res_dict["negative_labels"] = negative_labels
            object_scores = res_dict.get("object_scores", [])
            _elapsed = time.time() - start_time
            print(f"\n{'─'*_W}")
            print(f"  RESULT  ({_elapsed*1000:.1f} ms total)")
            print(f"{'─'*_W}")
            self.print_object_tree(self._graph.objects, object_scores, object_ids)
            print(f"{'─'*_W}\n")
            return (
                self._graph.floors[floor_id] if floor_id != -1 else None,
                (
                    [self._graph.floors[floor_id].rooms[k] for k in room_ids]
                    if floor_id != -1
                    else [self._graph.rooms[k] for k in room_ids]
                ),
                [self._graph.objects[i] for i in object_ids],
                res_dict,
            )

        room_ids = (
            self.query_room(room_query, floor_id=floor_id, query_method="label")
            if room_query is not None
            else []
        )
        print(f"  [3] Rooms    {room_ids if room_ids else '─ (no filter)'}")

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

        res_dict = {
            "room_query": room_query,
            "object_query": object_query,
            "negative_labels": negative_labels,
            "LLM_Parse_Time": llm_parse_time,
            "FastMatching": 0.0,
            "ObjectInImageCheck": 0.0,
            "VLM_Rethinking": 0.0,
            "Re_Matching": 0.0,
            "Total_Time": 0.0,
            "object_scores": object_scores,
        }
        _elapsed = time.time() - start_time
        print(f"\n{'─'*_W}")
        print(f"  RESULT  ({_elapsed*1000:.1f} ms total)")
        print(f"{'─'*_W}")
        self.print_object_tree(self._graph.objects, object_scores, object_ids)
        print(f"{'─'*_W}\n")
        return (
            self._graph.floors[floor_id] if floor_id != -1 else None,
            (
                [self._graph.floors[floor_id].rooms[k] for k in room_ids]
                if floor_id != -1
                else [self._graph.rooms[k] for k in room_ids]
            ),
            [self._graph.objects[i] for i in object_ids],
            res_dict,
        )

    # Backward-compatible alias
    query_hierarchy_protected_icra = query_hierarchy
