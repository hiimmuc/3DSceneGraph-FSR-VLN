"""Vision-language model utilities for image-based spatial reasoning.

Uses OWLv2 (Hugging Face) for grounded object detection — a more efficient
local alternative to remote LLM endpoints.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

_log = logging.getLogger(__name__)

# Default model for open-vocabulary object detection
_OWLv2_MODEL_ID = "google/owlv2-base-patch16-ensemble"


class VLMClient:
    """Grounding vision-language model client using OWLv2.

    Provides local, efficient image-grounded reasoning for object detection
    and frame selection without requiring external LLM endpoints.

    Usage::

        from memory.hmsg.tools.vlm import VLMClient

        vlm = VLMClient()

        queries = vlm.generate_object_queries("find a sofa in the living room")
        best_frame = vlm.choose_best_frame(image_paths, instruction)
        has_obj = vlm.detect_in_image(img_path, "chair")
    """

    def __init__(self, model_id: str = _OWLv2_MODEL_ID, device: str = "auto") -> None:
        """Initialize OWLv2 model for grounding object detection.

        Args:
            model_id: HuggingFace model identifier (default: google/owlv2-base-patch16-ensemble)
            device: Device to load model on ('auto', 'cpu', 'cuda', etc.)
        """
        self.model_id = model_id
        self.device = device
        _log.info(f"Loading OWLv2 model: {model_id} on device={device}")
        self.processor = Owlv2Processor.from_pretrained(model_id, local_files_only=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_id, device_map=device if device == "auto" else None, local_files_only=True
        )
        if device != "auto" and device != "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        _log.info("OWLv2 model loaded successfully")

    # ------------------------------------------------------------------
    # Core primitives
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> Image.Image:
        """Load image from local path or URL."""
        if path.startswith(("http://", "https://")):
            from io import BytesIO

            import requests

            response = requests.get(path)
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(path).convert("RGB")

    def float_score(self, img_path: str, query: str) -> float:
        """Compute confidence score [0, 1] for *query* in *img_path* using OWLv2.

        Returns 0.0 if no objects are detected or query is not found.
        """
        try:
            image = self._load_image(img_path)
            boxes, scores, _ = self.detect_objects(image, [query])
            if scores.numel() > 0:
                return float(scores[0].item())
            return 0.0
        except Exception as exc:
            _log.warning(f"float_score failed for {img_path}: {exc}")
            return 0.0

    def detect_objects(
        self, image: Image.Image, queries: List[str], threshold: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Run OWLv2 detection on an image.

        Args:
            image: PIL Image to analyze
            queries: List of text queries to search for
            threshold: Confidence threshold [0, 1]

        Returns:
            boxes: Bounding boxes [N, 4] in (x_min, y_min, x_max, y_max) format
            scores: Confidence scores [N]
            labels: Detected label names [N]
        """
        try:
            # Prepare queries for OWLv2 (as list of lists for batch processing)
            text_queries = [queries]

            inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(
                self.model.device
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process detections
            target_sizes = torch.tensor([(image.height, image.width)])
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
                text_labels=text_queries,
            )

            result = results[0]
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["text_labels"]

            _log.debug(f"Detected {len(labels)} objects in image: {list(zip(labels, scores))}")
            return boxes, scores, labels
        except Exception as exc:
            _log.error(f"detect_objects failed: {exc}")
            return torch.tensor([]), torch.tensor([]), []

    # ------------------------------------------------------------------
    # High-level reasoning operations
    # ------------------------------------------------------------------

    def generate_object_queries(self, instruction: str) -> List[str]:
        """Extract search phrases from a navigation instruction.

        Since we use local grounding VLM, we extract the core object query(ies)
        directly from the instruction. OWLv2 handles descriptive text well.

        Returns:
            List of search phrases (empty list if no object is identifiable).
        """
        # Remove common navigation prefixes to isolate the object query
        cleaned = instruction.strip()
        prefixes = [
            "find",
            "locate",
            "show",
            "bring",
            "get",
            "take",
            "navigate to",
            "go to",
            "where",
            "i need",
            "i'm looking for",
            "can you find",
            "please find",
            "show me",
            "find me",
            "look for",
        ]
        lower = cleaned.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                # Remove leading "me", "the", "a", "an", "the"
                for article in ["me", "the", "a", "an"]:
                    if cleaned.lower().startswith(article):
                        cleaned = cleaned[len(article) :].strip()
                break

        if not cleaned or len(cleaned) < 2:
            return []

        # For simple instructions like "sofa" or "chair in bedroom", return as-is
        # OWLv2 handles these descriptive queries directly
        queries = [cleaned]
        _log.debug(f"Extracted object query from '{instruction}' -> {queries}")
        return queries

    def choose_best_frame(self, image_paths: List[str], instruction: str) -> str:
        """Select the best frame from *image_paths* for the given *instruction*.

        Uses OWLv2 to detect the target object and ranks frames by detection
        confidence. Returns frame index wrapped in <frame_id> tag.

        Args:
            image_paths: List of image file paths
            instruction: Navigation instruction describing the target

        Returns:
            String like "<frame_id>0</frame_id>" with the best frame index.
        """
        queries = self.generate_object_queries(instruction)
        if not queries:
            _log.warning(f"No queries extracted from '{instruction}', using frame 0")
            return "<frame_id>0</frame_id>"

        query = queries[0]  # Use primary query
        scores = []

        for i, img_path in enumerate(image_paths):
            score = self.float_score(img_path, query)
            scores.append(score)
            _log.debug(f"Frame {i}: {img_path} -> score={score:.3f} for '{query}'")

        best_idx = int(np.argmax(scores)) if scores else 0
        _log.info(f"Selected frame {best_idx} (score={scores[best_idx]:.3f}) for query '{query}'")
        return f"<frame_id>{best_idx}</frame_id>"

    def detect_and_select(
        self, img_list: List[str], query: str, score_threshold: float = 0.5
    ) -> Tuple[List[bool], Optional[str]]:
        """Detect *query* in each image using OWLv2 and return ``(results, best_image_path)``.

        Uses OWLv2 detection with a single confidence threshold. Images with
        scores >= score_threshold are marked as containing the object.

        Args:
            img_list: List of image file paths
            query: Object query string
            score_threshold: Confidence threshold [0, 1]

        Returns:
            results: Per-image boolean detection flags.
            best_image_path: Path of the highest-scoring image, or None if
                             no image passed detection.
        """
        results, scores = [], []
        for img_path in img_list:
            score = self.float_score(img_path, query)
            has_object = score >= score_threshold
            results.append(has_object)
            scores.append(score)

            _log.debug(
                f"[VLM] Image: {img_path} → score={score:.3f}, "
                f"has_object={has_object}, query={query}"
            )

        best_image = img_list[int(np.argmax(scores))] if any(results) else None
        _log.debug(
            f"detect_and_select: found {sum(results)} objects out of {len(img_list)}, "
            f"best={best_image}"
        )
        return results, best_image

    def detect_in_image(self, img_path: str, query: str, score_threshold: float = 0.3) -> bool:
        """Return True if *query* object is present in *img_path* with sufficient confidence.

        Uses OWLv2 grounding detection.

        Args:
            img_path: Path to image file
            query: Object query string
            score_threshold: Confidence threshold [0, 1]

        Returns:
            True if object is detected with confidence >= score_threshold.
        """
        score = self.float_score(img_path, query)
        has_object = score >= score_threshold

        _log.debug(f"[VLM] {img_path} → score={score:.3f}, has_object={has_object}, query={query}")
        return has_object
