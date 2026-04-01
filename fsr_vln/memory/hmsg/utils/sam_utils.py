"""SAM mask and bounding box utilities. Provides mask filtering, cropping, and overlay operations."""

from typing import Dict, List, Tuple

import cv2
import numpy as np

# Type hints for SAM mask dictionaries
MaskDict = Dict[str, any]


class MaskProcessor:
    """Single responsibility: mask filtering and manipulation."""

    @staticmethod
    def filter_nested_masks(masks: List[MaskDict]) -> List[MaskDict]:
        """Remove masks that are contained within other masks, keeping only the largest.

        Args:
            masks: List of SAM mask dictionaries with 'segmentation' and 'area' keys.

        Returns:
            Filtered list of masks, sorted by area descending.
        """
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                mask_i = masks[i]["segmentation"]
                mask_j = masks[j]["segmentation"]
                overlap = np.logical_and(mask_i, mask_j)

                # If j is completely contained in i, remove j from i
                if np.all(overlap == mask_j):
                    masks[i]["segmentation"] = np.logical_xor(mask_i, mask_j)
                    masks[i]["area"] = np.sum(masks[i]["segmentation"])

        # Remove empty masks
        return [m for m in masks if np.sum(m["segmentation"]) > 0]

    @staticmethod
    def overlay_on_image(
        image: np.ndarray, mask: np.ndarray, alpha: float = 0.35, invert: bool = False
    ) -> np.ndarray:
        """Overlay a binary mask on top of an image.

        Args:
            image: Input image (H × W × 3).
            mask: Binary segmentation mask (H × W).
            alpha: Opacity of the mask overlay [0, 1].
            invert: If True, invert the mask before overlaying.

        Returns:
            Image with mask overlaid.
        """
        mask_3d = np.dstack([mask, mask, mask])
        if invert:
            mask_3d = np.invert(mask_3d)
        overlay = np.dstack([image, mask_3d * alpha])
        return overlay.astype(np.uint8)


class BBoxProcessor:
    """Single responsibility: bounding box operations."""

    @staticmethod
    def expand_bbox(
        bbox: Tuple[float, float, float, float], margin: int
    ) -> Tuple[int, int, int, int]:
        """Expand bounding box by margin while clamping to non-negative coordinates.

        Args:
            bbox: (x, y, width, height) in XYWH format.
            margin: Pixels to expand in each direction.

        Returns:
            Expanded bbox as (x, y, w, h) with x, y ≥ 0.
        """
        x, y, w, h = bbox
        x = max(0, x - margin)
        y = max(0, y - margin)
        w += 2 * margin
        h += 2 * margin
        return (int(x), int(y), int(w), int(h))

    @staticmethod
    def draw_bbox_on_image(
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        margin: int = 0,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw a bounding box on an image.

        Args:
            image: Input image.
            bbox: Bounding box in XYWH format.
            margin: Expansion margin in pixels.
            color: RGB color tuple.
            thickness: Line thickness in pixels.

        Returns:
            Image with bounding box drawn.
        """
        result = image.copy()
        x, y, w, h = BBoxProcessor.expand_bbox(bbox, margin)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        return result

    @staticmethod
    def crop_by_bbox(
        image: np.ndarray, bbox: Tuple[float, float, float, float], margin: int = 0
    ) -> np.ndarray:
        """Crop image to bounding box region with optional margin expansion.

        Args:
            image: Input image.
            bbox: Bounding box in XYWH format.
            margin: Expansion margin in pixels.

        Returns:
            Cropped image region.
        """
        x, y, w, h = BBoxProcessor.expand_bbox(bbox, margin)
        return image[y : y + h, x : x + w]

    @staticmethod
    def crop_by_mask(
        image: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[float, float, float, float],
        margin: int = 0,
    ) -> np.ndarray:
        """Crop image to bbox after masking with the given segmentation mask.

        Args:
            image: Input image.
            mask: Binary segmentation mask.
            bbox: Bounding box in XYWH format.
            margin: Expansion margin in pixels.

        Returns:
            Masked and cropped image.
        """
        masked = image * np.expand_dims(mask, axis=-1)
        return BBoxProcessor.crop_by_bbox(masked, bbox, margin)


def draw_bboxes_on_masks(
    image: np.ndarray,
    masks: List[MaskDict],
    iou_threshold: float = 0.89,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    bbox_margin: int = 0,
) -> np.ndarray:
    """Draw bounding boxes on image for masks above IoU threshold.

    Args:
        image: Input image.
        masks: List of SAM mask dictionaries.
        iou_threshold: Minimum predicted IoU to draw.
        color: RGB color tuple.
        thickness: Line thickness.
        bbox_margin: Margin expansion in pixels.

    Returns:
        Image with bounding boxes drawn.
    """
    result = image.copy()
    for mask in masks:
        if mask.get("predicted_iou", 0) > iou_threshold:
            result = BBoxProcessor.draw_bbox_on_image(
                result, mask["bbox"], bbox_margin, color, thickness
            )
    return result


def crop_masks_to_bboxes(
    image: np.ndarray,
    masks: List[MaskDict],
    use_mask: bool = True,
    bbox_margin: int = 0,
    target_size: Tuple[int, int] = (512, 512),
) -> List[np.ndarray]:
    """Crop and resize image regions for each mask.

    Args:
        image: Input image.
        masks: List of SAM mask dictionaries.
        use_mask: If True, use mask for cropping; if False, use bbox only.
        bbox_margin: Margin expansion in pixels.
        target_size: Target resize dimensions (height, width).

    Returns:
        List of cropped and resized image patches.
    """
    crops = []
    for mask in masks:
        if use_mask:
            crop = BBoxProcessor.crop_by_mask(
                image, mask["segmentation"], mask["bbox"], bbox_margin
            )
        else:
            crop = BBoxProcessor.crop_by_bbox(image, mask["bbox"], bbox_margin)
        crop = cv2.resize(crop, target_size)
        crops.append(crop)
    return crops
