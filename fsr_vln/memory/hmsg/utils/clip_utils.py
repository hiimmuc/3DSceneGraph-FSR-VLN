"""CLIP feature extraction utilities. Provides unified interface for image and text encoding."""

from typing import List, Tuple

import numpy as np
import open_clip
import torch
from PIL import Image

DEFAULT_BATCH_SIZE = 64


class CLIPExtractor:
    """Extract features from images and text using CLIP models."""

    # Template variations for improved feature diversity and robustness
    EXTENDED_TEMPLATES = [
        "{}", "a photo of {}", "a photo of the {}", "a photo of one {}",
        "I took a picture of {}.", "I took a picture of my {}.", "I took a picture of the {}.",
        "a photo of my {}", "a photo of many {}", "a good photo of {}",
        "a good photo of the {}", "a bad photo of {}", "a bad photo of the {}",
        "a photo of a nice {}", "a photo of the nice {}", "a photo of a cool {}",
        "a photo of the cool {}", "a photo of a weird {}", "a photo of the weird {}",
        "a photo of a small {}", "a photo of the small {}", "a photo of a large {}",
        "a photo of the large {}", "a photo of a clean {}", "a photo of the clean {}",
        "a photo of a dirty {}", "a photo of the dirty {}", "a bright photo of {}",
        "a bright photo of the {}", "a dark photo of {}", "a dark photo of the {}",
        "a photo of a hard to see {}", "a photo of the hard to see {}", "a low resolution photo of {}",
        "a low resolution photo of the {}", "a cropped photo of {}", "a cropped photo of the {}",
        "a close-up photo of {}", "a close-up photo of the {}", "a jpeg corrupted photo of {}",
        "a jpeg corrupted photo of the {}", "a blurry photo of {}", "a blurry photo of the {}",
        "a pixelated photo of {}", "a pixelated photo of the {}", "a black and white photo of the {}",
        "a black and white photo of {}", "a plastic {}", "the plastic {}", "a toy {}",
        "the toy {}", "a plushie {}", "the plushie {}", "a cartoon {}", "the cartoon {}",
        "an embroidered {}", "the embroidered {}", "a painting of the {}", "a painting of a {}",
    ]

    SIMPLE_TEMPLATES = [
        "{}", "a photo of {} in the scene.",
    ]

    def __init__(self, clip_model: torch.nn.Module, preprocess, feat_dim: int, device: str = "cuda"):
        """Initialize with a CLIP model.

        Args:
            clip_model: CLIP vision/text model.
            preprocess: Image preprocessing function.
            feat_dim: Feature dimension (e.g., 768 for ViT-L-14).
            device: Device to run on ("cuda" or "cpu").
        """
        self.model = clip_model
        self.preprocess = preprocess
        self.feat_dim = feat_dim
        self.device = device

    def _to_pil(self, img: np.ndarray) -> Image.Image:
        """Convert numpy uint8 image to PIL Image."""
        return Image.fromarray(np.uint8(img))

    def _normalize_features(self, features: torch.Tensor) -> np.ndarray:
        """Normalize and convert features to numpy."""
        normalized = features / features.norm(dim=-1, keepdim=True)
        return np.float32(normalized.cpu().detach())

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a single image.

        Args:
            image: RGB image (H × W × 3) as np.ndarray.

        Returns:
            Feature vector (feat_dim,) as np.float32.
        """
        img_pil = self._to_pil(image)
        img_tensor = self.preprocess(img_pil)[None, ...].to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(img_tensor).float()
        return self._normalize_features(feats).squeeze(0)

    def encode_images(
        self, images: List[np.ndarray], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> np.ndarray:
        """Extract features from a list of images.

        Args:
            images: List of images (H × W × 3).
            batch_size: Batch size for processing.

        Returns:
            Feature matrix (N × feat_dim).
        """
        n_images = len(images)
        feats = np.zeros((n_images, self.feat_dim), dtype=np.float32)

        for start_idx in range(0, n_images, batch_size):
            end_idx = min(start_idx + batch_size, n_images)
            batch_imgs = images[start_idx:end_idx]

            # Handle empty images
            batch_imgs = [
                np.zeros((1, 1, 3), dtype=np.uint8) if img.size == 0 else img
                for img in batch_imgs
            ]

            pil_imgs = [self._to_pil(img) for img in batch_imgs]
            img_tensors = torch.stack([self.preprocess(img) for img in pil_imgs]).to(self.device)

            with torch.no_grad():
                batch_feats = self.model.encode_image(img_tensors).float()
            feats[start_idx:end_idx] = self._normalize_features(batch_feats)

        return feats

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Extract features from a list of text descriptions.

        Args:
            texts: List of text strings.

        Returns:
            Feature matrix (N × feat_dim).
        """
        text_tokens = open_clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(text_tokens).float()
        return self._normalize_features(feats)

    def encode_text_with_templates(
        self, texts: List[str], templates: List[str] = None
    ) -> np.ndarray:
        """Extract features from text with template augmentation, averaged over templates.

        Args:
            texts: List of text strings.
            templates: List of format strings (e.g., ['{}', 'a photo of {}']).
                       Defaults to SIMPLE_TEMPLATES.

        Returns:
            Feature matrix (N × feat_dim), averaged over template variations.
        """
        if templates is None:
            templates = self.SIMPLE_TEMPLATES

        templated_texts = [template.format(text) for text in texts for template in templates]
        all_feats = self.encode_text(templated_texts)
        n_templates = len(templates)

        # Reshape and average: (N*T × D) -> (N × T × D) -> (N × D)
        return np.mean(all_feats.reshape(-1, n_templates, self.feat_dim), axis=1)

    def match_image_to_text(
        self, image: np.ndarray, texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match image to text descriptions by similarity scoring.

        Args:
            image: RGB image.
            texts: List of text descriptions.

        Returns:
            Tuple of (scores, image_feats, text_feats).
        """
        img_feats = self.encode_image(image)
        text_feats = self.encode_text(texts)
        scores = (img_feats @ text_feats.T).flatten()
        return scores, img_feats, text_feats

    def get_top_k_images(
        self, images: List[np.ndarray], text: str, k: int = 5
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Retrieve top-k images most similar to a text query.

        Args:
            images: List of candidate images.
            text: Text query description.
            k: Number of top results to return.

        Returns:
            Tuple of (top_k_indices, top_k_images, top_k_scores).
        """
        img_feats = self.encode_images(images)
        text_feats = self.encode_text([text])
        scores = (img_feats @ text_feats.T).flatten()

        top_indices = np.argsort(scores)[::-1][:k]
        top_scores = scores[top_indices]
        top_images = [images[i] for i in top_indices]

        return top_indices, top_images, top_scores


# Legacy function wrappers for backward compatibility
def get_text_feats(
    texts: List[str], clip_model: torch.nn.Module, feat_dim: int, batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """Get text features (legacy wrapper)."""
    extractor = CLIPExtractor(clip_model, None, feat_dim)
    return extractor.encode_text(texts)


def get_text_feats_62_templates(
    texts: List[str], clip_model: torch.nn.Module, feat_dim: int, batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """Get text features with extended templates (legacy wrapper)."""
    extractor = CLIPExtractor(clip_model, None, feat_dim)
    return extractor.encode_text_with_templates(texts, CLIPExtractor.EXTENDED_TEMPLATES)


def get_text_feats_multiple_templates(
    texts: List[str], clip_model: torch.nn.Module, feat_dim: int, batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """Get text features with simple templates (legacy wrapper)."""
    extractor = CLIPExtractor(clip_model, None, feat_dim)
    return extractor.encode_text_with_templates(texts, CLIPExtractor.SIMPLE_TEMPLATES)
