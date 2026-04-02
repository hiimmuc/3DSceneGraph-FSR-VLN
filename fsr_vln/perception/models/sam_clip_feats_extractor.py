import numpy as np
import torch
from memory.hmsg.utils.clip_utils import get_img_feats, get_img_feats_batch
from memory.hmsg.utils.sam_utils import crop_all_bounding_boxs


def extract_feats_raw(
    image,
    mask_generator,
    clip_model,
    preprocess,
    clip_feat_dim=768,
    bbox_margin=0,
    masked_weight=0.75,
):
    """Estimate the feature for each pixel in the image.

    Args:
        image: input image.
        mask_generator: sam model.
        clip_model: clip model.
        preprocess: clip preprocess.
        clip_feat_dim: clip feature dimension.
        bbox_margin: margin for cropped bounding box.
        masked_weight: weight for masked background.
        return:
        out_feat: pixel level feature.
        F_masks: feature for each mask.
        masks: all masks.
        cropped_images: cropped images.
        cropped_images_masked: cropped images with masked background.

    Returns:
        out_feat: pixel level feature.
    F_masks: feature for each mask.
    masks: all masks.
    cropped_images: cropped images.
    cropped_images_masked: cropped images with masked background."""
    LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = image.shape[0], image.shape[1]
    # run SAM on the full image.
    masks = mask_generator.generate(image)
    # run CLIP on the full image.
    F_g = get_img_feats(image, preprocess, clip_model)
    # crop all masks above certain thershold.
    cropped_images = crop_all_bounding_boxs(
        image, masks, block_background=False, bbox_margin=bbox_margin
    )
    cropped_images_masked = crop_all_bounding_boxs(
        image, masks, block_background=True, bbox_margin=bbox_margin
    )

    # run CLIP on all cropped images.
    F_l = []
    for img, img_masked in zip(cropped_images, cropped_images_masked):
        f_l_masked = get_img_feats(img_masked, preprocess, clip_model)
        f_l = get_img_feats(img, preprocess, clip_model)
        f_l = masked_weight * f_l_masked + (1 - masked_weight) * f_l
        f_l = torch.nn.functional.normalize(torch.from_numpy(f_l), p=2, dim=-1).cpu().numpy()
        F_l.append(f_l)
    F_l = np.array(F_l)
    F_l = torch.from_numpy(F_l).cuda()
    F_masks = F_l
    # interpolate F_p to the original image size
    out_feat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, clip_feat_dim).cuda()
    for i, mask in enumerate(masks):
        non_zero_indices = torch.argwhere(
            torch.from_numpy(np.array(mask["segmentation"])) == 1
        ).cuda()
        out_feat[non_zero_indices[:, 0], non_zero_indices[:, 1], :] += F_l[i, :]
        out_feat[non_zero_indices[:, 0], non_zero_indices[:, 1], :] = (
            torch.nn.functional.normalize(
                out_feat[non_zero_indices[:, 0], non_zero_indices[:, 1], :], p=2, dim=-1
            )
        )
    out_feat = out_feat.half()
    return out_feat.cpu(), F_masks.cpu(), masks, F_g


def extract_feats_per_pixel(
    image,
    mask_generator,
    clip_model,
    preprocess,
    clip_feat_dim=768,
    bbox_margin=0,
    masked_weight=0.75,
):
    """Estimate the feature for each pixel in the image using ConceptFusion

    method.

    Args:
        image: input image.
        mask_generator: sam model.
        clip_model: clip model.
        preprocess: clip preprocess.
        clip_feat_dim: clip feature dimension.
        bbox_margin: margin for cropped bounding box.
        masked_weight: weight for masked background.

    Returns:
        out_feat: pixel level feature.
        F_masks: feature for each mask.
        masks: all masks.
        F_g: global CLIP embedding for the image"""
    LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = image.shape[0], image.shape[1]

    masks = mask_generator.generate(image)

    F_g = get_img_feats(image, preprocess, clip_model)
    cropped_images = crop_all_bounding_boxs(
        image, masks, block_background=False, bbox_margin=bbox_margin
    )
    cropped_images_masked = crop_all_bounding_boxs(
        image, masks, block_background=True, bbox_margin=bbox_margin
    )
    number_of_masks = len(cropped_images)
    cropped_masked_feats = get_img_feats_batch(cropped_images_masked, preprocess, clip_model)
    cropped_feats = get_img_feats_batch(cropped_images, preprocess, clip_model)

    fused_crop_feats = torch.from_numpy(
        masked_weight * cropped_masked_feats + (1 - masked_weight) * cropped_feats
    )
    F_l = torch.nn.functional.normalize(fused_crop_feats, p=2, dim=-1).cpu().numpy()
    if F_l.shape[0] == 0:
        return None, None, None
    # 1. compute the cosine similarity etween the local feature fLi and the
    # global feature fG.
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    phi_l_G = cos(torch.from_numpy(F_l), torch.from_numpy(F_g))
    w_i = torch.nn.functional.softmax(phi_l_G, dim=0).reshape(-1, 1)
    # 2. compute the pixel-level feature Fp (one feauter vector for every
    # pixel in every mask) as a weighted sum of the local features
    F_p = w_i * F_g + (1 - w_i) * F_l.reshape(number_of_masks, clip_feat_dim)
    # 6. normalize F_p (TODO: no need because F_l and F_g are already
    # normalized, and normalize is costly)
    F_p = torch.nn.functional.normalize(F_p, p=2, dim=-1)
    # 7. interpolate F_p to the original image size
    F_p = F_p.cuda()
    out_feat = torch.zeros(LOAD_IMG_HEIGHT * LOAD_IMG_WIDTH, clip_feat_dim, device="cuda")
    non_zero_ids = torch.from_numpy(np.array([mask["segmentation"] for mask in masks])).reshape(
        (len(masks), -1)
    )
    for i, mask in enumerate(masks):
        non_zero_indices = torch.argwhere(non_zero_ids[i] == 1).cuda()
        out_feat[non_zero_indices, :] += F_p[i, :]
    out_feat = torch.nn.functional.normalize(out_feat, p=2, dim=-1)
    out_feat = out_feat.half()
    out_feat = out_feat.reshape((LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, clip_feat_dim))
    return out_feat.cpu(), F_p.cpu(), masks, F_g
