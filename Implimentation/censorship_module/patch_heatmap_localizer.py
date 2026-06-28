"""Weakly supervised patch-risk heatmap localization."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch

from . import config
from .convnext_frame_risk import score_frames_with_convnext


Box = dict


def _positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    values = list(range(0, max(1, length - patch_size + 1), max(1, stride)))
    last = length - patch_size
    if values[-1] != last:
        values.append(last)
    return sorted(set(int(v) for v in values))


def generate_patches(frame_bgr: np.ndarray, patch_sizes: Iterable[int], stride: int) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    h, w = frame_bgr.shape[:2]
    patches: List[np.ndarray] = []
    boxes: List[Tuple[int, int, int, int]] = []
    for raw_size in patch_sizes:
        size = int(min(max(1, int(raw_size)), h, w))
        for y in _positions(h, size, int(stride)):
            for x in _positions(w, size, int(stride)):
                x2 = min(w, x + size)
                y2 = min(h, y + size)
                patch = frame_bgr[y:y2, x:x2]
                if patch.size == 0:
                    continue
                patches.append(patch.copy())
                boxes.append((int(x), int(y), int(x2), int(y2)))
    return patches, boxes


def score_patches(model, patches: List[np.ndarray], device: str | torch.device) -> List[float]:
    rows = score_frames_with_convnext(patches, model, device, batch_size=int(config.PATCH_BATCH_SIZE))
    return [float(row["risk"]) for row in rows]


def build_risk_heatmap(
    frame_shape: Tuple[int, int] | Tuple[int, int, int],
    patch_boxes: List[Tuple[int, int, int, int]],
    patch_scores: List[float],
) -> np.ndarray:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    sums = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    for (x1, y1, x2, y2), score in zip(patch_boxes, patch_scores):
        sums[y1:y2, x1:x2] += float(score)
        counts[y1:y2, x1:x2] += 1.0
    heatmap = np.divide(sums, np.maximum(counts, 1.0), dtype=np.float32)
    if float(heatmap.max()) > float(heatmap.min()):
        heatmap = (heatmap - float(heatmap.min())) / (float(heatmap.max()) - float(heatmap.min()))
    return heatmap


def _pad_and_clip_box(box: Tuple[int, int, int, int], frame_shape) -> Tuple[int, int, int, int]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * float(config.BOX_PADDING_RATIO)))
    pad_y = int(round(bh * float(config.BOX_PADDING_RATIO)))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )


def heatmap_to_boxes(heatmap: np.ndarray, frame_shape) -> List[Box]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if heatmap.size == 0:
        return []
    percentile_threshold = float(np.percentile(heatmap, float(config.TOP_PERCENTILE)))
    threshold = max(float(config.PATCH_RISK_THRESHOLD), percentile_threshold)
    mask = (heatmap >= threshold).astype(np.uint8)
    if mask.max() == 0:
        return []

    kernel = np.ones((9, 9), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = float(config.MIN_BOX_AREA_RATIO) * float(h * w)
    boxes: List[Box] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        bw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        x1, y1, x2, y2 = _pad_and_clip_box((x, y, x + bw, y + bh), frame_shape)
        component_risk = float(np.max(heatmap[labels == label_idx])) if np.any(labels == label_idx) else 0.0
        if x2 > x1 and y2 > y1:
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "risk": component_risk})
    return boxes


def _union_area(boxes: List[Box], frame_shape) -> int:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(box[k]) for k in ("x1", "y1", "x2", "y2")]
        mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 1
    return int(mask.sum())


def limit_boxes_by_area(boxes: List[Box], frame_shape) -> Tuple[List[Box], bool]:
    max_area = float(config.MAX_CENSOR_AREA_RATIO) * float(frame_shape[0] * frame_shape[1])
    if not boxes or _union_area(boxes, frame_shape) <= max_area:
        return boxes, False

    kept: List[Box] = []
    for box in sorted(boxes, key=lambda item: float(item.get("risk", 0.0)), reverse=True):
        candidate = kept + [box]
        if _union_area(candidate, frame_shape) <= max_area:
            kept.append(box)
    return kept, True


def map_boxes_to_frame(boxes: List[Box], origin: Tuple[int, int], frame_shape) -> List[Box]:
    offset_x, offset_y = int(origin[0]), int(origin[1])
    h, w = int(frame_shape[0]), int(frame_shape[1])
    mapped: List[Box] = []
    for box in boxes:
        x1 = max(0, min(w, int(box["x1"]) + offset_x))
        y1 = max(0, min(h, int(box["y1"]) + offset_y))
        x2 = max(0, min(w, int(box["x2"]) + offset_x))
        y2 = max(0, min(h, int(box["y2"]) + offset_y))
        if x2 > x1 and y2 > y1:
            mapped.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "risk": float(box.get("risk", 0.0)),
                }
            )
    return mapped


def localize_sensitive_regions_with_metadata(
    frame_bgr: np.ndarray,
    image_model,
    device: str | torch.device,
    *,
    origin: Tuple[int, int] = (0, 0),
    full_frame_shape=None,
) -> dict:
    patches, boxes = generate_patches(frame_bgr, config.PATCH_SIZES, int(config.PATCH_STRIDE))
    if not patches:
        return {
            "boxes": [],
            "patch_boxes_before_roi_mapping": [],
            "max_area_filter_applied": False,
        }
    scores = score_patches(image_model, patches, device)
    heatmap = build_risk_heatmap(frame_bgr.shape, boxes, scores)
    crop_local_boxes = heatmap_to_boxes(heatmap, frame_bgr.shape)
    target_shape = full_frame_shape if full_frame_shape is not None else frame_bgr.shape
    mapped_boxes = map_boxes_to_frame(crop_local_boxes, origin, target_shape)
    limited, applied = limit_boxes_by_area(mapped_boxes, target_shape)
    return {
        "boxes": limited,
        "patch_boxes_before_roi_mapping": crop_local_boxes,
        "max_area_filter_applied": bool(applied),
    }


def localize_sensitive_regions(frame_bgr: np.ndarray, image_model, device: str | torch.device) -> List[Box]:
    result = localize_sensitive_regions_with_metadata(frame_bgr, image_model, device)
    return result["boxes"]
