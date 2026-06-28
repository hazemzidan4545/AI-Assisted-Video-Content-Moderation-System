"""Render pixelation or blur censorship for unsafe video segments."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch

from . import config
from .patch_heatmap_localizer import localize_sensitive_regions_with_metadata
from .video_window_detector import detect_person_roi_box


def pixelate_region(frame: np.ndarray, box: dict, factor: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(box[k]) for k in ("x1", "y1", "x2", "y2")]
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return frame
    factor = max(2, int(factor))
    small_w = max(1, region.shape[1] // factor)
    small_h = max(1, region.shape[0] // factor)
    small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = pixelated
    return frame


def blur_region(frame: np.ndarray, box: dict, kernel_size: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(box[k]) for k in ("x1", "y1", "x2", "y2")]
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return frame
    kernel = max(3, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    blurred = region
    for _ in range(max(1, int(getattr(config, "BLUR_PASSES", 1)))):
        blurred = cv2.GaussianBlur(blurred, (kernel, kernel), 0)
    frame[y1:y2, x1:x2] = blurred
    return frame


def apply_censorship_to_frame(frame: np.ndarray, boxes: List[dict], mode: str | None = None) -> np.ndarray:
    mode = str(mode or config.CENSOR_MODE).lower()
    output = frame.copy()
    for box in boxes:
        if mode == "pixelate":
            output = pixelate_region(output, box, int(config.PIXELATION_FACTOR))
        elif mode == "blur":
            output = blur_region(output, box, int(config.BLUR_KERNEL))
        else:
            raise ValueError(f"Unsupported censor mode: {mode}")
    return output


def make_center_fallback_box(frame_shape) -> dict:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    box_w = int(round(w * 0.40))
    box_h = int(round(h * 0.45))
    x1 = max(0, (w - box_w) // 2)
    y1 = max(0, (h - box_h) // 2)
    return {"x1": x1, "y1": y1, "x2": min(w, x1 + box_w), "y2": min(h, y1 + box_h), "risk": 0.0}


def make_person_roi_body_fallback_box(person_roi_box: dict) -> dict:
    x1 = int(person_roi_box["x1"])
    y1 = int(person_roi_box["y1"])
    x2 = int(person_roi_box["x2"])
    y2 = int(person_roi_box["y2"])
    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)
    h0, h1 = [float(v) for v in config.BODY_FALLBACK_HORIZONTAL_RANGE]
    v0, v1 = [float(v) for v in config.BODY_FALLBACK_VERTICAL_RANGE]
    return {
        "x1": int(round(x1 + roi_w * h0)),
        "y1": int(round(y1 + roi_h * v0)),
        "x2": int(round(x1 + roi_w * h1)),
        "y2": int(round(y1 + roi_h * v1)),
        "risk": 0.0,
    }


def _whole_frame_box(frame_shape) -> dict:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    return {"x1": 0, "y1": 0, "x2": w, "y2": h, "risk": 0.0}


def _inside_segment(timestamp: float, segments: List[dict]) -> bool:
    return any(float(seg["start"]) <= float(timestamp) <= float(seg["end"]) for seg in segments)


def _inside_effective_render_segment(timestamp: float, segments: List[dict]) -> bool:
    trim = max(0.0, float(getattr(config, "FULL_FRAME_END_TRIM_SECONDS", 0.0)))
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        if _resolve_censor_region_mode() == "full_frame":
            end = max(start, end - trim)
        if start <= float(timestamp) <= end:
            return True
    return False


def _sanitize_boxes(boxes: List[dict], frame_shape) -> List[dict]:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    out = []
    for box in boxes:
        x1 = max(0, min(w, int(box.get("x1", 0))))
        y1 = max(0, min(h, int(box.get("y1", 0))))
        x2 = max(0, min(w, int(box.get("x2", 0))))
        y2 = max(0, min(h, int(box.get("y2", 0))))
        if x2 > x1 and y2 > y1:
            out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "risk": float(box.get("risk", 0.0))})
    return out


def _fallback_for_missing_roi(frame_shape) -> Tuple[List[dict], bool, str | None]:
    fallback_mode = str(config.FALLBACK_MODE).lower()
    if fallback_mode == "center_region":
        return [make_center_fallback_box(frame_shape)], True, "center_region"
    if fallback_mode == "whole_frame_last_resort":
        return [_whole_frame_box(frame_shape)], True, "whole_frame_last_resort"
    if fallback_mode == "none":
        return [], False, "none"
    raise ValueError(f"Unsupported fallback mode: {config.FALLBACK_MODE}")


def _detect_person_roi_metadata(frame: np.ndarray, metadata: dict) -> dict | None:
    person_roi = detect_person_roi_box(frame)
    metadata["roi_found"] = person_roi is not None
    if person_roi is not None:
        metadata["person_roi_box"] = person_roi
        metadata["roi_valid"] = bool(person_roi.get("valid", False))
        metadata["roi_area_ratio"] = float(person_roi.get("area_ratio", 0.0))
    return person_roi


def _resolve_censor_region_mode() -> str:
    mode = getattr(config, "CENSOR_REGION_MODE", None)
    if mode is None:
        mode = getattr(config, "LOCALIZATION_SCOPE", "full_frame")
    mode = str(mode).lower()
    legacy_map = {
        "person_roi": "patch_debug",
        "person_roi_patch_debug": "patch_debug",
        "full_frame_debug": "full_frame_debug",
    }
    return legacy_map.get(mode, mode)


def _localize_unsafe_frame(frame: np.ndarray, image_model, device: str | torch.device) -> Tuple[List[dict], dict]:
    scope = str(getattr(config, "LOCALIZATION_SCOPE", "")).lower()
    mode = _resolve_censor_region_mode()
    metadata = {
        "censor_region_mode": mode,
        "roi_found": False,
        "roi_valid": False,
        "roi_area_ratio": None,
        "localization_scope": scope,
        "person_roi_box": None,
        "body_region_box": None,
        "patch_boxes_before_roi_mapping": [],
        "fallback_type": None,
        "fallback_used": False,
        "max_area_filter_applied": False,
        "used_full_frame_debug": False,
        "full_frame_censored": False,
        "person_roi_used": False,
        "patch_heatmap_used": False,
    }

    if mode == "full_frame":
        boxes = [_whole_frame_box(frame.shape)]
        metadata["fallback_type"] = "full_frame"
        metadata["full_frame_censored"] = True
        return boxes, metadata

    if mode == "full_frame_debug":
        metadata["patch_heatmap_used"] = True
        result = localize_sensitive_regions_with_metadata(
            frame,
            image_model,
            device,
            origin=(0, 0),
            full_frame_shape=frame.shape,
        )
        boxes = _sanitize_boxes(result["boxes"], frame.shape)
        metadata["patch_boxes_before_roi_mapping"] = result["patch_boxes_before_roi_mapping"]
        metadata["max_area_filter_applied"] = bool(result["max_area_filter_applied"])
        metadata["used_full_frame_debug"] = True
        if not boxes:
            boxes, fallback_used, fallback_type = _fallback_for_missing_roi(frame.shape)
            metadata["fallback_used"] = fallback_used
            metadata["fallback_type"] = fallback_type
        return boxes, metadata

    if mode not in {"person_body", "patch_debug"}:
        raise ValueError(
            f"Unsupported CENSOR_REGION_MODE/LOCALIZATION_SCOPE: "
            f"{getattr(config, 'CENSOR_REGION_MODE', None)!r}/{getattr(config, 'LOCALIZATION_SCOPE', None)!r}"
        )

    person_roi = _detect_person_roi_metadata(frame, metadata)

    if person_roi is not None and bool(person_roi.get("valid", False)):
        body_region_box = _sanitize_boxes([make_person_roi_body_fallback_box(person_roi)], frame.shape)
        metadata["body_region_box"] = body_region_box[0] if body_region_box else None

        if mode == "person_body":
            metadata["fallback_used"] = bool(body_region_box)
            metadata["fallback_type"] = "person_body" if body_region_box else "none"
            metadata["person_roi_used"] = bool(body_region_box)
            return body_region_box, metadata

        x1, y1, x2, y2 = [int(person_roi[k]) for k in ("x1", "y1", "x2", "y2")]
        roi_crop = frame[y1:y2, x1:x2]
        if roi_crop.size > 0:
            metadata["patch_heatmap_used"] = True
            result = localize_sensitive_regions_with_metadata(
                roi_crop,
                image_model,
                device,
                origin=(x1, y1),
                full_frame_shape=frame.shape,
            )
            boxes = _sanitize_boxes(result["boxes"], frame.shape)
            metadata["patch_boxes_before_roi_mapping"] = result["patch_boxes_before_roi_mapping"]
            metadata["max_area_filter_applied"] = bool(result["max_area_filter_applied"])
            if boxes:
                metadata["person_roi_used"] = True
                return boxes, metadata

        if bool(config.PERSON_ROI_BODY_FALLBACK):
            boxes = body_region_box
            metadata["fallback_used"] = bool(boxes)
            metadata["fallback_type"] = "person_body" if boxes else "none"
            metadata["person_roi_used"] = bool(boxes)
            return boxes, metadata
        return [], metadata

    boxes, fallback_used, fallback_type = _fallback_for_missing_roi(frame.shape)
    metadata["fallback_used"] = fallback_used
    metadata["fallback_type"] = fallback_type
    return boxes, metadata


def process_video_with_segments(
    input_video: str | Path,
    output_video: str | Path,
    segments: List[dict],
    image_model,
    device: str | torch.device,
    progress_callback: Callable[[float], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    progress_start: float = 0.50,
    progress_end: float = 0.90,
) -> Tuple[List[dict], dict]:
    input_video = Path(input_video)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output writer: {output_video}")

    frame_logs: List[dict] = []
    previous_boxes: List[dict] = []
    previous_metadata: dict | None = None
    frame_idx = 0
    unsafe_frame_count = 0
    fallback_used_frame_count = 0
    max_area_filter_frame_count = 0
    frames_with_person_roi = 0
    frames_with_valid_person_roi = 0
    frames_with_invalid_person_roi = 0
    frames_using_body_fallback = 0
    frames_using_center_fallback = 0
    frames_using_full_frame_debug = 0
    frames_using_patch_heatmap = 0
    frames_full_frame_censored = 0

    if status_callback is not None:
        status_callback("Rendering censored video...")
    if progress_callback is not None:
        progress_callback(float(progress_start))

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        timestamp = frame_idx / fps if fps > 0 else 0.0
        inside = _inside_effective_render_segment(timestamp, segments)
        boxes: List[dict] = []
        fallback_used = False
        max_area_filter_applied = False
        metadata = {
            "censor_region_mode": _resolve_censor_region_mode(),
            "roi_found": False,
            "roi_valid": False,
            "roi_area_ratio": None,
            "localization_scope": str(config.LOCALIZATION_SCOPE).lower(),
            "person_roi_box": None,
            "body_region_box": None,
            "patch_boxes_before_roi_mapping": [],
            "fallback_type": None,
            "full_frame_censored": False,
            "person_roi_used": False,
            "patch_heatmap_used": False,
        }

        if inside:
            unsafe_frame_count += 1
            if frame_idx % max(1, int(config.LOCALIZE_EVERY_N_FRAMES)) == 0 or not previous_boxes:
                boxes, localization_metadata = _localize_unsafe_frame(frame, image_model, device)
                previous_metadata = localization_metadata
                previous_boxes = boxes
            else:
                boxes = previous_boxes
                localization_metadata = previous_metadata or {}

            metadata.update(
                {
                    key: localization_metadata.get(key, metadata.get(key))
                    for key in (
                        "censor_region_mode",
                        "roi_found",
                        "roi_valid",
                        "roi_area_ratio",
                        "localization_scope",
                        "person_roi_box",
                        "body_region_box",
                        "patch_boxes_before_roi_mapping",
                        "fallback_type",
                        "full_frame_censored",
                        "person_roi_used",
                        "patch_heatmap_used",
                    )
                }
            )
            fallback_used = bool(localization_metadata.get("fallback_used", False))
            max_area_filter_applied = bool(localization_metadata.get("max_area_filter_applied", False))
            if max_area_filter_applied:
                max_area_filter_frame_count += 1

            if fallback_used:
                fallback_used_frame_count += 1
            if bool(metadata["roi_found"]):
                frames_with_person_roi += 1
            if bool(metadata["roi_valid"]):
                frames_with_valid_person_roi += 1
            elif bool(metadata["roi_found"]):
                frames_with_invalid_person_roi += 1
            if metadata["fallback_type"] == "person_body":
                frames_using_body_fallback += 1
            if metadata["fallback_type"] == "center_region":
                frames_using_center_fallback += 1
            if bool(localization_metadata.get("used_full_frame_debug", False)):
                frames_using_full_frame_debug += 1
            if bool(metadata.get("patch_heatmap_used", False)):
                frames_using_patch_heatmap += 1
            if bool(metadata.get("full_frame_censored", False)):
                frames_full_frame_censored += 1
            if boxes:
                frame = apply_censorship_to_frame(frame, boxes, mode=config.CENSOR_MODE)

        writer.write(frame)
        frame_logs.append(
            {
                "frame_idx": int(frame_idx),
                "time": float(timestamp),
                "inside_unsafe_segment": bool(inside),
                "boxes": boxes,
                "fallback_used": bool(fallback_used),
                "max_area_filter_applied": bool(max_area_filter_applied),
                **metadata,
            }
        )
        frame_idx += 1
        if progress_callback is not None and total_frames > 0 and (frame_idx == total_frames or frame_idx % 10 == 0):
            fraction = min(1.0, max(0.0, frame_idx / float(total_frames)))
            progress_callback(float(progress_start + (progress_end - progress_start) * fraction))

    cap.release()
    writer.release()
    if progress_callback is not None:
        progress_callback(float(progress_end))
    if status_callback is not None:
        status_callback("Finished rendering censored video.")

    fallback_percent = (
        100.0 * float(fallback_used_frame_count) / float(unsafe_frame_count)
        if unsafe_frame_count > 0
        else 0.0
    )
    stats = {
        "total_frames": int(frame_idx),
        "unsafe_frame_count": int(unsafe_frame_count),
        "fallback_used_frame_count": int(fallback_used_frame_count),
        "fallback_used_percent": float(fallback_percent),
        "max_area_filter_frame_count": int(max_area_filter_frame_count),
        "frames_with_person_roi": int(frames_with_person_roi),
        "frames_with_valid_person_roi": int(frames_with_valid_person_roi),
        "frames_with_invalid_person_roi": int(frames_with_invalid_person_roi),
        "frames_using_body_fallback": int(frames_using_body_fallback),
        "frames_using_center_fallback": int(frames_using_center_fallback),
        "frames_using_full_frame_debug": int(frames_using_full_frame_debug),
        "frames_using_patch_heatmap": int(frames_using_patch_heatmap),
        "frames_full_frame_censored": int(frames_full_frame_censored),
    }
    return frame_logs, stats
