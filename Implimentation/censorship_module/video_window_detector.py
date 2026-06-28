"""Unsafe video-window detection with visual fusion."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from . import config
from .convnext_frame_risk import compute_clip_frame_risk, preprocess_frame_for_convnext, score_frames_with_convnext
from .model_loader import IMPLEMENTATION_DIR, PROJECT_ROOT


_ROI_MODULE = None
_ROI_EXTRACTOR = None
_ROI_AVAILABLE = None
_ROI_WARNING_EMITTED = False


def get_video_metadata(video_path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = float(frame_count / fps) if fps > 0 else 0.0
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def read_raw_frame_at_time(video_path: str | Path, timestamp: float) -> Optional[np.ndarray]:
    metadata = get_video_metadata(video_path)
    fps = float(metadata.get("fps", 0.0) or 0.0)
    frame_count = int(metadata.get("frame_count", 0) or 0)
    if fps <= 0 or frame_count <= 0:
        return None

    frame_idx = max(0, min(frame_count - 1, int(round(float(timestamp) * fps))))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def scene_cut_score(frame_a_bgr: Optional[np.ndarray], frame_b_bgr: Optional[np.ndarray]) -> float:
    if frame_a_bgr is None or frame_b_bgr is None:
        return 0.0
    a_rgb = cv2.cvtColor(frame_a_bgr, cv2.COLOR_BGR2RGB)
    b_rgb = cv2.cvtColor(frame_b_bgr, cv2.COLOR_BGR2RGB)
    a_rgb = cv2.resize(a_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    b_rgb = cv2.resize(b_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(a_rgb.astype(np.float32) - b_rgb.astype(np.float32))))


def _resolve_local_yolo_weights() -> Optional[Path]:
    candidates = [
        Path.cwd() / "yolov8n.pt",
        IMPLEMENTATION_DIR / "yolov8n.pt",
        PROJECT_ROOT / "yolov8n.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _get_roi_module(require: bool = False) -> Tuple[Optional[object], bool, str]:
    global _ROI_MODULE, _ROI_EXTRACTOR, _ROI_AVAILABLE
    if _ROI_AVAILABLE is not None:
        return _ROI_MODULE, bool(_ROI_AVAILABLE), "" if _ROI_AVAILABLE else "ROI unavailable"

    for path in (IMPLEMENTATION_DIR, PROJECT_ROOT, Path.cwd()):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    yolo_weights = _resolve_local_yolo_weights()
    if yolo_weights is None:
        _ROI_AVAILABLE = False
        message = "Local yolov8n.pt was not found; ROI fallback will use full frames."
        if require:
            raise FileNotFoundError(message)
        return None, False, message

    errors = []
    for import_path in ("pipelines.roi_detector", "Implimentation.pipelines.roi_detector"):
        try:
            module = __import__(import_path, fromlist=["extract_person_crop"])
            if getattr(module, "YOLO", object()) is None:
                message = "ultralytics is not installed; ROI fallback will use full frames."
                _ROI_MODULE = None
                _ROI_EXTRACTOR = None
                _ROI_AVAILABLE = False
                if require:
                    raise ModuleNotFoundError(message)
                return None, False, message
            if hasattr(module, "_MODEL_NAME"):
                module._MODEL_NAME = str(yolo_weights)
            _ROI_MODULE = module
            _ROI_EXTRACTOR = module.extract_person_crop
            _ROI_AVAILABLE = True
            return _ROI_MODULE, True, ""
        except Exception as exc:
            errors.append(f"{import_path}: {exc}")

    _ROI_MODULE = None
    _ROI_EXTRACTOR = None
    _ROI_AVAILABLE = False
    message = "Could not import local ROI detector:\n" + "\n".join(errors)
    if require:
        raise RuntimeError(message)
    return None, False, message


def _get_roi_extractor(require: bool = False) -> Tuple[Optional[Callable], bool, str]:
    module, available, message = _get_roi_module(require=require)
    if not available or module is None:
        return None, available, message
    return getattr(module, "extract_person_crop", None), True, ""


def _maybe_emit_roi_warning(message: str) -> None:
    global _ROI_WARNING_EMITTED
    if message and not _ROI_WARNING_EMITTED:
        warnings.warn(message)
        _ROI_WARNING_EMITTED = True


def detect_person_roi_box(frame_bgr: np.ndarray, require: bool = False) -> Optional[dict]:
    """Return the largest local person ROI as original-frame coordinates.

    The existing project ROI helper returns only a crop, so this mirrors its
    local YOLO person-box logic without changing the old project file.
    """

    roi_mode = str(config.ROI_MODE).lower()
    if roi_mode == "none":
        return None

    module, available, message = _get_roi_module(require=require or roi_mode == "yolo")
    if not available or module is None:
        if roi_mode == "auto":
            _maybe_emit_roi_warning(message)
        return None
    if not hasattr(module, "_get_model"):
        if require or roi_mode == "yolo":
            raise RuntimeError("Local ROI module does not expose _get_model for ROI-box detection.")
        return None

    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    original = frame_rgb
    input_frame = frame_rgb
    scale_x = 1.0
    scale_y = 1.0

    height, width = original.shape[:2]
    inference_size = int(config.ROI_INFERENCE_SIZE) if config.ROI_INFERENCE_SIZE else None
    if inference_size and inference_size > 0 and max(height, width) > inference_size:
        if height >= width:
            resized_h = inference_size
            resized_w = max(1, int(round(width * inference_size / height)))
        else:
            resized_w = inference_size
            resized_h = max(1, int(round(height * inference_size / width)))
        input_frame = cv2.resize(original, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        scale_x = width / float(resized_w)
        scale_y = height / float(resized_h)

    try:
        model = module._get_model()
    except Exception as exc:
        message = f"Local ROI model unavailable; ROI fallback will use full frames. {exc}"
        if require or roi_mode == "yolo":
            raise RuntimeError(message) from exc
        if roi_mode == "auto":
            _maybe_emit_roi_warning(message)
        return None
    try:
        results = model(input_frame, classes=[0], conf=float(config.ROI_CONF), verbose=False, device="cpu")[0]
    except TypeError:
        results = model(input_frame, classes=[0], conf=float(config.ROI_CONF), verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return None

    boxes = results.boxes.xyxy.detach().cpu().numpy()
    if boxes.size == 0:
        return None
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes[best_idx].astype(int)
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    confidence = None
    if getattr(results.boxes, "conf", None) is not None:
        conf_values = results.boxes.conf.detach().cpu().numpy()
        if len(conf_values) > best_idx:
            confidence = float(conf_values[best_idx])

    roi_area = float((x2 - x1) * (y2 - y1))
    frame_area = float(max(1, width * height))
    area_ratio = roi_area / frame_area
    valid = float(config.MIN_PERSON_ROI_AREA_RATIO) <= area_ratio <= float(config.MAX_PERSON_ROI_AREA_RATIO)
    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "confidence": confidence,
        "area_ratio": float(area_ratio),
        "valid": bool(valid),
    }


def _sample_window_frames_with_stats(
    video_path: str | Path,
    start_time: float,
    end_time: float,
    seq_len: int,
) -> Tuple[List[np.ndarray], dict]:
    metadata = get_video_metadata(video_path)
    fps = max(float(metadata["fps"]), 1e-6)
    frame_count = int(metadata["frame_count"])
    if frame_count <= 0:
        return [], {
            "roi_mode": config.ROI_MODE,
            "roi_available": False,
            "roi_fallback_used": True,
            "roi_frames_found": 0,
            "roi_frames_total": 0,
        }

    start_frame = max(0, min(frame_count - 1, int(round(float(start_time) * fps))))
    end_frame = max(start_frame + 1, min(frame_count, int(round(float(end_time) * fps))))
    indices = np.linspace(start_frame, end_frame - 1, num=int(seq_len)).astype(np.int64)

    roi_mode = str(config.ROI_MODE).lower()
    require_roi = roi_mode == "yolo"
    use_roi = roi_mode in {"auto", "yolo"}
    roi_extractor = None
    roi_available = False
    roi_message = ""
    if use_roi:
        roi_extractor, roi_available, roi_message = _get_roi_extractor(require=require_roi)
        if not roi_available and roi_mode == "auto":
            _maybe_emit_roi_warning(roi_message)

    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    roi_found = 0
    roi_total = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            if frames:
                frames.append(frames[-1].copy())
            continue

        if use_roi and roi_available and roi_extractor is not None:
            roi_total += 1
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                crop_rgb = roi_extractor(
                    rgb,
                    conf=float(config.ROI_CONF),
                    inference_size=int(config.ROI_INFERENCE_SIZE) if config.ROI_INFERENCE_SIZE else None,
                )
            except Exception as exc:
                message = f"ROI extraction unavailable; continuing with full frames. {exc}"
                if require_roi:
                    cap.release()
                    raise RuntimeError(message) from exc
                _maybe_emit_roi_warning(message)
                roi_available = False
                roi_extractor = None
                crop_rgb = None
            if crop_rgb is not None and getattr(crop_rgb, "size", 0) > 0:
                if crop_rgb.shape[:2] != rgb.shape[:2]:
                    roi_found += 1
                frame_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

        frames.append(frame_bgr)
    cap.release()

    while frames and len(frames) < int(seq_len):
        frames.append(frames[-1].copy())

    roi_fallback_used = bool(use_roi and (not roi_available or roi_found < roi_total))
    return frames[: int(seq_len)], {
        "roi_mode": roi_mode,
        "roi_available": bool(roi_available),
        "roi_fallback_used": roi_fallback_used,
        "roi_frames_found": int(roi_found),
        "roi_frames_total": int(roi_total),
    }


def sample_window_frames(video_path: str | Path, start_time: float, end_time: float, seq_len: int) -> List[np.ndarray]:
    frames, _stats = _sample_window_frames_with_stats(video_path, start_time, end_time, seq_len)
    return frames


def predict_temporal_window(frames: List[np.ndarray], temporal_model, device: str | torch.device) -> float:
    if not frames:
        return 0.0
    tensor = torch.stack([preprocess_frame_for_convnext(frame) for frame in frames]).unsqueeze(0).to(device)
    temporal_model.eval()
    with torch.no_grad():
        probs = torch.softmax(temporal_model(tensor), dim=1)
    return float(probs[0, 1].detach().cpu().item())


def predict_frame_risk_for_window(frames: List[np.ndarray], image_model, device: str | torch.device) -> dict:
    rows = score_frames_with_convnext(frames, image_model, device)
    risk_summary = compute_clip_frame_risk([row["risk"] for row in rows])
    risk_summary["frame_scores"] = rows
    return risk_summary


def _should_compute_frame_risk_for_windows() -> bool:
    policy = str(config.FUSED_UNSAFE_POLICY).lower()
    region_mode = str(getattr(config, "CENSOR_REGION_MODE", "")).lower()
    if region_mode == "full_frame" and policy == "temporal_only":
        return bool(getattr(config, "COMPUTE_FRAME_RISK_IN_FULL_FRAME_MODE", False))
    return True


def fuse_window_scores(temporal_unsafe_prob: float, frame_clip_risk: float) -> dict:
    policy = str(config.FUSED_UNSAFE_POLICY).lower()
    temporal_score = float(temporal_unsafe_prob)
    frame_score = float(frame_clip_risk)

    if policy == "temporal_only":
        fused_score = temporal_score
        unsafe = fused_score >= float(config.TEMPORAL_UNSAFE_THRESHOLD)
    elif policy == "frame_only":
        fused_score = frame_score
        unsafe = fused_score >= float(config.FRAME_RISK_THRESHOLD)
    elif policy == "weighted":
        fused_score = float(config.TEMPORAL_WEIGHT) * temporal_score + float(config.FRAME_WEIGHT) * frame_score
        unsafe = fused_score >= float(config.FUSED_THRESHOLD)
    elif policy == "or":
        unsafe = (
            temporal_score >= float(config.TEMPORAL_UNSAFE_THRESHOLD)
            or frame_score >= float(config.FRAME_RISK_THRESHOLD)
        )
        fused_score = max(temporal_score, frame_score)
    else:
        raise ValueError(f"Unsupported FUSED_UNSAFE_POLICY: {config.FUSED_UNSAFE_POLICY}")

    return {"fused_score": float(fused_score), "unsafe": bool(unsafe)}


def predict_video_windows(video_path: str | Path, temporal_model, image_model, device: str | torch.device) -> List[dict]:
    metadata = get_video_metadata(video_path)
    duration = float(metadata["duration"])
    if duration <= 0:
        return []

    starts = list(np.arange(0.0, max(duration - 1e-6, 0.0), float(config.WINDOW_STRIDE_SECONDS)))
    rows: List[dict] = []
    previous_reference_frame = None
    for window_index, start in enumerate(starts):
        end = min(duration, float(start) + float(config.WINDOW_SECONDS))
        reference_time = min(duration, max(0.0, float(start)))
        reference_frame = read_raw_frame_at_time(video_path, reference_time)
        cut_score = scene_cut_score(previous_reference_frame, reference_frame) if previous_reference_frame is not None else 0.0
        scene_cut = bool(
            bool(config.ENABLE_SCENE_CUT_BOUNDARY)
            and window_index > 0
            and cut_score >= float(config.SCENE_CUT_THRESHOLD)
        )
        frames, roi_stats = _sample_window_frames_with_stats(video_path, start, end, int(config.SEQ_LEN))
        temporal_prob = predict_temporal_window(frames, temporal_model, device)
        if _should_compute_frame_risk_for_windows():
            frame_risk = predict_frame_risk_for_window(frames, image_model, device)
        else:
            frame_risk = {
                "frame_clip_risk": 0.0,
                "frame_max_risk": 0.0,
                "frame_scores": [],
                "frame_risk_computed": False,
            }
        fused = fuse_window_scores(temporal_prob, frame_risk["frame_clip_risk"])
        rows.append(
            {
                "window_index": int(window_index),
                "start": float(start),
                "end": float(end),
                "scene_cut_from_previous": bool(scene_cut),
                "scene_cut_score_from_previous": float(cut_score),
                "temporal_unsafe_prob": float(temporal_prob),
                "frame_clip_risk": float(frame_risk["frame_clip_risk"]),
                "frame_max_risk": float(frame_risk["frame_max_risk"]),
                "frame_risk_computed": bool(frame_risk.get("frame_risk_computed", _should_compute_frame_risk_for_windows())),
                "fused_score": float(fused["fused_score"]),
                "unsafe": bool(fused["unsafe"]),
                **roi_stats,
            }
        )
        previous_reference_frame = reference_frame
        if end >= duration:
            break
    return rows


def _scene_segment_bounds(predictions: List[dict], index: int) -> Tuple[int, int]:
    start = int(index)
    while start > 0 and not bool(predictions[start].get("scene_cut_from_previous", False)):
        start -= 1
    end = int(index) + 1
    while end < len(predictions) and not bool(predictions[end].get("scene_cut_from_previous", False)):
        end += 1
    return start, end


def smooth_window_predictions(predictions: List[dict]) -> List[dict]:
    if not predictions:
        return []
    smoothing_window = int(getattr(config, "SMOOTHING_WINDOW", 1))
    if smoothing_window <= 1:
        return [dict(row) for row in predictions]

    smoothed = [dict(row) for row in predictions]
    temporal = np.asarray([row["temporal_unsafe_prob"] for row in predictions], dtype=np.float32)
    frame = np.asarray([row["frame_clip_risk"] for row in predictions], dtype=np.float32)
    radius = max(0, smoothing_window // 2)
    for i, row in enumerate(smoothed):
        seg_lo, seg_hi = _scene_segment_bounds(predictions, i)
        lo = max(seg_lo, i - radius)
        hi = min(seg_hi, i + radius + 1)
        row["temporal_unsafe_prob"] = float(np.mean(temporal[lo:hi]))
        row["frame_clip_risk"] = float(np.mean(frame[lo:hi]))
        fused = fuse_window_scores(row["temporal_unsafe_prob"], row["frame_clip_risk"])
        row["fused_score"] = float(fused["fused_score"])
        row["unsafe"] = bool(fused["unsafe"])
    return smoothed


def _has_scene_cut_between(predictions: List[dict], previous_row: dict, current_row: dict, index_by_window: dict) -> bool:
    if not bool(config.ENABLE_SCENE_CUT_BOUNDARY):
        return False
    previous_index = int(index_by_window.get(int(previous_row.get("window_index", -1)), 0))
    current_index = int(index_by_window.get(int(current_row.get("window_index", -1)), 0))
    lo = min(previous_index, current_index) + 1
    hi = max(previous_index, current_index) + 1
    for idx in range(lo, min(hi, len(predictions))):
        if bool(predictions[idx].get("scene_cut_from_previous", False)):
            return True
    return False


def _apply_segment_padding(segment: dict, video_duration: Optional[float]) -> dict:
    duration = float(video_duration) if video_duration is not None else None
    pre_pad = max(0.0, float(config.PRE_PAD_SECONDS))
    post_pad = max(0.0, float(config.POST_PAD_SECONDS))
    original_start = float(segment["start"])
    original_end = float(segment["end"])
    segment["unpadded_start"] = original_start
    segment["unpadded_end"] = original_end
    segment["start"] = max(0.0, original_start - pre_pad)
    padded_end = original_end + post_pad
    if duration is not None and duration > 0:
        padded_end = min(duration, padded_end)
    segment["end"] = max(float(segment["start"]), float(padded_end))
    return segment


def merge_unsafe_segments(predictions: List[dict], video_duration: Optional[float] = None) -> List[dict]:
    unsafe_rows = [row for row in predictions if row.get("unsafe")]
    if not unsafe_rows:
        return []

    segments: List[dict] = []
    previous_unsafe_row = None
    index_by_window = {
        int(row.get("window_index", idx)): idx
        for idx, row in enumerate(predictions)
    }
    for row in unsafe_rows:
        gap = float(row["start"]) - float(segments[-1]["end"]) if segments else None
        crosses_cut = bool(
            previous_unsafe_row is not None
            and _has_scene_cut_between(predictions, previous_unsafe_row, row, index_by_window)
        )
        should_start_new = (
            not segments
            or gap is None
            or gap > float(config.MERGE_GAP_SECONDS)
            or crosses_cut
        )
        if should_start_new:
            segments.append(
                {
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "max_temporal_unsafe_prob": float(row["temporal_unsafe_prob"]),
                    "max_frame_clip_risk": float(row["frame_clip_risk"]),
                    "max_fused_score": float(row["fused_score"]),
                    "mean_fused_score": float(row["fused_score"]),
                    "scene_cut_boundary_before": bool(crosses_cut),
                    "_scores": [float(row["fused_score"])],
                }
            )
        else:
            seg = segments[-1]
            seg["end"] = max(float(seg["end"]), float(row["end"]))
            seg["max_temporal_unsafe_prob"] = max(float(seg["max_temporal_unsafe_prob"]), float(row["temporal_unsafe_prob"]))
            seg["max_frame_clip_risk"] = max(float(seg["max_frame_clip_risk"]), float(row["frame_clip_risk"]))
            seg["max_fused_score"] = max(float(seg["max_fused_score"]), float(row["fused_score"]))
            seg["_scores"].append(float(row["fused_score"]))
            seg["mean_fused_score"] = float(np.mean(seg["_scores"]))
        previous_unsafe_row = row

    for seg in segments:
        seg.pop("_scores", None)
        _apply_segment_padding(seg, video_duration)
    return segments
