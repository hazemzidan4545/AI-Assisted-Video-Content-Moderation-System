from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None


_MODEL = None
_MODEL_NAME = "yolov8n.pt"


def _get_model(model_name: str = _MODEL_NAME):
    global _MODEL
    if YOLO is None:
        raise ModuleNotFoundError(
            "ultralytics is required for ROI detection. Install it with `pip install ultralytics`."
        )
    if _MODEL is None:
        _MODEL = YOLO(model_name)
        try:
            _MODEL.to("cpu")
        except Exception:
            # If placement is not supported in this ultralytics build, inference call kwargs still request CPU.
            pass
    return _MODEL


def extract_person_crop(
    frame: np.ndarray,
    conf: float = 0.25,
    inference_size: Optional[int] = 320,
    device: str = "cpu",
) -> np.ndarray:
    if frame is None or getattr(frame, "size", 0) == 0:
        return frame

    original = frame
    input_frame = frame
    scale_x = 1.0
    scale_y = 1.0

    if inference_size and inference_size > 0:
        height, width = frame.shape[:2]
        if max(height, width) > inference_size:
            if height >= width:
                resized_h = int(inference_size)
                resized_w = max(1, int(round(width * inference_size / height)))
            else:
                resized_w = int(inference_size)
                resized_h = max(1, int(round(height * inference_size / width)))
            input_frame = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
            scale_x = width / float(resized_w)
            scale_y = height / float(resized_h)

    model = _get_model()
    try:
        results = model(input_frame, classes=[0], conf=float(conf), verbose=False, device=device)[0]
    except TypeError:
        # Backward compatibility for ultralytics builds that do not expose `device` in __call__.
        results = model(input_frame, classes=[0], conf=float(conf), verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return original

    boxes = results.boxes.xyxy.detach().cpu().numpy()
    if boxes.size == 0:
        return original

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))
    x1, y1, x2, y2 = boxes[best_idx].astype(int)

    height, width = original.shape[:2]
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return original

    crop = original[y1:y2, x1:x2]
    if crop.size == 0:
        return original

    return crop.copy()
