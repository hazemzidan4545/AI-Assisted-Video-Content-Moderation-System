"""Frame-level ConvNeXt risk scoring utilities."""

from __future__ import annotations

from typing import Iterable, List

import cv2
import numpy as np
import torch

from . import config


def preprocess_frame_for_convnext(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert an OpenCV BGR frame to a ConvNeXt input tensor in [0, 1]."""

    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        raise ValueError("Empty frame passed to preprocess_frame_for_convnext.")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def score_frames_with_convnext(
    frames_bgr: Iterable[np.ndarray],
    image_model,
    device: str | torch.device,
    batch_size: int | None = None,
) -> List[dict]:
    frames = list(frames_bgr)
    if not frames:
        return []

    batch_size = int(batch_size or config.PATCH_BATCH_SIZE)
    rows: List[dict] = []
    image_model.eval()
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            tensors = torch.stack([preprocess_frame_for_convnext(frame) for frame in chunk]).to(device)
            probs = torch.softmax(image_model(tensors), dim=1).detach().cpu().numpy()
            for prob in probs:
                prob_normal = float(prob[0])
                prob_sexy = float(prob[1])
                prob_nude = float(prob[2])
                risk = float(prob_nude + 0.5 * prob_sexy)
                rows.append(
                    {
                        "prob_normal": prob_normal,
                        "prob_sexy": prob_sexy,
                        "prob_nude": prob_nude,
                        "risk": risk,
                    }
                )
    return rows


def compute_clip_frame_risk(frame_risks: Iterable[float]) -> dict:
    risks = np.asarray(list(frame_risks), dtype=np.float32)
    if risks.size == 0:
        return {"frame_clip_risk": 0.0, "frame_max_risk": 0.0}
    topk = max(1, min(int(config.FRAME_TOPK), int(risks.size)))
    top_values = np.sort(risks)[-topk:]
    return {
        "frame_clip_risk": float(np.mean(top_values)),
        "frame_max_risk": float(np.max(risks)),
    }
