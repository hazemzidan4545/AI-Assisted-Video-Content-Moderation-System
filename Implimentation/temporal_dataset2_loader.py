import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from pipelines.roi_detector import extract_person_crop
except ModuleNotFoundError:
    from Implimentation.pipelines.roi_detector import extract_person_crop


_ROI_CACHE: "OrderedDict[Tuple[str, int, float, int], np.ndarray]" = OrderedDict()
_ROI_CACHE_TOTAL_BYTES = 0


def _roi_cache_get(key: Tuple[str, int, float, int]) -> Optional[np.ndarray]:
    cached = _ROI_CACHE.get(key)
    if cached is None:
        return None
    _ROI_CACHE.move_to_end(key)
    return cached.copy()


def _roi_cache_put(
    key: Tuple[str, int, float, int],
    crop: np.ndarray,
    max_size: int,
    max_total_bytes: int,
) -> None:
    global _ROI_CACHE_TOTAL_BYTES

    if max_size <= 0 or max_total_bytes <= 0:
        return

    crop_nbytes = int(getattr(crop, "nbytes", 0))
    if crop_nbytes <= 0 or crop_nbytes > max_total_bytes:
        return

    old = _ROI_CACHE.get(key)
    if old is not None:
        _ROI_CACHE_TOTAL_BYTES -= int(getattr(old, "nbytes", 0))

    _ROI_CACHE[key] = crop.copy()
    _ROI_CACHE_TOTAL_BYTES += crop_nbytes
    _ROI_CACHE.move_to_end(key)

    while len(_ROI_CACHE) > max_size or _ROI_CACHE_TOTAL_BYTES > max_total_bytes:
        _, removed = _ROI_CACHE.popitem(last=False)
        _ROI_CACHE_TOTAL_BYTES -= int(getattr(removed, "nbytes", 0))

    if _ROI_CACHE_TOTAL_BYTES < 0:
        _ROI_CACHE_TOTAL_BYTES = 0


def get_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def sample_frame_indices(total_frames: int, seq_len: int, random_sampling: bool = True) -> np.ndarray:
    total_frames = int(total_frames)
    seq_len = int(seq_len)

    if total_frames <= 0 or seq_len <= 0:
        return np.asarray([], dtype=np.int64)

    if total_frames <= seq_len:
        return np.linspace(0, total_frames - 1, num=seq_len).astype(np.int64)

    if not random_sampling:
        return np.linspace(0, total_frames - 1, num=seq_len).astype(np.int64)

    # Sample one frame per temporal segment so each training clip spans the full video.
    bin_edges = np.linspace(0, total_frames, num=seq_len + 1).astype(np.int64)
    indices: List[int] = []
    for i in range(seq_len):
        lo = int(bin_edges[i])
        hi = max(lo + 1, int(bin_edges[i + 1]))
        hi = min(hi, total_frames)
        if hi <= lo:
            idx = min(lo, total_frames - 1)
        else:
            idx = int(np.random.randint(lo, hi))
        indices.append(idx)
    return np.asarray(indices, dtype=np.int64)


def load_dataset2_from_directory_fallback(path: str, seq_len: int = 8) -> List[Dict[str, Any]]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    rows: List[Dict[str, Any]] = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if os.path.splitext(name)[1].lower() not in exts:
                continue
            n = name.lower()
            if "medium" in n:
                label = 1
            elif "extreme" in n or "nude" in n or "porn" in n:
                label = 2
            else:
                label = 0
            video_path = os.path.join(root, name)
            video_group = os.path.relpath(root, path)
            rows.append(
                {
                    "frames": {
                        "video_path": video_path,
                        "start_idx": 0,
                        "end_idx": None,
                    },
                    "label": int(label),
                    "video_id": video_group,
                    "source": "dataset2",
                }
            )
    return rows


def load_frames_fallback(
    frames_meta: Any,
    seq_len: int = 8,
    random_sampling: bool = True,
    use_roi: bool = False,
    roi_conf: float = 0.25,
    roi_inference_size: Optional[int] = 320,
    roi_cache_size: int = 512,
    roi_cache_max_bytes: int = 128 * 1024 * 1024,
) -> Optional[List[Image.Image]]:
    if isinstance(frames_meta, dict):
        video_path = frames_meta.get("video_path")
        if not video_path or not os.path.exists(video_path):
            return None
    elif isinstance(frames_meta, str):
        video_path = frames_meta
        if not os.path.exists(video_path):
            return None
    else:
        return None

    total_frames = get_video_frame_count(video_path)
    if total_frames <= 0:
        return None

    indices = sample_frame_indices(
        total_frames=total_frames,
        seq_len=int(seq_len),
        random_sampling=bool(random_sampling),
    ).tolist()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    out: List[Image.Image] = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if use_roi:
            cache_key = (
                str(video_path),
                int(i),
                round(float(roi_conf), 4),
                int(roi_inference_size or 0),
            )
            crop = _roi_cache_get(cache_key)
            if crop is None:
                crop = extract_person_crop(
                    frame,
                    conf=float(roi_conf),
                    inference_size=int(roi_inference_size) if roi_inference_size else None,
                )
                _roi_cache_put(
                    cache_key,
                    crop,
                    int(roi_cache_size),
                    int(roi_cache_max_bytes),
                )
            frame = crop
        out.append(Image.fromarray(frame))

    cap.release()
    return out


class TemporalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        transform,
        load_frames_fn,
        clip_length: int = 8,
        max_retries: int = 6,
        consistent_transform: bool = True,
    ):
        self.data = data
        self.transform = transform
        self.load_frames_fn = load_frames_fn
        self.clip_length = int(clip_length)
        self.max_retries = int(max_retries)
        self.consistent_transform = bool(consistent_transform)

        self.by_label = {0: [], 1: [], 2: []}
        for rec in data:
            lbl = int(rec.get("label", -1))
            if lbl in self.by_label:
                self.by_label[lbl].append(rec)

    def __len__(self) -> int:
        return len(self.data)

    def _make_clip(self, item: Dict[str, Any]) -> Optional[torch.Tensor]:
        frames = self.load_frames_fn(item["frames"])
        if frames is None:
            return None

        if torch.is_tensor(frames):
            from torchvision.transforms.functional import to_pil_image

            frame_list = [to_pil_image(frames[i].cpu()) for i in range(frames.shape[0])]
        else:
            frame_list = list(frames)

        frame_list = frame_list[: self.clip_length]
        if len(frame_list) == 0:
            return None
        if len(frame_list) < self.clip_length:
            frame_list.extend([frame_list[-1]] * (self.clip_length - len(frame_list)))

        if self.transform is None:
            return torch.stack([torch.from_numpy(np.asarray(f)) for f in frame_list])

        if not self.consistent_transform:
            return torch.stack([self.transform(f) for f in frame_list])

        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        seed = random.randint(0, 2**31 - 1)

        transformed_frames = []
        try:
            for frame in frame_list:
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed)
                transformed_frames.append(self.transform(frame))
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)

        return torch.stack(transformed_frames)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
        item = self.data[idx]
        lbl = int(item["label"])

        for attempt in range(self.max_retries):
            candidate = item if attempt == 0 else random.choice(self.by_label.get(lbl, [item]))
            clip = self._make_clip(candidate)
            if clip is not None:
                return clip, int(candidate["label"])

        return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    frames, labels = zip(*batch)
    return torch.stack(frames), torch.tensor(labels, dtype=torch.long)
