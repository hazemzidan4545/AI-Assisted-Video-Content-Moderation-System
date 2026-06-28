import os
import pickle
from collections import Counter

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data.dataloader import default_collate
from torchvision import datasets

try:
    import cv2
except ImportError:
    cv2 = None


def fast_transform_rgb224(img):
    """Convert PIL/NumPy RGB-like image to CHW float tensor in [0, 1]."""
    if cv2 is None:
        raise ImportError("opencv-python is required for fast_transform_rgb224")

    # ImageFolder yields PIL.Image by default; LMDB path already yields NumPy.
    if isinstance(img, Image.Image):
        img = np.array(img)
    elif not isinstance(img, np.ndarray):
        img = np.asarray(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = np.ascontiguousarray(img)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div(255.0)
    return img


class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError, ValueError):
            return None


def collate_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform

        # Live LMDB handles are intentionally not kept pickled.
        self.env = None
        self.txn = None
        self._lmdb_error = ()
        self.length = 0

        # Match ImageFolder-like API so downstream cells can stay generic.
        self.class_to_idx = self._load_class_to_idx()
        self.targets = []
        self._label_counts = Counter()

        self._read_metadata()

    def _open_env(self):
        import lmdb

        if self.env is None:
            # Linux generally benefits from readahead disabled for large random-read datasets.
            linux_default_readahead = False if os.name != "nt" else True
            readahead_env = os.environ.get("LMDB_READAHEAD")
            if readahead_env is None:
                use_readahead = linux_default_readahead
            else:
                use_readahead = readahead_env.strip().lower() in {"1", "true", "yes", "on"}

            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=use_readahead,
                meminit=False,
                max_readers=256,
            )
            self.txn = self.env.begin(write=False, buffers=True)
            self._lmdb_error = (lmdb.Error,)

    def _close_env(self):
        if self.txn is not None:
            self.txn = None
        if self.env is not None:
            self.env.close()
            self.env = None

    def __getstate__(self):
        # Remove non-picklable LMDB handles for multiprocessing spawn on Windows.
        state = self.__dict__.copy()
        state["env"] = None
        state["txn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        self._close_env()

    def close(self):
        """Explicitly release LMDB resources for notebook reruns."""
        self._close_env()

    def __enter__(self):
        self._open_env()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_env()
        return False

    def _load_class_to_idx(self):
        pkl_path = os.path.join(os.path.dirname(self.lmdb_path), "class_to_idx.pkl")
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    mapping = pickle.load(f)
                if isinstance(mapping, dict):
                    return mapping
            except Exception:
                pass
        return {}

    def _read_metadata(self):
        self._open_env()

        len_bytes = self.txn.get(b"__len__")
        class_to_idx_bytes = self.txn.get(b"__class_to_idx__")
        label_counts_bytes = self.txn.get(b"__label_counts__")

        if len_bytes is None:
            self._close_env()
            raise ValueError(
                "LMDB metadata missing (__len__). Recreate LMDB using the updated creation cell."
            )

        self.length = int(pickle.loads(len_bytes))

        if class_to_idx_bytes is not None:
            try:
                mapping = pickle.loads(class_to_idx_bytes)
                if isinstance(mapping, dict):
                    self.class_to_idx = mapping
            except Exception:
                pass

        if label_counts_bytes is not None:
            try:
                counts = pickle.loads(label_counts_bytes)
                if isinstance(counts, dict):
                    self._label_counts = Counter({int(k): int(v) for k, v in counts.items()})
            except Exception:
                pass

        # Keep empty by default to avoid per-sample metadata scan at startup.
        self.targets = []

    def get_label_counts(self):
        return dict(self._label_counts)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if cv2 is None:
            raise ImportError("opencv-python is required for LMDBDataset")

        try:
            self._open_env()
            key = f"{idx:08d}".encode()
            data = self.txn.get(key)

            if data is None:
                return None

            record = pickle.loads(data)
            if isinstance(record, dict):
                img_obj = record.get("img")
                label = record.get("label", 0)
            else:
                img_obj, label, _ = record

            if isinstance(img_obj, (bytes, bytearray, memoryview)):
                img_arr = np.frombuffer(img_obj, dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            else:
                img = img_obj
            if img is None:
                return None

            # Convert BGR -> RGB and keep as numpy for fast OpenCV-native transforms.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(img)
            else:
                img = fast_transform_rgb224(img)

            # Keep labels exactly as stored in LMDB metadata/samples.
            label = int(label)
            return img, label
        except self._lmdb_error:
            return None
        except (pickle.UnpicklingError, ValueError, TypeError, cv2.error):
            # Keep workers alive by skipping malformed records.
            return None


class RemapLabelDataset(torch.utils.data.Dataset):
    """
    Wrapper that remaps labels from one dataset to another.
    
    Used to align binary class indices between train/val/test datasets
    that may have different class_to_idx mappings (e.g., nude: 0 in train
    but nude: 1 in val). This ensures consistent evaluation metrics across splits.
    
    Args:
        base_ds: Base dataset (LMDBDataset, SafeImageFolder, etc.)
        remap_dict: Dict mapping {old_label: new_label}. If None, no remapping.
    
    Example:
        val_remap = {0: 1, 1: 0}  # Swap labels
        wrapped = RemapLabelDataset(val_dataset, remap_dict=val_remap)
    """
    def __init__(self, base_ds, remap_dict=None):
        self.base = base_ds
        self.remap = remap_dict

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        if item is None:
            return None
        img, y = item
        y = int(y)
        if self.remap is not None and y in self.remap:
            y = self.remap[y]
        return img, y


class DatasetConfig:
    """
    Centralized configuration container for train/val/test datasets.
    
    Encapsulates all dataset metadata and statistics, making it easy to pass
    consistent configuration across phases without relying on notebook globals.
    
    Attributes:
        train: LMDBDataset or SafeImageFolder for training
        val: LMDBDataset or SafeImageFolder for validation
        test: LMDBDataset or SafeImageFolder for testing (optional)
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing (optional)
        batch_size: Batch size used
        class_counts: Dict of {class_idx: count} total counts in train
        class_to_idx: Dict of {class_name: class_idx}
        device: Device (cuda/cpu) used
        seed: Random seed for reproducibility
    """
    def __init__(self, train, val, test=None, train_loader=None, val_loader=None,
                 test_loader=None, batch_size=56, class_counts=None, class_to_idx=None,
                 device='cpu', seed=42):
        self.train = train
        self.val = val
        self.test = test
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.class_counts = class_counts or {}
        self.class_to_idx = class_to_idx or {}
        self.device = device
        self.seed = seed
    
    def summary(self):
        """Return formatted summary string of dataset statistics."""
        lines = ["\nDataset Configuration Summary:"]
        lines.append(f"  Train: {len(self.train)}")
        lines.append(f"  Val:   {len(self.val)}")
        if self.test is not None:
            lines.append(f"  Test:  {len(self.test)}")
        lines.append(f"  Batch Size: {self.batch_size}")
        lines.append(f"  Device: {self.device}")
        return "\n".join(lines)
    
    def __repr__(self):
        return (f"DatasetConfig(train={len(self.train)}, val={len(self.val)}, "
                f"test={len(self.test) if self.test else 'None'}, batch={self.batch_size})")
