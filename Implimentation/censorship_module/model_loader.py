"""Robust local model loading for the censorship module."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import timm
import torch
import torch.nn as nn

from . import config


PACKAGE_DIR = Path(__file__).resolve().parent
IMPLEMENTATION_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = IMPLEMENTATION_DIR.parent


def _ensure_import_paths() -> None:
    for path in (IMPLEMENTATION_DIR, PROJECT_ROOT, Path.cwd()):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a checkpoint from common project-relative locations."""

    raw = Path(path).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                IMPLEMENTATION_DIR / raw,
                PROJECT_ROOT / raw,
                PACKAGE_DIR / raw,
            ]
        )

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in unique_candidates)
    raise FileNotFoundError(f"Checkpoint not found: {path}\nTried:\n{tried}")


def _unwrap_checkpoint_state(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("best_model_state_dict", "model_state_dict", "state_dict"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
        if all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise TypeError(
        "Unsupported checkpoint format. Expected raw state_dict or dict with "
        "'best_model_state_dict', 'model_state_dict', or 'state_dict'."
    )


def _strip_prefix_if_present(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state):
        return {}
    return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}


def _candidate_states(state: Dict[str, torch.Tensor]) -> Iterable[Tuple[str, Dict[str, torch.Tensor]]]:
    yield "raw", state
    for prefix in ("module.", "model.", "encoder."):
        stripped = _strip_prefix_if_present(state, prefix)
        if stripped:
            yield f"strip:{prefix}", stripped


def _matching_state(source: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in source.items()
        if key in target and getattr(value, "shape", None) == target[key].shape
    }


def _load_checked_state(
    model: nn.Module,
    raw_state: Dict[str, torch.Tensor],
    *,
    min_match_ratio: float,
    required_prefixes: Tuple[str, ...] = (),
    allow_head_shape_mismatch: bool = False,
) -> Tuple[int, int]:
    target = model.state_dict()
    best_name = ""
    best_matched: Dict[str, torch.Tensor] = {}
    for name, candidate in _candidate_states(raw_state):
        matched = _matching_state(candidate, target)
        if len(matched) > len(best_matched):
            best_name = name
            best_matched = matched

    if not best_matched:
        raise RuntimeError("No matching checkpoint keys found for model.")

    target_key_count = len(target)
    match_ratio = len(best_matched) / max(1, target_key_count)
    if match_ratio < float(min_match_ratio):
        raise RuntimeError(
            f"Too few checkpoint keys matched ({len(best_matched)}/{target_key_count}, "
            f"ratio={match_ratio:.3f}, candidate={best_name}). Refusing to continue."
        )

    missing_required = []
    for prefix in required_prefixes:
        required = [key for key in target if key.startswith(prefix)]
        missing_required.extend([key for key in required if key not in best_matched])
    if missing_required:
        preview = "\n".join(f"  - {key}" for key in missing_required[:20])
        raise RuntimeError(
            f"Required checkpoint keys were not loaded for prefixes {required_prefixes}:\n{preview}"
        )

    incompatible = [
        key
        for key, value in raw_state.items()
        if key in target and getattr(value, "shape", None) != target[key].shape
    ]
    if incompatible and not allow_head_shape_mismatch:
        preview = "\n".join(f"  - {key}: ckpt={tuple(raw_state[key].shape)} target={tuple(target[key].shape)}" for key in incompatible[:20])
        raise RuntimeError(f"Incompatible checkpoint tensor shapes:\n{preview}")

    load_result = model.load_state_dict(best_matched, strict=False)
    print(f"Loaded {len(best_matched)}/{target_key_count} keys using candidate '{best_name}'.")
    if load_result.missing_keys:
        print("Missing keys:")
        for key in load_result.missing_keys[:30]:
            print(f"  - {key}")
    if load_result.unexpected_keys:
        print("Unexpected keys:")
        for key in load_result.unexpected_keys[:30]:
            print(f"  - {key}")
    return len(best_matched), target_key_count


@contextlib.contextmanager
def _force_timm_pretrained_false():
    """Prevent accidental pretrained downloads inside existing project constructors."""

    original_create_model = timm.create_model

    def create_model_no_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = create_model_no_pretrained
    try:
        yield
    finally:
        timm.create_model = original_create_model


def _import_project_temporal_class():
    _ensure_import_paths()
    errors = []
    for import_path in ("Models.temporal_model", "Implimentation.Models.temporal_model"):
        try:
            module = __import__(import_path, fromlist=["ConvNeXtTemporalClassifier"])
            return module.ConvNeXtTemporalClassifier, import_path
        except Exception as exc:  # import can fail from notebook-style cwd differences
            errors.append(f"{import_path}: {exc}")
    return None, "\n".join(errors)


def _local_temporal_class():
    class TemporalAttentionHead(nn.Module):
        def __init__(self, embed_dim: int, hidden_dim: int = 192, num_classes: int = 2, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.attention = nn.Linear(hidden_dim * 2, 1)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)
            self.last_attention_weights = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
            self.last_attention_weights = weights.detach()
            context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
            return self.classifier(self.dropout(context))

    class ConvNeXtTemporalClassifier(nn.Module):
        def __init__(
            self,
            model_name: str = "convnext_small",
            head_type: str = "attention",
            freeze_encoder: bool = True,
            num_classes: int = 2,
            encoder_checkpoint_path: str = "",
        ):
            super().__init__()
            if head_type != "attention":
                raise ValueError("Local fallback only supports head_type='attention'.")
            self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="avg")
            self.embed_dim = int(getattr(self.encoder, "num_features", 768))
            self.freeze_encoder = bool(freeze_encoder)
            self.head_type = str(head_type)
            self.head = TemporalAttentionHead(self.embed_dim, num_classes=num_classes)
            if self.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            if self.freeze_encoder:
                with torch.no_grad():
                    emb = self.encoder(x)
            else:
                emb = self.encoder(x)
            emb = emb.reshape(b, t, self.embed_dim)
            return self.head(emb)

    return ConvNeXtTemporalClassifier


def load_binary_temporal_model(
    device: str | torch.device,
    checkpoint_path: str | Path = config.BINARY_TEMPORAL_CKPT,
    *,
    return_path: bool = False,
):
    resolved = resolve_checkpoint_path(checkpoint_path)
    temporal_class, import_info = _import_project_temporal_class()
    if temporal_class is None:
        print("Project ConvNeXtTemporalClassifier import failed; using local compatible fallback.")
        print(import_info)
        temporal_class = _local_temporal_class()
    else:
        print(f"Using project ConvNeXtTemporalClassifier from {import_info}.")

    with _force_timm_pretrained_false():
        model = temporal_class(
            model_name="convnext_small",
            head_type="attention",
            freeze_encoder=True,
            num_classes=2,
            encoder_checkpoint_path="",
        )

    checkpoint = torch.load(resolved, map_location="cpu")
    raw_state = _unwrap_checkpoint_state(checkpoint)
    _load_checked_state(
        model,
        raw_state,
        min_match_ratio=0.90,
        required_prefixes=("head.",),
        allow_head_shape_mismatch=False,
    )
    model.to(device)
    model.eval()
    model.resolved_checkpoint_path = str(resolved)
    return (model, resolved) if return_path else model


def load_convnext_image_classifier(
    device: str | torch.device,
    checkpoint_path: str | Path = config.IMAGE_BACKBONE_CKPT,
    *,
    return_path: bool = False,
):
    resolved = resolve_checkpoint_path(checkpoint_path)
    model = timm.create_model("convnext_small", pretrained=False, num_classes=3)
    checkpoint = torch.load(resolved, map_location="cpu")
    raw_state = _unwrap_checkpoint_state(checkpoint)
    _load_checked_state(
        model,
        raw_state,
        min_match_ratio=0.85,
        required_prefixes=("head.",),
        allow_head_shape_mismatch=True,
    )
    model.to(device)
    model.eval()
    model.resolved_checkpoint_path = str(resolved)
    return (model, resolved) if return_path else model
