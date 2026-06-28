import timm
import torch
import torch.nn as nn


def _unwrap_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("best_model_state_dict"), dict):
            return checkpoint["best_model_state_dict"]
        if isinstance(checkpoint.get("model_state_dict"), dict):
            return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint format")


def _strip_prefix_if_present(state_dict, prefix: str):
    prefix = str(prefix)
    out = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
    return out


def _filter_matching_state_dict(source_state, target_state):
    matched = {}
    for key, value in source_state.items():
        if key in target_state and getattr(value, "shape", None) == target_state[key].shape:
            matched[key] = value
    return matched


class TemporalLSTMHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 3, hidden_dim: int = 192, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.classifier(self.dropout(hidden[-1]))


class TemporalAttentionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 192,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
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
        head_type: str = "lstm",
        freeze_encoder: bool = False,
        num_classes: int = 3,
        encoder_checkpoint_path: str = "",
    ):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        self.embed_dim = int(getattr(self.encoder, "num_features", 768))
        self.freeze_encoder = bool(freeze_encoder)
        self.head_type = str(head_type).strip().lower()
        self.encoder_checkpoint_path = str(encoder_checkpoint_path or "").strip()

        if self.head_type == "lstm":
            self.head = TemporalLSTMHead(self.embed_dim, num_classes=num_classes)
        elif self.head_type == "attention":
            self.head = TemporalAttentionHead(
                embed_dim=self.embed_dim,
                hidden_dim=192,
                num_classes=num_classes,
                dropout=0.3,
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

        if self.encoder_checkpoint_path:
            self._load_encoder_checkpoint(self.encoder_checkpoint_path)

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _load_encoder_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        raw_state = _unwrap_checkpoint_state(checkpoint)
        encoder_state = self.encoder.state_dict()

        candidates = [
            raw_state,
            _strip_prefix_if_present(raw_state, "encoder."),
            _strip_prefix_if_present(raw_state, "model."),
            _strip_prefix_if_present(raw_state, "module."),
        ]

        matched_state = {}
        for candidate in candidates:
            matched_state = _filter_matching_state_dict(candidate, encoder_state)
            if matched_state:
                break

        if not matched_state:
            raise RuntimeError(f"No matching encoder weights found in checkpoint: {checkpoint_path}")

        self.encoder.load_state_dict(matched_state, strict=False)

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
