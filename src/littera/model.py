from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class ModelConfig:
    n_labels: int
    backbone: str = "resnet"
    context: str = "bilstm"
    height: int = 96
    channels: list[int] = field(default_factory=lambda: [64, 128, 192, 256])
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.1
    stride: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**d)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: tuple[int, int]):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity()
        if cin != cout or stride != (1, 1):
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + self.shortcut(x))


class ResNetBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        c = config.channels
        self.stem = nn.Sequential(nn.Conv2d(1, c[0], 3, padding=1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True))
        strides = [(2, 2), (2, 2), (2, 1), (2, 1)]
        stages, cin = [], c[0]
        for cout, stride in zip(c, strides):
            stages.append(nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, (1, 1))))
            cin = cout
        self.stages = nn.Sequential(*stages)
        rows = config.height // 16
        self.row_embedding = nn.Parameter(torch.zeros(1, cin, rows, 1))
        self.row_score = nn.Conv2d(cin, 1, 1)
        self.out_dim = cin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stages(self.stem(x))
        weights = torch.softmax(self.row_score(features + self.row_embedding), dim=2)
        return (features * weights).sum(2)


class VGSLBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        def conv(cin: int, cout: int, kh: int, kw: int) -> list[nn.Module]:
            return [nn.Conv2d(cin, cout, (kh, kw), padding=(kh // 2, kw // 2)), nn.ReLU(inplace=True), nn.Dropout2d(0.1)]

        self.layers = nn.Sequential(
            *conv(1, 32, 3, 13), nn.MaxPool2d((2, 2)),
            *conv(32, 32, 3, 13), nn.MaxPool2d((2, 2)),
            *conv(32, 64, 3, 9), nn.MaxPool2d((2, 1)),
            *conv(64, 64, 3, 9),
        )
        self.out_dim = 64 * (config.height // 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.layers(x)
        b, c, h, t = features.shape
        return features.reshape(b, c * h, t)


class BiLSTMContext(nn.Module):
    def __init__(self, din: int, hidden: int, layers: int, dropout: float, between: float = 0.0, after: float = 0.0):
        super().__init__()
        sizes = [din] + [2 * hidden] * (layers - 1)
        self.lstms = nn.ModuleList(nn.LSTM(size, hidden, batch_first=True, bidirectional=True) for size in sizes)
        self.between = nn.Dropout(max(dropout, between))
        self.after = nn.Dropout(after) if after else nn.Identity()
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        total = y.shape[1]
        for i, lstm in enumerate(self.lstms):
            if i:
                y = self.between(y)
            packed = pack_padded_sequence(y, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            y, _ = pad_packed_sequence(lstm(packed)[0], batch_first=True, total_length=total)
        return self.after(y).transpose(1, 2)


class LocalContext(nn.Module):
    def __init__(self, din: int, width: int, dilations: tuple[int, ...] = (1, 2, 4), dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(din, width, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv1d(width, width, 3, padding=d, dilation=d), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(dropout))
            for d in dilations
        )
        self.out_dim = width

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        y = self.inp(x)
        for block in self.blocks:
            y = y + block(y)
        return y


class LineModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.height % 16:
            raise ValueError("height must be a multiple of 16")
        self.backbone = ResNetBackbone(config) if config.backbone == "resnet" else VGSLBackbone(config)
        din = self.backbone.out_dim
        if config.context == "bilstm":
            self.context = BiLSTMContext(din, config.hidden, config.layers, config.dropout)
        elif config.context == "kraken":
            self.context = BiLSTMContext(din, 200, 3, 0.0, between=0.1, after=0.5)
        elif config.context == "local":
            self.context = LocalContext(din, 2 * config.hidden, dropout=config.dropout)
        else:
            raise ValueError(f"unknown context {config.context}")
        dim = self.context.out_dim
        self.char_head = nn.Conv1d(dim, config.n_labels, 1)
        self.boundary_head = nn.Conv1d(dim, 1, 3, padding=1)
        self.register_buffer("log_prior", torch.full((config.n_labels,), -float(torch.log(torch.tensor(float(config.n_labels))))))

    def frames(self, widths: torch.Tensor) -> torch.Tensor:
        return torch.div(widths + self.config.stride - 1, self.config.stride, rounding_mode="floor")

    def forward(self, images: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = self.frames(widths)
        features = self.backbone(images)
        lengths = lengths.clamp(max=features.shape[-1])
        context = self.context(features, lengths)
        logits = self.char_head(context).transpose(1, 2)
        boundary = self.boundary_head(context).squeeze(1)
        return logits, boundary, lengths


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


MODEL_SCHEMA = "littera/model/1"


def save_model(directory, model: nn.Module, config: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps({"schema": MODEL_SCHEMA, **config}, ensure_ascii=False, indent=1), encoding="utf-8")


def model_directory(path) -> Path:
    path = Path(path)
    if (path / "model.safetensors").exists():
        return path
    for name in ("best", "last"):
        if (path / name / "model.safetensors").exists():
            return path / name
    raise FileNotFoundError(f"no model.safetensors under {path}")


def load_model(path, device="cpu") -> tuple[LineModel, dict]:
    directory = model_directory(path)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = LineModel(ModelConfig.from_dict(config["model"]))
    model.load_state_dict(load_file(str(directory / "model.safetensors"), device=str(device)))
    return model.to(device).eval(), config
