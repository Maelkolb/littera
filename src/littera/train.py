from __future__ import annotations

import argparse
import functools
import io
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, Sampler

from littera import graph
from littera.geometry import GeometryConfig, prepare
from littera.graph import GraphOptions
from littera.model import LineModel, ModelConfig, model_directory, parameter_count, save_model
from littera.place import DECODE, Objective
from littera.text import ANY, Codec

POOL_COLUMNS = ["uid", "source", "split", "image", "text", "polygon", "baseline", "width", "height", "gold_cuts"]


@dataclass
class Augment:
    shear: float = 0.3
    shear_deg: float = 12.0
    height_jitter: float = 0.15
    stroke: float = 0.2
    contrast: float = 0.3
    blur: float = 0.2
    noise: float = 0.2
    neighbours: float = 0.3
    cut_edges: float = 0.15


@dataclass
class TrainConfig:
    pool: list[str]
    out: str
    name: str = "littera"
    init: str = ""
    split: str = "train"
    val_split: str = "val"
    sources: list[str] | None = None
    max_lines: int = 0
    val_lines: int = 1000
    model: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)
    objective: dict = field(default_factory=dict)
    graph: dict = field(default_factory=lambda: {"spaces": "required", "garbage_ends": True, "garbage_between": False})
    decode: dict = field(default_factory=dict)
    preset: str = "markup"
    presets: dict = field(default_factory=dict)
    min_count: int = 3
    rare_count: int = 50
    mask_rare: float = 0.2
    mask_any: float = 0.01
    boundary_weight: float = 1.0
    frame_weight: float = 1.0
    steps: int = 50_000
    lr: float = 1e-3
    warmup: int = 1000
    weight_decay: float = 0.01
    clip: float = 1.0
    batch_columns: int = 40_000
    max_batch: int = 48
    workers: int = 8
    amp: str = "fp16"
    seed: int = 0
    log_every: int = 50
    val_every: int = 2000
    save_every: float = 600.0
    prior_momentum: float = 0.999
    augment: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        return cls(**d)


def pool_files(paths: list[str]) -> list[Path]:
    files = []
    for path in map(Path, paths):
        if path.is_dir():
            files += sorted(f for f in path.rglob("*.parquet") if f.name != "index.parquet")
        elif path.exists():
            files.append(path)
    return files


def read_rows(paths: list[str], splits: list[str], sources: list[str] | None = None, max_lines: int = 0, seed: int = 0) -> pa.Table:
    tables = []
    for file in pool_files(paths):
        names = pq.read_schema(file).names
        if "image" not in names or "text" not in names:
            continue
        columns = [c for c in POOL_COLUMNS if c in names]
        filters = [("split", "in", list(splits))] if "split" in names and splits else None
        table = pq.read_table(file, columns=columns, filters=filters)
        if sources and "source" in table.column_names:
            table = table.filter(pc.is_in(table["source"], pa.array(sources)))
        if table.num_rows:
            tables.append(table)
    if not tables:
        return pa.table({"text": pa.array([], pa.string()), "image": pa.array([], pa.binary())})
    table = pa.concat_tables(tables, promote_options="default")
    if max_lines and table.num_rows > max_lines:
        picks = sorted(random.Random(seed).sample(range(table.num_rows), max_lines))
        table = table.take(pa.array(picks))
    return table


def _column(table: pa.Table, name: str) -> list:
    return table.column(name).to_pylist() if name in table.column_names else [None] * table.num_rows


class LineDataset(Dataset):
    def __init__(self, table: pa.Table, codec: Codec, config: TrainConfig, train: bool):
        self.images = table.column("image").combine_chunks()
        self.texts = _column(table, "text")
        self.presets = [config.presets.get(source, config.preset) for source in _column(table, "source")]
        self.polygons = _column(table, "polygon")
        self.baselines = _column(table, "baseline")
        self.cuts = _column(table, "gold_cuts")
        widths, heights = _column(table, "width"), _column(table, "height")
        self.codec = codec
        self.config = config
        self.geometry = geometry_config(config)
        topology = Objective.from_dict(config.objective).topology
        self.options = GraphOptions(**{**config.graph, "topology": "segmental" if topology == "segmental" else "ctc"})
        self.augment = Augment(**config.augment) if train else None
        self.rng = random.Random(config.seed)
        self.estimates = [int((w or 1000) * self.geometry.height / max(1, h or self.geometry.height)) for w, h in zip(widths, heights)]

    def __len__(self) -> int:
        return len(self.texts)

    def image(self, i: int) -> Image.Image:
        return Image.open(io.BytesIO(self.images[i].as_py())).convert("L")

    def __getitem__(self, i: int) -> dict:
        image = self.image(i)
        polygon = self.polygons[i] or None
        baseline = self.baselines[i] or None
        cuts = [(c["i"], c["x"]) for c in self.cuts[i]] if self.cuts[i] else None
        core_scale = 1.0
        if self.augment:
            image, polygon, baseline, cuts, core_scale = self._augment(image, polygon, baseline, cuts)
        strip = prepare(image, polygon, baseline, self.geometry, core_scale=core_scale)
        encoded = self.codec.encode(self.texts[i], self.presets[i])
        words = [[(self._mask(label), optional) for label, optional in word] for word in encoded.graph_words()]
        stride = self.geometry.stride
        frames = strip.width // stride
        targets = frame_labels = None
        if cuts:
            positions = np.round(strip.strip_u([x for _, x in cuts]) / stride).astype(np.int64)
            targets = sorted({int(p) for p in positions if 0 <= p < frames})
            if self.options.segmental:
                frame_labels = self._frame_labels(encoded, words, dict(cuts), strip, frames)
        return {"image": strip.image, "graph": graph.build(words, self.options), "units": sum(len(w) for w in words), "targets": targets,
                "frame_labels": frame_labels, "empty": strip.empty_frames(stride),
                "labels": [[label for label, _ in word] for word in words]}

    def _frame_labels(self, encoded, words, cut_at: dict, strip, frames: int) -> np.ndarray | None:
        labels = np.zeros(frames, np.int64)
        flat = [label for word in words for label, _ in word]
        for label, k in zip(flat, encoded.unit_order()):
            unit = encoded.seg.units[k]
            if unit.i0 not in cut_at or unit.i1 not in cut_at:
                return None
            f0, f1 = (int(round(float(strip.strip_u(cut_at[i])) / self.geometry.stride)) for i in (unit.i0, unit.i1))
            labels[max(0, f0) : max(0, min(frames, f1))] = label
        return labels

    def _mask(self, label: int) -> int:
        if not self.augment or label < 2:
            return label
        rare = self.codec.count(label) < self.config.rare_count
        if (rare and self.rng.random() < self.config.mask_rare) or self.rng.random() < self.config.mask_any:
            return ANY
        return label

    def _augment(self, image, polygon, baseline, cuts):
        a, rng = self.augment, self.rng
        if rng.random() < a.cut_edges:
            w, h = image.size
            top = int(h * rng.uniform(0, 0.12)) if rng.random() < 0.5 else 0
            bottom = int(h * rng.uniform(0, 0.12)) if rng.random() < 0.5 else 0
            if h - top - bottom > 8:
                image = image.crop((0, top, w, h - bottom))
                polygon = [(x, y - top) for x, y in polygon] if polygon else polygon
                baseline = [(x, y - top) for x, y in baseline] if baseline else baseline
        if rng.random() < a.neighbours and len(self) > 1:
            image = self._neighbours(image, self.image(rng.randrange(len(self))))
        if rng.random() < a.shear:
            w, h = image.size
            tangent = math.tan(math.radians(rng.uniform(-a.shear_deg, a.shear_deg)))
            offset = abs(tangent) * h / 2
            size = (int(math.ceil(w + 2 * offset)), h)
            image = image.transform(size, Image.Transform.AFFINE, (1, -tangent, -offset + tangent * h / 2, 0, 1, 0),
                                    resample=Image.Resampling.BILINEAR, fillcolor=255)

            def move(points):
                return [(x + offset + tangent * (y - h / 2), y) for x, y in points] if points else points

            polygon, baseline = move(polygon), move(baseline)
            cuts = [(i, x + offset) for i, x in cuts] if cuts else cuts
        if rng.random() < a.stroke:
            image = image.filter(ImageFilter.MinFilter(3) if rng.random() < 0.5 else ImageFilter.MaxFilter(3))
        if rng.random() < a.blur:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.4)))
        if rng.random() < a.contrast or rng.random() < a.noise:
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            if rng.random() < a.contrast:
                low, high = rng.uniform(0, 0.35), rng.uniform(0.75, 1.0)
                pixels = low + (high - low) * pixels ** rng.uniform(0.6, 1.6)
            if rng.random() < a.noise:
                pixels = pixels + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0.02, 0.08), pixels.shape)
            image = Image.fromarray((np.clip(pixels, 0, 1) * 255).astype(np.uint8))
        core_scale = 1.0 + rng.uniform(-a.height_jitter, a.height_jitter)
        return image, polygon, baseline, cuts, core_scale

    def _neighbours(self, image: Image.Image, other: Image.Image) -> Image.Image:
        w, h = image.size
        other = other.resize((max(1, int(other.width * h / max(1, other.height))), h))
        base = np.asarray(image, dtype=np.uint8).copy()
        donor = np.asarray(other, dtype=np.uint8)
        band = int(h * self.rng.uniform(0.15, 0.35))
        x0 = self.rng.randrange(max(1, donor.shape[1] - w + 1)) if donor.shape[1] > w else 0
        piece = donor[:, x0 : x0 + w]
        span = piece.shape[1]
        if self.rng.random() < 0.6:
            base[:band, :span] = np.minimum(base[:band, :span], piece[h - band :])
        if self.rng.random() < 0.6:
            base[h - band :, :span] = np.minimum(base[h - band :, :span], piece[:band])
        return Image.fromarray(base)


class WidthBatches(Sampler):
    def __init__(self, estimates: list[int], columns: int, max_batch: int, seed: int, shuffle: bool = True):
        self.estimates, self.columns, self.max_batch, self.shuffle = estimates, columns, max_batch, shuffle
        self.rng = random.Random(seed)
        self.batches = self._plan()

    def _plan(self) -> list[list[int]]:
        noisy = [(w * (1 + (self.rng.uniform(-0.1, 0.1) if self.shuffle else 0)), i) for i, w in enumerate(self.estimates)]
        batches, current, widest = [], [], 0
        for w, i in sorted(noisy):
            widest = max(widest, w)
            if current and (len(current) >= self.max_batch or widest * (len(current) + 1) > self.columns):
                batches.append(current)
                current, widest = [], w
            current.append(i)
        if current:
            batches.append(current)
        if self.shuffle:
            self.rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches, self.batches = self.batches, self._plan()
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches)


def collate(items: list[dict], stride: int = 4) -> dict:
    height = items[0]["image"].shape[0]
    width = max(item["image"].shape[1] for item in items)
    width += (-width) % stride
    images = torch.zeros(len(items), 1, height, width)
    for b, item in enumerate(items):
        images[b, 0, :, : item["image"].shape[1]] = torch.from_numpy(item["image"])
    frames = width // stride
    targets = torch.zeros(len(items), frames)
    target_mask = torch.zeros(len(items), frames, dtype=torch.bool)
    frame_labels = torch.full((len(items), frames), -100, dtype=torch.int64)
    empty = torch.ones(len(items), frames, dtype=torch.bool)
    for b, item in enumerate(items):
        n = item["image"].shape[1] // stride
        empty[b, :n] = torch.from_numpy(item["empty"][:n])
        if item["targets"] is not None:
            target_mask[b, :n] = True
            targets[b, item["targets"]] = 1.0
        if item.get("frame_labels") is not None:
            frame_labels[b, :n] = torch.from_numpy(item["frame_labels"][:n])
    return {"images": images, "widths": torch.tensor([item["image"].shape[1] for item in items]),
            "graphs": graph.pack([item["graph"] for item in items]), "units": torch.tensor([item["units"] for item in items]),
            "targets": targets, "target_mask": target_mask, "frame_labels": frame_labels, "empty": empty,
            "labels": [item["labels"] for item in items]}


def geometry_config(config: TrainConfig) -> GeometryConfig:
    height = config.model.get("height", 96)
    return GeometryConfig.from_dict({**config.geometry, "height": height})


def model_config_dict(config: TrainConfig, codec: Codec, objective: Objective, step: int, extra: dict | None = None) -> dict:
    model = ModelConfig(n_labels=len(codec), **config.model)
    topology = "segmental" if objective.topology == "segmental" else "ctc"
    ink_prior = GraphOptions(**config.graph).ink_prior
    garbage = {"garbage": "any"} if topology == "segmental" else {}
    decode = GraphOptions(**{**DECODE.to_dict(), "ink_prior": ink_prior, **garbage, **config.decode, "topology": topology})
    return {"name": config.name, "model": model.to_dict(), "codec": codec.to_dict(), "geometry": geometry_config(config).to_dict(),
            "objective": asdict(objective), "decode": decode.to_dict(), "preset": config.preset, "presets": config.presets,
            "trained": {"step": step, **(extra or {})}}


def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


class Trainer:
    def __init__(self, config: TrainConfig, device=None, on_save: Callable[[], None] | None = None):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.on_save = on_save or (lambda: None)
        self.out = Path(config.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.objective = Objective.from_dict(config.objective)
        self.options = GraphOptions(**{**config.graph, "topology": self.objective.topology})
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        train_rows = read_rows(config.pool, [config.split], config.sources, config.max_lines, config.seed)
        if train_rows.num_rows == 0:
            raise ValueError(f"no {config.split} rows under {config.pool}")
        val_rows = read_rows(config.pool, [config.val_split], config.sources, config.val_lines, config.seed) if config.val_split else None
        if val_rows is None or val_rows.num_rows == 0:
            held = min(config.val_lines, max(1, train_rows.num_rows // 20))
            val_rows = train_rows.slice(0, held)
        resume = self.out / "state.pt"
        init = model_directory(config.init) if config.init and not resume.exists() else None
        if resume.exists():
            saved = json.loads((self.out / "last" / "config.json").read_text(encoding="utf-8"))
            self.codec = Codec.from_dict(saved["codec"])
        elif init:
            saved = json.loads((init / "config.json").read_text(encoding="utf-8"))
            self.codec = Codec.from_dict(saved["codec"])
            config.model = {k: v for k, v in saved["model"].items() if k != "n_labels"}
        else:
            presets = [config.presets.get(source, config.preset) for source in _column(train_rows, "source")]
            self.codec = Codec.build(_column(train_rows, "text"), config.min_count, presets)
        self.model = LineModel(ModelConfig(n_labels=len(self.codec), **config.model)).to(self.device)
        if init:
            from safetensors.torch import load_file

            self.model.load_state_dict(load_file(str(init / "model.safetensors"), device=str(self.device)))
        self.train_set = LineDataset(train_rows, self.codec, config, train=True)
        self.val_set = LineDataset(val_rows, self.codec, config, train=False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.amp = config.amp if self.device.type == "cuda" else "off"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp == "fp16")
        self.step = 0
        self.best = math.inf
        if resume.exists():
            state = torch.load(resume, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scaler.load_state_dict(state["scaler"])
            self.step, self.best = state["step"], state["best"]

    def loader(self, dataset: LineDataset, shuffle: bool) -> DataLoader:
        stride = self.model.config.stride
        sampler = WidthBatches(dataset.estimates, self.config.batch_columns, self.config.max_batch, self.config.seed + self.step, shuffle)
        workers = self.config.workers
        return DataLoader(dataset, batch_sampler=sampler, num_workers=workers, collate_fn=functools.partial(collate, stride=stride),
                          worker_init_fn=_seed_worker if workers else None, pin_memory=self.device.type == "cuda")

    def learning_rate(self) -> float:
        c = self.config
        if self.step < c.warmup:
            return c.lr * (self.step + 1) / c.warmup
        progress = min(1.0, (self.step - c.warmup) / max(1, c.steps - c.warmup))
        return c.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))

    def losses(self, batch: dict, train: bool) -> dict:
        images = batch["images"].to(self.device, non_blocking=True)
        widths = batch["widths"].to(self.device)
        graphs = batch["graphs"].to(self.device)
        dtype = torch.float16 if self.amp == "fp16" else torch.bfloat16
        with torch.autocast(self.device.type, dtype=dtype, enabled=self.amp != "off"):
            logits, boundary, lengths = self.model(images, widths)
        logp = logits.float().log_softmax(-1)
        if train and self.objective.topology == "ctc_prior":
            self._update_prior(logp, lengths)
        empty = batch["empty"].to(self.device)
        prior_scale = self.objective.prior_scale if train else self.objective.decode_prior_scale
        nll = graph.nll(logp, boundary.float(), graphs, lengths, self.options, self.objective.normalise, self.model.log_prior,
                        prior_scale, empty)
        units = batch["units"].to(self.device)
        ok = torch.isfinite(nll)
        nll_sum = torch.where(ok, nll, torch.zeros_like(nll)).sum()
        unit_sum = torch.where(ok, units, torch.zeros_like(units)).sum().clamp(min=1)
        loss = nll_sum / unit_sum
        bce = torch.zeros((), device=self.device)
        mask = batch["target_mask"].to(self.device)
        if self.config.boundary_weight and mask.any():
            targets = batch["targets"].to(self.device)
            bce = F.binary_cross_entropy_with_logits(boundary.float()[mask], targets[mask])
            loss = loss + self.config.boundary_weight * bce
        frame = torch.zeros((), device=self.device)
        frame_labels = batch["frame_labels"].to(self.device)
        if self.config.frame_weight and (frame_labels >= 0).any():
            frame = F.nll_loss(logp.transpose(1, 2), frame_labels, ignore_index=-100)
            loss = loss + self.config.frame_weight * frame
        return {"loss": loss, "nll": nll_sum.detach(), "units": unit_sum.detach(), "bce": bce.detach(), "frame": frame.detach(),
                "dropped": (~ok).sum().detach(), "logp": logp.detach(), "boundary": boundary.detach().float(), "lengths": lengths,
                "empty": empty}

    @torch.no_grad()
    def _update_prior(self, logp: torch.Tensor, lengths: torch.Tensor) -> None:
        mask = torch.arange(logp.shape[1], device=logp.device)[None, :] < lengths[:, None]
        mean = logp.exp()[mask].mean(0).clamp(min=1e-12)
        momentum = self.config.prior_momentum
        prior = momentum * self.model.log_prior.exp() + (1 - momentum) * mean
        self.model.log_prior.copy_(prior.log())

    def train(self) -> Path:
        config = self.config
        log = (self.out / "log.jsonl").open("a", encoding="utf-8")
        params = parameter_count(self.model)
        log.write(json.dumps({"event": "start", "step": self.step, "params": params, "labels": len(self.codec),
                              "train_lines": len(self.train_set), "val_lines": len(self.val_set), "device": str(self.device)}) + "\n")
        last_save = time.time()
        window = {"nll": 0.0, "units": 0.0, "bce": 0.0, "frame": 0.0, "lines": 0, "dropped": 0, "wait": 0.0, "start": time.time()}
        self.model.train()
        while self.step < config.steps:
            fetched = time.time()
            for batch in self.loader(self.train_set, shuffle=True):
                window["wait"] += time.time() - fetched
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate()
                self.optimizer.zero_grad(set_to_none=True)
                try:
                    out = self.losses(batch, train=True)
                    self.scaler.scale(out["loss"]).backward()
                except torch.OutOfMemoryError:
                    out = None
                if out is None:
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    log.write(json.dumps({"event": "oom", "step": self.step, "lines": len(batch["labels"]),
                                          "width": int(batch["images"].shape[-1])}) + "\n")
                    fetched = time.time()
                    continue
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step += 1
                window["nll"] += float(out["nll"])
                window["units"] += float(out["units"])
                window["bce"] += float(out["bce"])
                window["frame"] += float(out["frame"])
                window["lines"] += len(batch["labels"])
                window["dropped"] += int(out["dropped"])
                if self.step % config.log_every == 0:
                    elapsed = time.time() - window["start"]
                    log.write(json.dumps({"step": self.step, "nll_per_unit": window["nll"] / max(1, window["units"]),
                                          "bce": window["bce"] / config.log_every, "frame": window["frame"] / config.log_every,
                                          "lr": self.learning_rate(),
                                          "lines_per_s": window["lines"] / max(1e-6, elapsed), "dropped": window["dropped"],
                                          "step_s": elapsed / config.log_every, "wait_share": window["wait"] / max(1e-6, elapsed),
                                          "gpu_gb": self._gpu_gb()}) + "\n")
                    log.flush()
                    window = {"nll": 0.0, "units": 0.0, "bce": 0.0, "frame": 0.0, "lines": 0, "dropped": 0, "wait": 0.0, "start": time.time()}
                if self.step % config.val_every == 0 or self.step >= config.steps:
                    scores = self.validate()
                    log.write(json.dumps({"step": self.step, **scores}) + "\n")
                    log.flush()
                    if scores["val_nll_per_unit"] < self.best:
                        self.best = scores["val_nll_per_unit"]
                        save_model(self.out / "best", self.model, model_config_dict(config, self.codec, self.objective, self.step, scores))
                    self.save()
                    last_save = time.time()
                elif time.time() - last_save > config.save_every:
                    self.save()
                    last_save = time.time()
                if self.step >= config.steps:
                    break
                fetched = time.time()
        self.save()
        log.close()
        return self.out

    def _gpu_gb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        peak = torch.cuda.max_memory_allocated(self.device) / 2**30
        torch.cuda.reset_peak_memory_stats(self.device)
        return round(peak, 2)

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        nll = units = errors = length = 0.0
        segmental = self.options.segmental
        for batch in self.loader(self.val_set, shuffle=False):
            out = self.losses(batch, train=False)
            nll += float(out["nll"])
            units += float(out["units"])
            logp, lengths = graph.with_ink_prior(out["logp"], out["empty"], self.options.ink_prior), out["lengths"]
            if segmental:
                logb, log1mb = graph.boundary_logs(out["boundary"], tuple(logp.shape[:2]), logp)
                readings = graph.free_viterbi(logp, logb, log1mb, lengths)
            else:
                readings = graph.greedy_ctc(logp, lengths)
            for segments, words in zip(readings, batch["labels"]):
                reading = [label for word in graph.reading_words(segments, segmental) for label in word]
                truth = [label for word in words for label in word]
                errors += edit_distance(reading, truth)
                length += len(truth)
        self.model.train()
        return {"val_nll_per_unit": nll / max(1.0, units), "val_cer": errors / max(1.0, length)}

    def save(self) -> None:
        save_model(self.out / "last", self.model, model_config_dict(self.config, self.codec, self.objective, self.step))
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                 "step": self.step, "best": self.best}
        torch.save(state, self.out / "state.tmp")
        (self.out / "state.tmp").replace(self.out / "state.pt")
        (self.out / "train_config.json").write_text(json.dumps(asdict(self.config), indent=1), encoding="utf-8")
        self.on_save()


def _seed_worker(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    seed = (torch.initial_seed() + worker_id) % (1 << 31)
    info.dataset.rng = random.Random(seed)
    np.random.seed(seed)


def run(config: TrainConfig, device=None, on_save: Callable[[], None] | None = None) -> Path:
    return Trainer(config, device, on_save).train()


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        target = out
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = parsed
    return out


def merge(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for key, value in extra.items():
        merged[key] = merge(merged.get(key, {}), value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="littera train")
    parser.add_argument("--config", help="JSON file with TrainConfig fields")
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. objective.topology=ctc model.backbone=vgsl")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    base = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    run(TrainConfig.from_dict(merge(base, parse_overrides(args.set))), args.device)


if __name__ == "__main__":
    main()
