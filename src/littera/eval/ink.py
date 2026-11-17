from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class InkProfile:
    mask: np.ndarray
    top: int
    bottom: int
    core: np.ndarray
    level: float

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])

    def extent(self) -> tuple[float, float]:
        columns = np.nonzero(self.core > self.level)[0]
        if len(columns) == 0:
            return 0.0, float(self.width)
        return float(columns[0]), float(columns[-1] + 1)

    def is_gap(self, x: float) -> bool:
        column = int(np.floor(x))
        if column < 0 or column >= self.width:
            return True
        return bool(self.core[column] <= self.level)


def ink_darkness(image: Image.Image, polygon: Sequence[Sequence[float]] | None = None) -> np.ndarray:
    grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    darkness = 1.0 - grey
    if polygon is not None and len(polygon) >= 3:
        inside = Image.new("L", image.size, 0)
        ImageDraw.Draw(inside).polygon([(float(x), float(y)) for x, y in polygon], fill=1)
        darkness = darkness * np.asarray(inside, dtype=np.float32)
    return darkness


def otsu_level(values: np.ndarray, bins: int = 64) -> float:
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centres = (edges[:-1] + edges[1:]) / 2
    weight0 = np.cumsum(hist)
    weight1 = total - weight0
    sum0 = np.cumsum(hist * centres)
    mean0 = np.divide(sum0, weight0, out=np.zeros_like(sum0), where=weight0 > 0)
    mean1 = np.divide(sum0[-1] - sum0, weight1, out=np.zeros_like(sum0), where=weight1 > 0)
    between = weight0 * weight1 * (mean0 - mean1) ** 2
    best = np.nonzero(between >= between.max() * (1 - 1e-9))[0]
    return float(edges[int(round(best.mean())) + 1])


def ink_profile(image: Image.Image, polygon: Sequence[Sequence[float]] | None = None, gap_share: float = 0.02) -> InkProfile:
    darkness = ink_darkness(image, polygon)
    background = float(np.median(darkness))
    spread = darkness - background
    level = otsu_level(np.clip(spread[spread > 0.02], 0, 1)) if np.any(spread > 0.02) else 1.0
    mask = spread > max(level, 0.08)
    rows = mask.mean(axis=1)
    peak = float(rows.max()) if rows.size else 0.0
    if peak > 0:
        dense = np.nonzero(rows >= 0.5 * peak)[0]
        pad = max(1, int(round(0.08 * mask.shape[0])))
        top, bottom = max(0, int(dense[0]) - pad), min(mask.shape[0], int(dense[-1]) + 1 + pad)
    else:
        top, bottom = 0, mask.shape[0]
    core = mask[top:bottom].sum(axis=0).astype(np.float32)
    gap_level = max(1.0, gap_share * float(core.max(initial=0.0)))
    return InkProfile(mask=mask, top=top, bottom=bottom, core=core, level=gap_level)
