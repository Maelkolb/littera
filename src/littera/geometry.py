from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

Point = tuple[float, float]


@dataclass
class GeometryConfig:
    height: int = 96
    pad: int = 16
    core_ratio: float = 0.2
    deslant: bool = True
    max_slant_deg: float = 50.0
    max_width: int = 4096
    stride: int = 4

    @property
    def core(self) -> float:
        return self.core_ratio * self.height

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> GeometryConfig:
        return cls(**(d or {}))


@dataclass
class Strip:
    image: np.ndarray
    crop_width: int
    crop_height: int
    scale: float
    pad: float
    slant: float
    v_mid: float
    core: float
    centre_x: np.ndarray
    centre_y: np.ndarray
    band: tuple[float, float]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    def to_crop(self, u, v) -> tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        x = (u - self.pad + self.slant * (self.v_mid - v)) * self.scale
        y = np.interp(x, self.centre_x, self.centre_y) + (v - self.v_mid) * self.scale
        return x, y

    def crop_x(self, u):
        return (np.asarray(u, dtype=np.float64) - self.pad) * self.scale

    def strip_u(self, x):
        return np.asarray(x, dtype=np.float64) / self.scale + self.pad

    def quad(self, u0: float, u1: float, top: float | None = None, bottom: float | None = None) -> list[Point]:
        top = self.band[0] if top is None else top
        bottom = self.band[1] if bottom is None else bottom
        xs, ys = self.to_crop([u0, u1, u1, u0], [top, top, bottom, bottom])
        xs = np.clip(xs, 0, self.crop_width)
        ys = np.clip(ys, 0, self.crop_height)
        return [(round(float(x), 1), round(float(y), 1)) for x, y in zip(xs, ys)]

    def ink_level(self) -> float:
        return otsu_level(self.image[self.image > 0.02])

    def empty_frames(self, stride: int, reach: int = 1) -> np.ndarray:
        top = max(0, int(math.floor(self.v_mid - self.core)))
        bottom = min(self.height, int(math.ceil(self.v_mid + self.core)))
        inked = (self.image[top:bottom] > self.ink_level()).sum(0) > 0
        frames = self.width // stride
        per_frame = inked[: frames * stride].reshape(frames, stride).any(1)
        near = per_frame.copy()
        for shift in range(1, reach + 1):
            near[shift:] |= per_frame[:-shift]
            near[:-shift] |= per_frame[shift:]
        return ~near

    def core_profile(self) -> np.ndarray:
        top = max(0, int(math.floor(self.v_mid - 0.75 * self.core)))
        bottom = min(self.height, int(math.ceil(self.v_mid + 0.75 * self.core)))
        return (self.image[top:bottom] > self.ink_level()).sum(0).astype(np.float64)


def ink_image(image: Image.Image) -> np.ndarray:
    grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if grey.size == 0 or min(grey.shape) < 2:
        return np.zeros((max(2, grey.shape[0]), max(2, grey.shape[1] if grey.ndim == 2 else 2)), np.float32)
    grey = _fill_dark_borders(grey)
    if float(np.median(grey)) < 0.4:
        grey = _fill_dark_borders(1.0 - grey)
    size = max(9, (min(grey.shape[0], 400) // 6) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    background = cv2.GaussianBlur(cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel), (0, 0), 2.0)
    ink = np.clip(background - grey, 0, None)
    level = max(float(np.percentile(ink, 98)), 0.12)
    ink = np.clip(ink / level, 0, 1)
    return np.clip((ink - 0.08) / 0.92, 0, 1).astype(np.float32)


def _fill_dark_borders(grey: np.ndarray) -> np.ndarray:
    h, w = grey.shape
    dark = (grey < 0.06).astype(np.uint8)
    if dark.mean() < 0.005:
        return grey
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    masked = np.zeros_like(dark, dtype=bool)
    for k in range(1, count):
        x, y, bw, bh, area = stats[k]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches and area >= 0.002 * h * w and area >= 0.3 * bw * bh and min(bw, bh) >= 0.05 * h:
            masked |= labels == k
    if not masked.any() or masked.all():
        return grey
    filled = grey.copy()
    filled[masked] = float(np.median(grey[~masked]))
    return filled


def smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if sigma < 0.5 or len(values) < 3:
        return values
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel / kernel.sum(), mode="valid")


def core_band(rows: np.ndarray, prefer: float, spread: float) -> tuple[int, int] | None:
    rows = np.asarray(rows, dtype=np.float64)
    if len(rows) == 0 or rows.max() <= 0:
        return None
    profile = smooth(rows, max(1.0, len(rows) / 80))
    weight = np.exp(-(((np.arange(len(rows)) - prefer) / max(spread, 1.0)) ** 2))
    peak = int(np.argmax(profile * weight))
    level, valley = 0.5 * profile[peak], 0.75 * profile[peak]
    top = peak
    def grows(inner: int, outer: int) -> bool:
        return profile[outer] >= level and not (profile[outer] > profile[inner] and profile[inner] < valley)

    top = peak
    while top > 0 and grows(top, top - 1):
        top -= 1
    bottom = peak
    while bottom < len(rows) - 1 and grows(bottom, bottom + 1):
        bottom += 1
    return top, bottom + 1


def otsu_level(values: np.ndarray, fallback: float = 0.35) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 16:
        return fallback
    hist, edges = np.histogram(values, bins=64, range=(0.0, 1.0))
    total = hist.sum()
    centres = np.arange(64)
    w0 = np.cumsum(hist)
    s0 = np.cumsum(hist * centres)
    w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = s0 / w0
        m1 = (s0[-1] - s0) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = -1
    best = int(np.argmax(between))
    if between[best] <= 0:
        return fallback
    return float(edges[best + 1])


def _sample(ink: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(ink, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0.0)


def _polygon_mask(polygon: Sequence[Point], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask


def _polygon_guide(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    columns = np.nonzero(mask.any(0))[0]
    if len(columns) < 2:
        return None
    h = mask.shape[0]
    top = np.argmax(mask[:, columns] > 0, axis=0)
    bottom = h - np.argmax(mask[::-1, columns] > 0, axis=0)
    centre = (top + bottom) / 2.0
    n = int(min(24, len(columns)))
    picks = np.linspace(0, len(columns) - 1, n).round().astype(int)
    xs = columns[picks].astype(np.float64)
    ys = np.array([np.median(centre[max(0, p - 8) : p + 9]) for p in picks])
    return xs, ys


def _without_rules(ink: np.ndarray) -> np.ndarray:
    h, w = ink.shape
    length = int(max(40, 1.5 * h))
    if w < length:
        return ink
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    return np.clip(ink - rules, 0, 1)


def _slice_bands(ink: np.ndarray, centre_at, spread: float, slice_width: float) -> list[tuple[float, float, float, float]]:
    h, w = ink.shape
    n = max(1, int(round(w / slice_width)))
    found = []
    for i in range(n):
        x0, x1 = i * w // n, (i + 1) * w // n
        rows = ink[:, x0:x1].sum(1)
        x = (x0 + x1) / 2
        band = core_band(rows, prefer=centre_at(x), spread=spread)
        if band is not None:
            found.append((x, (band[0] + band[1]) / 2, float(band[1] - band[0]), float(rows.sum())))
    return found


def _fit_line(found: list[tuple[float, float, float, float]], w: int) -> tuple[float, float]:
    xs, ys, weights = (np.array([f[k] for f in found]) for k in (0, 1, 3))
    weights = np.sqrt(weights)
    if len(found) < 2 or np.ptp(xs) <= 0:
        return 0.0, float(np.average(ys, weights=weights))
    slope = float(np.clip(np.polyfit(xs, ys, 1, w=weights)[0], -0.15, 0.15))
    return slope, float(np.average(ys - slope * xs, weights=weights))


def _ink_guide(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
    h, w = ink.shape
    found = _slice_bands(ink, lambda x: h / 2, 0.35 * h, max(h, 48))
    if not found:
        return np.array([0.0, float(w)]), np.array([h / 2, h / 2]), None
    heights = np.array([f[2] for f in found])
    xheight = float(np.median(heights))
    slope, intercept = _fit_line(found, w)
    for _ in range(2):
        found = _slice_bands(ink, lambda x, a=intercept, b=slope: a + b * x, max(xheight, 0.1 * h), max(h, 48))
        if not found:
            break
        residual = np.array([abs(f[1] - (intercept + slope * f[0])) for f in found])
        kept = [f for f, r in zip(found, residual) if r <= max(0.75 * xheight, 2.0)] or found
        xheight = float(np.median([f[2] for f in kept]))
        slope, intercept = _fit_line(kept, w)
    return np.array([0.0, float(w)]), np.array([intercept, intercept + slope * w]), xheight


def _baseline_guide(baseline: Sequence[Point]) -> tuple[np.ndarray, np.ndarray] | None:
    points = sorted((float(x), float(y)) for x, y in baseline)
    xs, ys = [], []
    for x, y in points:
        if xs and x <= xs[-1]:
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 2:
        return None
    return np.array(xs), np.array(ys)


def estimate_slant(strip: np.ndarray, v_mid: float, core: float, max_deg: float) -> float:
    h = strip.shape[0]
    top, bottom = max(0, int(v_mid - 2.2 * core)), min(h, int(v_mid + 2.2 * core))
    vs, us = np.nonzero(strip[top:bottom] > 0.35)
    if len(us) < 30:
        return 0.0
    rise = v_mid - (vs + top)

    def sharpness(degrees: float) -> float:
        columns = np.round(us - math.tan(math.radians(degrees)) * rise).astype(np.int64)
        hist = np.bincount(columns - columns.min()).astype(np.float64)
        return float((hist**2).sum())

    coarse = np.arange(-max_deg, max_deg + 1e-6, 2.5)
    best = max(coarse, key=sharpness)
    fine = np.arange(best - 2.0, best + 2.0 + 1e-6, 0.5)
    best = max(fine, key=sharpness)
    if sharpness(best) < 1.03 * sharpness(0.0):
        return 0.0
    return math.tan(math.radians(best))


def _ink_band(strip: np.ndarray, v_mid: float, core: float) -> tuple[float, float]:
    h = strip.shape[0]
    rows = smooth(strip.sum(1), 1.0)
    core_top, core_bottom = int(round(v_mid - 0.5 * core)), int(round(v_mid + 0.5 * core))
    peak = float(rows[core_top:core_bottom].max(initial=0))
    if peak <= 0:
        return max(0.0, v_mid - 1.5 * core), min(float(h), v_mid + 1.5 * core)
    level = 0.04 * peak
    limit_top, limit_bottom = max(0, int(v_mid - 2.2 * core)), min(h, int(math.ceil(v_mid + 2.2 * core)))
    top = core_top
    while top > limit_top and rows[top - 1] > level:
        top -= 1
    bottom = core_bottom
    while bottom < limit_bottom and rows[bottom] > level:
        bottom += 1
    return float(top), float(bottom)


def prepare(image: Image.Image, polygon: Sequence[Point] | None = None, baseline: Sequence[Point] | None = None,
            config: GeometryConfig | None = None, core_scale: float = 1.0, slant: float | None = None) -> Strip:
    config = config or GeometryConfig()
    ink = ink_image(image)
    h, w = ink.shape
    mask = None
    if polygon is not None and len(polygon) >= 3:
        mask = _polygon_mask(polygon, (h, w))
        ink = ink * cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(np.float32)
    guide, along_baseline, box, slice_height = None, False, False, None
    if baseline is not None and len(baseline) >= 2:
        guide = _baseline_guide(baseline)
        along_baseline = guide is not None
    if guide is None and mask is not None:
        guide = _polygon_guide(mask)
    measure = _without_rules(ink)
    if guide is None:
        gx, gy, slice_height = _ink_guide(measure)
        box = True
    else:
        gx, gy = guide

    reach = h
    columns = np.arange(w, dtype=np.float64)
    rows_v = np.arange(-reach, reach, dtype=np.float64)
    straight = _sample(measure, np.broadcast_to(columns, (len(rows_v), w)), np.interp(columns, gx, gy)[None, :] + rows_v[:, None])
    prefer = reach - (0.1 * h if along_baseline else 0.0)
    spread = slice_height if slice_height else (0.25 if along_baseline else 0.35) * h
    band = core_band(straight.sum(1), prefer=prefer, spread=spread)
    upper = h / 2.5 if box else h / 2.2
    if band is None:
        xheight, offset = h / 3, (-h / 6 if along_baseline else 0.0)
    else:
        xheight = float(np.clip(band[1] - band[0], max(4.0, h / 10), max(4.0, upper)))
        offset = (band[0] + band[1]) / 2 - reach
    centre_y = gy + offset

    height, core = config.height, config.core * core_scale
    v_mid = height / 2
    scale = float(np.clip(xheight / core, 0.1, 20.0))
    if w / scale + 2 * config.pad > config.max_width:
        scale = w / (config.max_width - 2 * config.pad)

    source, sx, sy = ink, 1.0, 1.0
    if scale > 1.25:
        size = (max(2, int(round(w / scale))), max(2, int(round(h / scale))))
        source = cv2.resize(ink, size, interpolation=cv2.INTER_AREA)
        sx, sy = size[0] / w, size[1] / h

    def render(sigma: float, pad: float, width: int) -> np.ndarray:
        u = np.arange(width, dtype=np.float64)[None, :]
        v = np.arange(height, dtype=np.float64)[:, None]
        x = (u - pad + sigma * (v_mid - v)) * scale
        y = np.interp(x, gx, centre_y) + (v - v_mid) * scale
        return _sample(source, x * sx, y * sy)

    straight_width = int(math.ceil(w / scale)) + 2 * config.pad
    if slant is None:
        slant = estimate_slant(render(0.0, config.pad, straight_width), v_mid, core, config.max_slant_deg) if config.deslant else 0.0
    extra = int(math.ceil(abs(slant) * max(v_mid, height - v_mid)))
    pad = config.pad + extra
    width = straight_width + 2 * extra
    width += (-width) % config.stride
    strip = render(slant, pad, width)
    return Strip(image=strip.astype(np.float32), crop_width=w, crop_height=h, scale=scale, pad=float(pad), slant=float(slant),
                 v_mid=v_mid, core=core, centre_x=np.asarray(gx, dtype=np.float64), centre_y=np.asarray(centre_y, dtype=np.float64),
                 band=_ink_band(strip, v_mid, core))


def best_gap(profile: np.ndarray, lo: int, hi: int, raw: float, level: float) -> tuple[float, float] | None:
    lo, hi = max(0, lo), min(len(profile), hi)
    if hi - lo < 1:
        return None
    empty = profile[lo:hi] <= level
    runs, start = [], None
    for k, is_empty in enumerate(list(empty) + [False]):
        if is_empty and start is None:
            start = k
        elif not is_empty and start is not None:
            runs.append((lo + start, lo + k))
            start = None
    if runs:
        s, e = max(runs, key=lambda r: (r[1] - r[0]) - 0.5 * abs((r[0] + r[1]) / 2 - raw))
        return float(s), float(e)
    window = profile[lo:hi]
    local = np.array([window[max(0, k - 1) : k + 2].sum() for k in range(len(window))])
    k = min(range(len(local)), key=lambda j: (local[j], abs(lo + j + 0.5 - raw)))
    return float(lo + k), float(lo + k + 1)


def ink_edge(profile: np.ndarray, raw: float, reach: float, level: float, side: str) -> float:
    n = len(profile)
    lo, hi = max(0, int(math.floor(raw - reach))), min(n, int(math.ceil(raw + reach)))
    if hi <= lo:
        return raw
    inked = profile[lo:hi] > level
    if side == "start":
        edges = [lo + k for k in range(hi - lo) if inked[k] and (lo + k == 0 or profile[lo + k - 1] <= level)]
    else:
        edges = [lo + k + 1 for k in range(hi - lo) if inked[k] and (lo + k + 1 >= n or profile[lo + k + 1] <= level)]
    if not edges:
        return raw
    return float(min(edges, key=lambda e: abs(e - raw)))


def ink_extent(profile: np.ndarray, level: float) -> tuple[float, float] | None:
    inked = np.nonzero(profile > level)[0]
    if len(inked) == 0:
        return None
    return float(inked[0]), float(inked[-1] + 1)
