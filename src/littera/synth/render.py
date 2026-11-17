from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from littera.types import word_spans

INK_THRESHOLD = 96


@lru_cache(maxsize=256)
def font_codepoints(path: str) -> frozenset[int]:
    from fontTools.ttLib import TTFont

    font = TTFont(path, lazy=True, fontNumber=0)
    cmap = font.getBestCmap() or {}
    return frozenset(cmap.keys())


def font_covers(path: str, text: str) -> bool:
    points = font_codepoints(path)
    return all(ord(ch) in points or ch.isspace() for ch in text)


@lru_cache(maxsize=512)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.BASIC)


def clean_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def x_height(font: ImageFont.FreeTypeFont) -> float:
    box = font.getbbox("xnvu", anchor="ls")
    height = -box[1]
    return float(height) if height > 0 else font.size * 0.45


def size_for_x_height(path: str, target: float) -> int:
    probe = load_font(path, 100)
    return max(8, int(round(100 * target / max(1.0, x_height(probe)))))


def pen_positions(font: ImageFont.FreeTypeFont, text: str) -> list[float]:
    positions = []
    for i in range(len(text)):
        positions.append(font.getlength(text[: i + 1]) - font.getlength(text[i]))
    positions.append(font.getlength(text))
    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i - 1])
    return positions


@dataclass
class Layout:
    text: str
    font_path: str
    size: int
    x_height: float
    pens: list[float]
    mask: np.ndarray
    word_masks: list[np.ndarray]
    origin_x: float
    baseline_y: float
    meta: dict = field(default_factory=dict)


def draw_text(canvas_size: tuple[int, int], text: str, font: ImageFont.FreeTypeFont, x: float, y: float) -> np.ndarray:
    image = Image.new("L", canvas_size, 0)
    ImageDraw.Draw(image).text((x, y), text, font=font, fill=255, anchor="ls")
    return np.asarray(image, dtype=np.uint8)


def layout_line(text: str, font_path: str, size: int, neighbours: tuple[str, str] | None = None, spacing: float = 3.2,
                neighbour_shift: tuple[float, float] = (0.0, 0.0)) -> Layout:
    font = load_font(font_path, size)
    xh = x_height(font)
    pens = pen_positions(font, text)
    left, top, right, bottom = font.getbbox(text, anchor="ls")
    pad = int(math.ceil(2 * xh))
    gap = spacing * xh
    above = int(math.ceil(gap + 2 * xh)) if neighbours else 0
    below = int(math.ceil(gap + 2 * xh)) if neighbours else 0
    origin_x = pad + max(0.0, -left)
    baseline_y = pad + max(0.0, -top) + above
    width = int(math.ceil(origin_x + max(right, pens[-1]) + pad))
    height = int(math.ceil(baseline_y + max(0.0, bottom) + pad + below))
    mask = draw_text((width, height), text, font, origin_x, baseline_y)
    word_masks = [draw_text((width, height), text[a:b], font, origin_x + pens[a], baseline_y) for a, b in word_spans(text)]
    if neighbours:
        for other, direction, shift in zip(neighbours, (-1, 1), neighbour_shift):
            if other:
                layer = draw_text((width, height), other, font, origin_x + shift * xh, baseline_y + direction * gap)
                mask = np.maximum(mask, layer)
    return Layout(text, font_path, size, xh, pens, mask, word_masks, origin_x, baseline_y)


def shear_matrix(slant: float, baseline_y: float, height: int) -> tuple[np.ndarray, int]:
    offsets = [slant * (baseline_y - y) for y in (0.0, float(height))]
    shift = -min(0.0, *offsets)
    extra = int(math.ceil(max(0.0, *offsets) + shift))
    matrix = np.array([[1.0, -slant, slant * baseline_y + shift], [0.0, 1.0, 0.0]], dtype=np.float32)
    return matrix, extra


def apply_shear(array: np.ndarray, matrix: np.ndarray, extra: int) -> np.ndarray:
    height, width = array.shape
    return cv2.warpAffine(array, matrix, (width + extra, height), flags=cv2.INTER_LINEAR, borderValue=0)


def apply_wobble(array: np.ndarray, amplitude: float, wavelength: float, phase: float) -> np.ndarray:
    if amplitude <= 0:
        return array
    height, width = array.shape
    xs = np.arange(width, dtype=np.float32)
    displacement = amplitude * np.sin(2 * np.pi * xs / max(1.0, wavelength) + phase)
    map_x = np.tile(xs, (height, 1))
    map_y = np.arange(height, dtype=np.float32)[:, None] - displacement[None, :]
    return cv2.remap(array, map_x, map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderValue=0)


def ink_columns(array: np.ndarray) -> tuple[float, float] | None:
    columns = np.nonzero((array >= INK_THRESHOLD).any(axis=0))[0]
    if len(columns) == 0:
        return None
    return float(columns[0]), float(columns[-1] + 1)


@dataclass
class Geometry:
    mask: np.ndarray
    gold_words: list[dict]
    gold_cuts: list[dict]
    baseline_y: float
    x_height: float
    slant: float


def place_geometry(layout: Layout, slant: float = 0.0, wobble: tuple[float, float, float] = (0.0, 100.0, 0.0)) -> Geometry:
    matrix, extra = shear_matrix(slant, layout.baseline_y, layout.mask.shape[0])
    shift = float(matrix[0, 2] - slant * layout.baseline_y)
    mask = apply_wobble(apply_shear(layout.mask, matrix, extra), *wobble)
    reference_y = layout.baseline_y - layout.x_height / 2
    text = layout.text

    def cut_x(i: int) -> float:
        return layout.origin_x + layout.pens[i] + slant * (layout.baseline_y - reference_y) + shift

    gold_cuts = []
    for i in range(len(text) + 1):
        touches = (i < len(text) and not text[i].isspace()) or (i > 0 and not text[i - 1].isspace())
        if touches:
            gold_cuts.append({"i": i, "x": cut_x(i)})
    gold_words = []
    for (a, b), word_mask in zip(word_spans(text), layout.word_masks):
        extent = ink_columns(apply_wobble(apply_shear(word_mask, matrix, extra), *wobble))
        x0, x1 = extent if extent else (cut_x(a), cut_x(b))
        gold_words.append({"i0": a, "i1": b, "x0": x0, "x1": x1})
    return Geometry(mask, gold_words, gold_cuts, layout.baseline_y, layout.x_height, slant)
