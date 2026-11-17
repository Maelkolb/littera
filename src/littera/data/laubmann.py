from __future__ import annotations

import io
import json
import re
import unicodedata

import numpy as np
from PIL import Image

OTHER_SPACES = re.compile("[ \t  -   　]+")
ACCEPTED_REASONS = ("", "not_selected", "dup_text_cap")
LEFT_OUT_COLLECTIONS = ("Laubmann_35",)
THIN_BOX = 0.6
VERTICAL_REACH = 0.3
MARKUP_REMNANTS = ("<", ">")
WIDTH_PER_CHAR = (0.08, 2.5)
MIN_HEIGHT = 16
MIN_PIXELS_PER_CHAR = 3
MAX_HEIGHT = 160


def clean_line(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\ufeff", "").replace("\u200b", "")
    return OTHER_SPACES.sub(" ", text).strip()


def region_lines(target: str) -> list[str]:
    return [clean_line(line) for line in target.replace("\r", "").split("\n")]


def visible_length(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace() and not unicodedata.combining(ch))


def long_runs(mask: np.ndarray, length: int) -> np.ndarray:
    runs = np.zeros_like(mask)
    for y in range(mask.shape[0]):
        edges = np.diff(np.concatenate([[0], mask[y].astype(np.int8), [0]]))
        for start, end in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            if end - start >= length:
                runs[y, start:end] = True
    return runs


def ink_mask(grey: np.ndarray, line_height: float) -> np.ndarray:
    darkness = 255.0 - grey.astype(np.float32)
    spread = darkness - float(np.median(darkness))
    level = max(28.0, 0.45 * float(np.percentile(spread, 99)))
    mask = spread > level
    rules = long_runs(mask, int(4 * line_height)) | long_runs(mask.T, int(2.5 * line_height)).T
    return mask & ~rules


def ink_span(mask: np.ndarray, top: int, bottom: int) -> tuple[int, int] | None:
    height = bottom - top
    core = mask[top + height // 4: bottom - height // 4]
    if core.size == 0:
        return None
    inked = (core.sum(axis=0) > 0).astype(np.float32)
    window = max(3, height // 4)
    density = np.convolve(inked, np.ones(window) / window, mode="same")
    columns = np.nonzero(density >= 0.3)[0]
    if len(columns) == 0:
        return None
    return int(columns[0]), int(columns[-1]) + 1


def cut_region(image_bytes: bytes, line_boxes: list[list[float]]) -> list[tuple[Image.Image, tuple[int, int, int, int]] | tuple[str, Image.Image]]:
    with Image.open(io.BytesIO(image_bytes)) as opened:
        region = opened.convert("RGB")
    typical_height = float(np.median([b[3] - b[1] for b in line_boxes]))
    mask = ink_mask(np.asarray(region.convert("L")), typical_height)
    width, height = region.size
    out: list = []
    for x0, y0, x1, y1 in line_boxes:
        x0, x1 = max(0, round(x0)), min(width, round(x1))
        y0, y1 = max(0, round(y0)), min(height, round(y1))
        if x1 - x0 < 8 or y1 - y0 < 8 or y1 - y0 < THIN_BOX * typical_height:
            out.append(("thin_box", region.crop((0, max(0, y0 - 10), width, min(height, y1 + 10)))))
            continue
        span = ink_span(mask, y0, y1)
        if span is None:
            out.append(("no_ink", region.crop((0, y0, width, y1))))
            continue
        if span[0] <= 1 or span[1] >= width - 1:
            out.append(("ink_at_region_edge", region.crop((0, y0, width, y1))))
            continue
        pad = max(6, round(0.25 * (y1 - y0)))
        reach = round(VERTICAL_REACH * (y1 - y0))
        box = (max(0, span[0] - pad), max(0, y0 - reach), min(width, span[1] + pad), min(height, y1 + reach))
        crop = region.crop(box)
        if crop.height > MAX_HEIGHT:
            crop = crop.resize((max(1, round(crop.width * MAX_HEIGHT / crop.height)), MAX_HEIGHT), Image.LANCZOS)
        out.append((crop, box))
    return out


def line_problem(text: str, width: int, height: int) -> str:
    visible = visible_length(text)
    if visible < 2:
        return "short_text"
    if any(mark in text for mark in MARKUP_REMNANTS):
        return "markup_remnant"
    if height < MIN_HEIGHT or width < MIN_PIXELS_PER_CHAR * visible:
        return "crop_too_small"
    per_char = width / (height * visible)
    if not WIDTH_PER_CHAR[0] <= per_char <= WIDTH_PER_CHAR[1]:
        return "width_per_char"
    return ""


def encode_jpeg(image: Image.Image, quality: int = 92) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    return buffer.getvalue()


def line_rows(region: dict, rejected: list | None = None) -> tuple[list[dict], dict[str, int]]:
    extra = json.loads(region["extra"] or "{}")
    texts = region_lines(region["target"])
    boxes = extra.get("line_boxes") or []
    dropped: dict[str, int] = {}
    if len(texts) != len(boxes):
        return [], {"line_count_mismatch": len(texts)}
    region_id = region["sample_id"].rsplit("/", 1)[-1]
    rows = []
    for number, (text, cut) in enumerate(zip(texts, cut_region(region["image"], boxes)), start=1):
        if isinstance(cut[0], str):
            dropped[cut[0]] = dropped.get(cut[0], 0) + 1
            if rejected is not None:
                rejected.append((cut[0], text, cut[1]))
            continue
        crop, box = cut
        problem = line_problem(text, crop.width, crop.height)
        if problem:
            dropped[problem] = dropped.get(problem, 0) + 1
            continue
        rows.append({
            "uid": f"laubmann/{region['collection']}/{region['page_id']}/{region_id}_l{number:02d}",
            "source": "laubmann", "doc": region["collection"], "page": region["page_id"], "hand": f"laubmann/{region['collection']}",
            "split": "heldout", "image": encode_jpeg(crop), "width": crop.width, "height": crop.height, "text": text,
            "script": region["script"], "language": region["language"], "licence": region["license"], "open_licence": False,
            "quality": region["quality"], "polygon": [], "baseline": [],
            "origin": f"kurrent-raw:/{region['file']}#{region['sample_id']}#line={number};box={','.join(str(v) for v in box)}",
        })
    return rows, dropped
