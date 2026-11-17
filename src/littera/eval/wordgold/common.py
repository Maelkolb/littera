from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from littera.types import word_spans


@dataclass
class WordBox:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class GoldLine:
    uid: str
    doc: str
    page: str
    words: list[WordBox]
    band: tuple[float, float] | None = None
    hand: str = ""
    extra: dict = field(default_factory=dict)


def merge_tokens(line_text: str, tokens: list[WordBox]) -> tuple[str, list[dict], list[WordBox]] | None:
    cursor, placed = 0, []
    for token in tokens:
        start = line_text.find(token.text, cursor)
        if start < 0:
            return None
        placed.append((start, start + len(token.text), token))
        cursor = start + len(token.text)
    words = []
    for i0, i1 in word_spans(line_text):
        members = [t for a, b, t in placed if a >= i0 and b <= i1]
        if not members:
            return None
        words.append((i0, i1, members))
    boxes = [WordBox(line_text[i0:i1], min(m.x0 for m in ms), min(m.y0 for m in ms), max(m.x1 for m in ms), max(m.y1 for m in ms))
             for i0, i1, ms in words]
    gold = [{"i0": i0, "i1": i1} for i0, i1, _ in words]
    return line_text, gold, boxes


def line_from_words(words: list[WordBox]) -> tuple[str, list[dict]]:
    text, gold = "", []
    for word in words:
        if text:
            text += " "
        gold.append({"i0": len(text), "i1": len(text) + len(word.text)})
        text += word.text
    return text, gold


def crop_line(page: Image.Image, boxes: list[WordBox], band: tuple[float, float] | None, margin_x: float = 0.6,
              margin_y: float = 0.15) -> tuple[Image.Image, tuple[int, int]]:
    heights = [b.y1 - b.y0 for b in boxes]
    typical = float(np.median(heights)) if heights else 40.0
    top = min(b.y0 for b in boxes)
    bottom = max(b.y1 for b in boxes)
    if band is not None:
        top, bottom = min(top, band[0]), max(bottom, band[1])
    left = min(b.x0 for b in boxes) - margin_x * typical
    right = max(b.x1 for b in boxes) + margin_x * typical
    top -= margin_y * typical
    bottom += margin_y * typical
    box = (int(max(0, np.floor(left))), int(max(0, np.floor(top))),
           int(min(page.width, np.ceil(right))), int(min(page.height, np.ceil(bottom))))
    return page.crop(box), (box[0], box[1])


def gold_row(line: GoldLine, page: Image.Image, text: str, gold: list[dict], boxes: list[WordBox], source: str, licence: str,
             origin: str, script: str = "latin", language: str = "", margin_x: float = 0.6) -> dict:
    from littera.eval.sets import encode_image

    crop, (left, _) = crop_line(page, boxes, line.band, margin_x=margin_x)
    gold_words = [{**g, "x0": float(b.x0 - left), "x1": float(b.x1 - left)} for g, b in zip(gold, boxes)]
    return {
        "uid": line.uid, "source": source, "doc": line.doc, "page": line.page, "hand": line.hand, "split": "eval",
        "image": encode_image(crop.convert("L"), "JPEG", 92), "width": crop.width, "height": crop.height, "text": text,
        "script": script, "language": language, "licence": licence, "open_licence": False, "quality": "gold-words",
        "polygon": [], "baseline": [], "origin": origin, "kind": "word", "gold_words": gold_words, "gold_cuts": [],
    }


def locate_word_crops(line_image: Image.Image, word_images: list[Image.Image]) -> list[tuple[float, float]] | None:
    import cv2

    line = np.asarray(line_image.convert("L"), dtype=np.float32)
    positions, cursor = [], 0
    for word in word_images:
        template = np.asarray(word.convert("L"), dtype=np.float32)
        if template.shape[0] > line.shape[0] or template.shape[1] > line.shape[1]:
            return None
        result = cv2.matchTemplate(line, template, cv2.TM_SQDIFF_NORMED)
        result[:, : max(0, cursor - template.shape[1] // 2)] = np.inf
        y, x = np.unravel_index(int(np.argmin(result)), result.shape)
        if result[y, x] > 0.05:
            return None
        positions.append((float(x), float(x + template.shape[1])))
        cursor = int(x + template.shape[1])
    return positions
