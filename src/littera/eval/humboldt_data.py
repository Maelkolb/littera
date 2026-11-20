"""The Humboldt lines of round 3: crops cut as the lookup build cuts them, the fine-tuning pool and the 1,000 evaluation lines.

``crop_fragments`` and ``fragments_box`` are copied from humboldt-htr (``src/humboldt_htr/crops.py``, the ``gap="box"`` path with
the polygon mask), so that a crop here equals the crop the lookup page's placement sees: the pieces' polygons widened by ``grow``
pixels on white, the upright box between two pieces kept.
"""

from __future__ import annotations

import json
import random
from typing import Any

from PIL import Image, ImageDraw

SOURCE = "humboldt_ft"
EVAL_SET = "humboldt-1000"
DEV_SPLITS = ("val", "test_page")
REPORT_SPLITS = ("test_bench", "test_journal", "letters")
SPLITS = (*DEV_SPLITS, *REPORT_SPLITS)
LICENCE = "CC BY-SA 4.0"


def polygon_bbox(points) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def fragments_box(size: tuple[int, int], fragments: list[dict[str, Any]], pad: int = 8) -> tuple[int, int, int, int]:
    pts = [q for f in fragments for q in (f.get("polygon") if len(f.get("polygon") or []) >= 3 else f.get("baseline") or [])]
    x0, y0, x1, y1 = polygon_bbox(pts)
    return max(0, x0 - pad), max(0, y0 - pad), min(size[0], x1 + pad), min(size[1], y1 + pad)


def _stencil(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], grow: int) -> None:
    draw.polygon(points, fill=255, outline=255)
    if grow:
        draw.line([*points, points[0]], fill=255, width=2 * grow + 1, joint="curve")


def crop_line(page: Image.Image, line: dict[str, Any], pad: int = 8, grow: int = 0) -> Image.Image:
    pad = max(pad, grow)
    points = line.get("polygon") or line.get("baseline")
    x0, y0, x1, y1 = polygon_bbox(points)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(page.width, x1 + pad), min(page.height, y1 + pad)
    crop = page.crop((x0, y0, x1, y1)).convert("RGB")
    if len(line.get("polygon") or []) < 3:
        return crop
    stencil = Image.new("L", crop.size, 0)
    _stencil(ImageDraw.Draw(stencil), [(x - x0, y - y0) for x, y in line["polygon"]], grow)
    white = Image.new("RGB", crop.size, (255, 255, 255))
    white.paste(crop, mask=stencil)
    return white


def crop_fragments(page: Image.Image, fragments: list[dict[str, Any]], pad: int = 8, grow: int = 0) -> Image.Image:
    pad = max(pad, grow)
    if len(fragments) == 1:
        return crop_line(page, fragments[0], pad, grow)
    boxes = [polygon_bbox(f["polygon"]) for f in fragments]
    order = sorted(range(len(fragments)), key=lambda i: boxes[i][0])
    x0 = max(0, min(b[0] for b in boxes) - pad)
    y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(page.width, max(b[2] for b in boxes) + pad)
    y1 = min(page.height, max(b[3] for b in boxes) + pad)
    crop = page.crop((x0, y0, x1, y1)).convert("RGB")
    stencil = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(stencil)
    for f in fragments:
        _stencil(draw, [(x - x0, y - y0) for x, y in f["polygon"]], grow)
    for a, b in zip(order, order[1:]):
        (_, ay0, ax1, ay1), (bx0, by0, _, by1) = boxes[a], boxes[b]
        if bx0 <= ax1:
            continue
        draw.rectangle((ax1 - x0, min(ay0, by0) - y0, bx0 - x0, max(ay1, by1) - y0), fill=255)
    white = Image.new("RGB", crop.size, (255, 255, 255))
    white.paste(crop, mask=stencil)
    return white


def production_crop(page: Image.Image, pieces: list[dict], grow: int = 6) -> tuple[Image.Image, tuple[int, int, int, int]]:
    return crop_fragments(page, pieces, grow=grow), fragments_box(page.size, pieces, pad=max(8, grow))


def crop_baselines(pieces: list[dict], box: tuple[int, int, int, int]) -> list[list[list[float]]]:
    x0, y0 = box[0], box[1]
    lines = [[[float(x - x0), float(y - y0)] for x, y in piece.get("baseline") or []] for piece in pieces]
    return sorted((line for line in lines if len(line) >= 2), key=lambda line: line[0][0])


def line_baseline(pieces: list[dict], box: tuple[int, int, int, int]) -> list[list[float]]:
    points = sorted((x, y) for piece in pieces for x, y in (piece.get("baseline") or []))
    return [[float(x - box[0]), float(y - box[1])] for x, y in points] if len(points) >= 2 else []


def text_on_crop(text: str, flags: str) -> tuple[str, list[int]]:
    kept, index = [], []
    for i, (ch, flag) in enumerate(zip(text, flags)):
        if flag == "a":
            continue
        if ch == " " and (not kept or kept[-1] == " "):
            continue
        kept.append(ch)
        index.append(i)
    while kept and kept[-1] == " ":
        kept.pop()
        index.pop()
    return "".join(kept), index


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def pool_plan(csv_rows: list[dict], recipe: list[dict], val_pages: int = 31, seed: int = 3) -> dict:
    recipe_pages = {entry["stem"].split("/")[1] for entry in recipe}
    kept = [r for r in csv_rows if r["source_split"] in ("train", "val") and r["page_id"] not in recipe_pages]
    pages = sorted({r["page_id"] for r in kept})
    held = set(random.Random(seed).sample(pages, min(val_pages, len(pages))))
    by_page: dict[str, dict] = {}
    for r in kept:
        text = clean_text(r["text_noadd"])
        page = by_page.setdefault(r["page_id"], {"stem": f"{r['journal']}/{r['page_id']}", "split": "val" if r["page_id"] in held else "train",
                                                 "lines": []})
        page["lines"].append({"uid": r["uid"], "text": text, "grade": r["grade"], "kind": r["kind"], "csv_split": r["source_split"]})
    dropped = {
        "rows on pages of the recipe": sum(r["source_split"] in ("train", "val") and r["page_id"] in recipe_pages for r in csv_rows),
        "rows of letters_train (not in the office build)": sum(r["source_split"] == "letters_train" for r in csv_rows),
        "rows of the test splits": sum(r["source_split"] not in ("train", "val", "letters_train") for r in csv_rows),
    }
    return {"pages": [by_page[p] for p in pages], "dropped": dropped, "recipe_pages": len(recipe_pages)}


def origin(stem: str, box, pieces: list[dict], **extra) -> str:
    return json.dumps({"stem": stem, "box": list(box), "pieces": len(pieces), "baselines": crop_baselines(pieces, box), **extra},
                      ensure_ascii=False)
