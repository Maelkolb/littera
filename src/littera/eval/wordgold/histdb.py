from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from littera.eval.wordgold.common import GoldLine, WordBox, gold_row, line_from_words, locate_word_crops

CODES = {"pt": ".", "cm": ",", "mi": "-", "sq": ";", "qo": ":", "qt": "'", "et": "&", "lb": "£", "bl": "(", "br": ")",
         "sl": "/", "s": "ſ", "dash": "-", "ex": "!", "qm": "?", "eq": "=", "pl": "+", "x": "×"}
LICENCE = "IAM-HistDB, registration required (FKI Bern); research use, evaluation only"
README = """IAM Historical Document Database ({name}), downloaded by the owner after registration at the FKI, University of Bern.
Word positions come from the database's word locations (polygons on the page) when present, otherwise from locating each
normalised word image inside its normalised line image (exact template match: the word images are crops of the line images).
Licence: registration-only research data. Evaluation only; keep this set private."""


def decode(token: str, unknown: dict[str, int]) -> str:
    out = []
    for part in token.split("-"):
        if part.startswith("s_"):
            code = part[2:]
            if code.isdigit():
                out.append(code)
            elif code in CODES:
                out.append(CODES[code])
            else:
                unknown[code] = unknown.get(code, 0) + 1
                out.append("?")
        else:
            out.append(part)
    return "".join(out)


def read_pairs(path: Path) -> dict[str, str]:
    pairs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, _, value = line.strip().partition(" ")
            pairs[key] = value.strip()
    return pairs


def svg_polygons(path: Path) -> dict[str, list[tuple[float, float]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    shapes = {}
    for match in re.finditer(r"<path[^>]*>", text):
        tag = match.group(0)
        ident = re.search(r'id="([^"]+)"', tag)
        data = re.search(r'\sd="([^"]+)"', tag)
        if ident and data:
            numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", data.group(1))]
            shapes[ident.group(1)] = list(zip(numbers[0::2], numbers[1::2]))
    return shapes


def find(root: Path, *names: str) -> Path | None:
    for name in names:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def build_rows(root: str | Path, name: str, language: str = "") -> tuple[list[dict], dict]:
    root = Path(root)
    unknown: dict[str, int] = {}
    word_labels_file = find(root, "word_labels.txt")
    labels = {k: decode(v, unknown) for k, v in read_pairs(word_labels_file).items()} if word_labels_file else {}
    by_line: dict[str, list[str]] = {}
    for word_id in sorted(labels):
        by_line.setdefault(word_id.rsplit("-", 1)[0], []).append(word_id)
    rows, stats = [], {"lines": len(by_line), "from_svg": 0, "from_crops": 0, "failed": 0}
    shapes: dict[str, list[tuple[float, float]]] = {}
    for svg in root.rglob("*.svg"):
        shapes.update(svg_polygons(svg))
    page_images = {p.stem: p for p in root.rglob("*.jpg")} | {p.stem: p for p in root.rglob("*.png") if "page" in str(p.parent).lower()}
    line_dir = find(root, "line_images_normalized")
    word_dir = find(root, "word_images_normalized")
    pages: dict[str, Image.Image] = {}
    for line_id, word_ids in sorted(by_line.items()):
        page_id = line_id.rsplit("-", 1)[0]
        if all(w in shapes for w in word_ids) and page_id in page_images:
            if page_id not in pages:
                pages = {page_id: Image.open(page_images[page_id]).convert("L")}
            boxes = []
            for word_id in word_ids:
                xs = [p[0] for p in shapes[word_id]]
                ys = [p[1] for p in shapes[word_id]]
                boxes.append(WordBox(labels[word_id], min(xs), min(ys), max(xs), max(ys)))
            text, gold = line_from_words(boxes)
            line = GoldLine(uid=f"{name}/{page_id}/{line_id}", doc=name, page=page_id, words=boxes)
            rows.append(gold_row(line, pages[page_id], text, gold, boxes, name, LICENCE, "IAM-HistDB word locations", language=language))
            stats["from_svg"] += 1
            continue
        if line_dir and word_dir and (line_dir / f"{line_id}.png").exists():
            from littera.eval.sets import encode_image

            line_image = Image.open(line_dir / f"{line_id}.png").convert("L")
            words = [Image.open(word_dir / f"{w}.png") for w in word_ids if (word_dir / f"{w}.png").exists()]
            spans = locate_word_crops(line_image, words) if len(words) == len(word_ids) else None
            if spans is None:
                stats["failed"] += 1
                continue
            boxes = [WordBox(labels[w], x0, 0, x1, line_image.height) for w, (x0, x1) in zip(word_ids, spans)]
            text, gold = line_from_words(boxes)
            rows.append({
                "uid": f"{name}/{page_id}/{line_id}", "source": name, "doc": name, "page": page_id, "hand": "", "split": "eval",
                "image": encode_image(line_image, "JPEG", 92), "width": line_image.width, "height": line_image.height, "text": text,
                "script": "latin", "language": language, "licence": LICENCE, "open_licence": False, "quality": "gold-words-normalised",
                "polygon": [], "baseline": [], "origin": "IAM-HistDB normalised line and word images", "kind": "word",
                "gold_words": [{**g, "x0": b.x0, "x1": b.x1} for g, b in zip(gold, boxes)], "gold_cuts": [],
            })
            stats["from_crops"] += 1
            continue
        stats["failed"] += 1
    stats["unknown_codes"] = unknown
    stats["rows"] = len(rows)
    return rows, stats
