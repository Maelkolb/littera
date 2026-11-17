from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from littera.eval.wordgold.common import WordBox, line_from_words, locate_word_crops

LICENCE = "Esposalles (CVC Barcelona), access on request; evaluation only"
README = """Esposalles marriage licence books (Barcelona Cathedral, 17th century; Romero et al. 2013), obtained by the owner from the
Computer Vision Center on request. Each word image is located inside its line image (word images are crops of the line images);
the gold word extents are those positions. Evaluation only; keep this set private."""

WORD = re.compile(r"^(?P<line>.+?)_?[Ww]ord_?(?P<n>\d+)$")


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    return ""


def build_rows(root: str | Path) -> tuple[list[dict], dict]:
    from littera.eval.sets import encode_image

    root = Path(root)
    images = {p.stem: p for p in root.rglob("*.png")} | {p.stem: p for p in root.rglob("*.jpg")}
    words_by_line: dict[str, list[tuple[int, Path]]] = {}
    for stem, path in images.items():
        match = WORD.match(stem)
        if match:
            words_by_line.setdefault(match.group("line").rstrip("_"), []).append((int(match.group("n")), path))
    rows, stats = [], {"lines": len(words_by_line), "failed": 0, "no_text": 0}
    for line_stem, members in sorted(words_by_line.items()):
        line_path = images.get(line_stem)
        if line_path is None:
            stats["failed"] += 1
            continue
        members.sort()
        texts = [read_text(p.with_suffix(".txt")) for _, p in members]
        if not all(texts):
            stats["no_text"] += 1
            continue
        line_image = Image.open(line_path).convert("L")
        spans = locate_word_crops(line_image, [Image.open(p) for _, p in members])
        if spans is None:
            stats["failed"] += 1
            continue
        boxes = [WordBox(t, x0, 0, x1, line_image.height) for t, (x0, x1) in zip(texts, spans)]
        text, gold = line_from_words(boxes)
        page = line_path.parent.parent.name if line_path.parent.name.lower() == "lines" else line_path.parent.name
        rows.append({
            "uid": f"esposalles/{page}/{line_stem}", "source": "esposalles", "doc": "esposalles", "page": page, "hand": "",
            "split": "eval", "image": encode_image(line_image, "JPEG", 92), "width": line_image.width, "height": line_image.height,
            "text": text, "script": "latin", "language": "ca", "licence": LICENCE, "open_licence": False, "quality": "gold-words",
            "polygon": [], "baseline": [], "origin": str(line_path.relative_to(root)), "kind": "word",
            "gold_words": [{**g, "x0": b.x0, "x1": b.x1} for g, b in zip(gold, boxes)], "gold_cuts": [],
        })
    stats["rows"] = len(rows)
    return rows, stats
