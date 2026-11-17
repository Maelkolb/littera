from __future__ import annotations

import io
import random
import tarfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image

from littera.eval.wordgold.common import WordBox, merge_tokens

RECORD = "https://zenodo.org/records/19003721"
TASK = "largeWriterIndependentTextLineRecognitionTask.zip"
FILES = {name: f"{RECORD}/files/{name}?download=1" for name in ("xml.tar", "lines.tar", TASK)}
LICENCE = "CC BY-NC 4.0 (IAM Handwriting Database 3.0, Zenodo 19003721, migrated from FKI Bern with the authors' permission)"
README = """IAM Handwriting Database 3.0 (Marti and Bunke 2002), modern English handwriting, lines of the test set of the large
writer-independent text line task (128 writers), only lines whose segmentation the database marks as `ok`. Word positions are the
union of each word's connected components (`cmp`) in the form XML: "extracted from pages of scanned text using an automatic
segmentation scheme and verified manually" (database description). IAM separates punctuation as tokens; tokens without a space
between them in the line text are merged into one gold word, so gold words are the line text's whitespace words. The crop is the
database's line image (the bounding box of the line's components).
Licence: CC BY-NC 4.0. Evaluation only. Source: """ + RECORD


def test_ids(task_zip: str | Path) -> set[str]:
    with zipfile.ZipFile(task_zip) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith("testset.txt"))
        return {line.strip() for line in archive.read(name).decode().splitlines() if line.strip()}


def parse_form(xml_bytes: bytes) -> list[dict]:
    root = ElementTree.fromstring(xml_bytes)
    writer = root.get("writer-id", "")
    lines = []
    for line in root.iter("line"):
        tokens = []
        for word in line.iter("word"):
            cmps = list(word.iter("cmp"))
            if not cmps:
                tokens.append(None)
                continue
            x0 = min(int(c.get("x")) for c in cmps)
            y0 = min(int(c.get("y")) for c in cmps)
            x1 = max(int(c.get("x")) + int(c.get("width")) for c in cmps)
            y1 = max(int(c.get("y")) + int(c.get("height")) for c in cmps)
            tokens.append(WordBox(word.get("text"), x0, y0, x1, y1))
        lines.append({"id": line.get("id"), "text": line.get("text"), "segmentation": line.get("segmentation"), "writer": writer,
                      "tokens": tokens})
    return lines


def build_rows(xml_tar: str | Path, lines_tar: str | Path, task_zip: str | Path, limit: int = 1000,
               seed: int = 3) -> tuple[list[dict], dict]:
    from littera.eval.sets import encode_image

    wanted = test_ids(task_zip)
    candidates = {}
    with tarfile.open(xml_tar) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".xml"):
                for line in parse_form(tar.extractfile(member).read()):
                    if line["id"] in wanted and line["segmentation"] == "ok" and all(t is not None for t in line["tokens"]):
                        candidates[line["id"]] = line
    chosen = sorted(candidates)
    random.Random(seed).shuffle(chosen)
    chosen = set(chosen[:limit])
    rows, stats = [], {"test_lines": len(wanted), "ok_lines": len(candidates), "size_mismatch": 0, "token_mismatch": 0}
    with tarfile.open(lines_tar) as tar:
        for member in tar:
            stem = Path(member.name).stem
            if not member.isfile() or stem not in chosen:
                continue
            line = candidates[stem]
            image = Image.open(io.BytesIO(tar.extractfile(member).read())).convert("L")
            tokens = line["tokens"]
            left, top = min(t.x0 for t in tokens), min(t.y0 for t in tokens)
            right, bottom = max(t.x1 for t in tokens), max(t.y1 for t in tokens)
            if (right - left, bottom - top) != image.size:
                stats["size_mismatch"] += 1
                continue
            text = line["text"].replace("&quot;", '"')
            merged = merge_tokens(text, tokens)
            if merged is None:
                stats["token_mismatch"] += 1
                continue
            text, gold, boxes = merged
            gold_words = [{**g, "x0": float(b.x0 - left), "x1": float(b.x1 - left)} for g, b in zip(gold, boxes)]
            form = stem.rsplit("-", 1)[0]
            rows.append({
                "uid": f"iam/{form}/{stem}", "source": "iam", "doc": form, "page": form, "hand": line["writer"], "split": "eval",
                "image": encode_image(image, "JPEG", 92), "width": image.width, "height": image.height, "text": text,
                "script": "latin", "language": "en", "licence": LICENCE, "open_licence": False, "quality": "gold-words",
                "polygon": [], "baseline": [], "origin": RECORD, "kind": "word", "gold_words": gold_words, "gold_cuts": [],
            })
    rows.sort(key=lambda r: r["uid"])
    stats["rows"] = len(rows)
    return rows, stats
