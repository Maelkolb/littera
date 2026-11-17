from __future__ import annotations

import tarfile
from pathlib import Path

from PIL import Image

from littera.eval.wordgold.common import GoldLine, WordBox, gold_row, line_from_words

URL = "https://ciir.cs.umass.edu/downloads/gw/gw_20p_wannot.tgz"
LICENCE = "CIIR UMass research-only, no redistribution (copyrightnotice.txt); scans Library of Congress"
README = """George Washington letter books, 20 pages (Library of Congress scans via CIIR, UMass Amherst; Lavrenko, Rath and Manmatha 2004).
Word boxes were corrected by hand on downscaled page images (README.txt of the archive); the transcription of every word was assigned
by a human annotator. The line bands of the archive are automatic and only group the words into lines. The line crop spans the union
of the line band and its word boxes plus a margin; `gold_words` are the hand-corrected box x extents in crop pixels.
Licence: research use only, no redistribution (CIIR copyright notice). Evaluation only; keep this set private.
Source: """ + URL


def extract(archive: str | Path, destination: str | Path) -> Path:
    destination = Path(destination)
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile() and not Path(m.name).is_absolute() and ".." not in Path(m.name).parts]
        tar.extractall(destination, members=members)
    return destination / "gw_20p_wannot"


def read_lines(root: str | Path) -> list[tuple[str, GoldLine]]:
    root = Path(root)
    annotations = (root / "annotations.txt").read_bytes().decode("latin-1").splitlines()
    pages = (root / "file_order.txt").read_text().split()
    cursor, out = 0, []
    for page_file in pages:
        stem = page_file[:-4]
        width, height = Image.open(root / page_file).size
        rows = [r.split() for r in (root / f"{stem}_boxes.txt").read_text().splitlines()[1:] if r.strip()]
        lines: dict[tuple[str, str], list[WordBox]] = {}
        bands: dict[tuple[str, str], tuple[float, float]] = {}
        for row in rows:
            rx0, rx1, ry0, ry1 = (float(v) for v in row[:4])
            key = (row[4], row[5])
            word = annotations[cursor].strip()
            cursor += 1
            box = WordBox(word, rx0 * (width - 1), ry0 * (height - 1), rx1 * (width - 1) + 1, ry1 * (height - 1) + 1)
            lines.setdefault(key, []).append(box)
            bands[key] = (float(row[4]) * (height - 1), float(row[5]) * (height - 1) + 1)
        for number, key in enumerate(sorted(lines, key=lambda k: float(k[0])), start=1):
            words = sorted(lines[key], key=lambda b: b.x0)
            out.append((page_file, GoldLine(uid=f"gw20/{stem}/{number:02d}", doc="gw20", page=stem, words=words, band=bands[key],
                                            hand="washington")))
    if cursor != len(annotations):
        raise ValueError(f"{cursor} boxes but {len(annotations)} annotations")
    return out


def build_rows(root: str | Path) -> list[dict]:
    root = Path(root)
    rows, pages = [], {}
    for page_file, line in read_lines(root):
        if page_file not in pages:
            pages = {page_file: Image.open(root / page_file).convert("L")}
        words = [w for w in line.words if w.text]
        if not words:
            continue
        text, gold = line_from_words(words)
        rows.append(gold_row(line, pages[page_file], text, gold, words, "gw20", LICENCE, URL, language="en"))
    return rows
