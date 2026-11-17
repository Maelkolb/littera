from __future__ import annotations

import math
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from lxml import etree
from PIL import Image

from littera.types import Placement, Placer, Point

WORD_FOLLOWERS = {"TextEquiv", "TextStyle", "UserDefined", "Labels"}


@dataclass
class PageLine:
    element: etree._Element
    id: str
    polygon: list[Point]
    baseline: list[Point]
    text: str


def namespace(tree: etree._ElementTree) -> str:
    tag = tree.getroot().tag
    return tag[1 : tag.index("}")] if tag.startswith("{") else ""


def parse_points(value: str | None) -> list[Point]:
    points = []
    for pair in (value or "").split():
        x, y = pair.split(",")[:2]
        points.append((float(x), float(y)))
    return points


def format_points(points: list[Point]) -> str:
    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in points)


def _local(element: etree._Element) -> str:
    return etree.QName(element).localname if isinstance(element.tag, str) else ""


def _child(element: etree._Element, name: str) -> etree._Element | None:
    return next((c for c in element if _local(c) == name), None)


def line_text(line: etree._Element) -> str:
    equivs = [c for c in line if _local(c) == "TextEquiv"]
    equivs.sort(key=lambda e: int(e.get("index", "0")) if e.get("index", "0").lstrip("-").isdigit() else 0)
    for equiv in equivs:
        unicode = _child(equiv, "Unicode")
        if unicode is not None and unicode.text:
            return unicode.text
    return ""


def read_lines(tree: etree._ElementTree) -> list[PageLine]:
    lines = []
    for element in tree.getroot().iter():
        if _local(element) != "TextLine":
            continue
        coords, baseline = _child(element, "Coords"), _child(element, "Baseline")
        lines.append(PageLine(element=element, id=element.get("id", f"line{len(lines)}"),
                              polygon=parse_points(coords.get("points")) if coords is not None else [],
                              baseline=parse_points(baseline.get("points")) if baseline is not None else [], text=line_text(element)))
    return lines


def crop_box(line: PageLine, size: tuple[int, int], margin: int, line_height: float) -> tuple[int, int, int, int]:
    if len(line.polygon) >= 3:
        xs, ys = [p[0] for p in line.polygon], [p[1] for p in line.polygon]
        top, bottom = min(ys), max(ys)
    else:
        xs, ys = [p[0] for p in line.baseline], [p[1] for p in line.baseline]
        top, bottom = min(ys) - 0.8 * line_height, max(ys) + 0.3 * line_height
    x0 = max(0, int(math.floor(min(xs))) - margin)
    y0 = max(0, int(math.floor(top)) - margin)
    x1 = min(size[0], int(math.ceil(max(xs))) + margin)
    y1 = min(size[1], int(math.ceil(bottom)) + margin)
    return x0, y0, max(x1, x0 + 2), max(y1, y0 + 2)


def _shift(points: list[Point], dx: float, dy: float) -> list[Point]:
    return [(x + dx, y + dy) for x, y in points]


def write_words(line: PageLine, placement: Placement, offset: tuple[float, float], ns: str) -> None:
    element = line.element
    for old in [c for c in element if _local(c) == "Word"]:
        element.remove(old)
    position = next((k for k, c in enumerate(element) if _local(c) in WORD_FOLLOWERS), len(element))
    dx, dy = offset

    def make(tag: str, **attributes) -> etree._Element:
        return etree.Element(f"{{{ns}}}{tag}" if ns else tag, **attributes)

    def equiv(text: str, conf: float) -> etree._Element:
        node = make("TextEquiv", conf=f"{conf:.3f}")
        unicode = make("Unicode")
        unicode.text = text
        node.append(unicode)
        return node

    for k, word in enumerate(placement.words):
        node = make("Word", id=f"{line.id}_w{k}", custom=f"littera {{fit:{word.fit:.3f};}}")
        node.append(make("Coords", points=format_points(_shift(word.poly, dx, dy))))
        glyphs: list[tuple[str, object]] = []
        for char in placement.chars[word.i0 : word.i1]:
            if glyphs and unicodedata.combining(char.ch):
                glyphs[-1] = (glyphs[-1][0] + char.ch, glyphs[-1][1])
            elif char.visible:
                glyphs.append((char.ch, char))
        for g, (text, char) in enumerate(glyphs):
            glyph = make("Glyph", id=f"{line.id}_w{k}_g{g}")
            glyph.append(make("Coords", points=format_points(_shift(char.poly, dx, dy))))
            glyph.append(equiv(text, char.q))
            node.append(glyph)
        node.append(equiv(placement.text[word.i0 : word.i1], word.q))
        element.insert(position, node)
        position += 1


def place_page(xml_path, image_path, placer: Placer, out_path, margin: int = 8) -> dict:
    tree = etree.parse(str(xml_path))
    ns = namespace(tree)
    image = Image.open(image_path)
    image.load()
    lines = read_lines(tree)
    heights = [max(p[1] for p in ln.polygon) - min(p[1] for p in ln.polygon) for ln in lines if len(ln.polygon) >= 3]
    line_height = statistics.median(heights) if heights else 60.0
    placed, fits = 0, []
    for line in lines:
        if not line.text.strip() or (len(line.polygon) < 3 and len(line.baseline) < 2):
            continue
        box = crop_box(line, image.size, margin, line_height)
        crop = image.crop(box)
        polygon = _shift(line.polygon, -box[0], -box[1]) if len(line.polygon) >= 3 else None
        baseline = _shift(line.baseline, -box[0], -box[1]) if len(line.baseline) >= 2 else None
        placement = placer.place(crop, line.text, polygon, baseline)
        write_words(line, placement, (box[0], box[1]), ns)
        placed += 1
        fits.append(placement.line_fit)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path), encoding="UTF-8", xml_declaration=True, pretty_print=True)
    return {"lines": len(lines), "placed": placed, "mean_line_fit": sum(fits) / len(fits) if fits else None}
