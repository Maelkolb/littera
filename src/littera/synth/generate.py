from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from littera.synth.degrade import compose, finish
from littera.synth.fonts import load_catalogue
from littera.synth.render import clean_text, font_covers, layout_line, place_geometry, size_for_x_height

SLANTED_STYLES = {"script", "hand", "kurrent"}


@dataclass
class SynthLine:
    image: Image.Image
    text: str
    gold_words: list[dict]
    gold_cuts: list[dict]
    polygon: list[list[float]]
    baseline: list[list[float]]
    font: dict
    params: dict


class SynthGenerator:
    def __init__(self, fonts_dir: str | Path, split: str = "train", seed: int = 0, styles: Sequence[str] | None = None,
                 texts: Sequence[str] | None = None):
        fonts = load_catalogue(fonts_dir)
        self.fonts = [f for f in fonts if split in ("all", f["split"]) and (styles is None or f["style"] in styles)]
        if not self.fonts:
            raise ValueError(f"no fonts for split {split!r} in {fonts_dir}")
        self.styles = sorted({f["style"] for f in self.fonts})
        self.rng = np.random.default_rng(seed)
        self.texts = [clean_text(t) for t in texts] if texts else []

    def pick_font(self, text: str, rng: np.random.Generator) -> dict | None:
        order = list(rng.permutation(self.styles))
        for style in order:
            candidates = [f for f in self.fonts if f["style"] == style and font_covers(f["path"], text)]
            if candidates:
                return candidates[int(rng.integers(len(candidates)))]
        return None

    def neighbour_texts(self, rng: np.random.Generator, text: str) -> tuple[str, str] | None:
        if rng.random() > 0.65:
            return None
        pool = self.texts or [text]
        picks = []
        for _ in range(2):
            picks.append(pool[int(rng.integers(len(pool)))] if rng.random() < 0.85 else "")
        return picks[0], picks[1]

    def render(self, text: str, font: dict | None = None, rng: np.random.Generator | None = None) -> SynthLine | None:
        rng = rng or self.rng
        text = clean_text(text)
        if len(text) < 2:
            return None
        font = font or self.pick_font(text, rng)
        if font is None:
            return None
        neighbours = self.neighbour_texts(rng, text)
        if neighbours:
            neighbours = tuple(n if font_covers(font["path"], n) else "" for n in neighbours)
        target = float(rng.uniform(14, 38))
        size = size_for_x_height(font["path"], target)
        if font["style"] in SLANTED_STYLES:
            slant = float(np.clip(rng.normal(0.08, 0.14), -0.2, 0.45))
        else:
            slant = 0.0 if rng.random() < 0.6 else float(np.clip(rng.normal(0.0, 0.08), -0.15, 0.2))
        spacing = float(rng.uniform(2.6, 4.2))
        shifts = (float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3)))
        layout = layout_line(text, font["path"], size, neighbours, spacing, shifts)
        xh = layout.x_height
        wobble = (float(rng.uniform(0, 0.12) * xh) if rng.random() < 0.5 else 0.0, float(rng.uniform(8, 30) * xh),
                  float(rng.uniform(0, 2 * math.pi)))
        geometry = place_geometry(layout, slant, wobble)
        ink_left = min(w["x0"] for w in geometry.gold_words)
        ink_right = max(w["x1"] for w in geometry.gold_words)
        height, width = geometry.mask.shape
        left = int(max(0, math.floor(ink_left - rng.uniform(0.2, 1.5) * xh)))
        right = int(min(width, math.ceil(ink_right + rng.uniform(0.2, 1.5) * xh)))
        top = int(max(0, math.floor(geometry.baseline_y - rng.uniform(1.7, 2.8) * xh)))
        bottom = int(min(height, math.ceil(geometry.baseline_y + rng.uniform(0.9, 1.8) * xh)))
        mask = geometry.mask[top:bottom, left:right]
        image = finish(compose(mask, rng, xh), rng)
        gold_words = [{**w, "x0": w["x0"] - left, "x1": w["x1"] - left} for w in geometry.gold_words]
        gold_cuts = [{**c, "x": c["x"] - left} for c in geometry.gold_cuts]
        baseline_y = geometry.baseline_y - top
        crop_w, crop_h = right - left, bottom - top
        band_top, band_bottom = max(0.0, baseline_y - 1.9 * xh), min(float(crop_h), baseline_y + 0.9 * xh)
        polygon = [[0.0, band_top], [float(crop_w), band_top], [float(crop_w), band_bottom], [0.0, band_bottom]]
        baseline = [[0.0, baseline_y], [float(crop_w), baseline_y]]
        params = {"size": size, "x_height": round(xh, 2), "slant": round(slant, 4), "wobble": round(wobble[0], 2),
                  "neighbours": bool(neighbours and any(neighbours))}
        font_info = {k: font[k] for k in ("name", "style", "split", "licence", "file")}
        return SynthLine(image, text, gold_words, gold_cuts, polygon, baseline, font_info, params)

    def stream(self, texts: Sequence[str], seed: int | None = None) -> Iterator[SynthLine]:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        while True:
            line = self.render(texts[int(rng.integers(len(texts)))], rng=rng)
            if line is not None:
                yield line


def to_row(line: SynthLine, uid: str, source: str = "synth", geometry: bool = True, origin: str = "") -> dict:
    from littera.eval.sets import encode_image

    return {
        "uid": uid, "source": source, "doc": line.font["name"], "page": uid.rsplit("/", 1)[-1], "hand": line.font["name"],
        "split": f"synth-{line.font['split']}", "image": encode_image(line.image, "JPEG", 95), "width": line.image.width,
        "height": line.image.height, "text": line.text, "script": line.font["style"], "language": "", "licence": line.font["licence"],
        "open_licence": True, "quality": "synthetic", "polygon": line.polygon if geometry else [],
        "baseline": line.baseline if geometry else [],
        "origin": origin or json.dumps({"font": line.font, **line.params}, ensure_ascii=False),
        "kind": "synthetic", "gold_words": line.gold_words, "gold_cuts": line.gold_cuts,
    }


def build_set(fonts_dir: str, texts: Sequence[str], count: int, split: str, seed: int, out: str, name: str,
              text_sources: Sequence[str] | None = None) -> dict:
    from littera.eval.sets import set_stats, write_readme, write_set

    generator = SynthGenerator(fonts_dir, split=split, seed=seed, texts=texts)
    rng = np.random.default_rng(seed + 1)
    rows, attempts = [], 0
    order = rng.permutation(len(texts))
    while len(rows) < count and attempts < 10 * count:
        index = int(order[attempts % len(order)])
        attempts += 1
        line = generator.render(texts[index], rng=rng)
        if line is None:
            continue
        text_from = text_sources[index] if text_sources else ""
        origin = json.dumps({"font": line.font, **line.params, "text_from": text_from}, ensure_ascii=False)
        rows.append(to_row(line, f"{name}/{len(rows):05d}", geometry=bool(rng.random() < 0.5), origin=origin))
    path = Path(out) / name / "lines.parquet"
    write_set(rows, path)
    stats = set_stats(rows)
    stats["fonts"] = {}
    for row in rows:
        stats["fonts"][row["doc"]] = stats["fonts"].get(row["doc"], 0) + 1
    licences = sorted({(f["name"], f["licence"]) for f in generator.fonts})
    write_readme(path.parent, name,
                 f"Synthetic lines rendered by `littera.synth` with the {split} fonts. Texts are lines of the pool's held-out sources "
                 "(text only, the images are rendered). Gold word extents are the ink extents of each word rendered alone at its pen "
                 "position; gold cuts are the pen positions (cumulative advance of the rendered prefix), taken at mid x-height after "
                 "the slant. Half of the rows carry a polygon and baseline. Fonts and licences:\n\n"
                 + "\n".join(f"- {n}: {lic}" for n, lic in licences), stats)
    return stats


POOL_SCRIPT = {"script": "latin_cursive", "hand": "latin_cursive", "kurrent": "kurrent", "fraktur": "fraktur", "medieval": "textualis",
               "print": "antiqua"}


def training_schema():
    import pyarrow as pa

    from littera.data.schema import POOL_SCHEMA
    from littera.eval.sets import GOLD_CUT, GOLD_WORD

    return pa.schema(list(POOL_SCHEMA) + [pa.field("gold_words", pa.list_(GOLD_WORD)), pa.field("gold_cuts", pa.list_(GOLD_CUT))])


def training_rows(fonts_dir: str, texts: Sequence[dict], count: int, seed: int, part: int, split: str = "train",
                  font_split: str = "train") -> list[dict]:
    from littera.eval.sets import encode_image

    generator = SynthGenerator(fonts_dir, split=font_split, seed=seed, texts=[t["text"] for t in texts])
    rng = np.random.default_rng(seed)
    rows, attempts = [], 0
    while len(rows) < count and attempts < 10 * count:
        attempts += 1
        source = texts[int(rng.integers(len(texts)))]
        line = generator.render(source["text"], rng=rng)
        if line is None:
            continue
        geometry = bool(rng.random() < 0.5)
        font_slug = line.font["file"].split("/")[0]
        rows.append({
            "uid": f"synth/{font_slug}/p{part:05d}/{len(rows)}", "source": "synth", "doc": font_slug, "page": f"p{part:05d}",
            "hand": font_slug, "split": split, "image": encode_image(line.image, "JPEG", 92), "width": line.image.width,
            "height": line.image.height, "text": line.text, "script": POOL_SCRIPT.get(line.font["style"], "mixed"),
            "language": source.get("language") or "",
            "licence": f"synthetic; font {line.font['licence']}; text {source.get('licence') or ''}",
            "open_licence": True, "quality": "synthetic", "polygon": line.polygon if geometry else [],
            "baseline": line.baseline if geometry else [],
            "origin": json.dumps({"font": line.font, **line.params, "text_from": source.get("uid", "")}, ensure_ascii=False),
            "gold_words": line.gold_words, "gold_cuts": line.gold_cuts,
        })
    return rows


def write_training_part(fonts_dir: str, texts: Sequence[dict], count: int, seed: int, part: int, out_dir: str | Path,
                        split: str = "train") -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = training_rows(fonts_dir, texts, count, seed, part, split)
    path = Path(out_dir) / "lines" / "synth" / f"part-{part:05d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=training_schema()), path, row_group_size=512)
    fonts: dict[str, int] = {}
    for row in rows:
        fonts[row["doc"]] = fonts.get(row["doc"], 0) + 1
    return {"part": part, "rows": len(rows), "path": str(path), "fonts": fonts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m littera.synth.generate")
    parser.add_argument("--fonts", default="data/fonts")
    parser.add_argument("--texts", required=True, help="a parquet with a text column (and uid) or a UTF-8 text file, one line each")
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/eval")
    parser.add_argument("--name", default="synth-heldfonts")
    args = parser.parse_args(argv)
    if args.texts.endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(args.texts, columns=["uid", "text"]).to_pylist()
        texts, sources = [r["text"] for r in table], [r["uid"] for r in table]
    else:
        texts = [t for t in Path(args.texts).read_text(encoding="utf-8").splitlines() if t.strip()]
        sources = None
    print(json.dumps(build_set(args.fonts, texts, args.count, args.split, args.seed, args.out, args.name, sources), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
