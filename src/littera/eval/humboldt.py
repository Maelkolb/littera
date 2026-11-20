"""Measures of letter placement on Humboldt lines without letter gold (round 3, docs/r3/MEASURE.md).

Everything here works on the upright production crop and on placements given as n + 1 edges in crop pixels.
The neutral strip is the crop with every column shifted up or down so that the line's ink runs level: no shear, no
scaling, so a column of the strip is a column of the crop. It is built from the crop alone and is not the strip of any placer.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from littera.eval.humboldt_data import DEV_SPLITS, REPORT_SPLITS
from littera.eval.ink import otsu_level
from littera.types import word_spans

MIN_RUN = 2
CORE_PAD = 0.25
SENSITIVITY = 0.5
TOLERANCE = 1.5
NEAR = 4.0
REFS = ("k2-snap-mid", "ft2-upright")
HALVES = {"dev": DEV_SPLITS, "report": REPORT_SPLITS}
LETTER_SIZE = 32


@dataclass
class Neutral:
    grey: np.ndarray
    ink: np.ndarray
    top: int
    bottom: int
    core: np.ndarray
    whole: np.ndarray
    level: float
    paper: float
    shift: np.ndarray
    margin: int

    @property
    def width(self) -> int:
        return int(self.grey.shape[1])

    @property
    def xheight(self) -> int:
        return self.bottom - self.top

    def empty(self) -> np.ndarray:
        return self.core <= self.level

    def extent(self) -> tuple[float, float]:
        columns = np.nonzero(self.whole >= 2)[0]
        if len(columns) == 0:
            return 0.0, float(self.width)
        return float(columns[0]), float(columns[-1] + 1)

    def runs(self, min_run: int = MIN_RUN) -> list[tuple[int, int]]:
        inked = np.nonzero(~self.empty())[0]
        if len(inked) == 0:
            return []
        out, start = [], None
        empty = self.empty()
        for u in range(int(inked[0]), int(inked[-1]) + 1):
            if empty[u] and start is None:
                start = u
            elif not empty[u] and start is not None:
                if u - start >= min_run:
                    out.append((start, u))
                start = None
        return out


def _ink_curve(ink: np.ndarray) -> np.ndarray:
    h, w = ink.shape
    width = max(48, h)
    step = max(1, width // 2)
    rows = np.arange(h)
    xs, ys = [], []
    for x0 in range(0, max(1, w - width // 2), step):
        counts = ink[:, x0 : x0 + width].sum(1)
        total = counts.sum()
        if total < 40:
            continue
        ys.append(float(rows[np.searchsorted(np.cumsum(counts), total / 2)]))
        xs.append(min(w - 1, x0 + width / 2))
    if not xs:
        return np.full(w, h / 2.0)
    ys = np.array(ys)
    if len(ys) >= 3:
        padded = np.pad(ys, 1, mode="edge")
        ys = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    return np.interp(np.arange(w), xs, ys)


def neutral_strip(image: Image.Image, sensitivity: float = SENSITIVITY) -> Neutral:
    grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    h, w = grey.shape
    inside = grey < 0.985
    darkness = 1.0 - grey
    values = darkness[inside]
    level = otsu_level(values) if values.size >= 64 else 0.5
    dark = (darkness > level) & inside
    paper = float(np.median(grey[inside & ~dark])) if (inside & ~dark).any() else 1.0
    curve = _ink_curve(dark)
    ink = (darkness > (1.0 - paper) + sensitivity * (level - (1.0 - paper))) & inside
    centre = float(np.mean(curve))
    shift = np.round(centre - curve).astype(np.int64)
    margin = int(np.abs(shift).max()) + 1
    rows = np.arange(h)[:, None] + shift[None, :] + margin
    columns = np.broadcast_to(np.arange(w)[None, :], rows.shape)
    flat_grey = np.ones((h + 2 * margin, w), np.float32)
    flat_ink = np.zeros((h + 2 * margin, w), bool)
    flat_grey[rows, columns] = grey
    flat_ink[rows, columns] = ink
    flat_dark = np.zeros((h + 2 * margin, w), bool)
    flat_dark[rows, columns] = dark
    density = flat_dark.mean(1)
    density = np.convolve(np.pad(density, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    middle = int(round(centre)) + margin
    reach = max(4, int(0.25 * h))
    lo, hi = max(0, middle - reach), min(len(density), middle + reach + 1)
    peak = lo + int(np.argmax(density[lo:hi]))
    half = 0.5 * density[peak]
    top = peak
    while top > 0 and density[top - 1] >= half and peak - top < 0.4 * h:
        top -= 1
    bottom = peak + 1
    while bottom < len(density) and density[bottom] >= half and bottom - peak < 0.4 * h:
        bottom += 1
    if bottom - top < 6:
        grow = (6 - (bottom - top) + 1) // 2
        top, bottom = max(0, top - grow), min(len(density), bottom + grow)
    pad = max(2, int(round(CORE_PAD * (bottom - top))))
    core = flat_ink[max(0, top - pad) : bottom + pad].sum(0).astype(np.float32)
    whole = flat_ink.sum(0).astype(np.float32)
    return Neutral(grey=flat_grey, ink=flat_ink, top=top, bottom=bottom, core=core, whole=whole,
                   level=max(1.0, 0.02 * float(core.max(initial=0.0))), paper=paper, shift=shift, margin=margin)


def fold_of(uid: str) -> int:
    return int(hashlib.md5(uid.encode("utf-8")).hexdigest(), 16) % 2


def inner_boundaries(text: str) -> list[int]:
    return [i for i in range(1, len(text)) if not text[i - 1].isspace() and not text[i].isspace()]


def even_line(neutral: Neutral, text: str) -> list[float]:
    x0, x1 = neutral.extent()
    n = max(1, len(text))
    return [round(x0 + (x1 - x0) * i / n, 1) for i in range(len(text) + 1)]


def even_inside(text: str, x: list[float]) -> list[float]:
    out = list(x)
    for a, b in word_spans(text):
        for i in range(a, b + 1):
            out[i] = round(x[a] + (x[b] - x[a]) * (i - a) / (b - a), 1)
    return out


def word_boundary_hits(neutral: Neutral, text: str, x: list[float]) -> tuple[int, int]:
    spans = word_spans(text)
    empty = neutral.empty()
    hits = 0
    for (_, a1), (b0, _) in zip(spans, spans[1:]):
        u = 0.5 * (x[a1] + x[b0])
        lo, hi = max(0, math.ceil(u - TOLERANCE - 0.5)), min(len(empty) - 1, math.floor(u + TOLERANCE - 0.5))
        hits += bool(hi >= lo and empty[lo : hi + 1].any())
    return hits, max(0, len(spans) - 1)


def _locate(spans: list[tuple[int, int]], x: list[float], c: float) -> tuple[str, int]:
    if not spans or c <= x[spans[0][0]] or c >= x[spans[-1][1]]:
        return "outside", -1
    for w, (a, b) in enumerate(spans):
        if x[a] < c < x[b]:
            return "lift", w
        if c <= x[a]:
            return "gap", w - 1
    return "outside", -1


def classify_runs(neutral: Neutral, text: str, refs: list[list[float]]) -> list[dict]:
    spans = word_spans(text)
    out = []
    for s, e in neutral.runs():
        found = {_locate(spans, x, 0.5 * (s + e)) for x in refs}
        kind, word = next(iter(found)) if len(found) == 1 else ("disputed", -1)
        out.append({"s": s, "e": e, "kind": kind, "word": word})
    return out


def connected_letters(text: str, runs: list[dict], refs: list[list[float]]) -> set[int]:
    spans = word_spans(text)
    centres = [0.5 * (r["s"] + r["e"]) for r in runs if r["kind"] == "disputed"]
    lifted = {r["word"] for r in runs if r["kind"] == "lift"}
    out: set[int] = set()
    for w, (a, b) in enumerate(spans):
        if w in lifted or b - a < 3:
            continue
        lo, hi = min(x[a] for x in refs), max(x[b] for x in refs)
        if any(lo < c < hi for c in centres):
            continue
        out.update(range(a + 1, b - 1))
    return out


def lift_cover(text: str, runs: list[dict], x: list[float], tolerance: float) -> list[int | None]:
    spans = word_spans(text)
    out = []
    for r in runs:
        if r["kind"] != "lift":
            continue
        a, b = spans[r["word"]]
        inside = [i for i in range(a + 1, b) if r["s"] - tolerance <= x[i] <= r["e"] + tolerance]
        out.append(inside[0] if inside else None)
    return out


def bootstrap(num, den, rounds: int = 1000, seed: int = 0) -> list[float] | None:
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    if den.sum() <= 0:
        return None
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(num), size=(rounds, len(num)))
    sums = den[picks].sum(1)
    values = 100.0 * num[picks].sum(1) / np.where(sums > 0, sums, 1)
    lo, hi = np.percentile(values[sums > 0], [2.5, 97.5])
    return [round(100.0 * float(num.sum() / den.sum()), 2), round(float(lo), 2), round(float(hi), 2), int(den.sum())]


def bootstrap_difference(num_a, num_b, den, rounds: int = 1000, seed: int = 0) -> list[float] | None:
    num_a, num_b, den = (np.asarray(v, dtype=np.float64) for v in (num_a, num_b, den))
    if den.sum() <= 0:
        return None
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(den), size=(rounds, len(den)))
    sums = den[picks].sum(1)
    values = 100.0 * (num_a[picks].sum(1) - num_b[picks].sum(1)) / np.where(sums > 0, sums, 1)
    lo, hi = np.percentile(values[sums > 0], [2.5, 97.5])
    return [round(100.0 * float((num_a.sum() - num_b.sum()) / den.sum()), 2), round(float(lo), 2), round(float(hi), 2)]


def groups_of(halves: list[str]) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for half in halves:
        for split in HALVES[half]:
            out[split] = (split,)
        out[half] = HALVES[half]
    return out


def inner_kept(text: str, inner: list[float], outer: list[float]) -> list[float]:
    out = list(outer)
    for a, b in word_spans(text):
        for i in range(a + 1, b):
            out[i] = min(max(inner[i], outer[a]), outer[b])
    return out


def base_names(names) -> list[str]:
    out = []
    for name in names:
        out += name.removeprefix("mix:").split("+") if name.startswith("mix:") else [name.removeprefix("even@")]
    return list(dict.fromkeys(out))


def resolve(name: str, placements: dict[str, dict], lines: dict[str, dict]) -> dict[str, list[float]]:
    if name.startswith("mix:"):
        inner, outer = name.removeprefix("mix:").split("+")
        return {uid: inner_kept(lines[uid]["text"], placements[inner][uid]["x"], row["x"]) for uid, row in placements[outer].items()
                if uid in lines and uid in placements[inner]}
    if name.startswith("even@"):
        base = placements[name.removeprefix("even@")]
        return {uid: even_inside(lines[uid]["text"], row["x"]) for uid, row in base.items() if uid in lines}
    return {uid: row["x"] for uid, row in placements[name].items() if uid in lines}


def score(lines: dict[str, dict], placements: dict[str, dict], names: list[str], halves: list[str], refs: tuple[str, ...] = REFS) -> dict:
    """``lines``: uid -> {"text", "split", "neutral"}; ``placements``: name -> uid -> row with "x"."""
    edges = {name: resolve(name, placements, lines) for name in dict.fromkeys([*names, *refs])}
    groups = groups_of(halves)
    wanted = {split for splits in groups.values() for split in splits}
    uids = [uid for uid, line in lines.items() if line["split"] in wanted and all(uid in edges[name] for name in edges)]
    per_line = {}
    for uid in uids:
        line = lines[uid]
        runs = classify_runs(line["neutral"], line["text"], [edges[r][uid] for r in refs])
        row = {"split": line["split"], "inner": len(inner_boundaries(line["text"])), "runs": len(runs),
               "lifts": sum(r["kind"] == "lift" for r in runs), "gaps": sum(r["kind"] == "gap" for r in runs),
               "disputed": sum(r["kind"] == "disputed" for r in runs), "words": len(word_spans(line["text"])), "placers": {}}
        letters = sum(not ch.isspace() for ch in line["text"])
        row["letters"] = letters
        row["connected_letters"] = len(connected_letters(line["text"], runs, [edges[r][uid] for r in refs]))
        first = lift_cover(line["text"], runs, edges[refs[0]][uid], TOLERANCE)
        for name in names:
            x = edges[name][uid]
            hits, total = word_boundary_hits(line["neutral"], line["text"], x)
            cover = lift_cover(line["text"], runs, x, TOLERANCE)
            near = lift_cover(line["text"], runs, x, NEAR)
            both = [(a, b) for a, b in zip(cover, first) if a is not None and b is not None]
            inner = inner_boundaries(line["text"])
            empty = line["neutral"].empty()
            on_gap = sum(bool(0 <= int(x[i]) < len(empty) and empty[int(x[i])]) for i in inner)
            row["placers"][name] = {"wb": hits, "wb_n": total, "cover": sum(c is not None for c in cover),
                                    "near": sum(c is not None for c in near), "both": len(both), "same": sum(a == b for a, b in both),
                                    "on_gap": on_gap}
        per_line[uid] = row
    out = {"refs": list(refs), "thresholds": {"min_run_px": MIN_RUN, "tolerance_px": TOLERANCE, "near_px": NEAR, "core_pad_xheights": CORE_PAD, "ink_sensitivity": SENSITIVITY, "gap_level": "max(1, 2 % of the largest column count in the core band)"},
           "lines": len(uids), "hand": {}, "placers": {name: {} for name in names}}
    for group, splits in groups.items():
        rows = [r for r in per_line.values() if r["split"] in splits]
        if not rows:
            continue
        column = lambda key: [r[key] for r in rows]  # noqa: E731
        out["hand"][group] = {
            "lines": len(rows), "inner_boundaries": sum(column("inner")), "runs": sum(column("runs")), "lifts": sum(column("lifts")),
            "word_gaps": sum(column("gaps")), "disputed": sum(column("disputed")), "word_boundaries": sum(r["words"] - 1 for r in rows if r["words"]),
            "lift_rate": bootstrap(column("lifts"), column("inner")),
            "word_gap_runs_per_boundary": bootstrap(column("gaps"), [max(0, r["words"] - 1) for r in rows]),
            "letters_in_connected_words": bootstrap(column("connected_letters"), column("letters")),
        }
        for name in names:
            p = [r["placers"][name] for r in rows]
            out["placers"][name][group] = {
                "word_boundary": bootstrap([v["wb"] for v in p], [v["wb_n"] for v in p]),
                "lift_cover": bootstrap([v["cover"] for v in p], column("lifts")),
                "lift_near": bootstrap([v["near"] for v in p], column("lifts")),
                "lift_same_boundary_as_first_ref": bootstrap([v["same"] for v in p], [v["both"] for v in p]),
                "inner_edges_on_empty_column": bootstrap([v["on_gap"] for v in p], column("inner")),
            }
            if refs[0] in names and name != refs[0]:
                base = [r["placers"][refs[0]] for r in rows]
                out["placers"][name][group]["word_boundary_minus_first_ref"] = bootstrap_difference(
                    [v["wb"] for v in p], [v["wb"] for v in base], [v["wb_n"] for v in p])
                out["placers"][name][group]["lift_cover_minus_first_ref"] = bootstrap_difference(
                    [v["cover"] for v in p], [v["cover"] for v in base], column("lifts"))
    return out


def width_control(lines: dict[str, dict], placements: dict[str, dict], name: str, bins: int = 20, min_class: int = 10) -> dict[str, dict]:
    edges = resolve(name, placements, lines)
    letters = {}
    for uid, x in edges.items():
        text = lines[uid]["text"]
        index = [i for i, ch in enumerate(text) if not ch.isspace()]
        widths = np.array([x[i + 1] - x[i] for i in index], dtype=np.float64)
        positive = widths[widths > 0]
        letters[uid] = ([text[i] for i in index], widths / (float(np.median(positive)) if len(positive) else 1.0))
    out = {}
    for test_fold in (0, 1):
        train = [uid for uid in letters if fold_of(uid) != test_fold]
        chars = [ch for uid in train for ch in letters[uid][0]]
        counts: dict[str, int] = {}
        for ch in chars:
            counts[ch] = counts.get(ch, 0) + 1
        classes = {ch for ch, n in counts.items() if n >= min_class}
        commonest = max(classes, key=lambda ch: counts[ch])
        rel = np.concatenate([letters[uid][1] for uid in train])
        cuts = np.unique(np.quantile(rel, np.linspace(0, 1, bins + 1)[1:-1]))
        slots = np.searchsorted(cuts, rel)
        votes: dict[int, dict[str, int]] = {}
        for slot, ch in zip(slots, chars):
            if ch in classes:
                votes.setdefault(int(slot), {})[ch] = votes.setdefault(int(slot), {}).get(ch, 0) + 1
        choice = {slot: max(v, key=v.get) for slot, v in votes.items()}
        for uid in letters:
            if fold_of(uid) != test_fold:
                continue
            kept = [(ch, int(slot)) for ch, slot in zip(letters[uid][0], np.searchsorted(cuts, letters[uid][1])) if ch in classes]
            out[uid] = {"split": lines[uid]["split"], "n": len(kept), "ok": sum(choice.get(slot, commonest) == ch for ch, slot in kept),
                        "base": sum(ch == commonest for ch, _ in kept)}
    return out


def letter_band(neutral: Neutral) -> tuple[int, int]:
    xh = neutral.xheight
    return neutral.top - int(round(1.2 * xh)), neutral.bottom + int(round(1.0 * xh))


def letter_crops(neutral: Neutral, text: str, x: list[float], size: int = LETTER_SIZE) -> tuple[np.ndarray, list[int]]:
    r0, r1 = letter_band(neutral)
    h, w = neutral.grey.shape
    dark = np.clip((neutral.paper - neutral.grey) / max(neutral.paper, 1e-3), 0, 1)
    dark[neutral.grey >= 0.985] = 0.0
    band = np.zeros((r1 - r0, w), np.float32)
    lo, hi = max(0, r0), min(h, r1)
    band[lo - r0 : hi - r0] = dark[lo:hi]
    images, index = [], []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        c0, c1 = max(0, int(math.floor(x[i]))), min(w, int(math.ceil(x[i + 1])))
        if c1 - c0 < 1:
            tile = np.zeros((size, size), np.float32)
        else:
            tile = cv2.resize(band[:, c0:c1], (size, size), interpolation=cv2.INTER_AREA if c1 - c0 >= size else cv2.INTER_LINEAR)
        images.append((255 * tile).astype(np.uint8))
        index.append(i)
    return np.stack(images) if images else np.zeros((0, size, size), np.uint8), index


def overlay(neutral: Neutral, text: str = "", x: list[float] | None = None, runs: list[dict] | None = None, tint: bool = False) -> Image.Image:
    picture = np.stack([255 * neutral.grey] * 3, axis=-1).astype(np.uint8)
    if tint:
        picture[neutral.ink] = (0.5 * picture[neutral.ink] + 0.5 * np.array([30, 90, 255])).astype(np.uint8)
    picture[neutral.top, :] = (60, 120, 220)
    picture[min(picture.shape[0] - 1, neutral.bottom), :] = (60, 120, 220)
    for r in runs or [{"s": s, "e": e, "kind": "run"} for s, e in neutral.runs()]:
        colour = {"lift": (230, 60, 60), "gap": (60, 170, 80), "disputed": (240, 170, 40)}.get(r["kind"], (150, 150, 150))
        picture[neutral.bottom + 2 : neutral.bottom + 7, r["s"] : r["e"]] = colour
    if x is not None:
        for i, edge in enumerate(x):
            column = int(np.clip(round(edge), 0, picture.shape[1] - 1))
            word_edge = i == 0 or i == len(text) or text[i - 1].isspace() or text[i].isspace()
            rows = slice(0, picture.shape[0]) if word_edge else slice(neutral.top - 6, neutral.bottom + 6)
            picture[rows, column] = (20, 20, 20) if word_edge else (200, 40, 200)
    return Image.fromarray(picture)
