from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from littera.geometry import Strip, free_runs, gap_level
from littera.text import GAP, Encoded

IGNORE = -100
KINDS = ("lift", "word", "start", "end")


def strip_runs(strip: Strip) -> list[tuple[int, int]]:
    profile = strip.core_profile()
    return free_runs(profile, gap_level(profile))


def run_at(runs: list[tuple[int, int]], u0: float, u1: float, reach: float) -> tuple[int, int] | None:
    near = [(s, e) for s, e in runs if s < u1 + reach and e > u0 - reach]
    if not near:
        return None
    return max(near, key=lambda r: (min(r[1], u1) - max(r[0], u0), r[1] - r[0]))


def gap_frames(start: float, end: float, stride: int) -> tuple[int, int]:
    first = min(int(round(start / stride)), int((start + end) / 2 // stride))
    return first, max(int(round(end / stride)), first + 1)


def _usable(analysis, gate: float) -> bool:
    return bool(analysis.words) and "fallback" not in analysis.extra and analysis.line_fit >= gate


def derive(plain, lifted, gate: float = 0.2, widest: float = 2.5) -> dict:
    """plain: Analysis decoded without lift arcs, snap "strip", lift_snap "cuts"; lifted: with lift arcs, lift_snap "lifts"."""
    record = {"gaps": [], "letters": []}
    if not (_usable(plain, gate) and _usable(lifted, gate)):
        return record
    strip, encoded = lifted.strip, lifted.encoded
    runs = set(strip_runs(strip))
    by_start = {s: (s, e) for s, e in runs}
    by_end = {e: (s, e) for s, e in runs}
    position = {unit: j for j, unit in enumerate(encoded.unit_order())}
    seen_plain = [wc for wc in plain.words if wc.units]
    seen = [wc for wc in lifted.words if wc.units]
    if len(seen) != len(seen_plain) or any(a.units != b.units for a, b in zip(seen_plain, seen)):
        return record

    def whole(edge: tuple[float, float]) -> tuple[int, int]:
        return int(round(edge[0])), int(round(edge[1]))

    gaps: dict[int, tuple[float, float, str]] = {}
    for a, b in zip(seen_plain, seen):
        for k, edge in b.edges.items():
            j = position[b.units[k]]
            if k in a.edges and whole(a.edges[k]) == whole(edge) and whole(edge) in runs and position[b.units[k - 1]] == j - 1:
                gaps[j] = (*whole(edge), "lift")
    for (a0, b0), (a1, b1) in zip(zip(seen_plain, seen), zip(seen_plain[1:], seen[1:])):
        edge = whole((b0.cuts[-1], b1.cuts[0]))
        j = position[b1.units[0]]
        if position[b0.units[-1]] == j - 1 and whole((a0.cuts[-1], a1.cuts[0])) == edge and edge in runs:
            gaps[j] = (*edge, "word")
    start, start_plain = int(round(seen[0].cuts[0])), int(round(seen_plain[0].cuts[0]))
    end, end_plain = int(round(seen[-1].cuts[-1])), int(round(seen_plain[-1].cuts[-1]))
    if start == start_plain and start in by_end:
        gaps[position[seen[0].units[0]]] = (*by_end[start], "start")
    if end == end_plain and end in by_start:
        gaps[position[seen[-1].units[-1]] + 1] = (*by_start[end], "end")
    order = encoded.unit_order()
    for j in sorted(gaps):
        s, e, kind = gaps[j]
        record["gaps"].append([j, round(float(strip.crop_x(s)), 2), round(float(strip.crop_x(e)), 2), kind])
        if j + 1 in gaps and j < len(order):
            width = gaps[j + 1][0] - e
            if 0 < width <= widest * lifted.letter:
                record["letters"].append([j, encoded.seg.units[order[j]].key])
    return record


def frame_targets(record: dict, strip: Strip, labels: list[int], frames: int, stride: int,
                  shift: float = 0.0) -> tuple[list[int], np.ndarray, np.ndarray]:
    runs = strip_runs(strip)
    targets: set[int] = set()
    known = np.zeros(frames, bool)
    frame_labels = np.full(frames, IGNORE, np.int64)
    placed: dict[int, tuple[int, int]] = {}
    last_end = -1
    for j, x0, x1, _ in sorted(record["gaps"]):
        u0, u1 = float(strip.strip_u(x0 + shift)), float(strip.strip_u(x1 + shift))
        run = run_at(runs, u0, u1, stride)
        if run is None:
            continue
        first, end = gap_frames(*run, stride)
        end = min(end, frames)
        if first <= last_end or first >= end:
            continue
        frame_labels[first:end] = GAP
        known[first:end] = True
        if first > 0:
            targets.add(first)
        if end < frames:
            targets.add(end)
            known[end] = True
        placed[j] = (first, end)
        last_end = end
    for j, _ in record["letters"]:
        if j in placed and j + 1 in placed and j < len(labels):
            a, b = placed[j][1], placed[j + 1][0]
            frame_labels[a:b] = labels[j]
            known[a:b] = True
    return sorted(targets), known, frame_labels


def lifts_at_cuts(encoded: Encoded, cut_at: dict, strip: Strip, stride: int, frame_labels: np.ndarray, targets: list[int]) -> list[int]:
    runs = strip_runs(strip)
    units = encoded.seg.units
    marked = set(targets)
    last_end = -1

    def frame(i: int) -> int:
        return int(round(float(strip.strip_u(cut_at[i])) / stride))

    for word in encoded.seg.word_units:
        for a, b in zip(word, word[1:]):
            left, right = units[a], units[b]
            if left.i1 != right.i0 or any(i not in cut_at for i in (left.i0, right.i0, right.i1)):
                continue
            u = float(strip.strip_u(cut_at[right.i0]))
            run = run_at(runs, u, u, stride)
            if run is None:
                continue
            first, end = gap_frames(*run, stride)
            start, cut, stop = frame(left.i0), frame(right.i0), frame(right.i1)
            if not (max(start, last_end, 0) < first and end < min(stop, len(frame_labels)) and start < cut < stop):
                continue
            before, after = frame_labels[cut - 1], frame_labels[cut]
            frame_labels[min(cut, first) : first] = before
            frame_labels[end : max(cut, end)] = after
            frame_labels[first:end] = GAP
            marked.discard(cut)
            marked.update((first, end))
            last_end = end
    return sorted(marked)


def write(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    records = {}
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            records[row["uid"]] = row
    return records


def for_rows(path: str | Path, uids: list[str]) -> list[dict | None]:
    records = read(path)
    return [records.get(uid) for uid in uids]


def summarise(records: list[dict]) -> dict:
    lines = max(1, len(records))
    kinds = Counter(gap[3] for record in records for gap in record["gaps"])
    letters = Counter(key for record in records for _, key in record["letters"])
    return {"lines": len(records), "lines_with_anchor": sum(1 for r in records if r["gaps"]),
            "gaps_per_1000_lines": {kind: round(1000 * kinds[kind] / lines, 1) for kind in KINDS},
            "letters_per_1000_lines": round(1000 * sum(letters.values()) / lines, 1),
            "letters": dict(letters.most_common())}


def tile(strip: Strip, left: float, right: float, caption: str, zoom: int = 3, margin: int = 28) -> Image.Image:
    lo, hi = max(0, int(left) - margin), min(strip.width, int(right) + margin)
    grey = (255 * (1 - np.clip(strip.image[:, lo:hi], 0, 1))).astype(np.uint8)
    picture = Image.fromarray(grey).convert("RGB").resize(((hi - lo) * zoom, strip.height * zoom), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(picture)
    for u in (left, right):
        x = int(round((u - lo) * zoom))
        draw.line([(x, 0), (x, picture.height)], fill=(220, 20, 20), width=2)
    draw.text((3, 2), caption, fill=(20, 60, 200))
    return picture


def sheet(tiles: list[Image.Image], columns: int = 4, gap: int = 6) -> Image.Image:
    width = max(t.width for t in tiles)
    height = max(t.height for t in tiles)
    rows = -(-len(tiles) // columns)
    page = Image.new("RGB", (columns * (width + gap) + gap, rows * (height + gap) + gap), (255, 255, 255))
    for k, picture in enumerate(tiles):
        page.paste(picture, (gap + (k % columns) * (width + gap), gap + (k // columns) * (height + gap)))
    return page
