from __future__ import annotations

import json
import random
from collections import defaultdict

import numpy as np

from littera.eval.ink import ink_profile
from littera.eval.mismatch import POOL_COLUMNS
from littera.eval.sets import decode_image

HELDOUT_EXTRA_TEST_SOURCES = ("tridis", "aaeb")

README = """Candidate lines for the owner's letter-boundary marking (kind `letter`, `gold_cuts` empty until marked). Drawn from every
held-out source of the pool and from the test pages of two training sources with other scripts and languages (`tridis`: medieval
Latin/French/Dutch cursive and textualis, abbreviations expanded in the text; `aaeb`: French cursive of the 16th to 18th century).
Printed antiqua and typewriter rows are left out (Fraktur print stays, from `austrian_newspapers`), and every text occurs once
per source. Per source about 60 lines with 15 to 60 characters,
crop height at least 40 px and a plausible width per character were drawn across pages. Each was scored by aligner-k2 and
McCATMuS (`line_fit`, the text's CTC likelihood against the model's own reading) and by the share of ink outside the line's core
band on the unmasked crop (neighbouring lines, ascenders of other lines). The best fit of the two models minus 1.5 times the
outside share ranks the lines; the best were kept, at most one per page where the source has enough pages. The origin column
records the scores. Licences: those of the pool rows. `laubmann` (internal, held out since round 2) was added later by the same
rules with its own draw (`modal/eval.py::add_candidates`); the round 1 rows are unchanged (`lines_round1.parquet`)."""


def plausible(row: dict) -> bool:
    text = row["text"]
    letters = sum(ch.isalpha() for ch in text)
    per_char = row["width"] / max(1, len(text))
    if row.get("script") in ("antiqua", "typewriter"):
        return False
    plausible_width = 0.12 * row["height"] <= per_char <= 1.6 * row["height"]
    return 15 <= len(text) <= 60 and row["height"] >= 40 and letters >= 0.6 * len(text) and plausible_width


def outside_share(image) -> float:
    profile = ink_profile(image)
    total = float(profile.mask.sum())
    if total == 0:
        return 1.0
    inside = float(profile.mask[profile.top:profile.bottom].sum())
    return 1.0 - inside / total


def build_candidates(pool_root, placers, total: int = 120, seed: int = 5, draw: int = 60,
                     extra_sources: tuple[str, ...] = HELDOUT_EXTRA_TEST_SOURCES, sources: list[str] | None = None,
                     exclude: tuple[str, ...] = ()) -> tuple[list[dict], dict]:
    from littera.data.pool import read_index, read_rows

    rng = random.Random(seed)
    placer_names = [getattr(p, "name", str(k)) for k, p in enumerate(placers)]
    columns = ["uid", "source", "page", "hand", "split", "text", "width", "height", "script"]
    heldout = read_index(pool_root, split="heldout", sources=sources, columns=columns).to_pylist()
    test = read_index(pool_root, split="test", sources=list(extra_sources), columns=columns).to_pylist() if extra_sources else []
    by_source: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    seen_texts = set()
    for row in heldout + test:
        if plausible(row) and (row["source"], row["text"]) not in seen_texts:
            seen_texts.add((row["source"], row["text"]))
            by_source[row["source"]][row["page"]].append(row)
    quota = max(1, round(total / max(1, len(by_source))))
    drawn = []
    for source, pages in sorted(by_source.items()):
        page_ids = sorted(pages)
        rng.shuffle(page_ids)
        for p in page_ids:
            rng.shuffle(pages[p])
        picked, round_ = [], 0
        while len(picked) < draw and any(len(pages[p]) > round_ for p in page_ids):
            for p in page_ids:
                if len(picked) < draw and len(pages[p]) > round_:
                    picked.append(pages[p][round_])
            round_ += 1
        drawn += picked
    full = {row["uid"]: row for row in read_rows([r["uid"] for r in drawn], pool_root)}
    scored = defaultdict(list)
    for row in drawn:
        record = full[row["uid"]]
        image = decode_image(record["image"])
        polygon = [tuple(p) for p in record["polygon"]] if record["polygon"] and len(record["polygon"]) >= 3 else None
        fits = []
        for placer in placers:
            try:
                fits.append(float(placer.place(image, record["text"], polygon=polygon).line_fit))
            except Exception:  # noqa: BLE001
                fits.append(0.0)
        fit = max(fits)
        outside = outside_share(image)
        scored[row["source"]].append((fit - 1.5 * outside, fit, outside, record, fits))
    rows, stats = [], {"quota": quota, "drawn": len(drawn), "per_source": {}}
    for source, items in sorted(scored.items()):
        items.sort(key=lambda item: -item[0])
        cap = max(1, -(-quota // len(by_source[source])))
        per_page: dict[str, int] = defaultdict(int)
        kept = []
        for score, fit, outside, record, fits in items:
            if len(kept) >= quota:
                break
            if per_page[record["page"]] >= cap or record["uid"] in exclude:
                continue
            per_page[record["page"]] += 1
            kept.append((score, fit, outside, record, fits))
        stats["per_source"][source] = {"kept": len(kept), "median_fit": float(np.median([k[1] for k in kept])) if kept else None}
        for score, fit, outside, record, fits in kept:
            origin = json.dumps({"pool_origin": record["origin"], "line_fit": dict(zip(placer_names, [round(f, 4) for f in fits])),
                                 "outside_share": round(outside, 4)})
            rows.append({**{k: record[k] for k in POOL_COLUMNS}, "origin": origin, "kind": "letter", "gold_words": [], "gold_cuts": []})
    stats["rows"] = len(rows)
    return rows, stats
