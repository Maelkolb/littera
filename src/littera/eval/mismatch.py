from __future__ import annotations

import random
from collections import defaultdict

from littera.types import word_spans

README = """Mismatch set from the pool's held-out sources (`split == heldout`). For each sampled line four rows share its image:
`right` (the line's own text, label 1), `swap` (the text of another line of the same page, or of the same document when the page
has a single line: variant `swap-doc`), `replace` (one word of at least three letters replaced by a word of similar length from
another line of the page or document), `delete` (one word removed; lines with at least three words). Label 0 for all but `right`.
`true_text` is the line's own text. Lines have 3 to 160 characters; at most 2 lines per page per source are drawn first, then more.
Licences: those of the pool rows (column `licence`); `abp` is internal archival material. Evaluation only."""

POOL_COLUMNS = ["uid", "source", "doc", "page", "hand", "split", "image", "width", "height", "text", "script", "language", "licence",
                "open_licence", "quality", "polygon", "baseline", "origin"]


def words_of(text: str) -> list[str]:
    return [text[a:b] for a, b in word_spans(text)]


def replace_word(text: str, donors: list[str], rng: random.Random) -> str | None:
    spans = [(a, b) for a, b in word_spans(text) if b - a >= 3 and any(ch.isalpha() for ch in text[a:b])]
    if not spans:
        return None
    a, b = spans[rng.randrange(len(spans))]
    target = text[a:b]
    options = [w for w in donors if w != target and abs(len(w) - len(target)) <= 2 and any(ch.isalpha() for ch in w)]
    if not options:
        options = [w for w in donors if w != target and len(w) >= 2]
    if not options:
        return None
    return text[:a] + options[rng.randrange(len(options))] + text[b:]


def delete_word(text: str, rng: random.Random) -> str | None:
    spans = word_spans(text)
    if len(spans) < 3:
        return None
    k = rng.randrange(len(spans))
    a, b = spans[k]
    if k + 1 < len(spans):
        return text[:a] + text[spans[k + 1][0]:]
    return text[: spans[k - 1][1]] + text[b:]


def choose_lines(index_rows: list[dict], per_source: int, rng: random.Random) -> list[dict]:
    by_source: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in index_rows:
        by_source[row["source"]][row["page"]].append(row)
    chosen = []
    for source, pages in sorted(by_source.items()):
        page_ids = sorted(pages)
        rng.shuffle(page_ids)
        candidates = {p: [r for r in pages[p] if 3 <= len(r["text"]) <= 160 and len(word_spans(r["text"])) >= 2] for p in page_ids}
        for p in page_ids:
            rng.shuffle(candidates[p])
        picked, round_ = [], 0
        while len(picked) < per_source and any(len(c) > round_ for c in candidates.values()):
            for p in page_ids:
                if len(picked) >= per_source:
                    break
                if len(candidates[p]) > round_:
                    picked.append(candidates[p][round_])
            round_ += 1
        chosen += picked
    return chosen


def build_mismatch_rows(pool_root, per_source: int = 120, seed: int = 11) -> tuple[list[dict], dict]:
    from littera.data.pool import read_index, read_rows

    rng = random.Random(seed)
    index = read_index(pool_root, split="heldout", columns=["uid", "source", "doc", "page", "text"]).to_pylist()
    by_page, by_doc = defaultdict(list), defaultdict(list)
    for row in index:
        by_page[(row["source"], row["page"])].append(row)
        by_doc[(row["source"], row["doc"])].append(row)
    chosen = choose_lines(index, per_source, rng)
    full = {row["uid"]: row for row in read_rows([r["uid"] for r in chosen], pool_root)}
    rows, counts = [], defaultdict(int)
    for line in chosen:
        base = {k: full[line["uid"]][k] for k in POOL_COLUMNS}
        truth = base["text"]
        same_page = [r for r in by_page[(line["source"], line["page"])] if r["text"] != truth]
        same_doc = [r for r in by_doc[(line["source"], line["doc"])] if r["text"] != truth]
        if same_page:
            swap_text, swap_variant = same_page[rng.randrange(len(same_page))]["text"], "swap"
        elif same_doc:
            swap_text, swap_variant = same_doc[rng.randrange(len(same_doc))]["text"], "swap-doc"
        else:
            swap_text, swap_variant = None, ""
        donors = [w for r in (same_page or same_doc) for w in words_of(r["text"])]
        variants = [("right", truth, 1), (swap_variant, swap_text, 0), ("replace", replace_word(truth, donors, rng), 0),
                    ("delete", delete_word(truth, rng), 0)]
        for variant, text, label in variants:
            if not text or (label == 0 and text == truth):
                continue
            counts[variant] += 1
            rows.append({**base, "uid": f"{base['uid']}#{variant}", "text": text, "kind": "mismatch", "true_text": truth,
                         "label": label, "variant": variant, "gold_words": [], "gold_cuts": []})
    return rows, {"lines": len(chosen), "variants": dict(counts)}
