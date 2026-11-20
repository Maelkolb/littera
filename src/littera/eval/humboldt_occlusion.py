"""Where a reader that is no aligner needs the ink of each token (TrOCR t11 of humboldt-htr, teacher-forced).

A paper-coloured bar slides along the line; for every token of the text the fall of its log-probability is recorded per bar
position. The token's evidence centre is the centre of mass of that fall around its peak. The evidence is computed once per line
and knows nothing of any placer; a placer is scored by where its span of the token's characters lies against the centre.
"""

from __future__ import annotations

import numpy as np

from littera.eval.humboldt import bootstrap, groups_of, resolve

BAR_LETTERS = 2.0
STEP_LETTERS = 0.5
MIN_PEAK = 1.0
SIDE = 384


def line_letter(width: int, text: str) -> float:
    return max(4.0, width / max(1, len(text)))


def evidence(model, tokenizer, crop, text: str, device: str = "cuda", batch: int = 96) -> dict:
    """``crop``: RGB PIL image. Returns the tokens with their character range, evidence centre in crop pixels and peak fall."""
    import torch
    from PIL import Image

    encoded = tokenizer(text, truncation=True, max_length=96, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    labels = torch.tensor(ids, device=device)
    start = model.config.decoder_start_token_id
    decoder_in = torch.cat([torch.tensor([start], device=device), labels[:-1]])[None]
    small = torch.from_numpy(np.asarray(crop.convert("RGB").resize((SIDE, SIDE), Image.BILINEAR), dtype=np.uint8).copy()).to(device)
    inside = small.min(2).values < 250
    paper = small[inside].float().median(0).values if inside.any() else torch.full((3,), 255.0, device=device)
    letter = line_letter(crop.width, text)
    to_small = SIDE / crop.width
    centres = np.arange(0.0, crop.width + 1e-6, STEP_LETTERS * letter)
    columns = torch.arange(SIDE, device=device)

    def log_probs(pictures: torch.Tensor) -> torch.Tensor:
        pixels = ((pictures.permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5).half()
        with torch.inference_mode():
            logits = model(pixel_values=pixels, decoder_input_ids=decoder_in.expand(len(pictures), -1)).logits.float()
        return torch.log_softmax(logits, -1).gather(-1, labels[None, :, None].expand(len(pictures), -1, 1)).squeeze(-1)

    base = log_probs(small[None])[0]
    falls = []
    for k in range(0, len(centres), batch):
        part = centres[k : k + batch]
        pictures = small[None].repeat(len(part), 1, 1, 1)
        for row, centre in enumerate(part):
            lo, hi = (centre - 0.5 * BAR_LETTERS * letter) * to_small, (centre + 0.5 * BAR_LETTERS * letter) * to_small
            bar = (columns + 0.5 >= lo) & (columns + 0.5 <= hi)
            cover = inside & bar[None, :]
            pictures[row][cover] = paper.to(pictures.dtype)
        falls.append((base[None] - log_probs(pictures)).clamp(min=0))
    fall = torch.cat(falls).cpu().numpy()
    tokens = []
    for t, (c0, c1) in enumerate(offsets):
        chars = [i for i in range(c0, c1) if not text[i].isspace()]
        if not chars:
            continue
        curve = fall[:, t]
        peak = int(curve.argmax())
        lo = hi = peak
        while lo > 0 and curve[lo - 1] >= 0.5 * curve[peak]:
            lo -= 1
        while hi < len(curve) - 1 and curve[hi + 1] >= 0.5 * curve[peak]:
            hi += 1
        mass = curve[lo : hi + 1]
        centre = float((mass * centres[lo : hi + 1]).sum() / mass.sum()) if mass.sum() > 0 else float(centres[peak])
        tokens.append({"c0": chars[0], "c1": chars[-1] + 1, "centre": round(centre, 1), "peak": round(float(curve[peak]), 3),
                       "base": round(float(base[t]), 3)})
    return {"letter": round(letter, 2), "bars": len(centres), "tokens": tokens}


def _ratio(num, den, scale: float) -> list[float] | None:
    value = bootstrap(num, den)
    return None if value is None else [round(value[0] * scale / 100, 3), round(value[1] * scale / 100, 3), round(value[2] * scale / 100, 3), value[3]]


def score(lines: dict[str, dict], placements: dict[str, dict], found: dict[str, dict], names: list[str], halves: list[str]) -> dict:
    """``lines``: uid -> {"text", "split"}; ``found``: uid -> the output of ``evidence``."""
    out = {"thresholds": {"bar_letters": BAR_LETTERS, "step_letters": STEP_LETTERS, "min_peak_nats": MIN_PEAK}, "placers": {}}
    for name in names:
        edges = resolve(name, placements, lines)
        rows = {}
        for uid, x in edges.items():
            if uid not in found:
                continue
            tokens = [t for t in found[uid]["tokens"] if t["peak"] >= MIN_PEAK]
            letter = found[uid]["letter"]
            inside = sum(x[t["c0"]] <= t["centre"] <= x[t["c1"]] for t in tokens)
            off = [(t["centre"] - 0.5 * (x[t["c0"]] + x[t["c1"]])) / letter for t in tokens]
            rows[uid] = {"split": lines[uid]["split"], "n": len(tokens), "inside": inside, "abs": float(np.abs(off).sum()), "signed": float(np.sum(off)),
                         "all": len(found[uid]["tokens"])}
        out["placers"][name] = {}
        for group, splits in groups_of(halves).items():
            part = [r for r in rows.values() if r["split"] in splits]
            if not part:
                continue
            n = [r["n"] for r in part]
            out["placers"][name][group] = {
                "centre_inside_span": bootstrap([r["inside"] for r in part], n),
                "mean_offset_letters": _ratio([r["abs"] for r in part], n, 1.0),
                "mean_signed_offset_letters": _signed([r["signed"] for r in part], n),
                "tokens_with_evidence": bootstrap(n, [r["all"] for r in part]),
            }
    return out


def _signed(num, den, rounds: int = 1000, seed: int = 0) -> list[float] | None:
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    if den.sum() <= 0:
        return None
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(num), size=(rounds, len(num)))
    sums = den[picks].sum(1)
    values = num[picks].sum(1) / np.where(sums > 0, sums, 1)
    lo, hi = np.percentile(values[sums > 0], [2.5, 97.5])
    return [round(float(num.sum() / den.sum()), 3), round(float(lo), 3), round(float(hi), 3)]
