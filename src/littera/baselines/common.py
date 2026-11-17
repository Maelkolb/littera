from __future__ import annotations

from collections.abc import Sequence

from littera.types import CharPlace, Placement, WordPlace, word_spans


def rectangle(x0: float, x1: float, top: float, bottom: float) -> list[tuple[float, float]]:
    return [(x0, top), (x1, top), (x1, bottom), (x0, bottom)]


def placement_from_cuts(text: str, cuts: Sequence[float], width: int, height: int, q: Sequence[float] | None = None,
                        line_fit: float = 0.5, word_fits: Sequence[float] | None = None, aligned: Sequence[bool] | None = None,
                        top: float = 0.0, bottom: float | None = None, placer: str = "", extra: dict | None = None) -> Placement:
    bottom = float(height) if bottom is None else bottom
    q = list(q) if q is not None else [0.5] * len(text)
    aligned = list(aligned) if aligned is not None else [True] * len(text)
    chars = []
    for i, ch in enumerate(text):
        x0, x1 = float(cuts[i]), float(cuts[i + 1])
        if ch.isspace():
            middle = (x0 + x1) / 2
            chars.append(CharPlace(i, ch, middle, middle, rectangle(middle, middle, top, bottom), 0.0, visible=False, aligned=False))
        else:
            chars.append(CharPlace(i, ch, x0, x1, rectangle(x0, x1, top, bottom), float(q[i]), visible=True, aligned=bool(aligned[i])))
    words = []
    for k, (i0, i1) in enumerate(word_spans(text)):
        x0, x1 = chars[i0].x0, chars[i1 - 1].x1
        word_q = min(chars[i].q for i in range(i0, i1))
        fit = float(word_fits[k]) if word_fits is not None else float(line_fit)
        words.append(WordPlace(i0, i1, x0, x1, rectangle(x0, x1, top, bottom), word_q, fit))
    return Placement(text=text, width=int(width), height=int(height), chars=chars, words=words, line_fit=float(line_fit),
                     placer=placer, extra=extra or {})
