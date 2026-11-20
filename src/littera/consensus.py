from __future__ import annotations

from typing import Sequence

from littera.types import word_spans


def blend_edges(text: str, own: Sequence[float], other: Sequence[float], margin: float = 0.25, reach: float = 1.5) -> tuple[list[float], int, int]:
    """Letter edges inside each word as the mean of two placements; word ends stay those of `own`.

    `own` has snapped word ends and its model's inner edges, `other` is a second aligner's unsnapped placement of the same
    text, both as n + 1 positions. A word keeps `own` when either placement skipped one of its characters, when the two
    disagree about where the word lies by more than `reach` letter widths, or when a mean edge would come within
    `margin` letter widths of a word end. Returns the edges, the words blended and the words with inner edges.
    """
    if len(own) != len(text) + 1 or len(other) != len(own):
        raise ValueError("own and other need one position more than the text has characters")
    out = [float(v) for v in own]
    blended = eligible = 0
    for i, j in word_spans(text):
        if j - i < 2:
            continue
        eligible += 1
        letter = (own[j] - own[i]) / (j - i)
        if letter <= 0 or any(own[k + 1] <= own[k] or other[k + 1] <= other[k] for k in range(i, j)):
            continue
        if abs(other[i] - own[i]) > reach * letter or abs(other[j] - own[j]) > reach * letter:
            continue
        mean = [(own[k] + other[k]) / 2 for k in range(i + 1, j)]
        if mean[0] < own[i] + margin * letter or mean[-1] > own[j] - margin * letter or any(b <= a for a, b in zip(mean, mean[1:])):
            continue
        out[i + 1 : j] = mean
        blended += 1
    return out, blended, eligible
