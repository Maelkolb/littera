from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol, Sequence

from PIL import Image

SCHEMA = "littera/placement/1"

Point = tuple[float, float]


@dataclass
class CharPlace:
    i: int
    ch: str
    x0: float
    x1: float
    poly: list[Point]
    q: float
    visible: bool = True
    aligned: bool = True


@dataclass
class WordPlace:
    i0: int
    i1: int
    x0: float
    x1: float
    poly: list[Point]
    q: float
    fit: float


@dataclass
class Placement:
    text: str
    width: int
    height: int
    chars: list[CharPlace]
    words: list[WordPlace]
    line_fit: float
    slant: float = 0.0
    placer: str = ""
    extra: dict = field(default_factory=dict)

    def cuts(self) -> list[float]:
        return [c.x0 for c in self.chars] + ([self.chars[-1].x1] if self.chars else [0.0])

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, d: dict) -> Placement:
        d = {k: v for k, v in d.items() if k != "schema"}
        d["chars"] = [CharPlace(**{**c, "poly": [tuple(p) for p in c["poly"]]}) for c in d["chars"]]
        d["words"] = [WordPlace(**{**w, "poly": [tuple(p) for p in w["poly"]]}) for w in d["words"]]
        return cls(**d)


def word_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, ch in enumerate(text + " "):
        if ch.isspace():
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    return spans


class Placer(Protocol):
    name: str

    def place(self, image: Image.Image, text: str, polygon: Sequence[Point] | None = None,
              baseline: Sequence[Point] | None = None) -> Placement: ...
