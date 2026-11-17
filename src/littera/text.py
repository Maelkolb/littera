from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from littera.types import word_spans

GAP = 0
ANY = 1
GAP_KEY = "<gap>"
ANY_KEY = "<any>"

INK = "ink"
NO_INK = "none"
OPTIONAL = "optional"

LINE_END_HYPHENS = "\u00ac\u2e17=-\u2010\u00ad"
ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff\u00ad"
SUPPLIED_BRACKETS = {"[": "]", "\u27e8": "\u27e9", "\u3008": "\u3009"}
TAG = re.compile(r"</?[A-Za-z][\w:.-]*(?:\s[^<>]*)?/?>")
ELLIPSIS = re.compile(r"^[\s.\u2026]*$")


def mark_none(text: str) -> list[str]:
    return [INK] * len(text)


def mark_plain(text: str) -> list[str]:
    roles = [NO_INK if ch in ZERO_WIDTH else INK for ch in text]
    last = next((i for i in range(len(text) - 1, -1, -1) if not text[i].isspace() and (roles[i] == INK or text[i] == "­")), None)
    if last is not None and text[last] in LINE_END_HYPHENS:
        roles[last] = OPTIONAL
    return roles


def mark_tags(text: str, roles: list[str] | None = None) -> list[str]:
    roles = mark_plain(text) if roles is None else roles
    for match in TAG.finditer(text):
        for i in range(match.start(), match.end()):
            roles[i] = NO_INK
    return roles


def mark_editorial(text: str, roles: list[str] | None = None) -> list[str]:
    roles = mark_plain(text) if roles is None else roles
    closers = {close: open_ for open_, close in SUPPLIED_BRACKETS.items()}
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in SUPPLIED_BRACKETS:
            end = text.find(SUPPLIED_BRACKETS[ch], i + 1)
            end = len(text) if end < 0 else end
            _supply(text, roles, i, end)
            i = end + 1
        elif ch in closers:
            _supply(text, roles, -1, i)
            i += 1
        else:
            i += 1
    return roles


def _supply(text: str, roles: list[str], open_at: int, close_at: int) -> None:
    if open_at >= 0:
        roles[open_at] = NO_INK
    if close_at < len(text):
        roles[close_at] = NO_INK
    inside = range(open_at + 1, close_at)
    content = text[open_at + 1 : close_at]
    for k in inside:
        if roles[k] != NO_INK and not text[k].isspace():
            roles[k] = NO_INK if ELLIPSIS.match(content) else OPTIONAL


def mark_all(text: str) -> list[str]:
    return mark_editorial(text, mark_tags(text))


PRESETS: dict[str, Callable[[str], list[str]]] = {
    "none": mark_none,
    "plain": mark_plain,
    "tags": mark_tags,
    "editorial": mark_editorial,
    "all": mark_all,
}

STRUCK_MARKERS = "\u301a\u301b\u27e6\u27e7"
NOTE = re.compile(r"\[\s*(?:\.{2,}|\u2026|\?|!|sic!?|illisible|unleserlich|unclear|gap)\s*\]", re.IGNORECASE)
APOSTROPHES = "'\u2019"


def _no_ink(roles: list[str], positions) -> None:
    for i in positions:
        roles[i] = NO_INK


def rule_tags(text: str, roles: list[str]) -> None:
    for match in TAG.finditer(text):
        _no_ink(roles, range(match.start(), match.end()))


def rule_struck(text: str, roles: list[str]) -> None:
    for match in re.finditer("~~", text):
        _no_ink(roles, range(match.start(), match.end()))
    _no_ink(roles, (i for i, ch in enumerate(text) if ch in STRUCK_MARKERS))


def rule_raised(text: str, roles: list[str]) -> None:
    _no_ink(roles, (i for i, ch in enumerate(text) if ch == "^"))


def rule_notes(text: str, roles: list[str]) -> None:
    for match in NOTE.finditer(text):
        _no_ink(roles, range(match.start(), match.end()))


def rule_brackets(text: str, roles: list[str]) -> None:
    mark_editorial(text, roles)


def rule_dollar(text: str, roles: list[str]) -> None:
    _no_ink(roles, (i for i, ch in enumerate(text) if ch == "$"))


def rule_bars(text: str, roles: list[str]) -> None:
    _no_ink(roles, (i for i, ch in enumerate(text) if ch == "|"))


def rule_braces(text: str, roles: list[str]) -> None:
    _no_ink(roles, (i for i, ch in enumerate(text) if ch in "{}"))


def rule_apostrophe(text: str, roles: list[str]) -> None:
    for i, ch in enumerate(text):
        if ch in APOSTROPHES and roles[i] == INK:
            roles[i] = OPTIONAL


RULES: dict[str, list[Callable[[str, list[str]], None]]] = {
    "markup": [rule_tags, rule_struck, rule_raised, rule_notes],
    "brackets": [rule_notes, rule_brackets],
    "dollar": [rule_dollar],
    "bars": [rule_bars],
    "braces": [rule_braces],
    "apostrophe": [rule_apostrophe],
}


def mark_rules(text: str, names: list[str]) -> list[str]:
    roles = [NO_INK if ch in ZERO_WIDTH else INK for ch in text]
    for name in names:
        for rule in RULES[name]:
            rule(text, roles)
    last = next((i for i in range(len(text) - 1, -1, -1) if not text[i].isspace() and roles[i] != NO_INK), None)
    if last is not None and text[last] in LINE_END_HYPHENS + "~":
        roles[last] = OPTIONAL
    return roles


def mark(text: str, preset: str = "plain") -> list[str]:
    if preset in PRESETS:
        return PRESETS[preset](text)
    names = preset.split("+")
    unknown = [name for name in names if name not in RULES]
    if unknown:
        raise KeyError(f"unknown markup preset {preset!r}: {unknown}")
    return mark_rules(text, names)


@dataclass
class Unit:
    key: str
    i0: int
    i1: int
    optional: bool = False


@dataclass
class Segmented:
    text: str
    roles: list[str]
    units: list[Unit]
    unit_of: list[int]
    word_units: list[list[int]]
    spans: list[tuple[int, int]] = field(default_factory=list)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def segment(text: str, preset: str = "plain") -> Segmented:
    roles = mark(text, preset)
    units: list[Unit] = []
    unit_of = [-1] * len(text)
    current = None
    for i, ch in enumerate(text):
        if ch.isspace():
            current = None
            continue
        if roles[i] == NO_INK:
            continue
        decomposed = nfd(ch)
        if unicodedata.combining(decomposed[0]) and current is not None:
            unit = units[current]
            unit.key = nfc(nfd(unit.key) + decomposed)
            unit.i1 = i + 1
            unit_of[i] = current
            continue
        units.append(Unit(key=nfc(decomposed), i0=i, i1=i + 1, optional=roles[i] == OPTIONAL))
        current = len(units) - 1
        unit_of[i] = current
    spans = word_spans(text)
    word_units = [sorted({unit_of[i] for i in range(a, b) if unit_of[i] >= 0}) for a, b in spans]
    return Segmented(text=text, roles=roles, units=units, unit_of=unit_of, word_units=word_units, spans=spans)


@dataclass
class Encoded:
    seg: Segmented
    labels: list[int]

    def graph_words(self) -> list[list[tuple[int, bool]]]:
        return [[(self.labels[k], self.seg.units[k].optional) for k in ks] for ks in self.seg.word_units if ks]

    def graph_word_index(self) -> list[int]:
        return [w for w, ks in enumerate(self.seg.word_units) if ks]

    def unit_order(self) -> list[int]:
        return [k for ks in self.seg.word_units for k in ks]


class Codec:
    def __init__(self, classes: Iterable[str], counts: dict[str, int] | None = None):
        self.classes = list(classes)
        self.index = {key: i + 2 for i, key in enumerate(self.classes)}
        self.counts = dict(counts or {})

    def __len__(self) -> int:
        return len(self.classes) + 2

    def label(self, key: str) -> int:
        if key in self.index:
            return self.index[key]
        return self.index.get(nfd(key)[0], ANY)

    def symbol(self, label: int) -> str:
        if label == GAP:
            return GAP_KEY
        if label == ANY:
            return ANY_KEY
        return self.classes[label - 2]

    def encode(self, text: str, preset: str = "plain") -> Encoded:
        seg = segment(text, preset)
        return Encoded(seg=seg, labels=[self.label(u.key) for u in seg.units])

    def decode(self, words: list[list[int]]) -> str:
        return " ".join("".join("�" if label == ANY else self.symbol(label) for label in word) for word in words)

    def count(self, label: int) -> int:
        return self.counts.get(self.symbol(label), 0) if label >= 2 else 0

    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 2, preset: str | Sequence[str] = "plain") -> Codec:
        texts = list(texts)
        presets = [preset] * len(texts) if isinstance(preset, str) else list(preset)
        counts = Counter(unit.key for text, name in zip(texts, presets) for unit in segment(text, name).units)
        kept = {key: n for key, n in counts.items() if n >= min_count}
        folded = Counter()
        for key, n in counts.items():
            base = nfd(key)[0]
            if key not in kept and base != key:
                folded[base] += n
        for base, n in folded.items():
            if base in kept:
                kept[base] += n
            elif counts.get(base, 0) + n >= min_count:
                kept[base] = counts.get(base, 0) + n
        return cls(sorted(kept), kept)

    def to_dict(self) -> dict:
        return {"classes": self.classes, "counts": self.counts}

    @classmethod
    def from_dict(cls, d: dict) -> Codec:
        return cls(d["classes"], d.get("counts"))
