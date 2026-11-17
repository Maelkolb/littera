from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from littera.baselines.common import placement_from_cuts
from littera.eval.ink import otsu_level
from littera.types import Placement, Point, word_spans


def load_recogniser(path: str):
    try:
        from kraken.models import load_models
    except ImportError:
        from kraken.lib.vgsl import TorchVGSLModel

        return TorchVGSLModel.load_model(path)
    return next(m for m in load_models(path) if "recognition" in str(getattr(m, "model_type", "recognition")))


def centres_to_cuts(centres: list[float | None], width: float) -> list[float]:
    known = [(i, c) for i, c in enumerate(centres) if c is not None]
    n = len(centres)
    if not known:
        return [width * i / max(1, n) for i in range(n + 1)]
    step = (known[-1][1] - known[0][1]) / max(1, known[-1][0] - known[0][0]) if len(known) > 1 else width / max(1, n)
    full = []
    for i in range(n):
        before = max(((j, c) for j, c in known if j <= i), default=None)
        after = min(((j, c) for j, c in known if j >= i), default=None)
        if before and after and before[0] != after[0]:
            full.append(before[1] + (after[1] - before[1]) * (i - before[0]) / (after[0] - before[0]))
        elif before and after:
            full.append(before[1])
        elif before:
            full.append(before[1] + step * (i - before[0]))
        else:
            full.append(after[1] - step * (after[0] - i))
    for i in range(1, n):
        full[i] = max(full[i], full[i - 1])
    half = step / 2
    cuts = [full[0] - half, *[(a + b) / 2 for a, b in zip(full, full[1:])], full[-1] + half]
    return [min(width, max(0.0, x)) for x in cuts]


def ctc_viterbi(log_probs: np.ndarray, labels: list[int], blank: int = 0) -> list[int]:
    frames = log_probs.shape[0]
    states = [blank]
    for label in labels:
        states += [label, blank]
    n_states = len(states)
    states_arr = np.asarray(states)
    can_skip = np.zeros(n_states, dtype=bool)
    for s in range(2, n_states):
        can_skip[s] = states[s] != blank and states[s] != states[s - 2]
    score = np.full(n_states, -np.inf)
    score[0] = log_probs[0, states[0]]
    if n_states > 1:
        score[1] = log_probs[0, states[1]]
    back = np.zeros((frames, n_states), dtype=np.int32)
    index = np.arange(n_states)
    for t in range(1, frames):
        stay = score
        step = np.concatenate([[-np.inf], score[:-1]])
        skip = np.where(can_skip, np.concatenate([[-np.inf, -np.inf], score[:-2]]), -np.inf)
        best = np.maximum(np.maximum(stay, step), skip)
        back[t] = np.where(best == stay, index, np.where(best == step, index - 1, index - 2))
        score = best + log_probs[t, states_arr]
    last = n_states - 1 if n_states == 1 or score[-1] >= score[-2] else n_states - 2
    path = [last]
    for t in range(frames - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def best_gap(profile: list[float], lo: int, hi: int, raw: float, level: float) -> tuple[float, float] | None:
    lo, hi = max(0, lo), min(len(profile), hi)
    if hi - lo < 1:
        return None
    runs, start = [], None
    for u in range(lo, hi + 1):
        empty = u < hi and profile[u] <= level
        if empty and start is None:
            start = u
        elif not empty and start is not None:
            runs.append((start, u))
            start = None
    if runs:
        s, e = max(runs, key=lambda r: (r[1] - r[0]) - 0.5 * abs((r[0] + r[1]) / 2 - raw))
        return float(s), float(e)
    smooth = [sum(profile[max(lo, u - 1): min(hi, u + 2)]) for u in range(lo, hi)]
    u = lo + min(range(len(smooth)), key=lambda k: (smooth[k], abs(lo + k - raw)))
    return float(u), float(u + 1)


@dataclass
class Alignment:
    text: str
    frames: int
    width_in: int
    crop_width: int
    strip: np.ndarray
    log_probs: np.ndarray
    spans: dict[int, tuple[int, int]] = field(default_factory=dict)
    q: dict[int, float] = field(default_factory=dict)
    frame_owner: list[int] = field(default_factory=list)
    frame_gain: list[float] = field(default_factory=list)
    line_fit: float = 0.0


class KrakenPlacer:
    def __init__(self, model_path: str, name: str = "kraken", centre_normalise: bool = True, padding: int = 16, snap: bool = True,
                 normal_form: str | None = None):
        import torch
        from kraken.lib.dataset import ImageInputTransforms

        self.torch = torch
        self.name = name
        self.net = load_recogniser(model_path)
        if hasattr(self.net, "eval"):
            self.net.eval()
        _, channels, height, width = self.net.input
        self.padding = padding
        self.snap = snap
        self.transform = ImageInputTransforms(1, height, width, channels, (padding, 0), valid_norm=centre_normalise)
        self.c2l = {c: list(labels) for c, labels in self.net.codec.c2l.items() if len(c) == 1}
        if normal_form is None:
            combining = any(unicodedata.combining(c) for c in self.c2l)
            precomposed = any(unicodedata.normalize("NFD", c) != c for c in self.c2l)
            normal_form = "NFD" if combining and not precomposed else "NFC"
        self.normal_form = normal_form

    def run(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        x = self.transform(image.convert("L"))
        with torch.inference_mode():
            output, lengths = self.net.nn(x.unsqueeze(0), torch.tensor([x.shape[-1]]))
        frames = int(lengths[0]) if lengths is not None else output.shape[-1]
        scores = output[0, :, 0, :frames].float()
        if float(scores.min()) >= 0 and abs(float(scores.sum(0).mean()) - 1.0) < 1e-3:
            log_probs = torch.log(scores.clamp_min(1e-12))
        else:
            log_probs = scores.log_softmax(0)
        return log_probs.T.numpy(), x[0].numpy()

    def labels(self, text: str) -> tuple[list[int], list[int]]:
        labels, owner = [], []
        for i, ch in enumerate(text):
            for unit in unicodedata.normalize(self.normal_form, ch):
                for label in self.c2l.get(unit, []):
                    labels.append(label)
                    owner.append(i)
        return labels, owner

    def line_fit(self, log_probs: np.ndarray, labels: list[int]) -> float:
        torch = self.torch
        best = log_probs.argmax(1).tolist()
        greedy = [c for t, c in enumerate(best) if c != 0 and (t == 0 or c != best[t - 1])]
        frames = log_probs.shape[0]
        emission = torch.from_numpy(log_probs)
        targets = torch.tensor(labels + greedy, dtype=torch.long)
        stacked = emission.unsqueeze(1).expand(frames, 2, emission.shape[1])
        nll = torch.nn.functional.ctc_loss(stacked, targets, torch.tensor([frames, frames]), torch.tensor([len(labels), len(greedy)]),
                                           blank=0, reduction="none", zero_infinity=False)
        if not torch.isfinite(nll[0]):
            return 0.0
        return min(1.0, math.exp(-(float(nll[0]) - float(nll[1])) / max(1, len(labels))))

    def align(self, image: Image.Image, text: str) -> Alignment:
        log_probs, strip = self.run(image)
        labels, owner = self.labels(text)
        al = Alignment(text=text, frames=log_probs.shape[0], width_in=strip.shape[-1], crop_width=image.width, strip=strip,
                       log_probs=log_probs)
        if not labels or al.frames < len(labels):
            return al
        path = ctc_viterbi(log_probs, labels)
        probs = np.exp(log_probs)
        best = log_probs.max(1)
        al.frame_owner = [-1] * al.frames
        al.frame_gain = [0.0] * al.frames
        for t, state in enumerate(path):
            if state % 2 == 1:
                k = (state - 1) // 2
                i = owner[k]
                lo, hi = al.spans.get(i, (t, t))
                al.spans[i] = (min(lo, t), max(hi, t))
                al.q[i] = max(al.q.get(i, 0.0), float(probs[t, labels[k]]))
                al.frame_owner[t] = i
                al.frame_gain[t] = float(log_probs[t, labels[k]] - best[t])
            else:
                al.frame_gain[t] = float(log_probs[t, 0] - best[t])
        al.line_fit = self.line_fit(log_probs, labels)
        return al

    def to_crop(self, al: Alignment, frame_x: float) -> float:
        net_scale = al.width_in / al.frames
        in_scale = al.crop_width / max(1, al.width_in - 2 * self.padding)
        return (frame_x * net_scale - self.padding) * in_scale

    def to_strip(self, al: Alignment, x: float) -> float:
        return x * max(1, al.width_in - 2 * self.padding) / al.crop_width + self.padding

    def from_strip(self, al: Alignment, u: float) -> float:
        return (u - self.padding) * al.crop_width / max(1, al.width_in - 2 * self.padding)

    def raw_cuts(self, al: Alignment) -> list[float]:
        centres: list[float | None] = [None] * len(al.text)
        for i, (lo, _) in al.spans.items():
            centres[i] = self.to_crop(al, lo + 0.5)
        return centres_to_cuts(centres, al.crop_width)

    def ink_band(self, al: Alignment) -> list[float]:
        strip = al.strip
        values = strip[strip > 0.02]
        level = otsu_level(values) if values.size else 0.5
        ink = (strip > level).astype(np.float32)
        rows = ink.mean(1)
        if float(rows.max()) > 0:
            idx = np.nonzero(rows >= 0.5 * float(rows.max()))[0]
            top, bottom = max(0, int(idx[0]) - 4), min(len(rows), int(idx[-1]) + 5)
        else:
            top, bottom = 0, len(rows)
        return ink[top:bottom].sum(0).tolist()

    def snap_words(self, al: Alignment, cuts: list[float]) -> list[float]:
        if not al.spans:
            return cuts
        band = self.ink_band(al)
        words = word_spans(al.text)
        extent = [[self.to_strip(al, cuts[a]), self.to_strip(al, cuts[b])] for a, b in words]
        seen = [w for w, (a, b) in enumerate(words) if any(i in al.spans for i in range(a, b))]
        gap_level = max(1.0, 0.02 * max(band, default=0))
        for v, w in zip(seen, seen[1:]):
            (_, a1), (b0, _) = words[v], words[w]
            left = self.to_strip(al, (cuts[a1 - 1] + cuts[a1]) / 2)
            right = self.to_strip(al, (cuts[b0] + cuts[b0 + 1]) / 2)
            raw = (self.to_strip(al, cuts[a1]) + self.to_strip(al, cuts[b0])) / 2
            run = best_gap(band, int(math.ceil(left)), int(math.floor(right)), raw, gap_level)
            if run:
                extent[v][1], extent[w][0] = run
        out = list(cuts)
        for (a, b), (u0, u1) in zip(words, extent):
            x0, x1 = self.from_strip(al, u0), self.from_strip(al, u1)
            old0, old1 = cuts[a], cuts[b]
            for i in range(a, b + 1):
                rel = (cuts[i] - old0) / (old1 - old0) if old1 > old0 else (i - a) / max(1, b - a)
                out[i] = x0 + rel * (x1 - x0)
        for i in range(1, len(out)):
            out[i] = max(out[i], out[i - 1])
        return [min(al.crop_width, max(0.0, x)) for x in out]

    def word_fits(self, al: Alignment) -> list[float]:
        fits = []
        for a, b in word_spans(al.text):
            frames = [t for t, i in enumerate(al.frame_owner) if a <= i < b]
            if not frames:
                fits.append(0.0)
                continue
            gains = al.frame_gain[frames[0]: frames[-1] + 1]
            fits.append(float(math.exp(sum(gains) / len(gains))))
        return fits

    def place(self, image: Image.Image, text: str, polygon: Sequence[Point] | None = None,
              baseline: Sequence[Point] | None = None) -> Placement:
        if polygon is not None and len(polygon) >= 3:
            image = mask_outside(image, polygon)
        al = self.align(image, text)
        cuts = self.raw_cuts(al)
        if self.snap:
            cuts = self.snap_words(al, cuts)
        q = [al.q.get(i, 0.0) for i in range(len(text))]
        aligned = [i in al.spans for i in range(len(text))]
        fits = self.word_fits(al) if al.spans else None
        return placement_from_cuts(text, cuts, image.width, image.height, q=q, line_fit=al.line_fit, word_fits=fits, aligned=aligned,
                                   placer=self.name, extra={"frames": al.frames, "normal_form": self.normal_form})


def mask_outside(image: Image.Image, polygon: Sequence[Point]) -> Image.Image:
    grey = image.convert("L")
    inside = Image.new("L", grey.size, 0)
    ImageDraw.Draw(inside).polygon([(float(x), float(y)) for x, y in polygon], fill=255)
    paper = int(np.median(np.asarray(grey)))
    return Image.composite(grey, Image.new("L", grey.size, paper), inside)
