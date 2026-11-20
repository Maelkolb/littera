from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from littera import graph
from littera.geometry import GeometryConfig, Strip, best_gap, free_runs, ink_edge, ink_extent, prepare
from littera.graph import GraphOptions
from littera.model import LineModel, load_model
from littera.text import Codec, Encoded
from littera.types import CharPlace, Placement, Point, WordPlace

SNAPS = ("strip", "upright", "off")


def snap_mode(snap: bool | str | None) -> str:
    """Where word edges are snapped into ink gaps: "strip" (the deslanted strip the model reads), "upright" (the same line
    without deslanting, for upright word crops), or "off"."""
    if snap is True:
        return "strip"
    if snap is False or snap is None:
        return "off"
    if snap not in SNAPS:
        raise ValueError(f"snap must be one of {SNAPS}, not {snap!r}")
    return snap


DECODE = GraphOptions(spaces="optional", space_penalty=2.0, skip_penalty=3.0, garbage_ends=True, garbage_between=True, garbage="filler")


@dataclass
class Objective:
    topology: str = "segmental"
    normalise: bool = True
    prior_scale: float = 0.3
    decode_prior_scale: float = 0.0

    @classmethod
    def from_dict(cls, d: dict | None) -> Objective:
        return cls(**(d or {}))


@dataclass
class WordCuts:
    word: int
    units: list[int]
    cuts: list[float]
    sigmas: list[float]
    lifts: dict[int, tuple[float, float]] = field(default_factory=dict)
    edges: dict[int, tuple[float, float]] = field(default_factory=dict)
    kept: bool = False

    def span(self, k: int) -> tuple[float, float]:
        left = self.edges[k][1] if k in self.edges else self.cuts[k]
        right = self.edges[k + 1][0] if k + 1 in self.edges else self.cuts[k + 1]
        return left, right


@dataclass
class Analysis:
    strip: Strip
    encoded: Encoded
    words: list[WordCuts]
    unit_q: dict[int, float]
    word_q: dict[int, float]
    word_fit: dict[int, float]
    line_fit: float
    letter: float
    frames: int
    log_likelihood: float
    best_log_likelihood: float
    reading: str
    extra: dict = field(default_factory=dict)


def _finite(value: float) -> float | None:
    return round(value, 4) if math.isfinite(value) else None


def _mass(sigma: float, radius: float, stride: int) -> float:
    spread = math.sqrt(sigma**2 + stride**2 / 12.0)
    return math.erf(radius / (spread * math.sqrt(2.0)))


class Aligner:
    def __init__(self, model: LineModel, codec: Codec, geometry: GeometryConfig | None = None, objective: Objective | None = None,
                 decode: GraphOptions | None = None, preset: str = "plain", snap: bool | str = True, name: str = "littera", device="cpu"):
        self.model = model.to(device).eval()
        self.codec = codec
        self.geometry = geometry or GeometryConfig(height=model.config.height, stride=model.config.stride)
        self.objective = objective or Objective()
        decode = decode or DECODE
        self.decode = GraphOptions(**{**decode.to_dict(), "topology": "segmental" if self.objective.topology == "segmental" else "ctc"})
        self.preset = preset
        self.snap = snap_mode(snap)
        self.name = name
        self.device = torch.device(device)

    @classmethod
    def load(cls, path, device="cpu", **overrides) -> Aligner:
        model, config = load_model(path, device)
        decode = GraphOptions.from_dict(config.get("decode")) if config.get("decode") else DECODE
        kwargs = dict(codec=Codec.from_dict(config["codec"]), geometry=GeometryConfig.from_dict(config.get("geometry")),
                      objective=Objective.from_dict(config.get("objective")), decode=decode, preset=config.get("preset", "plain"),
                      name=config.get("name", "littera"), device=device)
        kwargs.update(overrides)
        return cls(model, **kwargs)

    def config(self) -> dict:
        return {"name": self.name, "model": self.model.config.to_dict(), "codec": self.codec.to_dict(), "geometry": self.geometry.to_dict(),
                "objective": self.objective.__dict__, "decode": self.decode.to_dict(), "preset": self.preset}

    @torch.no_grad()
    def run(self, strip: Strip) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(strip.image).to(self.device)[None, None]
        logits, boundary, lengths = self.model(image, torch.tensor([strip.width], device=self.device))
        logp = logits.float().log_softmax(-1)
        scores = graph.sequence_scores(logp, self.objective.topology, self.model.log_prior, self.objective.decode_prior_scale)
        empty = torch.from_numpy(strip.empty_frames(self.geometry.stride)).to(self.device)[None, : scores.shape[1]]
        scores = graph.with_ink_prior(scores, empty, self.decode.ink_prior)
        return logp, scores, boundary.float(), lengths

    def _boundary_logs(self, boundary: torch.Tensor, scores: torch.Tensor):
        return graph.boundary_logs(boundary if self.decode.segmental else None, tuple(scores.shape[:2]), scores)

    def _lift_scores(self, scores: torch.Tensor, strip: Strip) -> torch.Tensor | None:
        if not (self.decode.lifts and self.decode.segmental):
            return None
        open_frames = torch.from_numpy(strip.open_frames(self.geometry.stride)).to(self.device)[None, : scores.shape[1]]
        return graph.lift_scores(scores, self.decode, open_frames)

    def _loglik(self, words: list[list[tuple[int, bool]]], logp, scores, logb, log1mb, lengths, lift=None) -> float:
        if not words:
            gap = scores[0, : int(lengths[0]), 0].sum()
            if self.decode.segmental:
                gap = gap + log1mb[0, 1 : int(lengths[0])].sum()
            return float(gap)
        packed = graph.pack([graph.build(words, self.decode)], self.device)
        E = graph.emissions(scores, packed, graph.garbage_scores(self._garbage_source(logp, scores), self.decode), lift)
        with torch.no_grad():
            return float(graph.log_likelihood(E, logb, log1mb, packed, lengths)[0])

    def _garbage_source(self, logp: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return scores if self.decode.garbage == "any" else logp

    def read(self, image: Image.Image, polygon: Sequence[Point] | None = None, baseline: Sequence[Point] | None = None) -> str:
        strip = prepare(image, polygon, baseline, self.geometry)
        logp, scores, boundary, lengths = self.run(strip)
        return self.codec.decode(self._best_reading(scores, boundary, lengths))

    def _best_reading(self, scores, boundary, lengths) -> list[list[int]]:
        if self.decode.segmental:
            logb, log1mb = self._boundary_logs(boundary, scores)
            segments = graph.free_viterbi(scores, logb, log1mb, lengths)[0]
        else:
            segments = graph.greedy_ctc(scores, lengths)[0]
        return graph.reading_words(segments, self.decode.segmental)

    def place(self, image: Image.Image, text: str, polygon: Sequence[Point] | None = None,
              baseline: Sequence[Point] | None = None) -> Placement:
        return self.to_placement(self.analyse(image, text, polygon, baseline), text)

    def analyse(self, image: Image.Image, text: str, polygon: Sequence[Point] | None = None,
                baseline: Sequence[Point] | None = None) -> Analysis:
        strip = prepare(image, polygon, baseline, self.geometry)
        encoded = self.codec.encode(text, self.preset)
        words = encoded.graph_words()
        stride = self.geometry.stride
        profile = strip.core_profile()
        level = max(1.0, 0.02 * float(profile.max(initial=0)))
        extent = ink_extent(profile, level) or (strip.pad, strip.width - strip.pad)
        n_units = sum(len(w) for w in words)
        letter = max(2.0, (extent[1] - extent[0]) / max(1, n_units))
        if not words:
            return Analysis(strip, encoded, [], {}, {}, {}, 0.0, letter, strip.width // stride, float("nan"), float("nan"), "")

        logp, scores, boundary, lengths = self.run(strip)
        logb, log1mb = self._boundary_logs(boundary, scores)
        lift = self._lift_scores(scores, strip)
        packed = graph.pack([graph.build(words, self.decode)], self.device)
        E = graph.emissions(scores, packed, graph.garbage_scores(self._garbage_source(logp, scores), self.decode), lift)
        post = graph.posteriors(E, logb, log1mb, packed, lengths)
        loglik = float(post.log_likelihood[0])
        best_words = self._best_reading(scores, boundary, lengths)
        reading = self.codec.decode(best_words)
        if not math.isfinite(loglik):
            cuts = self._even_cuts(encoded, extent)
            return Analysis(strip, encoded, cuts, {}, {}, {}, 0.0, letter, int(lengths[0]), loglik, float("nan"), reading,
                            {"fallback": "even"})

        frames = int(lengths[0])
        unit_states = packed.unit_states[0, :n_units]
        entry = post.entry()[0, :frames][:, unit_states].clamp(min=0).cpu().numpy().astype(np.float64)
        leave = post.exit()[0, :frames][:, unit_states].clamp(min=0).cpu().numpy().astype(np.float64)
        used = entry.sum(0)
        t = np.arange(frames, dtype=np.float64)[:, None]
        weight = np.maximum(used, 1e-9)
        start = (t * entry).sum(0) / weight
        start_var = np.maximum((t**2 * entry).sum(0) / weight - start**2, 0)
        end = ((t + 1) * leave).sum(0) / weight
        end_var = np.maximum(((t + 1) ** 2 * leave).sum(0) / weight - end**2, 0)
        order = encoded.unit_order()
        word_index = encoded.graph_word_index()
        lift_spans = self._lift_spans(post, packed, frames, n_units) if lift is not None else {}

        word_cuts: list[WordCuts] = []
        j = 0
        segmental = self.decode.segmental
        centres = (start + 0.5) * stride
        steps = []
        for units in words:
            visible = [j + k for k in range(len(units)) if used[j + k] >= 0.5]
            steps += [centres[b] - centres[a] for a, b in zip(visible, visible[1:])]
            j += len(units)
        half_step = 0.5 * float(np.median(steps)) if steps else 0.5 * letter
        j = 0
        for w, units in enumerate(words):
            visible = [j + k for k in range(len(units)) if used[j + k] >= 0.5]
            if segmental:
                cuts = [start[visible[0]] * stride] if visible else []
                sigmas = [math.sqrt(start_var[visible[0]]) * stride] if visible else []
                for a, b in zip(visible, visible[1:]):
                    cuts.append(0.5 * (end[a] + start[b]) * stride)
                    sigmas.append(math.sqrt(0.5 * (end_var[a] + start_var[b])) * stride)
                if visible:
                    cuts.append(end[visible[-1]] * stride)
                    sigmas.append(math.sqrt(end_var[visible[-1]]) * stride)
            else:
                spread = [math.sqrt(start_var[k]) * stride for k in visible]
                cuts = [centres[visible[0]] - half_step] if visible else []
                sigmas = [spread[0]] if visible else []
                for (a, sa), (b, sb) in zip(zip(visible, spread), zip(visible[1:], spread[1:])):
                    cuts.append(0.5 * (centres[a] + centres[b]))
                    sigmas.append(0.5 * math.sqrt(sa**2 + sb**2))
                if visible:
                    cuts.append(centres[visible[-1]] + half_step)
                    sigmas.append(spread[-1])
            word_cuts.append(WordCuts(word=word_index[w], units=[order[k] for k in visible], cuts=cuts, sigmas=sigmas))
            for k, (a, b) in enumerate(zip(visible, visible[1:]), start=1):
                found = [lift_spans[u] for u in range(a, b) if u in lift_spans]
                if found:
                    word_cuts[-1].lifts[k] = max(found, key=lambda span: span[1] - span[0])
            j += len(units)

        if self.snap == "strip":
            self._snap(word_cuts, profile, level, letter)
        self._monotone(word_cuts, strip.width)
        if self.snap == "upright" and strip.slant:
            self._snap_upright(image, polygon, baseline, strip, word_cuts, letter)
        elif self.snap == "upright":
            self._snap(word_cuts, profile, level, letter)
            self._monotone(word_cuts, strip.width)
        if self.decode.lift_snap != "off" and not (self.snap == "upright" and strip.slant):
            self._snap_lifts(word_cuts, profile, level)

        unit_q, word_q = {}, {}
        for wc in word_cuts:
            if not wc.units:
                continue
            masses = [_mass(s, 0.25 * letter, stride) for s in wc.sigmas]
            for k, unit in enumerate(wc.units):
                unit_q[unit] = masses[k] * masses[k + 1]
            word_q[wc.word] = _mass(wc.sigmas[0], 0.5 * letter, stride) * _mass(wc.sigmas[-1], 0.5 * letter, stride)

        gamma = post.occupancy[0, :frames]
        expected = (gamma * E[0, :frames].clamp(min=-1e4)).sum(1).cpu().numpy()
        best_frame = scores[0, :frames].max(1).values.cpu().numpy()
        word_fit = {}
        for wc, units in zip(word_cuts, words):
            if not wc.units:
                continue
            f0 = int(np.clip(math.floor(wc.cuts[0] / stride), 0, frames - 1))
            f1 = int(np.clip(math.ceil(wc.cuts[-1] / stride), f0 + 1, frames))
            gap = float(expected[f0:f1].sum() - best_frame[f0:f1].sum())
            word_fit[wc.word] = math.exp(min(0.0, gap) / len(units))

        best = self._loglik([[(label, False) for label in word] for word in best_words], logp, scores, logb, log1mb, lengths, lift)
        line_fit = math.exp(min(0.0, loglik - best) / n_units) if math.isfinite(best) else 0.0
        extra = {"used": [round(float(u), 4) for u in used]}
        if self.decode.inner_edges == "keep" and self.snap != "off":
            extra["words_rescaled"] = sum(1 for wc in word_cuts if len(wc.units) > 1 and not wc.kept)
        if lift is not None or self.decode.lift_snap != "off":
            extra["lifts"] = sum(len(wc.lifts) for wc in word_cuts)
            extra["lift_edges"] = sum(len(wc.edges) for wc in word_cuts)
        return Analysis(strip, encoded, word_cuts, unit_q, word_q, word_fit, line_fit, letter, frames, loglik, best, reading, extra)

    def _lift_spans(self, post: graph.Posteriors, packed: graph.Packed, frames: int, n_units: int) -> dict[int, tuple[float, float]]:
        states = packed.lift_states[0, :n_units].cpu().numpy()
        after = [j for j in range(n_units) if states[j] >= 0]
        if not after:
            return {}
        columns = torch.as_tensor(states[after], device=post.occupancy.device)
        entry = post.entry()[0, :frames][:, columns].clamp(min=0).cpu().numpy().astype(np.float64)
        leave = post.exit()[0, :frames][:, columns].clamp(min=0).cpu().numpy().astype(np.float64)
        t = np.arange(frames, dtype=np.float64)[:, None]
        visited = entry.sum(0)
        weight = np.maximum(visited, 1e-9)
        first = (t * entry).sum(0) / weight * self.geometry.stride
        last = ((t + 1) * leave).sum(0) / weight * self.geometry.stride
        return {j: (float(first[k]), float(max(first[k], last[k]))) for k, j in enumerate(after) if visited[k] >= 0.5}

    def _snap_lifts(self, words: list[WordCuts], profile: np.ndarray, level: float) -> None:
        stride = self.geometry.stride
        every_cut = self.decode.lift_snap == "cuts"
        for wc in words:
            for k in range(1, len(wc.cuts) - 1):
                cut = wc.cuts[k]
                span = wc.lifts.get(k) or ((cut - 0.5 * stride, cut + 0.5 * stride) if every_cut else None)
                if span is None:
                    continue
                floor = int(math.ceil(wc.edges[k - 1][1] if k - 1 in wc.edges else wc.cuts[k - 1]))
                ceiling = int(math.floor(wc.cuts[k + 1]))
                lo, hi = span[0] - stride, span[1] + stride
                runs = [(s, e) for s, e in free_runs(profile, level, floor, ceiling) if floor < s and e < ceiling and s < hi and e > lo]
                if not runs:
                    continue
                s, e = max(runs, key=lambda r: (min(r[1], span[1]) - max(r[0], span[0]), r[1] - r[0]))
                wc.cuts[k] = 0.5 * (s + e)
                if self.decode.lift_edges == "ink":
                    wc.edges[k] = (float(s), float(e))

    def _even_cuts(self, encoded: Encoded, extent: tuple[float, float]) -> list[WordCuts]:
        order = encoded.unit_order()
        total = max(1, len(order))
        step = (extent[1] - extent[0]) / total
        out, j = [], 0
        for w, units in zip(encoded.graph_word_index(), [ks for ks in encoded.seg.word_units if ks]):
            cuts = [extent[0] + (j + k) * step for k in range(len(units) + 1)]
            out.append(WordCuts(word=w, units=list(units), cuts=cuts, sigmas=[step] * len(cuts)))
            j += len(units)
        return out

    def _snap(self, words: list[WordCuts], profile: np.ndarray, level: float, letter: float) -> None:
        seen = [wc for wc in words if wc.units]
        new_edges = {id(wc): [wc.cuts[0], wc.cuts[-1]] for wc in seen}
        for left, right in zip(seen, seen[1:]):
            lo = 0.5 * (left.cuts[-2] + left.cuts[-1])
            hi = 0.5 * (right.cuts[0] + right.cuts[1])
            if hi - lo < 1:
                continue
            raw = 0.5 * (left.cuts[-1] + right.cuts[0])
            run = best_gap(profile, int(math.ceil(lo)), int(math.floor(hi)), raw, level)
            if run:
                new_edges[id(left)][1], new_edges[id(right)][0] = run
        if seen:
            first, last = seen[0], seen[-1]
            new_edges[id(first)][0] = ink_edge(profile, first.cuts[0], 0.5 * letter, level, "start")
            new_edges[id(last)][1] = ink_edge(profile, last.cuts[-1], 0.5 * letter, level, "end")
        for wc in seen:
            a, b = wc.cuts[0], wc.cuts[-1]
            na, nb = new_edges[id(wc)]
            if nb <= na:
                nb = na + 1.0
            margin = 0.25 * (nb - na) / len(wc.units)
            wc.kept = self.decode.inner_edges == "keep" and all(na + margin <= c <= nb - margin for c in wc.cuts[1:-1])
            if wc.kept:
                wc.cuts = [na] + wc.cuts[1:-1] + [nb]
            elif b > a:
                wc.cuts = [na + (c - a) * (nb - na) / (b - a) for c in wc.cuts]
                scale = (nb - na) / (b - a)
                wc.lifts = {k: (na + (u0 - a) * scale, na + (u1 - a) * scale) for k, (u0, u1) in wc.lifts.items()}
            else:
                n = len(wc.cuts) - 1
                wc.cuts = [na + k * (nb - na) / max(1, n) for k in range(n + 1)]
                wc.lifts = {}

    def _snap_upright(self, image, polygon, baseline, strip: Strip, words: list[WordCuts], letter: float) -> None:
        upright = prepare(image, polygon, baseline, self.geometry, slant=0.0)
        profile = upright.core_profile()
        level = max(1.0, 0.02 * float(profile.max(initial=0)))
        def to_upright(u: float) -> float:
            return float(upright.strip_u(strip.crop_x(u)))

        def to_strip(u: float) -> float:
            return float(strip.strip_u(upright.crop_x(u)))

        for wc in words:
            wc.cuts = [to_upright(u) for u in wc.cuts]
            wc.lifts = {k: (to_upright(u0), to_upright(u1)) for k, (u0, u1) in wc.lifts.items()}
        self._snap(words, profile, level, letter)
        self._monotone(words, upright.width)
        if self.decode.lift_snap != "off":
            self._snap_lifts(words, profile, level)
        for wc in words:
            wc.cuts = [to_strip(u) for u in wc.cuts]
            wc.lifts = {k: (to_strip(u0), to_strip(u1)) for k, (u0, u1) in wc.lifts.items()}
            wc.edges = {k: (to_strip(u0), to_strip(u1)) for k, (u0, u1) in wc.edges.items()}

    @staticmethod
    def _monotone(words: list[WordCuts], width: float) -> None:
        last = 0.0
        for wc in words:
            fixed = []
            for c in wc.cuts:
                last = min(float(width), max(last, float(c)))
                fixed.append(last)
            wc.cuts = fixed

    def to_placement(self, analysis: Analysis, text: str) -> Placement:
        strip, seg = analysis.strip, analysis.encoded.seg
        crop_w, crop_h = strip.crop_width, strip.crop_height
        span: dict[int, tuple[float, float]] = {}
        word_span: dict[int, tuple[float, float]] = {}
        for wc in analysis.words:
            for k, unit in enumerate(wc.units):
                span[unit] = wc.span(k)
            if wc.units:
                word_span[wc.word] = (wc.cuts[0], wc.cuts[-1])

        anchors: list[float | None] = [None] * len(text)
        position = None
        for i in range(len(text)):
            unit = seg.unit_of[i]
            if unit in span:
                anchors[i] = span[unit][0]
                position = span[unit][1]
            else:
                anchors[i] = position
        upcoming = None
        for i in range(len(text) - 1, -1, -1):
            unit = seg.unit_of[i]
            if unit in span:
                upcoming = span[unit][0]
            elif anchors[i] is None:
                anchors[i] = upcoming if upcoming is not None else strip.pad

        def x_of(u: float) -> float:
            return round(float(np.clip(strip.crop_x(u), 0, crop_w)), 1)

        chars = []
        for i, ch in enumerate(text):
            unit = seg.unit_of[i]
            if unit in span:
                u0, u1 = span[unit]
                q = round(analysis.unit_q.get(unit, 0.0), 4)
                chars.append(CharPlace(i=i, ch=ch, x0=x_of(u0), x1=x_of(u1), poly=strip.quad(u0, u1), q=q))
            else:
                u = anchors[i]
                chars.append(CharPlace(i=i, ch=ch, x0=x_of(u), x1=x_of(u), poly=strip.quad(u, u), q=0.0, visible=False, aligned=unit >= 0))
        words = []
        for w, (a, b) in enumerate(seg.spans):
            if w in word_span:
                u0, u1 = word_span[w]
                q, fit = round(analysis.word_q.get(w, 0.0), 4), round(analysis.word_fit.get(w, 0.0), 4)
                words.append(WordPlace(i0=a, i1=b, x0=x_of(u0), x1=x_of(u1), poly=strip.quad(u0, u1), q=q, fit=fit))
            else:
                u = anchors[a] if anchors[a] is not None else strip.pad
                words.append(WordPlace(i0=a, i1=b, x0=x_of(u), x1=x_of(u), poly=strip.quad(u, u), q=0.0, fit=1.0))
        extra = {"reading": analysis.reading, "frames": analysis.frames, "scale": round(strip.scale, 4),
                 "letter": round(strip.scale * analysis.letter, 2), "log_likelihood": _finite(analysis.log_likelihood),
                 "best_log_likelihood": _finite(analysis.best_log_likelihood), **analysis.extra}
        return Placement(text=text, width=crop_w, height=crop_h, chars=chars, words=words, line_fit=round(analysis.line_fit, 4),
                         slant=round(strip.slant, 4), placer=self.name, extra=extra)
