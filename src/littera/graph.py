from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from littera.text import ANY, GAP

NEUTRAL, ENTER, STAY = 0, 1, 2
UNIT, GAP_STATE, BLANK, GARBAGE = 0, 1, 2, 3
START = -1
GARBAGE_LABEL = -1
NEG = float("-inf")
TOPOLOGIES = ("ctc", "ctc_prior", "segmental")


@dataclass
class GraphOptions:
    topology: str = "segmental"
    spaces: str = "required"
    space_penalty: float = 2.0
    skip_penalty: float = 3.0
    garbage_ends: bool = True
    garbage_between: bool = False
    garbage_enter: float = 2.0
    garbage_frame: float = 1.0
    garbage: str = "any"
    ink_prior: float = 2.0

    @property
    def segmental(self) -> bool:
        return self.topology == "segmental"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> GraphOptions:
        return cls(**(d or {}))


@dataclass
class LineGraph:
    labels: np.ndarray
    roles: np.ndarray
    arcs: np.ndarray
    arc_weight: np.ndarray
    start: np.ndarray
    final: np.ndarray
    unit_states: np.ndarray
    word_of_unit: np.ndarray
    segmental: bool

    @property
    def n_states(self) -> int:
        return len(self.labels)


class _Builder:
    def __init__(self, options: GraphOptions):
        self.options = options
        self.segmental = options.segmental
        self.labels: list[int] = []
        self.roles: list[int] = []
        self.arcs: dict[tuple[int, int], tuple[int, float]] = {}
        self.start: dict[int, float] = {}
        self.unit_states: list[int] = []
        self.word_of_unit: list[int] = []

    def state(self, label: int, role: int) -> int:
        s = len(self.labels)
        self.labels.append(label)
        self.roles.append(role)
        segment = self.segmental and (role != GARBAGE or self.options.garbage == "any")
        self.arcs[(s, s)] = (STAY if segment else NEUTRAL, 0.0)
        return s

    def connect(self, frontier: list[tuple[int, float]], dst: int, cost: float = 0.0) -> None:
        kind = ENTER if self.segmental else NEUTRAL
        for src, c in frontier:
            weight = -(c + cost)
            if src == START:
                self.start[dst] = max(self.start.get(dst, NEG), weight)
                continue
            if self.roles[src] in (GAP_STATE, BLANK) and self.roles[dst] in (GAP_STATE, BLANK):
                continue
            if not self.segmental and self.roles[src] == UNIT and self.roles[dst] == UNIT and self.labels[src] == self.labels[dst]:
                continue
            old = self.arcs.get((src, dst))
            if old is None or old[1] < weight:
                self.arcs[(src, dst)] = (kind, weight)

    def unit(self, frontier: list[tuple[int, float]], label: int, optional: bool, blank_after: bool, word: int) -> list[tuple[int, float]]:
        u = self.state(label, UNIT)
        self.connect(frontier, u)
        out = [(u, 0.0)]
        if blank_after:
            blank = self.state(GAP, BLANK)
            self.connect([(u, 0.0)], blank)
            out.append((blank, 0.0))
        if optional:
            out += [(s, c + self.options.skip_penalty) for s, c in frontier]
        self.unit_states.append(u)
        self.word_of_unit.append(word)
        return out

    def junction(self, frontier: list[tuple[int, float]], required: bool, garbage: bool, skip_cost: float) -> list[tuple[int, float]]:
        entry = [(s, c) for s, c in frontier if s == START or self.roles[s] != BLANK]
        gap = self.state(GAP, GAP_STATE)
        self.connect(entry, gap)
        out = [(gap, 0.0)]
        if garbage:
            junk = self.state(GARBAGE_LABEL, GARBAGE)
            self.connect(entry + [(gap, 0.0)], junk, self.options.garbage_enter)
            after = self.state(GAP, GAP_STATE)
            self.connect([(junk, 0.0)], after)
            out += [(junk, 0.0), (after, 0.0)]
        if not required:
            out += [(s, c + skip_cost) for s, c in entry]
        return out


def build(words: list[list[tuple[int, bool]]], options: GraphOptions) -> LineGraph:
    b = _Builder(options)
    frontier = b.junction([(START, 0.0)], required=False, garbage=options.garbage_ends, skip_cost=0.0)
    for w, units in enumerate(words):
        if w:
            frontier = b.junction(frontier, required=options.spaces == "required", garbage=options.garbage_between,
                                  skip_cost=options.space_penalty)
        for j, (label, optional) in enumerate(units):
            frontier = b.unit(frontier, label, optional, blank_after=not b.segmental and j < len(units) - 1, word=w)
    frontier = b.junction(frontier, required=False, garbage=options.garbage_ends, skip_cost=0.0)
    n = len(b.labels)
    start = np.full(n, NEG, np.float32)
    for s, weight in b.start.items():
        start[s] = weight
    final = np.full(n, NEG, np.float32)
    for s, c in frontier:
        if s != START:
            final[s] = max(final[s], -c)
    arcs = np.array([(src, dst, kind) for (src, dst), (kind, _) in b.arcs.items()], dtype=np.int64).reshape(-1, 3)
    weights = np.array([weight for _, weight in b.arcs.values()], dtype=np.float32)
    return LineGraph(labels=np.array(b.labels, np.int64), roles=np.array(b.roles, np.int8), arcs=arcs, arc_weight=weights, start=start,
                     final=final, unit_states=np.array(b.unit_states, np.int64), word_of_unit=np.array(b.word_of_unit, np.int64),
                     segmental=b.segmental)


@dataclass
class Packed:
    labels: torch.Tensor
    roles: torch.Tensor
    pred: torch.Tensor
    weight: torch.Tensor
    enter: torch.Tensor
    stay: torch.Tensor
    self_loop: torch.Tensor
    succ: torch.Tensor
    succ_valid: torch.Tensor
    start: torch.Tensor
    final: torch.Tensor
    unit_states: torch.Tensor
    segmental: bool

    def to(self, device) -> Packed:
        return Packed(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in self.__dict__.items()})

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.pred.shape)


def _slots(keys: np.ndarray) -> np.ndarray:
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    slots = np.empty(len(keys), np.int64)
    slots[order] = np.arange(len(keys)) - np.searchsorted(ordered, ordered, side="left")
    return slots


def pack(graphs: list[LineGraph], device=None) -> Packed:
    batch = len(graphs)
    n = max(g.n_states for g in graphs)
    in_degree = [np.bincount(g.arcs[:, 1], minlength=g.n_states) for g in graphs]
    out_degree = [np.bincount(g.arcs[:, 0], minlength=g.n_states) for g in graphs]
    k_in = max(int(d.max()) for d in in_degree)
    k_out = max(int(d.max()) for d in out_degree)
    labels = np.zeros((batch, n), np.int64)
    roles = np.full((batch, n), -1, np.int8)
    pred = np.zeros((batch, n, k_in), np.int64)
    weight = np.full((batch, n, k_in), NEG, np.float32)
    kind = np.zeros((batch, n, k_in), np.int8)
    self_loop = np.zeros((batch, n, k_in), bool)
    succ = np.zeros((batch, n, k_out), np.int64)
    succ_valid = np.zeros((batch, n, k_out), bool)
    start = np.full((batch, n), NEG, np.float32)
    final = np.full((batch, n), NEG, np.float32)
    units = max(1, max(len(g.unit_states) for g in graphs))
    unit_states = np.full((batch, units), -1, np.int64)
    for b, g in enumerate(graphs):
        s = g.n_states
        labels[b, :s] = g.labels
        roles[b, :s] = g.roles
        start[b, :s] = g.start
        final[b, :s] = g.final
        unit_states[b, : len(g.unit_states)] = g.unit_states
        src, dst, arc_kind = g.arcs[:, 0], g.arcs[:, 1], g.arcs[:, 2]
        slot_in = _slots(dst)
        slot_out = _slots(src)
        pred[b, dst, slot_in] = src
        weight[b, dst, slot_in] = g.arc_weight
        kind[b, dst, slot_in] = arc_kind
        self_loop[b, dst, slot_in] = src == dst
        succ[b, src, slot_out] = dst * k_in + slot_in
        succ_valid[b, src, slot_out] = True
    packed = Packed(labels=torch.from_numpy(labels), roles=torch.from_numpy(roles), pred=torch.from_numpy(pred),
                    weight=torch.from_numpy(weight), enter=torch.from_numpy(kind == ENTER), stay=torch.from_numpy(kind == STAY),
                    self_loop=torch.from_numpy(self_loop), succ=torch.from_numpy(succ), succ_valid=torch.from_numpy(succ_valid),
                    start=torch.from_numpy(start), final=torch.from_numpy(final), unit_states=torch.from_numpy(unit_states),
                    segmental=graphs[0].segmental)
    return packed.to(device) if device is not None else packed


def garbage_scores(scores: torch.Tensor, options: GraphOptions) -> torch.Tensor:
    if options.garbage == "any":
        return scores[..., ANY] - options.garbage_frame
    log_gap = scores[..., GAP].clamp(max=-1e-6)
    return torch.log(-torch.expm1(log_gap)).clamp(min=-1e4) - options.garbage_frame


def emissions(scores: torch.Tensor, packed: Packed, garbage: torch.Tensor) -> torch.Tensor:
    batch, frames, n_labels = scores.shape
    extended = torch.cat([scores, garbage.unsqueeze(-1)], dim=-1)
    index = torch.where(packed.labels < 0, n_labels, packed.labels)
    return extended.gather(2, index.unsqueeze(1).expand(batch, frames, index.shape[1]))


def boundary_logs(boundary_logits: torch.Tensor | None, shape: tuple[int, int], like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if boundary_logits is None:
        zeros = like.new_zeros(shape)
        return zeros, zeros
    logits = boundary_logits.float()
    return torch.nn.functional.logsigmoid(logits), torch.nn.functional.logsigmoid(-logits)


def _arc_weights(g: Packed, logb_t: torch.Tensor, log1mb_t: torch.Tensor) -> torch.Tensor:
    if not g.segmental:
        return g.weight
    zero = logb_t.new_zeros(())
    return g.weight + torch.where(g.enter, logb_t[:, None, None], zero) + torch.where(g.stay, log1mb_t[:, None, None], zero)


def _arc_table(g: Packed, logb: torch.Tensor, log1mb: torch.Tensor) -> torch.Tensor:
    if not g.segmental:
        return g.weight[:, None].expand(-1, logb.shape[1], -1, -1)
    zero = logb.new_zeros(())
    return (g.weight[:, None] + torch.where(g.enter[:, None], logb[:, :, None, None], zero)
            + torch.where(g.stay[:, None], log1mb[:, :, None, None], zero))


def _active(lengths: torch.Tensor, frames: int) -> torch.Tensor:
    return torch.arange(frames, device=lengths.device)[None, :] < lengths[:, None]


def _alpha(E: torch.Tensor, logb: torch.Tensor, log1mb: torch.Tensor, g: Packed, lengths: torch.Tensor):
    batch, frames, n = E.shape
    k = g.pred.shape[2]
    pred = g.pred.reshape(batch, n * k)
    alpha = E.new_empty(batch, frames, n)
    weights = _arc_table(g, logb, log1mb)
    active = _active(lengths, frames)
    a = g.start + E[:, 0]
    alpha[:, 0] = a
    for t in range(1, frames):
        prev = a.gather(1, pred).view(batch, n, k)
        new = torch.logsumexp(prev + weights[:, t], dim=2) + E[:, t]
        a = torch.where(active[:, t, None], new, a)
        alpha[:, t] = a
    return alpha, torch.logsumexp(a + g.final, dim=1)


def _beta(E: torch.Tensor, logb: torch.Tensor, log1mb: torch.Tensor, g: Packed, lengths: torch.Tensor, alpha: torch.Tensor | None = None,
          logF: torch.Tensor | None = None):
    batch, frames, n = E.shape
    k = g.pred.shape[2]
    succ = g.succ.reshape(batch, -1)
    pred = g.pred.reshape(batch, n * k)
    beta = E.new_full((batch, frames, n), NEG)
    last = lengths - 1
    none = E.new_full((batch, n), NEG)
    sums = alpha is not None and g.segmental
    enter_sum = E.new_zeros(batch, frames) if sums else None
    stay_sum = E.new_zeros(batch, frames) if sums else None
    safe = torch.where(torch.isfinite(logF), logF, torch.zeros_like(logF)) if sums else None
    weights = _arc_table(g, logb, log1mb)
    for t in range(frames - 1, -1, -1):
        rec = none
        if t < frames - 1:
            arc = weights[:, t + 1] + (E[:, t + 1] + beta[:, t + 1]).unsqueeze(2)
            values = arc.reshape(batch, n * k).gather(1, succ).view(batch, n, -1).masked_fill(~g.succ_valid, NEG)
            rec = torch.logsumexp(values, dim=2)
            if sums:
                xi = torch.exp(alpha[:, t].gather(1, pred).view(batch, n, k) + arc - safe[:, None, None])
                enter_sum[:, t + 1] = (xi * g.enter).sum((1, 2))
                stay_sum[:, t + 1] = (xi * g.stay).sum((1, 2))
        beta[:, t] = torch.where((t == last)[:, None], g.final, torch.where((t < last)[:, None], rec, none))
    if sums:
        return beta, enter_sum, stay_sum
    return beta


def _arc_posteriors(E, logb, log1mb, g, alpha, beta, logF, want_self: bool):
    batch, frames, n = E.shape
    k = g.pred.shape[2]
    pred = g.pred.reshape(batch, n * k)
    enter_sum = E.new_zeros(batch, frames)
    stay_sum = E.new_zeros(batch, frames)
    self_post = E.new_zeros(batch, frames, n) if want_self else None
    safe = torch.where(torch.isfinite(logF), logF, torch.zeros_like(logF))
    weights = _arc_table(g, logb, log1mb)
    for t in range(1, frames):
        prev = alpha[:, t - 1].gather(1, pred).view(batch, n, k)
        xi = torch.exp(prev + weights[:, t] + (E[:, t] + beta[:, t]).unsqueeze(2) - safe[:, None, None])
        if g.segmental:
            enter_sum[:, t] = (xi * g.enter).sum((1, 2))
            stay_sum[:, t] = (xi * g.stay).sum((1, 2))
        if want_self:
            self_post[:, t] = (xi * g.self_loop).sum(2)
    return enter_sum, stay_sum, self_post


class _LogLikelihood(torch.autograd.Function):
    @staticmethod
    def forward(ctx, E, logb, log1mb, g, lengths):
        with torch.no_grad():
            alpha, logF = _alpha(E, logb, log1mb, g, lengths)
        ctx.save_for_backward(E, logb, log1mb, alpha, logF, lengths)
        ctx.graph = g
        return logF

    @staticmethod
    def backward(ctx, grad):
        E, logb, log1mb, alpha, logF, lengths = ctx.saved_tensors
        g = ctx.graph
        with torch.no_grad():
            ok = torch.isfinite(logF)
            safe = torch.where(ok, logF, torch.zeros_like(logF))
            scale = torch.where(ok, grad, torch.zeros_like(grad))
            grad_logb = grad_log1mb = None
            if g.segmental and (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
                beta, enter_sum, stay_sum = _beta(E, logb, log1mb, g, lengths, alpha, logF)
                grad_logb = enter_sum * scale[:, None]
                grad_log1mb = stay_sum * scale[:, None]
            else:
                beta = _beta(E, logb, log1mb, g, lengths)
            gamma = torch.exp(alpha + beta - safe[:, None, None]) * scale[:, None, None]
        return torch.nan_to_num(gamma), grad_logb, grad_log1mb, None, None


def log_likelihood(E: torch.Tensor, logb: torch.Tensor, log1mb: torch.Tensor, g: Packed, lengths: torch.Tensor) -> torch.Tensor:
    return _LogLikelihood.apply(E, logb, log1mb, g, lengths)


@dataclass
class Posteriors:
    log_likelihood: torch.Tensor
    occupancy: torch.Tensor
    self_loop: torch.Tensor

    def entry(self) -> torch.Tensor:
        return self.occupancy - self.self_loop

    def exit(self) -> torch.Tensor:
        following = torch.cat([self.self_loop[:, 1:], torch.zeros_like(self.self_loop[:, :1])], dim=1)
        return self.occupancy - following


@torch.no_grad()
def posteriors(E: torch.Tensor, logb: torch.Tensor, log1mb: torch.Tensor, g: Packed, lengths: torch.Tensor) -> Posteriors:
    alpha, logF = _alpha(E, logb, log1mb, g, lengths)
    beta = _beta(E, logb, log1mb, g, lengths)
    safe = torch.where(torch.isfinite(logF), logF, torch.zeros_like(logF))
    gamma = torch.nan_to_num(torch.exp(alpha + beta - safe[:, None, None]))
    _, _, self_post = _arc_posteriors(E, logb, log1mb, g, alpha, beta, logF, want_self=True)
    return Posteriors(log_likelihood=logF, occupancy=gamma, self_loop=torch.nan_to_num(self_post))


@torch.no_grad()
def viterbi(E: torch.Tensor, logb: torch.Tensor, log1mb: torch.Tensor, g: Packed,
            lengths: torch.Tensor) -> tuple[list[np.ndarray], torch.Tensor]:
    batch, frames, n = E.shape
    k = g.pred.shape[2]
    pred = g.pred.reshape(batch, n * k)
    pointers = torch.zeros(batch, frames, n, dtype=torch.int64, device=E.device)
    a = g.start + E[:, 0]
    for t in range(1, frames):
        prev = a.gather(1, pred).view(batch, n, k) + _arc_weights(g, logb[:, t], log1mb[:, t])
        best, arg = prev.max(dim=2)
        a = torch.where((t < lengths)[:, None], best + E[:, t], a)
        pointers[:, t] = arg
    score, end = (a + g.final).max(dim=1)
    pointers, pred_np, ends = pointers.cpu().numpy(), g.pred.cpu().numpy(), end.cpu().numpy()
    paths = []
    for b in range(batch):
        length = int(lengths[b])
        path = np.zeros(length, np.int64)
        s = int(ends[b])
        for t in range(length - 1, -1, -1):
            path[t] = s
            if t:
                s = int(pred_np[b, s, pointers[b, t, s]])
        paths.append(path)
    return paths, score


def free_log_partition(scores: torch.Tensor, logb: torch.Tensor, log1mb: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch, frames, n = scores.shape
    active = _active(lengths, frames)
    a = scores[:, 0]
    for t in range(1, frames):
        any_label = torch.logsumexp(a, dim=1, keepdim=True)
        not_gap = torch.logsumexp(a[:, 1:], dim=1, keepdim=True)
        enter_from = torch.cat([not_gap, any_label.expand(batch, n - 1)], dim=1)
        new = scores[:, t] + torch.logaddexp(a + log1mb[:, t, None], logb[:, t, None] + enter_from)
        a = torch.where(active[:, t, None], new, a)
    return torch.logsumexp(a, dim=1)


@torch.no_grad()
def free_viterbi(scores: torch.Tensor, logb: torch.Tensor, log1mb: torch.Tensor, lengths: torch.Tensor) -> list[list[tuple[int, int, int]]]:
    batch, frames, n = scores.shape
    stayed = torch.zeros(batch, frames, n, dtype=torch.bool, device=scores.device)
    from_any = torch.zeros(batch, frames, dtype=torch.int64, device=scores.device)
    from_text = torch.zeros(batch, frames, dtype=torch.int64, device=scores.device)
    a = scores[:, 0]
    for t in range(1, frames):
        best_any, arg_any = a.max(dim=1, keepdim=True)
        best_text, arg_text = a[:, 1:].max(dim=1, keepdim=True)
        enter = logb[:, t, None] + torch.cat([best_text, best_any.expand(batch, n - 1)], dim=1)
        stay = a + log1mb[:, t, None]
        new = scores[:, t] + torch.maximum(stay, enter)
        a = torch.where((t < lengths)[:, None], new, a)
        stayed[:, t] = stay >= enter
        from_any[:, t] = arg_any[:, 0]
        from_text[:, t] = arg_text[:, 0] + 1
    ends = a.argmax(dim=1).cpu().numpy()
    stayed, from_any, from_text = stayed.cpu().numpy(), from_any.cpu().numpy(), from_text.cpu().numpy()
    readings = []
    for b in range(batch):
        length = int(lengths[b])
        label, end, segments = int(ends[b]), length, []
        for t in range(length - 1, -1, -1):
            if t and stayed[b, t, label]:
                continue
            segments.append((label, t, end))
            end = t
            if t:
                label = int(from_text[b, t]) if label == GAP else int(from_any[b, t])
        readings.append(segments[::-1])
    return readings


def greedy_ctc(scores: torch.Tensor, lengths: torch.Tensor) -> list[list[tuple[int, int, int]]]:
    best = scores.argmax(dim=2).cpu().numpy()
    readings = []
    for b in range(scores.shape[0]):
        length = int(lengths[b])
        segments, prev = [], -1
        for t in range(length):
            label = int(best[b, t])
            if label != prev:
                segments.append((label, t, t + 1))
            else:
                segments[-1] = (label, segments[-1][1], t + 1)
            prev = label
        readings.append(segments)
    return readings


def reading_words(segments: list[tuple[int, int, int]], segmental: bool) -> list[list[int]]:
    words, current = [], []
    for label, _, _ in segments:
        if label == GAP:
            if segmental and current:
                words.append(current)
                current = []
            continue
        current.append(label)
    if current:
        words.append(current)
    return words


def sequence_scores(logp: torch.Tensor, topology: str, log_prior: torch.Tensor | None = None, prior_scale: float = 0.0) -> torch.Tensor:
    if topology == "ctc_prior" and log_prior is not None and prior_scale:
        return logp - prior_scale * log_prior.detach()
    return logp


def with_ink_prior(scores: torch.Tensor, empty: torch.Tensor | None, penalty: float) -> torch.Tensor:
    if empty is None or not penalty:
        return scores
    labels = torch.arange(scores.shape[-1], device=scores.device)
    return scores - penalty * (empty[..., None].to(scores.device) & (labels != GAP)).to(scores.dtype)


def nll(logp: torch.Tensor, boundary_logits: torch.Tensor | None, graphs: Packed, lengths: torch.Tensor, options: GraphOptions,
        normalise: bool = True, log_prior: torch.Tensor | None = None, prior_scale: float = 0.0,
        empty: torch.Tensor | None = None) -> torch.Tensor:
    scores = with_ink_prior(sequence_scores(logp, options.topology, log_prior, prior_scale), empty, options.ink_prior)
    batch, frames, _ = scores.shape
    logb, log1mb = boundary_logs(boundary_logits if options.segmental else None, (batch, frames), scores)
    E = emissions(scores, graphs, garbage_scores(scores if options.garbage == "any" else logp, options))
    loglik = log_likelihood(E, logb, log1mb, graphs, lengths)
    if options.segmental and normalise:
        loglik = loglik - free_log_partition(scores, logb, log1mb, lengths)
    return -loglik
