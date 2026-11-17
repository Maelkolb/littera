import itertools

import numpy as np
import torch
import torch.nn.functional as F

from littera import graph
from littera.graph import GraphOptions, build, pack


def ctc_options():
    return GraphOptions(topology="ctc", spaces="optional", space_penalty=0.0, garbage_ends=False)


def test_ctc_topology_equals_torch_ctc_loss():
    torch.manual_seed(0)
    targets = [[[2, 2, 3]], [[4, 5], [5, 5, 2]], [[1, 3, 3, 2, 2, 4, 4]], [[2]], [[3, 3, 3, 3]]]
    lengths = torch.tensor([30, 25, 28, 10, 12])
    logits = torch.randn(len(targets), 30, 6, dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(-1)
    options = ctc_options()
    graphs = pack([build([[(label, False) for label in word] for word in words], options) for words in targets])
    ours = graph.nll(logp, None, graphs, lengths, options)
    flat = [label for words in targets for word in words for label in word]
    target_lengths = torch.tensor([sum(len(w) for w in words) for words in targets])
    theirs = F.ctc_loss(logp.transpose(0, 1), torch.tensor(flat), lengths, target_lengths, blank=0, reduction="none", zero_infinity=False)
    assert torch.allclose(ours, theirs, atol=1e-8)
    grad_ours = torch.autograd.grad(ours.sum(), logits, retain_graph=True)[0]
    grad_theirs = torch.autograd.grad(theirs.sum(), logits)[0]
    assert torch.allclose(grad_ours, grad_theirs, atol=1e-8)


def test_ctc_infeasible_line_is_infinite_with_zero_gradient():
    logits = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    options = ctc_options()
    graphs = pack([build([[(2, False), (2, False), (2, False)]], options)])
    loss = graph.nll(logits.log_softmax(-1), None, graphs, torch.tensor([3]), options)
    assert torch.isinf(loss).all()
    (grad,) = torch.autograd.grad(loss.masked_fill(torch.isinf(loss), 0).sum() + 0 * logits.sum(), logits)
    assert torch.isfinite(grad).all()


def segmental_graph(options=None):
    options = options or GraphOptions(topology="segmental", spaces="optional", garbage_ends=True, garbage_between=True)
    words = [[(2, False), (3, True), (3, False)], [(4, False)], [(2, True), (5, False)]]
    return pack([build(words, options), build(words[:2], options)]), options


def test_segmental_gradients_match_finite_differences():
    torch.manual_seed(1)
    graphs, options = segmental_graph()
    batch, frames, labels = 2, 10, 6
    lengths = torch.tensor([10, 8])
    logp = torch.randn(batch, frames, labels, dtype=torch.float64).log_softmax(-1)
    E = graph.emissions(logp, graphs, graph.garbage_scores(logp, options)).detach().requires_grad_(True)
    blogit = torch.randn(batch, frames, dtype=torch.float64)
    logb = F.logsigmoid(blogit).requires_grad_(True)
    log1mb = F.logsigmoid(-blogit).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda e, a, b: graph.log_likelihood(e, a, b, graphs, lengths), (E, logb, log1mb), atol=1e-6)


def test_posteriors_sum_to_one_per_frame_and_viterbi_is_a_path():
    torch.manual_seed(2)
    graphs, options = segmental_graph()
    lengths = torch.tensor([14, 11])
    logp = torch.randn(2, 14, 6).log_softmax(-1)
    blogit = torch.randn(2, 14)
    logb, log1mb = graph.boundary_logs(blogit, (2, 14), logp)
    E = graph.emissions(logp, graphs, graph.garbage_scores(logp, options))
    post = graph.posteriors(E, logb, log1mb, graphs, lengths)
    for b, length in enumerate(lengths.tolist()):
        assert torch.allclose(post.occupancy[b, :length].sum(1), torch.ones(length), atol=1e-4)
        assert torch.all(post.occupancy[b, length:] == 0)
        entry = post.entry()[b, :length]
        assert entry.min() > -1e-4
    paths, score = graph.viterbi(E, logb, log1mb, graphs, lengths)
    pred = graphs.pred.numpy()
    weight = graphs.weight.numpy()
    for b, path in enumerate(paths):
        assert len(path) == lengths[b]
        assert graphs.start[b, path[0]] > -np.inf
        assert graphs.final[b, path[-1]] > -np.inf
        for s, t in zip(path[:-1], path[1:]):
            assert any(pred[b, t, k] == s and weight[b, t, k] > -np.inf for k in range(pred.shape[2]))
        units = graphs.unit_states[b].numpy()
        visited = set(path.tolist())
        required = [units[0], units[2], units[3]] if b == 0 else list(units[units >= 0])
        assert all(u in visited for u in required)
    assert torch.all(score <= post.log_likelihood + 1e-4)


def transcripts(alphabet, frames):
    for n in range(1, frames + 1):
        for symbols in itertools.product(list(alphabet) + [" "], repeat=n):
            text = "".join(symbols)
            if text.startswith(" ") or text.endswith(" ") or "  " in text:
                continue
            yield [[int(c) for c in word] for word in text.split(" ")]


def test_segmental_likelihood_is_normalised_over_all_transcripts():
    torch.manual_seed(3)
    frames, labels = 5, 3
    options = GraphOptions(topology="segmental", spaces="required", garbage_ends=False, garbage_between=False)
    words = list(transcripts("12", frames))
    batch = len(words)
    logp = torch.randn(1, frames, labels, dtype=torch.float64).log_softmax(-1)
    blogit = torch.randn(1, frames, dtype=torch.float64)
    graphs = pack([build([[(label, False) for label in w] for w in ws], options) for ws in words])
    lengths = torch.full((batch,), frames)
    logb, log1mb = graph.boundary_logs(blogit, (1, frames), logp)
    E = graph.emissions(logp.expand(batch, -1, -1), graphs, graph.garbage_scores(logp.expand(batch, -1, -1), options))
    logF = graph.log_likelihood(E, logb.expand(batch, -1), log1mb.expand(batch, -1), graphs, lengths)
    all_gap = logp[0, :, 0].sum() + log1mb[0, 1:].sum()
    total = torch.logsumexp(torch.cat([logF, all_gap.view(1)]), 0)
    logZ = graph.free_log_partition(logp, logb, log1mb, torch.tensor([frames]))[0]
    assert abs(float(total - logZ)) < 1e-9
    losses = graph.nll(logp.expand(batch, -1, -1), blogit.expand(batch, -1), graphs, lengths, options)
    probabilities = torch.exp(-losses)
    assert float(probabilities.sum() + torch.exp(all_gap - logZ)) == __import__("pytest").approx(1.0, abs=1e-9)


def test_free_viterbi_finds_the_best_consistent_labelling():
    torch.manual_seed(4)
    frames, labels = 6, 3
    logp = torch.randn(1, frames, labels, dtype=torch.float64).log_softmax(-1)
    blogit = torch.randn(1, frames, dtype=torch.float64)
    logb, log1mb = graph.boundary_logs(blogit, (1, frames), logp)
    (segments,) = graph.free_viterbi(logp, logb, log1mb, torch.tensor([frames]))
    assert segments[0][1] == 0 and segments[-1][2] == frames
    assert all(a[2] == b[1] for a, b in zip(segments, segments[1:]))

    def weight(segs):
        total = 0.0
        for label, t0, t1 in segs:
            total += float(logp[0, t0:t1, label].sum() + log1mb[0, t0 + 1 : t1].sum()) + (float(logb[0, t0]) if t0 else 0.0)
        return total

    best = -np.inf
    for sequence in itertools.product(range(labels), repeat=frames):
        for starts in itertools.product([False, True], repeat=frames - 1):
            cuts = [0] + [t + 1 for t, s in enumerate(starts) if s] + [frames]
            segs = [(sequence[a], a, b) for a, b in zip(cuts, cuts[1:])]
            if any(len(set(sequence[a:b])) > 1 for _, a, b in segs):
                continue
            if any(x[0] == 0 and y[0] == 0 for x, y in zip(segs, segs[1:])):
                continue
            best = max(best, weight(segs))
    assert abs(weight(segments) - best) < 1e-9
