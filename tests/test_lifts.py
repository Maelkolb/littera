import itertools
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from lines_dejavu import encode_png, render, sentences
from test_graph import transcripts
from test_place import check_contract, tiny_aligner
from test_train import config

from littera import anchors, graph
from littera.geometry import GeometryConfig, prepare
from littera.graph import ENTER, GAP_STATE, LIFT, STAY, GraphOptions, build, pack
from littera.text import ANY, GAP, Codec
from littera.train import Trainer

WORDS = [[(2, False), (3, True), (3, False)], [(4, False)], [(2, True), (5, False), (2, True)]]


def lift_options(**kwargs):
    return GraphOptions(topology="segmental", spaces="optional", garbage_ends=True, garbage_between=True, garbage="any", lifts=True,
                        **kwargs)


def test_options_off_build_the_released_graph():
    old = build(WORDS, GraphOptions(spaces="optional", garbage_between=True))
    new = build(WORDS, GraphOptions(spaces="optional", garbage_between=True, lift_enter=3.0, lift_frame=1.0, lift_snap="lifts"))
    assert np.array_equal(old.arcs, new.arcs) and np.array_equal(old.arc_weight, new.arc_weight) and np.array_equal(old.labels, new.labels)
    assert (new.lift_states == -1).all()


def test_lift_states_sit_between_letters_of_a_word_only():
    g = build(WORDS, lift_options(lift_enter=0.7))
    lifts = set(np.flatnonzero(g.roles == LIFT).tolist())
    assert len(lifts) == 4 and [s >= 0 for s in g.lift_states] == [True, True, False, False, True, True, False]
    for src, dst, kind in g.arcs:
        if src in lifts and dst != src:
            assert g.roles[dst] == 0 and kind == ENTER
        if dst in lifts and src != dst:
            weight = g.arc_weight[(g.arcs[:, 0] == src) & (g.arcs[:, 1] == dst)][0]
            assert g.roles[src] == 0 and kind == ENTER and weight == np.float32(-0.7)
        if src in lifts and dst == src:
            assert kind == STAY
    assert all(g.final[s] == -np.inf and g.start[s] == -np.inf for s in lifts)


def paths(g, frames):
    successors = {}
    for src, dst, kind in g.arcs.tolist():
        successors.setdefault(src, []).append((dst, kind))
    stack = [[(s, None)] for s in range(g.n_states) if g.start[s] > -np.inf]
    while stack:
        path = stack.pop()
        if len(path) == frames:
            if g.final[path[-1][0]] > -np.inf:
                yield path
            continue
        stack += [path + [(dst, kind)] for dst, kind in successors.get(path[-1][0], [])]


def test_lift_paths_are_distinct_paths_of_the_free_model():
    g = build(WORDS, lift_options())
    free_label = {s: (GAP if g.roles[s] in (GAP_STATE, LIFT) else ANY if g.labels[s] < 0 else int(g.labels[s])) for s in range(g.n_states)}
    seen = set()
    with_lift = 0
    for path in paths(g, 9):
        signature = tuple((free_label[s], kind == ENTER) for s, kind in path)
        assert signature not in seen
        seen.add(signature)
        segments = [signature[0][0]] + [label for label, entered in signature[1:] if entered]
        assert all(not (a == GAP and b == GAP) for a, b in zip(segments, segments[1:]))
        with_lift += any(g.roles[s] == LIFT for s, _ in path)
    assert with_lift > 0 and len(seen) > with_lift


def test_lift_likelihood_stays_under_the_free_partition():
    torch.manual_seed(5)
    frames, labels = 5, 3
    words = list(transcripts("12", frames))
    batch = len(words)
    lengths = torch.full((batch,), frames)
    for enter, frame in ((0.0, 0.0), (0.5, 0.3)):
        options = GraphOptions(topology="segmental", spaces="required", garbage_ends=False, lifts=True, lift_enter=enter, lift_frame=frame)
        plain = GraphOptions(topology="segmental", spaces="required", garbage_ends=False)
        graphs = pack([build([[(label, False) for label in w] for w in ws], options) for ws in words])
        plain_graphs = pack([build([[(label, False) for label in w] for w in ws], plain) for ws in words])
        for _ in range(5):
            logp = torch.randn(1, frames, labels, dtype=torch.float64).log_softmax(-1).expand(batch, -1, -1)
            blogit = torch.randn(1, frames, dtype=torch.float64).expand(batch, -1)
            with_lifts = graph.nll(logp, blogit, graphs, lengths, options)
            without = graph.nll(logp, blogit, plain_graphs, lengths, plain)
            assert float(with_lifts.min()) >= -1e-9
            assert torch.all(with_lifts <= without + 1e-9)
            closed = graph.nll(logp, blogit, graphs, lengths, options, open_frames=torch.zeros(batch, frames, dtype=torch.bool))
            assert torch.allclose(closed, without, atol=1e-6)
            some = torch.zeros(batch, frames, dtype=torch.bool)
            some[:, 2] = True
            partly = graph.nll(logp, blogit, graphs, lengths, options, open_frames=some)
            assert torch.all(partly <= without + 1e-9) and torch.all(partly >= with_lifts - 1e-9)


def test_lift_gradients_match_finite_differences():
    torch.manual_seed(6)
    options = lift_options(lift_enter=0.4, lift_frame=0.2)
    graphs = pack([build(WORDS, options), build(WORDS[:2], options)])
    lengths = torch.tensor([11, 9])
    logp = torch.randn(2, 11, 6, dtype=torch.float64).log_softmax(-1)
    lift = graph.lift_scores(logp, options, torch.rand(2, 11) > 0.3)
    E = graph.emissions(logp, graphs, graph.garbage_scores(logp, options), lift).detach().requires_grad_(True)
    blogit = torch.randn(2, 11, dtype=torch.float64)
    logb = F.logsigmoid(blogit).requires_grad_(True)
    log1mb = F.logsigmoid(-blogit).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda e, a, b: graph.log_likelihood(e, a, b, graphs, lengths), (E, logb, log1mb), atol=1e-6)


def test_viterbi_takes_a_lift_where_the_scores_say_gap():
    options = GraphOptions(topology="segmental", spaces="required", garbage_ends=False, lifts=True)
    graphs = pack([build([[(2, False), (3, False)]], options)])
    logp = torch.full((1, 7, 4), -8.0, dtype=torch.float64)
    for t, label in enumerate([2, 2, 2, 0, 3, 3, 3]):
        logp[0, t, label] = -0.01
    blogit = torch.full((1, 7), -4.0, dtype=torch.float64)
    blogit[0, 3] = blogit[0, 4] = 4.0
    logb, log1mb = graph.boundary_logs(blogit, (1, 7), logp)
    for open_frame, expected in ((True, [0, 0, 0, 1, 2, 2, 2]), (False, None)):
        open_frames = torch.zeros(1, 7, dtype=torch.bool)
        open_frames[0, 3] = open_frame
        E = graph.emissions(logp, graphs, graph.garbage_scores(logp, options), graph.lift_scores(logp, options, open_frames))
        (path,), _ = graph.viterbi(E, logb, log1mb, graphs, torch.tensor([7]))
        roles = graphs.roles[0, path].tolist()
        assert (LIFT in roles) == open_frame
        if expected:
            lift_state = int(graphs.lift_states[0, 0])
            assert [0 if s == int(graphs.unit_states[0, 0]) else 1 if s == lift_state else 2 for s in path.tolist()] == expected


def separated_line():
    text = "Haus am Tag"
    image, cuts, _ = render(text, size=40)
    return text, image, cuts


def test_edges_next_to_a_lift_are_the_ink_edges():
    text, image, _ = separated_line()
    for snap in ("strip", "upright", "off"):
        aligner = tiny_aligner(snap=snap)
        aligner.geometry.height = 32
        plain = aligner.place(image, text)
        aligner.decode.lift_snap = "cuts"
        placement = aligner.place(image, text)
        check_contract(placement, text)
        assert "lift_edges" not in plain.extra and placement.extra["lift_edges"] > 0
        assert [(w.x0, w.x1) for w in placement.words] == [(w.x0, w.x1) for w in plain.words]
        analysis = aligner.analyse(image, text)
        profile = analysis.strip.core_profile()
        for wc in analysis.words:
            for k, (s, e) in wc.edges.items():
                assert wc.cuts[k - 1] < s < e < wc.cuts[k + 1] and wc.span(k - 1)[1] == s and wc.span(k)[0] == e
                assert profile[int(s) : int(e)].max() <= 1.0 and profile[int(s) - 1] > 1.0 and profile[int(e)] > 1.0
        aligner.decode.lifts, aligner.decode.lift_snap, aligner.decode.lift_frame = True, "lifts", -20.0
        lifted = aligner.place(image, text)
        check_contract(lifted, text)
        assert lifted.extra["lifts"] > 0 and lifted.extra["lift_edges"] > 0


def strip_of(image, height=96):
    return prepare(image, None, None, GeometryConfig(height=height))


def test_anchor_frame_targets_mark_gaps_letters_and_boundaries():
    text, image, _ = separated_line()
    strip = strip_of(image)
    runs = [r for r in anchors.strip_runs(strip) if r[0] > 0 and r[1] < strip.width]
    (s0, e0), (s1, e1) = runs[0], runs[1]
    record = {"gaps": [[0, float(strip.crop_x(s0)), float(strip.crop_x(e0)), "lift"],
                       [1, float(strip.crop_x(s1)), float(strip.crop_x(e1)), "lift"], [5, -500.0, -490.0, "lift"]],
              "letters": [[0, "H"], [3, "s"]]}
    frames = strip.width // 4
    targets, known, labels = anchors.frame_targets(record, strip, [7, 8, 9, 10, 11, 12], frames, 4)
    a0, b0 = anchors.gap_frames(s0, e0, 4)
    a1, b1 = anchors.gap_frames(s1, e1, 4)
    assert targets == sorted({a0, b0, a1, b1})
    assert (labels[a0:b0] == GAP).all() and (labels[a1:b1] == GAP).all() and (labels[b0:a1] == 7).all() and b0 < a1
    assert known[a0 : b1 + 1].all() and not known[: a0].any() and not known[b1 + 1 :].any()
    assert (labels[~known] == anchors.IGNORE).all()
    shifted, _, _ = anchors.frame_targets(record, strip, [7, 8, 9, 10, 11, 12], frames, 4, shift=1.0)
    assert shifted == targets


def test_lifts_at_cuts_turn_exact_cuts_into_gap_frames():
    text, image, cuts = separated_line()
    strip = strip_of(image)
    codec = Codec.build([text], min_count=1)
    encoded = codec.encode(text)
    frames = strip.width // 4
    cut_at = dict(cuts)
    labels = np.zeros(frames, np.int64)
    for label, unit in zip(encoded.labels, encoded.seg.units):
        f0, f1 = (int(round(float(strip.strip_u(cut_at[i])) / 4)) for i in (unit.i0, unit.i1))
        labels[f0:f1] = label
    before = labels.copy()
    targets = sorted({int(round(float(strip.strip_u(x)) / 4)) for _, x in cuts})
    lifted = anchors.lifts_at_cuts(encoded, cut_at, strip, 4, labels, targets)
    assert len(lifted) > len(targets) and (labels != before).any()
    order = [int(k) for k, _ in itertools.groupby(labels.tolist())]
    letters = [label for label in order if label != GAP]
    assert letters == encoded.labels
    opened = strip.open_frames(4)
    changed = np.flatnonzero((labels == GAP) & (before != GAP))
    assert opened[changed].all()


def write_pool(path, n=12):
    rows, records = [], []
    for k, text in enumerate(sentences(n, seed=2)):
        image, cuts, _ = render(text, size=22)
        exact = k % 3 == 0
        rows.append({"uid": f"syn/d/p/{k}", "source": "syn", "split": "train", "image": encode_png(image), "text": text,
                     "width": image.width, "height": image.height, "polygon": [], "baseline": [],
                     "gold_cuts": [{"i": i, "x": float(x)} for i, x in cuts] if exact else None})
        if not exact:
            strip = strip_of(image, 32)
            run = next(r for r in anchors.strip_runs(strip) if r[0] > strip.pad + 2)
            gap = [1, float(strip.crop_x(run[0])), float(strip.crop_x(run[1])), "lift"]
            records.append({"uid": rows[-1]["uid"], "gaps": [gap], "letters": []})
    pq.write_table(pa.Table.from_pylist(rows), path)
    return records


def test_training_with_lifts_anchors_and_lifted_cuts(tmp_path):
    (tmp_path / "pool").mkdir()
    records = write_pool(tmp_path / "pool" / "part-00000.parquet")
    anchors.write(tmp_path / "anchors" / "part-0000.jsonl", records)
    graph_options = {"spaces": "required", "garbage_ends": True, "garbage_between": False, "lifts": True}
    trainer = Trainer(config(tmp_path, steps=6, graph=graph_options, decode={"lifts": True, "lift_snap": "lifts"},
                             anchors=str(tmp_path / "anchors"), lift_cuts=True, augment={"shear": 1.0}), device="cpu")
    batch = next(iter(trainer.loader(trainer.train_set, shuffle=False)))
    assert "open" in batch and batch["open"].any()
    partial = batch["target_mask"].any(1) & ~batch["target_mask"].all(1)
    assert partial.any() and batch["target_mask"].any()
    out = trainer.losses(batch, train=True)
    assert torch.isfinite(out["loss"]) and int(out["dropped"]) == 0 and float(out["bce"]) > 0 and float(out["frame"]) > 0
    trainer.train()
    saved = json.loads((tmp_path / "run" / "last" / "config.json").read_text(encoding="utf-8"))
    assert saved["decode"]["lifts"] is True and saved["decode"]["lift_snap"] == "lifts"
    log = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert all(r["dropped"] == 0 for r in log if "dropped" in r)


def test_kept_inner_edges_stay_where_the_model_put_them():
    text = "Hallo schrieb mir den Tag"
    image, _, _ = render(text, slant_deg=15)
    for snap in ("strip", "upright"):
        aligner = tiny_aligner(snap="off")
        aligner.geometry.height = 32
        raw = aligner.analyse(image, text)
        aligner.snap = snap
        rescaled = aligner.analyse(image, text)
        aligner.decode.inner_edges = "keep"
        kept = aligner.analyse(image, text)
        check_contract(aligner.to_placement(kept, text), text)
        moved = kept_words = 0
        for a, b, c in zip(raw.words, rescaled.words, kept.words):
            assert abs(c.cuts[0] - b.cuts[0]) < 1e-6 and abs(c.cuts[-1] - b.cuts[-1]) < 1e-6
            margin = 0.25 * (c.cuts[-1] - c.cuts[0]) / len(c.units)
            inside = all(c.cuts[0] + margin <= u <= c.cuts[-1] - margin for u in a.cuts[1:-1])
            assert c.kept == inside
            wanted = a.cuts if inside else b.cuts
            assert all(abs(w - u) < 1e-6 for u, w in zip(wanted[1:-1], c.cuts[1:-1]))
            assert all(x < y for x, y in zip(c.cuts, c.cuts[1:])) or not inside
            moved += sum(abs(v - u) > 1e-6 for u, v in zip(a.cuts[1:-1], b.cuts[1:-1]))
            kept_words += inside and len(c.units) > 1
        assert moved > 0 and kept_words > 0
