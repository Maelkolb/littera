import json

import pytest
import torch
from lines_dejavu import render, sentences

from littera.model import LineModel, ModelConfig, save_model
from littera.place import Aligner, Objective, snap_mode
from littera.text import Codec
from littera.types import Placement, word_spans


def tiny_aligner(topology="segmental", **kwargs):
    torch.manual_seed(0)
    codec = Codec.build(sentences(30) + ["Hallo [er] Tag ¬ ä"], min_count=1)
    model = LineModel(ModelConfig(n_labels=len(codec), height=32, channels=[8, 8, 16, 16], hidden=16, layers=1))
    aligner = Aligner(model, codec, objective=Objective(topology=topology), **kwargs)
    return aligner


def area(poly):
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1])) / 2


def check_contract(placement: Placement, text: str):
    assert placement.text == text
    assert [c.ch for c in placement.chars] == list(text)
    assert [c.i for c in placement.chars] == list(range(len(text)))
    assert [(w.i0, w.i1) for w in placement.words] == word_spans(text)
    assert 0 <= placement.line_fit <= 1
    last = 0.0
    for c in placement.chars:
        assert 0 <= c.x0 <= c.x1 <= placement.width
        assert 0 <= c.q <= 1
        assert len(c.poly) == 4 and area(c.poly) >= -1e-6
        if not c.visible:
            assert c.x0 == c.x1
        if c.ch.isspace():
            assert not c.visible and not c.aligned
        if c.visible:
            assert c.x0 >= last - 1e-6
            last = c.x0
    for w in placement.words:
        assert 0 <= w.q <= 1 and 0 <= w.fit <= 1 and w.x0 <= w.x1
        assert len(w.poly) == 4 and area(w.poly) >= -1e-6
    again = Placement.from_dict(json.loads(json.dumps(placement.to_dict())))
    assert again == placement


@pytest.mark.parametrize("topology", ["segmental", "ctc", "ctc_prior"])
def test_placement_contract(topology):
    aligner = tiny_aligner(topology)
    aligner.geometry.height = 32
    for text in ["Hallo schrieb mir den Tag", " Häus  am Tag¬ ", "x"]:
        image, cuts, centre = render(text.strip() or "x", slant_deg=12)
        check_contract(aligner.place(image, text), text)


def test_markup_combining_marks_and_degenerate_lines():
    aligner = tiny_aligner(preset="editorial")
    aligner.geometry.height = 32
    image, cuts, centre = render("dem Mann Tag")
    text = "d[er] Männ [...] Tag"
    placement = aligner.place(image, text)
    check_contract(placement, text)
    assert not placement.chars[1].aligned and not placement.chars[4].aligned
    assert placement.chars[7].x0 == placement.chars[8].x0 and placement.chars[7].x1 == placement.chars[8].x1
    assert placement.words[2].x0 == placement.words[2].x1
    for text in ["", "   ", "[...]"]:
        check_contract(aligner.place(image, text), text)
    long_text = "und " * 60
    tiny = image.crop((0, 0, 40, image.height))
    placement = aligner.place(tiny, long_text)
    check_contract(placement, long_text)
    assert placement.extra.get("fallback") == "even" and placement.line_fit == 0.0


def test_save_load_and_read(tmp_path):
    aligner = tiny_aligner("ctc_prior")
    aligner.geometry.height = 32
    save_model(tmp_path / "m", aligner.model, aligner.config())
    loaded = Aligner.load(tmp_path / "m")
    image, cuts, centre = render("Haus am Tag")
    a, b = aligner.place(image, "Haus am Tag"), loaded.place(image, "Haus am Tag")
    assert a == b
    assert isinstance(loaded.read(image), str)


@pytest.mark.parametrize("snap", ["strip", "upright", "off", True, False])
def test_snap_modes(snap):
    aligner = tiny_aligner(snap=snap)
    aligner.geometry.height = 32
    text = "Hallo schrieb mir den Tag"
    image, _, _ = render(text, slant_deg=15)
    placement = aligner.place(image, text)
    check_contract(placement, text)
    assert aligner.snap == snap_mode(snap)


def test_snap_mode_names():
    assert [snap_mode(v) for v in (True, False, None, "strip", "upright", "off")] == ["strip", "off", "off", "strip", "upright", "off"]
    with pytest.raises(ValueError):
        snap_mode("valley")
