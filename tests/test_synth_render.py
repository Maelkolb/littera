import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from littera.eval.sets import read_set, write_set
from littera.synth.generate import SynthGenerator, to_row
from littera.synth.render import layout_line, load_font, pen_positions, place_geometry

DEJAVU = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


@pytest.fixture()
def fonts_dir(tmp_path):
    folder = tmp_path / "fonts" / "dejavu"
    folder.mkdir(parents=True)
    (folder / "DejaVuSans.ttf").write_bytes(DEJAVU.read_bytes())
    record = {"name": "DejaVu Sans", "style": "print", "split": "heldout", "licence": "Bitstream Vera", "url": "", "member": "",
              "licence_url": "", "file": "dejavu/DejaVuSans.ttf", "sha256": "", "licence_file": ""}
    (tmp_path / "fonts" / "fonts.json").write_text(json.dumps({"fonts": [record], "failures": []}))
    return tmp_path / "fonts"


def test_pen_positions_match_prefix_advance():
    font = load_font(str(DEJAVU), 40)
    text = "Wolf Tavern"
    pens = pen_positions(font, text)
    assert pens[0] == 0 and pens[-1] == font.getlength(text)
    assert all(b >= a for a, b in zip(pens, pens[1:]))
    assert pens[5] == font.getlength("Wolf T") - font.getlength("T")


def test_word_ink_inside_cuts_without_slant():
    layout = layout_line("Hallo Welt", str(DEJAVU), 48)
    geometry = place_geometry(layout, slant=0.0)
    cuts = {c["i"]: c["x"] for c in geometry.gold_cuts}
    for word in geometry.gold_words:
        assert cuts[word["i0"]] - 4 <= word["x0"] <= cuts[word["i1"]]
        assert word["x1"] <= cuts[word["i1"]] + 4
    assert 5 not in cuts or cuts[5] <= cuts[6]


def test_slant_moves_cuts_by_shear():
    layout = layout_line("ab cd", str(DEJAVU), 48)
    flat = place_geometry(layout, slant=0.0)
    slanted = place_geometry(layout, slant=0.3)
    shift = [s["x"] - f["x"] for s, f in zip(slanted.gold_cuts, flat.gold_cuts)]
    assert max(shift) - min(shift) < 1e-6
    assert slanted.mask.shape[1] > flat.mask.shape[1]


def test_generator_rows_round_trip(fonts_dir, tmp_path):
    generator = SynthGenerator(fonts_dir, split="heldout", seed=1, texts=["neighbour line here", "another one"])
    line = generator.render("Der Brief vom 3ten März")
    assert line is not None and line.text == "Der Brief vom 3ten März"
    xs = [c["x"] for c in line.gold_cuts]
    assert all(0 <= x <= line.image.width for x in xs) and xs == sorted(xs)
    assert len(line.gold_words) == 5
    path = tmp_path / "set" / "lines.parquet"
    write_set([to_row(line, "synth/00000")], path)
    row = read_set(path)[0]
    assert row["kind"] == "synthetic" and len(row["gold_cuts"]) == len(line.gold_cuts)
    assert np.isclose(row["gold_words"][0]["x0"], line.gold_words[0]["x0"], atol=1e-3)
    assert generator.render("中文") is None
