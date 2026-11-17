import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from littera.data.laubmann import line_problem, line_rows
from littera.text import mark
from tools.mark.build import cut_model, prefill
from tools.mark.ingest import gold_rows


def region(text: str, boxes: list[list[int]]) -> dict:
    image = Image.new("RGB", (900, 170), (236, 228, 210))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(28)
    for line, box in zip(text.split("\n"), boxes):
        draw.text((40, box[1] + 10), line, fill=(40, 30, 20), font=font)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return {"extra": json.dumps({"line_boxes": boxes}), "target": text, "sample_id": "laubmann/L/p/region/r01", "collection": "L",
            "page_id": "p", "image": buffer.getvalue(), "script": "kurrent", "language": "de", "license": "internal", "quality": "gold",
            "file": "staged/laubmann/w000.parquet"}


def test_line_rows_cut_each_line_and_keep_the_text():
    rows, dropped = line_rows(region("Hier befand ich mich\nMittags in der Zugwelle", [[8, 10, 890, 62], [8, 80, 890, 135]]))
    assert dropped == {}
    assert [r["text"] for r in rows] == ["Hier befand ich mich", "Mittags in der Zugwelle"]
    assert rows[0]["uid"] == "laubmann/L/p/r01_l01" and rows[1]["split"] == "heldout" and not rows[1]["open_licence"]
    assert all(r["width"] < 600 for r in rows)


def test_line_problem_rules():
    assert line_problem("a", 100, 40) == "short_text"
    assert line_problem("Wort</u> und", 300, 40) == "markup_remnant"
    assert line_problem("ab", 600, 40) == "width_per_char"
    assert line_problem("Hier befand ich", 300, 40) == ""


def test_cut_model_word_edges_inner_cuts_and_markup():
    text = "ab <u>cd</u>"
    cuts, letters = cut_model(text, mark(text, "tags"))
    assert [(c["kind"], c["i"]) for c in cuts] == [("start", 0), ("inner", 1), ("end", 2), ("start", 6), ("inner", 7), ("end", 8)]
    assert cuts[3]["idx"] == [3, 4, 5, 6] and cuts[5]["idx"] == [8, 9, 10, 11, 12]
    assert [(t["g"], t["l"], t["r"]) for t in letters] == [("a", 0, 1), ("b", 1, 2), ("c", 3, 4), ("d", 4, 5)]


def test_cut_model_keeps_combining_marks_on_their_base():
    text = "qu̅e"
    cuts, letters = cut_model(text, mark(text, "plain"))
    assert [c["i"] for c in cuts] == [0, 1, 3, 4]
    assert letters[1]["g"] == "u̅"


def test_prefill_without_placement_spreads_over_the_ink():
    xs, used = prefill([{}, {}, {}], None, 200, (20.0, 120.0))
    assert used == "even" and xs == [20.0, 70.0, 120.0]


def test_gold_rows_take_done_lines_only():
    candidates = {"s/d/p/1": {"uid": "s/d/p/1", "text": "ab", "origin": "{}"}, "s/d/p/2": {"uid": "s/d/p/2", "text": "cd", "origin": "{}"}}
    marks = {"exported": "now", "placer": "even", "lines": [
        {"uid": "s/d/p/1", "text": "ab", "status": "done", "gold_cuts": [{"i": 1, "x": 5.0}, {"i": 0, "x": 1.0}, {"i": 2, "x": 9.0}]},
        {"uid": "s/d/p/2", "text": "cd", "status": "mismatch", "gold_cuts": []}]}
    rows, counts = gold_rows(marks, candidates)
    assert len(rows) == 1 and rows[0]["kind"] == "letter"
    assert rows[0]["gold_cuts"] == [{"i": 0, "x": 1.0}, {"i": 1, "x": 5.0}, {"i": 2, "x": 9.0}]
    assert counts["done"] == 1 and counts["mismatch"] == 1
