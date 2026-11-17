import math

import numpy as np
from PIL import Image, ImageDraw

from littera.baselines.common import placement_from_cuts
from littera.baselines.even import EvenPlacer
from littera.baselines.kraken_ctc import centres_to_cuts, ctc_viterbi
from littera.eval.ink import ink_profile, otsu_level
from littera.eval.metrics import aggregate, auc, changed_words, expected_calibration_error, predicted_cut, row_scores
from littera.types import word_spans


def blocks_image(spans, width=200, height=40):
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    for x0, x1 in spans:
        draw.rectangle([x0, 12, x1 - 1, 28], fill=0)
    return image


def test_auc_and_calibration():
    assert auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auc([0.5, 0.5], [1, 0]) == 0.5
    assert auc([0.3], [1]) is None
    assert expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    assert math.isclose(expected_calibration_error([0.95, 0.95], [False, False]), 0.95)


def test_changed_words():
    assert changed_words("a x c d", "a b c d") == [False, True, False, False]
    assert changed_words("a c d", "a b c d") == [False, False, False]
    assert changed_words("same", "same") == [False]


def test_perfect_placement_scores_zero():
    text = "ab cd"
    cuts = [10, 20, 30, 40, 50, 60]
    placement = placement_from_cuts(text, cuts, 100, 40, q=[0.9] * 5)
    assert [(w.x0, w.x1) for w in placement.words] == [(10, 30), (40, 60)]
    assert placement.chars[2].x0 == placement.chars[2].x1 == 35
    assert predicted_cut(placement, 1) == 20
    gold_words = [{"i0": 0, "i1": 2, "x0": 10, "x1": 30}, {"i0": 3, "i1": 5, "x0": 40, "x1": 60}]
    row = {"uid": "u", "text": text, "width": 100, "gold_words": gold_words,
           "gold_cuts": [{"i": 1, "x": 20.0}, {"i": 4, "x": 50.0}]}
    scores = row_scores(placement, row)
    summary = aggregate([scores])
    assert summary["word"]["mean"] == 0 and summary["word"]["iou"] == 1
    assert summary["letter"]["n"] == 2 and summary["letter"]["within_0.25"] == 1


def test_mismatch_auc_from_line_fit():
    placement_right = placement_from_cuts("ab cd", [0, 1, 2, 3, 4, 5], 10, 10, line_fit=0.9)
    placement_wrong = placement_from_cuts("ab xd", [0, 1, 2, 3, 4, 5], 10, 10, line_fit=0.2)
    right = row_scores(placement_right, {"uid": "a", "text": "ab cd", "width": 10, "label": 1, "variant": "right", "true_text": "ab cd"})
    wrong = row_scores(placement_wrong, {"uid": "b", "text": "ab xd", "width": 10, "label": 0, "variant": "replace", "true_text": "ab cd"})
    fit = aggregate([right, wrong])["fit"]
    assert fit["line_fit_auc"] == 1.0 and fit["line_fit_auc_replace"] == 1.0
    assert wrong["word_fits"][1][1] == 0 and wrong["word_fits"][0][1] == 1


def test_ink_profile_gaps_and_even_placer():
    image = blocks_image([(20, 60), (80, 140)])
    profile = ink_profile(image)
    assert profile.extent() == (20.0, 140.0)
    assert profile.is_gap(70) and not profile.is_gap(30)
    placement = EvenPlacer().place(image, "ab cd")
    assert placement.words[0].x0 == 20 and math.isclose(placement.words[-1].x1, 140)
    assert len(placement.chars) == 5 and not placement.chars[2].visible
    assert 0.3 < otsu_level(np.array([0.1] * 50 + [0.9] * 50)) < 0.9


def test_ctc_viterbi_and_centres():
    log_probs = np.log(np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]) + 1e-9)
    path = ctc_viterbi(log_probs, [1, 2])
    assert path == [1, 2, 3, 4]
    cuts = centres_to_cuts([10.0, None, 30.0], 100)
    assert cuts == [5.0, 15.0, 25.0, 35.0]
    assert word_spans(" a  bc ") == [(1, 2), (4, 6)]
