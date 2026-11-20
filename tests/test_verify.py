import random

from littera.verify import Corruptor, _levenshtein

CONFUSIONS = {"sub": {"e": {"a": 5, "n": 2}, "n": {"m": 4}}, "del": {"e": 3, " ": 1}, "ins": {"e": 2, "i": 1}, "edits": {"1": 10, "2": 4, "3": 1}}


def test_corruptions_differ_and_stay_close():
    corruptor = Corruptor(CONFUSIONS, ["Herz", "Harz", "Hertz", "machen", "wachen"])
    rng = random.Random(0)
    text = "das Herz zu machen und eine Entdeckung"
    made = [corruptor(text, rng) for _ in range(200)]
    made = [m for m in made if m]
    assert len(made) > 150
    assert all(m != text for m in made)
    assert all(_levenshtein(m, text, 12) <= 6 for m in made)


def test_corruptions_repeat_with_the_same_seed():
    corruptor = Corruptor(CONFUSIONS, ["Herz", "Harz"])
    a = [corruptor("das Herz schlug", random.Random(3)) for _ in range(5)]
    b = [corruptor("das Herz schlug", random.Random(3)) for _ in range(5)]
    assert a == b


def test_no_words_no_corruption():
    assert Corruptor(CONFUSIONS, []).__call__("12 3 .", random.Random(1)) is None


def test_neighbour_is_a_vocabulary_word():
    corruptor = Corruptor(CONFUSIONS, ["Harz", "Hertz"])
    assert corruptor._neighbour("Herz", random.Random(0)) in {"Harz", "Hertz"}
