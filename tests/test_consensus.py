import pytest

from littera.consensus import blend_edges


def test_inner_edges_are_the_mean_and_word_ends_stay():
    text = "abc de"
    own = [10, 20, 30, 40, 50, 60, 70]
    other = [12, 24, 36, 44, 52, 66, 72]
    edges, blended, eligible = blend_edges(text, own, other)
    assert edges == [10, 22, 33, 40, 50, 63, 70]
    assert (blended, eligible) == (2, 2)


def test_word_with_a_skipped_character_keeps_own():
    own = [10, 20, 20, 40]
    other = [10, 18, 30, 40]
    assert blend_edges("abc", own, other) == ([10, 20, 20, 40], 0, 1)


def test_word_the_other_placement_puts_elsewhere_keeps_own():
    own = [10, 20, 30, 40]
    other = [60, 70, 80, 90]
    assert blend_edges("abc", own, other)[0] == [10, 20, 30, 40]


def test_mean_edge_too_close_to_a_word_end_keeps_own():
    own = [10, 13, 30, 40]
    other = [10, 11, 30, 40]
    assert blend_edges("abc", own, other)[0] == [10, 13, 30, 40]


def test_lengths_must_match():
    with pytest.raises(ValueError):
        blend_edges("abc", [0, 1, 2], [0, 1, 2, 3])
