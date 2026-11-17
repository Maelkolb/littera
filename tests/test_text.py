from littera.text import ANY, GAP, INK, NO_INK, OPTIONAL, Codec, segment


def test_combining_marks_ride_on_their_base():
    text = "Häus ä"
    seg = segment(text)
    assert [u.key for u in seg.units] == ["H", "ä", "u", "s", "ä"]
    assert seg.unit_of[1] == seg.unit_of[2] == 1
    assert seg.unit_of[5] == -1
    assert seg.word_units == [[0, 1, 2, 3], [4]]
    assert seg.spans == [(0, 5), (6, 7)]


def test_line_end_hyphens_are_optional():
    for text in ["Ge-", "Ge¬", "Ge=", "Ge⸗", "Ge­", "Ge- "]:
        seg = segment(text)
        assert seg.units[-1].optional, text
    assert not any(u.optional for u in segment("a - b").units)
    assert segment("Ge­hen").roles[2] == NO_INK


def test_presets_mark_characters_without_ink():
    seg = segment("d[er] Mann [...] x", "editorial")
    assert seg.roles[1] == NO_INK and seg.roles[4] == NO_INK
    assert seg.roles[2] == OPTIONAL and seg.roles[3] == OPTIONAL
    assert all(seg.roles[i] == NO_INK for i in range(11, 16))
    assert [u.key for u in seg.units] == ["d", "e", "r", "M", "a", "n", "n", "x"]
    assert len(seg.word_units[2]) == 0
    tags = segment('<del rend="x">ab</del> c', "tags")
    assert [u.key for u in tags.units] == ["a", "b", "c"]
    assert segment("<b>", "plain").roles == [INK, INK, INK]
    both = segment("<i>d[er]</i>", "all")
    assert [u.key for u in both.units] == ["d", "e", "r"] and both.units[1].optional


def test_codec_folds_rare_clusters_and_round_trips():
    texts = ["aaa bbb", "á c", "ab"]
    codec = Codec.build(texts, min_count=2)
    assert codec.classes == ["a", "b"]
    assert codec.label("á") == codec.label("a")
    assert codec.label("á") == codec.label("a")
    assert codec.label("z") == ANY
    assert codec.counts["a"] == 5
    again = Codec.from_dict(codec.to_dict())
    assert again.classes == codec.classes and len(again) == len(codec) == 4
    encoded = codec.encode("ab z")
    assert encoded.labels == [codec.label("a"), codec.label("b"), ANY]
    assert encoded.graph_words() == [[(2, False), (3, False)], [(ANY, False)]]
    assert codec.decode([[2, 3], [ANY]]) == "ab �"
    assert codec.symbol(GAP) == "<gap>"


def roles_of(text: str, preset: str) -> str:
    return "".join({INK: "i", NO_INK: ".", OPTIONAL: "o"}[r] for r in segment(text, preset).roles)


def test_markup_preset_drops_markers_and_keeps_struck_ink():
    assert roles_of("ge~~str~~n", "markup") == "ii..iii..i"
    assert roles_of("2^te^ <u>Haus</u>", "markup") == "i.ii.i...iiii...."
    assert roles_of("Wort [sic!] ab [?] x", "markup") == "iiiii......iiii...ii"
    assert roles_of("[Johann] x", "markup") == "iiiiiiiiii"
    assert roles_of("〚unt〛 x", "markup") == ".iii.ii"
    assert roles_of("ein~", "markup") == "iiio"


def test_markup_line_end_hyphen_before_closing_markers_is_optional():
    assert roles_of("~~ge¬~~", "markup") == "..iio.."
    assert roles_of("vnd ge¬ ", "markup") == "iiiiiioi"
    assert segment("ge¬~~", "plain").roles[2] == INK


def test_composed_presets():
    assert roles_of("10t[en] [...]", "markup+brackets") == "iii.oo.i....."
    seg = segment("$hye$ in|den {x} l'of", "markup+brackets+dollar+bars+braces+apostrophe")
    assert [u.key for u in seg.units] == list("hyeindenxl'of")
    assert seg.units[-3].optional and seg.roles[18] == OPTIONAL
    try:
        segment("x", "markup+nothing")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown rule accepted")


def test_codec_build_takes_a_preset_per_text():
    codec = Codec.build(["^a^ ^b^", "^c^"], min_count=1, preset=["markup", "plain"])
    assert codec.classes == ["^", "a", "b", "c"]
    assert codec.counts["^"] == 2
