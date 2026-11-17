import io
import json
import sys

from lines_dejavu import render
from lxml import etree
from PIL import Image
from test_place import tiny_aligner

from littera import cli
from littera.model import save_model
from littera.pagexml import parse_points, read_lines

NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def make_page(tmp_path):
    page = Image.new("L", (520, 200), 255)
    lines = []
    for k, text in enumerate(["Hallo schrieb mir", "den Tag am Haus"]):
        image, cuts, centre = render(text, size=24)
        x, y = 30, 30 + 80 * k
        page.paste(image, (x, y))
        w, h = image.size
        lines.append((text, [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], [(x + 5, y + centre + 6), (x + w - 5, y + centre + 6)]))
    page.save(tmp_path / "page.png")
    region = "".join(
        f'<TextLine id="l{k}"><Coords points="{" ".join(f"{int(a)},{int(b)}" for a, b in poly)}"/>'
        f'<Baseline points="{" ".join(f"{int(a)},{int(b)}" for a, b in base)}"/>'
        f"<TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>"
        for k, (text, poly, base) in enumerate(lines)
    )
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><PcGts xmlns="{NS}"><Page imageFilename="page.png" imageWidth="520" imageHeight="200">'
           f'<TextRegion id="r0"><Coords points="0,0 520,0 520,200 0,200"/>{region}</TextRegion></Page></PcGts>')
    (tmp_path / "page.xml").write_text(xml, encoding="utf-8")


def run_cli(argv, monkeypatch):
    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer, encoding="utf-8"))
    cli.main(argv)
    sys.stdout.flush()
    return json.loads(buffer.getvalue().decode("utf-8"))


def test_page_and_place_commands(tmp_path, monkeypatch):
    aligner = tiny_aligner()
    aligner.geometry.height = 32
    save_model(tmp_path / "model", aligner.model, aligner.config())
    make_page(tmp_path)
    summary = run_cli(["page", str(tmp_path / "page.xml"), "--image", str(tmp_path / "page.png"), "--model", str(tmp_path / "model"),
                       "-o", str(tmp_path / "out.xml")], monkeypatch)
    assert summary["placed"] == 2
    tree = etree.parse(str(tmp_path / "out.xml"))
    ns = {"p": NS}
    for line, text in zip(read_lines(tree), ["Hallo schrieb mir", "den Tag am Haus"]):
        words = line.element.findall("p:Word", ns)
        assert [w.find("p:TextEquiv/p:Unicode", ns).text for w in words] == text.split()
        children = [etree.QName(c).localname for c in line.element]
        assert children == ["Coords", "Baseline", "Word", "Word", "Word", "Word"][: 2 + len(words)] + ["TextEquiv"]
        for word in words:
            glyphs = word.findall("p:Glyph", ns)
            assert len(glyphs) == len(word.find("p:TextEquiv/p:Unicode", ns).text)
            for x, y in parse_points(word.find("p:Coords", ns).get("points")):
                assert 0 <= x <= 520 and 0 <= y <= 200
            ys = [y for _, y in parse_points(word.find("p:Coords", ns).get("points"))]
            top = min(p[1] for p in line.polygon) - 10
            assert min(ys) >= top
    placement = run_cli(["place", str(tmp_path / "page.png"), "Hallo", "--model", str(tmp_path / "model")], monkeypatch)
    assert placement["schema"] == "littera/placement/1" and len(placement["chars"]) == 5
