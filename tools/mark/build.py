from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from littera.eval.ink import ink_profile
from littera.eval.placers import make_placer, placer_label
from littera.eval.run import baseline_or_none, load_placements, polygon_or_none
from littera.eval.sets import decode_image, read_set, set_path
from littera.geometry import prepare
from littera.text import NO_INK, mark
from littera.types import Placement, word_spans

ROOT = Path(__file__).resolve().parents[2]
PAGE = Path(__file__).with_name("page.html")
FONT = ROOT / "data" / "fonts" / "junicode-two-beta" / "JunicodeTwoBeta-Regular.ttf"
SOURCES = ROOT / "configs" / "sources.yaml"
SET = "letter-gold-candidates"
DEFAULT_OUT = Path.home() / "Downloads" / "Littera_Letter_Gold"
CENTRE_POINTS = 33


def markup_presets() -> dict[str, str]:
    import yaml

    sources = yaml.safe_load(SOURCES.read_text(encoding="utf-8"))["sources"]
    return {name: entry.get("markup", "plain") or "plain" for name, entry in sources.items()}


def roles_of(text: str, preset: str) -> list[str]:
    try:
        return mark(text, preset)
    except KeyError:
        return mark(text, "plain")


def cut_model(text: str, roles: list[str]) -> tuple[list[dict], list[dict]]:
    def indices(lo: int, hi: int) -> list[int]:
        return [j for j in range(lo, hi + 1) if j == len(text) or not unicodedata.combining(text[j])]

    cuts, letters = [], []
    for word, (a, b) in enumerate(word_spans(text)):
        inked = [j for j in range(a, b) if roles[j] != NO_INK and not unicodedata.combining(text[j])]
        if not inked:
            continue
        first = len(cuts)
        cuts.append({"i": inked[0], "idx": indices(a, inked[0]), "kind": "start", "w": word, "after": None, "before": inked[0]})
        for previous, current in pairwise(inked):
            cuts.append({"i": current, "idx": indices(previous + 1, current), "kind": "inner", "w": word, "after": previous, "before": current})
        end = inked[-1] + 1
        while end < b and unicodedata.combining(text[end]):
            end += 1
        cuts.append({"i": end, "idx": indices(end, b), "kind": "end", "w": word, "after": inked[-1], "before": None})
        for n, j in enumerate(inked):
            glyph = text[j]
            k = j + 1
            while k < len(text) and unicodedata.combining(text[k]):
                glyph += text[k]
                k += 1
            letters.append({"g": glyph, "j": j, "l": first + n, "r": first + n + 1})
    return cuts, letters


def prefill(cuts: list[dict], placement: Placement | None, width: int, ink: tuple[float, float]) -> tuple[list[float], str]:
    if placement is None or len(placement.chars) != len(placement.text) or not cuts:
        xs = np.linspace(ink[0], ink[1], max(1, len(cuts)))
        return [round(float(x), 2) for x in xs], "even"
    chars = placement.chars
    xs = []
    for cut in cuts:
        if cut["kind"] == "start":
            x = chars[cut["before"]].x0
        elif cut["kind"] == "end":
            x = chars[cut["after"]].x1
        else:
            x = (chars[cut["after"]].x1 + chars[cut["before"]].x0) / 2
        xs.append(min(max(float(x), ink[0]), ink[1]))
    xs = np.maximum.accumulate(np.asarray(xs))
    return [round(float(x), 2) for x in xs], placement.placer or ""


def line_geometry(image, row: dict) -> dict:
    strip = prepare(image, polygon_or_none(row.get("polygon")), baseline_or_none(row.get("baseline")))
    xs = np.linspace(0, image.width, CENTRE_POINTS)
    ys = np.interp(xs, strip.centre_x, strip.centre_y)
    return {"slant": round(float(strip.slant), 4), "centre": [[round(float(x), 1), round(float(y), 1)] for x, y in zip(xs, ys)],
            "band": [round(float((strip.band[0] - strip.v_mid) * strip.scale), 1), round(float((strip.band[1] - strip.v_mid) * strip.scale), 1)],
            "xheight": round(float(strip.core * strip.scale), 2)}


def modal_cli(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONUTF8": "1", "MSYS_NO_PATHCONV": "1"}
    return subprocess.run([sys.executable, "-m", "modal", *args], capture_output=True, text=True, encoding="utf-8", env=env, check=False)


def fetch_placements(spec: str, runs: Path) -> Path:
    folder = runs / placer_label(spec) / SET
    if any(folder.glob("placements-*.jsonl")):
        return folder
    listing = modal_cli("volume", "ls", "--json", "littera", f"/eval_runs/{placer_label(spec)}/{SET}")
    if listing.returncode != 0:
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    for entry in json.loads(listing.stdout):
        name = Path(entry["filename"]).name
        if name.startswith("placements-"):
            modal_cli("volume", "get", "--force", "littera", "/" + entry["filename"], str(folder / name))
    return folder


def read_placements(spec: str, rows: list[dict], runs: Path) -> dict[str, Placement]:
    folder = fetch_placements(spec, runs)
    placements: dict[str, Placement] = {}
    for path in sorted(folder.glob("placements-*.jsonl")):
        placements.update(load_placements(path))
    missing = [r for r in rows if r["uid"] not in placements]
    if missing and not spec.startswith("kraken"):
        placer = make_placer(spec)
        for row in missing:
            image = decode_image(row["image"])
            placements[row["uid"]] = placer.place(image, row["text"], polygon=polygon_or_none(row.get("polygon")),
                                                  baseline=baseline_or_none(row.get("baseline")))
    return placements


def letters_font(texts: list[str]) -> str:
    from fontTools import subset

    options = subset.Options()
    options.flavor = "woff"
    options.layout_features = ["ccmp", "locl", "mark", "mkmk", "kern"]
    font = subset.load_font(str(FONT), options)
    subsetter = subset.Subsetter(options)
    wanted = set("".join(texts)) | set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:-")
    subsetter.populate(text="".join(sorted(wanted)))
    subsetter.subset(font)
    buffer = io.BytesIO()
    font.save(buffer)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return ("@font-face { font-family: \"Junicode Letters\"; font-display: block; "
            f"src: url(data:font/woff;base64,{data}) format(\"woff\"); }}\n")


def image_name(number: int, data: bytes) -> str:
    return f"lines/{number:03d}.{'jpg' if data[:2] == b'\xff\xd8' else 'png'}"


def build(placer: str, out: Path, root: str = "data/eval", runs: Path = ROOT / "runs" / "eval") -> dict:
    rows = sorted(read_set(set_path(SET, root)), key=lambda r: (r["source"], r["uid"]))
    placements = read_placements(placer, rows, runs)
    presets = markup_presets()
    if (out / "lines").exists():
        shutil.rmtree(out / "lines")
    (out / "lines").mkdir(parents=True, exist_ok=True)
    lines, prefill_counts = [], {}
    for number, row in enumerate(rows, start=1):
        image = decode_image(row["image"])
        text = row["text"]
        roles = roles_of(text, presets.get(row["source"], "plain"))
        cuts, letters = cut_model(text, roles)
        xs, used = prefill(cuts, placements.get(row["uid"]), image.width, ink_profile(image, polygon_or_none(row.get("polygon"))).extent())
        prefill_counts[used] = prefill_counts.get(used, 0) + 1
        name = image_name(number, row["image"])
        (out / name).write_bytes(row["image"])
        lines.append({"uid": row["uid"], "source": row["source"], "page": row["page"], "text": text, "image": name,
                      "width": image.width, "height": image.height, "prefill": used, "ink": [r != NO_INK for r in roles],
                      "cuts": [{"i": c["i"], "idx": c["idx"], "kind": c["kind"], "w": c["w"], "x": x} for c, x in zip(cuts, xs)],
                      "letters": letters, **line_geometry(image, row)})
    data = {"set": SET, "placer": placer, "built": datetime.now().isoformat(timespec="seconds"), "lines": lines}
    (out / "data.js").write_text("window.LETTER_GOLD = " + json.dumps(data, ensure_ascii=True, separators=(",", ":")) + ";\n", encoding="utf-8")
    (out / "letters.css").write_text(letters_font([r["text"] for r in rows]), encoding="utf-8")
    shutil.copyfile(PAGE, out / "index.html")
    return {"out": str(out), "lines": len(lines), "cuts": sum(len(line["cuts"]) for line in lines), "prefill": prefill_counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.mark.build")
    parser.add_argument("--placer", default="kraken-mccatmus", help="even, kraken-k2, kraken-mccatmus or littera:<run or model dir>")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default="data/eval")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.placer, Path(args.out), args.root), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
