from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

GOOGLE = "https://raw.githubusercontent.com/google/fonts/main"
FONTLIBRARY = "https://fontlibrary.org"
WIEGEL = "http://www.peter-wiegel.de/Fonts"


@dataclass(frozen=True)
class FontSpec:
    name: str
    style: str
    split: str
    licence: str
    url: str
    member: str = ""
    licence_url: str = ""


def google(name: str, style: str, split: str, directory: str, filename: str, licence: str = "OFL-1.1") -> FontSpec:
    licence_file = "LICENSE.txt" if directory.startswith("apache/") else "OFL.txt"
    return FontSpec(name, style, split, licence, f"{GOOGLE}/{directory}/{filename}", "", f"{GOOGLE}/{directory}/{licence_file}")


def fontlibrary(name: str, style: str, split: str, path: str, member: str, licence: str = "OFL-1.1") -> FontSpec:
    return FontSpec(name, style, split, licence, f"{FONTLIBRARY}{path}", member, "")


def wiegel(name: str, style: str, split: str, archive: str, member: str) -> FontSpec:
    return FontSpec(name, style, split, "OFL-1.1", f"{WIEGEL}/{archive}", member, "")


CATALOGUE: list[FontSpec] = [
    google("Pinyon Script", "script", "train", "ofl/pinyonscript", "PinyonScript-Regular.ttf"),
    google("Mrs Saint Delafield", "script", "heldout", "ofl/mrssaintdelafield", "MrsSaintDelafield-Regular.ttf"),
    google("Herr Von Muellerhoff", "script", "train", "ofl/herrvonmuellerhoff", "HerrVonMuellerhoff-Regular.ttf"),
    google("Monsieur La Doulaise", "script", "train", "ofl/monsieurladoulaise", "MonsieurLaDoulaise-Regular.ttf"),
    google("Italianno", "script", "train", "ofl/italianno", "Italianno-Regular.ttf"),
    google("Tangerine", "script", "heldout", "ofl/tangerine", "Tangerine-Regular.ttf"),
    google("Great Vibes", "script", "train", "ofl/greatvibes", "GreatVibes-Regular.ttf"),
    google("Petit Formal Script", "script", "train", "ofl/petitformalscript", "PetitFormalScript-Regular.ttf"),
    google("Alex Brush", "script", "train", "ofl/alexbrush", "AlexBrush-Regular.ttf"),
    google("Allura", "script", "heldout", "ofl/allura", "Allura-Regular.ttf"),
    google("Rouge Script", "script", "train", "ofl/rougescript", "RougeScript-Regular.ttf"),
    google("Mr De Haviland", "script", "train", "ofl/mrdehaviland", "MrDeHaviland-Regular.ttf"),
    google("Qwigley", "script", "train", "ofl/qwigley", "Qwigley-Regular.ttf"),
    google("Meddon", "script", "heldout", "ofl/meddon", "Meddon.ttf"),
    google("Euphoria Script", "script", "train", "ofl/euphoriascript", "EuphoriaScript-Regular.ttf"),
    google("Corinthia", "script", "train", "ofl/corinthia", "Corinthia-Regular.ttf"),
    google("Lovers Quarrel", "script", "train", "ofl/loversquarrel", "LoversQuarrel-Regular.ttf"),
    google("Ephesis", "script", "heldout", "ofl/ephesis", "Ephesis-Regular.ttf"),
    google("Imperial Script", "script", "train", "ofl/imperialscript", "ImperialScript-Regular.ttf"),
    google("Ms Madi", "script", "train", "ofl/msmadi", "MsMadi-Regular.ttf"),
    google("Sacramento", "script", "train", "ofl/sacramento", "Sacramento-Regular.ttf"),
    google("Lavishly Yours", "script", "heldout", "ofl/lavishlyyours", "LavishlyYours-Regular.ttf"),
    google("Dancing Script", "script", "train", "ofl/dancingscript", "DancingScript[wght].ttf"),
    google("Marck Script", "script", "train", "ofl/marckscript", "MarckScript-Regular.ttf"),
    google("Homemade Apple", "hand", "train", "apache/homemadeapple", "HomemadeApple-Regular.ttf", "Apache-2.0"),
    google("Cedarville Cursive", "hand", "train", "ofl/cedarvillecursive", "Cedarville-Cursive.ttf"),
    google("La Belle Aurore", "hand", "heldout", "ofl/labelleaurore", "LaBelleAurore.ttf"),
    google("Kristi", "hand", "train", "ofl/kristi", "Kristi-Regular.ttf"),
    google("Zeyada", "hand", "train", "ofl/zeyada", "Zeyada.ttf"),
    google("Dawning of a New Day", "hand", "heldout", "ofl/dawningofanewday", "DawningofaNewDay.ttf"),
    google("Mrs Sheppards", "hand", "train", "ofl/mrssheppards", "MrsSheppards-Regular.ttf"),
    google("Ruthie", "hand", "train", "ofl/ruthie", "Ruthie-Regular.ttf"),
    google("Nothing You Could Do", "hand", "train", "ofl/nothingyoucoulddo", "NothingYouCouldDo.ttf"),
    google("Reenie Beanie", "hand", "heldout", "ofl/reeniebeanie", "ReenieBeanie.ttf"),
    google("Caveat", "hand", "train", "ofl/caveat", "Caveat[wght].ttf"),
    google("Kaushan Script", "hand", "train", "ofl/kaushanscript", "KaushanScript-Regular.ttf"),
    google("Charm", "hand", "train", "ofl/charm", "Charm-Regular.ttf"),
    wiegel("18th Century Kurrent", "kurrent", "train", "18thCentury.zip", "18thCtrKurTxt.ttf"),
    wiegel("Greifswalder Deutsche Schrift", "kurrent", "heldout", "greifswalderd.zip", "Greifswalder Deutsche.ttf"),
    wiegel("Kanzleyrath", "kurrent", "train", "Kanzleyrath.zip", "Kanzleyrath.ttf"),
    wiegel("Kanzler", "kurrent", "heldout", "Kanzler.zip", "Kanzler.ttf"),
    google("UnifrakturMaguntia", "fraktur", "train", "ofl/unifrakturmaguntia", "UnifrakturMaguntia-Book.ttf"),
    google("UnifrakturCook", "fraktur", "heldout", "ofl/unifrakturcook", "UnifrakturCook-Bold.ttf"),
    google("Pirata One", "fraktur", "train", "ofl/pirataone", "PirataOne-Regular.ttf"),
    google("Grenze Gotisch", "fraktur", "train", "ofl/grenzegotisch", "GrenzeGotisch[wght].ttf"),
    google("Texturina", "fraktur", "train", "ofl/texturina", "Texturina[opsz,wght].ttf"),
    google("Fruktur", "fraktur", "heldout", "ofl/fruktur", "Fruktur-Regular.ttf"),
    google("New Rocker", "fraktur", "train", "ofl/newrocker", "NewRocker-Regular.ttf"),
    fontlibrary("Grobe Deutschmeister", "fraktur", "train",
                "/assets/downloads/grobe-deutschmeister/748dcc62617890a56fe54fd7f872598e/grobe-deutschmeister.zip",
                "grobe-deutschmeister"),
    fontlibrary("KJV 1611", "fraktur", "heldout", "/assets/downloads/kjv-1611/361c6269b9bad57bb03a7513af5678b1/kjv-1611.zip", "KJV1611"),
    fontlibrary("Gamaliel", "fraktur", "train", "/assets/downloads/gamaliel/e0c9601992c81cd08b8d449411bc0ac6/gamaliel.zip", "Gamaliel"),
    fontlibrary("Blackletter Hand", "kurrent", "train",
                "/assets/downloads/blackletter-hand/fd0f5d97dfa7a8630e390317f058fdf8/blackletter-hand.zip",
                "BlackletterHand"),
    google("Uncial Antiqua", "medieval", "train", "ofl/uncialantiqua", "UncialAntiqua-Regular.ttf"),
    google("MedievalSharp", "medieval", "heldout", "ofl/medievalsharp", "MedievalSharp.ttf"),
    google("Almendra", "medieval", "train", "ofl/almendra", "Almendra-Regular.ttf"),
    google("Fondamento", "medieval", "train", "ofl/fondamento", "Fondamento-Regular.ttf"),
    google("IM Fell DW Pica", "print", "train", "ofl/imfelldwpica", "IMFePIrm28P.ttf"),
    google("IM Fell English", "print", "train", "ofl/imfellenglish", "IMFeENrm28P.ttf"),
    google("IM Fell French Canon", "print", "heldout", "ofl/imfellfrenchcanon", "IMFeFCrm28P.ttf"),
    google("IM Fell Great Primer", "print", "train", "ofl/imfellgreatprimer", "IMFeGPrm28P.ttf"),
    google("IM Fell Double Pica Italic", "print", "train", "ofl/imfelldoublepica", "IMFELLDoublePica-Italic.ttf"),
    google("EB Garamond", "print", "train", "ofl/ebgaramond", "EBGaramond[wght].ttf"),
    google("Old Standard TT", "print", "heldout", "ofl/oldstandardtt", "OldStandard-Regular.ttf"),
    google("Libre Caslon Text", "print", "train", "ofl/librecaslontext", "LibreCaslonText[wght].ttf"),
    google("Sorts Mill Goudy", "print", "train", "ofl/sortsmillgoudy", "SortsMillGoudy-Regular.ttf"),
    google("Cormorant Garamond", "print", "heldout", "ofl/cormorantgaramond", "CormorantGaramond[wght].ttf"),
    google("Rosarivo", "print", "train", "ofl/rosarivo", "Rosarivo-Regular.ttf"),
    google("Goudy Bookletter 1911", "print", "train", "ofl/goudybookletter1911", "GoudyBookletter1911.ttf"),
    google("Cardo", "print", "heldout", "ofl/cardo", "Cardo-Regular.ttf"),
    FontSpec("Junicode Two Beta", "print", "train", "OFL-1.1",
             "https://raw.githubusercontent.com/psb1558/Junicode-font/HEAD/webfiles/JunicodeTwoBeta-Regular.ttf", "",
             "https://raw.githubusercontent.com/psb1558/Junicode-font/HEAD/OFL.txt"),
]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "littera-font-fetch"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def pick_member(archive: zipfile.ZipFile, member: str) -> tuple[str, bytes]:
    names = [n for n in archive.namelist() if n.lower().endswith((".ttf", ".otf"))]
    exact = [n for n in names if Path(n).name == member]
    stem = [n for n in names if Path(n).stem == member or Path(n).stem.lower() == member.lower()]
    choice = (exact or stem or [n for n in names if member.lower() in n.lower()] or names)[0]
    return Path(choice).name, archive.read(choice)


def licence_text(archive: zipfile.ZipFile) -> bytes:
    for name in archive.namelist():
        base = Path(name).name.lower()
        if base.endswith(".txt") and ("ofl" in base or "licen" in base) and "faq" not in base:
            return archive.read(name)
    return b""


def embedded_licence(font: bytes) -> bytes:
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return b""
    names = TTFont(io.BytesIO(font), lazy=True)["name"]
    parts = [n.toUnicode() for n in names.names if n.nameID in (0, 13, 14) and n.platformID == 3]
    return "\n\n".join(dict.fromkeys(parts)).encode("utf-8")


def fetch(spec: FontSpec, destination: Path) -> dict:
    folder = destination / slug(spec.name)
    folder.mkdir(parents=True, exist_ok=True)
    payload = download(spec.url)
    if spec.url.endswith(".zip"):
        archive = zipfile.ZipFile(io.BytesIO(payload))
        filename, font = pick_member(archive, spec.member)
        licence = licence_text(archive) or embedded_licence(font)
    else:
        filename, font = Path(spec.url).name, payload
        licence = download(spec.licence_url) if spec.licence_url else b""
    (folder / filename).write_bytes(font)
    if licence:
        (folder / "LICENCE.txt").write_bytes(licence)
    return {**asdict(spec), "file": f"{slug(spec.name)}/{filename}", "sha256": hashlib.sha256(font).hexdigest(),
            "licence_file": f"{slug(spec.name)}/LICENCE.txt" if licence else ""}


def fetch_all(destination: str | Path, names: list[str] | None = None) -> list[dict]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for spec in CATALOGUE:
        if names and spec.name not in names:
            continue
        try:
            records.append(fetch(spec, destination))
            print(f"ok   {spec.name}")
        except Exception as error:  # noqa: BLE001
            failures.append({"name": spec.name, "error": repr(error)})
            print(f"FAIL {spec.name}: {error!r}")
    (destination / "fonts.json").write_text(json.dumps({"fonts": records, "failures": failures}, indent=1, ensure_ascii=False),
                                            encoding="utf-8")
    return records


def load_catalogue(directory: str | Path) -> list[dict]:
    directory = Path(directory)
    data = json.loads((directory / "fonts.json").read_text(encoding="utf-8"))
    return [{**record, "path": str(directory / record["file"])} for record in data["fonts"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m littera.synth.fonts")
    parser.add_argument("command", choices=["fetch", "list"])
    parser.add_argument("--dest", default="data/fonts")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch_all(args.dest)
    else:
        for spec in CATALOGUE:
            print(f"{spec.split:8} {spec.style:9} {spec.licence:11} {spec.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
