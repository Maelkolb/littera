from __future__ import annotations

import argparse
import json
import sys

from PIL import Image


def _points(value: str | None):
    return [tuple(p) for p in json.loads(value)] if value else None


def _write(data) -> None:
    sys.stdout.buffer.write(json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "train":
        from littera.train import main as train_main

        train_main(argv[1:])
        return
    parser = argparse.ArgumentParser(prog="littera")
    commands = parser.add_subparsers(dest="command", required=True)
    place = commands.add_parser("place", help="place a transcription on a line image, JSON on stdout")
    place.add_argument("image")
    place.add_argument("text")
    place.add_argument("--model", required=True)
    place.add_argument("--polygon", help="JSON list of [x, y] in image px")
    place.add_argument("--baseline", help="JSON list of [x, y] in image px")
    place.add_argument("--preset")
    place.add_argument("--snap", choices=["strip", "upright", "off"], default="strip",
                       help="snap word edges into ink gaps of the deslanted strip (default), of the upright line, or not at all")
    place.add_argument("--no-snap", action="store_true", help="same as --snap off")
    place.add_argument("--device", default="cpu")
    page = commands.add_parser("page", help="place every TextLine of a PAGE-XML file, write Word and Glyph elements")
    page.add_argument("xml")
    page.add_argument("--image", required=True)
    page.add_argument("--model", required=True)
    page.add_argument("-o", "--out", required=True)
    page.add_argument("--margin", type=int, default=8)
    page.add_argument("--preset")
    page.add_argument("--snap", choices=["strip", "upright", "off"], default="strip")
    page.add_argument("--device", default="cpu")
    commands.add_parser("train", help="train a model (littera train --help)")
    args = parser.parse_args(argv)

    from littera.place import Aligner

    overrides = {"device": args.device}
    if args.preset:
        overrides["preset"] = args.preset
    if args.command == "place":
        aligner = Aligner.load(args.model, snap="off" if args.no_snap else args.snap, **overrides)
        placement = aligner.place(Image.open(args.image), args.text, _points(args.polygon), _points(args.baseline))
        _write(placement.to_dict())
    elif args.command == "page":
        from littera.pagexml import place_page

        aligner = Aligner.load(args.model, snap=args.snap, **overrides)
        _write(place_page(args.xml, args.image, aligner, args.out, args.margin))


if __name__ == "__main__":
    main()
