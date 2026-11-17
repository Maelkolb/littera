from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from littera.eval.sets import set_stats, write_readme, write_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m littera.eval.wordgold")
    parser.add_argument("dataset", choices=["gw20", "iam", "washington", "parzival", "saintgall", "esposalles"])
    parser.add_argument("--input", required=True, help="extracted dataset folder (iam: folder with xml.tar, lines.tar and the task zip)")
    parser.add_argument("--out", default="data/eval")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args(argv)
    source = Path(args.input)
    if args.dataset == "gw20":
        from littera.eval.wordgold import gw20

        rows, readme, extra = gw20.build_rows(source), gw20.README, {}
        name = "gw20-words"
    elif args.dataset == "iam":
        from littera.eval.wordgold import iam

        rows, extra = iam.build_rows(source / "xml.tar", source / "lines.tar", source / "largeWriterIndependentTextLineRecognitionTask.zip",
                                     limit=args.limit)
        readme, name = iam.README, "iam-words"
    elif args.dataset == "esposalles":
        from littera.eval.wordgold import esposalles

        rows, extra = esposalles.build_rows(source)
        readme, name = esposalles.README, "esposalles-words"
    else:
        from littera.eval.wordgold import histdb

        language = {"washington": "en", "parzival": "gmh", "saintgall": "la"}[args.dataset]
        rows, extra = histdb.build_rows(source, args.dataset, language)
        readme, name = histdb.README.format(name=args.dataset), f"{args.dataset}-words"
    write_set(rows, Path(args.out) / name / "lines.parquet")
    stats = {**set_stats(rows), "conversion": extra}
    write_readme(Path(args.out) / name, name, readme, stats)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
