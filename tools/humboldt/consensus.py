"""Blends littera's placement files of the Humboldt's Hand lookup with aligner-k2's unsnapped cuts (`littera.consensus`).

    uv run python -m tools.humboldt.consensus --own runs/index/ft2-keep --other runs/index/k2raw --out runs/index/ft2-keep-consensus
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from littera.consensus import blend_edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    count = Counter()
    for f in sorted(Path(args.own).glob("*/*.json")):
        own = json.loads(f.read_text(encoding="utf-8"))
        other_file = Path(args.other) / f.parent.name / f.name
        other = json.loads(other_file.read_text(encoding="utf-8")) if other_file.exists() else {}
        for lid, r in own.items():
            count["lines"] += 1
            o = other.get(lid)
            if o is None or o["t"] != r["t"] or o["x0"] != r["x0"] or len(o["cuts"]) != len(r["cuts"]):
                count["lines without the other placement"] += 1
                continue
            cuts, blended, eligible = blend_edges(r["t"], r["cuts"], o["cuts"])
            r["cuts"] = [round(c, 1) for c in cuts]
            count["words blended"] += blended
            count["words with inner edges"] += eligible
        dst = Path(args.out) / f.parent.name / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(own, ensure_ascii=False), encoding="utf-8")
    print(dict(count))


if __name__ == "__main__":
    main()
