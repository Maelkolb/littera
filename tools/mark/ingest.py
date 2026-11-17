from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from littera.eval.sets import read_set, set_path, set_stats, write_readme, write_set

SCHEMA = "littera/letter-marks/1"
SOURCE_SET = "letter-gold-candidates"
OUT_SET = "letter-gold"

README = """Letter boundaries marked by hand by the owner on the marking page (`tools/mark`, prefilled from a placer, every cut moved or
confirmed by eye) over the lines of `letter-gold-candidates`. Only lines marked "done" are here; lines marked "text does not
match" or "cannot tell" are counted in the stats and left out. `gold_cuts`: the boundary before `text[i]` at the line's mid
x-height in crop pixels (`x`), one entry per text index; a cut that stands for several indices (letters without ink by the
source's `littera.text` preset) repeats its `x`, and cuts the owner could not place are left out. `origin` keeps the candidate's
origin and the marking record (slant used for the cut lines, prefill placer, cuts moved and visited, seconds). `marks.json` is
the page's export as imported. Licences: those of the pool rows (`abp`, `laubmann` internal; evaluation only)."""


def gold_rows(marks: dict, candidates: dict[str, dict]) -> tuple[list[dict], Counter]:
    rows, counts = [], Counter()
    for line in marks["lines"]:
        counts[line["status"]] += 1
        if line["status"] != "done":
            continue
        base = candidates.get(line["uid"])
        if base is None or base["text"] != line["text"]:
            counts["text_changed_or_missing"] += 1
            continue
        gold = dict(sorted({int(c["i"]): float(c["x"]) for c in line["gold_cuts"]}.items()))
        try:
            candidate_origin = json.loads(base["origin"])
        except (TypeError, ValueError):
            candidate_origin = base["origin"]
        record = {k: line.get(k) for k in ("slant", "prefill", "moved", "visited", "cuts_total", "seconds", "skipped")}
        origin = {"candidate": candidate_origin, "marked": {**record, "exported": marks.get("exported"), "placer": marks.get("placer")}}
        rows.append({**base, "kind": "letter", "gold_cuts": [{"i": i, "x": x} for i, x in gold.items()],
                     "origin": json.dumps(origin, ensure_ascii=False)})
        counts["cuts"] += len(gold)
    return rows, counts


def upload(folder: Path) -> int:
    env = {**os.environ, "PYTHONUTF8": "1", "MSYS_NO_PATHCONV": "1"}
    return subprocess.run([sys.executable, "-m", "modal", "volume", "put", "--force", "littera", str(folder), f"/eval/{OUT_SET}"],
                          env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.mark.ingest")
    parser.add_argument("marks", help="the JSON file exported by the marking page")
    parser.add_argument("--root", default="data/eval")
    parser.add_argument("--no-upload", action="store_true", help="write data/eval/letter-gold only, not /eval/letter-gold on the volume")
    args = parser.parse_args(argv)
    marks = json.loads(Path(args.marks).read_text(encoding="utf-8"))
    if marks.get("schema") != SCHEMA or marks.get("set") != SOURCE_SET:
        raise SystemExit(f"{args.marks} is not an export of the letter marking page")
    candidates = {row["uid"]: row for row in read_set(set_path(SOURCE_SET, args.root))}
    rows, counts = gold_rows(marks, candidates)
    folder = Path(args.root) / OUT_SET
    write_set(rows, folder / "lines.parquet")
    shutil.copyfile(args.marks, folder / "marks.json")
    stats = {**set_stats(rows), "statuses": dict(counts), "exported": marks.get("exported"), "prefill": marks.get("placer")}
    write_readme(folder, OUT_SET, README, stats)
    print(json.dumps({"lines": len(rows), **dict(counts), "folder": str(folder)}, indent=1))
    if not args.no_upload:
        return upload(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
