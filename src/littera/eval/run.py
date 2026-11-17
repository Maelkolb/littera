from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from littera.eval.ink import ink_profile
from littera.eval.metrics import aggregate, row_scores
from littera.eval.placers import make_placer, placer_label
from littera.eval.sets import decode_image, iter_set, set_path
from littera.types import Placement


def polygon_or_none(points):
    return [tuple(p) for p in points] if points is not None and len(points) >= 3 else None


def baseline_or_none(points):
    return [tuple(p) for p in points] if points is not None and len(points) >= 2 else None


def place_and_score(placer, rows, placements_file: Path, scores_file: Path, shard: int = 0, shards: int = 1,
                    limit: int | None = None) -> dict:
    placements_file.parent.mkdir(parents=True, exist_ok=True)
    done, errors, seconds = 0, 0, 0.0
    with placements_file.open("w", encoding="utf-8") as placements_out, scores_file.open("w", encoding="utf-8") as scores_out:
        for index, row in enumerate(rows):
            if index % shards != shard:
                continue
            if limit is not None and done + errors >= limit:
                break
            image = decode_image(row["image"])
            polygon, baseline = polygon_or_none(row.get("polygon")), baseline_or_none(row.get("baseline"))
            started = time.perf_counter()
            try:
                placement = placer.place(image, row["text"], polygon=polygon, baseline=baseline)
            except Exception as error:  # noqa: BLE001
                errors += 1
                placements_out.write(json.dumps({"uid": row["uid"], "error": repr(error)}, ensure_ascii=False) + "\n")
                continue
            elapsed = time.perf_counter() - started
            seconds += elapsed
            done += 1
            profile = ink_profile(image, polygon)
            scores = row_scores(placement, row, profile)
            placements_out.write(json.dumps({"uid": row["uid"], "seconds": round(elapsed, 4), "placement": placement.to_dict()},
                                            ensure_ascii=False) + "\n")
            scores_out.write(json.dumps(scores, ensure_ascii=False) + "\n")
    return {"placed": done, "errors": errors, "seconds": round(seconds, 2)}


def shard_files(directory: Path, prefix: str, shards: int | None) -> list[Path]:
    if shards is None:
        return sorted(directory.glob(f"{prefix}-*.jsonl"))
    return [directory / f"{prefix}-{k:04d}.jsonl" for k in range(shards) if (directory / f"{prefix}-{k:04d}.jsonl").exists()]


def merge_scores(directory: Path, extra: dict | None = None, shards: int | None = None) -> dict:
    row_level = []
    for path in shard_files(directory, "rowscores", shards):
        row_level += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = 0
    for path in shard_files(directory, "placements", shards):
        errors += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if '"error"' in line[:200])
    result = {"errors": errors, **aggregate(row_level), **(extra or {})}
    (directory / "scores.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def rescore(set_name: str, root: str, directory: Path) -> dict:
    files = sorted(directory.glob("placements-*.jsonl"))
    placements = {}
    for path in files:
        shard = path.stem.split("-")[-1]
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "placement" in record:
                placements[record["uid"]] = (shard, Placement.from_dict(record["placement"]))
    outputs: dict[str, list[str]] = {path.stem.split("-")[-1]: [] for path in files}
    for row in iter_set(set_path(set_name, root)):
        if row["uid"] not in placements:
            continue
        shard, placement = placements[row["uid"]]
        polygon = polygon_or_none(row.get("polygon"))
        scores = row_scores(placement, row, ink_profile(decode_image(row["image"]), polygon))
        outputs[shard].append(json.dumps(scores, ensure_ascii=False))
    for shard, lines in outputs.items():
        (directory / f"rowscores-{shard}.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return {"rows": sum(len(v) for v in outputs.values()), "shards": len(outputs)}


def run(placer_spec: str, set_name: str, root: str, out: str, shard: int = 0, shards: int = 1, limit: int | None = None,
        placer=None) -> dict:
    path = set_path(set_name, root)
    directory = Path(out) / placer_label(placer_spec) / path.parent.name
    placer = placer or make_placer(placer_spec)
    stats = place_and_score(placer, iter_set(path), directory / f"placements-{shard:04d}.jsonl",
                            directory / f"rowscores-{shard:04d}.jsonl", shard, shards, limit)
    return {"directory": str(directory), **stats}


def load_placements(path: str | Path) -> dict[str, Placement]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "placement" in record:
            out[record["uid"]] = Placement.from_dict(record["placement"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m littera.eval.run")
    parser.add_argument("command", choices=["place", "rescore", "score", "all"])
    parser.add_argument("--placer", default="even")
    parser.add_argument("--set", required=True, help="set name under --root, a set directory, or a lines.parquet path")
    parser.add_argument("--root", default="data/eval")
    parser.add_argument("--out", default="runs/eval")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--merge-shards", type=int, default=None, help="score only shards 0..n-1 (default: every shard file)")
    args = parser.parse_args(argv)
    directory = Path(args.out) / placer_label(args.placer) / set_path(args.set, args.root).parent.name
    if args.command in ("place", "all"):
        print(json.dumps(run(args.placer, args.set, args.root, args.out, args.shard, args.shards, args.limit)))
    if args.command == "rescore":
        print(json.dumps(rescore(args.set, args.root, directory)))
    if args.command in ("rescore", "score", "all"):
        print(json.dumps(merge_scores(directory, {"placer": args.placer, "set": args.set}, args.merge_shards), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
