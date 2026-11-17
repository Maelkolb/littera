"""Pilot jobs (app littera-eval, CPU): fixed training draw, crops for markup checks, evaluation of pilot runs, read rate.

    uv run modal run modal/pilot.py::draw                       # /vol/pilot/v1: 50k train, val, test rows from /pool/v1
    uv run modal run modal/pilot.py::crops --uids-file u.json --out c.parquet
    uv run modal run modal/pilot.py::evaluate --runs pilot-k2,pilot-seg --sets gw20-words,...
    uv run modal run modal/pilot.py::read_rate --runs pilot-k2,pilot-seg
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

app = modal.App("littera-eval")
volume = modal.Volume.from_name("littera", create_if_missing=True)
VOL = Path("/vol")
POOL = VOL / "pool" / "v1"
PILOT = VOL / "pilot" / "v1"
EVAL = VOL / "eval"
RUNS = VOL / "eval_runs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy==2.5.4", "pillow==13.0.0", "opencv-python-headless==5.0.0.93", "pyarrow==26.0.1", "lxml==6.1.3",
                 "safetensors==0.8.0")
    .env({"PYTHONUTF8": "1", "OMP_NUM_THREADS": "1"})
    .add_local_python_source("littera")
)

TRAIN_LINES = 50_000
VAL_LINES = 1_000
TEST_PER_SOURCE = 30
SEED = 20261110
EVAL_CONTAINERS = int(os.environ.get("LITTERA_EVAL_CONTAINERS", "6"))


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=4096, timeout=1800)
def fetch_rows(uids: list[str]) -> list[dict]:
    from littera.data import read_rows

    return [{"uid": r["uid"], "source": r["source"], "text": r["text"], "image": r["image"]} for r in read_rows(uids, POOL)]


@app.local_entrypoint()
def crops(uids_file: str, out: str):
    import pyarrow as pa
    import pyarrow.parquet as pq

    uids = json.loads(Path(uids_file).read_text(encoding="utf-8"))
    rows = fetch_rows.remote(uids)
    pq.write_table(pa.Table.from_pylist(rows), out)
    print(f"{len(rows)} rows -> {out}")


def plan_draw(index_rows: list[dict]) -> dict[str, list[str]]:
    import random
    from collections import defaultdict

    rng = random.Random(SEED)
    by_split_source: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in index_rows:
        by_split_source[(row["split"], row["source"])].append(row["uid"])
    picks: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split, total in (("train", TRAIN_LINES), ("val", VAL_LINES)):
        groups = {s: sorted(u) for (sp, s), u in by_split_source.items() if sp == split}
        available = sum(len(u) for u in groups.values())
        for source, uids in sorted(groups.items()):
            share = round(total * len(uids) / available)
            picks[split] += rng.sample(uids, min(share, len(uids)))
    for (split, source), uids in sorted(by_split_source.items()):
        if split == "test":
            picks["test"] += rng.sample(sorted(uids), min(TEST_PER_SOURCE, len(uids)))
    return picks


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=8192, timeout=1800, max_containers=6)
def write_pilot_part(part: str, rows: list[int], name: str) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(POOL / part).take(pa.array(rows))
    out = PILOT / "lines" / "pool" / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, row_group_size=512, compression="zstd")
    volume.commit()
    return table.num_rows


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=4096, timeout=1800)
def plan_job() -> dict:
    import pyarrow.parquet as pq

    index = pq.read_table(POOL / "index.parquet", columns=["uid", "source", "split", "part", "part_row", "open_licence"])
    rows = index.to_pylist()
    picks = plan_draw([r for r in rows if r["split"] in ("train", "val", "test") and r["open_licence"]])
    chosen = {uid for uids in picks.values() for uid in uids}
    by_part: dict[str, list[int]] = {}
    for row in rows:
        if row["uid"] in chosen:
            by_part.setdefault(row["part"], []).append(row["part_row"])
    PILOT.mkdir(parents=True, exist_ok=True)
    (PILOT / "draw.json").write_text(json.dumps({"seed": SEED, "index_rows": len(rows), **picks}), encoding="utf-8")
    volume.commit()
    return {"parts": {part: sorted(r) for part, r in by_part.items()}, "counts": {k: len(v) for k, v in picks.items()}}


@app.local_entrypoint()
def draw():
    plan = plan_job.remote()
    print(json.dumps(plan["counts"]))
    jobs = [(part, rows, part.replace("lines/", "").replace("/", "-").removesuffix(".parquet")) for part, rows in sorted(plan["parts"].items())]
    total = sum(write_pilot_part.starmap(jobs))
    print(f"wrote {total} rows in {len(jobs)} parts under {PILOT}/lines/pool")


SET_PRESETS = {"synth-heldfonts": "plain"}
ALIGNERS: dict[tuple[str, str], object] = {}


def run_dir(run: str) -> str:
    return run if run.startswith("/") else f"/vol/runs/{run}"


def spec_of(run: str) -> str:
    name, variant = split_variant(run)
    return f"littera:{run_dir(name)}" + (f"@{variant}" if variant else "")


def split_variant(run: str) -> tuple[str, str]:
    name, _, variant = run.partition("@")
    return name, variant


def aligner_for(run: str, preset: str | None):
    import torch

    from littera.place import Aligner

    torch.set_num_threads(1)
    key = (run, preset or "")
    if key not in ALIGNERS:
        name, variant = split_variant(run)
        aligner = Aligner.load(run_dir(name), device="cpu")
        if preset:
            aligner.preset = preset
        if variant == "nosnap":
            aligner.snap = False
        elif variant == "gany":
            aligner.decode.garbage = "any"
        elif variant.startswith("dps"):
            aligner.objective.decode_prior_scale = float(variant[3:])
        ALIGNERS[key] = aligner
    return ALIGNERS[key]


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=2048, timeout=3600, max_containers=EVAL_CONTAINERS)
def place_shard(run: str, set_name: str, shard: int, shards: int) -> dict:
    from littera.eval.run import run as eval_run

    volume.reload()
    placer = aligner_for(run, SET_PRESETS.get(set_name))
    result = eval_run(spec_of(run), set_name, str(EVAL), str(RUNS), shard, shards, placer=placer)
    volume.commit()
    return {"run": run, "set": set_name, "shard": shard, **result}


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=4096, timeout=1800, max_containers=EVAL_CONTAINERS)
def merge_set(run: str, set_name: str, shards: int) -> dict:
    from littera.eval.placers import placer_label
    from littera.eval.run import merge_scores

    volume.reload()
    spec = spec_of(run)
    result = merge_scores(RUNS / placer_label(spec) / set_name, {"placer": spec, "set": set_name, "shards": shards}, shards)
    volume.commit()
    return result


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=2048, timeout=600)
def finished_shards(jobs: list[tuple[str, str, int, int]], sizes: dict) -> list[list]:
    from littera.eval.placers import placer_label

    volume.reload()
    done = []
    for run, set_name, shard, shards in jobs:
        path = RUNS / placer_label(spec_of(run)) / set_name / f"placements-{shard:04d}.jsonl"
        expected = len(range(shard, sizes[set_name], shards))
        if path.exists() and sum(1 for _ in path.open(encoding="utf-8")) == expected:
            done.append([run, set_name, shard, shards])
    return done


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=2048, timeout=600)
def set_sizes(set_names: list[str]) -> dict:
    import pyarrow.parquet as pq

    return {name: pq.ParquetFile(EVAL / name / "lines.parquet").metadata.num_rows for name in set_names}


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=3072, timeout=3600, max_containers=EVAL_CONTAINERS)
def read_rate_shard(run: str, shard: int, shards: int) -> dict:
    import io
    import time

    import pyarrow.parquet as pq
    from PIL import Image

    from littera.geometry import prepare
    from littera.train import edit_distance

    volume.reload()
    aligner = aligner_for(run, None)
    name, _ = split_variant(run)
    presets = json.loads((Path(run_dir(name)) / "best" / "config.json").read_text(encoding="utf-8")).get("presets", {})
    table = pq.read_table(PILOT / "lines", filters=[("split", "=", "test")], columns=["uid", "source", "text", "image", "polygon", "baseline"])
    rows = [r for k, r in enumerate(table.to_pylist()) if k % shards == shard]
    records, started = [], time.time()
    for row in rows:
        image = Image.open(io.BytesIO(row["image"])).convert("L")
        baseline = [tuple(p) for p in row["baseline"]] if row["baseline"] and len(row["baseline"]) >= 2 else None
        strip = prepare(image, None, baseline, aligner.geometry)
        _, scores, boundary, lengths = aligner.run(strip)
        read_labels = [label for word in aligner._best_reading(scores, boundary, lengths) for label in word]
        encoded = aligner.codec.encode(row["text"], presets.get(row["source"], aligner.preset))
        truth_labels = [label for word in encoded.graph_words() for label, _ in word]
        records.append({"uid": row["uid"], "source": row["source"], "units": len(truth_labels),
                        "errors": edit_distance(read_labels, truth_labels), "reading": aligner.codec.decode([read_labels])})
    out = Path(run_dir(name)) / "read_rate" / f"test-{shard:04d}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    volume.commit()
    errors, units = sum(r["errors"] for r in records), sum(r["units"] for r in records)
    return {"run": run, "shard": shard, "lines": len(records), "errors": errors, "units": units, "seconds": round(time.time() - started, 1)}


@app.local_entrypoint()
def evaluate(runs: str, sets: str = "gw20-words,iam-words,synth-heldfonts,mismatch-heldout", rows_per_shard: int = 400,
             read_shards: int = 2, out: str = "runs/eval/pilot.json", merge_only: bool = False):
    run_list, set_list = runs.split(","), [s for s in sets.split(",") if s]
    sizes = set_sizes.remote(set_list) if set_list else {}
    jobs = [(run, name, shard, max(1, -(-sizes[name] // rows_per_shard))) for run in run_list for name in set_list
            for shard in range(max(1, -(-sizes[name] // rows_per_shard)))]
    done = {tuple(job) for job in finished_shards.remote(jobs, sizes)} if jobs else set()
    jobs = [job for job in jobs if job not in done]
    print(f"{len(jobs)} placement shards to run ({len(done)} already done), {len(run_list) * read_shards} read-rate shards", flush=True)
    read_results = []
    for result in place_shard.starmap([] if merge_only else jobs, order_outputs=False):
        print(json.dumps(result), flush=True)
    for result in read_rate_shard.starmap([(run, k, read_shards) for run in run_list for k in range(read_shards)], order_outputs=False):
        print(json.dumps(result))
        read_results.append(result)
    pairs = sorted({(run, name, n) for run, name, _, n in jobs} | {(run, name, n) for run, name, _, n in done})
    merged = dict(zip([f"{run}|{name}" for run, name, _ in pairs], merge_set.starmap(pairs)))
    for run in run_list if read_shards else []:
        parts = [r for r in read_results if r["run"] == run]
        errors, units = sum(r["errors"] for r in parts), sum(r["units"] for r in parts)
        merged[f"{run}|test-read"] = {"lines": sum(r["lines"] for r in parts), "units": units, "cer": errors / max(1, units),
                                      "read_rate": 1 - errors / max(1, units)}
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(merged)
    path.write_text(json.dumps(existing, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=1024, timeout=600)
def copy_best_job(run: str, decode: dict) -> dict:
    import shutil

    volume.reload()
    source = Path(run_dir(run)) / "best"
    target = VOL / "models" / "pilot-best"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    config_path = target / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["decode"] = {**config["decode"], **decode}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    (target / "SOURCE.txt").write_text(f"{source}\ndecode changed: {json.dumps(decode)}\n", encoding="utf-8")
    volume.commit()
    return {"from": str(source), "to": str(target), "files": sorted(p.name for p in target.iterdir()), "decode": config["decode"]}


@app.local_entrypoint()
def copy_best(run: str, decode: str = ""):
    print(json.dumps(copy_best_job.remote(run, json.loads(decode) if decode else {})))
