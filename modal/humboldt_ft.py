"""Round 3 fine-tunes on Humboldt's hand (app littera-humboldt-ft).

    uv run modal run modal/humboldt_ft.py::inspect                              # pool rows, baselines, synthetic parts
    uv run modal run modal/humboldt_ft.py::anchors --lift-frame -3 --lift-enter -4    # /vol/pool/humboldt_ft_anchors + sheets
    uv run modal run --detach modal/humboldt_ft.py::main --runs r3-control,r3-lifts
    uv run modal run modal/humboldt_ft.py::status
    uv run modal run modal/humboldt_ft.py::throughput --model /vol/runs/r3-lifts/last
    uv run modal run modal/humboldt_ft.py::release --run r3-lifts --decode '{"inner_edges": "keep"}'    # or --run /vol/models/<name>
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("littera-humboldt-ft")
volume = modal.Volume.from_name("littera", create_if_missing=True)
VOL = "/vol"
POOL = "/vol/pool/humboldt_ft"
ANCHORS = "/vol/pool/humboldt_ft_anchors"
FT2 = "/vol/models/littera-humboldt-ft2"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy==2.5.4", "pillow==13.0.0", "opencv-python-headless==5.0.0.93", "pyarrow==26.0.1", "lxml==6.1.3",
                 "safetensors==0.8.0")
    .env({"PYTHONUTF8": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("littera")
)

BASE = {"pool": [POOL], "init": FT2, "presets": {"humboldt_ft": "markup", "synth": "plain"}, "steps": 1500, "warmup": 100,
        "lr": 2e-4, "val_every": 500, "val_lines": 992, "batch_columns": 48000, "max_batch": 64, "save_every": 600.0}
LIFTS = {"graph": {"lifts": True}, "decode": {"lifts": True, "lift_snap": "lifts", "inner_edges": "keep"}}
SYNTH = [f"/vol/synth/v1/lines/synth/part-{k:05d}.parquet" for k in range(2)]
RUNS = {
    "r3-control": {},
    "r3-lifts": LIFTS,
    "r3-lifts-anchors": {**LIFTS, "anchors": ANCHORS},
    "r3-lifts-anchors-synth": {**LIFTS, "anchors": ANCHORS, "lift_cuts": True, "pool": [POOL] + SYNTH},
    "r3-local-lifts-anchors": {**LIFTS, "anchors": ANCHORS, "init": "/vol/runs/pilot-seg-local", "steps": 3000, "warmup": 200, "lr": 5e-4,
                               "val_every": 1000},
}


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=4096, timeout=900)
def inspect_job() -> dict:
    import pyarrow.parquet as pq

    volume.reload()
    files = sorted((Path(POOL) / "lines").rglob("*.parquet"))
    table = pq.read_table(files[0], columns=["uid", "split", "text", "baseline", "polygon", "width", "height"])
    rows = table.slice(0, 3).to_pylist()
    with_baseline = sum(1 for b in table.column("baseline").to_pylist() if b and len(b) >= 2)
    synth = {path: pq.ParquetFile(path).metadata.num_rows for path in SYNTH if Path(path).exists()}
    summary = Path(POOL) / "summary.json"
    return {"files": len(files), "rows_first_file": table.num_rows, "with_baseline_first_file": with_baseline, "sample": rows,
            "synth": synth, "summary": json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else None}


@app.local_entrypoint()
def inspect():
    print(json.dumps(inspect_job.remote(), indent=1, ensure_ascii=False, default=str))


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=4096, timeout=3600, max_containers=10)
def anchor_shard(shard: int, shards: int, lift_frame: float, lift_enter: float, limit: int, tiles: int) -> dict:
    import io
    import time

    import pyarrow.parquet as pq
    import torch
    from PIL import Image

    from littera import anchors as anchoring
    from littera.place import Aligner

    torch.set_num_threads(1)
    volume.reload()
    plain = Aligner.load(FT2, snap="strip")
    lifted = Aligner.load(FT2, snap="strip")
    plain.decode.lift_snap = "cuts"
    plain.decode.inner_edges = lifted_edges = "keep"
    lifted.decode.lifts, lifted.decode.lift_snap, lifted.decode.inner_edges = True, "lifts", lifted_edges
    lifted.decode.lift_frame, lifted.decode.lift_enter = lift_frame, lift_enter
    files = sorted((Path(POOL) / "lines").rglob("*.parquet"))[shard::shards]
    records, letter_tiles, lift_tiles, started, errors = [], [], [], time.time(), 0
    for file in files:
        table = pq.read_table(file, columns=["uid", "split", "text", "image", "baseline"], filters=[("split", "=", "train")])
        for row in table.to_pylist():
            if limit and len(records) >= limit:
                break
            picture = Image.open(io.BytesIO(row["image"])).convert("L")
            baseline = [tuple(p) for p in row["baseline"]] if row["baseline"] and len(row["baseline"]) >= 2 else None
            try:
                first = plain.analyse(picture, row["text"], baseline=baseline)
                second = lifted.analyse(picture, row["text"], baseline=baseline)
                record = anchoring.derive(first, second)
            except Exception as error:  # noqa: BLE001
                errors += 1
                record = {"gaps": [], "letters": [], "error": repr(error)}
                second = None
            record["uid"] = row["uid"]
            record["lifts_decoded"] = second.extra.get("lifts", 0) if second else 0
            record["cut_edges"] = first.extra.get("lift_edges", 0) if second else 0
            records.append(record)
            if second is None or len(records) % 23:
                continue
            gaps = {gap[0]: gap for gap in record["gaps"]}
            strip, text = second.strip, row["text"]
            units, order = second.encoded.seg.units, second.encoded.unit_order()
            if record["letters"] and len(letter_tiles) < tiles:
                j, key = record["letters"][len(records) % len(record["letters"])]
                unit = units[order[j]]
                caption = f"{text[max(0, unit.i0 - 8):unit.i0]}[{key}]{text[unit.i1:unit.i1 + 8]}"
                letter_tiles.append(anchoring.tile(strip, float(strip.strip_u(gaps[j][2])), float(strip.strip_u(gaps[j + 1][1])), caption))
            lifts = [gap for gap in record["gaps"] if gap[3] == "lift"]
            if lifts and len(lift_tiles) < tiles:
                j, x0, x1, _ = lifts[len(records) % len(lifts)]
                at = units[order[j]].i0
                caption = f"{text[max(0, at - 8):at]}|{text[at:at + 8]}"
                lift_tiles.append(anchoring.tile(strip, float(strip.strip_u(x0)), float(strip.strip_u(x1)), caption))
    anchoring.write(Path(ANCHORS) / f"part-{shard:04d}.jsonl", records)
    for name, pictures in (("letters", letter_tiles), ("lifts", lift_tiles)):
        if pictures:
            anchoring.sheet(pictures).save(Path(ANCHORS) / f"sheet-{name}-{shard:04d}.png")
    volume.commit()
    return {"shard": shard, "records": [{k: r[k] for k in ("gaps", "letters", "lifts_decoded", "cut_edges")} for r in records],
            "errors": errors, "seconds": round(time.time() - started, 1)}


@app.local_entrypoint()
def anchors(lift_frame: float = 0.0, lift_enter: float = 0.0, shards: int = 10, limit: int = 0, tiles: int = 12,
            out: str = "results/r3_approach/anchors.json"):
    from littera.anchors import summarise

    parts = list(anchor_shard.starmap([(k, shards, lift_frame, lift_enter, limit, tiles) for k in range(shards)]))
    records = [r for part in parts for r in part["records"]]
    lines = max(1, len(records))
    summary = {**summarise(records), "errors": sum(p["errors"] for p in parts), "lift_frame": lift_frame, "lift_enter": lift_enter,
               "lifts_decoded_per_1000_lines": round(1000 * sum(r["lifts_decoded"] for r in records) / lines, 1),
               "cuts_with_ink_gap_per_1000_lines": round(1000 * sum(r["cut_edges"] for r in records) / lines, 1),
               "seconds_per_line": round(sum(p["seconds"] for p in parts) / lines, 3)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


@app.function(image=image, gpu="A10G", cpu=4, memory=16384, volumes={VOL: volume}, timeout=3 * 3600, max_containers=4)
def train(config: dict) -> str:
    from littera.train import TrainConfig, run

    volume.reload()
    out = run(TrainConfig.from_dict(config), device="cuda", on_save=volume.commit)
    volume.commit()
    return str(out)


@app.local_entrypoint()
def main(runs: str = "r3-control,r3-lifts", set: str = ""):
    from littera.train import merge, parse_overrides

    for name in runs.split(","):
        config = merge(merge({**BASE, "out": f"/vol/runs/{name}", "name": name}, RUNS[name]), parse_overrides(set.split()))
        call = train.spawn(config)
        print(f"{name}: spawned {call.object_id}\n{json.dumps(config, ensure_ascii=False)}")


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=1024, timeout=600)
def status_job(runs: list[str]) -> dict:
    volume.reload()
    out = {}
    for name in runs:
        path = Path(VOL) / "runs" / name / "log.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        steps = [r for r in rows if "nll_per_unit" in r]
        out[name] = {"start": next((r for r in rows if r.get("event") == "start"), None), "last": steps[-1] if steps else None,
                     "val": [r for r in rows if "val_cer" in r], "oom": sum(1 for r in rows if r.get("event") == "oom"),
                     "lines_per_s": round(sum(r["lines_per_s"] for r in steps) / max(1, len(steps)), 1)}
    return out


@app.local_entrypoint()
def status(runs: str = ",".join(RUNS)):
    print(json.dumps(status_job.remote(runs.split(",")), indent=1))


@app.function(image=image, gpu="A10G", cpu=2, memory=8192, volumes={VOL: volume}, timeout=1800)
def throughput_job(model: str, options: dict, lines: int) -> dict:
    import io
    import time

    import pyarrow.parquet as pq
    import torch
    from PIL import Image

    from littera.place import Aligner

    volume.reload()
    columns = ["uid", "split", "text", "image", "baseline"]
    rows = pq.read_table(Path(VOL) / "eval" / "humboldt-1000" / "lines.parquet", columns=columns).to_pylist()
    rows = [r for r in rows if r["split"] in ("val", "test_page")][:lines]
    pictures = [Image.open(io.BytesIO(r["image"])).convert("L") for r in rows]
    result = {"gpu": torch.cuda.get_device_name(0), "lines": len(rows)}
    for device in ("cuda", "cpu"):
        aligner = Aligner.load(model, device=device, snap=options.get("snap", "upright"))
        for key, value in options.items():
            if key != "snap":
                setattr(aligner.decode, key, value)
        subset = list(zip(rows, pictures))[: len(rows) if device == "cuda" else 40]
        for row, picture in subset[:5]:
            aligner.place(picture, row["text"], baseline=[tuple(p) for p in row["baseline"]] if row["baseline"] else None)
        started = time.perf_counter()
        placements = [aligner.place(picture, row["text"], baseline=[tuple(p) for p in row["baseline"]] if row["baseline"] else None)
                      for row, picture in subset]
        result[f"{device}_lines_per_s"] = round(len(subset) / (time.perf_counter() - started), 2)
        inner = sum(max(0, w.i1 - w.i0 - 1) for p in placements for w in p.words)
        result.update({"words": sum(1 for p in placements for w in p.words if w.i1 - w.i0 > 1),
                       "words_rescaled": sum(p.extra.get("words_rescaled", 0) for p in placements), "inner_boundaries": inner,
                       "lifts_decoded": sum(p.extra.get("lifts", 0) for p in placements),
                       "lift_edges": sum(p.extra.get("lift_edges", 0) for p in placements)} if device == "cuda" else {})
    return result


@app.local_entrypoint()
def throughput(model: str = FT2, options: str = "{}", lines: int = 200):
    print(json.dumps(throughput_job.remote(model, json.loads(options), lines)))


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=1024, timeout=600)
def release_job(run: str, checkpoint: str, name: str, decode: dict, note: str) -> dict:
    import shutil

    volume.reload()
    source = Path(run) if run.startswith("/") else Path(VOL) / "runs" / run / checkpoint
    target = Path(VOL) / "models" / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    config["name"] = name
    config["decode"] = {**config["decode"], **decode}
    (target / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    (target / "SOURCE.txt").write_text(f"{source}\ndecode set: {json.dumps(decode)}\n{note}\n", encoding="utf-8")
    volume.commit()
    return {"from": str(source), "to": str(target), "decode": config["decode"], "trained": config.get("trained")}


@app.local_entrypoint()
def release(run: str, checkpoint: str = "last", name: str = "littera-humboldt-ft3", decode: str = "{}", note: str = ""):
    print(json.dumps(release_job.remote(run, checkpoint, name, json.loads(decode), note), indent=1))
