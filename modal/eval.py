"""Evaluation jobs for littera (app littera-eval, CPU only).

    uv run modal run modal/eval.py::fetch_models                       # McCATMuS -> /vol/models/mccatmus
    uv run modal run modal/eval.py::probe                              # load the kraken placers once, place one synthetic line
    uv run modal run modal/eval.py::build_iam                          # IAM test lines with word gold -> /vol/eval/iam-words
    uv run modal run modal/eval.py::build_synth --count 2000           # /vol/eval/synth-heldfonts from heldout pool texts
    uv run modal run modal/eval.py::build_mismatch                     # /vol/eval/mismatch-heldout
    uv run modal run modal/eval.py::build_candidates                   # /vol/eval/letter-gold-candidates
    uv run modal run modal/eval.py::baselines --placers even,kraken-k2,kraken-mccatmus --sets gw20-words,iam-words,...
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("littera-eval")
volume = modal.Volume.from_name("littera", create_if_missing=True)
humboldt = modal.Volume.from_name("humboldt-htr")
VOL = Path("/vol")
EVAL = VOL / "eval"
RUNS = VOL / "eval_runs"
POOL = VOL / "pool" / "v1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("kraken>=7.1,<7.2", "pyarrow", "fonttools", "opencv-python-headless", "pillow", "numpy")
    .env({"PYTHONUTF8": "1", "LITTERA_K2_MODEL": "/hh/runs/aligner-k2/model/best.safetensors",
          "LITTERA_MCCATMUS_MODEL": "/vol/models/mccatmus/McCATMuS_nfd_nofix_V1.mlmodel"})
    .add_local_python_source("littera")
)
volumes = {VOL: volume, "/hh": humboldt.read_only()}
MCCATMUS = "https://zenodo.org/records/13788177/files/McCATMuS_nfd_nofix_V1.mlmodel?download=1"


def download(url: str, path: Path) -> Path:
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "littera-eval"})
        with urllib.request.urlopen(request, timeout=600) as response, path.open("wb") as out:
            while chunk := response.read(1 << 22):
                out.write(chunk)
    return path


@app.function(image=image, volumes=volumes, cpu=1.0, memory=2048, timeout=900)
def fetch_models_job() -> dict:
    path = download(MCCATMUS, VOL / "models" / "mccatmus" / "McCATMuS_nfd_nofix_V1.mlmodel")
    download("https://zenodo.org/records/13788177/files/metadata.json?download=1", path.parent / "metadata.json")
    volume.commit()
    return {"path": str(path), "bytes": path.stat().st_size}


@app.local_entrypoint()
def fetch_models():
    print(fetch_models_job.remote())


@app.function(image=image, volumes=volumes, cpu=2.0, memory=4096, timeout=900)
def probe_job() -> dict:
    import time

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from littera.eval.placers import make_placer

    picture = Image.new("L", (900, 80), 255)
    ImageDraw.Draw(picture).text((20, 55), "und schrieb mir den Brief", font=ImageFont.load_default(40), fill=0, anchor="ls")
    out = {}
    for spec in ("kraken-k2", "kraken-mccatmus"):
        started = time.perf_counter()
        placer = make_placer(spec)
        loaded = time.perf_counter() - started
        log_probs, strip = placer.run(picture)
        placement = placer.place(picture, "und schrieb mir den Brief")
        out[spec] = {"load_s": round(loaded, 1), "input": list(placer.net.input), "classes": log_probs.shape[1],
                     "frames": log_probs.shape[0], "normal_form": placer.normal_form, "strip": list(strip.shape),
                     "strip_range": [float(strip.min()), float(strip.max())], "row_sum": float(np.exp(log_probs).sum(1).mean()),
                     "line_fit": placement.line_fit, "words": [(w.x0, w.x1, round(w.fit, 3)) for w in placement.words],
                     "greedy": greedy_text(placer, log_probs)}
    return out


def greedy_text(placer, log_probs) -> str:
    best = log_probs.argmax(1).tolist()
    labels = [c for t, c in enumerate(best) if c != 0 and (t == 0 or c != best[t - 1])]
    try:
        return "".join(item[0] for item in placer.net.codec.decode([(label, 0, 0, 1.0) for label in labels]))
    except Exception as error:  # noqa: BLE001
        return repr(error)


@app.local_entrypoint()
def probe():
    print(json.dumps(probe_job.remote(), indent=1))


@app.function(image=image, volumes=volumes, cpu=4.0, memory=8192, timeout=3600)
def build_iam_job(limit: int = 1000) -> dict:
    from littera.eval.sets import set_stats, write_readme, write_set
    from littera.eval.wordgold import iam

    folder = Path("/tmp/iam")
    paths = {name: download(url, folder / name) for name, url in iam.FILES.items()}
    rows, extra = iam.build_rows(paths["xml.tar"], paths["lines.tar"], paths["largeWriterIndependentTextLineRecognitionTask.zip"], limit)
    write_set(rows, EVAL / "iam-words" / "lines.parquet")
    stats = {**set_stats(rows), "conversion": extra}
    write_readme(EVAL / "iam-words", "iam-words", iam.README, stats)
    volume.commit()
    return stats


@app.local_entrypoint()
def build_iam(limit: int = 1000):
    print(json.dumps(build_iam_job.remote(limit), indent=1))


def heldout_index(columns: list[str]):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(POOL / "index.parquet", columns=columns)
    return table.filter(pc.equal(table["split"], "heldout"))


@app.function(image=image, volumes=volumes, cpu=4.0, memory=8192, timeout=3600)
def build_synth_job(count: int = 2000, seed: int = 7) -> dict:
    import random

    from littera.synth.generate import build_set

    rows = heldout_index(["uid", "source", "text", "split"]).to_pylist()
    rows = [r for r in rows if 8 <= len(r["text"]) <= 90]
    random.Random(seed).shuffle(rows)
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    share = max(1, (3 * count) // max(1, len(by_source)))
    picked = [r for group in by_source.values() for r in group[:share]]
    random.Random(seed + 1).shuffle(picked)
    stats = build_set(str(VOL / "fonts"), [r["text"] for r in picked], count, "heldout", seed, str(EVAL), "synth-heldfonts",
                      [r["uid"] for r in picked])
    volume.commit()
    return stats


@app.local_entrypoint()
def build_synth(count: int = 2000, seed: int = 7):
    print(json.dumps(build_synth_job.remote(count, seed), indent=1, ensure_ascii=False))


@app.function(image=image, volumes=volumes, cpu=4.0, memory=16384, timeout=3600)
def build_mismatch_job(per_source: int = 120, seed: int = 11) -> dict:
    from littera.eval.mismatch import README, build_mismatch_rows
    from littera.eval.sets import set_stats, write_readme, write_set

    rows, extra = build_mismatch_rows(POOL, per_source=per_source, seed=seed)
    write_set(rows, EVAL / "mismatch-heldout" / "lines.parquet")
    stats = {**set_stats(rows), **extra}
    write_readme(EVAL / "mismatch-heldout", "mismatch-heldout", README, stats)
    volume.commit()
    return stats


@app.local_entrypoint()
def build_mismatch(per_source: int = 120, seed: int = 11):
    print(json.dumps(build_mismatch_job.remote(per_source, seed), indent=1, ensure_ascii=False))


@app.function(image=image, volumes=volumes, cpu=4.0, memory=16384, timeout=3600)
def build_candidates_job(total: int = 120, seed: int = 5) -> dict:
    from littera.eval.candidates import README, build_candidates
    from littera.eval.placers import make_placer
    from littera.eval.sets import set_stats, write_readme, write_set

    rows, extra = build_candidates(POOL, [make_placer("kraken-k2"), make_placer("kraken-mccatmus")], total=total, seed=seed)
    write_set(rows, EVAL / "letter-gold-candidates" / "lines.parquet")
    stats = {**set_stats(rows), **extra}
    write_readme(EVAL / "letter-gold-candidates", "letter-gold-candidates", README, stats)
    volume.commit()
    return stats


@app.local_entrypoint()
def build_candidates(total: int = 120, seed: int = 5):
    print(json.dumps(build_candidates_job.remote(total, seed), indent=1, ensure_ascii=False))


@app.function(image=image, volumes=volumes, cpu=4.0, memory=16384, timeout=3600)
def add_candidates_job(sources: list[str], per_source: int = 12, seed: int = 5, exclude: tuple[str, ...] = ()) -> dict:
    import shutil

    from littera.eval.candidates import README, build_candidates
    from littera.eval.placers import make_placer
    from littera.eval.sets import read_set, set_stats, write_readme, write_set

    volume.reload()
    folder = EVAL / "letter-gold-candidates"
    if not (folder / "lines_round1.parquet").exists():
        shutil.copy2(folder / "lines.parquet", folder / "lines_round1.parquet")
    rows, extra = build_candidates(POOL, [make_placer("kraken-k2"), make_placer("kraken-mccatmus")], total=per_source * len(sources),
                                   seed=seed, extra_sources=(), sources=sources, exclude=tuple(exclude))
    kept = [r for r in read_set(folder / "lines.parquet") if r["source"] not in sources]
    write_set(kept + rows, folder / "lines.parquet")
    stats = {**set_stats(kept + rows), "added": {"sources": sources, "excluded_by_eye": list(exclude), **extra}}
    write_readme(folder, "letter-gold-candidates", README, stats)
    volume.commit()
    return {"kept": len(kept), "added": len(rows), **extra,
            "rows": [{"uid": r["uid"], "text": r["text"], "height": r["height"], "origin": r["origin"]} for r in rows]}


@app.local_entrypoint()
def add_candidates(sources: str = "laubmann", per_source: int = 12, seed: int = 5, exclude: str = ""):
    excluded = tuple(u for u in exclude.split(",") if u)
    print(json.dumps(add_candidates_job.remote(sources.split(","), per_source, seed, excluded), indent=1, ensure_ascii=False))


SYNTH_TRAIN = VOL / "synth" / "v1"


@app.function(image=image, volumes=volumes, cpu=1.0, memory=3072, timeout=3600, max_containers=10)
def synth_train_part(part: int, count: int, seed: int) -> dict:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from littera.synth.generate import write_training_part

    table = pq.read_table(POOL / "index.parquet", columns=["uid", "split", "open_licence", "text", "language", "licence"])
    table = table.filter(pc.and_(pc.equal(table["split"], "train"), table["open_licence"]))
    texts = [r for r in table.to_pylist() if 8 <= len(r["text"]) <= 100]
    result = write_training_part(str(VOL / "fonts"), texts, count, seed + part, part, SYNTH_TRAIN)
    volume.commit()
    return result


@app.function(image=image, volumes=volumes, cpu=1.0, memory=1024, timeout=600)
def synth_train_readme(total: int, parts: list[dict]) -> None:
    import shutil

    from littera.synth.fonts import load_catalogue

    fonts = [f for f in load_catalogue(VOL / "fonts") if f["split"] == "train"]
    lines = "\n".join(f"- {f['name']} ({f['style']}): {f['licence']}" for f in fonts)
    (SYNTH_TRAIN / "README.md").write_text(
        "# Synthetic training lines v1\n\nRendered by `littera.synth.generate.write_training_part` with the training fonts (the held-out "
        "fonts of `synth-heldfonts` are never used here) from texts of open-licence train rows of /pool/v1. Pool schema plus `gold_words` "
        "and `gold_cuts` (boundary before text[i], pen position at mid x-height after the slant), split `train`, source `synth`. Add "
        f"`/vol/synth/v1` to `TrainConfig.pool`.\n\nRows: {total}\n\nFonts:\n\n{lines}\n\n```json\n"
        + json.dumps(parts, indent=1) + "\n```\n",
        encoding="utf-8")
    shutil.copy(VOL / "fonts" / "fonts.json", SYNTH_TRAIN / "fonts.json")
    volume.commit()


@app.local_entrypoint()
def build_synth_train(count: int = 20000, parts: int = 10, first_part: int = 0, seed: int = 1000):
    per_part = -(-count // parts)
    results = list(synth_train_part.starmap([(first_part + k, per_part, seed) for k in range(parts)]))
    total = sum(r["rows"] for r in results)
    synth_train_readme.remote(total, [{k: r[k] for k in ("part", "rows")} for r in results])
    print(json.dumps({"rows": total, "parts": [(r["part"], r["rows"]) for r in results]}))


PLACERS: dict[str, object] = {}


@app.function(image=image, volumes=volumes, cpu=2.0, memory=4096, timeout=3600, max_containers=10)
def place_shard(placer_spec: str, set_name: str, shard: int, shards: int) -> dict:
    from littera.eval.placers import make_placer
    from littera.eval.run import run

    volume.reload()
    if placer_spec not in PLACERS:
        PLACERS[placer_spec] = make_placer(placer_spec)
    result = run(placer_spec, set_name, str(EVAL), str(RUNS), shard, shards, placer=PLACERS[placer_spec])
    volume.commit()
    return {"placer": placer_spec, "set": set_name, "shard": shard, **result}


@app.function(image=image, volumes=volumes, cpu=1.0, memory=4096, timeout=1800, max_containers=1)
def merge_job(placer_spec: str, set_name: str, shards: int | None = None) -> dict:
    from littera.eval.placers import placer_label
    from littera.eval.run import merge_scores

    volume.reload()
    result = merge_scores(RUNS / placer_label(placer_spec) / set_name, {"placer": placer_spec, "set": set_name, "shards": shards}, shards)
    volume.commit()
    return result


@app.function(image=image, volumes=volumes, cpu=1.0, memory=2048, timeout=600)
def set_sizes(set_names: list[str]) -> dict:
    import pyarrow.parquet as pq

    return {name: pq.ParquetFile(EVAL / name / "lines.parquet").metadata.num_rows for name in set_names
            if (EVAL / name / "lines.parquet").exists()}


@app.function(image=image, volumes=volumes, cpu=1.0, memory=4096, timeout=1800, max_containers=5)
def rescore_job(placer_spec: str, set_name: str) -> dict:
    from littera.eval.placers import placer_label
    from littera.eval.run import rescore

    volume.reload()
    result = rescore(set_name, str(EVAL), RUNS / placer_label(placer_spec) / set_name)
    volume.commit()
    return {"placer": placer_spec, "set": set_name, **result}


@app.local_entrypoint()
def rescore(placers: str = "even,kraken-k2,kraken-mccatmus", sets: str = "synth-heldfonts"):
    pairs = [(p, s) for p in placers.split(",") for s in sets.split(",")]
    for result in rescore_job.starmap(pairs):
        print(json.dumps(result))


@app.local_entrypoint()
def merge(placers: str = "even,kraken-k2,kraken-mccatmus", sets: str = "gw20-words", rows_per_shard: int = 400,
          out: str = "runs/eval/baselines.json"):
    set_list = sets.split(",")
    sizes = set_sizes.remote(set_list)
    pairs = [(p, s, max(1, -(-sizes[s] // rows_per_shard))) for p in placers.split(",") for s in set_list if s in sizes]
    scores = dict(zip([f"{p}|{s}" for p, s, _ in pairs], merge_job.starmap(pairs)))
    path = Path(out)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(scores)
    path.write_text(json.dumps(existing, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


@app.local_entrypoint()
def baselines(placers: str = "even,kraken-k2,kraken-mccatmus", sets: str = "gw20-words", rows_per_shard: int = 400,
              out: str = "runs/eval/baselines.json"):
    placer_list, set_list = placers.split(","), sets.split(",")
    sizes = set_sizes.remote(set_list)
    jobs = []
    for placer_spec in placer_list:
        for set_name in set_list:
            if set_name not in sizes:
                print(f"missing set {set_name}")
                continue
            shards = max(1, -(-sizes[set_name] // rows_per_shard))
            jobs += [(placer_spec, set_name, shard, shards) for shard in range(shards)]
    print(f"{len(jobs)} shards: {sizes}")
    for result in place_shard.starmap(jobs, order_outputs=False):
        print(json.dumps(result))
    pairs = sorted({(p, s, n) for p, s, _, n in jobs})
    scores = dict(zip([f"{p}|{s}" for p, s, _ in pairs], merge_job.starmap(pairs)))
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(scores)
    path.write_text(json.dumps(existing, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


torch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy==2.5.4", "pillow==13.0.0", "opencv-python-headless==5.0.0.93", "pyarrow==26.0.1", "lxml==6.1.3",
                 "safetensors==0.8.0")
    .env({"PYTHONUTF8": "1", "OMP_NUM_THREADS": "1"})
    .add_local_python_source("littera")
)
DECODE_PRESETS = {"synth-heldfonts": "plain"}
DECODE_PARTS = {"tune": 0, "report": 1}


@app.function(image=torch_image, volumes={VOL: volume}, cpu=1.0, memory=3072, timeout=3600, max_containers=6)
def decode_shard(model: str, set_name: str, variants: dict, part: str, shard: int, shards: int, limit: int) -> dict:
    import time

    import torch

    from littera.eval.ink import ink_profile
    from littera.eval.metrics import row_scores
    from littera.eval.run import baseline_or_none, polygon_or_none
    from littera.eval.sets import decode_image, iter_set, set_path
    from littera.place import Aligner

    torch.set_num_threads(1)
    rows = [row for k, row in enumerate(iter_set(set_path(set_name, str(EVAL)))) if k % 2 == DECODE_PARTS[part]]
    rows = (rows[:limit] if limit else rows)[shard::shards]
    aligner = Aligner.load(VOL / "models" / model, device="cpu")
    aligner.preset = DECODE_PRESETS.get(set_name, aligner.preset)
    released = aligner.decode.to_dict()
    out = {}
    for name, options in variants.items():
        snap = options.get("snap", "strip")
        aligner.snap = snap
        aligner.decode = type(aligner.decode)(**{**released, **{k: v for k, v in options.items() if k != "snap"}})
        scores, lifts, edges, started = [], 0, 0, time.perf_counter()
        for row in rows:
            image = decode_image(row["image"])
            polygon, baseline = polygon_or_none(row.get("polygon")), baseline_or_none(row.get("baseline"))
            placement = aligner.place(image, row["text"], polygon=polygon, baseline=baseline)
            lifts += placement.extra.get("lifts", 0)
            edges += placement.extra.get("lift_edges", 0)
            scores.append(row_scores(placement, row, ink_profile(image, polygon)))
        out[name] = {"scores": scores, "lifts": lifts, "lift_edges": edges, "seconds": round(time.perf_counter() - started, 1)}
    return out


@app.local_entrypoint()
def decode_grid(variants: str, model: str = "littera-pilot-seg", sets: str = "synth-heldfonts,gw20-words", part: str = "tune",
                limit: int = 400, shards: int = 6, out: str = "results/r3_approach/decode.json"):
    from littera.eval.metrics import aggregate

    grid = json.loads(Path(variants).read_text(encoding="utf-8"))
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for set_name in sets.split(","):
        parts = list(decode_shard.starmap([(model, set_name, grid, part, shard, shards, limit) for shard in range(shards)]))
        for name, options in grid.items():
            scores = [s for p in parts for s in p[name]["scores"]]
            result = aggregate(scores, by_source=False)
            cuts = sum(len(s["cuts"]) for s in scores)
            row = {"options": options, "lines": len(scores), "lifts": sum(p[name]["lifts"] for p in parts),
                   "lift_edges": sum(p[name]["lift_edges"] for p in parts), "inner_cuts": cuts,
                   "seconds_per_line": round(sum(p[name]["seconds"] for p in parts) / max(1, len(scores)), 3),
                   "line_fit_mean": round(sum(s["line_fit"] for s in scores) / max(1, len(scores)), 4),
                   **{k: result[k] for k in ("word", "letter", "char", "gap_rate") if k in result}}
            table[f"{model}|{set_name}|{part}|{name}"] = row
            letter, word, char = row.get("letter", {}), row.get("word", {}), row.get("char", {})
            print(f"{set_name:16s} {part:6s} {name:28s} letters<=.25 {letter.get('within_0.25')} med {letter.get('median')} "
                  f"char {char.get('both_cuts_within_0.25')} word<=.25 {word.get('within_0.25')} med {word.get('median')} "
                  f"gap {row.get('gap_rate')} fit {row['line_fit_mean']} lifts {row['lifts']} edges {row['lift_edges']} "
                  f"s/line {row['seconds_per_line']}", flush=True)
    path.write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
