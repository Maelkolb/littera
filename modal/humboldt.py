"""Humboldt measurement of round 3 (app littera-humboldt; CPU, and one GPU job for the occlusion evidence). docs/r3/MEASURE.md.

    uv run modal run modal/humboldt.py::build                                   # /vol/pool/humboldt_ft and /vol/eval/humboldt-1000
    uv run modal run modal/humboldt.py::import_k2                               # k2-raw-mid, k2-snap-mid, even from place-k2.jsonl
    uv run modal run modal/humboldt.py::place --model /vol/models/littera-humboldt-ft2 --name ft2-upright --options '{"snap": "upright"}'
    uv run modal run modal/humboldt.py::score --names k2-snap-mid,ft2-off,ft2-upright [--halves dev,report]
    uv run modal run modal/humboldt.py::classify --names k2-snap-mid,ft2-upright [--halves dev,report] [--reuse]
    uv run modal run modal/humboldt.py::widths --names k2-snap-mid,ft2-upright
    uv run modal run modal/humboldt.py::occlusion --names k2-snap-mid,ft2-upright
    uv run modal run modal/humboldt.py::sheets --a k2-snap-mid --b ft2-upright
    uv run modal run modal/humboldt.py::peek --uids <uid,uid> --names <a,b> --out <folder>   # crops, neutral strip, edges to look at

Placements: /vol/eval_runs/humboldt-1000/<name>.jsonl, one row per line, edges in crop pixels of the upright crop.
Derived placers need no file: even@<name> (even spacing inside the words of <name>) and mix:<inner>+<outer> (word edges of
<outer>, letter edges inside words of <inner>).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("littera-humboldt")
volume = modal.Volume.from_name("littera", create_if_missing=True)
humboldt = modal.Volume.from_name("humboldt-htr")
VOL = Path("/vol")
HH = Path("/hh")
POOL = VOL / "pool" / "humboldt_ft"
EVAL = VOL / "eval" / "humboldt-1000"
RUNS = VOL / "eval_runs" / "humboldt-1000"
HUMBOLDT_HTR = "C:/Users/totom/Projects/humboldt-htr"

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("kraken>=7.1,<7.2", "pyarrow", "fonttools", "opencv-python-headless", "pillow", "numpy")
    .env({"PYTHONUTF8": "1"})
)
image = base_image.add_local_python_source("littera")
sheet_image = base_image.apt_install("fonts-dejavu-core").add_local_python_source("littera")
volumes = {VOL: volume, HH: humboldt.read_only()}


def chunks(items: list, n: int) -> list[list]:
    n = max(1, min(n, len(items)))
    return [items[k::n] for k in range(n)]


# ---------------------------------------------------------------------------- data


@app.function(image=image, volumes=volumes, cpu=2.0, memory=4096, timeout=3600, max_containers=10)
def pool_part(part: int, pages: list[dict]) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from littera.data.schema import POOL_SCHEMA
    from littera.eval import humboldt_data as hd
    from littera.eval.sets import encode_image

    records, skipped = [], {"not matched": 0, "empty text": 0, "no page": 0}
    for page in pages:
        match_path = HH / "lookup" / "lines" / f"{page['stem']}.json"
        image_path = HH / "images" / f"{page['stem']}.jpg"
        if not match_path.exists() or not image_path.exists():
            skipped["no page"] += len(page["lines"])
            continue
        matched = {ln["id"]: ln for ln in json.loads(match_path.read_text(encoding="utf-8"))["lines"]}
        picture = Image.open(image_path).convert("RGB")
        journal, page_id = page["stem"].split("/")
        for line in page["lines"]:
            if not line["text"]:
                skipped["empty text"] += 1
                continue
            match = matched.get(line["uid"])
            if match is None or not match.get("pieces"):
                skipped["not matched"] += 1
                continue
            crop, box = hd.production_crop(picture, match["pieces"])
            records.append({
                "uid": line["uid"], "source": hd.SOURCE, "doc": journal, "page": page_id, "hand": "humboldt", "split": page["split"],
                "image": encode_image(crop, "JPEG", 95), "width": crop.width, "height": crop.height, "text": line["text"],
                "script": "latin", "language": "", "licence": hd.LICENCE, "open_licence": True, "quality": line["grade"],
                "polygon": [], "baseline": hd.line_baseline(match["pieces"], box),
                "origin": hd.origin(page["stem"], box, match["pieces"], kind=line["kind"], csv_split=line["csv_split"]),
            })
    name = f"lines/{hd.SOURCE}/part-{part:05d}.parquet"
    path = POOL / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=POOL_SCHEMA), path, row_group_size=256, compression="zstd")
    volume.commit()
    index = [{**{k: v for k, v in r.items() if k != "image"}, "part": name, "part_row": k} for k, r in enumerate(records)]
    return {"part": name, "index": index, "skipped": skipped}


@app.function(image=image, volumes=volumes, cpu=1.0, memory=4096, timeout=1800)
def pool_finish(index: list[dict], notes: dict) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from littera.data.schema import INDEX_SCHEMA

    volume.reload()
    pq.write_table(pa.Table.from_pylist(index, schema=INDEX_SCHEMA), POOL / "index.parquet", compression="zstd")
    stats = {"rows": len(index), "splits": {}, **notes}
    for split in ("train", "val"):
        rows = [r for r in index if r["split"] == split]
        stats["splits"][split] = {"lines": len(rows), "pages": len({r["page"] for r in rows}), "journals": len({r["doc"] for r in rows}),
                                  "characters": sum(len(r["text"]) for r in rows)}
    stats["grades"] = {g: sum(r["quality"] == g for r in index) for g in sorted({r["quality"] for r in index})}
    stats["median_height"] = sorted(r["height"] for r in index)[len(index) // 2]
    stats["median_width"] = sorted(r["width"] for r in index)[len(index) // 2]
    (POOL / "summary.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    volume.commit()
    return stats


@app.function(image=image, volumes=volumes, cpu=2.0, memory=8192, timeout=3600)
def eval_set(recipe: list[dict]) -> dict:
    from PIL import Image

    from littera.eval import humboldt_data as hd
    from littera.eval.sets import encode_image, write_readme, write_set

    rows = []
    pages: dict[str, Image.Image] = {}
    for entry in recipe:
        if entry["stem"] not in pages:
            pages = {entry["stem"]: Image.open(HH / "images" / f"{entry['stem']}.jpg").convert("RGB")}
        crop, box = hd.production_crop(pages[entry["stem"]], entry["pieces"])
        journal, page_id = entry["stem"].split("/")
        rows.append({
            "uid": entry["uid"], "source": hd.EVAL_SET, "doc": journal, "page": page_id, "hand": "humboldt", "split": entry["split"],
            "image": encode_image(crop, "PNG"), "width": crop.width, "height": crop.height, "text": hd.text_on_crop(entry["t"], entry["f"])[0],
            "polygon": [], "baseline": hd.line_baseline(entry["pieces"], box), "script": "latin", "licence": hd.LICENCE, "open_licence": False, "kind": "word", "true_text": entry["t"], "variant": entry["f"],
            "origin": hd.origin(entry["stem"], box, entry["pieces"], kind=entry["k"], other=entry["other"]),
        })
    write_set(rows, EVAL / "lines.parquet")
    stats = {"rows": len(rows), "splits": {s: sum(r["split"] == s for r in rows) for s in hd.SPLITS},
             "lines with text above the line removed": sum(r["text"] != r["true_text"] for r in rows),
             "characters": sum(len(r["text"]) for r in rows), "pages": len({r["page"] for r in rows})}
    write_readme(EVAL, hd.EVAL_SET, """
The 1,000 lines of humboldt-htr's aligner evaluation (`runs/aligner-k2/eval/recipe.json`), 200 per split, cut from the page as
the lookup build cuts them (`crop_fragments(page, pieces, grow=6)`), stored as PNG. `text` is the recipe's text as the lookup build
places it (`modal/humboldt_index.py::line_text_on_crop`: characters flagged `a`, written above the line, removed, the double
space this leaves collapsed), `true_text` the recipe's full text, `variant` its flags. `baseline` holds all baseline points of the
line's pieces, sorted by x, in crop pixels, and `polygon` is empty: production places with `aligner.place(crop, text,
baseline=points)`. `origin` holds the page stem, the crop's box on the page and the baselines piece by piece. `kind` is `word` for the schema's sake; there is no gold. Dev half: val, test_page. Report half: test_bench, test_journal,
letters. Evaluation only.
""", stats)
    volume.commit()
    return stats


def read_csv(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@app.local_entrypoint()
def build(humboldt_htr: str = HUMBOLDT_HTR, only: str = "pool,eval", parts: int = 20, val_pages: int = 31, seed: int = 3):
    from littera.eval import humboldt_data as hd

    base = Path(humboldt_htr) / "runs" / "aligner-k2"
    recipe = json.loads((base / "eval" / "recipe.json").read_text(encoding="utf-8"))
    eval_call = eval_set.spawn(recipe) if "eval" in only else None
    if "pool" in only:
        plan = hd.pool_plan(read_csv(base / "lines-r6k.csv"), recipe, val_pages, seed)
        groups = chunks(sorted(plan["pages"], key=lambda p: -len(p["lines"])), parts)
        results = list(pool_part.starmap(list(enumerate(groups))))
        skipped: dict[str, int] = {}
        for result in results:
            for key, count in result["skipped"].items():
                skipped[key] = skipped.get(key, 0) + count
        index = [row for result in results for row in result["index"]]
        notes = {"dropped_before": plan["dropped"], "skipped": skipped, "recipe_pages": plan["recipe_pages"], "val_pages": val_pages, "seed": seed}
        print(json.dumps(pool_finish.remote(index, notes), indent=1, ensure_ascii=False))
    if eval_call is not None:
        print(json.dumps(eval_call.get(), indent=1, ensure_ascii=False))


# ---------------------------------------------------------------------------- placements


def read_eval(uids: list[str] | None = None, columns: list[str] | None = None) -> list[dict]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(EVAL / "lines.parquet", columns=columns)
    if uids is not None:
        table = table.filter(pc.is_in(table["uid"], pa.array(list(uids), pa.string())))
    return table.to_pylist()


def read_placements(name: str) -> dict[str, dict]:
    path = RUNS / f"{name}.jsonl"
    return {row["uid"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def make_aligner(model: str, options: dict):
    import dataclasses

    from littera.geometry import GeometryConfig
    from littera.graph import GraphOptions
    from littera.place import Aligner, Objective

    options = dict(options)
    decode = dict(options.pop("decode", None) or {})
    objective = options.pop("objective", None)
    geometry = options.pop("geometry", None)
    for key in [k for k in options if k in {f.name for f in dataclasses.fields(GraphOptions)}]:
        decode[key] = options.pop(key)
    aligner = Aligner.load(model, **options)
    if objective:
        aligner.objective = Objective.from_dict({**dataclasses.asdict(aligner.objective), **objective})
    if geometry:
        aligner.geometry = GeometryConfig.from_dict({**aligner.geometry.to_dict(), **geometry})
    if decode:
        aligner.decode = GraphOptions.from_dict({**aligner.decode.to_dict(), **decode})
    return aligner


@app.function(image=image, volumes=volumes, cpu=2.0, memory=4096, timeout=3600, max_containers=10)
def place_shard(model: str, options: dict, uids: list[str]) -> list[dict]:
    import time

    import torch

    from littera.eval.sets import decode_image

    torch.set_num_threads(2)
    volume.reload()
    aligner = make_aligner(model, options)
    out = []
    for row in read_eval(uids, ["uid", "image", "text", "baseline"]):
        picture = decode_image(row["image"]).convert("RGB")
        baseline = [(float(x), float(y)) for x, y in row["baseline"] or []]
        started = time.perf_counter()
        placement = aligner.place(picture, row["text"], baseline=baseline if len(baseline) >= 2 else None)
        out.append({"uid": row["uid"], "x": [round(float(v), 1) for v in placement.cuts()], "visible": [c.visible for c in placement.chars],
                    "q": [round(float(c.q), 3) for c in placement.chars], "line_fit": placement.line_fit, "slant": placement.slant,
                    "letter": placement.extra.get("letter"), "seconds": round(time.perf_counter() - started, 3)})
    return out


@app.function(image=image, volumes=volumes, cpu=1.0, memory=2048, timeout=900)
def write_placements(name: str, rows: list[dict], meta: dict) -> dict:
    volume.reload()
    lengths = {r["uid"]: len(r["text"]) for r in read_eval(None, ["uid", "text"])}
    wrong = [r["uid"] for r in rows if len(r["x"]) != lengths.get(r["uid"], -2) + 1]
    if wrong:
        raise ValueError(f"{len(wrong)} rows of {name} do not have len(text) + 1 edges, e.g. {wrong[:3]}")
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (RUNS / f"{name}.meta.json").write_text(json.dumps({**meta, "lines": len(rows)}, indent=1), encoding="utf-8")
    volume.commit()
    return {"name": name, "lines": len(rows), "path": str(RUNS / f"{name}.jsonl")}


@app.function(image=image, volumes=volumes, cpu=1.0, memory=2048, timeout=600)
def eval_uids(halves: list[str]) -> list[str]:
    from littera.eval.humboldt import HALVES

    volume.reload()
    wanted = {split for half in halves for split in HALVES[half]}
    return [r["uid"] for r in read_eval(None, ["uid", "split"]) if r["split"] in wanted]


@app.local_entrypoint()
def place(model: str = "/vol/models/littera-humboldt-ft2", name: str = "ft2-upright", options: str = '{"snap": "upright"}',
          halves: str = "dev,report", shards: int = 10):
    """``options``: JSON. ``snap`` and any other keyword of ``Aligner.load``; fields of ``GraphOptions`` at the top level or under
    ``decode``; ``objective`` and ``geometry`` as dicts merged into the model's."""
    chosen = json.loads(options)
    uids = eval_uids.remote(halves.split(","))
    rows = [r for part in place_shard.starmap([(model, chosen, group) for group in chunks(uids, shards)]) for r in part]
    seconds = sum(r["seconds"] for r in rows)
    meta = {"model": model, "options": chosen, "halves": halves, "cpu_seconds_per_line": round(seconds / max(1, len(rows)), 3)}
    print(json.dumps({**write_placements.remote(name, rows, meta), **meta}))


@app.function(image=image, volumes=volumes, cpu=2.0, memory=4096, timeout=1800)
def even_job() -> list[dict]:
    from littera.eval.humboldt import even_line, neutral_strip
    from littera.eval.sets import decode_image

    volume.reload()
    out = []
    for row in read_eval(None, ["uid", "image", "text"]):
        x = even_line(neutral_strip(decode_image(row["image"]), sensitivity=1.0), row["text"])
        out.append({"uid": row["uid"], "x": x, "visible": [not ch.isspace() for ch in row["text"]], "q": [0.0] * len(row["text"]),
                    "line_fit": 0.0, "slant": 0.0})
    return out


@app.local_entrypoint()
def import_k2(humboldt_htr: str = HUMBOLDT_HTR, variants: str = "raw-mid,snap-mid"):
    from littera.eval.humboldt_data import text_on_crop

    base = Path(humboldt_htr) / "runs" / "aligner-k2" / "eval"
    recipe = {e["uid"]: e for e in json.loads((base / "recipe.json").read_text(encoding="utf-8"))}
    placed = [json.loads(line) for line in (base / "place-k2.jsonl").read_text(encoding="utf-8").splitlines()]
    for variant in variants.split(","):
        rows = []
        for row in placed:
            entry, k2 = recipe[row["uid"]], row["models"]["k2"]
            text, index = text_on_crop(entry["t"], entry["f"])
            cuts = k2["cuts"][variant]
            x = [cuts[i] for i in index] + [cuts[index[-1] + 1] if index else 0.0]
            rows.append({"uid": row["uid"], "x": x, "visible": [not ch.isspace() for ch in text], "q": [k2["q"][i] / 10 for i in index],
                         "line_fit": k2["line_q"], "slant": 0.0, "w": row["w"]})
        print(write_placements.remote(f"k2-{variant}", rows, {"model": "aligner-k2", "source": "humboldt-htr place-k2.jsonl", "variant": variant}))
    print(write_placements.remote("even", even_job.remote(), {"model": "even spacing over the ink extent of the neutral strip"}))


# ---------------------------------------------------------------------------- measures


def neutral_lines(halves: list[str]) -> dict[str, dict]:
    from littera.eval.humboldt import HALVES, neutral_strip
    from littera.eval.sets import decode_image

    wanted = {split for half in halves for split in HALVES[half]}
    lines = {}
    for row in read_eval(None, ["uid", "split", "image", "text", "width"]):
        if row["split"] in wanted:
            lines[row["uid"]] = {"text": row["text"], "split": row["split"], "width": row["width"], "neutral": neutral_strip(decode_image(row["image"]))}
    return lines


def base_names(names: list[str]) -> list[str]:
    from littera.eval.humboldt import base_names as bases

    return bases(names)


@app.function(image=image, volumes=volumes, cpu=4.0, memory=8192, timeout=1800)
def score_job(names: list[str], halves: list[str], refs: list[str]) -> dict:
    from littera.eval.humboldt import score as score_lines

    volume.reload()
    placements = {name: read_placements(name) for name in base_names([*names, *refs])}
    widths = {}
    lines = neutral_lines(halves)
    for name, rows in placements.items():
        widths[name] = sum(1 for uid, r in rows.items() if uid in lines and "w" in r and r["w"] != lines[uid]["width"])
    result = score_lines(lines, placements, names, halves, tuple(refs))
    result["crop_width_mismatches"] = {k: v for k, v in widths.items() if v}
    return result


def table(result: dict, measure: str, title: str) -> str:
    groups = list(next(iter(result["placers"].values())).keys())
    rows = [f"| {title} | " + " | ".join(groups) + " |", "|---|" + "---:|" * len(groups)]
    for name, by_group in result["placers"].items():
        cells = []
        for group in groups:
            value = by_group[group].get(measure)
            cells.append("" if value is None else f"{value[0]:.1f} ({value[1]:.1f} to {value[2]:.1f})")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


MEASURES = (("word_boundary", "word boundary on an ink-free column, %"), ("lift_cover", "lifts holding a letter edge (1.5 px), %"),
            ("lift_near", "lifts with a letter edge within 4 px, %"), ("lift_same_boundary_as_first_ref", "same boundary in the lift as the first reference, %"),
            ("inner_edges_on_empty_column", "inner edges on an ink-free column, %"),
            ("word_boundary_minus_first_ref", "word boundaries, points above the first reference (paired)"),
            ("lift_cover_minus_first_ref", "lifts held, points above the first reference (paired)"))


@app.local_entrypoint()
def score(names: str = "even,k2-raw-mid,k2-snap-mid,even@k2-snap-mid,pilot-upright,ft2-off,ft2-upright,even@ft2-upright", halves: str = "dev",
          refs: str = "k2-snap-mid,ft2-upright", out: str = ""):
    result = score_job.remote(names.split(","), halves.split(","), refs.split(","))
    path = Path(out or f"results/r3/score_{halves.replace(',', '_')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["hand"], indent=1))
    for measure, title in MEASURES:
        print("\n" + table(result, measure, title))
    print(f"\nwrote {path}")


@app.function(image=image, volumes=volumes, cpu=2.0, memory=4096, timeout=900)
def peek_job(uids: list[str], names: list[str], pool: bool) -> dict[str, bytes]:
    import io

    import pyarrow.parquet as pq
    from PIL import ImageDraw

    from littera.eval.humboldt import REFS, classify_runs, neutral_strip, overlay
    from littera.eval.sets import decode_image

    volume.reload()
    if pool:
        rows = pq.read_table(POOL / "lines" / "humboldt_ft" / "part-00000.parquet").slice(0, 400).to_pylist()
        rows = [r for r in rows if r["uid"] in uids] if uids else [rows[k] for k in (0, 137, 311)]
    else:
        rows = read_eval(uids)
    placements = {name: read_placements(name) for name in names}
    out = {}

    def png(picture) -> bytes:
        buffer = io.BytesIO()
        picture.save(buffer, "PNG")
        return buffer.getvalue()

    for row in rows:
        picture = decode_image(row["image"])
        marked = picture.convert("RGB")
        if row["baseline"]:
            ImageDraw.Draw(marked).line([tuple(p) for p in row["baseline"]], fill=(220, 40, 40), width=1)
        out[f"{row['uid']}.crop.png"] = png(marked)
        neutral = neutral_strip(picture)
        out[f"{row['uid']}.neutral.png"] = png(overlay(neutral))
        for sensitivity in (1.0, 0.5, 0.3):
            out[f"{row['uid']}.ink{int(100 * sensitivity)}.png"] = png(overlay(neutral_strip(picture, sensitivity), tint=True))
        runs = None
        if all(ref in placements and row["uid"] in placements[ref] for ref in REFS):
            runs = classify_runs(neutral, row["text"], [placements[ref][row["uid"]]["x"] for ref in REFS])
        for name in names:
            if row["uid"] in placements[name]:
                out[f"{row['uid']}.{name}.png"] = png(overlay(neutral, row["text"], placements[name][row["uid"]]["x"], runs))
        out[f"{row['uid']}.txt"] = (row["text"] + "\n" + json.dumps({k: row[k] for k in ("split", "width", "height", "origin")})).encode("utf-8")
    return out


@app.local_entrypoint()
def peek(uids: str = "", names: str = "", pool: bool = False, out: str = "results/r3/peek"):
    files = peek_job.remote([u for u in uids.split(",") if u], [n for n in names.split(",") if n], pool)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    print(f"{len(files)} files -> {folder}")


# ---------------------------------------------------------------------------- letter-crop classifier


@app.function(image=image, volumes=volumes, cpu=4.0, memory=8192, timeout=3600, max_containers=10)
def classify_job(name: str, half: str, refs: list[str]) -> dict:
    import torch

    from littera.eval.humboldt_cnn import classify as classify_letters

    torch.set_num_threads(4)
    volume.reload()
    placements = {n: read_placements(n) for n in base_names([name, *refs])}
    result = classify_letters(neutral_lines([half]), placements, name, tuple(refs))
    folder = RUNS / "classify"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name.replace('@', '_at_').replace(':', '_')}.{half}.json").write_text(json.dumps(result), encoding="utf-8")
    volume.commit()
    return result


@app.function(image=image, volumes=volumes, cpu=1.0, memory=2048, timeout=600)
def saved_classify(pairs: list[tuple[str, str]]) -> list[tuple[str, str, dict]]:
    volume.reload()
    out = []
    for name, half in pairs:
        path = RUNS / "classify" / f"{name.replace('@', '_at_').replace(':', '_')}.{half}.json"
        if path.exists():
            out.append((name, half, json.loads(path.read_text(encoding="utf-8"))))
    return out


@app.function(image=image, volumes=volumes, cpu=4.0, memory=8192, timeout=1800)
def connected_job(halves: list[str], refs: list[str]) -> dict[str, list[int]]:
    from littera.eval.humboldt import classify_runs, connected_letters, resolve

    volume.reload()
    lines = neutral_lines(halves)
    placements = {name: read_placements(name) for name in base_names(refs)}
    edges = [resolve(ref, placements, lines) for ref in refs]
    out = {}
    for uid, line in lines.items():
        if all(uid in e for e in edges):
            refs_x = [e[uid] for e in edges]
            out[uid] = sorted(connected_letters(line["text"], classify_runs(line["neutral"], line["text"], refs_x), refs_x))
    return out


def letter_counts(result: dict, splits: tuple[str, ...]) -> dict[str, dict[str, list[float]]]:
    out = {kind: {"num": [], "den": [], "uid": []} for kind in ("all", "connected", "other")}
    for uid in sorted(result):
        row = result[uid]
        if row["split"] not in splits:
            continue
        for kind in out:
            picks = [k for k, c in enumerate(row["connected"]) if kind == "all" or c == (kind == "connected")]
            out[kind]["num"].append(sum(row["ok"][k] for k in picks))
            out[kind]["den"].append(len(picks))
    return out


@app.local_entrypoint()
def classify(names: str = "even,k2-raw-mid,k2-snap-mid,even@k2-snap-mid,pilot-upright,ft2-off,ft2-upright,even@ft2-upright", halves: str = "dev",
             refs: str = "k2-snap-mid,ft2-upright", against: str = "k2-snap-mid", out: str = "", reuse: bool = False):
    """``reuse``: take the per-letter results already on the volume (``classify/<placer>.<half>.json``) instead of training again.
    Only right while the placement file of that name has not changed since."""
    from littera.eval.humboldt import bootstrap, bootstrap_difference, groups_of

    name_list, half_list, ref_list = names.split(","), halves.split(","), refs.split(",")
    jobs = [(name, half, ref_list) for name in name_list for half in half_list]
    merged: dict[str, dict] = {name: {} for name in name_list}
    if reuse:
        saved = saved_classify.remote([(name, half) for name, half, _ in jobs])
        for name, half, result in saved:
            merged[name].update(result)
        done = {(name, half) for name, half, _ in saved}
        jobs = [job for job in jobs if (job[0], job[1]) not in done]
        print(f"{len(done)} saved results reused, {len(jobs)} to run")
    connected_call = connected_job.spawn(half_list, ref_list)
    for (name, _, _), result in zip(jobs, classify_job.starmap(jobs)):
        merged[name].update(result)
    connected = connected_call.get()
    for rows in merged.values():
        for uid, row in rows.items():
            inside = set(connected.get(uid, []))
            row["connected"] = [i in inside for i in row["i"]]
    summary = {"refs": ref_list, "against": against, "recipe": "32 x 32 crops, 3 conv blocks, 15 epochs, 2 folds of lines per half, 2 seeds",
               "placers": {name: {} for name in name_list}}
    for group, splits in groups_of(half_list).items():
        base = letter_counts(merged[against], splits) if against in merged else None
        for name in name_list:
            counts = letter_counts(merged[name], splits)
            cell = {kind: bootstrap(counts[kind]["num"], counts[kind]["den"]) for kind in counts}
            if base is not None and name != against:
                for kind in ("all", "connected"):
                    if base[kind]["den"] == counts[kind]["den"]:
                        cell[f"{kind}_minus_{against}"] = bootstrap_difference(counts[kind]["num"], base[kind]["num"], counts[kind]["den"])
            summary["placers"][name][group] = cell
    path = Path(out or f"results/r3/classify_{halves.replace(',', '_')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    groups = list(groups_of(half_list))
    for kind in ("all", "connected", "other", f"all_minus_{against}", f"connected_minus_{against}"):
        print(f"\n| letter classifier, {kind}, % | " + " | ".join(groups) + " |\n|---|" + "---:|" * len(groups))
        for name in name_list:
            cells = []
            for group in groups:
                value = summary["placers"][name][group].get(kind)
                cells.append("" if value is None else f"{value[0]:.1f} ({value[1]:.1f} to {value[2]:.1f})" + (f" n={value[3]}" if len(value) > 3 and group in half_list else ""))
            print(f"| {name} | " + " | ".join(cells) + " |")
    print(f"\nwrote {path}")


# ---------------------------------------------------------------------------- contact sheets for the blind check


@app.function(image=sheet_image, volumes=volumes, cpu=2.0, memory=8192, timeout=1800)
def sheets_job(a: str, b: str, halves: list[str], count: int, seed: int, disagree: float, per_sheet: int, gate: float) -> dict[str, bytes]:
    import io
    import random

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from littera.eval.humboldt import HALVES, resolve
    from littera.eval.sets import decode_image
    from littera.types import word_spans

    volume.reload()
    wanted = {split for half in halves for split in HALVES[half]}
    rows = {r["uid"]: r for r in read_eval(None, ["uid", "split", "image", "text", "width"]) if r["split"] in wanted}
    placements = {n: read_placements(n) for n in base_names([a, b])}
    edges = {n: resolve(n, placements, rows) for n in (a, b)}
    rng = random.Random(seed)
    found = []
    for uid in sorted(rows):
        if uid not in edges[a] or uid not in edges[b]:
            continue
        if any(0 < placements[n][uid].get("line_fit", 0) < gate for n in base_names([a, b])):
            continue
        text, xa, xb = rows[uid]["text"], edges[a][uid], edges[b][uid]
        widths = [x[i + 1] - x[i] for x in (xa, xb) for i, ch in enumerate(text) if not ch.isspace() and x[i + 1] > x[i]]
        if not widths:
            continue
        letter = float(np.median(widths))
        here = []
        for w0, w1 in word_spans(text):
            for n in (2, 3):
                for i in range(w0, w1 - n + 1):
                    if not text[i : i + n].isalpha():
                        continue
                    apart = max(abs(xa[i] - xb[i]), abs(xa[i + n] - xb[i + n]))
                    if apart > disagree * letter and min(xa[i + n] - xa[i], xb[i + n] - xb[i]) >= 2:
                        here.append({"uid": uid, "split": rows[uid]["split"], "i": i, "n": n, "seq": text[i : i + n], "letter": round(letter, 1),
                                     "apart_letters": round(apart / letter, 2)})
        rng.shuffle(here)
        taken: list[dict] = []
        for cand in here:
            if len(taken) < 2 and all(cand["i"] + cand["n"] <= t["i"] or t["i"] + t["n"] <= cand["i"] for t in taken):
                taken.append(cand)
        found += taken
    candidates = len(found)
    rng.shuffle(found)
    found = found[:count]
    tiles = []
    for cand in found:
        for name in (a, b):
            x = edges[name][cand["uid"]]
            tiles.append({**cand, "placer": name, "x0": x[cand["i"]], "x1": x[cand["i"] + cand["n"]],
                          "view": [min(edges[a][cand["uid"]][cand["i"]], edges[b][cand["uid"]][cand["i"]]) - 3 * cand["letter"],
                                   max(edges[a][cand["uid"]][cand["i"] + cand["n"]], edges[b][cand["uid"]][cand["i"] + cand["n"]]) + 3 * cand["letter"]]})
    rng.shuffle(tiles)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    tile_h, tile_w, label_h, columns = 170, 420, 30, 3
    out, key = {}, {}
    pictures: dict[str, Image.Image] = {}
    for start in range(0, len(tiles), per_sheet):
        part = tiles[start : start + per_sheet]
        sheet = Image.new("RGB", (columns * (tile_w + 12) + 12, -(-len(part) // columns) * (tile_h + label_h + 14) + 12), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for k, tile in enumerate(part):
            tile_id = f"T{start + k + 1:03d}"
            key[tile_id] = {key_name: tile[key_name] for key_name in ("uid", "split", "i", "n", "seq", "placer", "x0", "x1", "apart_letters")}
            if tile["uid"] not in pictures:
                pictures = {tile["uid"]: decode_image(rows[tile["uid"]]["image"]).convert("RGB")}
            crop = pictures[tile["uid"]]
            v0, v1 = max(0, int(tile["view"][0])), min(crop.width, int(round(tile["view"][1])))
            region = np.asarray(crop.crop((v0, 0, v1, crop.height)), dtype=np.float32)
            inked = np.nonzero((region.min(2) < 250).any(1))[0]
            if len(inked):
                region = region[max(0, int(inked[0]) - 4) : int(inked[-1]) + 5]
            s0, s1 = int(round(tile["x0"])) - v0, int(round(tile["x1"])) - v0
            faded = 255 - 0.5 * (255 - region)
            faded[:, max(0, s0) : max(0, s1)] = region[:, max(0, s0) : max(0, s1)]
            view = Image.fromarray(faded.astype(np.uint8))
            scale = min(tile_w / view.width, tile_h / view.height, 4.0)
            view = view.resize((max(1, int(view.width * scale)), max(1, int(view.height * scale))), Image.LANCZOS)
            px = 12 + (k % columns) * (tile_w + 12)
            py = 12 + (k // columns) * (tile_h + label_h + 14)
            draw.text((px, py), f"{tile_id}   {tile['seq']}", font=font, fill=(0, 0, 0))
            sheet.paste(view, (px, py + label_h))
            for edge in (s0, s1):
                ex = px + int(round(edge * scale))
                draw.line([(ex, py + label_h), (ex, py + label_h + 8)], fill=(200, 30, 30), width=2)
                draw.line([(ex, py + label_h + view.height - 8), (ex, py + label_h + view.height)], fill=(200, 30, 30), width=2)
            draw.rectangle((px - 2, py + label_h - 2, px + tile_w + 1, py + label_h + tile_h + 1), outline=(190, 190, 190))
        buffer = io.BytesIO()
        sheet.save(buffer, "PNG")
        out[f"sheet_{start // per_sheet + 1:02d}.png"] = buffer.getvalue()
    meta = {"a": a, "b": b, "halves": halves, "seed": seed, "disagree_letters": disagree, "line_fit_gate": gate, "candidates": candidates,
            "sequences": len(found), "tiles": len(tiles)}
    out["key.json"] = json.dumps({"meta": meta, "tiles": key}, indent=1, ensure_ascii=False).encode("utf-8")
    out["judged.txt"] = ("# one line per tile: id, then right / cut / neighbour / both / unclear\n" + "".join(f"{t} \n" for t in key)).encode("utf-8")
    return out


@app.local_entrypoint()
def sheets(a: str = "k2-snap-mid", b: str = "ft2-upright", halves: str = "dev,report", count: int = 60, seed: int = 5, disagree: float = 0.5,
           per_sheet: int = 12, gate: float = 0.2, out: str = ""):
    """Letter sequences of two and three letters where ``a`` and ``b`` differ by more than ``disagree`` letter widths at either end,
    each shown once per placer (the claimed span in full contrast between red ticks, the rest faded), shuffled."""
    files = sheets_job.remote(a, b, halves.split(","), count, seed, disagree, per_sheet, gate)
    folder = Path(out or f"results/r3/sheets/{a.replace('@', '_at_').replace(':', '_')}_vs_{b.replace('@', '_at_').replace(':', '_')}")
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        if name != "judged.txt" or not (folder / name).exists():
            (folder / name).write_bytes(data)
    print(json.loads(files["key.json"])["meta"], "->", folder)


# ---------------------------------------------------------------------------- occlusion evidence of TrOCR t11

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision==0.23.0")
    .pip_install("transformers==4.57.6", "pyarrow", "pillow", "numpy<2.3", "safetensors", "opencv-python-headless")
    .env({"PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("littera")
)
TROCR = HH / "runs" / "t11_r6g6_AL" / "model"
EVIDENCE = RUNS / "occlusion" / "evidence.json"


@app.function(image=gpu_image, gpu=["L4", "A10G", "T4"], cpu=4.0, memory=16384, volumes=volumes, timeout=3600, max_containers=2)
def evidence_shard(uids: list[str]) -> dict[str, dict]:
    import time

    import torch
    from transformers import AutoTokenizer, VisionEncoderDecoderModel

    from littera.eval.humboldt_occlusion import evidence
    from littera.eval.sets import decode_image

    volume.reload()
    model = VisionEncoderDecoderModel.from_pretrained(str(TROCR)).to("cuda").half().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(TROCR), use_fast=True)
    out, started = {}, time.perf_counter()
    for row in read_eval(uids, ["uid", "image", "text"]):
        if row["text"].strip():
            out[row["uid"]] = evidence(model, tokenizer, decode_image(row["image"]), row["text"])
    print(f"{len(out)} lines, {sum(v['bars'] for v in out.values())} bars in {time.perf_counter() - started:.0f} s on {torch.cuda.get_device_name(0)}")
    return out


@app.function(image=image, volumes=volumes, cpu=2.0, memory=4096, timeout=900)
def occlusion_job(names: list[str], halves: list[str], found: dict | None) -> dict:
    from littera.eval.humboldt import HALVES
    from littera.eval.humboldt_occlusion import score as score_evidence

    volume.reload()
    if found is not None:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(json.dumps(found), encoding="utf-8")
        volume.commit()
    if not EVIDENCE.exists():
        return {"missing": True}
    found = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    wanted = {split for half in halves for split in HALVES[half]}
    lines = {r["uid"]: r for r in read_eval(None, ["uid", "split", "text"]) if r["split"] in wanted}
    placements = {name: read_placements(name) for name in base_names(names)}
    result = score_evidence(lines, placements, found, names, halves)
    result["evidence_lines"] = len(found)
    return result


@app.local_entrypoint()
def occlusion(names: str = "even,k2-raw-mid,k2-snap-mid,even@k2-snap-mid,pilot-upright,ft2-off,ft2-upright,even@ft2-upright", halves: str = "dev",
              out: str = "", rebuild: bool = False, limit: int = 0):
    """The evidence is placer-free and built once for all 1,000 lines (two L4 containers); later calls only score."""
    result = {"missing": True} if rebuild else occlusion_job.remote(names.split(","), halves.split(","), None)
    if result.get("missing"):
        uids = eval_uids.remote(["dev", "report"])
        uids = uids[:limit] if limit else uids
        found = {}
        for part in evidence_shard.map(chunks(uids, 4)):
            found.update(part)
        result = occlusion_job.remote(names.split(","), halves.split(","), found)
    path = Path(out or f"results/r3/occlusion_{halves.replace(',', '_')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    groups = list(next(iter(result["placers"].values())).keys())
    for measure in ("centre_inside_span", "mean_offset_letters", "mean_signed_offset_letters", "tokens_with_evidence"):
        print(f"\n| {measure} | " + " | ".join(groups) + " |\n|---|" + "---:|" * len(groups))
        for name, by_group in result["placers"].items():
            cells = ["" if by_group[g][measure] is None else f"{by_group[g][measure][0]} ({by_group[g][measure][1]} to {by_group[g][measure][2]})" for g in groups]
            print(f"| {name} | " + " | ".join(cells) + " |")
    print(f"\nwrote {path}")


# ---------------------------------------------------------------------------- control: letters named from the span's width alone


@app.function(image=image, volumes=volumes, cpu=2.0, memory=4096, timeout=900)
def width_job(names: list[str], halves: list[str]) -> dict:
    from littera.eval.humboldt import HALVES, bootstrap, groups_of, width_control

    volume.reload()
    wanted = {split for half in halves for split in HALVES[half]}
    lines = {r["uid"]: r for r in read_eval(None, ["uid", "split", "text"]) if r["split"] in wanted}
    placements = {name: read_placements(name) for name in base_names(names)}
    out = {"recipe": "20 quantile bins of width / median width of the line; the commonest letter of the bin", "placers": {}}
    for name in names:
        out["placers"][name] = {}
        for half in halves:
            rows = width_control({uid: line for uid, line in lines.items() if line["split"] in HALVES[half]}, placements, name)
            for group, splits in groups_of([half]).items():
                part = [r for r in rows.values() if r["split"] in splits]
                out["placers"][name][group] = {"width_only": bootstrap([r["ok"] for r in part], [r["n"] for r in part]),
                                               "commonest_letter": bootstrap([r["base"] for r in part], [r["n"] for r in part])}
    return out


@app.local_entrypoint()
def widths(names: str = "even,k2-raw-mid,k2-snap-mid,even@k2-snap-mid,pilot-upright,ft2-off,ft2-upright,even@ft2-upright", halves: str = "dev", out: str = ""):
    result = width_job.remote(names.split(","), halves.split(","))
    path = Path(out or f"results/r3/widths_{halves.replace(',', '_')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, by_group in result["placers"].items():
        print(name.ljust(28), "  ".join(f"{g} {v['width_only'][0]:.1f} (commonest {v['commonest_letter'][0]:.1f})" for g, v in by_group.items()))
    print(f"wrote {path}")
