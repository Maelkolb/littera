"""Laubmann held-out lines for the littera pool, cut from the Kurrent staged region crops (volume kurrent-raw, read-only).

Run from the project root, in this order:
  PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/laubmann.py::survey          # regions whose line boxes match the text
  PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/laubmann.py::cut --trial     # contact sheets of a few pages only
  PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/laubmann.py::cut             # lines of 4,000 planned lines -> build/laubmann
  PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/laubmann.py::check           # k2 and McCATMuS fit of own vs neighbour texts
  PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/laubmann.py::write --drop drop_regions.json   # part + its index part
  PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/laubmann.py::backup          # check build/index against index.parquet, back up
  PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/pool.py --steps finish       # index.parquet + summary.json from build/index
  PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/laubmann.py::verify
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

RAW_DIR = "/raw"
VOL_DIR = "/vol"
POOL = Path(VOL_DIR, "pool", "v1")
WORK = POOL / "build" / "laubmann"
BACKUP = POOL / "build" / "backup_2026-11-09_pre_laubmann"
PART = "lines/laubmann/part-00000.parquet"
STAGED = Path(RAW_DIR, "staged", "laubmann")
DECISIONS = f"{RAW_DIR}/build_corpus_v1_5_r5/decisions.parquet"
META_COLUMNS = ["sample_id", "collection", "page_id", "unit", "task", "target", "convention", "script", "language", "quality", "split",
                "license", "img_w", "img_h", "geom_lines", "holdout_set", "extra"]
MAX_LINES = 2_000
SEED = 20261109
SHEET_LINES = 40

app = modal.App("littera-data")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("fonts-dejavu-core")
    .pip_install("pyarrow>=18", "pandas>=2.2", "numpy>=2", "pillow>=11")
    .add_local_python_source("littera")
)
reader_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("kraken>=7.1,<7.2", "pyarrow", "fonttools", "opencv-python-headless", "pillow", "numpy")
    .env({"PYTHONUTF8": "1", "LITTERA_K2_MODEL": "/hh/runs/aligner-k2/model/best.safetensors",
          "LITTERA_MCCATMUS_MODEL": "/vol/models/mccatmus/McCATMuS_nfd_nofix_V1.mlmodel"})
    .pip_install("pandas")
    .add_local_python_source("littera")
)
RAW = modal.Volume.from_name("kurrent-raw").read_only()
VOL = modal.Volume.from_name("littera")
HUMBOLDT = modal.Volume.from_name("humboldt-htr").read_only()
VOLUMES = {RAW_DIR: RAW, VOL_DIR: VOL}


def staged_regions():
    import pandas as pd
    import pyarrow.parquet as pq

    frames = []
    for path in sorted(STAGED.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        columns = [c for c in META_COLUMNS if c in parquet.schema_arrow.names]
        frame = parquet.read(columns=columns).to_pandas()
        frame["file"] = path.relative_to(RAW_DIR).as_posix()
        frame["row"] = range(len(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@app.function(image=image, volumes=VOLUMES, cpu=2, memory=16384, timeout=1800)
def survey_job() -> dict:
    import collections

    import pyarrow.parquet as pq

    meta = staged_regions()
    regions = meta[(meta.unit == "region") & (meta.task == "transcribe")].copy()
    extras = [json.loads(e or "{}") for e in regions.extra]
    regions["line_boxes"] = [len(e.get("line_boxes") or []) for e in extras]
    regions["target_lines"] = [t.count("\n") + 1 for t in regions.target]
    regions["target_origin"] = [e.get("target_origin", "") for e in extras]
    regions["region_type"] = [e.get("region_type", "") for e in extras]
    matching = regions[(regions.line_boxes > 0) & (regions.line_boxes == regions.target_lines)]
    decisions = pq.read_table(DECISIONS, columns=["sample_id", "source", "reason"], filters=[("source", "=", "laubmann")]).to_pandas()
    matching = matching.merge(decisions[["sample_id", "reason"]].rename(columns={"reason": "kurrent_reason"}), on="sample_id", how="left")
    matching["kurrent_reason"] = matching.kurrent_reason.fillna("")
    out = {"staged_rows": len(meta), "regions": len(regions), "with_line_boxes": int((regions.line_boxes > 0).sum()),
           "matching": len(matching), "matching_lines": int(matching.target_lines.sum()),
           "matching_pages": int(matching[["collection", "page_id"]].drop_duplicates().shape[0]),
           "split_quality": {f"{s}/{q}": int(n) for (s, q), n in matching.groupby(["split", "quality"]).size().items()},
           "kurrent_reason": {k: int(v) for k, v in collections.Counter(matching.kurrent_reason).items()}}
    WORK.mkdir(parents=True, exist_ok=True)
    matching.to_parquet(WORK / "regions.parquet", index=False)
    VOL.commit()
    return out


@app.local_entrypoint()
def survey():
    print(json.dumps(survey_job.remote(), indent=1, ensure_ascii=False, default=str))


def choose_pages(regions, limit: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    pages = regions.groupby(["collection", "page_id"]).agg(lines=("target_lines", "sum"), held=("split", lambda s: bool((s != "train").any())))
    pages = pages.reset_index()
    pages["key"] = rng.random(len(pages))
    pages["rank"] = pages.groupby("collection").key.rank(method="first")
    collection_keys = dict(zip(sorted(pages.collection.unique()), rng.random(pages.collection.nunique())))
    pages["collection_key"] = pages.collection.map(collection_keys)
    pages["priority"] = (~pages.held).astype(int)
    pages = pages.sort_values(["priority", "rank", "collection_key"])
    chosen, total = [], 0
    for collection, page_id, lines in pages[["collection", "page_id", "lines"]].itertuples(index=False):
        if total + lines <= limit:
            chosen.append((collection, page_id))
            total += lines
        if total >= limit:
            break
    return chosen


def whole_pages_up_to(rows: list[dict], page_order: list[tuple[str, str]], limit: int) -> list[dict]:
    by_page: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_page.setdefault((row["doc"], row["page"]), []).append(row)
    kept: list[dict] = []
    for key in page_order:
        page_rows = by_page.get(key, [])
        if len(kept) + len(page_rows) <= limit:
            kept += page_rows
        if len(kept) >= limit:
            break
    return kept


def read_images(regions) -> dict[str, bytes]:
    import bisect

    import pyarrow.parquet as pq

    images = {}
    for file, group in regions.groupby("file"):
        parquet = pq.ParquetFile(Path(RAW_DIR, file))
        starts, offset = [], 0
        for number in range(parquet.num_row_groups):
            starts.append(offset)
            offset += parquet.metadata.row_group(number).num_rows
        wanted: dict[int, list[tuple[str, int]]] = {}
        for sample_id, row in zip(group.sample_id, group.row):
            number = bisect.bisect_right(starts, row) - 1
            wanted.setdefault(number, []).append((sample_id, row - starts[number]))
        for number, items in wanted.items():
            table = parquet.read_row_group(number, columns=["sample_id", "image"])
            ids, data = table.column("sample_id"), table.column("image")
            for sample_id, local in items:
                if ids[local].as_py() != sample_id:
                    raise ValueError(f"row mismatch in {file}: {sample_id}")
                images[sample_id] = data[local].as_py()
    return images


def contact_sheet(rows: list[dict], path: Path, zoom: float = 1.5) -> None:
    import io

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    width = 1700
    blocks = []
    for row in rows:
        crop = Image.open(io.BytesIO(row["image"])).convert("RGB")
        scale = min(zoom, (width - 8) / crop.width)
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
        block = Image.new("RGB", (width, crop.height + 44), "white")
        block.paste(crop, (4, 2))
        draw = ImageDraw.Draw(block)
        draw.text((4, crop.height + 5), row["text"][:160], fill=(0, 0, 150), font=font)
        draw.text((4, crop.height + 27), f"{row['uid']}  {row['quality']}  {row['width']}x{row['height']}", fill=(110, 110, 110), font=small)
        blocks.append(block)
    sheet = Image.new("RGB", (width, sum(b.height + 6 for b in blocks) + 6), (190, 190, 190))
    y = 6
    for block in blocks:
        sheet.paste(block, (0, y))
        y += block.height + 6
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


@app.function(image=image, volumes=VOLUMES, cpu=2, memory=16384, timeout=3600)
def cut_job(trial: bool = False, limit: int = MAX_LINES) -> dict:
    import collections
    import io

    import pandas as pd

    from littera.data.laubmann import ACCEPTED_REASONS, LEFT_OUT_COLLECTIONS, line_rows, region_lines

    regions = pd.read_parquet(WORK / "regions.parquet")
    report = {"regions": len(regions)}
    regions = regions[regions.kurrent_reason.isin(ACCEPTED_REASONS) & ~regions.collection.isin(LEFT_OUT_COLLECTIONS)]
    report["regions_accepted"] = len(regions)
    chosen = choose_pages(regions, 60 if trial else 2 * limit, SEED)
    keys = set(chosen)
    picked = regions[[(c, p) in keys for c, p in zip(regions.collection, regions.page_id)]].copy()
    order = {key: k for k, key in enumerate(chosen)}
    picked["page_order"] = [order[(c, p)] for c, p in zip(picked.collection, picked.page_id)]
    picked = picked.sort_values(["page_order", "sample_id"])
    images = read_images(picked)
    rows, dropped, rejected = [], collections.Counter(), []
    for region in picked.to_dict("records"):
        region["image"] = images[region["sample_id"]]
        made, lost = line_rows(region, rejected if trial else None)
        texts = region_lines(region["target"])
        for row in made:
            number = int(row["origin"].split("#line=")[1].split(";")[0])
            row.update(region=region["sample_id"], page_order=region["page_order"], line_number=number,
                       prev_text=texts[number - 2] if number > 1 else "", next_text=texts[number] if number < len(texts) else "")
        rows += made
        dropped.update(lost)
    report.update({"pages_cut": len(chosen), "regions_cut": len(picked), "planned_lines": int(picked.target_lines.sum()), "lines": len(rows),
                   "dropped": dict(dropped)})
    if trial:
        sheets = WORK / "sheets_trial"
        for number in range(0, len(rows), SHEET_LINES):
            contact_sheet(rows[number:number + SHEET_LINES], sheets / f"sample_{number // SHEET_LINES}.png")

        def as_row(reason, text, band):
            buffer = io.BytesIO()
            band.save(buffer, "JPEG")
            return {"image": buffer.getvalue(), "text": text, "uid": reason, "quality": "", "width": band.width, "height": band.height}

        if rejected:
            contact_sheet([as_row(*item) for item in rejected[:SHEET_LINES]], sheets / "rejected.png", zoom=1.0)
    else:
        pd.DataFrame(rows).to_parquet(WORK / "cut_lines.parquet", index=False)
        (WORK / "cut_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    VOL.commit()
    return report


@app.local_entrypoint()
def cut(trial: bool = False, limit: int = MAX_LINES):
    print(json.dumps(cut_job.remote(trial, limit), indent=1, ensure_ascii=False))


@app.function(image=reader_image, volumes={VOL_DIR: VOL, "/hh": HUMBOLDT}, cpu=2, memory=4096, timeout=3600, max_containers=4)
def check_shard(shard: int, shards: int) -> int:
    import io

    import pandas as pd
    from PIL import Image

    from littera.eval.placers import make_placer

    lines = pd.read_parquet(WORK / "cut_lines.parquet", columns=["uid", "image", "text", "prev_text", "next_text"]).iloc[shard::shards]
    placers = {"k2": make_placer("kraken-k2"), "mcc": make_placer("kraken-mccatmus")}
    out = []
    for row in lines.itertuples():
        image = Image.open(io.BytesIO(row.image)).convert("L")
        record = {"uid": row.uid}
        for name, placer in placers.items():
            log_probs, _ = placer.run(image)
            for which, text in (("own", row.text), ("prev", row.prev_text), ("next", row.next_text)):
                labels, _ = placer.labels(text)
                usable = bool(text) and 0 < len(labels) <= log_probs.shape[0]
                record[f"{name}_{which}"] = placer.line_fit(log_probs, labels) if usable else None
        out.append(record)
    folder = WORK / "check"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(folder / f"part-{shard:02d}.parquet", index=False)
    VOL.commit()
    return len(out)


@app.local_entrypoint()
def check(shards: int = 4):
    print(sum(check_shard.starmap([(k, shards) for k in range(shards)])))


@app.function(image=image, volumes={VOL_DIR: VOL}, cpu=2, memory=16384, timeout=3600)
def write_job(drop_regions: list[str], limit: int = MAX_LINES) -> dict:
    import collections
    import random

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from littera.data.schema import INDEX_SCHEMA, POOL_SCHEMA

    VOL.reload()
    lines = pd.read_parquet(WORK / "cut_lines.parquet")
    report = {"cut_lines": len(lines), "regions_dropped": len(set(drop_regions)),
              "lines_in_dropped_regions": int(lines.region.isin(drop_regions).sum())}
    lines = lines[~lines.region.isin(drop_regions)].sort_values(["page_order", "region", "line_number"])
    order = lines[["doc", "page"]].drop_duplicates()
    rows = [{k: r[k] for k in POOL_SCHEMA.names} for r in lines.to_dict("records")]
    rows = whole_pages_up_to(rows, list(zip(order.doc, order.page)), limit)
    for row in rows:
        row["polygon"], row["baseline"] = [], []
    uids = [r["uid"] for r in rows]
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate uids")
    report.update({"lines": len(rows), "pages": len({(r["doc"], r["page"]) for r in rows}), "volumes": len({r["doc"] for r in rows}),
                   "quality": dict(collections.Counter(r["quality"] for r in rows)),
                   "script": dict(collections.Counter(r["script"] for r in rows))})
    (POOL / PART).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=POOL_SCHEMA), POOL / PART, row_group_size=512, compression="zstd")
    index_rows = [{**{k: v for k, v in r.items() if k != "image"}, "part": PART, "part_row": k} for k, r in enumerate(rows)]
    pq.write_table(pa.Table.from_pylist(index_rows, schema=INDEX_SCHEMA), POOL / "build" / "index" / PART.replace("/", "__"))
    sample = random.Random(SEED).sample(rows, min(len(rows), 4 * SHEET_LINES))
    for number in range(0, len(sample), SHEET_LINES):
        contact_sheet(sample[number:number + SHEET_LINES], WORK / "sheets" / f"final_{number // SHEET_LINES}.png")
    (WORK / "write_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    VOL.commit()
    return report


@app.local_entrypoint()
def write(drop: str = ""):
    regions = json.loads(Path(drop).read_text(encoding="utf-8")) if drop else []
    print(json.dumps(write_job.remote(regions), indent=1, ensure_ascii=False))


@app.function(image=image, volumes={VOL_DIR: VOL}, cpu=2, memory=16384, timeout=1800)
def backup_job() -> dict:
    import shutil

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    VOL.reload()
    current = pq.read_table(POOL / "index.parquet")
    parts = [p for p in sorted((POOL / "build" / "index").glob("*.parquet")) if "laubmann" not in p.name]
    rebuilt = pa.concat_tables([pq.read_table(p) for p in parts])
    if "laubmann" in set(current["source"].to_pylist()):
        raise ValueError("index.parquet already has laubmann rows")
    same = current.num_rows == rebuilt.num_rows and current.select(["uid", "part", "part_row"]).equals(rebuilt.select(["uid", "part", "part_row"]))
    if not same:
        raise ValueError(f"build/index differs from index.parquet ({rebuilt.num_rows} vs {current.num_rows} rows)")
    BACKUP.mkdir(parents=True, exist_ok=True)
    for name in ("index.parquet", "summary.json"):
        if not (BACKUP / name).exists():
            shutil.copy2(POOL / name, BACKUP / name)
    VOL.commit()
    return {"rows": current.num_rows, "same_as_build_index": same, "backup": str(BACKUP),
            "heldout": int(pc.sum(pc.equal(current["split"], "heldout")).as_py())}


@app.local_entrypoint()
def backup():
    print(json.dumps(backup_job.remote(), indent=1))


@app.function(image=image, volumes={VOL_DIR: VOL}, cpu=2, memory=16384, timeout=1800)
def verify_job() -> dict:
    import random

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    VOL.reload()
    index = pq.read_table(POOL / "index.parquet")
    old = pq.read_table(BACKUP / "index.parquet", columns=["uid"])
    uids = index["uid"].to_pylist()
    laubmann = index.filter(pc.equal(index["source"], "laubmann"))
    problems = []
    if len(set(uids)) != len(uids):
        problems.append("duplicate uids")
    if index.num_rows != old.num_rows + laubmann.num_rows:
        problems.append("row count is not old + laubmann")
    if set(laubmann["split"].to_pylist()) != {"heldout"} or any(laubmann["open_licence"].to_pylist()):
        problems.append("laubmann rows not heldout / not closed")
    closed = index.filter(pc.invert(index["open_licence"]))
    if set(closed["split"].to_pylist()) - {"heldout"}:
        problems.append("non-open rows outside heldout")
    pages = index.select(["source", "doc", "page", "split"]).to_pandas().groupby(["source", "doc", "page"]).split.nunique()
    if (pages > 1).any():
        problems.append("page in two splits")
    positions = [k for k, uid in enumerate(uids) if uid.startswith("laubmann/")] + random.Random(1).sample(range(index.num_rows), 300)
    parts, rows_in_part = index["part"].to_pylist(), index["part_row"].to_pylist()
    by_part: dict[str, list[tuple[int, str]]] = {}
    for k in positions:
        by_part.setdefault(parts[k], []).append((rows_in_part[k], uids[k]))
    checked = 0
    for part, items in by_part.items():
        found = pq.read_table(POOL / part, columns=["uid"])["uid"].to_pylist()
        for row, uid in items:
            checked += 1
            if found[row] != uid:
                problems.append(f"{part}:{row} has {found[row]} not {uid}")
                break
    summary = json.loads((POOL / "summary.json").read_text(encoding="utf-8"))
    return {"rows": index.num_rows, "laubmann": laubmann.num_rows, "checked_pointers": checked, "problems": problems,
            "summary_laubmann": summary["sources"].get("laubmann"), "summary_heldout": summary["splits"]["heldout"]}


@app.local_entrypoint()
def verify():
    print(json.dumps(verify_job.remote(), indent=1, ensure_ascii=False))
