"""Line pool v1 for littera from the Kurrent VLM staged pool (volume kurrent-raw, read-only) into volume littera:/pool/v1.

Run from the project root:
  PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run --detach modal/pool.py --steps scan,plan,sample,write,finish
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = "/raw"
VOL_DIR = "/vol"
POOL = f"{VOL_DIR}/pool/v1"
BUILD = f"{POOL}/build"
SAMPLE = f"{POOL}/sample"
DECISIONS = f"{RAW_DIR}/build_corpus_v1_5_r5/decisions.parquet"

TRAIN_TOTAL = 300_000
SOURCE_CAP = 30_000
PRINT_CAP = 40_000
VAL_PAGE_SHARE = 0.01
VAL_MAX_LINES = 500
TEST_MAX_LINES = 300
HELDOUT_MAX_LINES = 2_000
MIN_CHARS = 2
WIDTH_PER_CHAR = (0.08, 2.5)
MIN_HEIGHT = 16
MIN_PIXELS_PER_CHAR = 3
BASELINE_SLACK = 0.25
PART_ROWS = 10_000
PART_BYTES = 200_000_000
ROW_GROUP_ROWS = 512
SCAN_FILES_PER_TASK = 60
SAMPLE_PER_GROUP = {"train": 6, "val": 2, "test": 3, "heldout": 12}
SEED = 20261109

ACCEPTED_REASONS = ("", "not_selected", "line_region_twin", "page_shared_with_test", "geometry_only_target", "abbreviation_star")
HELD_SPLITS = ("test", "holdout")
PRINT_SCRIPTS = ("fraktur", "antiqua", "typewriter")
MULTI_HAND_SOURCES = ("zenodo19c", "catmus", "tridis", "scads", "swiss_council", "bullinger", "icfhr2018", "read_icfhr16", "konzilsprotokolle",
                      "mkn", "hanse_xvi", "hanse_xvii", "enhg15", "emanuscripta", "hisclima")
PAGE_NUMBER_SUFFIX = re.compile(r"[_\-. ](?:p|s|f|fol|page|seite)?\d+[rv]?$", re.IGNORECASE)
OTHER_SPACES = re.compile("[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
EDITION_LINE_END_BAR = ("semper",)
EDITION_MARKERS = {"semper": "TRENN|QUOTE|KLAMMER|SKORR|XXXX"}
TEST_SHARE_LIMIT = 0.1
HTR_PAGE_SOURCES = ("konzilsprotokolle",)
HTR_PAGE_SHARE = 0.2
SPACED_PUNCTUATION = re.compile(r"\s[,.;:](?:\s|$)")

app = modal.App("littera-data")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("fonts-dejavu-core")
    .pip_install("pyarrow>=18", "pandas>=2.2", "numpy>=2", "pillow>=11", "pyyaml>=6")
    .add_local_file(ROOT / "src" / "littera" / "data" / "schema.py", "/root/pool_schema.py")
)
RAW = modal.Volume.from_name("kurrent-raw").read_only()
VOL = modal.Volume.from_name("littera", create_if_missing=True)
VOLUMES = {RAW_DIR: RAW, VOL_DIR: VOL}


def parse_extra(value) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def clean_line_text(text: str, source: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\ufeff", "").replace("\u200b", "")
    if source in EDITION_LINE_END_BAR:
        text = text.replace(" |", "").replace("|", "")
    return OTHER_SPACES.sub(" ", text).strip()


def visible_length(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace() and not unicodedata.combining(ch))


def document_of(page_id: str) -> str:
    return PAGE_NUMBER_SUFFIX.sub("", page_id) or page_id


def hand_of(source: str, collection: str, page_id: str, writer: str, script: str) -> str:
    if script in PRINT_SCRIPTS:
        return ""
    if writer:
        return f"{source}/w/{writer}"
    if source == "scads":
        return f"{source}/{page_id or collection}"
    if source == "zenodo19c" and page_id:
        return f"{source}/{document_of(page_id).removeprefix('Low_')}"
    if source in MULTI_HAND_SOURCES and page_id:
        return f"{source}/{collection}/{document_of(page_id)}"
    return f"{source}/{collection}"


def uid_of(sample_id: str) -> str:
    source, collection, page, _unit, index = sample_id.split("/")
    return f"{source}/{collection}/{page}/{index}"


def crop_baseline(points, bbox, width: int, height: int) -> list[list[float]]:
    if not points or bbox is None or len(bbox) != 4:
        return []
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return []
    sx, sy = width / (x2 - x1), height / (y2 - y1)
    moved = [((float(x) - x1) * sx, (float(y) - y1) * sy) for x, y in points]
    slack_x, slack_y = BASELINE_SLACK * height, BASELINE_SLACK * height
    if any(x < -slack_x or x > width + slack_x or y < -slack_y or y > height + slack_y for x, y in moved):
        return []
    return [[round(min(max(x, 0.0), width), 1), round(min(max(y, 0.0), height), 1)] for x, y in moved]


def load_config(config_text: str) -> dict:
    import yaml

    return yaml.safe_load(config_text)["sources"]


def read_decisions(columns: list[str], sources: list[str]):
    import pyarrow.parquet as pq

    filters = [("unit", "=", "line"), ("task", "=", "transcribe"), ("source", "in", sources)]
    return pq.read_table(DECISIONS, columns=columns, filters=filters).to_pandas()


@app.function(image=image, volumes=VOLUMES, cpu=2, memory=24576, timeout=1800)
def scan_tasks(config_text: str) -> list[tuple[str, list[str]]]:
    config = load_config(config_text)
    wanted = [s for s, c in config.items() if c["role"] in ("train", "heldout")]
    files = read_decisions(["file"], wanted).file.drop_duplicates()
    tasks = []
    for folder, group in files.groupby(files.str.rsplit("/", n=1).str[0]):
        names = sorted(Path(RAW_DIR, folder).glob("*.parquet"))
        for k in range(0, len(names), SCAN_FILES_PER_TASK):
            tasks.append((folder, [p.name for p in names[k:k + SCAN_FILES_PER_TASK]]))
    return tasks


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=8192, timeout=3600, max_containers=10)
def scan_folder(task: tuple[str, list[str]]) -> dict:
    import io

    import pandas as pd
    import pyarrow.parquet as pq
    from PIL import Image

    folder, names = task
    frames = []
    for name in names:
        path = Path(RAW_DIR, folder, name)
        parquet = pq.ParquetFile(path)
        available = parquet.schema_arrow.names
        columns = [c for c in ("sample_id", "unit", "task", "target", "img_w", "img_h", "extra") if c in available]
        sizes_missing = "img_w" not in available
        offset = 0
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            frame = parquet.read_row_group(group, columns=columns + (["image"] if sizes_missing else [])).to_pandas()
            frame["row"] = range(offset, offset + count)
            offset += count
            frame = frame[(frame.unit == "line") & (frame.task == "transcribe")]
            if sizes_missing:
                sizes = [Image.open(io.BytesIO(b)).size for b in frame.image]
                frame["img_w"], frame["img_h"] = [s[0] for s in sizes], [s[1] for s in sizes]
                frame = frame.drop(columns=["image"])
            frame["file"] = path.relative_to(RAW_DIR).as_posix()
            frames.append(frame)
    meta = pd.concat(frames, ignore_index=True)
    extras = [parse_extra(e) for e in meta.extra]
    meta["staged_writer"] = [str(e.get("writer_id") or e.get("scribe") or "") for e in extras]
    meta["baseline_points"] = [len(e.get("baseline_page") or []) for e in extras]
    meta = meta.drop(columns=["extra", "unit", "task"]).rename(columns={"target": "staged_target"})
    out = Path(BUILD, "scan", f"{folder.replace('/', '__')}__{names[0].removesuffix('.parquet')}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(out, index=False)
    VOL.commit()
    return {"folder": folder, "files": len(names), "line_rows": len(meta)}


def spread_order(rows, seed: int, priority=None):
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = rows.copy()
    rows["_group"] = rows.hand.where(rows.hand != "", rows.source + "/" + rows.doc)
    pages = rows[["_group", "doc", "page"]].drop_duplicates().copy()
    pages["_page_key"] = rng.random(len(pages))
    pages["_page_rank"] = pages.groupby("_group")._page_key.rank(method="first")
    groups = pages[["_group"]].drop_duplicates().copy()
    groups["_group_key"] = rng.random(len(groups))
    rows = rows.merge(pages, on=["_group", "doc", "page"]).merge(groups, on="_group")
    rows["_line_key"] = rng.random(len(rows))
    rows["_line_rank"] = rows.groupby(["doc", "page"])._line_key.rank(method="first")
    rows["_priority"] = 0 if priority is None else (~rows[priority]).astype(int)
    rows = rows.sort_values(["_priority", "_line_rank", "_page_rank", "_group_key"])
    return rows.drop(columns=["_group", "_page_key", "_page_rank", "_group_key", "_line_key", "_line_rank", "_priority"])


def page_order(rows, seed: int, priority=None):
    import numpy as np

    rng = np.random.default_rng(seed)
    pages = rows.groupby(["doc", "page"], sort=False).agg(lines=("uid", "size"), group=("hand", "first"), source=("source", "first"),
                                                         held=(priority or "uid", "any" if priority else "size")).reset_index()
    pages["group"] = pages.group.where(pages.group != "", pages.source + "/" + pages.doc)
    pages["_key"] = rng.random(len(pages))
    pages["_rank"] = pages.groupby("group")._key.rank(method="first")
    group_keys = {g: k for g, k in zip(pages.group.unique(), rng.random(pages.group.nunique()))}
    pages["_group_key"] = pages.group.map(group_keys)
    pages["_priority"] = (~pages.held.astype(bool)).astype(int) if priority else 0
    return pages.sort_values(["_priority", "_rank", "_group_key"])


def whole_pages(rows, limit: int, seed: int, priority=None):
    chosen, total = [], 0
    for doc, page, lines in page_order(rows, seed, priority)[["doc", "page", "lines"]].itertuples(index=False):
        if total + lines <= limit:
            chosen.append((doc, page))
            total += lines
        if total >= limit:
            break
    keys = set(chosen)
    return rows[[(d, p) in keys for d, p in zip(rows.doc, rows.page)]]


def water_fill(available: dict[str, int], total: int, cap: int) -> int:
    if sum(min(n, cap) for n in available.values()) <= total:
        return cap
    low, high = 0, cap
    while low < high:
        mid = (low + high + 1) // 2
        if sum(min(n, mid) for n in available.values()) <= total:
            low = mid
        else:
            high = mid - 1
    return low


@app.function(image=image, volumes=VOLUMES, cpu=4, memory=49152, timeout=3600)
def make_plan(config_text: str) -> dict:
    import numpy as np
    import pandas as pd

    config = load_config(config_text)
    roles = {s: c["role"] for s, c in config.items()}
    wanted = [s for s, role in roles.items() if role in ("train", "heldout")]
    decisions = read_decisions(["sample_id", "source", "collection", "page_id", "split", "keep", "reason", "file", "row", "writer", "license",
                                "script", "language", "quality", "holdout_set", "image_bytes", "overlay_file"], wanted)
    staged_bytes = decisions[decisions.overlay_file.fillna("") == ""]
    bytes_per_row = staged_bytes.groupby("source").image_bytes.mean().reindex(wanted).fillna(decisions.image_bytes.mean()).to_dict()
    scans = pd.concat([pd.read_parquet(p) for p in sorted(Path(BUILD, "scan").glob("*.parquet"))], ignore_index=True)
    rows = decisions.merge(scans, on=["sample_id", "file", "row"], how="inner", validate="one_to_one")
    report = {"decisions_rows": len(decisions), "matched_rows": len(rows), "unmatched": int(len(decisions) - len(rows))}
    for column in ("collection", "page_id", "writer", "reason", "holdout_set", "staged_writer", "staged_target"):
        rows[column] = rows[column].fillna("")
    rows["uid"] = rows.sample_id.map(uid_of)
    rows["doc"], rows["page"] = rows.collection, rows.page_id
    rows["text"] = [clean_line_text(t, s) for t, s in zip(rows.staged_target, rows.source)]
    rows["visible"] = rows.text.map(visible_length)
    rows["width_per_char"] = rows.img_w / (rows.img_h * rows.visible.clip(lower=1))
    rows["role"] = rows.source.map(roles)
    writer = rows.writer.where(rows.writer != "", rows.staged_writer)
    rows["hand"] = [hand_of(s, c, p, w, k) for s, c, p, w, k in zip(rows.source, rows.collection, rows.page_id, writer, rows.script)]
    rows["origin"] = "kurrent-raw:/" + rows.file + "#" + rows.sample_id
    rows["licence"] = rows.license

    def row_open(source: str, licence: str) -> bool:
        entry = config[source]
        if entry["open_licence"] == "per_row":
            return bool(entry.get("row_licences", {}).get(licence, False))
        return bool(entry["open_licence"]) and licence == entry["licence"]

    rows["open_licence"] = [row_open(s, lic) for s, lic in zip(rows.source, rows.licence)]
    report["licence_mismatch"] = rows[(rows.licence != rows.source.map(lambda s: config[s]["licence"])) & (rows.source.map(lambda s: config[s]["open_licence"]) != "per_row")] \
        .groupby(["source", "licence"]).size().reset_index().values.tolist()
    report["width_per_char_quantiles"] = {s: [round(float(v), 3) for v in g.width_per_char.quantile([0.0005, 0.005, 0.5, 0.995, 0.9995])]
                                          for s, g in rows.groupby("source")}

    rows["drop"] = ""

    def drop(mask, reason) -> None:
        mask = mask & (rows["drop"] == "")
        rows.loc[mask, "drop"] = reason if isinstance(reason, str) else reason[mask]

    drop(~rows.reason.isin(ACCEPTED_REASONS), "build:" + rows.reason)
    drop(rows.text.str.contains("\n"), "multi_line_text")
    for source, markers in EDITION_MARKERS.items():
        drop((rows.source == source) & rows.text.str.contains(markers), "edition_markup")
    htr_source = rows.source.isin(HTR_PAGE_SOURCES)
    spaced_share = (htr_source & rows.text.str.contains(SPACED_PUNCTUATION)).groupby([rows.source, rows.doc, rows.page]).transform("mean")
    drop(htr_source & (spaced_share >= HTR_PAGE_SHARE), "uncorrected_htr_page")
    drop(rows.visible < MIN_CHARS, "short_text")
    drop((rows.width_per_char < WIDTH_PER_CHAR[0]) | (rows.width_per_char > WIDTH_PER_CHAR[1]), "width_per_char")
    drop((rows.img_h < MIN_HEIGHT) | (rows.img_w < MIN_PIXELS_PER_CHAR * rows.visible), "crop_too_small")
    drop(~rows.open_licence & (rows.role == "train"), "licence_not_open")
    drop(~rows.open_licence & (rows.role == "heldout") & rows.source.map(lambda s: config[s]["open_licence"] == "per_row"), "licence_not_open")
    report["dropped"] = {s: g["drop"].value_counts().to_dict() for s, g in rows.groupby("source")}

    held_rows = rows.split.isin(HELD_SPLITS) | (rows.holdout_set != "")
    held_pages = set(zip(rows.source[held_rows], rows.doc[held_rows], rows.page[held_rows]))
    rows["kurrent_held"] = [(s, d, p) in held_pages for s, d, p in zip(rows.source, rows.doc, rows.page)]
    rows["kurrent_split"] = rows.split
    candidates = rows[rows["drop"] == ""].copy()

    drawn, train_pools, per_source = [], {}, {}
    for number, (source, group) in enumerate(sorted(candidates.groupby("source"), key=lambda item: item[0])):
        seed = SEED + number
        info = {"role": roles[source], "candidates": len(group), "candidate_pages": int(group[["doc", "page"]].drop_duplicates().shape[0])}
        if roles[source] == "heldout":
            chosen = whole_pages(group, HELDOUT_MAX_LINES, seed, priority="kurrent_held")
            drawn.append(chosen.assign(split="heldout"))
            per_source[source] = info
            continue
        test_target = min(TEST_MAX_LINES, max(1, int(TEST_SHARE_LIMIT * len(group))))
        held = group[group.kurrent_held]
        test_pages = held
        if len(held) < test_target:
            test_pages = pd.concat([held, whole_pages(group[~group.kurrent_held], test_target - len(held), seed)])
        test = spread_order(test_pages, seed).head(max(test_target, min(len(held), TEST_MAX_LINES)))
        test_keys = set(zip(test_pages.doc, test_pages.page))
        rest = group[[(d, p) not in test_keys for d, p in zip(group.doc, group.page)]]
        pages = page_order(rest, seed + 1000)
        val_count = max(1, round(VAL_PAGE_SHARE * len(pages))) if len(pages) >= 2 else 0
        val_keys = set(zip(pages.doc.head(val_count), pages.page.head(val_count)))
        on_val = np.array([(d, p) in val_keys for d, p in zip(rest.doc, rest.page)], dtype=bool)
        val = spread_order(rest[on_val], seed + 2000).head(VAL_MAX_LINES)
        train_pools[source] = spread_order(rest[~on_val], seed + 3000)
        drawn += [test.assign(split="test"), val.assign(split="val")]
        info.update({"test_pages_from_kurrent_held": bool(len(held)), "test_pages": len(test_keys), "val_pages": val_count, "train_available": int((~on_val).sum())})
        per_source[source] = info

    available = {s: len(pool) for s, pool in train_pools.items()}
    cap = water_fill(available, TRAIN_TOTAL, SOURCE_CAP)
    train = pd.concat([pool.head(min(len(pool), cap)) for pool in train_pools.values()], ignore_index=True)
    is_print = train.script.isin(PRINT_SCRIPTS)
    if is_print.sum() > PRINT_CAP:
        keep_share = PRINT_CAP / is_print.sum()
        print_rows = train[is_print].groupby("source", group_keys=False).apply(lambda g: g.head(int(len(g) * keep_share)))
        train = pd.concat([train[~is_print], print_rows], ignore_index=True)
    drawn.append(train.assign(split="train"))
    plan = pd.concat(drawn, ignore_index=True)
    if plan.uid.duplicated().any():
        raise ValueError(f"duplicate uids: {plan.uid[plan.uid.duplicated()].head().tolist()}")
    page_splits = plan.groupby(["source", "doc", "page"]).split.nunique()
    if (page_splits > 1).any():
        raise ValueError(f"pages in two splits: {page_splits[page_splits > 1].head().index.tolist()}")

    parts = []
    for source, group in plan.groupby("source"):
        group = group.sort_values(["file", "row"]).reset_index(drop=True)
        rows_per_part = int(min(PART_ROWS, max(1_000, PART_BYTES // bytes_per_row[source])))
        group["part"] = [f"lines/{source}/part-{k // rows_per_part:05d}.parquet" for k in range(len(group))]
        group["part_row"] = [k % rows_per_part for k in range(len(group))]
        parts.append(group)
    plan = pd.concat(parts, ignore_index=True)
    columns = ["uid", "source", "doc", "page", "hand", "split", "img_w", "img_h", "text", "script", "language", "licence", "open_licence", "quality",
               "origin", "file", "row", "part", "part_row", "kurrent_split", "kurrent_held", "baseline_points"]
    plan = plan[columns]
    Path(BUILD).mkdir(parents=True, exist_ok=True)
    plan.to_parquet(f"{BUILD}/plan.parquet", index=False)

    counts = plan.groupby(["source", "split"]).size().unstack(fill_value=0)
    for source, info in per_source.items():
        info["drawn"] = {k: int(v) for k, v in counts.loc[source].items() if v} if source in counts.index else {}
    report.update({"train_cap_per_source": int(cap), "train_available": available, "per_source": per_source,
                   "splits": plan.split.value_counts().to_dict(), "print_train": int(plan[(plan.split == "train") & plan.script.isin(PRINT_SCRIPTS)].shape[0]),
                   "parts": int(plan.part.nunique())})
    Path(BUILD, "plan_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    VOL.commit()
    return {k: report[k] for k in ("matched_rows", "unmatched", "train_cap_per_source", "splits", "print_train", "parts")}


def read_records(plan, config: dict) -> list[dict]:
    import bisect
    import io
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from PIL import Image

    def from_file(item) -> list[tuple[int, dict]]:
        file, rows = item
        parquet = pq.ParquetFile(Path(RAW_DIR, file))
        starts, offset = [], 0
        for group in range(parquet.num_row_groups):
            starts.append(offset)
            offset += parquet.metadata.row_group(group).num_rows
        wanted: dict[int, list] = {}
        for position, row in zip(rows.index, rows.row):
            group = bisect.bisect_right(starts, row) - 1
            wanted.setdefault(group, []).append((position, row - starts[group]))
        out = []
        for group, items in wanted.items():
            table = parquet.read_row_group(group, columns=["sample_id", "image", "extra", "bbox"])
            ids, images, extras, boxes = (table.column(c) for c in ("sample_id", "image", "extra", "bbox"))
            for position, local in items:
                planned = plan.loc[position]
                if ids[local].as_py() != planned.origin.split("#", 1)[1]:
                    raise ValueError(f"row mismatch in {file} row {planned.row}")
                data = images[local].as_py()
                with Image.open(io.BytesIO(data)) as crop:
                    width, height = crop.size
                baseline = []
                if config[planned.source].get("baseline", True):
                    baseline = crop_baseline(parse_extra(extras[local].as_py()).get("baseline_page"), boxes[local].as_py(), width, height)
                out.append((position, {
                    "uid": planned.uid, "source": planned.source, "doc": planned.doc, "page": planned.page, "hand": planned.hand, "split": planned.split,
                    "image": data, "width": width, "height": height, "text": planned.text, "script": planned.script, "language": planned.language,
                    "licence": planned.licence, "open_licence": bool(planned.open_licence), "quality": planned.quality, "polygon": [],
                    "baseline": baseline, "origin": planned.origin,
                }))
        return out

    with ThreadPoolExecutor(8) as pool:
        results = [r for part in pool.map(from_file, plan.groupby("file")) for r in part]
    order = {position: k for k, position in enumerate(plan.index)}
    return [record for _, record in sorted(results, key=lambda item: order[item[0]])]


def write_table(records: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pool_schema import POOL_SCHEMA

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=POOL_SCHEMA), path, row_group_size=ROW_GROUP_ROWS, compression="zstd")


def index_table(records: list[dict], part: str):
    import pyarrow as pa
    from pool_schema import INDEX_SCHEMA

    rows = [{**{k: v for k, v in r.items() if k != "image"}, "part": part, "part_row": k} for k, r in enumerate(records)]
    return pa.Table.from_pylist(rows, schema=INDEX_SCHEMA)


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=8192, timeout=3600, max_containers=10)
def write_part(part: str, config_text: str) -> dict:
    import pandas as pd
    import pyarrow.parquet as pq

    config = load_config(config_text)
    plan = pd.read_parquet(f"{BUILD}/plan.parquet", filters=[("part", "=", part)]).sort_values("part_row")
    records = read_records(plan, config)
    write_table(records, Path(POOL, part))
    index_path = Path(BUILD, "index", part.replace("/", "__"))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(index_table(records, part), index_path)
    VOL.commit()
    return {"part": part, "rows": len(records), "with_baseline": sum(1 for r in records if r["baseline"])}


def contact_sheet(records: list[dict], path: Path) -> None:
    import io

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    width = 1400
    blocks = []
    for record in records:
        crop = Image.open(io.BytesIO(record["image"])).convert("RGB")
        scale = min(1.0, width / crop.width)
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
        draw = ImageDraw.Draw(crop)
        if len(record["baseline"]) >= 2:
            draw.line([(x * scale, y * scale) for x, y in record["baseline"]], fill=(230, 20, 20), width=2)
        block = Image.new("RGB", (width, crop.height + 30), "white")
        block.paste(crop, (0, 0))
        ImageDraw.Draw(block).text((4, crop.height + 4), f"{record['split']}  {record['text']}"[:150], fill=(0, 0, 160), font=font)
        blocks.append(block)
    sheet = Image.new("RGB", (width, sum(b.height + 6 for b in blocks) + 6), (200, 200, 200))
    y = 6
    for block in blocks:
        sheet.paste(block, (0, y))
        y += block.height + 6
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


@app.function(image=image, volumes=VOLUMES, cpu=2, memory=16384, timeout=3600)
def make_sample(config_text: str) -> dict:
    import shutil

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    config = load_config(config_text)
    plan = pd.read_parquet(f"{BUILD}/plan.parquet")
    picked = []
    for (source, split), group in plan.groupby(["source", "split"]):
        picked.append(group.sample(min(len(group), SAMPLE_PER_GROUP[split]), random_state=SEED))
    sample = pd.concat(picked).sort_values(["source", "split", "uid"])
    shutil.rmtree(SAMPLE, ignore_errors=True)
    indexes = []
    for source, group in sample.groupby("source"):
        records = read_records(group, config)
        part = f"lines/{source}/part-00000.parquet"
        write_table(records, Path(SAMPLE, part))
        indexes.append(index_table(records, part))
        contact_sheet(records, Path(SAMPLE, "sheets", f"{source}.png"))
    pq.write_table(pa.concat_tables(indexes), f"{SAMPLE}/index.parquet")
    VOL.commit()
    return {"rows": len(sample), "sources": int(sample.source.nunique())}


@app.function(image=image, volumes=VOLUMES, cpu=4, memory=32768, timeout=3600)
def finish() -> dict:
    import collections

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    index = pa.concat_tables([pq.read_table(p) for p in sorted(Path(BUILD, "index").glob("*.parquet"))])
    pq.write_table(index, f"{POOL}/index.parquet", compression="zstd")
    frame = index.select(["uid", "source", "doc", "page", "hand", "split", "width", "height", "text", "script", "open_licence", "baseline"]).to_pandas()
    frame["chars"] = frame.text.str.len()
    frame["has_baseline"] = frame.baseline.map(len) > 0
    frame["print"] = frame.script.isin(PRINT_SCRIPTS)

    def stats(group) -> dict:
        return {"rows": len(group), "pages": int(group[["source", "doc", "page"]].drop_duplicates().shape[0]),
                "hands": int(group.hand[group.hand != ""].nunique()), "chars": int(group.chars.sum()),
                "print_rows": int(group["print"].sum()), "with_baseline": int(group.has_baseline.sum()), "open_licence_rows": int(group.open_licence.sum())}

    bins = list(range(0, 161, 16)) + [10_000]
    summary = {
        "pool": "littera:/pool/v1", "rows": len(frame),
        "splits": {split: stats(g) for split, g in frame.groupby("split")},
        "sources": {source: {split: stats(g) for split, g in group.groupby("split")} for source, group in frame.groupby("source")},
        "height": {split: {"quantiles": {str(q): float(g.height.quantile(q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
                           "histogram_16px": {f"{lo}-{hi}": int(c) for lo, hi, c in zip(bins[:-1], bins[1:], np.histogram(g.height, bins=bins)[0])}}
                   for split, g in frame.groupby("split")},
        "width": {split: {str(q): float(g.width.quantile(q)) for q in (0.01, 0.5, 0.99)} for split, g in frame.groupby("split")},
        "characters": {split: dict(collections.Counter("".join(g.text)).most_common()) for split, g in frame.groupby("split")},
    }
    Path(POOL, "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    VOL.commit()
    return {"rows": summary["rows"], "splits": {k: v["rows"] for k, v in summary["splits"].items()}}


@app.function(image=image, volumes=VOLUMES, cpu=0.5, memory=2048, timeout=6 * 3600)
def build(config_text: str, steps: str) -> dict:
    import shutil

    steps_wanted = steps.split(",")
    result = {}
    if "scan" in steps_wanted:
        shutil.rmtree(Path(BUILD, "scan"), ignore_errors=True)
        VOL.commit()
        tasks = scan_tasks.remote(config_text)
        scanned = list(scan_folder.map(tasks))
        result["scan"] = {"tasks": len(tasks), "line_rows": sum(s["line_rows"] for s in scanned)}
        print("scan", result["scan"], flush=True)
    VOL.reload()
    if "plan" in steps_wanted:
        result["plan"] = make_plan.remote(config_text)
        print("plan", json.dumps(result["plan"]), flush=True)
    VOL.reload()
    if "sample" in steps_wanted:
        result["sample"] = make_sample.remote(config_text)
        print("sample", result["sample"], flush=True)
    if "write" in steps_wanted:
        import pandas as pd

        VOL.reload()
        for stale in ("index.parquet", "summary.json"):
            Path(POOL, stale).unlink(missing_ok=True)
        for stale in ("lines", "build/index"):
            shutil.rmtree(Path(POOL, stale), ignore_errors=True)
        VOL.commit()
        parts = sorted(pd.read_parquet(f"{BUILD}/plan.parquet", columns=["part"]).part.unique())
        written = list(write_part.map(parts, kwargs={"config_text": config_text}))
        result["write"] = {"parts": len(written), "rows": sum(w["rows"] for w in written), "with_baseline": sum(w["with_baseline"] for w in written)}
        print("write", result["write"], flush=True)
    VOL.reload()
    if "finish" in steps_wanted:
        result["finish"] = finish.remote()
        print("finish", result["finish"], flush=True)
    return result


@app.local_entrypoint()
def main(steps: str = "scan,plan,sample,write,finish"):
    config_text = (ROOT / "configs" / "sources.yaml").read_text(encoding="utf-8")
    print(json.dumps(build.remote(config_text, steps), ensure_ascii=False, indent=1))
