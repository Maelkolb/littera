from __future__ import annotations

import io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

KINDS = ("word", "letter", "synthetic", "mismatch")

POINTS = pa.list_(pa.list_(pa.float32()))
GOLD_WORD = pa.struct([("i0", pa.int32()), ("i1", pa.int32()), ("x0", pa.float32()), ("x1", pa.float32())])
GOLD_CUT = pa.struct([("i", pa.int32()), ("x", pa.float32())])

SCHEMA = pa.schema([
    ("uid", pa.string()),
    ("source", pa.string()),
    ("doc", pa.string()),
    ("page", pa.string()),
    ("hand", pa.string()),
    ("split", pa.string()),
    ("image", pa.binary()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("text", pa.string()),
    ("script", pa.string()),
    ("language", pa.string()),
    ("licence", pa.string()),
    ("open_licence", pa.bool_()),
    ("quality", pa.string()),
    ("polygon", POINTS),
    ("baseline", POINTS),
    ("origin", pa.string()),
    ("kind", pa.string()),
    ("gold_words", pa.list_(GOLD_WORD)),
    ("gold_cuts", pa.list_(GOLD_CUT)),
    ("true_text", pa.string()),
    ("label", pa.int8()),
    ("variant", pa.string()),
])

DEFAULTS = {
    "hand": "", "split": "eval", "script": "", "language": "", "licence": "", "open_licence": False, "quality": "",
    "polygon": [], "baseline": [], "origin": "", "gold_words": [], "gold_cuts": [], "true_text": None, "label": None,
    "variant": "",
}


def encode_image(image: Image.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buffer = io.BytesIO()
    if fmt == "JPEG":
        image = image.convert("RGB") if image.mode not in ("L", "RGB") else image
        image.save(buffer, "JPEG", quality=quality)
    else:
        image.save(buffer, fmt)
    return buffer.getvalue()


def decode_image(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def make_row(**fields) -> dict:
    row = {**DEFAULTS, **fields}
    missing = [name for name in SCHEMA.names if name not in row]
    if missing:
        raise ValueError(f"row is missing {missing}")
    if row["kind"] not in KINDS:
        raise ValueError(f"unknown kind {row['kind']!r}")
    return {name: row[name] for name in SCHEMA.names}


def write_set(rows: Iterable[dict], path: str | Path) -> int:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([make_row(**row) for row in rows], schema=SCHEMA)
    pq.write_table(table, path, row_group_size=256)
    return len(rows)


def read_set(path: str | Path, columns: list[str] | None = None, limit: int | None = None) -> list[dict]:
    table = pq.read_table(path, columns=columns)
    if limit is not None:
        table = table.slice(0, limit)
    return table.to_pylist()


def iter_set(path: str | Path, columns: list[str] | None = None, batch_size: int = 64) -> Iterator[dict]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def set_path(name_or_path: str, root: str | Path = "data/eval") -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix == ".parquet":
        return candidate
    if candidate.is_dir():
        return candidate / "lines.parquet"
    return Path(root) / name_or_path / "lines.parquet"


def write_readme(directory: str | Path, title: str, body: str, stats: dict | None = None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    text = f"# {title}\n\n{body.strip()}\n"
    if stats:
        text += "\n```json\n" + json.dumps(stats, indent=1, ensure_ascii=False) + "\n```\n"
    (directory / "README.md").write_text(text, encoding="utf-8")


def set_stats(rows: list[dict]) -> dict:
    stats: dict = {"rows": len(rows), "kinds": {}, "sources": {}}
    for row in rows:
        stats["kinds"][row["kind"]] = stats["kinds"].get(row["kind"], 0) + 1
        stats["sources"][row["source"]] = stats["sources"].get(row["source"], 0) + 1
    stats["gold_words"] = sum(len(row["gold_words"] or []) for row in rows)
    stats["gold_cuts"] = sum(len(row["gold_cuts"] or []) for row in rows)
    return stats
