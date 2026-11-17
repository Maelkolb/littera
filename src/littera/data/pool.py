from __future__ import annotations

import io
import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

from littera.data.schema import INDEX_SCHEMA, POOL_SCHEMA, SPLITS

VOLUME_POOL = Path("/vol/pool/v1")
LOCAL_SAMPLE = Path(__file__).resolve().parents[3] / "data" / "pool_sample"


def pool_root(root: str | Path | None = None) -> Path:
    if root:
        return Path(root)
    if os.environ.get("LITTERA_POOL"):
        return Path(os.environ["LITTERA_POOL"])
    if (VOLUME_POOL / "index.parquet").exists():
        return VOLUME_POOL
    return LOCAL_SAMPLE


def _select(table: pa.Table, split: str | None, sources: Iterable[str] | None) -> pa.Table:
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
        table = table.filter(pc.equal(table["split"], split))
    if sources is not None:
        table = table.filter(pc.is_in(table["source"], pa.array(list(sources), pa.string())))
    return table


def read_index(root: str | Path | None = None, split: str | None = None, sources: Iterable[str] | None = None,
               columns: Sequence[str] | None = None) -> pa.Table:
    wanted = None if columns is None else list(dict.fromkeys([*columns, "split", "source"]))
    table = _select(pq.read_table(pool_root(root) / "index.parquet", columns=wanted), split, sources)
    return table if columns is None else table.select(list(columns))


def list_sources(root: str | Path | None = None) -> list[str]:
    return sorted(p.name for p in (pool_root(root) / "lines").iterdir() if p.is_dir())


def part_files(root: str | Path | None = None, sources: Iterable[str] | None = None) -> list[Path]:
    base = pool_root(root) / "lines"
    names = sorted(sources) if sources is not None else sorted(p.name for p in base.iterdir() if p.is_dir())
    return [path for name in names for path in sorted((base / name).glob("part-*.parquet"))]


def iter_rows(root: str | Path | None = None, split: str | None = None, sources: Iterable[str] | None = None,
              columns: Sequence[str] | None = None, batch_size: int = 512) -> Iterator[dict]:
    wanted = None if columns is None else list(dict.fromkeys([*columns, "split", "source"]))
    for path in part_files(root, sources):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=wanted):
            table = _select(pa.Table.from_batches([batch]), split, None)
            if columns is not None:
                table = table.select(list(columns))
            yield from table.to_pylist()


def read_rows(uids: Sequence[str], root: str | Path | None = None) -> list[dict]:
    base = pool_root(root)
    index = pq.read_table(base / "index.parquet", columns=["uid", "part", "part_row"])
    index = index.filter(pc.is_in(index["uid"], pa.array(list(uids), pa.string())))
    found: dict[str, dict] = {}
    for part in set(index["part"].to_pylist()):
        rows = index.filter(pc.equal(index["part"], part))
        table = pq.read_table(base / part).take(rows["part_row"])
        for row in table.to_pylist():
            found[row["uid"]] = row
    missing = [u for u in uids if u not in found]
    if missing:
        raise KeyError(f"{len(missing)} uids not in the pool, e.g. {missing[:3]}")
    return [found[u] for u in uids]


def load_image(row: dict) -> Image.Image:
    with Image.open(io.BytesIO(row["image"])) as image:
        image.load()
        return image.copy()


def points(row: dict, key: str = "baseline") -> list[tuple[float, float]]:
    return [(round(float(x), 2), round(float(y), 2)) for x, y in row.get(key) or []]


__all__ = ["INDEX_SCHEMA", "POOL_SCHEMA", "SPLITS", "iter_rows", "list_sources", "load_image", "part_files", "points", "pool_root", "read_index",
           "read_rows"]
