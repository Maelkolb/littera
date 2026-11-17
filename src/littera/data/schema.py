import pyarrow as pa

SPLITS = ("train", "val", "test", "heldout")
POINTS = pa.list_(pa.list_(pa.float32()))

POOL_SCHEMA = pa.schema([
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
])

INDEX_SCHEMA = pa.schema([f for f in POOL_SCHEMA if f.name != "image"] + [pa.field("part", pa.string()), pa.field("part_row", pa.int32())])
