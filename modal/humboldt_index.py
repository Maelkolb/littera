"""Places every matched edition line of the Humboldt's Hand lookup with a littera model.

Reads pages and match files from the volume `humboldt-htr` (read only) and writes one placement file per page to
`/vol/humboldt_index/<tag>/<doc>/<page>.json` on the volume `littera`, in the layout of humboldt-htr's
`data/lookup/place/<tag>` with littera's slant, line centre and x-height added.

    uv run modal run modal/humboldt_index.py::place --model littera-humboldt-ft2 --tag ft2 [--pages H0016412/H0016412_2r,...]
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("littera-humboldt-index")
littera_volume = modal.Volume.from_name("littera")
humboldt_volume = modal.Volume.from_name("humboldt-htr")
HUMBOLDT_SOURCE = "C:/Users/totom/Projects/humboldt-htr/src/humboldt_htr"
GROW = 6

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy==2.5.4", "pillow==13.0.0", "opencv-python-headless==5.0.0.93", "pyarrow==26.0.1", "lxml==6.1.3",
                 "safetensors==0.8.0", "scipy", extra_index_url="https://download.pytorch.org/whl/cpu")
    .env({"PYTHONUTF8": "1"})
    .add_local_python_source("littera")
    .add_local_dir(HUMBOLDT_SOURCE, "/root/humboldt_htr", ignore=["__pycache__", "viewer_ui", "*.pyc"])
)


def line_text_on_crop(text: str, flags: str) -> tuple[str, list[int]]:
    kept, index = [], []
    for i, (ch, flag) in enumerate(zip(text, flags)):
        if flag == "a":
            continue
        if ch == " " and (not kept or kept[-1] == " "):
            continue
        kept.append(ch)
        index.append(i)
    while kept and kept[-1] == " ":
        kept.pop()
        index.pop()
    return "".join(kept), index


def spread_cuts(n: int, index: list[int], cuts: list[float]) -> tuple[list[float], list[bool]]:
    start = [None] * n
    end = [None] * n
    for k, i in enumerate(index):
        start[i], end[i] = cuts[k], cuts[k + 1]
    out, placed, position = [], [], cuts[0] if cuts else 0.0
    for i in range(n):
        if start[i] is not None:
            position = start[i]
            out.append(position)
            position = end[i]
            placed.append(True)
        else:
            out.append(position)
            placed.append(False)
    out.append(position)
    return out, placed


@app.function(image=image, cpu=2.0, memory=4096, volumes={"/vol": littera_volume, "/hh": humboldt_volume}, timeout=2 * 3600, max_containers=10)
def place_chunk(stems: list[str], model: str, tag: str, overrides: dict, redo: bool) -> dict:
    import numpy as np
    import torch
    from humboldt_htr.crops import crop_fragments, fragments_box
    from humboldt_htr.lookup import ink_extent
    from PIL import Image

    from littera.graph import GraphOptions
    from littera.place import Aligner

    torch.set_num_threads(2)
    options = dict(overrides)
    decode = options.pop("decode", None)
    aligner = Aligner.load(f"/vol/models/{model}", **options)
    if decode:
        aligner.decode = GraphOptions.from_dict({**aligner.decode.to_dict(), **decode})
    done = {}
    for stem in stems:
        dst = Path("/vol/humboldt_index") / tag / f"{stem}.json"
        if dst.exists() and not redo:
            done[stem] = len(json.loads(dst.read_text(encoding="utf-8")))
            continue
        match = json.loads(Path(f"/hh/lookup/lines/{stem}.json").read_text(encoding="utf-8"))
        page = Image.open(f"/hh/images/{stem}.jpg").convert("RGB")
        result = {}
        for ln in match["lines"]:
            text, flags = ln["t"], ln["f"]
            crop = crop_fragments(page, ln["pieces"], grow=GROW)
            bx0, by0 = fragments_box(page.size, ln["pieces"], pad=max(8, GROW))[:2]
            on_crop, index = line_text_on_crop(text, flags)
            if not on_crop.strip():
                continue
            base = sorted((x, y) for p in ln["pieces"] for x, y in (p.get("baseline") or []))
            in_crop = [(x - bx0, y - by0) for x, y in base]
            analysis = aligner.analyse(crop, on_crop, baseline=in_crop if len(in_crop) >= 2 else None)
            placement = aligner.to_placement(analysis, on_crop)
            cuts, placed = spread_cuts(len(text), index, placement.cuts())
            cuts = [round(bx0 + c, 1) for c in cuts]
            visible = [False] * len(text)
            q = [0.0] * len(text)
            for k, i in enumerate(index):
                visible[i] = placement.chars[k].visible
                q[i] = placement.chars[k].q
            strip = analysis.strip
            centre_offset = None
            if len(base) >= 2:
                xs = np.linspace(cuts[0], cuts[-1], 9)
                base_y = np.interp(xs, [x for x, _ in base], [y for _, y in base])
                centre_y = np.interp(xs - bx0, strip.centre_x, strip.centre_y) + by0
                centre_offset = round(float(np.mean(base_y - centre_y)), 2)
            entry = {"t": text, "x0": bx0, "w": crop.width, "cuts": cuts, "q": [round(v, 3) for v in q], "visible": visible,
                     "lq": placement.line_fit, "sl": round(placement.slant, 4), "xh": round(strip.core * strip.scale, 2), "co": centre_offset,
                     "letter": placement.extra.get("letter"), "reading": placement.extra.get("reading")}
            ink = ink_extent(page, ln["pieces"], cuts, text, flags)
            if ink:
                entry["ink"] = ink
            result[ln["id"]] = entry
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        done[stem] = len(result)
    littera_volume.commit()
    return done


@app.function(image=image, cpu=1.0, memory=2048, volumes={"/hh": humboldt_volume}, timeout=600)
def list_stems() -> list[tuple[str, int]]:
    out = []
    for f in sorted(Path("/hh/lookup/lines").glob("*/*.json")):
        out.append((f"{f.parent.name}/{f.stem}", len(json.loads(f.read_text(encoding="utf-8"))["lines"])))
    return out


@app.local_entrypoint()
def place(model: str = "littera-humboldt-ft2", tag: str = "ft2", pages: str = "", overrides: str = '{"snap": "upright"}', chunks: int = 40,
          redo: bool = False):
    stems = list_stems.remote()
    if pages:
        wanted = set(pages.split(","))
        stems = [s for s in stems if s[0] in wanted]
    stems.sort(key=lambda s: -s[1])
    parts = [[] for _ in range(min(chunks, max(1, len(stems))))]
    for k, (stem, _) in enumerate(stems):
        parts[k % len(parts)].append(stem)
    print(f"{len(stems)} pages, {sum(n for _, n in stems)} lines, {len(parts)} chunks, model {model}, tag {tag}")
    total = 0
    for done in place_chunk.map(parts, kwargs={"model": model, "tag": tag, "overrides": json.loads(overrides), "redo": redo}):
        total += sum(done.values())
        print(f"placed {total} lines")
