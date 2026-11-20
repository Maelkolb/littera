"""aligner-k2's unsnapped letter cuts ("raw-mid") for every matched edition line of the Humboldt's Hand lookup.

Writes `/vol/humboldt_index/k2raw/<doc>/<page>.json` on the volume `littera`: per line `{"t", "x0", "cuts", "lq"}`, cuts in
the stored page's pixels. Pages, match files and the model come from the volume `humboldt-htr` (read only).

    uv run modal run modal/humboldt_k2raw.py::place [--pages H0016412/H0016412_2r,...]
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("littera-humboldt-k2raw")
littera_volume = modal.Volume.from_name("littera")
humboldt_volume = modal.Volume.from_name("humboldt-htr")
HUMBOLDT_SOURCE = "C:/Users/totom/Projects/humboldt-htr/src/humboldt_htr"
MODEL = "/hh/runs/aligner-k2/model/best.safetensors"
GROW = 6

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("kraken>=7.1,<7.2", "pillow")
    .env({"PYTHONUTF8": "1"})
    .add_local_dir(HUMBOLDT_SOURCE, "/root/humboldt_htr", ignore=["__pycache__", "viewer_ui", "*.pyc"])
)


@app.function(image=image, cpu=1.0, memory=2048, volumes={"/vol": littera_volume, "/hh": humboldt_volume}, timeout=2 * 3600, max_containers=10)
def place_chunk(stems: list[str], redo: bool) -> dict:
    import torch
    from humboldt_htr.crops import crop_fragments, fragments_box
    from humboldt_htr.tokens import TokenPlacer
    from PIL import Image

    torch.set_num_threads(1)
    placer = TokenPlacer(MODEL, cut="mid", snap=False)
    done = {}
    for stem in stems:
        dst = Path("/vol/humboldt_index/k2raw") / f"{stem}.json"
        if dst.exists() and not redo:
            done[stem] = len(json.loads(dst.read_text(encoding="utf-8")))
            continue
        match = json.loads(Path(f"/hh/lookup/lines/{stem}.json").read_text(encoding="utf-8"))
        page = Image.open(f"/hh/images/{stem}.jpg").convert("RGB")
        result = {}
        for ln in match["lines"]:
            crop = crop_fragments(page, ln["pieces"], grow=GROW)
            x0 = fragments_box(page.size, ln["pieces"], pad=max(8, GROW))[0]
            skip = {i for i, flag in enumerate(ln["f"]) if flag == "a"}
            placed = placer.place(crop, ln["t"], skip=skip) if skip else placer.place(crop, ln["t"])
            result[ln["id"]] = {"t": ln["t"], "x0": x0, "cuts": [round(x0 + c, 1) for c in placed["cuts"]], "lq": placed["line_q"]}
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        done[stem] = len(result)
    littera_volume.commit()
    return done


@app.function(image=image, cpu=1.0, memory=2048, volumes={"/hh": humboldt_volume}, timeout=600)
def list_stems() -> list[tuple[str, int]]:
    return [(f"{f.parent.name}/{f.stem}", len(json.loads(f.read_text(encoding="utf-8"))["lines"]))
            for f in sorted(Path("/hh/lookup/lines").glob("*/*.json"))]


@app.local_entrypoint()
def place(pages: str = "", chunks: int = 60, redo: bool = False):
    stems = list_stems.remote()
    if pages:
        wanted = set(pages.split(","))
        stems = [s for s in stems if s[0] in wanted]
    stems.sort(key=lambda s: -s[1])
    parts = [[] for _ in range(min(chunks, max(1, len(stems))))]
    for k, (stem, _) in enumerate(stems):
        parts[k % len(parts)].append(stem)
    print(f"{len(stems)} pages, {sum(n for _, n in stems)} lines, {len(parts)} chunks")
    total = 0
    for done in place_chunk.map(parts, kwargs={"redo": redo}):
        total += sum(done.values())
        print(f"placed {total} lines")
