"""littera as a verifier of Humboldt readings (app littera-humboldt-verify): ft3 fine-tuned with the verifier's margin
between a line's text and reader-like corruptions of it (src/littera/verify.py). Confusions from the line readers of
ehd-htr on humboldt-htr's test_page split (no evaluation page): /vol/verify/confusions.json.

    PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal volume put littera <confusions.json> /verify/confusions.json
    PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/humboldt_verify.py::main --runs v1-smoke
    PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run --detach modal/humboldt_verify.py::main --runs v1
    PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/humboldt_verify.py::status --runs v1
    PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run modal/humboldt_ft.py::release --run v1 --checkpoint last --name littera-humboldt-verify1
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("littera-humboldt-verify")
volume = modal.Volume.from_name("littera", create_if_missing=True)
VOL = "/vol"
POOL = "/vol/pool/humboldt_ft"
FT3 = "/vol/models/littera-humboldt-ft3"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy==2.5.4", "pillow==13.0.0", "opencv-python-headless==5.0.0.93", "pyarrow==26.0.1", "lxml==6.1.3",
                 "safetensors==0.8.0")
    .env({"PYTHONUTF8": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("littera")
)

BASE = {"pool": [POOL], "init": FT3, "presets": {"humboldt_ft": "markup", "synth": "plain"}, "steps": 2000, "warmup": 100,
        "lr": 5e-5, "val_every": 500, "val_lines": 992, "batch_columns": 24000, "max_batch": 32, "save_every": 600.0}
VERIFY = {"confusions": "/vol/verify/confusions.json", "negatives": 6, "weight": 1.0, "margin": 4.0, "tau": 2.0}
RUNS = {
    "v1-smoke": {"train": {"steps": 20, "max_lines": 400, "val_lines": 64, "val_every": 20, "workers": 2}, "verify": {}},
    "v1": {"train": {}, "verify": {}},
    "v1-strong": {"train": {"lr": 1e-4}, "verify": {"weight": 3.0}},
}


@app.function(image=image, gpu="A10G", cpu=4, memory=16384, volumes={VOL: volume}, timeout=4 * 3600, max_containers=2)
def train(config: dict, verify: dict) -> str:
    from littera.train import TrainConfig
    from littera.verify import VerifyConfig, run

    volume.reload()
    out = run(TrainConfig.from_dict(config), VerifyConfig(**verify), device="cuda", on_save=volume.commit)
    volume.commit()
    return str(out)


@app.local_entrypoint()
def main(runs: str = "v1-smoke"):
    for name in runs.split(","):
        config = {**BASE, "out": f"/vol/runs/{name}", "name": name, **RUNS[name]["train"]}
        verify = {**VERIFY, **RUNS[name]["verify"]}
        if name.endswith("smoke"):
            print(train.remote(config, verify))
            print(json.dumps(status_job.remote([name]), indent=1))
        else:
            call = train.spawn(config, verify)
            print(f"{name}: spawned {call.object_id}\n{json.dumps(config)}\n{json.dumps(verify)}")


@app.function(image=image, volumes={VOL: volume}, cpu=1.0, memory=1024, timeout=600)
def status_job(runs: list[str]) -> dict:
    volume.reload()
    out = {}
    for name in runs:
        path = Path(VOL) / "runs" / name / "log.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        steps = [r for r in rows if "nll_per_unit" in r]
        out[name] = {"last": steps[-1] if steps else None, "val": [r for r in rows if "val_cer" in r],
                     "oom": sum(1 for r in rows if r.get("event") == "oom")}
    return out


@app.local_entrypoint()
def status(runs: str = "v1"):
    print(json.dumps(status_job.remote(runs.split(",")), indent=1))
