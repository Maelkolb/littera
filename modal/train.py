from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("littera-train")
volume = modal.Volume.from_name("littera", create_if_missing=True)
ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy==2.5.4", "pillow==13.0.0", "opencv-python-headless==5.0.0.93", "pyarrow==26.0.1", "lxml==6.1.3",
                 "safetensors==0.8.0")
    .env({"PYTHONUTF8": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("littera")
)

GPUS = ["L4", "A10G"]
PILOT_POOL = ["/vol/pilot/v1/lines"] + [f"/vol/synth/v1/lines/synth/part-{k:05d}.parquet" for k in range(5)]
PILOT_RUNS = {
    "pilot-k2": {"model": {"backbone": "vgsl", "context": "kraken"}, "objective": {"topology": "ctc"}},
    "pilot-ctc": {"model": {"backbone": "resnet", "context": "bilstm"}, "objective": {"topology": "ctc"}},
    "pilot-prior": {"model": {"backbone": "resnet", "context": "bilstm"}, "objective": {"topology": "ctc_prior"}},
    "pilot-seg": {"model": {"backbone": "resnet", "context": "bilstm"}, "objective": {"topology": "segmental"}},
    "pilot-seg-local": {"model": {"backbone": "resnet", "context": "local"}, "objective": {"topology": "segmental"}},
}


def source_presets() -> dict:
    import yaml

    sources = yaml.safe_load((ROOT / "configs" / "sources.yaml").read_text(encoding="utf-8"))["sources"]
    return {"synth": "plain", **{name: entry["markup"] for name, entry in sources.items() if entry.get("markup")}}


@app.function(image=image, gpu=GPUS, cpu=4, memory=16384, volumes={"/vol": volume}, timeout=12 * 3600)
def train(config: dict) -> str:
    from littera.train import TrainConfig, run

    volume.reload()
    out = run(TrainConfig.from_dict(config), device="cuda", on_save=volume.commit)
    volume.commit()
    return str(out)


@app.local_entrypoint()
def main(name: str, config: str = "", set: str = "", wait: bool = False):
    from littera.train import merge, parse_overrides

    base = json.loads(Path(config).read_text(encoding="utf-8")) if config else {}
    base.setdefault("pool", ["/vol/pool/v1/lines"])
    base.setdefault("presets", source_presets())
    base["out"] = f"/vol/runs/{name}"
    base["name"] = name
    merged = merge(base, parse_overrides(set.split()))
    print(json.dumps(merged, indent=1, ensure_ascii=False))
    if wait:
        print(train.remote(merged))
    else:
        call = train.spawn(merged)
        print(f"spawned {call.object_id}; log: /vol/runs/{name}/log.jsonl")


@app.function(image=image, gpu=GPUS, cpu=4, memory=16384, volumes={"/vol": volume}, timeout=660)
def smoke_job(base: dict, runs: dict, steps: int, eval_rows: int, tag: str = "smoke") -> dict:
    import functools
    import os
    import shutil
    import time

    import torch

    from littera import train as training
    from littera.eval.run import run as eval_run
    from littera.place import Aligner
    from littera.train import TrainConfig, Trainer, merge

    started = time.time()
    report = {"cpus": len(os.sched_getaffinity(0)), "gpu": torch.cuda.get_device_name(0), "runs": {}}
    original = training.read_rows
    cached = functools.lru_cache(maxsize=4)(lambda paths, splits, sources, max_lines, seed: original(
        list(paths), list(splits), list(sources) if sources else None, max_lines, seed))
    training.read_rows = lambda paths, splits, sources=None, max_lines=0, seed=0: cached(
        tuple(paths), tuple(splits), tuple(sources) if sources else None, max_lines, seed)
    tic = time.time()
    rows = training.read_rows(base["pool"], ["train"], None, 0, 0)
    report["read_s"] = round(time.time() - tic, 1)
    report["train_rows"] = rows.num_rows

    def keep(stage: str) -> None:
        report["stage"] = stage
        report["elapsed_s"] = round(time.time() - started, 1)
        path = Path("/vol/pilot/smoke") / f"{tag}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        volume.commit()

    for name, change in runs.items():
        out = Path("/tmp/smoke") / name
        shutil.rmtree(out, ignore_errors=True)
        config = TrainConfig.from_dict(merge({**base, "out": str(out), "name": name, "steps": steps}, change))
        tic = time.time()
        try:
            trainer = Trainer(config, device="cuda")
        except Exception as error:  # noqa: BLE001
            report["runs"][name] = {"error": repr(error)}
            keep(f"setup {name}")
            continue
        setup = time.time() - tic
        if not report.get("loader"):
            loader = trainer.loader(trainer.train_set, shuffle=True)
            tic, lines, batches = time.time(), 0, 0
            for batch in loader:
                lines += len(batch["labels"])
                batches += 1
                if batches == 40:
                    break
            report["loader"] = {"lines_per_s": round(lines / (time.time() - tic), 1), "lines_per_batch": round(lines / batches, 1)}
        tic = time.time()
        error = None
        try:
            trainer.train()
        except Exception as caught:  # noqa: BLE001
            error = repr(caught)
        log = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        steps_log = [r for r in log if "nll_per_unit" in r]
        report["runs"][name] = {"setup_s": round(setup, 1), "train_s": round(time.time() - tic, 1), "params": log[0]["params"],
                                "labels": log[0]["labels"], "log": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                                                                    for r in steps_log],
                                "val": [r for r in log if "val_cer" in r], "oom": [r for r in log if r.get("event") == "oom"],
                                "error": error}
        del trainer
        torch.cuda.empty_cache()
        keep(f"trained {name}")

    for name in runs:
        out = Path("/tmp/smoke") / name
        if not (out / "best").exists():
            continue
        aligner = Aligner.load(out, device="cpu")
        torch.set_num_threads(2)
        timings = {}
        for set_name in ("gw20-words", "synth-heldfonts"):
            tic = time.time()
            result = eval_run(f"littera:{out}", set_name, "/vol/eval", f"/tmp/smoke_eval/{name}", limit=eval_rows, placer=aligner)
            timings[set_name] = {"rows_s": round((time.time() - tic) / max(1, result["placed"]), 3), **result,
                                 "scores": training_scores(Path(result["directory"]))}
        report["runs"][name]["eval"] = timings
        keep(f"evaluated {name}")
    report["total_s"] = round(time.time() - started, 1)
    keep("done")
    return report


def training_scores(directory: Path) -> dict:
    from littera.eval.run import merge_scores

    result = merge_scores(directory)
    return {k: v for k, v in result.items() if k in ("errors", "lines", "word", "letter", "gap_rate")}


@app.local_entrypoint()
def smoke(steps: int = 40, eval_rows: int = 12, runs: str = ",".join(PILOT_RUNS), set: str = "", out: str = "", tag: str = "smoke"):
    from littera.train import merge, parse_overrides

    base = {"pool": PILOT_POOL, "presets": source_presets(), "warmup": 10, "log_every": 10, "val_every": 10_000, "val_lines": 120,
            "save_every": 1e9}
    base = merge(base, parse_overrides(set.split()))
    report = smoke_job.remote(base, {name: PILOT_RUNS[name] for name in runs.split(",")}, steps, eval_rows, tag)
    text = json.dumps(report, indent=1, ensure_ascii=False)
    print(text)
    if out:
        Path(out).write_text(text, encoding="utf-8")
