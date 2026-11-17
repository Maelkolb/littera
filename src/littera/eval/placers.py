from __future__ import annotations

import os

MODEL_PATHS = {
    "k2": os.environ.get("LITTERA_K2_MODEL", "/hh/runs/aligner-k2/model/best.safetensors"),
    "mccatmus": os.environ.get("LITTERA_MCCATMUS_MODEL", "/vol/models/mccatmus/McCATMuS_nfd_nofix_V1.mlmodel"),
}

KRAKEN_PRESETS = {
    "kraken-k2": ("k2", True),
    "kraken-mccatmus": ("mccatmus", False),
}


def make_placer(spec: str):
    if spec == "even":
        from littera.baselines.even import EvenPlacer

        return EvenPlacer()
    if spec in KRAKEN_PRESETS or spec.startswith("kraken:"):
        from littera.baselines.kraken_ctc import KrakenPlacer

        if spec in KRAKEN_PRESETS:
            model, centre = KRAKEN_PRESETS[spec]
            return KrakenPlacer(MODEL_PATHS[model], name=spec, centre_normalise=centre)
        path, _, centre = spec.removeprefix("kraken:").partition("@")
        return KrakenPlacer(path, name=f"kraken:{os.path.basename(path)}", centre_normalise=centre != "raw")
    if spec.startswith("littera:"):
        from littera.place import Aligner

        aligner = Aligner.load(spec.removeprefix("littera:"))
        if not getattr(aligner, "name", ""):
            aligner.name = spec
        return aligner
    raise ValueError(f"unknown placer {spec!r}")


def placer_label(spec: str) -> str:
    return spec.replace(":", "_").replace("/", "_").replace("@", "_").strip("_")
