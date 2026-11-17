import json

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lines_dejavu import encode_png, render, sentences

from safetensors.torch import load_file

from littera.model import model_directory
from littera.place import Aligner, Objective
from littera.text import Codec
from littera.train import TrainConfig, Trainer, merge, model_config_dict, parse_overrides

TINY = {"channels": [8, 16, 16, 24], "hidden": 32, "layers": 1, "height": 32}


def write_pool(path, n=24):
    rows = []
    for k, text in enumerate(sentences(n, seed=1)):
        image, cuts, centre = render(text, size=22)
        rows.append({"uid": f"syn/d/p/{k}", "source": "syn", "split": "train", "image": encode_png(image), "text": text,
                     "width": image.width, "height": image.height, "polygon": [], "baseline": [],
                     "gold_cuts": [{"i": i, "x": float(x)} for i, x in cuts] if k % 2 == 0 else None})
    pq.write_table(pa.Table.from_pylist(rows), path)


def config(tmp_path, **kwargs):
    base = dict(pool=[str(tmp_path / "pool")], out=str(tmp_path / "run"), val_split="", lr=3e-3, warmup=5, workers=0, log_every=10,
                val_every=1000, min_count=1, model=TINY, batch_columns=8 * 400, max_batch=8,
                augment={"shear": 0.2, "neighbours": 0.2, "cut_edges": 0.2, "stroke": 0.1, "blur": 0.1, "noise": 0.1, "contrast": 0.1})
    return TrainConfig(**{**base, **kwargs})


def test_overfits_resumes_and_loads(tmp_path):
    (tmp_path / "pool").mkdir()
    write_pool(tmp_path / "pool" / "part-00000.parquet")
    Trainer(config(tmp_path, steps=60), device="cpu").train()
    log = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    losses = [r["nll_per_unit"] for r in log if "nll_per_unit" in r]
    assert losses[-1] < 0.75 * losses[0]
    assert any("val_cer" in r for r in log)
    assert all(r["dropped"] == 0 for r in log if "dropped" in r)
    resumed = Trainer(config(tmp_path, steps=70), device="cpu")
    assert resumed.step == 60
    resumed.train()
    assert torch.load(tmp_path / "run" / "state.pt", weights_only=False)["step"] == 70
    aligner = Aligner.load(tmp_path / "run")
    image, cuts, centre = render("Haus am Tag", size=22)
    placement = aligner.place(image, "Haus am Tag")
    assert len(placement.chars) == 11 and placement.placer == "littera"


def test_ctc_prior_tracks_label_prior(tmp_path):
    (tmp_path / "pool").mkdir()
    write_pool(tmp_path / "pool" / "part-00000.parquet", n=8)
    trainer = Trainer(config(tmp_path, steps=3, objective={"topology": "ctc_prior"}, prior_momentum=0.5), device="cpu")
    before = trainer.model.log_prior.clone()
    trainer.train()
    assert not torch.allclose(before, trainer.model.log_prior)
    assert abs(float(trainer.model.log_prior.exp().sum()) - 1) < 1e-3


def test_overrides():
    parsed = parse_overrides(["objective.topology=ctc", "steps=10", "model.channels=[8,8,8,8]", "name=x"])
    assert parsed == {"objective": {"topology": "ctc"}, "steps": 10, "model": {"channels": [8, 8, 8, 8]}, "name": "x"}
    assert merge({"model": {"hidden": 32}, "steps": 1}, parsed)["model"] == {"hidden": 32, "channels": [8, 8, 8, 8]}


def test_ctc_prior_reaches_the_loss(tmp_path):
    (tmp_path / "pool").mkdir()
    write_pool(tmp_path / "pool" / "part-00000.parquet", n=4)
    trainer = Trainer(config(tmp_path, steps=1, objective={"topology": "ctc_prior"}), device="cpu")
    assert trainer.options.topology == "ctc_prior" and not trainer.options.segmental
    batch = next(iter(trainer.loader(trainer.train_set, shuffle=False)))
    trainer.model.log_prior.copy_(torch.linspace(-8, -1, len(trainer.codec)).log_softmax(0))
    with_prior = float(trainer.losses(batch, train=True)["nll"])
    plain = float(trainer.losses(batch, train=False)["nll"])
    trainer.objective.prior_scale = 0.0
    without = float(trainer.losses(batch, train=True)["nll"])
    assert abs(with_prior - without) > 1e-3 and abs(plain - without) < 1e-3


def test_presets_follow_the_source_column(tmp_path):
    (tmp_path / "pool").mkdir()
    rows = []
    for k, (source, text) in enumerate([("marked", "^a^ ~~b~~"), ("raw", "^a^ ~~b~~")]):
        image, _, _ = render(text, size=22)
        rows.append({"uid": f"{source}/d/p/{k}", "source": source, "split": "train", "image": encode_png(image), "text": text,
                     "width": image.width, "height": image.height, "polygon": [], "baseline": [], "gold_cuts": None})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "pool" / "part-00000.parquet")
    trainer = Trainer(config(tmp_path, steps=1, preset="plain", presets={"marked": "markup"}), device="cpu")
    assert "~" in trainer.codec.classes and "^" in trainer.codec.classes
    units = {trainer.train_set.texts[i] + trainer.train_set.presets[i]: trainer.train_set[i]["units"] for i in range(2)}
    assert units == {"^a^ ~~b~~markup": 2, "^a^ ~~b~~plain": 8}


def test_segmental_models_decode_with_the_trained_garbage(tmp_path):
    codec = Codec(["a", "b"])
    segmental = model_config_dict(config(tmp_path), codec, Objective(topology="segmental"), 0)
    ctc = model_config_dict(config(tmp_path), codec, Objective(topology="ctc"), 0)
    forced = model_config_dict(config(tmp_path, decode={"garbage": "filler"}), codec, Objective(topology="segmental"), 0)
    assert segmental["decode"]["garbage"] == "any" and ctc["decode"]["garbage"] == "filler" and forced["decode"]["garbage"] == "filler"


def test_init_starts_from_a_trained_model(tmp_path):
    (tmp_path / "pool").mkdir()
    write_pool(tmp_path / "pool" / "part-00000.parquet")
    Trainer(config(tmp_path, steps=30), device="cpu").train()
    trained_dir = model_directory(tmp_path / "run")
    base = json.loads((trained_dir / "config.json").read_text(encoding="utf-8"))
    tuned = Trainer(config(tmp_path, out=str(tmp_path / "tuned"), init=str(tmp_path / "run"), steps=0, model={"hidden": 999}), device="cpu")
    assert tuned.codec.to_dict() == base["codec"]
    assert tuned.model.config.hidden == base["model"]["hidden"]
    trained = load_file(str(trained_dir / "model.safetensors"))
    assert all(torch.equal(trained[k], v) for k, v in tuned.model.state_dict().items() if k in trained)
