"""littera as a verifier: fine-tuning so that a line's true text is more likely on its ink than texts a line reader would
misread it as.

Beside the training loss of ``littera.train``, every training line carries ``negatives`` corruptions of its text, drawn
from a reader's confusions (``Corruptor``: substitutions, deletions and insertions of characters as a reader makes them,
a word swapped for a word of the pool's vocabulary that looks alike, a capital changed). The true text and its
corruptions are scored on the line's own network output with the DECODE graph (the graph ``Aligner.analyse`` scores a text
with), and the loss asks for a margin between them:

    loss += weight * mean(softplus((margin - (nll(corruption) - nll(truth))) / tau))

Validation adds ``val_verify_acc`` (share of pairs the truth wins) and ``val_verify_margin`` (median nll gap) over fixed
corruptions of the validation lines.
"""

from __future__ import annotations

import functools
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from littera import graph
from littera.graph import GraphOptions
from littera.train import LineDataset, TrainConfig, Trainer, WidthBatches, _seed_worker, collate

WORD = re.compile(r"[^\W\d_]{2,}")


def _levenshtein(a: str, b: str, cap: int) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


class Corruptor:
    """Texts a reader might have read instead of ``text``: a word that looks alike, a capital, or one or two edits inside a word."""

    def __init__(self, confusions: dict, vocabulary: list[str]):
        self.sub = {a: (list(bs), list(bs.values())) for a, bs in confusions["sub"].items() if bs}
        self.dele = confusions["del"]
        self.ins = (list(confusions["ins"]), list(confusions["ins"].values()))
        total = {k: sum(v.values()) if k != "sub" else sum(sum(b.values()) for b in confusions["sub"].values())
                 for k, v in (("sub", confusions["sub"]), ("del", confusions["del"]), ("ins", confusions["ins"]))}
        self.kinds = (list(total), list(total.values()))
        edits = {int(k): v for k, v in confusions["edits"].items() if int(k) <= 2}
        self.edits = (list(edits), list(edits.values()))
        self.by_length: dict[int, list[str]] = {}
        for word in sorted(set(vocabulary)):
            self.by_length.setdefault(len(word), []).append(word)

    def _neighbour(self, word: str, rng: random.Random) -> str | None:
        pool = [w for n in (len(word) - 1, len(word), len(word) + 1) for w in self.by_length.get(n, [])]
        if not pool:
            return None
        for other in rng.sample(pool, min(60, len(pool))):
            if other != word and other.lower() != word.lower() and _levenshtein(word, other, 2) <= 2:
                return other
        return None

    def _edit(self, chars: list[str], lo: int, hi: int, rng: random.Random) -> None:
        kind = rng.choices(*self.kinds)[0]
        places = list(range(lo, hi))
        if kind == "sub":
            choices = [i for i in places if chars[i] in self.sub]
            if choices:
                i = rng.choice(choices)
                chars[i] = rng.choices(*self.sub[chars[i]])[0]
                return
            kind = "del"
        if kind == "del" and hi - lo > 1:
            weights = [self.dele.get(chars[i], 0.5) for i in places]
            del chars[rng.choices(places, weights)[0]]
            return
        chars.insert(rng.randint(lo, hi), rng.choices(*self.ins)[0])

    def __call__(self, text: str, rng: random.Random) -> str | None:
        words = [m.span() for m in WORD.finditer(text)]
        if not words:
            return None
        roll = rng.random()
        if roll < 0.35:  # a word that looks alike
            a, b = rng.choice(words)
            other = self._neighbour(text[a:b], rng)
            if other:
                return text[:a] + other + text[b:]
        if roll < 0.45:  # a capital changed
            a, b = rng.choice(words)
            w = text[a:b]
            return text[:a] + (w[0].lower() if w[0].isupper() else w[0].upper()) + w[1:] + text[b:]
        chars = list(text)
        a, b = rng.choice(words) if rng.random() < 0.9 else (0, len(text))
        for _ in range(rng.choices(*self.edits)[0]):
            if b <= a:
                break
            before = len(chars)
            self._edit(chars, a, b, rng)
            b += len(chars) - before
        out = "".join(chars)
        return out if out != text and out.strip() else None


@dataclass
class VerifyConfig:
    confusions: str  # json of littera-ehd's confusions (sub, del, ins, edits)
    negatives: int = 3
    weight: float = 1.0
    margin: float = 4.0
    tau: float = 2.0
    decode: dict = field(default_factory=dict)  # over the init model's decode options
    val_pairs_seed: int = 7


class VerifyDataset(LineDataset):
    """A ``LineDataset`` whose items also carry ``verify``: the DECODE graphs of the true text and of its corruptions."""

    @classmethod
    def around(cls, base: LineDataset, corruptor: Corruptor, negatives: int, options: GraphOptions, fixed: bool) -> VerifyDataset:
        self = cls.__new__(cls)
        self.__dict__.update(base.__dict__)
        self.corruptor, self.negatives, self.verify_options, self.fixed = corruptor, negatives, options, fixed
        return self

    def __getitem__(self, i: int) -> dict:
        item = super().__getitem__(i)
        text, preset = self.texts[i], self.presets[i]
        rng = random.Random(i * 7919 + 13) if self.fixed else self.rng
        texts, tries = [text], 0
        while len(texts) < 1 + self.negatives and tries < 4 * self.negatives:
            tries += 1
            neg = self.corruptor(text, rng)
            if neg and neg not in texts:
                texts.append(neg)
        item["verify"] = [graph.build(self.codec.encode(t, preset).graph_words(), self.verify_options) for t in texts] if len(texts) > 1 else []
        return item


def verify_collate(items: list[dict], stride: int = 4) -> dict:
    batch = collate(items, stride)
    rows, graphs, sizes = [], [], []
    for b, item in enumerate(items):
        if item.get("verify"):
            rows += [b] * len(item["verify"])
            graphs += item["verify"]
            sizes.append(len(item["verify"]))
    batch["v_graphs"] = graph.pack(graphs) if graphs else None
    batch["v_rows"] = torch.tensor(rows, dtype=torch.long)
    batch["v_sizes"] = sizes
    return batch


class VerifyTrainer(Trainer):
    def __init__(self, config: TrainConfig, verify: VerifyConfig, device=None, on_save=None):
        super().__init__(config, device, on_save)
        self.verify = verify
        confusions = json.loads(Path(verify.confusions).read_text(encoding="utf-8"))
        vocabulary = [w for t in self.train_set.texts for w in WORD.findall(t)]
        corruptor = Corruptor(confusions, vocabulary)
        init_decode = json.loads((Path(config.init) / "config.json").read_text(encoding="utf-8")).get("decode", {}) if config.init else {}
        self.v_options = GraphOptions.from_dict({**init_decode, **verify.decode, "topology": self.objective.topology})
        self.train_set = VerifyDataset.around(self.train_set, corruptor, verify.negatives, self.v_options, fixed=False)
        self.val_set = VerifyDataset.around(self.val_set, corruptor, verify.negatives, self.v_options, fixed=True)
        self.decode_extra = {**init_decode, **verify.decode}

    def loader(self, dataset, shuffle: bool) -> DataLoader:
        sampler = WidthBatches(dataset.estimates, self.config.batch_columns, self.config.max_batch, self.config.seed + self.step, shuffle)
        workers = self.config.workers
        return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                          collate_fn=functools.partial(verify_collate, stride=self.model.config.stride),
                          worker_init_fn=_seed_worker if workers else None, pin_memory=self.device.type == "cuda")

    def _gaps(self, batch: dict, logp: torch.Tensor, boundary: torch.Tensor, lengths: torch.Tensor, empty: torch.Tensor) -> torch.Tensor:
        """nll(corruption) - nll(truth) of every pair in the batch (finite pairs only)."""
        if batch.get("v_graphs") is None:
            return torch.zeros(0, device=logp.device)
        rows = batch["v_rows"].to(self.device)
        nll = graph.nll(logp[rows], boundary.float()[rows], batch["v_graphs"].to(self.device), lengths[rows], self.v_options, False,
                        None, 0.0, empty[rows], None)
        gaps, k = [], 0
        for size in batch["v_sizes"]:
            truth, negs = nll[k], nll[k + 1 : k + size]
            k += size
            ok = torch.isfinite(negs) & torch.isfinite(truth)
            gaps.append((negs - truth)[ok])
        return torch.cat(gaps) if gaps else torch.zeros(0, device=logp.device)

    def losses(self, batch: dict, train: bool) -> dict:
        images = batch["images"].to(self.device, non_blocking=True)
        widths = batch["widths"].to(self.device)
        dtype = torch.float16 if self.amp == "fp16" else torch.bfloat16
        with torch.autocast(self.device.type, dtype=dtype, enabled=self.amp != "off"):
            logits, boundary, lengths = self.model(images, widths)
        out = self._from_outputs(batch, train, logits, boundary, lengths)
        logp = logits.float().log_softmax(-1)
        empty = batch["empty"].to(self.device)
        gaps = self._gaps(batch, logp, boundary, lengths, empty)
        v = self.verify
        vloss = F.softplus((v.margin - gaps) / v.tau).mean() if gaps.numel() else torch.zeros((), device=self.device)
        out["loss"] = out["loss"] + v.weight * vloss
        out["v_gaps"] = gaps.detach()
        return out

    def _from_outputs(self, batch, train, logits, boundary, lengths) -> dict:
        """``Trainer.losses`` from the network's outputs (the forward pass is shared with the verifier's term)."""
        model = self.model
        forward = model.forward
        model.forward = lambda *a, **k: (logits, boundary, lengths)
        try:
            return Trainer.losses(self, batch, train)
        finally:
            model.forward = forward

    def validate(self) -> dict:
        scores = super().validate()
        self.model.eval()
        gaps = []
        with torch.no_grad():
            for batch in self.loader(self.val_set, shuffle=False):
                out = self.losses(batch, train=False)
                gaps.append(out["v_gaps"].float().cpu())
        self.model.train()
        g = torch.cat(gaps) if gaps else torch.zeros(0)
        scores["val_verify_acc"] = float((g > 0).float().mean()) if g.numel() else float("nan")
        scores["val_verify_margin"] = float(g.median()) if g.numel() else float("nan")
        scores["val_verify_pairs"] = int(g.numel())
        return scores


def run(config: TrainConfig, verify: VerifyConfig, device=None, on_save=None) -> Path:
    return VerifyTrainer(config, verify, device, on_save).train()
