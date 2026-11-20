"""The letter-crop classifier: a small CNN that names a letter from the crop a placer cut for it.

Trained on the letters of one fold of lines and tested on the other, with the same recipe for every placer. It knows nothing
of any placer. Better cuts give more consistent crops and so a higher accuracy.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from torch import nn

from littera.eval.humboldt import classify_runs, connected_letters, fold_of, letter_crops, resolve

MIN_CLASS = 10
EPOCHS = 15
BATCH = 128
SEEDS = (0, 1)


class LetterNet(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        layers, channels = [], 1
        for width in (32, 64, 128):
            layers += [nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, classes: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = LetterNet(classes)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x = torch.from_numpy(train_x).float().div(255).unsqueeze(1)
    y = torch.from_numpy(train_y).long()
    steps = EPOCHS * ((len(x) + BATCH - 1) // BATCH)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=2e-3, total_steps=max(1, steps))
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), generator=generator)
        for k in range(0, len(x), BATCH):
            picks = order[k : k + BATCH]
            loss = nn.functional.cross_entropy(model(x[picks]), y[picks], label_smoothing=0.05)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
    model.eval()
    out = []
    with torch.no_grad():
        tx = torch.from_numpy(test_x).float().div(255).unsqueeze(1)
        for k in range(0, len(tx), 1024):
            out.append(model(tx[k : k + 1024]).argmax(1))
    return torch.cat(out).numpy() if out else np.zeros(0, np.int64)


def classify(lines: dict[str, dict], placements: dict[str, dict], name: str, refs: tuple[str, ...]) -> dict[str, dict]:
    """Per line of ``lines``: the letters tested (index in the text), the share of seeds that named each right, and whether the
    letter lies inside a connected word (no ink-free run in the word, not its first or last letter)."""
    edges = resolve(name, placements, lines)
    ref_edges = [resolve(r, placements, lines) for r in refs]
    uids = sorted(uid for uid in lines if uid in edges and all(uid in r for r in ref_edges))
    crops = {}
    for uid in uids:
        images, index = letter_crops(lines[uid]["neutral"], lines[uid]["text"], edges[uid])
        crops[uid] = (images, index, [lines[uid]["text"][i] for i in index])
    out = {}
    for test_fold in (0, 1):
        train = [uid for uid in uids if fold_of(uid) != test_fold]
        test = [uid for uid in uids if fold_of(uid) == test_fold]
        counts = Counter(ch for uid in train for ch in crops[uid][2])
        classes = {ch: k for k, ch in enumerate(sorted(ch for ch, n in counts.items() if n >= MIN_CLASS))}
        train_x = np.concatenate([crops[uid][0][[k for k, ch in enumerate(crops[uid][2]) if ch in classes]] for uid in train])
        train_y = np.array([classes[ch] for uid in train for ch in crops[uid][2] if ch in classes])
        kept = {uid: [k for k, ch in enumerate(crops[uid][2]) if ch in classes] for uid in test}
        test_x = np.concatenate([crops[uid][0][kept[uid]] for uid in test])
        test_y = np.array([classes[crops[uid][2][k]] for uid in test for k in kept[uid]])
        right = np.mean([fit_predict(train_x, train_y, test_x, len(classes), seed) == test_y for seed in SEEDS], axis=0)
        position = 0
        for uid in test:
            line = lines[uid]
            refs_x = [r[uid] for r in ref_edges]
            connected = connected_letters(line["text"], classify_runs(line["neutral"], line["text"], refs_x), refs_x)
            letters = [crops[uid][1][k] for k in kept[uid]]
            scores = right[position : position + len(letters)]
            position += len(letters)
            out[uid] = {"split": line["split"], "i": letters, "ok": [round(float(v), 2) for v in scores],
                        "connected": [i in connected for i in letters], "fold": test_fold, "classes": len(classes), "train_letters": len(train_y)}
    return out
