# littera

littera places a line's transcription on the line image. For every character and every word it returns a span on the
line, a box that follows the slant of the writing, and a confidence. For every word and for the whole line it returns a
fit score: how well the text explains the ink. A low line fit flags a transcription that belongs to another line, or a
crop that misses part of the text.

It is meant for historical handwriting and print where a transcription exists but its position on the scan does not:
cutting word images from an edition's scans, making word- and letter-level ground truth, or checking that edition lines
were matched to the right lines on the page.

**Status:** research preview, version 0.1. The general model is a pilot: 800 training steps on 50,000 lines, where the
planned main run will use the full pool of 300,000 lines. A model fine-tuned on Alexander von Humboldt's handwriting is
available. See [Status](#status).

## Install

With Python 3.12 (3.13 is not supported yet) and [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/Maelkolb/littera
cd littera
uv sync
```

`uv sync` installs the CPU build of PyTorch, which is all that placement needs. The commands below then run as
`uv run littera ...`. Without uv, `pip install git+https://github.com/Maelkolb/littera` installs the same package and the
command is plain `littera`.

## Models

| model | trained on | for |
|---|---|---|
| `littera-pilot-seg` | 50,006 lines from 31 openly licensed collections of historical handwriting and print, and 12,500 synthetic lines | placing text on any Latin-script hand, and as the starting point for a fine-tune |
| `littera-humboldt-ft2` | `littera-pilot-seg`, fine-tuned on 19,713 lines from Humboldt's journals and letters | Humboldt's hand only |

"seg" stands for the segmental objective described under [Model and alignment](#model-and-alignment); "ft2" is the
second Humboldt fine-tune, the one released.

Download the zip files from the [v0.1.0 release](https://github.com/Maelkolb/littera/releases/tag/v0.1.0) and unzip them
anywhere. A model is a folder with `config.json` and `model.safetensors`, and you load it by its path. Training details,
scores and licences are in [docs/MODELS.md](docs/MODELS.md).

## Place a line

```python
from PIL import Image
from littera.place import Aligner

aligner = Aligner.load("models/littera-humboldt-ft2", snap="upright")
placement = aligner.place(Image.open("line.jpg"), "ziehen. Das reine Wasser bringt ja gar")

print(placement.line_fit)
for word in placement.words:
    print(placement.text[word.i0:word.i1], word.x0, word.x1, word.q, word.fit)
```

The image is a crop of one line. If you know the line's baseline or polygon in the crop's pixels, pass it to `place`
(`baseline=[(x, y), ...]` or `polygon=[...]`) and littera straightens the line along it. Without one, it estimates the
line's course from the ink.

From the shell, with the placement as JSON on stdout:

```
uv run littera place line.jpg "ziehen. Das reine Wasser bringt ja gar" --model models/littera-humboldt-ft2 --snap upright
uv run littera page page.xml --image page.jpg --model models/littera-pilot-seg -o page.words.xml
```

`littera page` takes each `TextLine` of a PAGE-XML file with its own text, polygon and baseline, places the text, and
writes `Word` elements with `Glyph` elements inside, replacing any `Word` elements the line had. Coordinates are page
pixels, and each `TextEquiv` carries the confidence.

### Output

`placement.to_dict()` gives JSON with the schema `littera/placement/1`. Positions are pixels of the line crop.

| field | meaning |
|---|---|
| `chars[]` | one entry per character of the text: `x0`, `x1`, `poly` (the four corners of a box that follows the slant), `q`, `visible` and `aligned` |
| `words[]` | one entry per space-separated word: `i0` and `i1` (its characters), `x0`, `x1`, `poly`, `q` and `fit` |
| `line_fit` | 0 to 1: near 1 when the text explains the ink as well as the model's own reading of the line does, low for a wrong or incomplete text |
| `slant` | the slant that was removed, as a shear factor |

- `aligned` is false for characters the model does not look for on the page: spaces, and editorial markup such as
  `[...]` or tags, by the rules of the model's markup preset (`littera.text`). `visible` is false for these and also for
  characters the alignment skipped. Characters that are not visible get a zero-width span at their place in the text.
- `q` is 0 to 1: how sure the model is of the edges. Each edge has an expected position and a spread; `q` is the share of
  that spread within a quarter letter of the position for a character (both edges multiplied), within half a letter for
  a word.
- `fit` is a fit score for the word alone, 0 to 1.

The line fit compares the likelihood of the given text with that of the model's own reading of the line without the
text, per character, as `exp(-difference)`. With `littera-humboldt-ft2`, a gate at 0.2 kept 90 % of right Humboldt
lines and none of 1,000 wrong ones ([docs/HUMBOLDT.md](docs/HUMBOLDT.md)). The pilot's line fit is much weaker on a hand
it has not seen; calibrate a gate on your own material.

### Word edges (`snap`)

littera reads the line as a *strip*: a copy straightened along the baseline, deslanted, and scaled to 96 px height.
After placing the text it moves the edges between two words into the ink-free gap between them. This changes `x0`,
`x1` and `poly` at word ends; the characters inside a word move in proportion.

- `snap="strip"` (the default) looks for the gap on the deslanted strip. Use it with the slanted boxes in `poly`.
- `snap="upright"` looks for the gap on the same line without deslanting. Use it when you cut upright rectangles between
  `x0` and `x1`, as word crops usually are.
- `snap="off"` keeps the model's own edges.

## Model and alignment

The network reads the strip in columns of 4 px; each column is one *frame*. A residual CNN in four stages reduces the
strip's height from 96 to 6 rows, attention pooling over those rows gives one vector per frame, and two BiLSTM layers add
context along the line. Two heads read the result: a softmax over the characters, and a sigmoid for "a letter starts
here". The model has 7.0 million parameters.

Aligners built on CTC recognisers, such as kraken's, emit each character at one frame and fill the frames around it with
blanks. Their training loss scores the character sequence only, so where a letter begins or ends is left to heuristics
afterwards. littera's segmental objective models every character as a run of frames with a start and an end, scored by
both heads. The transcript becomes a graph: its characters in order, optional spaces between words that touch,
characters that may be skipped at a cost, and a garbage state for ink the text does not cover (line numbers, a
neighbouring line's ascenders, struck text that was not transcribed). Forward-backward over that graph gives every
character edge an expected position and a spread; `q` is computed from the spreads.

Training maximises the transcript's likelihood under the graph. On synthetic lines, where every letter's position is
known, a second loss trains the boundary head directly. The code also has a CTC objective, used as a control.

The design and the decisions behind it are in [docs/PLAN.md](docs/PLAN.md).

## Results

Comparisons from the pilot round ([docs/r2/PILOT.md](docs/r2/PILOT.md)), errors in letter widths. *Even spacing* spreads
the characters evenly over the line's ink and is the floor. *aligner-k2* is a kraken recogniser trained on Humboldt's
hand and used as a forced aligner; *McCATMuS* is kraken's generic model for print and handwriting of the 16th to 21st
century. The placement measures run on synthetic lines and on the hand-marked words of George Washington's letters
(GW20). The last column places held-out lines once with their own text and once with one word replaced, and asks
whether the line fit tells the two apart.

| placer | synthetic letters: edges within 1/4 letter | GW20 word edges: median error | calibration error of `q` (ECE) | replaced word detected (AUC) |
|---|---|---|---|---|
| even spacing | 31 % | 0.66 | 0.28 | 0.50 |
| kraken aligner-k2 | 75 % | 0.29 | 0.33 | 0.69 |
| kraken McCATMuS | 76 % | 0.28 | 0.33 | 0.72 |
| littera-pilot-seg | 86 % | 0.34 | 0.11 | 0.77 |

On synthetic lines the pilot places the letters inside words better than the CTC aligners, and its confidences are
better calibrated. On GW20 word edges it is slightly behind the two kraken models, which were trained to the end; the
pilot had 800 steps. The synthetic test lines come from the same generator as the synthetic training lines, in other
fonts, so the letter result has to be confirmed on hand-marked letters of real hands.

On 1,000 test lines of Humboldt's hand ([docs/HUMBOLDT.md](docs/HUMBOLDT.md)), the share of word boundaries that fall on
an ink-free column, measured on aligner-k2's own strip:

| | aligner-k2 | littera-humboldt-ft2 |
|---|---:|---:|
| no snapping | 89.9 % | 90.5 % |
| own snapping (k2's as deployed; littera `strip` / `upright`) | 96.8 % | 93.1 % / 95.2 % |
| aligner-k2's snapping applied to both | 96.8 % | 97.4 % |

The two models place word edges about equally well. k2's own snapping scores higher than littera's on this measure
because it searches the same strip, with the same ink threshold, that the measure inspects. Both separate the right line
text from another line's text with AUC 0.999 (littera) and 0.998 (k2).

## Train or fine-tune

```
uv run littera train --config run.json [--set key=value ...]
```

A run reads line pools: Parquet files with one row per line, following `littera.data.POOL_SCHEMA` (the crop as image
bytes, its text, `split` as `train` or `val`, a `source` name, and the baseline and polygon in crop pixels where known).
There is no converter from PAGE-XML yet. A fine-tune starts from a trained model with `init`, which also fixes the
alphabet and the architecture:

```json
{"pool": ["data/my_pool"], "out": "runs/my-ft", "init": "models/littera-pilot-seg", "steps": 6000, "lr": 0.0005, "warmup": 200}
```

Training needs a GPU and a CUDA build of PyTorch in the environment. The Humboldt fine-tune took about 40 minutes on one
RTX 5090; the pilots were trained on Modal, a cloud GPU service (`modal/train.py`).

Evaluation code is in `littera.eval` (`python -m littera.eval.run`): readers for word-level ground truth (GW20, IAM,
Esposalles, HistDB), synthetic lines with exact letter positions, mismatched texts, and the metrics above; see
[docs/r1/EVAL.md](docs/r1/EVAL.md).

## Data and licences

- Code: MIT ([LICENSE](LICENSE)).
- Training data: openly licensed sources only (MIT, CC BY, CC BY-SA, CC0, public domain). Non-commercial, no-derivatives
  and unpublished archival material is used for evaluation only. The decision for each source is in
  [configs/sources.yaml](configs/sources.yaml) and [docs/DATA_CARD.md](docs/DATA_CARD.md).
- Model weights: CC BY-SA 4.0, because part of the training data is CC BY-SA ([docs/MODELS.md](docs/MODELS.md)).

## Status

- The general model is a pilot. The main run on the full pool (299,992 training lines) waits for about 100 lines with
  hand-marked letter boundaries. They will show whether the segmental objective also places real letters better than
  CTC, as it does on synthetic lines; if not, the main run uses CTC ([docs/PLAN.md](docs/PLAN.md)).
- Letter placement inside words has been measured on synthetic lines only.
- Latin script only so far.
- Placement takes about half a second per line on one CPU core.

## Repository

| path | contents |
|---|---|
| `src/littera/` | geometry, text units, model, alignment graph, placement, training, PAGE-XML, command line |
| `src/littera/eval/`, `src/littera/baselines/` | metrics, readers for word-level ground truth, even-spacing and kraken baselines |
| `src/littera/synth/` | synthetic lines with exact letter positions |
| `src/littera/data/` | the line-pool reader and schema |
| `modal/` | data, training and evaluation jobs on Modal |
| `configs/` | licence and role of every source, pilot configs |
| `docs/` | design (`PLAN.md`), data card, models, the Humboldt fine-tune, and the reports of the development rounds (`r1/`, `r2/`) |
| `results/` | scores of the baselines and pilots |
| `tools/mark/` | the page for marking letter boundaries by hand |
| `tests/` | `uv run pytest` |

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff); GitHub shows it under "Cite this repository".
