# Baselines (round 1, 2026-11-09)

Three placers on every evaluation set (Modal app `littera-eval`, CPU, 2 cores per container, shards of 400 rows):

- `even`: every character of the text (spaces included) gets the same width over the line's ink extent in its core band;
  `q` = `line_fit` = 0.5 everywhere. The floor.
- `kraken-k2`: `aligner-k2` (Humboldt fine-tune, `humboldt-htr:/runs/aligner-k2/model/best.safetensors`, read-only), kraken
  7.1.1, centre-normalised strip 120 px high, 4 px per frame, codec NFC. CTC Viterbi forced alignment of the text, the k2 rule:
  every character's spike = first frame of its label, cuts midway between consecutive spikes (`charalign.centres_to_cuts`),
  word boundaries snapped into the widest ink gap of the strip's core band between the neighbouring characters' centres,
  characters inside a word kept in proportion. `q` = the label's highest probability on its frames; `line_fit` =
  exp(-(CTC loss of the text - CTC loss of the greedy reading) / labels) as in `humboldt_htr.tokens`; word `fit` = exp(mean
  over the word's frames of log p(path label) - log p(best label)). Reimplemented in `littera.baselines.kraken_ctc`
  (own Viterbi instead of kraken's trellis; kraken 7.1 returns logits, so log-softmax as in the predecessor).
- `kraken-mccatmus`: the same rule with McCATMuS (Zenodo 13788177, CC BY 4.0, generic 16th to 21st century print and
  handwriting, `McCATMuS_nfd_nofix_V1.mlmodel`), no centre normalisation (it was trained on baseline-extracted lines),
  8 px per frame, codec NFD (combining marks owned by their base character). Polygons, when a row has one, mask the crop.

## Placement

Errors in letter widths (line ink width / visible characters). Word edges: both ends of every gold word. Letter cuts:
boundaries between two characters of a word (synthetic set only until the owner's letter gold exists). Ink-gap rate:
predicted word boundaries on an ink-free column of the core band (for the mismatch set over all variants, wrong texts included).

| set | placer | lines | words | word edge error median | within 0.25 / 0.5 (%) | word x-IoU | both ends within 0.5 (%) | letter cut error median | letter within 0.25 / 0.5 (%) | ink-gap rate (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| gw20-words | even | 657 | 4,860 | 0.66 | 20.7 / 39.9 | 0.63 | 20.2 | - | - | 42.0 |
| gw20-words | kraken-k2 | 657 | 4,860 | 0.29 | 43.4 / 78.6 | 0.80 | 62.5 | - | - | 92.5 |
| gw20-words | kraken-mccatmus | 657 | 4,860 | 0.28 | 45.6 / 77.3 | 0.80 | 60.3 | - | - | 94.0 |
| iam-words | even | 999 | 7,786 | 0.45 | 33.4 / 53.5 | 0.70 | 36.1 | - | - | 56.9 |
| iam-words | kraken-k2 | 999 | 7,786 | 0.03 | 72.7 / 79.4 | 0.81 | 71.6 | - | - | 97.1 |
| iam-words | kraken-mccatmus | 999 | 7,786 | 0.01 | 91.0 / 97.0 | 0.95 | 94.2 | - | - | 99.7 |
| synth-heldfonts | even | 2,000 | 13,286 | 0.37 | 39.3 / 60.0 | 0.77 | 41.9 | 0.44 | 31.5 / 55.7 | 45.1 |
| synth-heldfonts | kraken-k2 | 2,000 | 13,286 | 0.05 | 86.7 / 93.3 | 0.93 | 88.6 | 0.13 | 74.9 / 91.8 | 95.8 |
| synth-heldfonts | kraken-mccatmus | 2,000 | 13,286 | 0.03 | 86.1 / 91.7 | 0.94 | 83.9 | 0.14 | 76.4 / 93.0 | 99.2 |
| mismatch-heldout | even | 3,785 | - | - | - | - | - | - | - | 18.6 |
| mismatch-heldout | kraken-k2 | 3,785 | - | - | - | - | - | - | - | 66.0 |
| mismatch-heldout | kraken-mccatmus | 3,785 | - | - | - | - | - | - | - | 73.1 |
| letter-gold-candidates | even | 119 | - | - | - | - | - | - | - | 17.3 |
| letter-gold-candidates | kraken-k2 | 119 | - | - | - | - | - | - | - | 54.9 |
| letter-gold-candidates | kraken-mccatmus | 119 | - | - | - | - | - | - | - | 64.0 |

Reading the table:

- The k2 rule (spikes, mid cuts, snapping) more than halves the even floor's word-edge error on every set. On GW20 (hand-drawn
  boxes with a margin around the ink) both kraken models are within 0.5 letter for 77 to 79 % of word edges.
- IAM is easy for word edges: the crop is the bounding box of the line's components and the gold box is the union of a
  word's components, so a snapped boundary lands on the gold edge exactly when the right gap is found (median 0.01 letter
  for McCATMuS). k2 finds the wrong gap more often on modern English (p90 1.96 letters against 0.23).
- Synthetic held-out fonts give the only letter-level numbers so far: median cut error 0.13 to 0.14 letter, 92 to 93 %
  within half a letter, 75 to 76 % within a quarter, for either kraken model; even spacing 56 % / 32 %. These fonts are
  cleaner than handwriting, so this is an upper bound of what the CTC rule does inside words; the owner's letter gold will
  say how much of it holds on real hands.
- The ink-gap rate on real heldout lines (mismatch set, 73 % / 66 %) is far below the Humboldt figure of 95 to 99 %
  (ALIGNER.md): box-cropped sources (`abp` 43 to 49 %, `humboldt` 52 to 63 %, `parzival` 51 to 57 %) carry neighbouring
  lines' ascenders and descenders into the core band, so fewer columns are empty at all; on cleanly masked crops
  (`bullinger` 83 to 87 %, `austrian_newspapers` 82 to 93 %) it is close to the Humboldt figure.

## Fit (mismatch-heldout: 960 lines of 8 heldout sources, right text against swapped, replaced, deleted)

| placer | rows | line_fit AUC, all | swap | swap-doc | replace | delete | word fit AUC (replaced word vs the rest) | mean line_fit, right text |
|---|---|---|---|---|---|---|---|---|
| even | 3,785 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| kraken-k2 | 3,785 | 0.774 | 0.972 | 0.977 | 0.688 | 0.655 | 0.844 | 0.240 |
| kraken-mccatmus | 3,785 | 0.816 | 0.991 | 0.994 | 0.723 | 0.729 | 0.858 | 0.149 |

line_fit AUC over all variants, per source:

| source | kraken-k2 | kraken-mccatmus |
|---|---|---|
| abp | 0.779 | 0.787 |
| austrian_newspapers | 0.777 | 0.915 |
| bullinger | 0.715 | 0.799 |
| humboldt | 0.914 | 0.756 |
| klosterneuburg_cod48 | 0.749 | 0.874 |
| konzilsprotokolle | 0.810 | 0.794 |
| parzival | 0.839 | 0.929 |
| swiss_council | 0.764 | 0.837 |

A whole wrong line is caught (AUC 0.97 to 0.99, as on Humboldt); one wrong or missing word is not (0.66 to 0.73): the
line fit averages over all characters, so one word moves it little. The word fit sees the replaced word much better (0.84
to 0.86). k2 is best on its own Humboldt hands (0.91) and below McCATMuS elsewhere.

Text/crop check: right against swap, per source, is 0.94 to 1.00 for both models; there is no source whose texts are paired
with the wrong crops. Only 2 of 960 lines have a swapped text that both models prefer (`abp/1706/ABP_B5_1706_00303/39`, a
two-word line with the markup `~~O~~`, and `abp/1768/ABP_B5_1768_01013/16`, a box crop with the neighbouring line in full);
both texts are right on inspection.

## Characters and `q` (synth-heldfonts, 71,156 characters with both gold cuts)

| placer | both cuts within 0.25 letter (%) | mean `q` | `q` AUC | `q` ECE |
|---|---|---|---|---|
| even | 22.0 | 0.50 | 0.50 | 0.28 |
| kraken-k2 | 65.1 | 0.53 | 0.61 | 0.33 |
| kraken-mccatmus | 65.3 | 0.76 | 0.58 | 0.33 |

This is agent A's definition of a character's `q` (the probability that both of its cuts lie within 0.25 letter), so the
littera models' `q` can be compared directly. Words (`q` = minimum over the word, against both ends within 0.5 letter):
AUC 0.51 / 0.52 on GW20, 0.61 / 0.61 on IAM, 0.60 / 0.58 on the synthetic set (k2 / McCATMuS), ECE 0.4 to 0.7. The spike
probability is not the probability of a right boundary, as in ALIGNER.md (AUC about 0.6).

## Cost and time

CPU seconds per line (placement only): k2 about 0.3 on GW20/IAM, 0.17 to 0.2 on synthetic and heldout lines; McCATMuS 0.1
to 0.16. All baseline runs together (including a first GW20 run and the merges) cost about 0.16 USD.

## Rerun

```
uv run modal run modal/eval.py::fetch_models
uv run modal run modal/eval.py::baselines --placers even,kraken-k2,kraken-mccatmus \
    --sets gw20-words,iam-words,synth-heldfonts,mismatch-heldout,letter-gold-candidates      # -> runs/eval/baselines.json
uv run python -m littera.eval.report                                                          # tables
uv run modal run modal/eval.py::baselines --placers littera:/vol/runs/<run> --sets ...        # a littera model, same sets
```

Placements and per-row scores stay on the volume: `/vol/eval_runs/<placer>/<set>/placements-<shard>.jsonl`,
`rowscores-<shard>.jsonl`, `scores.json`.
