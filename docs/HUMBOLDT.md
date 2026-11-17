# Fine-tuning on Humboldt's hand

`littera-humboldt-ft2` is `littera-pilot-seg` fine-tuned on the journals and letters of Alexander von Humboldt
(1769–1859), for the word crops of the Humboldt's Hand lookup page (https://maelkolb.github.io/humboldt-hand-lookup/).
It is compared with aligner-k2, a kraken CTC recogniser trained on Humboldt's hand, which the lookup page uses now as a
forced aligner in the humboldt-htr pipeline (a separate, private repository).

## Data

The lines are the edition humboldt digital's text (CC BY-SA 4.0) matched to the detected lines of the scans by
humboldt-htr. Only pages of humboldt-htr's train and validation splits were used, and no page that holds a line of any evaluation
below. Text written above the line was removed from the transcription, because it is not on the line crop.

| | pages | lines |
|---|---:|---:|
| train | 584 | 19,713 |
| validation | 31 | 909 |

## Training

From `littera-pilot-seg` with `init`, so the alphabet and the architecture are the pilot's:

```json
{"pool": ["data/ft_pool"], "out": "runs/ft2", "name": "littera-humboldt-ft2", "init": "models/littera-pilot-seg",
 "steps": 6000, "warmup": 200, "lr": 0.0005, "val_every": 500, "batch_columns": 48000, "max_batch": 64,
 "presets": {"humboldt_ft": "markup"}}
```

`presets` maps the pool's `source` name to a markup preset, which says which characters of a transcription have no ink.

About 40 minutes on one RTX 5090. The validation character error rate (reading without the text) fell from 45.0 % at
step 500 to 23.5 % at step 5,500, the checkpoint with the lowest validation loss. A first run of 2,000 steps (ft1)
placed word edges within 0.6 points of ft2 on every split (with `strip` snapping) and read worse (28.9 %); ft2 is the
one released.

## Comparison with aligner-k2

The test set is humboldt-htr's aligner evaluation: 1,000 lines, 200 from each of five splits of humboldt-htr. `val` and
`test_page` are pages of the journals it trains on that were kept out of training, `test_bench` is a benchmark of 30
pages, `test_journal` two journals never trained on (Austria 1797/98, France 1798), and `letters` 34 pages of held-out
letters. None of these pages is in littera's fine-tuning data. The lines are cut from the page as the lookup page cuts
them.

### Word boundaries

A word boundary counts as right when the midpoint between one word's end and the next word's start lies within 1.5 px of
a column without ink in the core band (the rows around the x-height) of aligner-k2's normalised line strip.

| placer, word-edge snapping | val | test_page | test_bench | test_journal | letters | all |
|---|---:|---:|---:|---:|---:|---:|
| aligner-k2, off | 86.6 | 86.6 | 92.3 | 87.9 | 95.1 | 89.9 |
| aligner-k2, on k2's strip (as the lookup page uses it) | 94.9 | 94.8 | 97.4 | 96.8 | 99.3 | 96.8 |
| littera-pilot-seg, `strip` | 85.9 | 84.1 | 88.3 | 86.3 | 91.3 | 87.4 |
| littera-humboldt-ft2, `off` | 88.0 | 87.5 | 92.7 | 88.1 | 95.5 | 90.5 |
| littera-humboldt-ft2, `strip` | 91.4 | 89.5 | 94.3 | 91.9 | 97.4 | 93.1 |
| littera-humboldt-ft2, `upright` | 93.8 | 92.7 | 95.7 | 94.9 | 98.0 | 95.2 |
| littera-humboldt-ft2, then aligner-k2's snapping | 96.3 | 95.8 | 97.7 | 96.8 | 99.7 | 97.4 |

aligner-k2's snapping moves every word boundary into an ink-free column of the same strip that the measure inspects,
with the same ink threshold. It reaches 96.8 %, close to the share of boundaries that have an empty column within reach
at all (96.6 %, counted from k2's raw edges). littera's `strip` snapping looks for the gap on its own deslanted line and
`upright` on its own upright line, each with its own ink threshold; measured on k2's columns they reach 93.1 and 95.2 %.

Two comparisons hold the snapping equal: the two models without snapping (littera 90.5 %, k2 89.9 %) and the two with
aligner-k2's snapping (97.4 % and 96.8 %). littera is ahead by 0.2 to 1.4 points per split in both, except on
test_journal with k2's snapping, where the two are level. Each split has between 920 and 1,518 word boundaries.

With `strip` snapping, littera's word edges lie a median 0.17 letters from those of aligner-k2 as the lookup page uses
it; per split, 83 to 85 % are within half a letter.

### Line fit: right text against another line's text

Each line is placed twice, with its own text and with the text of another line of the same page, and the line fit has
to tell them apart.

| | AUC | right lines kept at the strictest gate that lets no wrong line through | at gate 0.2 |
|---|---:|---:|---:|
| aligner-k2 | 0.998 | 94.0 % | 90.8 % |
| littera-pilot-seg | 0.840 | 33.8 % | 8.5 % |
| littera-humboldt-ft2 | 0.999 | 93.4 % | 90.2 % |

No wrong line passed at gate 0.2 for either aligner-k2 or littera-humboldt-ft2.

### Not yet measured

aligner-k2 was also judged by reading its word crops with a TrOCR model and by blind visual checks of word and letter
crops (humboldt-htr, `docs/lookup/ALIGNER.md`). Those checks have not been run for littera.

## In the Humboldt's Hand lookup page

A build of the lookup page with an index from `littera-humboldt-ft2` (`snap="upright"`, line-fit gate 0.2) was made for
review on 17 November 2026; it is not deployed. It holds 31,168 lines, against 30,361 with aligner-k2. On the 29,390
lines in both, the word edges of the two lie a median 0.05 letters apart, and 88.9 % within half a letter. Of the 1,778
lines only in the littera build, 315 are lines in which aligner-k2 found a run of at least 8 characters without support,
which usually means the crop misses part of the text. littera's confidences do not flag such lines, so only the
line-fit gate filters them.
