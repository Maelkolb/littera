# Models

Both models have the same architecture: ResNet backbone (4 stages, 64 to 256 channels) with attention pooling over rows,
two BiLSTM layers of 256 units, a character head and a boundary head; input height 96 px, one frame per 4 px; segmental
objective; 6,985,737 parameters; 327 output labels. A model folder holds `config.json` (architecture, alphabet,
geometry, decoding) and `model.safetensors`. Load it with `Aligner.load(folder)`.

Weights licence: CC BY-SA 4.0. Part of the training data is CC BY-SA (see [DATA_CARD.md](DATA_CARD.md)), and the
Humboldt texts are CC BY-SA 4.0.

## littera-pilot-seg

Chosen from five pilot runs (CTC, CTC with label priors, segmental with a BiLSTM or with local context, and kraken's
architecture with CTC) for its letter placement and calibration ([r2/PILOT.md](r2/PILOT.md)). It is the general model
of version 0.1.

| | |
|---|---|
| training lines | 50,006 lines drawn from the train split of littera's line pool (one sixth of each of the 31 training sources, see [DATA_CARD.md](DATA_CARD.md)), plus 12,500 synthetic lines with exact letter positions (20 % of the rows) |
| training | 800 steps (about one pass), AdamW, learning rate 1e-3 with cosine decay (planned for 1,200 steps, stopped at 800 to keep budget for the evaluation), batches of at most 64,000 strip columns and 72 lines, one A10G on Modal |
| decoding | garbage segments as in training (`garbage = "any"`), optional spaces, skippable characters; word-edge snapping as chosen when loading (`strip` by default) |

Scores ([r2/PILOT.md](r2/PILOT.md), section 4; errors in letter widths):

| measure | littera-pilot-seg | kraken aligner-k2 | kraken McCATMuS |
|---|---|---|---|
| synthetic letters, edge within 1/4 letter | 86 % | 75 % | 76 % |
| synthetic characters, both edges within 1/4 letter | 83 % | 65 % | 65 % |
| GW20 word edges, median error | 0.34 | 0.29 | 0.28 |
| GW20 word edges within 1/2 letter | 71 % | 79 % | 77 % |
| IAM word edges, median error | 0.06 | 0.03 | 0.01 |
| calibration error of the character confidence (ECE) | 0.11 | 0.33 | 0.33 |
| line with a replaced / deleted word detected by the line fit (AUC) | 0.77 / 0.76 | 0.69 / 0.66 | 0.72 / 0.73 |
| characters read right without the text (test lines of the 31 training sources) | 53 % | - | - |

Its line fit is weak on a hand it has not seen: on Humboldt test lines it tells the right text from another line's text
with AUC 0.840, against 0.999 after the fine-tune ([HUMBOLDT.md](HUMBOLDT.md)).

## littera-humboldt-ft2

`littera-pilot-seg` fine-tuned on Alexander von Humboldt's hand. Details and the comparison with aligner-k2:
[HUMBOLDT.md](HUMBOLDT.md).

| | |
|---|---|
| training lines | 19,713 lines from 584 pages of Humboldt's journals and letters (the edition humboldt digital's text matched to the scans by the humboldt-htr pipeline, a separate private repository); 909 lines from 31 further pages for validation. Pages with any line used in an evaluation were left out. |
| training | from `littera-pilot-seg` (`init`), 6,000 steps, learning rate 5e-4, warm-up 200 steps, batches of at most 48,000 columns and 64 lines, fp16, about 40 minutes on one RTX 5090 |
| checkpoint | the lowest validation loss, at step 5,500 (validation character error rate without the text: 23.5 %) |
| used with | `snap="upright"`, because the Humboldt's Hand lookup page cuts upright word crops |

Use it for Humboldt's hand only. On a synthetic printed line, where `littera-pilot-seg` gives a line fit of 0.58, this
model gives 0.02.

The line images come from scans served by the digilib server of the Berlin-Brandenburg Academy of Sciences and
Humanities (BBAW); their rights stay with the holding libraries and
archives. The weights contain no images.
