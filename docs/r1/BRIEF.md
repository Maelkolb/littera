# Round 1 (2026-11-09): package, line pool, evaluation

Three agents work in parallel in the littera checkout, each in its own files. Read `PLAN.md`,
`CLAUDE.md` and `src/littera/types.py` first. Background on the predecessor: humboldt-htr's
`src/humboldt_htr/tokens.py` and `.../docs/lookup/ALIGNER.md` (read-only, do not copy its structure blindly; littera
is a general tool).

Round budget on Modal: core 0, data 2.50 USD, eval 1.50 USD. Stop and report rather than overrun.

## Contracts (fixed for the round)

### Placement

`littera.types.Placement` (schema `littera/placement/1`), in pixel coordinates of the image passed to `place`
(x right, y down). `chars[k]` belongs to `text[k]` of the original string, one entry per character including spaces.
Characters without ink (markup, skipped, spaces) have `x0 == x1` at their place and `visible`/`aligned` set
accordingly. `poly` is the character's or word's slanted quadrilateral, clockwise from top left. Words are the
maximal non-space runs (`word_spans`). `q`, `fit`, `line_fit` are in 0..1, higher is surer. Every placer (the new
model and every baseline) implements `littera.types.Placer`.

### Line pool (data agent writes, the others read)

Volume `littera`, `/pool/v1/`:

- `lines/<source>/part-00000.parquet ...` (about 200 MB per part) with columns
  `uid` str (unique, `<source>/<doc>/<page>/<n>`), `source` str, `doc` str, `page` str, `hand` str (`""` unknown),
  `split` str, `image` binary (encoded crop as stored, JPEG or PNG), `width` int, `height` int, `text` str (NFC),
  `script` str, `language` str, `licence` str, `open_licence` bool, `quality` str,
  `polygon` list<list<float>> (crop px, empty if unknown), `baseline` list<list<float>> (same), `origin` str
  (where the row came from, e.g. the staged sample id).
- `index.parquet`: every column except `image`, for sampling without reading images.
- `splits`: `train` and `val` (about 1 % of pages of a training source) only for open-licence rows of training
  sources; `test` = held-out pages of training sources (in-domain, at most 300 lines per source); `heldout` = every
  row of a held-out source (at most 2,000 lines per source, whole pages).
- Held-out sources (never in train or val): all Humboldt variants (`humboldt`, `humboldt_md`, `humboldt_tight`,
  `humboldt_pages`), `parzival`, `konzilsprotokolle`, `bullinger`, `swiss_council`, `klosterneuburg_cod48`,
  `austrian_newspapers`. Rows that are not openly licensed may appear only as `heldout` (e.g. `abp`, `laubmann`
  as extra evaluation sources).

### Evaluation sets (eval agent writes)

Volume `littera`, `/eval/<set>/lines.parquet` (and the same file under local `data/eval/<set>/` when under 50 MB):
the pool columns plus `kind` str (`word`, `letter`, `synthetic`, `mismatch`), `gold_words`
list<struct<i0 int, i1 int, x0 float, x1 float>> (character range in `text`, x extent in crop px),
`gold_cuts` list<struct<i int, x float>> (the boundary before `text[i]`), and for `mismatch` rows `true_text` str
and `label` int (1 = `text` is right). `/eval/<set>/README.md` says where the set comes from and its licence.

### Model checkpoint (core agent defines, everybody loads through `littera.place.Aligner.load(path)`)

A directory with `model.safetensors` and `config.json` (model config, codec, geometry settings, objective).

## Agent A: core package

Files: `src/littera/{text,geometry,model,graph,place,train,pagexml,cli}.py`, `tests/`, `modal/train.py`,
report `docs/r1/CORE.md`.

1. `text`: Unicode handling. NFD units, combining marks attached to their base (zero width), presets that mark
   characters without ink (Transkribus-style tags, editorial brackets, line-end hyphen variants ¬ = ⸗ -), and a
   `Codec` (built from texts with a minimum count, gap/blank, `<any>`, saved in `config.json`) with a mapping from
   original string indices to aligned units and back.
2. `geometry`: `prepare(image, polygon=None, baseline=None, height=96)` returns the network strip (grey, ink high,
   16 px padding) and the transform back to the crop: straighten along the baseline when given, otherwise find the
   main text band of a box crop that may contain neighbouring lines; estimate the slant and shear it out; keep the
   slant so cuts map back as slanted lines. Ink profile helpers for snapping.
3. `model`: the PLAN architecture, configurable for the pilot: backbone `resnet` (PLAN) or `vgsl` (kraken's stock
   recogniser with one column pool removed, the k2 control), context `bilstm` or `local` (dilated 1-D convolutions,
   receptive field about three letters), heads `char` and `boundary`.
4. `graph`: one alignment graph per line used for the loss and for decoding, topologies `ctc`, `ctc_prior`,
   `segmental` (PLAN), with optional spaces, skippable units at a penalty, garbage states at the ends and between
   words. Batched log-space forward on GPU, forward-backward posteriors, Viterbi. Test: the `ctc` topology gives the
   same loss as `torch.nn.functional.ctc_loss`.
5. `place`: `Aligner.load(path)`, `Aligner.place(...) -> Placement`: expected boundaries and their spread from the
   posteriors (spread -> `q`), word ends snapped into ink gaps on the strip, slanted polygons back in crop pixels,
   `line_fit` (the text's loss against the model's own best reading, per aligned unit) and per-word `fit`.
6. `train`: reads the pool parquet (local path or `/vol/pool/v1`), width-bucketed batches, augmentation (shear,
   height jitter, stroke width, contrast, blur, noise, neighbouring-line ink above and below, ascenders cut at the crop
   edge, random masking of rare characters to `<any>`), the three objectives, an optional exact-boundary loss on the
   boundary head for samples that carry cuts, mixed precision, checkpoint + resume, `log.jsonl`. `modal/train.py`:
   an L4 job around it (do not launch it this round). Local test: overfit a tiny model on a few dozen lines rendered
   with matplotlib's DejaVu font, CPU, seconds.
7. `pagexml` + `cli`: read PAGE-XML TextLines (Coords, Baseline, TextEquiv), crop, place, write Word and Glyph
   elements with Coords and TextEquiv; `littera place <image> <text>` (JSON) and
   `littera page <page.xml> --image <img> --model <dir> -o <out.xml>`.

Keep CPU tests under a minute in total.

## Agent B: line pool and data card

Files: `src/littera/data/`, `modal/pool.py`, `configs/sources.yaml`, `docs/DATA_CARD.md`, report `docs/r1/DATA.md`.

Source of rows: the Kurrent VLM staged pool on volume `kurrent-raw` (read-only): `/staged/<source>`,
`/staged_v14/...`, `/staged_v15/<source>` (parquet shards, schema in kurrent-vlm's
`src/kurrent_corpus/schema.py` and `docs/CONVERTER_CONTRACT.md`; per-source licences and pool sizes in
`.../docs/DATA_CARD_v1_5.md`; build decisions with rule outcomes under `/build_corpus_v1_5_pool` or
`/build_corpus_v1_5_r5`). Only `unit == line` with `task == transcribe`.

1. `configs/sources.yaml`: every candidate source with its licence as recorded, `open_licence` true/false and a
   one-line reason, role `train` / `heldout` / `out`. Open = MIT, CC BY, CC BY-SA, CC0, PDM. Where the record is mixed
   per row (e.g. zenodo19c), decide per row. Recorded conflicts (dresden, nzz, mkn, finnish images, leopold images)
   follow the stricter reading unless both readings are open.
2. Row rules: non-empty text, at least 2 characters, plausible width per character for the crop height, rows that
   passed the corpus build's rules where that outcome exists. Pages, not lines, decide the split.
3. Draw: about 300k train lines, at most 30k per source sampled across as many pages and hands as possible, print
   at most 40k in total; val about 1 % of pages; test and heldout as in the contract.
4. `modal/pool.py` (app `littera-data`, CPU, at most 10 containers): plan from the index columns first, read images
   only for the drawn rows, write the pool and `index.parquet`, then a summary (rows, pages, hands, characters per
   split and source, character inventory with counts, crop height distribution).
5. `littera.data.pool`: a small reader (iterate rows of a split, optionally by source, from a local path or the
   volume) used by training and evaluation.
6. `docs/DATA_CARD.md`: sources, licences, counts, held-out list, rules, how to rebuild. Copy a few hundred
   index rows locally to `data/pool_sample/` with their crops for the other agents' tests.

## Agent C: evaluation

Files: `src/littera/eval/`, `src/littera/synth/`, `src/littera/baselines/`, `modal/eval.py`, report
`docs/r1/EVAL.md`.

1. Metrics (`littera.eval.metrics`): word boundary error in letter widths (line ink width over visible characters),
   share within 0.25 and 0.5 letters, word x-IoU; the same for letter cuts; ink-gap rate of word boundaries; AUC of
   `line_fit` and of word `fit` on mismatch rows; calibration of `q`. A runner `littera.eval.run` (placer x set ->
   `placements.jsonl` + `scores.json`) that works locally and inside Modal.
2. Synthetic exact-boundary lines (`littera.synth`): fonts with open licences only (OFL, Apache, public domain;
   record each font's licence), handwriting scripts, Fraktur, Kurrent/Sütterlin-like where openly licensed,
   historical print; boundaries from the cumulative advance of the rendered prefix; degradations (paper, ink colour,
   blur, noise, slant, baseline wobble, neighbouring lines). Split the fonts: training fonts and held-out fonts.
   Eval set `synth-heldfonts` (2,000 lines, texts from heldout pool lines). The generator must be usable by training
   in the next round.
3. Word gold: find word-level ground truth made by hand that can be downloaded without registration (for example the
   ICFHR 2016 keyword-spotting sets; check each candidate's licence and how the word boxes were made). Convert to line
   crops with `gold_words`. For sets that need registration (IAM-HistDB George Washington, Parzival, Saint Gall; IAM;
   Esposalles) write the converter and the download instructions for the owner; do not register anywhere.
4. Mismatch sets from heldout pool lines: text swapped with another line of the same page, one word replaced, one
   word deleted, right text (label 1).
5. Baselines, run on every set on Modal CPU (app `littera-eval`): `EvenPlacer` (characters spread evenly over the
   line's ink), and a kraken CTC forced-alignment placer with the k2 rule (cut midway between spikes, word ends
   snapped into ink gaps) for `aligner-k2` (volume `humboldt-htr`, `/runs/aligner-k2/model/best.safetensors`) and for
   a public generic kraken model (e.g. McCATMuS). Results table in `docs/r1/BASELINES.md`.
6. Letter-gold candidates: about 120 lines (about 10 sources, mostly heldout, plus `abp` and `laubmann` if the pool
   has them as heldout), clean crops, 15 to 60 characters, stored as `/eval/letter-gold-candidates/lines.parquet`.
   The marking page comes next round.

The pool appears during the round. Start with what does not need it (metrics, synthetic lines, word gold, baselines
code); poll `/pool/v1/index.parquet` before steps 4 to 6.

## Reports

Each agent writes its report file: what was built, decisions and why, measured numbers, spend, what is open. Keep it
short and factual.
