# Round 1, agent A: core package (2026-11-09)

Built: `src/littera/{text,geometry,model,graph,place,train,pagexml,cli}.py`, `tests/` (26 tests of mine, helper
`tests/lines_dejavu.py`), `modal/train.py` (written, not launched). Modal spend: 0. No change to `types.py` needed.

## What each module does

| module | content |
|---|---|
| `text` | NFD units: a base character with its combining marks is one unit (class = its NFC form, falling back to the base letter, then `<any>`). Presets mark characters without ink: `none`, `plain` (default: zero-width characters no ink, line-end `¬ ⸗ = - ‐` and soft hyphen optional), `tags` (`<...>` no ink), `editorial` (`[ ] ⟨ ⟩ 〈 〉` no ink, their content optional, `[...]` no ink), `all`. `Codec` (label 0 = gap/blank, 1 = `<any>`, then classes; built with a minimum count, rare marked clusters fold onto their base letter; counts kept for rare-character masking). `Encoded` maps original indices to units (`unit_of`, `word_units`) and back. |
| `geometry` | `prepare(image, polygon, baseline, config)` -> `Strip`: grey, ink high (background by morphological closing, so masked white or black crop borders do not become ink; solid dark border regions are filled first), polygon mask, guide line (baseline, else polygon centre, else a robust slice-wise fit of the main text band of a box crop), x-height from the core band relative to the guide, scale so the x-height is 19 px (`core_ratio` 0.2 of 96), slant estimated on the straightened strip (shear search to ±50°, kept only if it sharpens the column profile by 3 %) and sheared out, 16 px padding plus the shear overhang, width a multiple of 4. `Strip.to_crop(u, v)` maps back (cuts become slanted lines), `strip_u(x)` maps crop x to the strip. Ink helpers: `core_profile`, `empty_frames`, `best_gap`, `ink_edge`, `ink_extent`. |
| `model` | `LineModel`: backbone `resnet` (PLAN: stem + 4 stages of 2 basic blocks, 64/128/192/256, strides 2x2, 2x2, 2x1, 2x1, attention pooling over the 6 rows with a row embedding) or `vgsl` (kraken stock spec with the third pool `Mp2,1`, plain PyTorch, height 96 -> 12 rows x 64 ch = 768 per frame); context `bilstm` (2 x BiLSTM 256, packed), `local` (1x1 + 3 residual dilated convs 1/2/4, receptive field 15 frames = 60 px, about three letters), `kraken` (3 x BiLSTM 200 with kraken's dropouts); char head (softmax) and boundary head (conv k=3, sigmoid); `log_prior` buffer for `ctc_prior`. `save_model` / `load_model`. |
| `graph` | One alignment graph for loss and decoding (below): `build`, `pack`, custom autograd log-likelihood (forward in log space, backward = explicit beta pass), `posteriors` (occupancy, entries, exits), `viterbi`, `free_log_partition`, `free_viterbi`, `greedy_ctc`, `nll`. |
| `place` | `Aligner.load(path)` (a model dir, or a run dir with `best/` or `last/`), `Aligner.place(...) -> Placement`, `Aligner.analyse` (all intermediate numbers), `Aligner.read` (best reading). |
| `train` | Parquet reader (`lines/**/*.parquet`, skips `index.parquet`, split and source filters, optional `gold_cuts`), width-bucketed batches, augmentation, the three objectives, exact-boundary losses, AMP, checkpoint + resume, `log.jsonl`, validation (nll per unit, greedy CER). `littera train --config c.json --set objective.topology=ctc ...` |
| `pagexml`, `cli` | PAGE-XML lines (Coords, Baseline, TextEquiv) -> crop (polygon box + 8 px) -> place -> `Word` and `Glyph` elements with Coords (page px) and TextEquiv (`conf` = q, word `custom="littera {fit:...;}"`), inserted before the line's TextEquiv. `littera place <image> <text> --model DIR` (JSON), `littera page <xml> --image <img> --model DIR -o <out.xml>`. |
| `modal/train.py` | App `littera-train`, L4, 8 CPU, 32 GB, volume `littera` at `/vol`, pool default `/vol/pool/v1/lines`, out `/vol/runs/<name>`, volume commit at every save, resumes from `state.pt`. Launch (next round): `PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run --detach modal/train.py::main --name pilot-seg --set "objective.topology=segmental max_lines=50000 steps=20000"` (`--detach` is required: the entrypoint spawns). |

Parameter counts (200 labels; the heads grow with the alphabet by 513 per label):

| backbone | context | total | backbone | context |
|---|---|---|---|---|
| resnet | bilstm (PLAN) | **6.92 M** | 4.19 M | 2.63 M |
| resnet | local | 6.79 M | 4.19 M | 2.50 M |
| vgsl | kraken (= k2 control) | **3.77 M** | 0.21 M | 3.48 M |
| vgsl | bilstm / local | 3.99 / 3.07 M | 0.21 M | |

PLAN said "about 8M"; the architecture as specified comes to 6.9M. Widths are configurable (`model.channels`, `hidden`).

## The alignment graph

States per line: units (one per text unit), gap states (label 0) at junctions, garbage states, and for the CTC
topologies an optional blank after every unit inside a word. A junction (line start, between words, line end) is
`gap -> [garbage -> gap]`: the second gap is reachable only from garbage, so every frame labelling has one path.
Line-end junctions are optional; a space is required (at least one gap or garbage frame) or optional at a penalty.
Skippable units (optional characters) add bypass arcs at `skip_penalty`. Rules that keep the graph unambiguous: no
gap/blank -> gap/blank arcs, in-word blanks never enter a junction, and in CTC no unit -> unit arc for equal labels.
The `ctc` graph with optional spaces and no garbage is exactly CTC: the test checks loss **and gradients** against
`torch.nn.functional.ctc_loss` (repeats, multi-word, varying lengths, 1e-8). Batched over lines, vectorised over
states (predecessor/successor tables, in-degree <= ~6), loop only over time. CPU, 32 lines x 300 frames: segmental
loss forward 0.10 s, backward 0.17 s; ctc 0.07 / 0.08 s. GPU speed not measured.

### Segmental topology as a normalised likelihood

Per frame the model gives a label distribution `p_t` and a boundary probability `b_t`. A frame labelling is a
sequence of segments, each a run of one label; a labelling is consistent if no two gap segments are adjacent. Its
weight is `prod_t p_t(c_t) * prod_{t>0} (b_t if a segment starts at t else 1 - b_t)`. `Z` is the sum over all
consistent labellings, computed by a forward pass over a free label loop (`free_log_partition`, O(T x C)):
`a_t(c) = p_t(c) [ (1 - b_t) a_{t-1}(c) + b_t S_{t-1}(c) ]`, `S` the total mass excluding gap when `c` is gap.
The transcript graph sums the weights of exactly the labellings that read as the transcript (units in order, one
segment each, gap segments at spaces and optionally at the ends). With required spaces, no garbage and no skips,
every consistent labelling belongs to exactly one transcript, so `P(y|x) = F_y / Z` sums to 1 over all transcripts:
the test enumerates every transcript of up to 5 frames over a 3-label alphabet and gets `sum = 1` to 1e-9. The loss
is `-(log F_y - log Z)`, normalised per unit. Both heads are trained by it: frames inside a letter must emit the
letter (not blank), and `b` must fire where the labels change.

Two findings while building it:

- **Garbage must stay inside the normalised space.** A filler garbage state scored by `1 - p_t(gap)` without
  boundary factors lets training push the loss below zero (seen: -1.08 nats per unit after 150 steps). Training
  garbage is therefore an `<any>` segment (`garbage="any"`, entry and per-frame penalties), which is an exact subset.
  Decoding uses the more permissive filler (`garbage="filler"`), where normalisation does not matter.
  Penalties (garbage, skips, optional spaces) are priors: they keep `F_y <= Z` but `P` is then the weight of a union
  of transcripts, not of one.
- **Nothing in a sequence objective ties labels to ink.** The first segmental run (tiny model, 1,500 steps) read its
  validation lines well (CER 6.5 %) but put letters 2 letters off (median, before snapping): the BiLSTM emitted the
  first word inside the left padding. Two anchors fix it: an **ink prior** (`ink_prior`, default 2 nats) subtracts a
  penalty from every non-gap label on frames with no ink within one frame in the core band (applied to the scores,
  so numerator and `Z` both see it and the normalisation holds; also used in decoding), and for samples with exact
  cuts a **framewise char loss** (`frame_weight`, segmental only) next to the boundary BCE (`boundary_weight`).

`ctc_prior`: CTC with `log p - 0.3 log prior` during training; the prior is an EMA (momentum 0.999) of the mean
posterior over valid frames, stored in the model. Decoding uses plain posteriors (`decode_prior_scale` 0; a pilot knob).

## Placement

Posteriors over the text graph give, per unit, the distribution of its entry and exit frame (occupancy minus
self-loop mass). Segmental: a letter spans expected entry to expected exit; the cut between two letters is the mean of
exit and next entry, its spread from the variances. CTC: the k2 rule on expected spike positions (cut midway between
spikes, word ends half a median step outside). Then word boundaries snap into ink gaps on the strip (k2's rule:
widest near-empty column run near the raw boundary, inner cuts rescaled), the outer edges of the line move to the
nearest ink edge within half a letter, cuts are made monotone, mapped through the slant (slanted quadrilaterals,
clockwise from top left, top/bottom = the line's ink band) and clipped to the crop. Optional units with use
probability < 0.5 get zero width.

- `q` (char) = P(both cuts within 0.25 letter) under Gaussian spreads (plus frame quantisation); word `q` the same
  with 0.5 letter. Letter = line ink width / aligned units (the eval metric's unit).
- `line_fit` = `exp(min(0, log F_text - log F_best) / units)`, `F_best` the forward score of the model's own best
  reading (free Viterbi for segmental, greedy for CTC) through the same decode graph.
- word `fit` = `exp(min(0, expected emission of the text path - best frame scores) / units)` over the word's frames.
- Conventions (inside the contract): a combining mark gets its base's span; a space is zero width at the previous
  word's end, `aligned=False`; no-ink markup is zero width, `aligned=False`; a skipped optional unit is zero width,
  `aligned=True`; words without units are zero width with `q=0`, `fit=1`; lines too short for their text fall back
  to even spacing over the ink with `extra.fallback="even"` and `line_fit=0`.

## Measured

`uv run pytest`: 36 passed in 8.5 s (26 mine; the training test about 5 s). `prepare` on the pool sample: 15 ms per
line mean, 50 ms max (CPU).

Local check of the whole chain (CPU, about 6 min per run, tiny model 0.33 M params, 180 DejaVu lines 20 to 29 px,
half of them with exact cuts, 1,500 steps; error on 20 held-out lines in letter widths):

| run | val CER | letter starts: median / within 0.25 / within 0.5 | word edges within 0.25 (with snap) |
|---|---|---|---|
| segmental, no anchors | 6.5 % | 1.96 / 4 % / 8 % | 33 % |
| segmental + ink prior, boundary BCE | 0 % | 0.11 / 74 % / 85 % | 93 % |
| segmental + ink prior + frame loss (default) | 0 % | **0.07 / 100 % / 100 %** | 98 % (100 % without snap) |
| ctc + ink prior | 0 % | 0.26 / 49 % / 85 % (snap: 0.17 / 59 % / 91 %) | 93 % |

This is one synthetic font, in domain; it shows that coordinates, graph, posteriors and cuts fit together, not how the
objectives compare on real hands. That is the pilot's job.

Geometry on the pool sample (117 lines, 3 per source, inspected on contact sheets): straight, deslanted and scaled as
intended for about 110. Off: dense crops with three equally strong lines (semper, zenodo19c, inzigkofen; the strip
keeps all three, small), a noisy bern_xix crop, two Humboldt crops (an underlined single word, a sparse line).

## Open

- The pilot decides snapping (it costs a little when the model is already exact), `ink_prior` strength,
  `frame_weight`, `decode_prior_scale`, training spaces required vs optional. Real-data behaviour of the anchors is
  untested: real lines carry no cuts, so only the ink prior and the synthetic share anchor them.
- The graph runs one Python loop over time (alpha; beta fused with the boundary gradient; the free graph via autograd).
  On GPU this is many small kernels; if the pilot shows it dominating, CUDA graphs or a fused kernel are the next step.
- Geometry failure cases above (about 5 % of the sample). The x-height still drives the scale; a crop with no clear
  core falls back to clamps (x-height between 0.1 and 0.4 of a box crop's height).
- Whole-word skips next to garbage can count one labelling twice (penalised paths only); harmless for decoding.

Notes for the other agents:

- Eval (C): `make_placer("littera:<dir>")` works with a run directory or a model directory. Training reads
  `gold_cuts` exactly as in the eval-set contract (list of `{i, x}`, crop px). For the next round, the synthetic
  generator can write pool-style parquet rows (`image`, `text`, `width`, `height`, `split`, empty `polygon`/`baseline`,
  `gold_cuts`); `TrainConfig.pool` takes several paths, so pool and synthetic shards mix without code changes.
- Data (B): `train.py` uses its own small parquet reader (same columns); it can move to `littera.data.pool` later.
  The sample has 0 polygons and 266 of 437 rows with baselines. One text looks wrong for its crop:
  `konzilsprotokolle/Copy_of_1796-97/Copy_of_1796-97_00000004/10` has "General in Aber . M Amor Catechisiren dem das
  Honoraria", the crop reads "General en Chef, Baron Cederström, dem das Gouverne-" (held-out source, so evaluation
  only, but mismatch sets and fit AUC would count it as a right line).
