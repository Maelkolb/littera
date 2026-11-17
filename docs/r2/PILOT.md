# Round 2, agent D: markup presets and pilot

(draft, filled in as the round goes)

## 1. Markup inventory and presets

Every training and held-out source's texts in `/pool/v1/index.parquet` (328,517 rows) were counted for markup characters, and
93 crops with the doubtful marks were looked at on contact sheets (12 groups: square brackets per source, `[...]`, `〚〛`, `$`,
`#`, apostrophes, single `~`, `|`, line-end `¬`, line-end `_`, `@`, `{`).

What the crops show:

| mark | sources | ink? | treatment |
|---|---|---|---|
| `<u>...</u>`, `<sic>`, `<unclear>` | aaeb, basel_grundbuch, bern_xix, grandtour, hanse, semper, unibe_xx, zenodo19c, abp, klosterneuburg | tags no ink | no ink |
| `~~...~~` | aaeb, semper (18 % of rows), nuremberg (13 %), hanse, zenodo19c, ... | struck text has ink, the markers none | markers no ink |
| `^...^`, lectaurep's single `^` | lectaurep, basel_grundbuch, aaeb, hanse, read_icfhr16, ... | raised letters have ink, the markers none | `^` no ink |
| `〚...〛` (mkn), `⟦⟧` (catmus) | mkn | struck text (seen struck on the crop) | markers no ink |
| `[...]`, `[…]` | nuremberg, colmar_adhr, zenodo19c, faithful, hanse | unread or lost text: ink of unknown letters, or none | no ink (as before; the neighbours or the line-end garbage take the ink) |
| `[?]`, `[sic]`, `[sic!]`, `[illisible]` | emanuscripta, fanny, mkn, grandtour, zh_rrp, bern_xix, lectaurep, icfhr2018 | editorial notes | no ink |
| other `[...]` | grandtour `10t[en]`, `u[nd]`; icfhr2018 `R[icchezz]a`; fanny `Columne[n]`; tridis `[H]erveus`; zh_rrp, mkn, scads, semper, hisclima `[Temp]` | supplied letters and editorial words, no brackets on the page | brackets no ink, content optional (`brackets`) |
| other `[...]` | austrian_newspapers (print), basel_grundbuch `[1399]`, `Herr [Johann]`, bern_xix, unibe_xx (typewritten), lectaurep `[d'aubervilliers` | brackets written or printed | ink (left as characters) |
| `$...$` | tridis | struck text (seen struck) | `$` no ink (`dollar`) |
| `|` | tridis, icfhr2018 `in|den`, inzigkofen `ieglicher|mit` | not on the page | no ink (`bars`); hanse and mkn `|: ... :|` are written brackets and stay ink |
| `{` `}` | tridis (interlinear insertion marks) | not on the page | no ink (`braces`) |
| `'` `’` | aaeb, tridis (French elisions) | usually not written | optional (`apostrophe`) |
| line-end `¬` `⸗` `=` `-` `~` | zh_rrp (30 %), bern_xix, hanse_xvii, mkn, ... | sometimes a written hyphen, sometimes none | optional, now also when struck-text or tag markers follow |
| `*`, `/`, `@` | nuremberg, enhg15 | abbreviation sign, virgule, sign written like a 2 | ink (characters) |
| `#` | aaeb (a written sign), mkn `heil#:` (an unread letter), enhg15 (line end) | ink or unknown | ink (a class of its own) |
| line-end `_`, `‗` | zenodo19c, unibe_xx | a written hyphen (seen) | ink |
| single `~` mid-line | nuremberg (583 rows), lectaurep (a dash) | unclear (nuremberg), ink (lectaurep) | ink |

Presets (`littera.text`): the old ones (`none`, `plain`, `tags`, `editorial`, `all`) are unchanged. New rules combine with `+`:
`markup` (tags, `~~`, `〚〛⟦⟧`, `^`, editorial notes, line-end hyphen and `~` optional after all markers are removed), `brackets`
(other square brackets editorial: no ink, content optional), `dollar`, `bars`, `braces`, `apostrophe`. `Codec.build` takes one
preset per text. Each source's preset is the `markup:` key in `configs/sources.yaml`; `modal/train.py` reads them into
`TrainConfig.presets` (source -> preset; the synthetic rows are `plain`, since their texts are rendered with every character),
and `LineDataset` picks the preset by the row's `source`. A model's default preset (for placing new text) is `markup`.

Effect on the pool (units removed against `plain`, rows changed): semper 3.4 % / 18.6 %, nuremberg 1.5 % / 13.8 %, zenodo19c
1.5 % / 6.7 %, klosterneuburg 1.5 % / 5.4 %, basel_grundbuch 1.1 % / 4.4 %, aaeb 0.7 % / 16.8 % (apostrophes optional), others
below 0.5 % of units.

## 2. GPU smoke

`modal/train.py::smoke` (one container, the five pilot configs in turn, 30 steps each on the real pilot data, then
`Aligner.load` and `littera.eval.run` on 10 GW20 and 10 synthetic lines on the container's CPU). L4 had no capacity tonight
(two attempts waited 5 minutes "to be scheduled on a GPU_L4 worker"); the functions now take `["L4", "A10G"]`, and the smoke ran
on an A10G. A first attempt that fell back to a T4 (15 GB) ran out of memory with 60,000 columns per batch; the training loop
now skips a batch that runs out of memory and logs it (`event: oom`), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

A10G, 4 CPU cores (8 threads), 8 loader workers, batches of 40,000 strip columns (34 lines on average):

| config | params | s/step | lines/s | peak GPU memory | nll/unit step 10 -> 30 | eval CPU s/line (GW20, synth) |
|---|---|---|---|---|---|---|
| vgsl + kraken, ctc | 3.82 M | 0.45 | 79 | 3.6 GB | 36.5 -> 3.8 | 0.36, 0.18 |
| resnet + bilstm, ctc | 6.99 M | 0.73 | 48 | 8.4 GB | 30.3 -> 3.7 | 0.56, 0.25 |
| resnet + bilstm, ctc_prior | 6.99 M | 0.72 | 49 | 8.4 GB | 16.7 -> -5.8 (prior-shifted scores) | 0.54, 0.25 |
| resnet + bilstm, segmental | 6.99 M | 1.32 | 27 | 8.5 GB | 26.9 -> 4.2 | 0.64, 0.28 |
| resnet + local, segmental | 6.86 M | 1.20 | 29 | 8.2 GB | 29.7 -> 4.6 | 0.60, 0.27 |

Data loader alone: 133 lines/s; the GPU never waited for data (wait share 0.1 to 0.2 % after the first steps). Reading the
62,506 training rows from the volume took 2.6 s, codec and dataset setup 5 to 9 s. Every checkpoint loaded through
`Aligner.load` and placed its rows without errors. The step time is dominated by the Python loops over frames in the
alignment graph (the k2 config with the same CTC graph takes 0.45 s; segmental adds the free-partition pass and the boundary
sums), not by the CNN, so larger batches buy throughput nearly for free; the pilot uses 64,000 columns and at most 72 lines.
`graph.py` now precomputes the per-frame arc weights and the active-frame mask once per call (same numbers, fewer kernels).

Two fixes found on the way:

- `ctc_prior` trained as plain `ctc`: the trainer built its graph options with topology `ctc` for both CTC objectives, and
  `graph.nll` applies the prior only when the topology is `ctc_prior`. Fixed (test `test_ctc_prior_reaches_the_loss`).
- Validation of `ctc_prior` used the moving prior, so `best/` would have been picked on a moving target; validation now scores
  with `decode_prior_scale` (0 by default, the posteriors decoding uses).

## 3. Pilot grid

Data (`modal/pilot.py::draw`, fixed once, `/vol/pilot/v1/`): the index was read once at 22:37 and the draw written as parquet
(`lines/pool/<source>-part-NNNNN.parquet`, uid lists in `draw.json`, seed 20261110): 50,006 `train` rows, stratified over the
31 training sources in proportion to their pool share (one sixth of every source's train rows), 999 `val` rows the same way, and
30 `test` rows per source (930) for the read rate. Agent E's Laubmann rows (heldout) are not in it. Synthetic share: the first
five `/vol/synth/v1` parts, 12,500 rows, 20 % of the 62,506 training rows (source `synth`, preset `plain`, frame and boundary
losses where the row has cuts).

Same for all five runs (`configs/pilot/<run>.json`, launched with `modal/train.py::main --config`): steps 800 (see below),
warmup 120, lr 1e-3 cosine to 5 %, AdamW wd 0.01, batches of at most 64,000 strip columns and 72 lines (56.5 lines on
average), seed 0 (the same 45,164 lines in the same batches in every run), augmentation defaults, graph defaults (spaces required,
garbage at the line ends, ink prior 2 nats), boundary weight 1, frame weight 1 (segmental only), validation on 600 val lines at
steps 400 and 800. Only `model.backbone`, `model.context` and `objective.topology` differ.

Steps: the configs say 1,200; all five were stopped right after their step-800 validation (checkpoint committed) because the
billed rate turned out higher than planned (about 1.83 USD per A10G container hour with its 4 CPU cores: CPU billing was half of
the GPU's), and 1,200 steps would have left nothing for the evaluation. The cosine schedule was therefore cut at two thirds (lr
3.4e-4 at the stop); this is the same for every run.

### Segmental decoding needs the garbage it was trained with

The first evaluation of `pilot-seg` placed GW20 word edges 3.24 letters off (median; 9 % within 0.25) although its free reading of
the same lines was close to right ("Winchester, and abont two honsand wigkt"), and its line fit did not separate right from
wrong texts (AUC 0.51). The decode graph (`place.DECODE`, saved into every model's config) uses the permissive `filler` garbage
(score `log(1 - p(gap)) - 1` per frame, no boundary factors) at the line ends and between words. Against an 800-step segmental
model, whose letter posteriors are still spread over a few classes, a filler frame is cheaper than a letter frame plus its
boundary costs, so the graph puts whole words into garbage and squeezes the text into the rest of the line (first word of
"Winchester, and about two thousand weight" placed at x 626 instead of 46). Training used `garbage="any"` (an `<any>` segment
inside the normalised space). On 30 GW20 lines:

| decode garbage | segmental word-edge median | within 0.25 / 0.5 | ctc word-edge median |
|---|---|---|---|
| filler, ends and between words (default) | 3.27 | 8 / 16 % | 0.31 |
| filler, ends only | 11.94 | 2 / 4 % | 0.31 |
| any (as trained), ends and between | 0.34 | 37 / 69 % | 0.31 |
| none | 0.34 | 37 / 69 % | - |

CTC models are indifferent (their posteriors are spikes and blanks). Fix: `model_config_dict` now saves `garbage="any"` for
segmental models unless the config says otherwise (test `test_segmental_models_decode_with_the_trained_garbage`). The two
segmental pilot checkpoints were saved before the fix, so they are evaluated with that one decode option changed
(`pilot-seg@gany`, `pilot-seg-local@gany` in `modal/pilot.py`); the default-decode numbers of `pilot-seg` are kept in the table
for reference.

## 4. Results

Agent D's session ended during the evaluation; the orchestrator resumed it (finished shards were kept) and wrote sections
4 to 6. Every number is in `runs/eval/pilot.json` (pilots) and `runs/eval/baselines.json` (round 1). Errors in letter widths;
`%` = share within the limit. Pilots: 800 steps, about one pass over the 50k lines plus 12.5k synthetic rows.

| placer | GW20 word edge median | GW20 within 0.25 / 0.5 | GW20 ink-gap rate | IAM median / within 0.5 | synth letters within 0.25 / 0.5 | synth chars, both cuts within 0.25 | char q ECE | mismatch AUC replace / delete / swap | word-fit AUC | in-domain read rate |
|---|---|---|---|---|---|---|---|---|---|---|
| even spacing | 0.66 | 21 / 40 | 42 | 0.45 / 54 | 31 / 56 | 22 | 0.28 | 0.50 / 0.50 / 0.50 | 0.50 | - |
| kraken aligner-k2 | 0.29 | 43 / 79 | 93 | 0.03 / 79 | 75 / 92 | 65 | 0.33 | 0.69 / 0.66 / 0.97 | 0.84 | - |
| kraken McCATMuS | **0.28** | **46 / 77** | **94** | **0.01 / 97** | 76 / 93 | 65 | 0.33 | 0.72 / 0.73 / 0.99 | 0.86 | - |
| pilot-k2 (vgsl + kraken, ctc) | 0.32 | 38 / 73 | 87 | 0.08 / 86 | 62 / 88 | 52 | 0.17 | 0.56 / 0.42 / 0.75 | 0.70 | 14 |
| pilot-ctc (resnet + bilstm) | 0.32 | 37 / 75 | 89 | 0.06 / 95 | 74 / **95** | 65 | 0.33 | 0.79 / 0.72 / **1.00** | **0.95** | **64** |
| pilot-prior (resnet + bilstm, ctc_prior) | 3.46 | 10 / 19 | 59 | 1.20 / 40 | 17 / 29 | 13 | 0.24 | 0.50 / 0.50 / 0.55 | 0.54 | 9 |
| pilot-seg (resnet + bilstm, segmental) | 0.34 | 34 / 71 | 89 | 0.06 / 92 | **86 / 95** | **83** | **0.11** | 0.77 / 0.76 / **1.00** | 0.86 | 53 |
| pilot-seg-local (resnet + local, segmental) | 0.35 | 33 / 70 | 89 | 0.06 / 91 | 85 / 94 | 82 | 0.13 | **0.80 / 0.77 / 1.00** | 0.85 | 57 |

Segmental runs are scored with the trained garbage (`@gany`).

What the pilot shows:

- Letters: the segmental objective places letters inside words much better than any CTC placer: 86 % of synthetic letter cuts
  within a quarter letter against 74 to 76 % (pilot-ctc, k2, McCATMuS), 83 % of characters with both cuts within a quarter
  letter against 65 %. Its per-character `q` is calibrated (ECE 0.11 against 0.33). Caveat: the synthetic set comes from the
  same generator as the boundary anchor (other fonts). Only the owner's letter gold can confirm this on real hands.
- Word edges on a real hand (GW20): all trained placers are within 0.06 letters of each other. The fully trained kraken models
  lead slightly (0.28 to 0.29), the 800-step pilots follow (0.32 to 0.35). Word edges are mostly decided by ink-gap snapping.
- Mismatch: every resnet pilot detects a replaced or deleted word better than the kraken baselines (AUC 0.77 to 0.80 against
  0.69 to 0.73; word fit 0.85 to 0.95 against 0.84 to 0.86). Swapped lines are separated perfectly.
- `ctc_prior` collapsed: it reads 9 % of the characters and places worse than even spacing (see the decoding check below).
- `vgsl` + `kraken` (k2's design) is the slowest learner: 14 % read rate after 800 steps, behind on every measure.
- Local context against BiLSTM: no difference that matters (letters 85 against 86 %, mismatch slightly better, read rate 57
  against 53 %, 10 % slower per step).

Decoding check for `ctc_prior` (`runs/eval/pilot_prior_decode.json`): dividing the posteriors by the prior at decoding does not
rescue it (GW20 word-edge median 4.27 with prior scale 0.5, 4.78 with 1.0; synthetic letters within 0.25: 14 and 12 %; read rate
11 and 7 %). The model itself collapsed in training: the prior was subtracted at full scale from the first step. The paper's
recipe starts the prior after a warm-up and at a scale around 0.3 to 0.5. Not pursued: the segmental objective already does
what the prior was meant to do.

The marking page (the local folder `Littera_Letter_Gold`) is rebuilt with `pilot-seg@gany` cuts (`/vol/eval_runs/
littera__vol_runs_pilot-seg_gany/letter-gold-candidates`).

## 5. Recommendation for the main run

Decision rule once the owner's letter gold is in: score `pilot-seg@gany`, `pilot-ctc`, k2 and McCATMuS on it. If the
segmental pilot beats the CTC placers on real letters, as on the synthetic set, the main run is segmental. If it does not,
ship a CTC model (PLAN, exit after phase 3).

Main-run config (segmental case):

| setting | value | why |
|---|---|---|
| model | `resnet` + `bilstm`, char + boundary heads (6.99 M) | equal to `local` on every measure, slightly better letters |
| objective | `segmental`, ink prior 2 nats, frame weight 1, boundary weight 1 | as piloted |
| decode | `garbage = "any"` (saved in the config now), snapping on, optional spaces | the trained garbage; snapping as piloted |
| data | the full pool train split (299,992 lines) + synthetic rows at 20 % (about 75,000: 55,000 more from `build_synth_train`, about 0.30 USD) | the pilot used one sixth of the pool |
| length | 2 passes, about 13,400 steps of at most 64,000 columns / 72 lines | the pilot's 800 steps were about one pass over 62k rows |
| schedule | lr 1e-3, warm-up 300, cosine to 5 % | as piloted, now run to the end |
| noise | after the first pass, drop each source's rows whose loss per unit is in the top 5 % of that source | the silver sources (bern_xix, zh_rrp, aaeb, basel_grundbuch, unibe_xx) without a separate scoring pass |
| hardware | A10G (L4 if available), 2 CPU cores instead of 4 | GPU-bound at about 48 lines/s; the loader needs fewer cores |

Estimated cost: about 4.3 GPU hours at 1.5 to 1.8 USD per hour, 6.5 to 8 USD, plus 1 USD for the final evaluation. The step
time is set by the per-frame Python loop of the alignment graph (CNN and data are idle most of the time); a fused recursion
(CUDA graph or a small Triton kernel) could halve it and is worth trying before the main run.

Knobs left for the letter gold: snapping on or off inside words, ink-prior strength, garbage enter and frame costs, optional
spaces, and how far a word end may move to an ink gap.

## 6. Spend

Round 2, agent D and the resumed evaluation: training 2.38 USD (smoke and five pilots), evaluation about 1.55 USD (pilots, the
decoding checks, the candidate placements). Agent E 0.07 USD. Project total by 2026-11-10 morning: about 4.9 of 25 USD.
