# Round 3, agent T: pen-lift arcs, edges on the ink, anchors

2026-11-17. Code: `src/littera/graph.py`, `place.py`, `geometry.py`, `train.py`, `anchors.py`, `modal/eval.py`
(`decode_grid`), `modal/humboldt_ft.py`, `tests/test_lifts.py`. Numbers: `results/r3_approach/`. Humboldt scores are
agent M's measures (`docs/r3/MEASURE.md`) through `modal/humboldt.py::place`, `score`, `classify`.

## Summary

- The hypothesis of the round is not confirmed. Pen-lift arcs, letter edges on the ink and anchors did not make the
  letter crops on Humboldt more consistent than `littera-humboldt-ft2` with its inner edges left alone, and on exact
  synthetic gold they did not raise the share of letter cuts within a quarter letter (sections 2 and 5).
- What helped is agent M's finding, now an option: `inner_edges = "keep"` (word ends snapped, the letter edges inside
  the word left where the model put them). On exact synthetic gold `littera-pilot-seg` goes from 86.4 to 95.4 % of
  letter cuts within a quarter letter; on Humboldt dev the letter classifier goes from 38.0 to 46.1 % for ft2.
- No fine-tune beat ft2 with kept inner edges on dev: control -0.2, pen-lift topology +0.2, topology with anchors
  -0.6, the limited-context model from the pilot with 3,000 steps -0.5 classifier points on all letters, all inside or
  at the edge of the classifier's noise. `/vol/models/littera-humboldt-ft3` is therefore the weights of ft2 with
  `inner_edges = "keep"` in its config. Report half: 52.9 % of letters named, against 44.3 % for ft2 as deployed and
  46.7 % for aligner-k2 as deployed.
- Inside words aligner-k2's unsnapped edges are still ahead of littera's by 2.0 points on dev (4.7 in connected words).

## 1. What was built

Everything is off by default. With the defaults the graph has the same states, arcs and weights as before
(`test_options_off_build_the_released_graph`), and 60 placements (5 GW20 and 5 synthetic lines, both released models,
the three snap modes) are identical as JSON before and after the change (checked again after the last edit).
`uv run pytest`: 72 tests pass (56 old, 11 new in `tests/test_lifts.py`, 5 of the orchestrator's `tests/test_consensus.py`).

New fields of `GraphOptions` (saved in `config.json["decode"]`, read by `Aligner.load`):

| field | default | values |
|---|---|---|
| `lifts` | `false` | optional gap state between two letters of a word |
| `lift_enter`, `lift_frame` | 0, 0 | cost of entering a lift, cost per frame in it |
| `lift_snap` | `"off"` | `"lifts"`: letter edges next to a decoded lift go to the ink edges of its gap; `"cuts"`: every cut inside a word with an ink-free run within 1.5 frames |
| `lift_edges` | `"ink"` | `"mid"`: both letters share the middle of the run |
| `inner_edges` | `"rescale"` | `"keep"`: word-end snapping leaves the inner cuts where the model put them |

### Pen-lift arcs (`graph.py`)

`lifts` puts one optional state after every letter of a word except the last (role `LIFT`). Its arcs:

| arc | kind | weight |
|---|---|---|
| letter -> lift | ENTER (the boundary head's `log b` at the lift's first frame) | `-lift_enter` |
| lift -> lift | STAY (`log(1 - b)`) | 0 |
| lift -> next letter | ENTER (`log b` at the letter's first frame) | 0, plus the skip penalty where an optional letter is skipped |

A frame in the lift state scores `score[gap] - lift_frame`, and only on frames that hold an ink-free column of the strip's
core profile (`Strip.open_frames`, the profile and threshold that the word-edge snapping uses); elsewhere it costs 10,000.
So a lift cannot become a blank between two touching letters, which is what the segmental objective was chosen to avoid.

- No chains of gaps: a lift is entered from its own letter only, it does not lead into a junction (word gap, garbage) or to
  the end of the line, and `connect` refuses lift-to-gap arcs. A lift next to a skipped optional letter still leads to the
  letter after it.
- Normalisation: a path through a lift is a path of the free model with the labelling letter, gap, letter (two ENTER
  events, STAY inside), and with `lift_enter >= 0`, `lift_frame >= 0` its weight is not larger. No two paths of one
  transcript graph map to the same free path (`test_lift_paths_are_distinct_paths_of_the_free_model` enumerates them), so
  the free partition stays an upper bound and the loss stays non-negative
  (`test_lift_likelihood_stays_under_the_free_partition`, every transcript over two letters on five frames). What is lost:
  the transcripts "ab" and "a b" now share paths, so the likelihoods of all transcripts no longer sum to one. Negative
  costs (a bonus) are for decoding with a model that was not trained with lifts; they break the bound and must not be
  trained with.
- The boundary head has to fire twice at a lift, at its first frame and at the first frame of the next letter. A model
  trained without lifts fires once per letter boundary and gives the gap label little mass inside words, so the released
  models take the arc almost never without a bonus (section 2).

### Edges on the ink (`place.py`)

`lift_snap = "lifts"` sets the two letter edges next to a decoded lift (visited with posterior at least 0.5) to the ends
of the ink-free run that overlaps the lift's expected frames (one frame of slack on either side); `"cuts"` does the same
for every cut inside a word, without any lift arcs (the control for "is it the topology or the snapping"). The run is
looked for on the strip that the word-edge snapping uses: the deslanted strip for `snap = "strip"` and `"off"`, the
upright line for `snap = "upright"`. With `lift_edges = "ink"` the letter before the gap ends at the run's start and the
letter after it starts at the run's end (`CharPlace.x1` of one is then smaller than `x0` of the next; `Placement.cuts()`,
which agent M's placement files use, takes each letter's `x0`, so there a letter's crop runs from its own ink to the next
letter's ink). Word edges are not touched.

### Inner edges kept (`place.py`, after agent M's finding)

`_snap` moves a word's two ends into the ink gap and, as released, rescales every cut inside the word in proportion.
With `inner_edges = "keep"` the inner cuts stay, provided all of them lie at least a quarter of the word's mean letter
width inside both snapped ends; otherwise that word is rescaled as before (the orchestrator's refinement: a plain clip
collapsed letters at the side of a word whose end moved inward past them). On dev 41 of 2,570 words with more than one
letter take the fallback (1.6 %). `Analysis.extra["words_rescaled"]` counts them per line.

### Anchors (`anchors.py`, `train.py`)

`anchors.derive` takes two analyses of one line with `snap = "strip"` and inner edges kept: without lift arcs and with
`lift_snap = "cuts"`, and with lift arcs (enter bonus 8) and `lift_snap = "lifts"`. It keeps what both agree on:

- a pen lift: the same ink-free run chosen for the same pair of adjoining letters by both,
- a word gap: both snapped the two word edges to the two ends of one ink-free run,
- line start and end: the first and last letter edge at an ink edge in both,
- a single letter: a letter with an anchored gap on either side and at most 2.5 mean letter widths between them.

Lines whose line fit is below 0.2 in either analysis give nothing. A record holds gaps as crop x positions, so the
augmentation can shift them; `anchors.frame_targets` finds each run again on the strip it is given (a run within one
frame, else the anchor is dropped for that sample) and returns frame labels (gap on the run's frames, the letter on a
single letter's frames, ignore elsewhere), boundary targets (1 at the run's first frame and at the first frame after it,
0 inside runs and inside single letters) and the mask of frames that have a target. `LineDataset` passes them to the
existing `bce` and frame losses (`TrainConfig.anchors`); the collate function takes a per-frame mask where a row has one.
`TrainConfig.lift_cuts` rewrites the exact cuts of synthetic rows into the same convention (gap frames between two
letters that do not touch); it is tested, and no run of this round used it.

## 2. The decoding change alone

### Exact gold: `littera-pilot-seg` on held-out synthetic fonts and GW20

`modal/eval.py::decode_grid`, `snap = "strip"`. Each set is split by row parity. The penalties were chosen on the even
rows (400 lines per set) and the table after it is the odd rows, which were run once with the chosen settings.

Tuning part, 400 synthetic lines (about 11,600 letter cuts) and 400 GW20 lines:

| decode | synthetic letter cuts within 0.25 letters, % | median error | lifts taken | GW20 word edges within 0.25, % | GW20 ink-gap rate, % |
|---|---:|---:|---:|---:|---:|
| released | 87.0 | 0.084 | 0 | 33.7 | 88.1 |
| released + every cut snapped to ink (`cuts`) | 89.0 | 0.034 | 0 | 33.7 | 88.1 |
| released + lift arcs, no bonus | 87.1 | 0.084 | 27 | 33.8 | 88.0 |
| released + lift arcs, enter bonus 4 | 87.2 | 0.080 | 493 | 34.3 | 87.9 |
| released + lift arcs, enter bonus 8 | 89.2 | 0.052 | 4,867 | 35.3 | 87.1 |
| released + lift arcs, frame bonus 3 | 82.1 | 0.090 | 1,098 | 20.7 | 62.9 |
| inner edges kept | **95.9** | 0.056 | 0 | 33.7 | 88.1 |
| kept + `cuts` | 95.4 | 0.030 | 0 | 33.7 | 88.1 |
| kept + lift arcs, enter bonus 6 | 95.6 | 0.046 | 1,954 | 34.9 | 87.4 |
| kept + lift arcs, enter bonus 8 | 95.5 | 0.037 | 4,867 | 35.3 | 87.1 |
| kept + lift arcs, enter bonus 10 | 95.0 | 0.031 | 7,884 | 35.0 | 86.5 |
| kept + lift arcs, enter bonus 12 | 94.4 | 0.031 | 9,225 | 34.7 | 86.0 |
| kept + lift arcs, enter bonus 10, frame cost 1 | 95.5 | 0.034 | 6,447 | 34.5 | 88.0 |

Lift arcs are decoded with `lift_snap = "lifts"`. A bonus per frame lets lifts swallow word gaps and wrecks the word
edges (frame bonus 6: 18.1 % of GW20 word edges within 0.25), so only the enter bonus was searched further. Chosen for
the report part: enter bonus 6, no frame cost (the best share within a quarter letter among the lift variants).

Report part, 1,000 synthetic lines (28,971 letter cuts) and 328 GW20 lines (2,436 words):

| decode | letter cuts within 0.25, % | median error | characters with both cuts within 0.25, % | lifts taken | GW20 word edges within 0.25, % | GW20 ink-gap rate, % |
|---|---:|---:|---:|---:|---:|---:|
| released | 86.4 | 0.084 | 82.5 | 0 | 34.1 | 89.6 |
| inner edges kept | 95.4 | 0.056 | 92.1 | 0 | 34.1 | 89.6 |
| kept + `cuts` | 94.7 | 0.031 | 89.5 | 0 | 34.1 | 89.6 |
| kept + lift arcs (bonus 6) + lift edges | 95.5 | 0.046 | 91.4 | 5,066 | 35.0 | 89.2 |
| kept + lift arcs (bonus 6), no edge snapping | 95.4 | 0.048 | 92.2 | 5,066 | 35.0 | 89.2 |

- The proportional rescale of the released snapping costs 9 points of letter cuts on exact gold; keeping the inner
  edges gives them back. This is the one large effect, and it is agent M's finding, confirmed here where the letter
  cuts are known.
- With the inner edges kept, the pen-lift decoding does not raise the share of cuts within a quarter letter (95.4
  against 95.5 and 94.7 %). It makes the typical cut more exact (median 0.056 to 0.046 and 0.031 letter widths) and
  adds a few large errors, cuts moved into a gap that belongs to another boundary or lies inside a letter. Under the
  released rescale the same snapping looked like a gain of 2 points (87.0 to 89.0 %), because it repaired part of
  what the rescale had moved.
- A released model takes a lift arc almost never without a bonus (27 lifts on 400 lines).
- GW20 word edges gain about one point from the lift arcs (34.1 to 35.0 % within a quarter letter) and the ink-gap
  rate falls by 0.4.

### Humboldt dev, `littera-humboldt-ft2`

Agent M's `place` (`snap = "upright"`, baseline passed) and `classify`. Three classifier jobs; a difference is paired
within its job, against the first row of the job.

| decode | lifts held (1.5 px), % | letters named, % | difference |
|---|---:|---:|---:|
| job A, released rescale: ft2 as deployed | 37.9 | 38.0 | |
| + every cut snapped to an ink run of the deslanted strip | 40.6 | 36.9 | -1.1 |
| + lift arcs, enter bonus 8, lift edges | 37.2 | 37.6 | -0.4 |
| job B, inner edges kept (plain clip): ft2 | 48.6 | 46.3 | |
| + `cuts` on the deslanted strip | 44.4 | 44.6 | -1.7 (-2.2 to -1.2) |
| + lift arcs, enter bonus 8, lift edges on the deslanted strip | 44.9 | 45.1 | -1.1 (-1.6 to -0.6) |
| job C, inner edges kept: ft2 | 48.6 | 46.2 | |
| + `cuts` on the upright line, ink edges | 55.7 | 45.4 | -0.8 (-1.3 to -0.3) |
| + `cuts` on the upright line, middle of the run | 63.0 | 45.9 | -0.3 (-0.8 to 0.2) |

Moving letter edges onto ink gaps at decoding time makes the letter crops slightly less consistent on Humboldt in
every variant. Two things were learnt on the way. Gaps of littera's deslanted strip are not the upright ink-free runs
of agent M's neutral strip: snapping to them lowered the share of M's lifts that hold an edge (48.6 to 44.4 %), so
for upright crops the run has to be looked for on the upright line, which is what `snap = "upright"` now does. And
"lifts held" rises to 63 % for a placer that snaps to upright runs while its classifier score does not rise: for
such a placer the measure is circular, as M's report says.

## 3. Anchors

`modal/humboldt_ft.py::anchors`, ft2, about one CPU second per line for the two analyses, all 19,655 training lines
(`results/r3_approach/anchors.json`). Two of the ten shards were preempted and came late: the two runs with anchors
trained with the other eight (15,570 lines, the same rates within 3 %, `anchors_8of10.json`).

| per 1,000 pool lines | |
|---|---:|
| lines with at least one anchor | 928 |
| pen lifts both analyses agree on | 1,498 |
| word gaps | 4,932 |
| line starts, line ends | 790, 599 |
| single letters with both edges anchored | 1,234 |
| for comparison: cuts of the plain analysis with an ink-free run within 1.5 frames | 5,549 |
| for comparison: lifts taken by the analysis with lift arcs (enter bonus 8) | 3,195 |

A line has about 26 letter boundaries inside words, so an anchored pen lift sits at 6 % of them; the plain analysis
finds an ink-free run next to 21 %. The 24,257 single letters are 3 % of the letters and are not the letters of running
words: 34.6 % digits, 22.1 % punctuation and signs, 6.9 % capitals, 36.4 % lower case. The commonest: `1` 6.5 %, `.`
5.8 %, `u` 5.6 % (the abbreviation for "und", a word of its own), `2` 5.0 %, `4`, `3`, `p` (page references), `5`,
`à`, `6`, `a`, `0`, `(`, `s`. A left-edge convention learnt from these is the convention of digits and one-letter
words.

By eye (`docs/r3/anchors/`, two sheets of twelve tiles from shard 3, text context printed on each tile): the twelve
single letters (`u`, `1`, `u`, a sign, `G`, `d`, `(`, `.`, `1`, `i`, `p`, `g`) all sit between two real gaps and show
one letter; the twelve lifts (among them `Pap|pelwälder`, `Kalo|gheros`, `Ha|fen`, `aus|zuzahlen`, `d|ie`, digits and
brackets) are all gaps between the two letters named. I saw no wrong anchor in the 24 and cannot judge the two
narrowest tiles (`.` and `co|mme`, two pixels wide).

## 4. Fine-tunes

`modal/humboldt_ft.py`, one A10G each, pool `/vol/pool/humboldt_ft` (19,655 train lines, baselines present; the runs
read it at 20:54 and 22:08), batches of at most 48,000 columns and 64 lines, warm-up 100, cosine to 5 %, seed 0,
augmentation defaults, the checkpoint after the last step.

| run | from | objective | steps, lr | lines/s in training | validation nll per unit | validation CER, % |
|---|---|---|---|---:|---:|---:|
| `r3-control` | ft2 | as ft2 | 1,500, 2e-4 | 57.7 | 0.522 | 12.7 |
| `r3-lifts` | ft2 | graph with lift arcs (costs 0) | 1,500, 2e-4 | 57.0 | 0.514 | 12.6 |
| `r3-lifts-anchors` | ft2 | lift arcs + anchors (`bce` and frame loss on anchored frames) | 1,500, 2e-4 | 54.6 | 0.682 | 14.8 |
| `r3-local-lifts-anchors` | `pilot-seg-local` (limited context) | lift arcs + anchors | 3,000, 5e-4 | 48.8 | 0.992 | 23.2 |

The validation rows are agent M's new `val` split (992 lines of 31 pages that ft2 probably trained on; ft2's own
validation pages are not known), so these numbers compare the runs with each other only. `r3-lifts-anchors` was
preempted near step 1,250 and went on from its step-1,000 state. The limited-context run is the orchestrator's
choice for the fourth run; it starts from a model without Humboldt training and is not step-matched with the others
(ft2 had 5,500 steps from `pilot-seg`).

What the objectives did to the model, on 200 dev lines with 5,418 letter boundaries inside words:

| model | lifts taken when decoded with lift arcs (cost 0) |
|---|---:|
| ft2 | 15 |
| `r3-lifts` | 24 |
| `r3-lifts-anchors` | 479 (359 of them with an upright ink run for the edges) |

The topology alone is inert. The normalised likelihood is as content with a letter that owns the gap as with a lift,
and a model that starts from ft2 stays where it is. Only the anchors, which label the gap frames and ask the boundary
head to fire on both sides, move it.

## 5. Dev scores and the choice

Agent M's `score` and `classify` on the dev half (val and test_page, 400 lines, 12,872 letters, 2,665 of them inside
connected words). Every littera row: `snap = "upright"`, `inner_edges = "keep"`. One score job and one classifier job
for the first nine rows (`results/r3_approach/score_dev_final.json`, `classify_dev_final.json`); differences are
paired with "ft2, inner edges kept" inside the job.

| placer | word boundaries, % | lifts held, % | letters named, % | in connected words, % | all, minus ft2 kept | connected, minus ft2 kept |
|---|---:|---:|---:|---:|---:|---:|
| aligner-k2: raw-mid inner edges in its snapped words (`mix:`) | 92.4 | 49.8 | 48.1 | 49.0 | +2.0 (1.1 to 3.1) | +4.7 (2.7 to 6.4) |
| ft2: `off` inner edges in its `upright` words (`mix:`, agent M) | 92.9 | 48.6 | 46.3 | 45.4 | +0.2 (-0.2 to 0.6) | +1.1 (0.3 to 2.0) |
| **ft2, inner edges kept** (`inner_edges = "keep"`, the baseline) | 92.9 | 48.5 | 46.1 | 44.3 | | |
| (a) control: 1,500 more steps, old objective | 93.1 | 46.7 | 45.9 | 44.3 | -0.2 (-0.7 to 0.3) | 0.0 (-1.1 to 1.1) |
| (b) pen-lift topology, decoded as trained (lift arcs, lift edges) | 92.9 | 48.4 | 46.3 | 44.7 | +0.2 (-0.2 to 0.7) | +0.3 (-0.7 to 1.4) |
| (b) decoded without lift arcs | 93.1 | 48.4 | 45.7 | 43.8 | -0.3 (-0.8 to 0.1) | -0.5 (-1.6 to 0.5) |
| (c) pen-lift topology + anchors, decoded as trained | 91.1 | 51.9 | 45.4 | 44.3 | -0.6 (-1.2 to -0.1) | 0.0 (-1.1 to 1.1) |
| (c) lift arcs, no lift edges | 91.1 | 51.9 | 45.6 | 44.2 | -0.5 (-1.1 to 0.0) | -0.1 (-1.2 to 1.0) |
| (c) decoded without lift arcs | 91.3 | 41.0 | 44.0 | 44.0 | -2.0 (-2.6 to -1.4) | -0.4 (-1.7 to 0.8) |
| (d) limited context from the pilot, lift arcs + anchors, decoded as trained | 90.9 | 56.3 | 45.5 | 44.8 | -0.5 (-1.2 to 0.1) | +0.4 (-1.2 to 2.0) |
| (d) decoded without lift arcs | 90.8 | 37.8 | 42.8 | 43.6 | -3.2 (-4.0 to -2.5) | -0.7 (-2.3 to 0.8) |

The two (d) rows are from a second job (`classify_dev_local.json`), in which the baseline row came out the same (46.1 and 44.3).

For reference, ft2 as deployed until today (proportional rescale): 92.9, 37.9, 38.0, 36.8.

How much a difference means: the two ft2 rows differ only in the 1.6 % of words that take the fallback, and the
classifier puts them 1.1 points apart on connected words in this job (0.3 the other way round in an earlier job; the
`mix:` row itself scored 45.4 in three jobs and 44.5 in two). Each placer's classifier is trained on its own crops,
and how much that training moves the score is not in the bootstrap intervals. A difference below about one point on
the connected set, or half a point on all letters, is not a result.

- (a) More steps with the old objective change nothing.
- (b) The topology alone changes nothing: the model hardly uses the arcs (section 4).
- (c) The anchors make the model use lifts, and the result is not better: classifier -0.6 on all letters, 0.0 inside
  connected words, word boundaries -1.8 points. It holds more of M's lifts (51.9 against 48.5 %), which is circular
  for this model. Decoded without lift arcs it is clearly worse (-2.0), as expected of a model that has learnt to give
  gap frames away. The hope of the hypothesis was the connected words (a left-edge convention carried over from
  anchored letters): nothing moved there.
- (d) The limited-context model, started from the pilot without Humboldt training, reaches ft2's level in 3,000 steps
  with lift arcs and anchors (-0.5 on all letters, +0.4 in connected words, both inside the noise) and loses 2.0 points
  of word boundaries. It does not clear the bar below. It is not a test of limited context against the BiLSTM on
  equal terms: start, steps and objective all differ from ft2's.

Rule, fixed on dev before the report half was looked at: a fine-tune replaces ft2's weights only if its paired
difference to "ft2, inner edges kept" is above zero with the interval excluding zero on all letters or on connected
words, without a loss on the other and without more than one point of word boundaries. No run
clears it, so ft3 keeps ft2's weights and changes the decoding only.

`/vol/models/littera-humboldt-ft3`: the weights of `littera-humboldt-ft2`, `config.json` with

```json
"decode": {"topology": "segmental", "spaces": "optional", "space_penalty": 2.0, "skip_penalty": 3.0,
           "garbage_ends": true, "garbage_between": true, "garbage_enter": 2.0, "garbage_frame": 1.0,
           "garbage": "any", "ink_prior": 2.0, "lifts": false, "lift_enter": 0.0, "lift_frame": 0.0,
           "lift_snap": "off", "lift_edges": "ink", "inner_edges": "keep"}
```

`Aligner.load("/vol/models/littera-humboldt-ft3", snap="upright")` needs nothing else. Placed that way through
`modal/humboldt.py::place`, the 400 dev lines have the line fits of "ft2, inner edges kept" exactly and the same edges
up to rounding: 306 of 15,771 edges differ by 0.1 to 0.3 px between two containers (floating point on different
machines). The folder needs this round's code: an older littera does not know the new decode fields.

## 6. Report half

The chosen placement is "ft2, inner edges kept". The orchestrator scored it on the report half in his own job before
I did (`ft2-keep`, `results/r3/classify_consensus_final.json`, `score_consensus_final.json`), so I stopped my copy of
that job before it placed a line and did not look at the report half a second time. Report half: test_bench,
test_journal, letters; 600 lines, 21,278 letters, 4,415 of them inside connected words.

| placer | word boundaries, % | lifts held, % | letters named, % | in connected words, % | all, minus k2 as deployed | connected, minus k2 as deployed |
|---|---:|---:|---:|---:|---:|---:|
| **littera-humboldt-ft3** (ft2's weights, inner edges kept) | 94.0 | 38.0 | 52.9 (51.8 to 54.1) | 54.2 (52.4 to 55.8) | +6.2 (5.4 to 6.9) | +6.7 (5.1 to 8.1) |
| littera-humboldt-ft2 as deployed (`upright`, rescale) | 94.0 | 30.6 | 44.3 (43.2 to 45.3) | 45.0 (43.3 to 46.7) | -2.4 (-3.2 to -1.6) | -2.5 (-4.2 to -0.8) |
| aligner-k2 as deployed (`snap-mid`) | 95.1 | 31.5 | 46.7 (45.6 to 47.9) | 47.5 (45.8 to 49.2) | | |
| aligner-k2's raw-mid inner edges in ft2's snapped words | 94.0 | | 56.2 (55.1 to 57.4) | 59.1 (57.1 to 60.8) | +9.5 (8.8 to 10.2) | +11.6 (10.2 to 13.0) |
| the orchestrator's consensus of ft2's and k2's inner edges | 94.0 | 42.0 | 56.9 (55.7 to 58.1) | 59.3 (57.6 to 61.0) | +10.2 (9.4 to 10.9) | +11.9 (10.3 to 13.2) |

Against the two placements that are deployed, ft3 is 8.6 points ahead of ft2 and 6.2 ahead of aligner-k2 on the
letter classifier, with ft2's word boundaries (1.1 points behind k2's). All of that comes from not rescaling the inner
edges. With the same treatment, aligner-k2's inner edges are 3.4 points ahead of littera's on the report half (4.9 in
connected words; agent M's `mix:k2-raw-mid+k2-snap-mid` has 56.5 and 58.5).

## 7. Throughput

| | lines per second |
|---|---:|
| placing, one line at a time, ft2, on an A10G | 3.0 (same container, its 2 CPU cores without the GPU: 2.8) |
| the same on an A10 machine Modal gave for a later call | 4.6 (CPU: 4.1) |
| placing with lift arcs (the model with anchors) on an A10G | 3.2 |
| agent M's `place` on 2 CPU cores | 2.3 to 2.9 (0.34 to 0.44 s per line) |
| training, A10G, batches of 48,000 columns | 58 (old objective), 57 (lift arcs), 55 (with anchors), 49 (limited context) |

The GPU buys nothing for placement one line at a time: the frame loop of the alignment in Python sets the pace, as in
training (PILOT.md). Lift arcs nearly double the states inside words and cost 1 % of training speed, the anchors
another 4 %. A batched `analyse` would be the way to use a GPU for an index build.

## 8. What did not work, and why I think so

1. Edges on the ink at decoding time. On exact gold they make most cuts more exact and a few much worse; on Humboldt
   the classifier loses 0.3 to 1.7 points. In this hand an ink-free run inside a word is often not a letter boundary:
   M's report gives 20 runs per 100 letter boundaries, of which the best placers hold half, and hairlines below the
   ink threshold and open letter forms give runs too. A rule that moves every nearby cut into a run moves some into
   the wrong one.
2. Lift arcs without supervision. A trained model does not start to use them.
3. Anchors. The checks keep what two analyses of the same model agree on, so the anchors mostly repeat what ft2 does
   already, and the single letters are digits, punctuation and one-letter words. The model learnt the gap label at
   lifts, lost word boundaries (a lift and an optional word gap now compete for the same empty frames) and gained
   nothing in connected words.
4. My first version of the upright mode looked for lift runs on the deslanted strip. For upright crops that is the
   wrong strip (section 2).
5. The released proportional rescale was the largest fault found this round, and it was not in the hypothesis.

Limits of what was measured: no letter gold on Humboldt; the classifier's noise is about a point; one seed per run;
1,500 steps; anchors from eight of ten shards; `Placement.cuts()` hands the gap after a letter to that letter, so with
`lift_edges = "ink"` a letter's crop in M's measures includes the gap to its right (the `mid` variant was 0.5 points
better than `ink` on dev, still below no snapping).

## 9. What I would do next

1. Ship `inner_edges = "keep"` (done in ft3) and make it the default of the next release, with the fallback rule.
2. The gap to aligner-k2 inside words (2.0 points, 4.7 in connected words on dev) is the open question, and nothing of
   this round touched it. The orchestrator's consensus of the two placers' inner edges (49.6 on dev) says the two make
   different errors. Candidates, in the order I would try them: (i) ft2 never had a loss on exact cuts during
   fine-tuning; continue it with 20 % synthetic rows so that `bce` and the frame loss act on connected letters
   (`TrainConfig.lift_cuts` exists for the lift convention, plain synthetic rows need no new code); (ii) frame stride 2
   instead of 4, since a letter is only a few frames wide here (the survey reports stride 2 as best in speech alignment);
   (iii) distil aligner-k2's inner cuts where both placers agree into boundary targets, which is a stronger anchor
   than ink gaps because it covers connected letters.
3. Twenty lines of hand-marked letters on Humboldt would replace the classifier and settle differences of one point.
4. If pen lifts are taken up again: only with an independent check of the anchor (a second model, or the owner's
   marks), on the upright line, and with the word gap kept apart from the lift in the graph (a cost for a lift that
   is wider than a letter).

## 10. Spend

`modal billing report --for today`, 23:16, apps I started: 7.7 USD of my 10.

| | USD |
|---|---:|
| `littera-humboldt-ft`: four fine-tunes (2.4 A10G hours), two anchor passes over the pool (the first was discarded when the rescale was found), throughput jobs | 6.0 |
| `littera-eval`: the two decode grids and the report part on synthetic lines and GW20 | 0.5 |
| `littera-humboldt`: my calls of agent M's `place`, `score`, `classify` (those in my logs 1.16, a few early ones not logged) | 1.2 |

At most three GPU containers ran at once. CPU containers: I was above the 10 of CLAUDE.md for parts of the evening (the
anchor pass with 8 to 10 containers beside placement and classifier jobs, up to about 18).

## Files

Changed: `src/littera/graph.py`, `src/littera/place.py`, `src/littera/geometry.py`, `src/littera/train.py`,
`modal/eval.py`. New: `src/littera/anchors.py`, `modal/humboldt_ft.py`, `tests/test_lifts.py`, `docs/r3/APPROACH.md`,
`docs/r3/anchors/` (four sheets), `results/r3_approach/`. On the volume: `/vol/models/littera-humboldt-ft3`,
`/vol/runs/r3-control`, `r3-lifts`, `r3-lifts-anchors`, `r3-local-lifts-anchors`, `/vol/pool/humboldt_ft_anchors`,
placements `t-*` and `f-*` under `/vol/eval_runs/humboldt-1000`.
