# Round 3 brief: is littera better than aligner-k2 on Humboldt, and what would be

Orchestrator's brief, 2026-11-17 evening. The owner gives a talk tomorrow morning (slide 31 shows littera's
architecture), so finish within about three hours of wall time. A complete, measured result on few variants beats many
unfinished ones. Report numbers as measured, including results that go against the hypothesis below.

## The owner's questions

1. Is littera (`littera-humboldt-ft2`) actually better than the earlier CTC alignment (kraken `aligner-k2`)?
2. Is the architecture (7.0 M parameters: ResNet, row attention, 2 BiLSTM, character and boundary head) the right one,
   or are others better?
3. Does it build on something like the kraken feature map used before?
4. How can it be improved? The owner asks for a clearly better approach, shown by measurement.

## What is known (read before starting)

- `docs/HUMBOLDT.md`, `docs/MODELS.md`, `docs/r2/PILOT.md` in this repo.
- `C:/Users/totom/Projects/humboldt-htr/docs/lookup/ALIGNER.md` (read only): the aligner-k2 evaluation. Its key
  finding: inside words the CTC spikes were not shown to beat even spacing (blind check 62 against 63 of 77), and cuts
  between letters lay on a local minimum of the ink profile no more often than even cuts (21.5 against 20.4 %).
- With equal snapping littera-humboldt-ft2 and aligner-k2 differ by 0.6 points on word boundaries. Letter placement
  inside words has never been measured on Humboldt for littera. There is no hand-marked letter gold (the owner marked
  one line).
- `littera-humboldt-ft2` was fine-tuned on Humboldt lines only. The boundary loss (`bce` in `train.py`) acts only on
  rows with exact cuts, which are the synthetic rows, and the fine-tuning pool had none. So for 5,500 steps nothing but
  the transcript likelihood shaped where letters start. Round 1 found that without the ink prior and the loss on exact
  cuts, letters drift by about two letters. Whether ft2 drifted inside words is open.
- In the segmental graph (`graph.py`), letters of a word adjoin directly: `blank_after` is false, and only junctions
  between words have a gap state. An ink-free column run inside a word (a pen lift) has to be owned by a letter, at
  the ink prior's cost of 2.0 per frame, and nothing draws a letter boundary to it.

## Hypothesis of this round: ink-anchored alignment

Where the writer lifts the pen inside a word, the ink itself marks a letter boundary exactly. Use that three times:

1. **Pen-lift arcs.** An optional gap state between two letters of a word (new `GraphOptions` field, off by default so
   released models decode as before). A pen lift is then explained by the gap label, and the boundary falls into it.
2. **Edges on the ink.** Letter edges next to a lift are set to the ink's edges of that gap, below frame resolution,
   as word edges are today (`_snap`).
3. **Anchors as supervision in the hand.** Pieces between gaps that hold exactly one letter give that letter's exact
   extent; word starts and ends at snapped gaps give exact left and right edges. These are real-ink rows with partial
   frame labels and boundary targets for the existing `bce` and frame losses (with a mask). A letter's left-edge
   convention is then fixed by its anchored instances and carried into connected runs by the shared network.

This is a hypothesis. If the measurements say pen lifts are rare in this hand, or anchors do not help connected
letters, say so and stop spending on it.

## Shared data on the Modal volume `littera` (mounted `/vol`)

- Models: `/vol/models/littera-humboldt-ft2`, `/vol/models/littera-pilot-seg` (also local in `models/`).
- Volume `humboldt-htr` (read only): `/images/<stem>.jpg` pages, `/lookup/lines/<stem>.json` matched lines with
  `pieces` polygons, `t`, `f`; `/lines/r6g6` line crops of the 28,419 r6 lines; `/runs/aligner-k2/model/best.safetensors`;
  TrOCR runs used by `modal_tokens.py::eval_read_job`.
- Local, read only: `humboldt-htr/runs/aligner-k2/lines-r6k.csv` (uid, page_id, split, `text_noadd`),
  `humboldt-htr/runs/aligner-k2/eval/recipe.json` (the 1,000 evaluation lines, 200 per split, with `pieces`),
  `humboldt-htr/runs/aligner-k2/eval/place-k2.jsonl` (k2 cuts per variant in crop pixels), `humboldt-htr/modal_tokens.py`,
  `humboldt-htr/modal_lookup.py`, `humboldt-htr/src/humboldt_htr/{crops,tokens,lookup}.py`.

Contracts (agent M makes these first, agent T reads them):

- `/vol/eval/humboldt-1000/`: littera eval-set schema (`src/littera/eval/sets.py`), the recipe's 1,000 lines, crops
  cut as production cuts them (`crop_fragments(page, pieces, grow=6)`), `uid` and `split` kept, text as in the recipe
  with characters flagged `a` (added above the line) handled as the lookup build does (skipped).
- `/vol/pool/humboldt_ft/`: pool schema (`src/littera/data/schema.py`), `source = "humboldt_ft"`, splits `train` and
  `val` from `lines-r6k.csv` (`text_noadd`), without any page that holds a line of the recipe. The office build had
  19,713 train and 909 validation lines from 584 and 31 pages; report your counts.
- Dev and report halves of the 1,000 lines: **dev = `val` + `test_page`**, **report = `test_bench` + `test_journal` +
  `letters`**. Every choice (penalties, thresholds, checkpoints) is made on dev. The report half is scored once per
  final variant.
- Placements: `/vol/eval_runs/humboldt-1000/<placer>.jsonl`, one row per line:
  `{"uid", "x": [n+1 edges in crop px, upright crop], "visible": [n bools], "q": [n], "line_fit", "slant"}` with `n` the
  number of characters of the text. Agent M's app writes these for any model folder plus decode options, and scores any
  set of them. Agent T uses M's entry points and does not write its own scorer.

## Agents

All: follow `CLAUDE.md`. No commits, no pushes. Do not touch `humboldt-hand-lookup` or tracked files of `humboldt-htr`.
Local machine: code, unit tests and a handful of lines only; no local bulk inference or training. Modal budget of
the whole round is 20 USD; each agent has a share and reports its spend from `modal billing report`.
Use `--detach` for jobs over ten minutes and poll with `modal app logs` or the volume.

### S: survey (no compute, web)

Report `docs/r3/SURVEY.md`, two pages at most, every claim with a checked source (open the page, do not cite from
memory):

- Methods that place a known transcription on a handwriting or print line at character level: CTC forced alignment and
  its known localisation faults, label priors, segmental and semi-Markov models, attention-based readers, weakly
  supervised character detection (CRAFT and successors), generative sprite models (Learnable Typewriter), anything
  newer than these up to 2026.
- What is known about architecture and localisation: recurrent whole-line context against limited context,
  transformers, frame stride, 2D against collapsed 1D features. Evidence, not opinion.
- Public datasets with letter-level ground truth on handwriting that could serve as a test (licence, size, how to get
  them): online handwriting with character labels, palaeographic letter annotations, others.
- A verdict in ten lines: which of these would plausibly beat littera's present design for letter placement in one
  writer's cursive hand, and what it would cost to try.

### M: measurement on Humboldt (4 USD) — files `modal/humboldt.py`, `src/littera/eval/humboldt*.py`, `docs/r3/MEASURE.md`, `results/r3/`

1. Build the two data contracts above. Announce the pool in `docs/r3/DATA.md` as soon as it exists (T waits for it).
2. `modal/humboldt.py::place --model <vol path> --name <placer> --options '<json of decode and snap overrides>'` and
   `::score --names a,b,c`. Import k2's cuts from `place-k2.jsonl` (`raw-mid` and `snap-mid`) and an even-spacing floor
   (whole line, and inside the snapped words of each placer).
3. Measures, each per split and for dev and report halves, with bootstrap intervals over lines:
   - **Word boundaries on an ink-free column**, on a strip that belongs to neither placer's snapping (state exactly
     which strip and threshold), raw and as deployed.
   - **Pen lifts:** how many letter boundaries inside words have an ink-free column run at all (the hand's lift rate),
     and for each placer the share of such runs that hold a letter edge. Report this as circular for any placer that
     snaps to lifts.
   - **Letter-crop classifier (independent of all placers):** cut every letter exactly at its span, scale to a fixed
     size, train a small CNN on the letters of half the lines and test on the other half, the same recipe per placer.
     Better cuts give more consistent letters. Report accuracy per placer, also separately for letters inside
     connected runs (no lift on either side), which is the case the hypothesis has to prove.
   - **Occlusion evidence from a reader that is none of the aligners (TrOCR t11 of humboldt-htr):** slide a
     paper-coloured bar along the line, record how far each token's teacher-forced log-probability falls, and compare
     each token's evidence centre with the placed span of its characters (share inside the span, mean offset in
     letter widths). Do this if it fits the budget after the three measures above.
   - **Contact sheets for a blind check:** letter sequences of two and three letters where two named placers disagree
     by more than half a letter, the same tiles under each placer, shuffled, with a key file kept apart. The
     orchestrator judges them.
4. First table: even, k2 raw-mid, k2 snap-mid, littera-pilot-seg, littera-humboldt-ft2 (`off`, `upright`).

### T: the approach (10 USD) — files `src/littera/graph.py`, `place.py`, `train.py`, `src/littera/anchors.py`, `modal/humboldt_ft.py`, `tests/`, `docs/r3/APPROACH.md`

Changes to shared files must be additive and off by default: released models must place exactly as before (check
with the existing tests and one line placed before and after).

1. Pen-lift arcs and edges on the ink, at decoding time. Test on `littera-pilot-seg` with the existing synthetic
   held-out fonts and GW20 (`modal/eval.py`), where exact gold exists, and on Humboldt dev with M's entry points.
2. Anchors from the fine-tuning pool, and the masked losses for them.
3. Fine-tunes continued from `littera-humboldt-ft2`, about 1,500 steps each on one A10G, at most four runs:
   a control with the old objective (to separate "more steps" from the change), pen-lift topology alone, pen-lift
   topology with anchors, and one of your choice (for example with 20 % synthetic rows from `/vol/synth/v1`).
4. Score every run with M's measures on dev, pick one, score it on the report half, and put the chosen model at
   `/vol/models/littera-humboldt-ft3`. Report throughput (lines per second on the GPU used).

## Afterwards (orchestrator)

Blind check, re-placement of the lookup index with the chosen model, a local copy of the lookup site for the owner's
review. Nothing is pushed to `humboldt-hand-lookup` before the owner has seen it.
