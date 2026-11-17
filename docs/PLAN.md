# littera: plan

Given a line image and its transcription, littera places every character and word on the line: a span, a slanted box,
a confidence, and a fit score for the line and each word. It generalises the Humboldt aligner (`aligner-k2` in
Maelkolb/humboldt-htr, `src/humboldt_htr/tokens.py`, measured in `docs/lookup/ALIGNER.md`) to any hand, script
and transcript.

## Owner decisions (2026-11-09)

- Private repo `Maelkolb/littera` (public since 2026-11-17, version 0.1).
- Training data: openly licensed only (MIT, CC BY, CC BY-SA, CC0, public domain). Non-commercial, no-derivatives,
  internal or registration-only material is for evaluation only.
- The owner marks about 100 lines of letter boundaries by hand, later. The main training run waits for that set.
- Budget: 25 USD for Modal and APIs over the whole project.
- Scope v1: Latin-script hands and print (Kurrent, Fraktur, medieval book hands, Latin cursive). Assumed, not yet confirmed.

## Why a new model

What `aligner-k2` showed (Humboldt, ALIGNER.md):

- A better reader is not a better placer: k1 to k2 cut CER from 20.1 to 15.3 % and moved placement by about one point.
- The cut rule gave 8 to 9 points, snapping word ends into the ink gap 95 to 99 % right word boundaries.
- Inside words the CTC spikes did not beat even spacing (62 against 63 of 77 disputed sequences).
- Per-character confidence `q` predicts a right word with AUC about 0.6; the line fit separates right from wrong
  text with AUC about 1.0.

CTC is only trained to emit the right sequence. A spike may sit anywhere on its letter and blank takes the rest, so
nothing trains letter boundaries. littera changes the objective, the decoding and the geometry, and measures letters
against hand-marked boundaries.

## Architecture

```
line crop (+ polygon / baseline when known)
  geometry: straighten along the baseline, estimate and remove slant, height 96 px, grey, ink high, 16 px padding
grey strip 96 x W
  residual CNN, 4 stages of 2 blocks
     64 ch  down 2x2   48 x W/2
    128 ch  down 2x2   24 x W/4
    192 ch  down 2x1   12 x W/4
    256 ch  down 2x1    6 x W/4
  attention pooling over the 6 rows -> 256-d per frame (4 px of the strip)
2 x BiLSTM 256 -> 512-d per frame
  character head: alphabet + gap + <any>   (softmax)
  boundary head:  a letter starts here     (sigmoid)
```

About 8M parameters. One alignment graph serves as training loss and as decoder:

- `ctc`: standard CTC topology (control).
- `ctc_prior`: CTC with label priors subtracted during training (Huang et al., ICASSP 2024).
- `segmental`: every character is a run of at least one frame, no blank inside a word, a gap state only at spaces and
  line ends; the boundary head scores the transitions. Synthetic lines with exact boundaries add a direct loss on the
  boundary head.

Decoding runs forward-backward on the transcript's graph: every boundary is an expected position with a spread, the
spread is the character's confidence. Arcs for real transcripts: optional spaces (touching words), skippable characters
at a penalty (expansions, normalised spellings), a garbage state at the line ends and between words (line numbers,
neighbouring ink, struck text not transcribed). Cuts are mapped back through the slant as slanted lines; word ends snap
into ink gaps on the strip.

Alphabet: NFD, combining marks get zero width on their base, MUFI and variant letters are classes of their own, `<any>`
(trained by masking rare characters) aligns unseen symbols. Markup presets mark characters that have no ink.

## Phases and budget (USD)

| phase | content | budget |
|---|---|---|
| 0 | package: text, geometry, graph, model, placement, PAGE-XML, CLI, tests | 0 |
| 1 | line pool from open sources with held-out sources, data card | 2.5 |
| 2 | evaluation sets, synthetic exact-boundary lines, baselines, metrics | 1.5 |
| 3 | pilot on 50k lines: backbone x objective x context, ranked on word gold + synthetic | 3 |
| - | owner marks ~100 letter-gold lines | 0 |
| 4 | main run of the winner on the full pool | 6 |
| 5 | evaluation and report | 2 |
| 6 | Humboldt fine-tune and comparison with k2, ONNX, CLI release | 1.5 |
| | reserve | 8.5 |

Exit after phase 3: if `ctc_prior` and `segmental` do not beat `ctc` on letters, ship the general CTC model with
posterior decoding, slant and snapping, and keep the money.

Round documents: `docs/r1/` (phases 0 to 2), `docs/r2/` (phase 3). Scores: `results/`.

## Status (2026-11-17)

- Phases 0 to 3 done; Modal spend about 4.9 USD.
- Pool v1: 299,992 training lines from 31 open sources, 16,319 held-out lines from 9 sources (Laubmann and ABP
  evaluation only).
- Pilot (`docs/r2/PILOT.md`): the segmental objective places synthetic letters within a quarter letter 86 % of the time
  against 74 to 76 % for CTC (pilot and kraken baselines), with calibrated confidences; word edges on George Washington
  are within 0.06 letters of the kraken baselines; `ctc_prior` collapsed. Best pilot: `pilot-seg`, released as
  `littera-pilot-seg`.
- Humboldt fine-tune (part of phase 6, done before the main run, on our own GPU): `littera-humboldt-ft2`, compared with
  aligner-k2 in `docs/HUMBOLDT.md`. With the same word-edge snapping it places word boundaries as well as k2 or better on
  every split; the line fit separates right from wrong text as well as k2's.
- New in placement: `snap="upright"` snaps word edges on the upright line, for upright word crops.
- Waiting for the owner: about 100 lines in the marking page (a local folder `Littera_Letter_Gold`, not published,
  prefilled by `pilot-seg`), then `uv run python -m tools.mark.ingest <export.json>`.
- Next: score the pilots on the letter gold, decide segmental or CTC, main run (6.5 to 8 USD), final evaluation, ONNX.
