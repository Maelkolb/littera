# Round 2 (2026-11-09 night): pilot, marking page, Laubmann lines

Round 1 is committed (c88a9d8 .. 55adc89): core package (`docs/r1/CORE.md`), line pool v1 (`docs/r1/DATA.md`,
`docs/DATA_CARD.md`), evaluation sets and baselines (`docs/r1/EVAL.md`, `docs/r1/BASELINES.md`). Spend so far
0.85 USD of 25. Read `PLAN.md`, `CLAUDE.md` and the round 1 reports that concern you first.

Two agents work in parallel, each in its own files. Round budget: pilot 3.50 USD, marking page and Laubmann 0.40 USD.
Together never more than 10 Modal containers at once: pilot at most 6, the other agent at most 4.

## Decisions taken after round 1 (orchestrator)

- Pool texts stay as published. Markup without ink (`<u>`, `^`, `~~`, `[...]`, `¬`, ...) is handled by `littera.text`
  presets, configured per source in `configs/sources.yaml` (`markup:` key). Nuremberg's `*` is an abbreviation sign
  with ink and stays a character.
- Silver sources keep the same cap; after the pilot, rows whose line fit is far below their source's median are
  dropped from the main run.
- mkn stays in (public domain mark and CC BY-SA are both open), koenigsfelden stays out.
- Laubmann becomes a held-out evaluation source (internal, `open_licence = false`): lines cut from region crops with
  checked line boxes.
- Ranking without letter gold: GW20 word edges and `synth-heldfonts` letter cuts first, mismatch AUC second. The
  synthetic set comes from the same generator as the boundary anchor (other fonts), so it flatters runs that use it;
  the owner's letter gold decides the main run.

## Agent D: markup presets and pilot

Files: `configs/sources.yaml` (only the `markup:` keys), `src/littera/text.py` (new presets only, with tests),
`configs/pilot/*.yaml` or equivalent, `modal/train.py` and `src/littera/train.py` (fixes only), report `docs/r2/PILOT.md`.

1. Markup inventory: per source, which characters and tags in the pool texts have no ink (sample texts and crops,
   by eye on contact sheets). Presets in `littera.text`, the source's preset in `configs/sources.yaml`, applied in
   training through the source column.
2. GPU smoke on an L4 (at most 10 minutes): steps per second, memory, data loader speed for each pilot config; check
   that loss falls and that a checkpoint loads through `Aligner.load` and runs `littera.eval.run` end to end.
3. Pilot grid, all on the same 50k training lines (stratified over the 31 sources) plus the synthetic rows
   `/vol/synth/v1` at a fixed share, same steps and seed:
   - `vgsl` + `kraken` context, `ctc` (the k2 design on general data)
   - `resnet` + `bilstm`, `ctc`
   - `resnet` + `bilstm`, `ctc_prior`
   - `resnet` + `bilstm`, `segmental`
   - `resnet` + `local`, `segmental`

   Size the steps from the smoke so the five runs fit the budget with the evaluation. Run them in parallel.
4. Evaluate every run with `littera.eval.run` on `gw20-words`, `iam-words`, `synth-heldfonts`, `mismatch-heldout`
   and the in-domain `test` split (read rate of the model's own best reading as a sanity number), beside the round 1
   baselines. Contact sheets of 30 heldout lines (Humboldt, Swiss Federal Council, Bullinger, ABP) for the best two runs
   and k2, judged blind by eye.
5. `docs/r2/PILOT.md`: table, ranking, the recommended main-run config (objective, context, synthetic share,
   ink-prior and frame-loss weights, steps), estimated main-run cost, and which knobs remain open for the letter gold.
   Copy the best pilot run's `best/` checkpoint to `/vol/models/pilot-best/`.

## Agent E: Laubmann lines and the letter marking page

Files: `modal/laubmann.py` (or a function in a new module under `src/littera/data/laubmann.py`), `tools/mark/`
(the page and its builder), report `docs/r2/MARK.md`.

1. Laubmann held-out lines: the Kurrent pool has Laubmann only as region crops (`kurrent-raw`, staged laubmann
   sources; about 4,000 regions have checked line boxes in `extra.line_boxes`). Cut line crops from regions whose
   line count matches the target's lines, pages decide, at most 2,000 lines, written into the pool as
   `/pool/v1/lines/laubmann/` with `split = heldout`, `open_licence = false`, and appended to `index.parquet`
   (keep the index consistent; back up the old one first). Add 12 Laubmann lines to `letter-gold-candidates`.
2. The marking page for about 120 letter-gold lines. Load the skills `edition-ui` and `frontend-design` first, write a
   short design note, and close with a `web-design-guidelines` audit and screenshots (headless Edge).
   - Offline only (internal and unpublished sources): a local folder `Littera_Letter_Gold/` with `index.html`,
     the crops as image files and the line data as a `.js` file, so it opens from file:// without a server.
   - Each line: the crop large, the text under it with each letter under its span, the cuts between letters of a word
     pre-filled from a placer (the best available; McCATMuS from round 1 now, the pilot model later through a rebuild),
     drawn as slanted lines through the line's mid x-height (slant and centre line from `littera.geometry`), dragged
     horizontally with mouse or touch, nudged with the keyboard. Per line: "text does not match", "cannot tell",
     done. Word ends count too.
   - Autosave in localStorage, export one JSON file in the eval-set contract (`gold_cuts` as `i` and crop `x` at mid
     x-height, plus the slant and the line's status) and an importer that turns it into
     `/eval/letter-gold/lines.parquet`.
   - Few, very polished features; the owner marks about 100 lines in about two hours, so speed of marking matters
     most (keyboard flow from cut to cut and line to line).
   - The builder (`uv run python -m tools.mark.build --placer <name> --out <dir>`) re-runs with another placer.
