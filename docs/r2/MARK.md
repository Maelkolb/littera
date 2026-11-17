# Round 2, agent E: Laubmann lines and the letter marking page

## Design note (written before the build)

Job: the owner places about 3,500 letter boundaries on about 120 lines in about two hours, so roughly two seconds per
cut. Everything on the page serves that rate; the crop is the document, the chrome stays out of the way.

Material: line crops of manuscripts and print on paper of every tone (beige diary paper, grey newsprint, parchment),
inks from iron-gall brown to Laubmann's violet and blue. The marks must read on all of them and must not look like ink.

Palette (light; dark mode mirrors it on a near-black slate):

| token | hex | role |
|---|---|---|
| board | `#E7E8E4` | ground: grey archival board, cooler than any of the papers so the crop stands out |
| ink | `#1E2227` | text |
| pencil | `#5D636B` | secondary text, idle letters |
| rule | `#C6C9C3` | hairlines, overview frame |
| cut | `#00707A` | prefilled and confirmed cuts: a drafting-pen teal that no source uses |
| active | `#C40D2B` | the active cut and nothing else: the editor's red pencil |
| doubt | `#A87400` | "cannot tell", skipped cuts |

Type: one serif family for the page (Palatino Linotype, Book Antiqua, Georgia), tabular lining numerals; the letters
under the crop in Junicode (embedded subset, OFL) because medieval lines carry MUFI letters and combining marks. No
sans, no monospace, no capitals for labels.

Layout (1400 px):

```
littera letter gold                                         41 of 131 done          Export
|||||||||||||||||||| ||||||||||| |||||||||||| ...   one tick per line, gap between sources
humboldt  H0005449_97v  line 4                                      prefill kraken-mccatmus
[ whole line, fit to width, cut hairlines, frame of the magnified view ]
[ magnified crop: slanted cuts through mid x-height, faint centre line                     ]
[ letters, each centred under its span; cut lines run down between them                   ]
  Done  Enter        Text does not match  M        Cannot tell  C
  key line in pencil grey
```

At 390 px the same column, buttons wrap, the key line is hidden, cuts are dragged by touch.

Principles:

- One loud thing: the red active cut. Everything else is a hairline or a tick.
- The letters under their spans are the check: a wrong cut shows as a letter sitting under the wrong ink.
- Keyboard first, the mouse as an equal: drag a cut, or click on the ink to put the active cut there and move on.
- Cuts push their neighbours instead of crossing them, so order is never broken.
- Nothing is lost: every change is saved in the browser at once; an exported JSON dropped onto the page restores it.

Cut model: per word, a start cut, a cut before every inked letter after the first, and an end cut. Letters without ink
(markup by the source's `littera.text` preset) and combining marks get no cut of their own; an exported cut repeats
its `x` for every text index it stands for, so zero-width markup keeps zero width in `gold_cuts`.

## Laubmann held-out lines

`modal/laubmann.py` (app `littera-data`, at most 4 containers) and `src/littera/data/laubmann.py`; rules and checks in
`docs/DATA_CARD.md` (held-out sources). Result: `/pool/v1/lines/laubmann/part-00000.parquet`, **2,000 lines from 234 pages of 34
volumes**, `split = heldout`, `open_licence = false`, licence as recorded, 42 lines with human-corrected text and 1,958 with the
knowledge-graph text layer. `index.parquet` was rebuilt by the pool's own `finish` step after checking that `build/index` equals
the old index (330,517 rows; uids unique, no page in two splits, no closed row outside `heldout`, 2,300 part/row pointers read back);
the old index and summary are in `/pool/v1/build/backup_2026-11-09_pre_laubmann/`.

What it took to get the lines right, in order:

- Volume 35 is the species register (numerals, region boxes that cut the line ends, tag remnants in the corrected texts): left out.
- Ruled lines of the squared paper made every line look like it ran into the region edge; long horizontal and vertical ink runs
  are removed before the line's ink extent is measured.
- Line-count matching is not enough: in about one region in seven a band was missed and another split, so the texts shift by one
  line (seen on the first sheets). Both kraken readers now score each crop's own text against its neighbours' texts; a region is
  dropped if either prefers a neighbour anywhere (64 of 441 regions). Every flagged line looked at was a real shift.
- Print clippings pasted into the diaries carry `script = kurrent` from the staging; not fixed in the pool (open below).

Letter-gold candidates: `modal/eval.py::add_candidates` (new) draws Laubmann with the round 1 rules (`build_candidates` gained
`sources` and `exclude`) and appends to `/eval/letter-gold-candidates/lines.parquet`; the 119 round 1 rows are unchanged
(`lines_round1.parquet` next to it). Two of the first twelve were print clippings labelled `kurrent` (a Fraktur and a typewritten
line); they were excluded by uid (recorded in the README stats) and the next-ranked lines of other pages took their place. Now
**131 candidates** (12 per source, `austrian_newspapers` 11). k2 and McCATMuS were rerun on the whole set (`baselines`, 131 placed,
0 errors), so `/vol/eval_runs/kraken-{k2,mccatmus}/letter-gold-candidates/` cover the Laubmann lines. Local copy refreshed in
`data/eval/letter-gold-candidates/`. Some silver Laubmann candidates have machine-text errors (e.g. "Mehlschwalben" where the ink
reads "Rauchschwalben"); the owner marks those "text does not match".

## The marking page

A local folder `Littera_Letter_Gold/` (never published): `index.html`, `data.js`, `letters.css` (Junicode
subset, OFL, as a data URI), `lines/001.jpg` to `lines/131.jpg` (the crops as stored); 4.9 MB, opens from file://.

Use: open `index.html`; the red cut is the active one. Arrow keys move it (Shift by 5, Alt with the rest of the word), Tab goes to
the next cut, or a click on the ink puts the active cut there and moves on; Enter when the line is right, M if the text does not
match, C if you cannot tell. Marks save in the browser at every change; Export (Ctrl S) saves the JSON file, then
`uv run python -m tools.mark.ingest <file>` writes `data/eval/letter-gold/` and uploads `/eval/letter-gold/`.

- Lines sorted by source; one tick per line at the top (teal done, red text does not match, ochre cannot tell); up and down arrows
  or a click on a tick change lines; `#line=n` in the URL keeps the place.
- Cuts: 5,234 on 131 lines (start, inner and end cuts of every word). Drawn as slanted lines through the line's mid x-height with
  the slant and centre line from `littera.geometry.prepare` (computed locally by the builder); solid over the x-height, faint above
  and below; word starts and ends carry small feet. The bracket keys change the slant per line (exported). X marks one cut "cannot
  place" (left out of `gold_cuts`, listed in `skipped`). Cuts push their neighbours instead of crossing. Ctrl Z undoes.
- Under the crop each letter sits centred between its two cuts; the transcript below underlines the letters at the active cut and
  greys characters without ink.
- Magnified so the x-height is about 42 screen px (at least 1.5x); the view follows the active cut; a thin overview of the whole
  line with the visible frame appears when the line is wider than the screen; Z shows the whole line, plus and minus zoom.
- Prefill: the placer's cuts (start = `x0` of the first letter, end = `x1` of the last, inner = midpoint), clamped to the line's ink
  extent, made monotone. Rebuild with another placer: `uv run python -m tools.mark.build --placer littera:<run> --out <dir>`
  (kraken placements are fetched from `/vol/eval_runs/`, `littera:` and `even` placers run locally). Marks already made stay:
  the browser keeps them per line uid and text, and dropping an exported JSON onto the page restores it (undoable).

Export, schema `littera/letter-marks/1`: `set`, `placer`, `built`, `exported`, and per touched line `uid`, `text`, `status`
(`done`, `mismatch`, `unsure`, `open`), `slant`, `prefill`, `moved`, `visited`, `cuts_total`, `seconds`, `cuts` (`i`, `kind`, `x`,
`skip`), `gold_cuts` (done lines: `{i, x}` in crop px at mid x-height, one per text index), `skipped`. The importer keeps done
lines whose text still matches the candidate, writes the eval-set contract (`kind = letter`, `gold_cuts` sorted by `i`, the marking
record in `origin`), `marks.json` and a README.

Checks:

- Behaviour in headless Edge (synthetic keys and pointer events on a copy of the folder, results read back with `--dump-dom`):
  nudges, Tab and Shift Tab, click-to-place, push order, skip, slant, the three statuses and advancing, undo of moves, statuses and
  restore, line keys, gold cuts per index without skipped cuts, localStorage, export and restore: 26 of 26 pass.
- `tests/test_letter_marks.py` (6 tests): line cutting and its rules, the cut model with markup and combining marks, prefill
  fallback, importer. A synthetic export was imported and scored with `littera.eval.run` (letter cuts read).
- Screenshots (1400 px light and dark, 390 px in an iframe) looked at after each change; final ones in the session scratchpad
  (`agentE/shots/final_*.png`).
- `web-design-guidelines` audit: fixed theme-color, `translate="no"` on names and historical text, tick touch and hover states,
  overview marked `aria-hidden` (a pointer shortcut duplicating Tab, Home and End), Intl date for "last exported", undo for restore,
  no text selection while dragging, balanced headings, a message instead of a blank page without `data.js`, cached ticks. Left
  as is: no skip link (focus starts on the line), the live region announces every cut move (terse, polite).

## Spend

About 0.07 USD of 0.40 (today's billing by app id: `littera-data` survey, three trial cuts, cuts, reader check 0.046, write,
backup, finish, verify 0.065; `littera-eval` candidate draws and baselines about 0.008).

## Open

- Print clippings in the Laubmann lines keep `script = kurrent` (4 of 64 lines read); a script check (by eye per region or a
  print/hand classifier) would fix the column. They are right texts on real lines, so evaluation can use them.
- Laubmann texts are mostly silver (CER 2 to 6 %); fit and mismatch numbers on this source mix model error and text error.
- Laubmann crops are small (region crops at most 1,600 px wide; x-height about 11 px); the page magnifies them about 4x.
- 5,234 cuts at two seconds each is close to three hours; the owner may stop at about 100 lines (done lines are what counts).
- The geometry's x-height is too large on some three-line box crops (Humboldt); the page caps the cut length and the zoom floor
  covers it, but the centre line there should be checked by eye while marking (the bracket keys fix slant, not the centre).
