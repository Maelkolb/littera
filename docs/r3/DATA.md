# Round 3 data on the Modal volume `littera` (agent M, 2026-11-17)

Both sets are on the volume. Built by `modal/humboldt.py::build` (app `littera-humboldt`, 2 minutes on 10 CPU containers)
from the volume `humboldt-htr` (pages `/images`, matched lines `/lookup/lines`) and the two local files
`humboldt-htr/runs/aligner-k2/lines-r6k.csv` and `eval/recipe.json`.

## `/vol/pool/humboldt_ft` (fine-tuning pool)

Pool schema, `source = "humboldt_ft"`, files `lines/humboldt_ft/part-00000.parquet` to `part-00019.parquet`,
`index.parquet`, `summary.json`. Use it as `"pool": ["/vol/pool/humboldt_ft"]`, `"presets": {"humboldt_ft": "markup"}`.

| split | lines | pages | journals | characters |
|---|---:|---:|---:|---:|
| train | 19,655 | 585 | 10 | 773,807 |
| val | 992 | 31 | 6 | 35,895 |

The office build had 19,713 and 909 lines from 584 and 31 pages.

How the rows were chosen:

- Rows of `lines-r6k.csv` whose `source_split` is `train` or `val` (20,740 rows on 617 pages after the next step).
  The csv's `split == "train"` also holds 2,622 lines of the training letters (`source_split == "letters_train"`,
  169 pages). They are left out: the office build's 615 pages equal the journal pages alone, so ft2 never saw them, and
  a control run "more steps, same objective" needs the same data. They could be added as a further source later.
- No page that holds a line of the recipe: 175 pages, 1,031 csv rows dropped. This removes all but 2 pages (5 lines)
  of humboldt-htr's `val` split, because the recipe's 200 `val` lines lie on 34 of its 36 pages.
- So `val` here is not humboldt-htr's `val`: it is 31 pages drawn from the kept pages with `random.Random(3).sample`
  over the sorted page ids, the rest is `train`. The split is by page.
- Dropped while cutting: 70 rows without a matched line in `/lookup/lines`, 15 rows on two pages without a match file,
  8 rows with empty `text_noadd`.
- Grades: 17,381 A, 3,266 L (`quality`). Median crop 1,219 x 149 px.

What a row holds:

- `image`: the production crop, `crop_fragments(page, pieces, grow=6)` with the pieces of `/lookup/lines` (polygons
  widened by 6 px on white, upright box between two pieces), JPEG quality 95. The cutting code is a copy of
  humboldt-htr's `crops.py` (`src/littera/eval/humboldt_data.py`); on four recipe lines the copy and the original give
  identical pixels and the same page box.
- `text`: `text_noadd` of the csv with whitespace runs collapsed (struck words kept, text above the line removed).
- `baseline`: all baseline points of the line's pieces, sorted by x, in crop pixels (page point minus
  `fragments_box(page.size, pieces, pad=8)[:2]`). `polygon` is empty. This is how the live lookup index was placed
  (`aligner.place(crop, text, baseline=points)`, orchestrator's check on H0017682_3r: identical cuts only with the
  baseline), so `prepare()` straightens a training line along the same guide as placement does. **The first build of
  this pool (about 20 minutes earlier) had an empty baseline column; it was rebuilt, same rows and counts. A run that
  read the pool before the rebuild trained without baselines and should be restarted.**
- `origin`: JSON with `stem`, `box` (the crop's box on the page, so page x = crop x + box[0]), `pieces` (count),
  `baselines` (the detector's baseline of every piece, in crop pixels, left to right), `kind`, `csv_split`.
- `split`: `train` or `val`. `doc` is the journal, `page` the page id, `hand` is `humboldt`, licence CC BY-SA 4.0.

## `/vol/eval/humboldt-1000` (evaluation lines)

littera eval-set schema, `lines.parquet` and `README.md`. 1,000 lines on 175 pages, 200 per split, 40,633 characters.

- `image`: the same production crop, stored as PNG (lossless: the pixels placement saw in production).
- `text`: the recipe's `t` as production places it (`modal/humboldt_index.py::line_text_on_crop`, copied as
  `humboldt_data.text_on_crop`): characters flagged `a` (written above the line) removed and the double space this
  leaves collapsed. 20 lines differ from the recipe. `true_text` is the recipe's `t`, `variant` its flags `f`.
- `baseline` as in the pool, `polygon` empty. `uid` and `split` as in the recipe. `origin` as above plus `other`
  (another line's text of the same page, for the line-fit check).
- Dev half: `val`, `test_page` (400 lines). Report half: `test_bench`, `test_journal`, `letters` (600 lines).

## Placements

`/vol/eval_runs/humboldt-1000/<placer>.jsonl`, one row per line:
`{"uid", "x": [n+1 edges], "visible": [n], "q": [n], "line_fit", "slant"}`, `n = len(text)` of the eval set's `text`,
edges in pixels of the upright crop; character `i` spans `x[i]` to `x[i+1]`, a space spans the gap between two words.
On the volume now: `even`, `k2-raw-mid`, `k2-snap-mid` (aligner-k2's October cuts, imported), `pilot-upright`, `ft2-off`,
`ft2-upright`. Per-letter classifier results are in `classify/`, the occlusion evidence in `occlusion/evidence.json`.
Entry points: see `docs/r3/MEASURE.md` and the head of `modal/humboldt.py`.

## Checked by eye

Three pool rows of `part-00000` (rows 0, 137, 311; `modal/humboldt.py::peek --pool`) and two evaluation rows, each as
the crop with the stored baseline drawn on it:

- `H0016412_24r_z003_l006` (4 pieces, "bestim̅t um 1° zu weit gegen Osten gezeichnet"): crop and text agree, the
  baseline runs under the letters of all four pieces, the upright boxes between the pieces show paper and a little of
  the line above.
- `H0016412_63v_z005_l002` (1 piece, "gewirkt. Alle Bergketten, die hohe südlich von"): crop and text agree, the
  baseline fits.
- `H1242132_7v_z001_l000` (3 pieces, "le Schiffe gescheitert. Die Inseln Píritu sind"): the crop holds one piece more on
  the left (a marginal note, about a fifth of the width) than the text. The row is grade A. humboldt-htr's audit puts
  such crop-extent faults at about a quarter of r6; production crops have them too, so they are left in.
- Evaluation rows `H0019155_12v_z004_l005` and `H1242132_17r_z003_l026`: crop, text and baseline agree; in the first
  the leading word "stets" is struck through (flags `s`), and aligner-k2 and littera-humboldt-ft2 disagree on where
  it ends.

So one of the three pool rows looked at has a crop that holds more ink than its text. Three rows are no estimate of
a rate, but a fine-tune on this pool has to tolerate ink without text at the line ends (the decode options'
`garbage_ends`), and an exact word-edge anchor taken from such a row is wrong at that end. The recipe's evaluation
lines come from the same matching, so the same kind of fault can be in the evaluation set.
