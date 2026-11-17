# Round 1, agent B: line pool and data card

## Built

- `configs/sources.yaml`: 62 candidate sources with the licence as recorded, `open_licence` (true, false or `per_row` with
  `row_licences`), role (`train` 31, `heldout` 8, `out` 23) and a one-line reason.
- `modal/pool.py` (app `littera-data`, CPU, at most 10 containers): `scan` (staged metadata of every line row, 50 tasks), `plan`
  (rules, splits, draw from metadata only), `sample`, `write` (reads the images of the drawn rows only, one task per part),
  `finish` (index and summary). One command rebuilds everything: see `docs/DATA_CARD.md`.
- Volume `littera:/pool/v1/`: `lines/<source>/part-*.parquet` (71 parts, 10.6 GB), `index.parquet` (plus `part`, `part_row` for
  random access), `summary.json`, `build/` (plan, plan report, scans), `sample/` (437 rows with contact sheets), `checks/` (audit
  sheets).
- `src/littera/data/` (`schema.py`, `pool.py`): `POOL_SCHEMA`, `INDEX_SCHEMA`, `read_index`, `iter_rows`, `read_rows`,
  `load_image`, `points`, `list_sources`, `pool_root` (explicit path, `$LITTERA_POOL`, `/vol/pool/v1`, else `data/pool_sample`).
  pyarrow and PIL only. Checked on the local sample.
- `data/pool_sample/`: 437 rows (train 186, val 62, test 93, heldout 96) from all 39 sources, pool layout, 15 MB.
- `docs/DATA_CARD.md`.

## Pool

| split | rows | pages | hands (proxy) | characters |
|---|---:|---:|---:|---:|
| train | 299,992 | 49,169 | 8,472 | 12.0M |
| val | 5,595 | 546 | 435 | 0.23M |
| test | 8,611 | 1,787 | 1,051 | 0.33M |
| heldout | 14,319 | 2,434 | 2,201 | 0.59M |

31 training sources, train cap 14,385 lines per source (water-filling to 300,000; 13 sources reach it, the other 18 give every
line left after test and val). Print in train 10,042 (cap 40,000 not binding). Baselines in crop pixels on 70 % of the rows
(24 sources). Crop height median 124 px, p5 48, ceiling 160. 467 distinct characters in train. Held-out: humboldt, bullinger,
swiss_council, abp, konzilsprotokolle 2,000 each; austrian_newspapers 1,976, klosterneuburg_cod48 1,955, parzival 388 (all it
has). Checks on the written index: uids unique, no page in two splits, no non-open row outside `heldout`, no held-out source
outside `heldout`.

## Audit of text against crop

Asked by the orchestrator after agent A found a konzilsprotokolle line whose text did not match its crop. 20 random pool rows of
every source (780 rows) looked at by eye, plus the 437 sample rows and 24 more konzilsprotokolle rows: no source pairs texts with
the wrong line crop (no index shift as in the Kurrent forst rows). The konzilsprotokolle line is the same line, but its page carries
uncorrected recognition output (`General in Aber . M Amor Catechisiren` for `General en Chef, Baron Cederström`). Such pages
tokenise punctuation (`Ursache , weshalb`): 7 of the source's 330 pages have it on 35 to 63 % of their lines, every other page on
0 %. Those 7 pages (184 candidate lines; 2 of the pages were in the first heldout draw) are now dropped (`uncorrected_htr_page`),
and the pool was rebuilt. Other findings, left as they are: one garbled Bullinger line (`ofisit e i i ui t si oi it ei,`, Bullinger is silver);
crops that show two or three lines with the text of the middle one (zh_rrp, colmar_adhr, grandtour, humboldt). Spaced punctuation
elsewhere is the sources' own convention (nuremberg_letterbooks, tridis, icfhr2018 Janauschek, emanuscripta registers, hanse_xv),
checked on their sheets.

## Decisions and why

- Rows: the Kurrent v1.5 build decisions (`build_corpus_v1_5_r5/decisions.parquet`, 1.36M line/transcribe rows of the wanted
  sources) joined with the staged shards they point to. The build's rule outcome is honoured except reasons that do not concern a
  line crop (`not_selected`, `line_region_twin`, `page_shared_with_test`, `geometry_only_target`, `abbreviation_star`).
- Text: the staged transcription as published (NFC, spaces folded), not the build's canonical target. The canonical target expands
  abbreviations on half the rows that carry an expansion and normalises letters, which moves the text away from the ink. Markup
  stays (`~~`, `<u>`, `^`, `[...]`, `¬`, Nuremberg's `*` for an abbreviation sign); the data card lists it. Semper: the edition's
  line-end bar removed and 142 rows with edition markers (`TRENNUNG`, `QUOTE`, ...) dropped.
- Crops: the staged crops as stored (no re-cut). The Kurrent v1.4 re-cut overlay (levelled, tighter crops for 750k rows) is not
  used, so many crops show parts of the neighbouring lines; the staged page baselines, mapped into the crop, mark the right line
  on 70 % of the rows (checked on the sheets and by position: they span 94 to 100 % of the width, at 55 to 70 % of the height).
- Extra row rules: at least 2 visible characters, width per character 0.08 to 2.5 crop heights, crop at least 16 px high and 3 px
  wide per character (8 rows of 9 to 15 px height had slipped through), single line.
- Splits by page. `test` takes the pages the Kurrent project already holds out (published test sets, unseen-hand pages) first, so
  littera's in-domain test agrees with the sources' own splits; at most 300 lines and at most 10 % of a source's candidates (small
  sources such as colmar_adhr would otherwise lose 40 %). Every reserved test page is excluded from train and val. `val` 1 % of
  the remaining pages, at most 500 lines per source.
- Spread: one line from every page before a second line from any page, hands in turn. bern_xix's 14,385 train lines come from
  14,385 pages and 260 hand proxies; zh_rrp's from 2,648 pages, semper's from 1,656, nuremberg_letterbooks' from 1,667 pages and
  16 writers.
- Hand: writer where recorded (Bullinger, Nuremberg, Inzigkofen), otherwise the Kurrent selection's proxy (document or
  collection); counts are proxies.
- Humboldt: rows from `staged/humboldt_tight` (the folder the v1.5 build used; `humboldt` and `humboldt_md` are other cuts of the
  same 28,187 lines). Held-out pages: the Kurrent test and `humboldt_llm_test50` pages first.
- Licences: zenodo19c per row (106 CC BY-NC-SA rows out), semper per row (both open). mkn: the brief lists it as a recorded
  conflict, but the Kurrent records state Public Domain Mark 1.0 on every row and nowhere anything different; it is in.
  koenigsfelden is open but out: its text is a graphemic edition with editorial punctuation (the v1.5 build drops all its rows).
- Parts are sized from the mean staged crop size per source (aim 200 MB); sources whose rows the Kurrent re-cut replaced estimate
  low, so parts run from 5 to 380 MB (median 170 MB).

## Spend

`littera-data` on 2026-11-09: 0.43 USD billed when this was written, about 0.46 once the last run is billed in full, of 2.50
(survey 0.01; scan and plan 0.08; plan and sample 0.04; four full rebuilds about 0.08 each as rules were added; audit sheets
0.01).

## Open (owner)

1. Laubmann: the staged pool has no line rows for it (regions and pages only), so it is not a held-out source in v1. Lines could
   be cut from its region crops with line boxes (about 4,000 regions have checked boxes); that needs new crops. Wanted?
2. Text form: the pool keeps each source's own markup and conventions (struck text, underline tags, raised-letter markers,
   Nuremberg `*`, Hanse expansions). The core text presets must handle them; alternatively the pool could strip `<u>` tags and the
   `^` markers before training.
3. Silver text at full weight: zh_rrp (edition text aligned to the images), aaeb, basel_grundbuch, bern_xix, unibe_xx are
   high_silver and get the same cap as gold sources. Down-weight or cap them lower?
4. Re-cut crops: use the Kurrent v1.4 re-cut overlay (levelled, one text band) for training instead of, or next to, the staged
   crops? The owner asked for the staged crops as stored; the baselines cover 70 % of the rows.
5. mkn and koenigsfelden as above: confirm.
