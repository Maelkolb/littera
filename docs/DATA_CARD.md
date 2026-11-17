# littera line pool v1: data card

Line crops with their transcriptions, for training and evaluating littera's placer. Built 2026-11-09 from the Kurrent VLM
staged pool (Modal volume `kurrent-raw`, read-only) by `modal/pool.py` (app `littera-data`). Licence decisions per source:
`configs/sources.yaml`. Reader: `littera.data` (`src/littera/data/pool.py`).

## Where

Modal volume `littera`, `/pool/v1/`:

- `lines/<source>/part-NNNNN.parquet`: the rows with their crops (sized for about 200 MB from the mean staged crop size; 5 to
  380 MB, median 170 MB; row groups of 512 rows, zstd).
- `index.parquet`: every column except `image`, plus `part` and `part_row` (the row's file and position, for random access).
- `summary.json`: rows, pages, hands and characters per split and source, character inventory per split with counts, crop height
  and width distribution.
- `build/`: `plan.parquet` (the drawn rows with their pointers into `kurrent-raw`), `plan_report.json` (rule outcomes per source,
  per-source width-per-character quantiles, the train cap), `scan/` (staged metadata of the line rows), `index/` (index per part).
- `sample/`: 437 rows across every source and split in the same layout (also local in `data/pool_sample/`, with contact sheets
  on the volume under `sample/sheets/`).
- `checks/`: the audit contact sheets (20 rows per source).

Columns as in the round contract (`docs/r1/BRIEF.md`, schema `littera.data.POOL_SCHEMA`): `uid` (`<source>/<doc>/<page>/<n>`),
`source`, `doc`, `page`, `hand`, `split`, `image` (the stored JPEG), `width`, `height`, `text` (NFC), `script`, `language`,
`licence`, `open_licence`, `quality`, `polygon`, `baseline`, `origin` (`kurrent-raw:/<staged shard>#<staged sample id>`).

## Size

330,517 rows from 40 sources (31 training sources, 9 held-out sources), 72 parts (10.6 GB), 13.2M characters. Laubmann (2,000 held-out
lines) was added in round 2 by `modal/laubmann.py` (below).

| split | rows | pages | hands (proxy) | characters | print rows | rows with baseline |
|---|---:|---:|---:|---:|---:|---:|
| train | 299,992 | 49,169 | 8,472 | 12,008,388 | 10,042 | 212,404 |
| val | 5,595 | 546 | 435 | 227,585 | 105 | 4,593 |
| test | 8,611 | 1,787 | 1,051 | 333,442 | 409 | 5,511 |
| heldout | 16,319 | 2,668 | 2,235 | 667,110 | 2,083 | 6,306 |

Per source (candidates = line rows that passed every rule below; pages = pages in the pool over all splits; hands = proxies, largest
split; baseline = yes when most rows carry one):

| source | role | licence (as recorded) | open | candidates | train | val | test | heldout | pages | hands | baseline |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| aaeb | train | MIT | yes | 50,223 | 14,385 | 356 | 300 |  | 1,725 | 333 | yes |
| basel_grundbuch | train | MIT | yes | 11,663 | 11,252 | 111 | 300 |  | 924 | 14 | yes |
| bern_xix | train | MIT | yes | 508,411 | 14,385 | 500 | 300 |  | 14,591 | 260 | yes |
| catmus | train | CC BY 4.0 | yes | 1,676 | 1,494 | 15 | 167 |  | 1,676 | 1,379 | no |
| colmar_adhr | train | MIT | yes | 738 | 645 | 23 | 70 |  | 32 | 25 | yes |
| emanuscripta | train | CC BY-SA / CC0 text, PDM images | yes | 7,705 | 7,343 | 62 | 300 |  | 635 | 43 | no |
| enhg15 | train | CC BY 4.0 | yes | 2,121 | 1,866 | 44 | 211 |  | 50 | 1 | no |
| faithful_transcriptions | train | CC0 1.0 | yes | 8,592 | 8,203 | 89 | 300 |  | 180 | 12 | no |
| fanny_mendelssohn | train | CC BY 4.0 / CC0 1.0 | yes | 1,985 | 1,774 | 13 | 198 |  | 163 | 1 | yes |
| fondue_wolfflin | train | CC BY 4.0 | yes | 1,761 | 1,572 | 13 | 176 |  | 820 | 65 | yes |
| grandtour | train | CC BY-SA 4.0 | yes | 32,624 | 14,385 | 305 | 300 |  | 1,125 | 6 | yes |
| gt4histocr | train | CC BY 4.0 | yes | 9,763 | 9,368 | 95 | 300 |  | 9,763 | (print) | no |
| hanse_xv | train | MIT | yes | 14,352 | 13,935 | 117 | 300 |  | 429 | 47 | yes |
| hanse_xvi | train | MIT | yes | 26,594 | 14,385 | 261 | 300 |  | 1,145 | 1,106 | yes |
| hanse_xvii | train | MIT | yes | 29,198 | 14,385 | 280 | 300 |  | 1,296 | 1,234 | yes |
| hisclima | train | CC BY 4.0 | yes | 3,728 | 1,460 | 13 | 300 |  | 469 | 359 | no |
| htr_school_2024 | train | CC BY-SA 4.0 | yes | 15,613 | 14,385 | 141 | 300 |  | 350 | 5 | yes |
| icfhr2018 | train | CC BY 4.0 | yes | 12,448 | 11,240 | 161 | 300 |  | 990 | 822 | no |
| inzigkofen | train | CC BY 4.0 | yes | 8,067 | 7,668 | 106 | 293 |  | 279 | 3 | yes |
| lectaurep | train | CC BY 4.0 | yes | 21,025 | 14,385 | 203 | 298 |  | 322 | 8 | yes |
| mkn | train | Public Domain Mark 1.0 | yes | 18,650 | 14,385 | 191 | 300 |  | 770 | 8 | no |
| nuremberg_letterbooks | train | CC BY 4.0 | yes | 49,390 | 14,385 | 500 | 300 |  | 1,695 | 16 | yes |
| popp | train | CC BY 4.0 | yes | 6,871 | 5,440 | 60 | 300 |  | 235 | 74 | no |
| rats_richtebuecher | train | MIT | yes | 5,586 | 5,241 | 45 | 300 |  | 266 | 25 | yes |
| read_icfhr16 | train | CC BY 4.0 | yes | 10,025 | 8,794 | 97 | 300 |  | 450 | 2 | yes |
| scads | train | CC BY 4.0 | yes | 11,464 | 11,065 | 99 | 300 |  | 771 | 741 | no |
| semper | train | CC BY 4.0 or CC BY-SA 4.0 per row | yes | 71,635 | 14,385 | 500 | 299 |  | 1,682 | 10 | yes |
| tridis | train | MIT (dataset); originals CC BY | yes | 19,137 | 14,385 | 158 | 300 |  | 3,083 | 1,778 | no |
| unibe_xx | train | MIT | yes | 42,053 | 14,385 | 500 | 300 |  | 2,642 | 40 | yes |
| zenodo19c | train | CC BY 4.0 per row (NC-SA rows out) | yes | 8,596 | 4,627 | 37 | 300 |  | 257 | 13 | yes |
| zh_rrp | train | MIT (hub); text CC BY-SA 4.0 | yes | 76,283 | 14,385 | 500 | 299 |  | 2,687 | 90 | yes |
| abp | heldout | internal archival material | **no** | 10,890 |  |  |  | 2,000 | 99 | 24 | no |
| austrian_newspapers | heldout | CC BY 4.0 | yes | 55,073 |  |  |  | 1,976 | 6 | (print) | yes |
| bullinger | heldout | CC BY-SA 4.0 | yes | 48,763 |  |  |  | 2,000 | 160 | 160 | no |
| humboldt | heldout | CC BY-SA 4.0 | yes | 23,591 |  |  |  | 2,000 | 61 | 9 | no |
| klosterneuburg_cod48 | heldout | CC BY 4.0 | yes | 2,791 |  |  |  | 1,955 | 28 | 1 | yes |
| konzilsprotokolle | heldout | CC BY 4.0 | yes | 8,202 |  |  |  | 2,000 | 76 | 4 | yes |
| parzival | heldout | MIT | yes | 388 |  |  |  | 388 | 4 | 3 | yes |
| swiss_council | heldout | CC BY 4.0 | yes | 2,643 |  |  |  | 2,000 | 2,000 | 2,000 | no |
| laubmann | heldout | unpublished archival material (Nachlass Alfred Laubmann, Bayerisches Hauptstaatsarchiv), internal research use | **no** | 45,102 |  |  |  | 2,000 | 234 | 34 | no |

Scripts in train: Kurrent and other German cursive, Latin cursive (French, English, Swedish, Norwegian, Italian), textualis and
bastarda (inzigkofen, tridis, faithful_transcriptions, htr_school_2024), Fraktur print (gt4histocr) and Antiqua print (lectaurep and
hisclima headers). Languages: mostly German and Low German, then French, Latin, English, Dutch, Scandinavian.

## Licences

Training rows (`train`, `val`) and in-domain `test` rows come only from openly licensed sources: MIT, CC BY, CC BY-SA, CC0, Public
Domain Mark. Every row carries the licence string recorded on its staged row and `open_licence`. `zenodo19c` and `semper` are
decided per row (`row_licences` in `configs/sources.yaml`): the 106 zenodo19c rows from Digitale Schriftkunde (CC BY-NC-SA 4.0) are
out, semper's CC BY and CC BY-SA rows are both open. Recorded conflicts follow the stricter reading unless both readings are open:
gt4histocr (CC BY / CC BY-SA) and fanny_mendelssohn (CC BY / CC0) are in, dresden (CC BY / CC BY-NC-SA), nzz_newspapers
(CC BY / CC BY-NC) and valais_census (CC0 / CC BY-NC) are out. Finnish migration records and the Leopold I letters are out (image
rights not open). Rows that are not openly licensed appear only as `heldout` with `open_licence = false`: `abp` and `laubmann` (internal
archival material, evaluation only). For mkn the
Kurrent records state Public Domain Mark 1.0 on every row and no conflicting statement; it is in.

Out with reasons (one line each in `configs/sources.yaml`): adv_forms (NC-ND), dresden, stabs (NC), nzz_newspapers,
finnish_migration, leopold_letters, hennig_diaries (no licence stated), valais_census, forst (internal), reuss (NC
facsimile, no line rows), koenigsfelden (open, but its transcription text is a graphemic edition with editorial punctuation and
normalised spellings; the v1.5 build drops every transcription row), and the sources without line transcription rows
(reichsanzeiger, chronicling_germany, cbad2019, read_abp_table, uos_logbooks, rainfall_rescue, ctdar19_archival, sbb_tables,
beeproject, pipeline, synth_layout).

## Held-out sources

Never in `train` or `val`; their rows are `heldout` only, whole pages, at most 2,000 lines per source: humboldt (all variants;
the rows come from `staged/humboldt_tight`, the folder the v1.5 build used; `humboldt_pages` has page rows only), parzival,
konzilsprotokolle, bullinger, swiss_council, klosterneuburg_cod48, austrian_newspapers, abp and laubmann (both not open). Pages the
Kurrent project already holds out (its `test` and `holdout` splits, e.g. `humboldt_llm_test50`, `abp_clean_test`) are taken first, then
pages in turn across hands.

Laubmann (round 2, `modal/laubmann.py`, work files in `/pool/v1/build/laubmann/`). The staged pool has Laubmann's diaries only as region
crops (`staged/laubmann`). Lines are cut from the 5,030 transcribed regions whose `extra.line_boxes` count equals the text's line count
(45,102 lines, 3,451 pages), without the regions the Kurrent v1.5 build rejected for crop or label reasons (kept: no reason,
`not_selected`, `dup_text_cap`) and without volume 35 (the species register: Roman numerals and page numbers, region boxes that cut the
line ends, tag remnants in its texts). A line crop is its box widened to the line's ink (pad a quarter of the box height; ruled lines of
the squared paper removed from the ink mask) and 30 % of the box height higher and lower, so ascenders and some neighbouring ink show,
as in the other box-cropped sources. Dropped lines: a box thinner than 0.6 of the region's median, ink running into the region crop's
edge (the line continues beyond the region box; also drops some lines next to the book edge), text with tag remnants, and the pool's
width and size rules. Text/crop alignment was checked by two readers: on every cut line k2 and McCATMuS scored the line's own text
against the texts of the lines above and below in the region; a region was dropped when either reader preferred a neighbour's text on
any of its lines (64 of 441 regions, 803 lines; every flagged line looked at was shifted by one line, mostly after a short line whose
band was missed). Pages were then taken whole, held diary pages first (gold text), then in turn across the 34 volumes, up to 2,000 lines:
2,000 lines, 234 pages; 42 lines with human-corrected text (`quality = gold`), 1,958 with the machine text layer of the Laubmann
knowledge graph (`high_silver`; CER 2 to 6 % by band in the Kurrent calibration). 64 random lines of the final part were read by eye
against their crops (contact sheets in `build/laubmann/sheets/`, 160 lines): no shifted line left; residues are machine-text errors on silver lines (e.g. "zurück" where the
ink reads "Zugspitze") and newspaper and typewritten clippings pasted into the volumes (4 of the 64 lines read), which the
Kurrent staging labelled `script = kurrent` unless it recognised them as inserts (107 rows `antiqua`). `hand` is the volume
(`laubmann/<volume>`); `polygon` and `baseline` are empty; `origin` names the staged region and the line's box in it. A pool rebuild
with the `write` step removes `lines/laubmann/`; run `modal/laubmann.py` `cut`, `check`, `write` and then `pool.py --steps finish`
again (the docstring lists the commands).

## Rules

A staged row is a candidate when

1. `unit == line` and `task == transcribe` in the Kurrent v1.5 build decisions (`kurrent-raw:/build_corpus_v1_5_r5/decisions.parquet`,
   which reads `staged_v15`, then `staged_v14`, then `staged`; a re-staged folder replaces the older one);
2. it passed that build's rules (`keep`, or rejected only by the selection or by rules that do not concern a line crop:
   `not_selected`, `line_region_twin`, `page_shared_with_test`, `geometry_only_target`, `abbreviation_star`). Every other build
   reason drops the row: duplicates and near-duplicate crops, held-out leakage by page hash, cross-source duplicate pages, the density,
   aspect and ink checks, short crops, readers against the label, editorial brackets, unbalanced markup, the quality-gate drops;
3. its text, after NFC, removal of U+FEFF and U+200B, and folding tabs and Unicode spaces into one space, is a single line with at
   least 2 visible characters (combining marks not counted);
4. the crop width per visible character lies between 0.08 and 2.5 crop heights, the crop is at least 16 px high and at least 3 px
   wide per visible character;
5. semper: rows with the edition's markers (`TRENNUNG`, `QUOTE`, `KLAMMER...`) are dropped and its line-end bar `|` (not on the page)
   is removed; konzilsprotokolle: pages whose text is uncorrected recognition output are dropped whole (7 of 330 pages:
   Copy_of_1794-95 pages 112, 113, 117 to 119 and Copy_of_1796-97 pages 3 and 4; found by eye, recognised by tokenised punctuation
   `wort , wort` on at least 20 % of a page's lines, which no corrected page of the source has);
6. its licence is open, or the source is a held-out source.

Splits are decided by page (`source`, `doc`, `page`); a page belongs to one split only:

- `test` (training sources): the source's pages that the Kurrent project holds out (its published test sets and unseen-hand pages)
  come first; if they hold fewer lines than the target, whole random pages are added. At most 300 lines and at most 10 % of the
  source's candidates. Every reserved page is excluded from train and val, even where only some of its lines are drawn.
- `val`: 1 % of the remaining pages of every training source (at least 1), at most 500 lines per source.
- `train`: the remaining pages. One cap for every source (water-filling to 300,000 lines, at most 30,000 per source) gives
  14,385 lines per source; sources with fewer candidates give all of theirs. Within a source the draw takes one line of every page
  before a second line of any page, and moves through the hands in turn (first page of every hand, then the second ...), so the lines
  spread over as many pages and hands as the source has. Print (Fraktur, Antiqua, typewriter) is capped at 40,000 lines in total;
  the pool has 10,042 (the cap does not bind).
- `heldout`: whole pages of the held-out sources, at most 2,000 lines each.

`hand` is the writer where the source records one (Bullinger, Nuremberg, Inzigkofen scribes), otherwise the hand proxy of the
Kurrent selection: the document for collections that mix writers, the collection otherwise (Grand Tour per document, SCADS per
transcript). Print rows have `hand = ""`. Hand counts are therefore counts of proxies, not of verified writers.

## Text

`text` is the staged transcription as the source publishes it (only technical clean-up, see rule 3), not the Kurrent build's
canonical rendering: that rendering expands abbreviations on half the rows that carry an expansion and normalises letters, which
moves the text away from the ink. Conventions therefore differ per source (`configs/sources.yaml`, and the Kurrent profiles in
`kurrent-vlm/configs/conventions_v1_5.yaml`). Markup that appears in `text`:

| mark | meaning | where mostly |
|---|---|---|
| `~~...~~` | struck text (ink present) | semper, nuremberg_letterbooks, aaeb, hanse_xv-xvii |
| `<u>...</u>` | underlined (tags have no ink) | basel_grundbuch, grandtour, unibe_xx, zenodo19c, bern_xix |
| `^...^` | raised letters (markers have no ink) | lectaurep, basel_grundbuch, aaeb, hanse_xvi |
| `[...]` | gap, unread text | nuremberg_letterbooks, zenodo19c |
| `¬`, `⸗`, `=`, `-` at the line end | word continues on the next line (ink: a hyphen sign; `¬` is added by some editions where the page has none) | zh_rrp, bern_xix, hanse, read_icfhr16, htr_school_2024 |
| `*` | abbreviation sign as written (nuremberg_letterbooks) | nuremberg_letterbooks |
| `✳`, `·`, `•` | scribal punctuation | colmar_adhr, inzigkofen, klosterneuburg_cod48, htr_school_2024 |
| combining marks (U+0305, U+0304, U+0364, U+035B, ...) | abbreviation bars, small letters above | medieval sources |

Struck words are dropped from Humboldt's text (`deletions: dropped`), Hanse texts expand abbreviations in place, zh_rrp is an
edition text aligned to the images (high_silver), faithful_transcriptions is crowd work (high_silver).

## Crops and geometry

Crops are the staged crops as stored: the line polygon's bounding box with about 10 % vertical and 4 to 8 px horizontal padding,
cut at source resolution and resized to a height of at most 160 px (JPEG q85). Many crops of slanted or tightly written lines
contain parts of the neighbouring lines. Humboldt's crops are deskewed (`staged/humboldt_tight`). `polygon` is empty everywhere (the
staged rows do not carry line polygons). `baseline` is the source's page baseline mapped into crop pixels (`(x - x1) * w / (x2 - x1)`,
same for y) for the sources that record one; dropped when a point falls more than a quarter of the crop height outside the crop.
Checked on contact sheets (`sample/sheets/`) and by position: baselines span 94 to 100 % of the crop width and sit at 55 to 70 % of
its height in every source that has them.

Crop height in px (the staged crops are never upscaled, so 160 is the ceiling; 30 % of the train crops sit at it):

| split | p1 | p5 | median | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| train | 36 | 48 | 124 | 160 | 160 |
| val | 42 | 62 | 131 | 160 | 160 |
| test | 37 | 48 | 121 | 160 | 160 |
| heldout | 35 | 44 | 141 | 160 | 160 |

Median crop width about 1,040 px (p1 122, p99 2,724 in train). Smallest crop 16 px high (rule 4).

## Audit by eye

20 random pool rows of every source (780 rows; an ad-hoc contact-sheet job, sheets on the volume under `checks/`) plus the sample
sheets: text and crop agree in every source; no source pairs texts with the wrong line (no index shift as in the Kurrent forst
rows). Found: uncorrected recognition text on 7 konzilsprotokolle pages (now dropped, rule 5), one garbled Bullinger line
(`ofisit e i i ui t si oi it ei,`; Bullinger is silver), and crops that show two or three lines (zh_rrp, colmar_adhr, grandtour,
humboldt) where the text is the middle one; the baseline marks it where the source has one.

## Characters

467 distinct characters in train (189 of them fewer than 10 times, 297 fewer than 100), 187 in heldout. Counts per split are in
`summary.json` (`characters`). Frequent non-letters besides punctuation: `*` (39k, nuremberg), `~` (32k, struck-text markup), `ſ`
(100k), `¬` (19k), `ʒ` (18k), combining overline and macron (11k), combining e and zigzag above (7k), `⸗`, `✳`, `·`, `•`, `^`,
`<u>` tags, Greek letters (semper), `℔`.

## Rebuild

```
PYTHONUTF8=1 MSYS_NO_PATHCONV=1 uv run modal run --detach modal/pool.py --steps scan,plan,sample,write,finish
```

Steps: `scan` (staged metadata of every line row of the wanted sources, 50 tasks), `plan` (rules, splits, draw; no image is read),
`sample`, `write` (reads the images of the drawn rows only, one task per part, at most 10 containers), `finish` (index and summary).
The draw is deterministic (seed 20261109). Copy the sample locally with
`uv run modal volume get littera /pool/v1/sample <dir>` and keep `lines/` and `index.parquet`.

## Reading

```python
from littera.data import iter_rows, load_image, read_index, read_rows

index = read_index(split="train", columns=["uid", "source", "width", "height", "text"])   # local sample, $LITTERA_POOL or /vol/pool/v1
for row in iter_rows(split="heldout", sources=["humboldt"]):
    image = load_image(row)
rows = read_rows(["humboldt/H0005449/H0005449_97v/z004_l002"])
```

`pool_root()` takes an explicit path, else `$LITTERA_POOL`, else `/vol/pool/v1` when mounted, else `data/pool_sample`.
