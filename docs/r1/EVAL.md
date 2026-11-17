# Round 1, agent C: evaluation

Built: metrics and runner (`littera.eval`), synthetic exact-boundary lines (`littera.synth`), word-gold converters
(`littera.eval.wordgold`), mismatch and letter-gold-candidate builders, two baselines (`littera.baselines`), the Modal app
`littera-eval` (`modal/eval.py`) and 20,000 synthetic training rows. Baseline numbers: [BASELINES.md](BASELINES.md).

## Evaluation sets

Volume `littera`, `/eval/<set>/lines.parquet` (+ `README.md` with source, licence and counts), schema of the brief
(`littera.eval.sets.SCHEMA`; mismatch rows add `variant`). Copies under `data/eval/<set>/` (gitignored) where under 50 MB.

| set | rows | kind | gold | source | licence |
|---|---|---|---|---|---|
| `gw20-words` | 657 lines, 4,860 words | word | hand-corrected word boxes | George Washington letter books, 20 pages (CIIR UMass, Lavrenko/Rath/Manmatha 2004) | research only, no redistribution; eval only |
| `iam-words` | 999 lines, 7,786 words | word | word = union of its connected components; "automatic segmentation, verified manually" | IAM 3.0 test set (128 writers), Zenodo 19003721 | CC BY-NC 4.0; eval only |
| `synth-heldfonts` | 2,000 lines, 13,286 words, 84,442 cuts | synthetic | exact word ink extents and letter cuts | 19 held-out fonts, texts of heldout pool lines | fonts OFL-1.1; texts from the pool |
| `mismatch-heldout` | 3,785 rows (960 lines x right/swap/replace/delete) | mismatch | `label`, `true_text`, `variant` | 8 heldout pool sources, 120 lines each | pool licences (`abp` internal) |
| `letter-gold-candidates` | 119 lines (106 pages) | letter | none yet (owner marks) | 8 heldout sources + test pages of `tridis`, `aaeb` | pool licences (`abp` internal) |

Plus `/synth/v1/lines/synth/part-0000[0-7].parquet` (8 x 2,500 rows, about 53 MB each, `README.md`, `fonts.json`): 20,000
synthetic **training** rows (training fonts only, texts of
open-licence train rows, pool schema + `gold_words` + `gold_cuts`, split `train`). Add `/vol/synth/v1` to
`TrainConfig.pool`; agent A's `read_rows` reads them unchanged (checked locally).

Text/crop check (coordinator's warning): 7 to 8 `right` rows per heldout source viewed by eye (60 rows, including two rows of
the flagged page `konzilsprotokolle/Copy_of_1796-97/Copy_of_1796-97_00000004`): every text matched its crop. A systematic
check over all 960 mismatch lines with the two kraken models is in BASELINES.md (per-source line_fit AUC, right vs swap).

## Metrics (`littera.eval.metrics`)

- Letter width = the line's ink width (gold words' extent, else the ink extent of the core band) / visible characters.
- Word boundary error: |predicted - gold| of both ends of every gold word, in letter widths; median, p90, share within
  0.25 and 0.5 letters; word x-IoU; share of words with both ends within 0.5.
- Letter cuts: the same for every gold cut between two visible characters of a word (`(chars[i-1].x1 + chars[i].x0) / 2`).
  Gold cut = the boundary before `text[i]`.
- Ink-gap rate: share of boundaries between consecutive predicted words (midpoint of the gap) that fall on an ink-free
  column of the line's core band (Otsu mask, rows at least half as dense as the densest row, padded; at most 2 % of the
  densest column counts as empty).
- Mismatch: AUC of `line_fit` (right text against each wrong variant and all together), AUC of word `fit` with the
  replaced word as the negative.
- Calibration of `q` (A's definition): per character with both gold cuts, "both cuts within 0.25 letter" against its `q`
  (share, mean `q`, AUC, expected calibration error over 10 bins); per word, "both ends within 0.5 letter" against the
  word's `q`; also `min(q)` at a cut against "cut within 0.5 letter".

Runner: `python -m littera.eval.run all --placer <spec> --set <name|dir|parquet> [--root data/eval] [--out runs/eval]
[--shard k --shards n]` writes `<out>/<placer>/<set>/placements-<k>.jsonl` (uid + `Placement.to_dict()`),
`rowscores-<k>.jsonl` and `scores.json`. Placer specs: `even`, `kraken-k2`, `kraken-mccatmus`, `kraken:<model>[@raw]`,
`littera:<run or model dir>` (agent A's `Aligner.load`). On Modal: `modal run modal/eval.py::baselines --placers ...
--sets ...` (shards of 400 rows, at most 10 containers), then `python -m littera.eval.report` prints the tables.

## Synthetic lines (`littera.synth`)

- Fonts: 70 openly licensed fonts (`python -m littera.synth.fonts fetch --dest data/fonts`; `fonts.json` records URL,
  licence, sha256, licence file; uploaded to `/vol/fonts`). Google Fonts GitHub (OFL, one Apache), fontlibrary.org (OFL)
  and Peter Wiegel's site (OFL/GPL+FE dual, the zips carry the OFL text). Textura Libera (M+ licence) was dropped to keep
  to OFL/Apache/PD. Styles: 24 script, 13 hand, 5 Kurrent/Kanzlei/Sütterlin (18th Century Kurrent, Blackletter Hand,
  Kanzleyrath | held out: Greifswalder Deutsche Schrift, Kanzler), 10 Fraktur/blackletter, 4 medieval, 14 historical print
  (IM Fell, Junicode, Garamonds, Caslon...). 19 fonts (about a quarter of each style) are held out for `synth-heldfonts`
  and never used in training rows.
- Exact cuts: basic layout (no ligatures), pen position before `text[i]` = advance of `text[:i+1]` minus advance of
  `text[i]` (keeps kerning), non-decreasing. Slant is a shear about the baseline, applied to image and cuts; the cut is
  reported at mid x-height. Gold word extents: each word rendered alone at its pen position, through the same shear and
  baseline wobble, ink columns >= 96/255.
- Degradations: x-height 14 to 38 px, slant (handwriting styles around 0.08 +- 0.14, print mostly 0), baseline wobble,
  neighbouring lines above and below (65 %), stroke dilation/erosion, paper colour + texture + stains, ink colour and
  density variation, blur, noise, down/up-scaling, grey conversion, JPEG 55 to 95, random crop margins (descenders and
  ascenders may be cut). Half the rows carry a band polygon and a baseline.
- Use in training: `SynthGenerator(fonts_dir, split="train", texts=...).render(text)` gives one `SynthLine`;
  `write_training_part(...)` writes pool-style parquet; Modal `build_synth_train --count N --parts P --first-part K`
  appends parts. Fonts with no glyph for a character of the text are skipped; a text no font covers is skipped.

## Word gold: what exists without registration

| candidate | status |
|---|---|
| ICFHR 2016 H-KWS, Botany and Konzilsprotokolle (prhlt.upv.es) | **not downloadable**: the data URLs loop on a 301 to themselves (checked 2026-11-09, with the missing intermediate certificate supplied), the Wayback Machine has the page but none of the zips. Also, only 3 training pages per set had hand-verified word boxes (b0001, b0002, b0009 / k0007-k0009); the rest came from forced alignment. If the owner gets the data from the organisers (PRHLT Valencia / DUTH Xanthi): the XML is `<spot word image x y w h>`, a 30-line converter in the style of `gw20.py`. |
| H-KWS 2014 Bentham (vc.ee.duth.gr) | downloadable, 50 pages, hand-made word boxes, **no transcriptions**; pairing with the tranScriptorium Bentham transcripts (Zenodo, ICFHR 2014 / ICDAR 2015 HTRtS) is possible but not done |
| GW20 (CIIR UMass) | **built** (`gw20-words`) |
| IAM 3.0 | now on Zenodo without registration (record 19003721, CC BY-NC 4.0, migrated with the authors' permission): **built** (`iam-words`) |

### Owner downloads (registration or request; not done by me)

1. IAM-HistDB (Saint Gall, Parzival, Washington): register at https://fki.tic.heia-fr.ch/databases/iam-historical-document-database,
   download the three archives, unpack each into its own folder, then
   `uv run python -m littera.eval.wordgold saintgall --input <folder>` (also `parzival`, `washington`) -> `data/eval/<name>-words/`.
   The converter uses word polygons from the SVG locations when present, else locates each normalised word image inside
   its normalised line image (exact template match); it decodes the `s_pt`-style codes and reports unknown ones. It is
   written against the documented layout and **not tested** on the real archives: check the printed stats.
   Parzival lines are in the pool as held out; the IAM-HistDB word positions would be the matching word gold.
2. Esposalles (CVC Barcelona, on request): http://dag.cvc.uab.es/the-esposalles-database/ (the site did not answer on
   2026-11-09). `uv run python -m littera.eval.wordgold esposalles --input <folder>` pairs each line image with its word
   images (`..._Word<k>`) and their `.txt` transcripts. Also untested.
3. Upload a built set with `uv run modal volume put littera data/eval/<name>-words /eval/<name>-words`.

## Mismatch set

From 120 lines per heldout source (round-robin over pages, 3 to 160 characters, at least 2 words): `right` (label 1),
`swap` (another line of the same page; `swap-doc` when the page has only one line, mostly `swiss_council`), `replace` (one
word of >= 3 letters replaced by a word of similar length from the page or document), `delete` (one word removed, lines
with >= 3 words). `abp` is the only non-open source among them.

## Letter-gold candidates

119 lines, 12 per source (`austrian_newspapers` 11: six pages): the 8 heldout sources (`abp`, `austrian_newspapers`,
`bullinger`, `humboldt`, `klosterneuburg_cod48`, `konzilsprotokolle`, `parzival`, `swiss_council`) plus test pages of
`tridis` (medieval Latin/French/Dutch, textualis and cursive; its texts expand abbreviations, so some letters have no ink of
their own) and `aaeb` (French cursive, 16th to 18th century). `laubmann` is not in the pool. `hisclima` was tried first and
dropped: its "handwritten" rows are mostly printed form headers whose handwritten values lie outside the crop. 60 drawn per source
(15 to 60 characters, height >= 40 px, plausible width per character, no antiqua/typewriter print, unique texts), ranked by
max(line_fit of aligner-k2, line_fit of McCATMuS) - 1.5 x share of ink outside the core band, at most one per page where
the source has enough pages. Box-cropped sources (`humboldt`, `abp`) still show neighbouring lines; the marking page
should show the crop as is and let the owner mark on the main line. The baselines' placements on this set
(`/vol/eval_runs/<placer>/letter-gold-candidates/placements-*.jsonl`) can pre-fill the marking page.

## Spend

Modal app `littera-eval` on 2026-11-09 (`modal billing report --for today`, summed by app name): **0.41 USD** of 1.50.
Baselines 0.16 (including a first GW20 run and merges), synthetic training rows 0.11 plus 0.05 for a first launch I stopped
because it would have run 21 containers at once, set builds (IAM, synthetic, mismatch, three candidate rounds) 0.08, probes
and model download 0.01. No paid APIs.

## Open

- Letter gold: the owner marks `letter-gold-candidates` (marking page next round); then rerun the baselines on it.
- Registration-only word gold (IAM-HistDB Saint Gall, Parzival, Washington; Esposalles): owner download, converters untested.
- H-KWS 2016 Botany/Konzilsprotokolle: data offline; only via the organisers. Bentham (H-KWS 2014): boxes without text,
  pairing with the tranScriptorium transcripts not done.
- `iam-words` is easy for word edges (tight component boxes and crops); weight GW20 and the letter gold more.
- `swiss_council` has one line per page, so its swaps come from the same document (`swap-doc`).
- If agent B rebuilds the pool, the eval sets stay valid (they carry their own images and texts); `uid`s point to
  `/pool/v1` as of 22:09 (heldout rows unchanged across the rebuilds I saw).
- `q` of the baselines is badly calibrated (character ECE 0.33, word ECE 0.4 to 0.7); the metrics now use A's thresholds
  (0.25 letter per character, 0.5 per word). `python -m littera.eval.run rescore` / Modal `rescore` recompute row scores
  from stored placements when a metric changes.
