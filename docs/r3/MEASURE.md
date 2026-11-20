# Round 3 measurement on Humboldt (agent M, 2026-11-17)

Code: `modal/humboldt.py` (app `littera-humboldt`), `src/littera/eval/humboldt.py`, `humboldt_cnn.py`,
`humboldt_occlusion.py`, `humboldt_data.py`. Numbers: `results/r3/*_first_table.json`. Data: `docs/r3/DATA.md`.
There is no letter gold on Humboldt. Every measure below is a proxy, and what each can show is stated with it.

## Placers

All on the 1,000 lines of `/vol/eval/humboldt-1000`, edges in pixels of the upright production crop.

| name | what |
|---|---|
| `even` | characters (spaces too) spread evenly over the line's ink extent |
| `k2-raw-mid`, `k2-snap-mid` | aligner-k2's cuts from humboldt-htr's `place-k2.jsonl`, variants `raw-mid` and `snap-mid` (the lookup page's) |
| `pilot-upright` | `littera-pilot-seg`, `snap="upright"` |
| `ft2-off`, `ft2-upright` | `littera-humboldt-ft2`, `snap="off"` and `snap="upright"` (the lookup build under review) |
| `even@<name>` | the word edges of `<name>`, the characters of each word spread evenly between them |
| `mix:<inner>+<outer>` | the word edges of `<outer>`; the letter edges inside each word from `<inner>`, clipped to the word |

All littera placements are made as production makes them: `aligner.place(crop, text, baseline=points)` with the
baseline points of the line's pieces in crop pixels and no polygon; the text is the recipe's without the characters
written above the line. aligner-k2's cuts were made in October on the same crops (the crop widths in `place-k2.jsonl`
equal ours on all 1,000 lines) with the same characters skipped. littera takes 0.32 to 0.35 s per line on 2 CPU
cores. `even@` and `mix:` are derived when scoring and have no file.

## The neutral strip

Both placers snap word edges on a strip of their own (aligner-k2: kraken's centre-normalised line; littera: its
straightened and scaled line along the detector baseline). The measures use a third one, built from the crop alone
(`humboldt.neutral_strip`):

1. Grey crop, 0 to 1. Pixels at 0.985 or above are the white outside the line's mask and are left out.
2. Otsu's threshold over the darkness of the remaining pixels (64 bins) separates paper and dark ink. The line's
   course and the x-height band (steps 3 to 5) use this mask.
3. The line's course: in windows as wide as the crop is high (at least 48 px, half overlapping) the row that halves the
   window's ink; a running median over three windows; linear interpolation between window centres.
4. Every column is shifted up or down by a whole number of pixels so that this course is level. Nothing is sheared or
   scaled: column `x` of the strip is column `x` of the crop.
5. x-height band: the rows around the densest row (within a quarter of the crop height of the course) whose ink
   density is at least half of the peak, contiguous. Core band: the x-height band widened by a quarter of its height
   (at least 2 px) above and below, so that joining strokes on the baseline count.
6. Ink for the column profile: darker than the point halfway between the paper level (median of the pixels below
   Otsu's threshold) and Otsu's threshold. A column is ink-free when the core band holds at most
   `max(1, 2 % of the largest column count)` such pixels in it (the rule both placers use on their own strips).

Two choices were made after first results, both from looking at overlays and ink masks of single lines, not at the
comparison of placers. (a) The widening in step 5: the unwidened band was 8 to 13 px high. (b) The halved threshold
in step 6: in the later letters, hairlines stay below Otsu's threshold and whole letters counted as ink-free
(`Wenn` in `H0001182_1v_z001_l023`). At a third of the way the mask picks up paper specks; halfway it does not on
the lines looked at, and some hairlines are still missed. With Otsu's threshold the lift rate is higher, most of all
in the letters:

| | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| lifts per 100 letter boundaries, Otsu's threshold | 25.5 (23.3 to 28.1) | 27.9 (25.3 to 30.8) | 26.6 (24.9 to 28.4) | 29.2 (26.5 to 32.1) | 23.0 (21.1 to 25.3) | 44.1 (40.1 to 48.6) | 33.0 (30.9 to 35.3) |

The first table computed with Otsu's threshold is kept in `results/r3/score_first_table_otsu.json`; the order of the
placers is the same in both.

## Measures

### Word boundaries

For every pair of neighbouring words: the midpoint between the end of one and the start of the
next lies within 1.5 px of the centre of an ink-free column. All word pairs of the text count, for every placer.
Raw (`k2-raw-mid`, `ft2-off`) and as deployed (`k2-snap-mid`, `ft2-upright`). It cannot tell a boundary in the right
gap from one in a wrong gap, and it says nothing about letters.

### Pen lifts

A run is at least 2 neighbouring ink-free columns between the first and the last inked column of the
line. Two reference placements, `k2-snap-mid` and `ft2-upright`, sort the runs: a run whose centre lies inside the
same word for both is a *lift*; between the same two words for both, a word gap; otherwise disputed (left out).
This sorting uses the references' word extents only; their letter edges play no part in it.
The hand's lift rate is lifts per 100 letter boundaries inside words. A placer *holds* a lift when one of its letter
edges inside that word lies in the run or within 1.5 px of it (also given: within 4 px).
What it cannot show: a run inside a word is not always between two letters. A hairline below the ink threshold, an
open letter form or the gap between two strokes of one letter gives a run too, so the ceiling of this measure is below
100 % by an unknown amount, and the lift rate is an upper bound on pen lifts between letters. Lifts hidden by the
slant of the hand (no upright ink-free column) are not counted. Even spacing holds a quarter of the lifts by chance.
The measure is circular for any placer that moves letter edges into ink-free runs. A high value for such a placer
means that its snapping works and that its strip agrees with this one. It does not mean that the letter is right.
None of the placers in the first table does this (their snapping moves word edges only).

### Same boundary

Among the lifts that a placer and `k2-snap-mid` both hold: the share where both put the same letter
boundary of the text into the run. This is agreement between two placers. Both can be wrong.

### Letter-crop classifier (`humboldt_cnn.py`)

For each placer every character that is not a space is cut at its
span (columns `floor(x[i])` to `ceil(x[i+1])`) from the neutral strip's grey image, rows from 1.2 x-heights above the
x-height band to 1.0 below it, as darkness against the line's paper level, and scaled to 32 x 32. A character without
width gives an empty crop. Within each half the lines are split in two folds by the MD5 of the uid; a CNN (three
blocks of convolution, batch norm and pooling with 32, 64, 128 channels; 256 hidden units; AdamW, one-cycle to 0.002,
15 epochs, batch 128, label smoothing 0.05, no augmentation) is trained on one fold's letters and names the other
fold's. Classes are the characters with at least 10 letters in the training fold. Two seeds; a letter scores the
share of seeds that named it. The recipe and the test letters are the same for every placer.
Connected words: these are words of three or more characters that hold no lift and no disputed run; their letters except the
first and the last are the letters "inside connected runs". The set comes from the two references' word extents and
the neutral strip and is the same for every placer.
What it can and cannot show: more consistent crops of a letter give a higher accuracy, so a placer whose spans drift
loses. It does not test the convention: spans shifted the same way everywhere are as easy to learn as correct ones,
and a span that takes in the same piece of the neighbour every time is not punished. The training folds are small
(dev: about 6,500 letters), so accuracies are low in absolute terms; only differences between placers on the same
letters mean something. Differences are given against `k2-snap-mid` with a paired bootstrap.
Control for the span's width: a placer knows the text, and a span's width can give the letter away. The control
names a letter from its width alone (width over the line's median width, 20 quantile bins, the commonest letter of
the bin, same folds). If it is no better than always saying the commonest letter, widths leak nothing.

### Occlusion evidence (`humboldt_occlusion.py`)

The reader is TrOCR `t11_r6g6_AL` of humboldt-htr, which reads
whole lines and is none of the aligners. The crop is resized to 384 x 384 as the reader expects. A bar two mean
character widths wide (crop width over the number of characters) slides along the line in steps of half a character
width; inside the bar every pixel of the line's mask is set to the paper colour (median of the mask's pixels). For
each bar position the text is scored teacher-forced, and for each token the fall of its log-probability against the
unoccluded line is kept (falls below zero count as zero). A token's evidence centre is the centre of mass of the fall
over the bar centres around the peak where the fall is at least half the peak. Tokens whose peak fall is below
1.0 nat have no evidence and are left out. The evidence is computed once per line and is the same for every placer.
For a placer: the share of tokens whose centre lies inside the span of the token's characters, and the mean distance
between the centre and the middle of that span in character widths (signed: positive when the evidence lies to the
right of the span's middle).
What it cannot show: one ViT patch is 16 of 384 pixels, about 1.7 character widths on a line of 40 characters, so the
measure resolves about a letter or two, and tokens are word pieces of several letters. It can show a placer whose
spans are off by a letter or more; it cannot rank placers that differ by half a letter. The reader also uses context,
so a token's evidence can sit on its most distinctive stroke and not on its middle.

### Intervals

95 % percentile intervals from 1,000 bootstrap resamples of lines (seed 0), within the group shown.

## First table

Dev half (val, test_page; 400 lines):

| placer | word boundaries | lifts held | letters named | letters named, connected words | letters, minus k2 snap-mid | connected, minus k2 snap-mid | evidence centre inside the token's span | its offset, letters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 40.3 (37.8 to 42.5) | 26.4 (24.3 to 28.4) | 12.0 (11.5 to 12.6) | 14.5 (13.3 to 15.7) | -26.5 (-27.8 to -25.2) | -23.1 (-25.2 to -21.0) | 44.8 (42.5 to 47.3) | 2.36 (2.01 to 2.76) |
| aligner-k2, raw-mid | 87.4 (85.3 to 89.3) | 49.8 (47.4 to 51.9) | 48.0 (46.4 to 49.8) | 48.6 (46.1 to 51.2) | 9.5 (8.6 to 10.5) | 11.0 (9.0 to 13.0) | 74.7 (72.7 to 76.4) | 1.29 (1.01 to 1.71) |
| aligner-k2, snap-mid (deployed) | 92.4 (90.6 to 93.8) | 37.2 (34.8 to 39.5) | 38.5 (37.1 to 39.9) | 37.6 (35.5 to 39.7) |  |  | 72.2 (70.2 to 73.8) | 1.29 (1.01 to 1.71) |
| k2: snap-mid's word edges, raw-mid's inner edges | 92.4 (90.6 to 93.8) | 49.8 (47.4 to 51.9) | 48.1 (46.4 to 49.7) | 48.9 (46.4 to 51.3) | 9.6 (8.7 to 10.5) | 11.2 (9.0 to 13.3) | 74.2 (72.3 to 76.0) | 1.28 (0.99 to 1.70) |
| even inside k2 snap-mid's words | 92.4 (90.6 to 93.8) | 40.1 (37.8 to 42.3) | 28.6 (27.5 to 29.8) | 25.8 (24.2 to 27.5) | -9.9 (-10.8 to -9.0) | -11.9 (-13.7 to -9.9) | 70.7 (68.6 to 72.3) | 1.32 (1.03 to 1.74) |
| littera-pilot-seg, `upright` | 87.0 (84.6 to 89.0) | 31.6 (29.5 to 33.7) | 24.4 (23.4 to 25.6) | 23.4 (21.8 to 25.3) | -14.1 (-15.1 to -13.0) | -14.2 (-16.4 to -11.8) | 59.8 (57.7 to 61.9) | 1.68 (1.40 to 2.01) |
| littera-humboldt-ft2, `off` | 86.1 (84.2 to 87.7) | 48.6 (45.8 to 51.1) | 46.1 (44.5 to 47.8) | 45.2 (42.7 to 47.6) | 7.6 (6.5 to 8.8) | 7.5 (5.2 to 9.7) | 73.6 (72.0 to 75.3) | 1.23 (0.96 to 1.55) |
| littera-humboldt-ft2, `upright` (deployed) | 92.9 (91.3 to 94.4) | 37.9 (35.5 to 40.3) | 38.0 (36.6 to 39.5) | 36.8 (34.7 to 38.7) | -0.5 (-1.4 to 0.4) | -0.9 (-2.8 to 0.9) | 73.2 (71.5 to 74.8) | 1.26 (0.99 to 1.58) |
| ft2: `upright`'s word edges, `off`'s inner edges | 92.9 (91.3 to 94.4) | 48.6 (45.8 to 51.1) | 46.3 (44.7 to 48.0) | 45.4 (42.8 to 47.9) | 7.8 (6.7 to 8.9) | 7.8 (5.5 to 10.0) | 73.9 (72.1 to 75.6) | 1.24 (0.97 to 1.56) |
| even inside ft2 `upright`'s words | 92.9 (91.3 to 94.4) | 37.0 (34.8 to 39.1) | 28.3 (27.2 to 29.4) | 25.5 (23.8 to 27.4) | -10.2 (-11.2 to -9.2) | -12.1 (-14.2 to -10.2) | 70.2 (68.5 to 71.9) | 1.29 (1.02 to 1.61) |

Report half (test_bench, test_journal, letters; 600 lines):

| placer | word boundaries | lifts held | letters named | letters named, connected words | letters, minus k2 snap-mid | connected, minus k2 snap-mid | evidence centre inside the token's span | its offset, letters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 43.4 (41.2 to 45.5) | 25.2 (23.6 to 26.8) | 14.7 (14.2 to 15.2) | 16.8 (15.7 to 17.9) | -32.0 (-33.2 to -31.0) | -30.8 (-32.7 to -28.9) | 48.2 (46.5 to 49.7) | 1.72 (1.59 to 1.85) |
| aligner-k2, raw-mid | 90.4 (89.1 to 91.6) | 39.3 (37.6 to 41.2) | 56.8 (55.7 to 58.0) | 59.0 (57.0 to 60.8) | 10.1 (9.4 to 10.8) | 11.4 (9.9 to 12.8) | 75.7 (74.6 to 76.7) | 0.93 (0.85 to 1.04) |
| aligner-k2, snap-mid (deployed) | 95.1 (94.2 to 96.0) | 31.5 (29.9 to 33.4) | 46.7 (45.6 to 47.9) | 47.6 (45.9 to 49.5) |  |  | 72.4 (71.4 to 73.4) | 0.95 (0.87 to 1.05) |
| k2: snap-mid's word edges, raw-mid's inner edges | 95.1 (94.2 to 96.0) | 39.3 (37.6 to 41.2) | 56.5 (55.4 to 57.7) | 58.5 (56.6 to 60.3) | 9.9 (9.2 to 10.5) | 10.9 (9.5 to 12.2) | 75.2 (74.1 to 76.2) | 0.93 (0.84 to 1.03) |
| even inside k2 snap-mid's words | 95.1 (94.2 to 96.0) | 34.5 (32.9 to 36.3) | 35.1 (34.1 to 36.0) | 32.9 (31.5 to 34.4) | -11.6 (-12.4 to -10.9) | -14.7 (-16.3 to -13.1) | 70.6 (69.5 to 71.6) | 0.99 (0.91 to 1.09) |
| littera-pilot-seg, `upright` | 87.9 (86.2 to 89.5) | 27.5 (26.0 to 29.1) | 29.2 (28.1 to 30.2) | 29.5 (27.8 to 31.0) | -17.5 (-18.4 to -16.5) | -18.1 (-19.8 to -16.3) | 60.7 (59.2 to 62.1) | 1.26 (1.14 to 1.40) |
| littera-humboldt-ft2, `off` | 90.1 (88.8 to 91.3) | 38.1 (36.3 to 40.1) | 53.5 (52.4 to 54.6) | 53.8 (52.0 to 55.4) | 6.9 (6.1 to 7.6) | 6.1 (4.6 to 7.7) | 74.4 (73.4 to 75.4) | 0.91 (0.82 to 1.01) |
| littera-humboldt-ft2, `upright` (deployed) | 94.0 (92.8 to 95.0) | 30.6 (28.8 to 32.5) | 44.3 (43.2 to 45.3) | 45.0 (43.3 to 46.7) | -2.4 (-3.1 to -1.6) | -2.6 (-4.3 to -0.9) | 73.5 (72.4 to 74.6) | 0.96 (0.86 to 1.05) |
| ft2: `upright`'s word edges, `off`'s inner edges | 94.0 (92.8 to 95.0) | 38.1 (36.3 to 40.1) | 53.0 (51.9 to 54.2) | 54.0 (52.3 to 55.7) | 6.4 (5.6 to 7.1) | 6.4 (4.8 to 7.9) | 74.9 (73.9 to 75.9) | 0.93 (0.84 to 1.03) |
| even inside ft2 `upright`'s words | 94.0 (92.8 to 95.0) | 31.1 (29.4 to 32.7) | 34.0 (33.2 to 35.0) | 32.9 (31.5 to 34.3) | -12.6 (-13.4 to -11.8) | -14.8 (-16.5 to -13.2) | 70.0 (69.0 to 71.1) | 1.00 (0.91 to 1.09) |

Columns: word boundaries on an ink-free column, %; lifts holding a letter edge within 1.5 px, %; letters named right
by the classifier, all and inside connected words, %; the paired differences of both to `k2-snap-mid` in points; the
occlusion measure (share of tokens, %, and mean distance in character widths).

Reading of the table. Where two numbers are given, the first is the dev half and the second the report half.

1. Word boundaries are level. As deployed, aligner-k2 has 92.4 and 95.1 %, littera-humboldt-ft2 92.9 and 94.0 %; the
   paired difference ft2 minus k2 is +0.5 (-0.6 to 1.6) and -1.1 (-2.0 to -0.3) points. Without snapping 87.4 and
   90.4 % against 86.1 and 90.1 %. In `docs/HUMBOLDT.md` k2 led by 1.6 points on its own strip.
2. Both models know more inside a word than its two ends. Their unsnapped letter edges hold 49.8 (k2) and 48.6 %
   (ft2) of the lifts on dev, even spacing inside the same words 40.1 and 37.0 %. The classifier names 48.0 and 46.1 %
   of the letters from the unsnapped spans, 28.6 and 28.3 % from even spans inside the snapped words. The width
   control is at the level of the commonest letter for k2 (12.6 against 12.7 %) and 2.8 points above it for ft2
   (15.5 %), so the span's width explains none of this for k2 and a few points at most for ft2.
3. The snapping that is deployed loses all of that on the lifts and about half of it on the classifier. `snap-mid` and `upright` move a word's two ends into the ink
   gap and stretch all letter edges of the word in proportion. A letter edge inside a word moves by a median of 0.22
   (k2) and 0.20 (ft2) letter widths, and 13 % of them by more than half a letter. After it the letter edges hold no
   more lifts than even spacing inside the word (k2 37.2 against 40.1 %, ft2 37.9 against 37.0 %), and the classifier
   loses 9.5 (k2) and 8.1 (ft2) points on dev, 10.1 and 9.2 on the report half. The `mix:` rows keep the unsnapped
   letter edges and take only the word edges from the snapped placement: they have the snapped word boundaries, the
   unsnapped lift numbers and the unsnapped letters (classifier 48.1 and 46.3 % on dev, 56.5 and 53.0 % on the
   report half). This needs no new model: `_snap` in `place.py` (and `post` in humboldt-htr's `tokens.py`) would move
   the outer edges and clip the inner ones instead of rescaling them.
4. Inside words littera-humboldt-ft2 is not better than aligner-k2, and on the classifier it is a little worse.
   With the same snapping rule on both sides (`results/r3/classify_raw_against_raw.json`, `classify_mix_against_mix.json`),
   ft2 minus k2 on all letters: unsnapped -1.9 (-2.9 to -0.8) and -3.2 (-4.0 to -2.5) points; `mix:` -1.8 (-2.8 to
   -0.8) and -3.5 (-4.2 to -2.8); as deployed -0.5 (-1.4 to 0.4) and -2.4 (-3.1 to -1.6). On letters inside connected
   words: unsnapped -3.5 (-5.4 to -1.5) and -5.2 (-6.7 to -3.8); as deployed -0.9 (-2.8 to 0.9) and -2.6 (-4.3 to
   -0.9). The width control favours ft2, so its deficit is not an artefact of widths. Lifts held are equal within the
   intervals in every variant. The occlusion measure sees no difference at its resolution (as deployed 73.2 against
   72.2 % of tokens on dev, 73.5 against 72.4 % on the report half; unsnapped 73.6 against 74.7 and 74.4 against 75.7)
   and rules out a drift of ft2 by a letter or more: its mean distance is that of k2.
5. `littera-pilot-seg` without the fine-tune is far behind on every measure (classifier 24.4 and 29.2 %, below even
   spacing inside the fine-tuned model's words).

For the hypothesis of this round: ink-free runs inside words exist at about 20 per 100 letter boundaries (19.8 and
21.7; 24.1 in the letters), and that is an upper bound on usable pen lifts. The best placers hold half of them within
1.5 px and 62 to 64 % within 4 px on dev, so some edges sit a few pixels beside a lift. A fifth of the
letters lie inside connected words (20.6 %), where a lift cannot help directly. A placer that puts edges on lifts will
score high on "lifts held" by construction; the classifier, the letters inside connected words and the blind check
have to show whether its letters are better.

## The hand: ink-free runs

| | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| lines | 200 | 200 | 400 | 200 | 200 | 200 | 600 |
| word boundaries | 1,225 | 1,097 | 2,322 | 920 | 1,302 | 1,518 | 3,740 |
| letter boundaries inside words | 5,418 | 4,909 | 10,327 | 4,495 | 5,850 | 6,837 | 17,182 |
| ink-free runs of 2 px or more | 2,593 | 2,427 | 5,020 | 2,571 | 2,758 | 4,022 | 9,351 |
| of these: between two words | 1,139 | 1,002 | 2,141 | 1,204 | 1,176 | 1,689 | 4,069 |
| inside a word (lifts) | 1,023 | 1,020 | 2,043 | 997 | 1,082 | 1,650 | 3,729 |
| the two references disagree | 287 | 261 | 548 | 252 | 346 | 578 | 1,176 |
| lifts per 100 letter boundaries inside words | 18.9 (17.1 to 20.9) | 20.8 (19.0 to 22.8) | 19.8 (18.4 to 21.1) | 22.2 (19.8 to 24.8) | 18.5 (16.8 to 20.4) | 24.1 (21.9 to 26.8) | 21.7 (20.4 to 23.1) |
| letters inside connected words, % of letters | 21.1 (19.0 to 23.1) | 19.9 (17.5 to 22.2) | 20.6 (18.9 to 22.1) | 21.3 (18.8 to 23.6) | 24.6 (22.6 to 26.6) | 16.7 (14.9 to 18.6) | 20.6 (19.4 to 21.9) |

## Per split

Word boundaries on an ink-free column, %:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 40.6 (37.3 to 44.2) | 39.9 (36.5 to 43.6) | 40.3 (37.8 to 42.5) | 49.7 (45.8 to 53.9) | 36.7 (33.9 to 39.9) | 45.3 (41.4 to 49.0) | 43.4 (41.2 to 45.5) |
| aligner-k2, raw-mid | 87.5 (84.8 to 90.2) | 87.3 (84.5 to 89.9) | 87.4 (85.3 to 89.3) | 93.5 (91.6 to 95.2) | 88.3 (86.0 to 90.4) | 90.3 (88.1 to 92.3) | 90.4 (89.1 to 91.6) |
| aligner-k2, snap-mid (deployed) | 92.0 (89.7 to 94.2) | 92.8 (90.6 to 95.0) | 92.4 (90.6 to 93.8) | 96.5 (94.8 to 97.9) | 95.7 (94.2 to 97.0) | 93.8 (92.2 to 95.3) | 95.1 (94.2 to 96.0) |
| k2: snap-mid's word edges, raw-mid's inner edges | 92.0 (89.7 to 94.2) | 92.8 (90.6 to 95.0) | 92.4 (90.6 to 93.8) | 96.5 (94.8 to 97.9) | 95.7 (94.2 to 97.0) | 93.8 (92.2 to 95.3) | 95.1 (94.2 to 96.0) |
| even inside k2 snap-mid's words | 92.0 (89.7 to 94.2) | 92.8 (90.6 to 95.0) | 92.4 (90.6 to 93.8) | 96.5 (94.8 to 97.9) | 95.7 (94.2 to 97.0) | 93.8 (92.2 to 95.3) | 95.1 (94.2 to 96.0) |
| littera-pilot-seg, `upright` | 86.3 (83.5 to 89.3) | 87.7 (84.7 to 90.6) | 87.0 (84.6 to 89.0) | 89.1 (86.2 to 91.9) | 89.5 (87.0 to 91.9) | 85.8 (83.0 to 88.4) | 87.9 (86.2 to 89.5) |
| littera-humboldt-ft2, `off` | 85.5 (82.8 to 88.0) | 86.8 (84.3 to 89.3) | 86.1 (84.2 to 87.7) | 92.9 (91.0 to 94.8) | 88.2 (85.8 to 90.3) | 90.0 (87.8 to 92.0) | 90.1 (88.8 to 91.3) |
| littera-humboldt-ft2, `upright` (deployed) | 92.7 (90.5 to 94.8) | 93.2 (91.2 to 95.0) | 92.9 (91.3 to 94.4) | 95.4 (93.4 to 97.1) | 94.5 (92.9 to 96.0) | 92.8 (90.8 to 94.6) | 94.0 (92.8 to 95.0) |
| ft2: `upright`'s word edges, `off`'s inner edges | 92.7 (90.5 to 94.8) | 93.2 (91.2 to 95.0) | 92.9 (91.3 to 94.4) | 95.4 (93.4 to 97.1) | 94.5 (92.9 to 96.0) | 92.8 (90.8 to 94.6) | 94.0 (92.8 to 95.0) |
| even inside ft2 `upright`'s words | 92.7 (90.5 to 94.8) | 93.2 (91.2 to 95.0) | 92.9 (91.3 to 94.4) | 95.4 (93.4 to 97.1) | 94.5 (92.9 to 96.0) | 92.8 (90.8 to 94.6) | 94.0 (92.8 to 95.0) |

Lifts holding a letter edge within 1.5 px, %:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 26.7 (24.1 to 29.5) | 26.2 (23.1 to 29.1) | 26.4 (24.3 to 28.4) | 24.3 (21.2 to 27.2) | 25.5 (22.5 to 28.6) | 25.6 (23.4 to 27.8) | 25.2 (23.6 to 26.8) |
| aligner-k2, raw-mid | 49.5 (45.9 to 53.1) | 50.2 (47.0 to 53.2) | 49.8 (47.4 to 51.9) | 36.3 (33.0 to 39.7) | 46.9 (43.4 to 50.4) | 36.2 (33.7 to 38.8) | 39.3 (37.6 to 41.2) |
| aligner-k2, snap-mid (deployed) | 35.7 (32.0 to 39.0) | 38.8 (35.6 to 42.1) | 37.2 (34.8 to 39.5) | 28.2 (25.0 to 31.5) | 39.3 (36.2 to 42.2) | 28.5 (26.0 to 30.9) | 31.5 (29.9 to 33.4) |
| k2: snap-mid's word edges, raw-mid's inner edges | 49.5 (45.9 to 53.1) | 50.2 (47.0 to 53.2) | 49.8 (47.4 to 51.9) | 36.3 (33.0 to 39.7) | 46.9 (43.4 to 50.4) | 36.2 (33.7 to 38.8) | 39.3 (37.6 to 41.2) |
| even inside k2 snap-mid's words | 39.0 (35.4 to 42.6) | 41.3 (38.5 to 44.0) | 40.1 (37.8 to 42.3) | 30.2 (27.1 to 33.5) | 39.7 (36.6 to 43.0) | 33.6 (31.4 to 36.0) | 34.5 (32.9 to 36.3) |
| littera-pilot-seg, `upright` | 30.5 (27.7 to 33.4) | 32.6 (29.1 to 36.1) | 31.6 (29.5 to 33.7) | 25.7 (22.8 to 28.7) | 31.6 (29.0 to 34.2) | 26.0 (23.5 to 28.6) | 27.5 (26.0 to 29.1) |
| littera-humboldt-ft2, `off` | 47.9 (43.8 to 51.9) | 49.2 (45.6 to 52.4) | 48.6 (45.8 to 51.1) | 34.3 (30.7 to 37.8) | 46.3 (43.0 to 49.7) | 35.0 (32.3 to 38.0) | 38.1 (36.3 to 40.1) |
| littera-humboldt-ft2, `upright` (deployed) | 37.1 (33.6 to 40.7) | 38.7 (35.8 to 41.7) | 37.9 (35.5 to 40.3) | 28.2 (25.1 to 31.3) | 37.7 (34.3 to 40.8) | 27.4 (24.9 to 29.9) | 30.6 (28.8 to 32.5) |
| ft2: `upright`'s word edges, `off`'s inner edges | 47.9 (43.8 to 51.9) | 49.2 (45.6 to 52.4) | 48.6 (45.8 to 51.1) | 34.3 (30.7 to 37.8) | 46.3 (43.0 to 49.7) | 35.0 (32.3 to 38.0) | 38.1 (36.3 to 40.1) |
| even inside ft2 `upright`'s words | 36.0 (32.9 to 39.1) | 38.0 (35.2 to 40.9) | 37.0 (34.8 to 39.1) | 28.0 (24.5 to 31.3) | 34.5 (31.5 to 37.6) | 30.7 (28.2 to 33.0) | 31.1 (29.4 to 32.7) |

Lifts with a letter edge within 4 px, %:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 38.4 (34.9 to 42.0) | 36.0 (32.1 to 39.5) | 37.2 (34.7 to 39.8) | 34.8 (31.6 to 38.2) | 38.1 (34.8 to 41.5) | 36.1 (33.5 to 38.4) | 36.3 (34.6 to 38.1) |
| aligner-k2, raw-mid | 64.2 (60.6 to 67.5) | 64.1 (60.7 to 67.2) | 64.2 (61.5 to 66.4) | 47.8 (43.9 to 51.4) | 62.8 (59.3 to 66.2) | 50.0 (47.1 to 52.7) | 53.1 (51.2 to 55.2) |
| aligner-k2, snap-mid (deployed) | 50.9 (47.1 to 54.3) | 53.9 (50.9 to 56.9) | 52.4 (49.9 to 54.7) | 40.9 (37.3 to 44.6) | 53.8 (50.6 to 56.9) | 41.8 (39.0 to 44.6) | 45.0 (43.1 to 47.1) |
| k2: snap-mid's word edges, raw-mid's inner edges | 64.2 (60.6 to 67.5) | 64.1 (60.7 to 67.2) | 64.2 (61.5 to 66.4) | 47.8 (43.9 to 51.4) | 62.8 (59.3 to 66.2) | 50.0 (47.1 to 52.7) | 53.1 (51.2 to 55.2) |
| even inside k2 snap-mid's words | 53.5 (49.8 to 57.1) | 56.8 (53.7 to 59.6) | 55.1 (52.7 to 57.5) | 43.1 (39.5 to 46.9) | 56.2 (53.1 to 59.3) | 45.2 (42.8 to 47.9) | 47.8 (46.0 to 49.9) |
| littera-pilot-seg, `upright` | 41.9 (38.5 to 45.2) | 43.3 (39.7 to 46.9) | 42.6 (40.4 to 44.9) | 37.7 (34.3 to 41.1) | 46.1 (43.3 to 49.1) | 37.1 (34.2 to 40.1) | 39.9 (37.9 to 41.8) |
| littera-humboldt-ft2, `off` | 60.9 (57.1 to 64.5) | 63.1 (59.8 to 66.3) | 62.0 (59.5 to 64.6) | 45.7 (41.8 to 49.4) | 61.5 (58.3 to 65.0) | 45.8 (42.7 to 48.8) | 50.4 (48.3 to 52.4) |
| littera-humboldt-ft2, `upright` (deployed) | 51.4 (47.7 to 55.4) | 55.0 (51.7 to 58.2) | 53.2 (50.7 to 55.6) | 40.3 (36.3 to 44.1) | 53.3 (50.0 to 56.6) | 39.3 (36.2 to 42.2) | 43.6 (41.5 to 45.9) |
| ft2: `upright`'s word edges, `off`'s inner edges | 60.9 (57.1 to 64.5) | 63.1 (59.8 to 66.3) | 62.0 (59.5 to 64.6) | 45.7 (41.8 to 49.4) | 61.5 (58.3 to 65.0) | 45.8 (42.7 to 48.8) | 50.4 (48.3 to 52.4) |
| even inside ft2 `upright`'s words | 51.9 (48.6 to 55.2) | 52.6 (49.5 to 55.6) | 52.3 (50.0 to 54.8) | 41.7 (37.7 to 45.6) | 52.5 (49.5 to 55.6) | 42.9 (40.3 to 45.6) | 45.4 (43.4 to 47.4) |

Same boundary in the lift as `k2-snap-mid`, % of the lifts both hold:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 43.1 (33.3 to 52.5) | 41.6 (31.9 to 50.4) | 42.3 (36.1 to 48.9) | 52.0 (40.7 to 62.0) | 44.9 (37.4 to 52.4) | 45.7 (38.2 to 53.5) | 47.0 (42.3 to 51.6) |
| aligner-k2, raw-mid | 95.4 (92.3 to 98.0) | 96.8 (94.8 to 98.7) | 96.2 (94.3 to 97.7) | 95.6 (92.4 to 98.1) | 95.9 (92.9 to 98.4) | 94.2 (91.1 to 96.8) | 95.2 (93.5 to 96.7) |
| aligner-k2, snap-mid (deployed) | 100.0 (100.0 to 100.0) | 100.0 (100.0 to 100.0) | 100.0 (100.0 to 100.0) | 100.0 (100.0 to 100.0) | 100.0 (100.0 to 100.0) | 100.0 (100.0 to 100.0) | 100.0 (100.0 to 100.0) |
| k2: snap-mid's word edges, raw-mid's inner edges | 95.4 (92.3 to 98.0) | 96.8 (94.8 to 98.7) | 96.2 (94.3 to 97.7) | 95.6 (92.4 to 98.1) | 95.9 (92.9 to 98.4) | 94.2 (91.1 to 96.8) | 95.2 (93.5 to 96.7) |
| even inside k2 snap-mid's words | 87.5 (82.9 to 92.3) | 83.8 (78.5 to 88.5) | 85.6 (82.4 to 88.9) | 91.0 (85.9 to 95.2) | 86.6 (81.4 to 91.7) | 79.2 (74.0 to 83.9) | 84.6 (81.5 to 87.6) |
| littera-pilot-seg, `upright` | 63.4 (55.4 to 71.3) | 68.7 (60.6 to 76.7) | 66.2 (60.3 to 71.6) | 64.2 (52.9 to 74.0) | 69.2 (61.9 to 77.0) | 53.3 (46.4 to 60.3) | 62.2 (57.4 to 66.9) |
| littera-humboldt-ft2, `off` | 90.8 (86.7 to 94.4) | 92.4 (89.0 to 95.2) | 91.6 (88.9 to 94.0) | 95.6 (92.2 to 98.3) | 94.1 (91.2 to 97.0) | 91.1 (87.5 to 94.5) | 93.3 (91.3 to 95.2) |
| littera-humboldt-ft2, `upright` (deployed) | 91.0 (86.9 to 94.6) | 91.0 (86.7 to 94.5) | 91.0 (88.1 to 93.6) | 90.7 (84.7 to 95.5) | 91.8 (88.0 to 95.2) | 84.2 (77.9 to 89.3) | 88.7 (85.8 to 91.3) |
| ft2: `upright`'s word edges, `off`'s inner edges | 90.8 (86.7 to 94.4) | 92.4 (89.0 to 95.2) | 91.6 (88.9 to 94.0) | 95.6 (92.2 to 98.3) | 94.1 (91.2 to 97.0) | 91.1 (87.5 to 94.5) | 93.3 (91.3 to 95.2) |
| even inside ft2 `upright`'s words | 82.0 (76.3 to 87.8) | 82.2 (77.0 to 87.8) | 82.1 (78.4 to 86.0) | 86.4 (79.7 to 92.0) | 78.7 (71.1 to 85.8) | 75.0 (69.4 to 80.4) | 79.0 (74.9 to 82.8) |

Letter edges inside words that lie on an ink-free column, %:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 17.7 (16.3 to 19.1) | 17.1 (15.7 to 18.7) | 17.4 (16.5 to 18.5) | 15.0 (13.8 to 16.1) | 15.9 (14.6 to 17.1) | 21.2 (19.7 to 22.9) | 17.8 (16.9 to 18.7) |
| aligner-k2, raw-mid | 10.8 (9.5 to 12.1) | 11.0 (9.6 to 12.4) | 10.9 (9.9 to 11.8) | 7.4 (6.5 to 8.4) | 9.1 (8.0 to 10.2) | 9.8 (8.6 to 11.2) | 8.9 (8.3 to 9.7) |
| aligner-k2, snap-mid (deployed) | 8.4 (7.2 to 9.6) | 8.7 (7.5 to 10.1) | 8.6 (7.7 to 9.4) | 5.8 (5.0 to 6.8) | 7.8 (6.7 to 8.8) | 7.6 (6.4 to 8.8) | 7.2 (6.6 to 7.9) |
| k2: snap-mid's word edges, raw-mid's inner edges | 10.8 (9.5 to 12.1) | 11.0 (9.6 to 12.4) | 10.9 (9.9 to 11.8) | 7.4 (6.5 to 8.4) | 9.1 (8.0 to 10.2) | 9.8 (8.6 to 11.2) | 8.9 (8.3 to 9.7) |
| even inside k2 snap-mid's words | 8.9 (7.8 to 10.1) | 9.1 (7.8 to 10.5) | 9.0 (8.1 to 9.9) | 6.2 (5.4 to 7.1) | 6.5 (5.6 to 7.4) | 9.0 (7.9 to 10.2) | 7.4 (6.8 to 8.1) |
| littera-pilot-seg, `upright` | 8.1 (7.0 to 9.2) | 8.9 (7.8 to 10.1) | 8.5 (7.7 to 9.3) | 7.7 (6.7 to 8.7) | 7.7 (6.8 to 8.6) | 11.1 (9.7 to 12.5) | 9.0 (8.3 to 9.8) |
| littera-humboldt-ft2, `off` | 10.0 (8.5 to 11.4) | 9.2 (8.0 to 10.6) | 9.6 (8.6 to 10.6) | 7.0 (6.0 to 8.1) | 7.9 (6.9 to 8.9) | 9.2 (7.8 to 10.8) | 8.2 (7.5 to 8.9) |
| littera-humboldt-ft2, `upright` (deployed) | 8.2 (6.9 to 9.4) | 7.7 (6.7 to 8.9) | 8.0 (7.2 to 8.9) | 6.0 (5.2 to 7.0) | 6.6 (5.7 to 7.5) | 8.3 (7.0 to 9.7) | 7.1 (6.5 to 7.8) |
| ft2: `upright`'s word edges, `off`'s inner edges | 10.0 (8.5 to 11.4) | 9.2 (8.0 to 10.6) | 9.6 (8.6 to 10.6) | 7.0 (6.0 to 8.1) | 7.8 (6.9 to 8.9) | 9.2 (7.7 to 10.8) | 8.2 (7.4 to 8.9) |
| even inside ft2 `upright`'s words | 8.2 (7.0 to 9.4) | 8.1 (6.9 to 9.4) | 8.2 (7.3 to 9.1) | 5.9 (5.1 to 6.9) | 5.7 (4.9 to 6.6) | 9.3 (8.0 to 10.6) | 7.2 (6.6 to 7.9) |

Letters named right by the classifier, all, %:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 12.2 (11.3 to 12.9) | 11.8 (11.0 to 12.7) | 12.0 (11.5 to 12.6) | 15.2 (14.2 to 16.3) | 13.8 (12.9 to 14.7) | 15.1 (14.3 to 15.8) | 14.7 (14.2 to 15.2) |
| aligner-k2, raw-mid | 49.3 (46.9 to 51.7) | 46.6 (44.1 to 48.9) | 48.0 (46.4 to 49.8) | 58.5 (56.5 to 60.3) | 57.3 (55.5 to 59.1) | 55.3 (53.2 to 57.2) | 56.8 (55.7 to 58.0) |
| aligner-k2, snap-mid (deployed) | 39.1 (37.1 to 41.0) | 37.8 (35.7 to 39.7) | 38.5 (37.1 to 39.9) | 47.9 (46.0 to 49.8) | 48.3 (46.5 to 50.0) | 44.5 (42.5 to 46.4) | 46.7 (45.6 to 47.9) |
| k2: snap-mid's word edges, raw-mid's inner edges | 49.7 (47.3 to 52.1) | 46.3 (43.9 to 48.6) | 48.1 (46.4 to 49.7) | 57.6 (55.6 to 59.5) | 57.1 (55.3 to 59.0) | 55.4 (53.5 to 57.2) | 56.5 (55.4 to 57.7) |
| even inside k2 snap-mid's words | 29.8 (28.1 to 31.3) | 27.3 (25.7 to 29.0) | 28.6 (27.5 to 29.8) | 36.4 (34.7 to 38.0) | 36.2 (34.6 to 37.7) | 33.3 (31.7 to 34.9) | 35.1 (34.1 to 36.0) |
| littera-pilot-seg, `upright` | 24.9 (23.4 to 26.4) | 23.8 (22.2 to 25.4) | 24.4 (23.4 to 25.6) | 30.8 (28.8 to 32.6) | 30.5 (29.0 to 32.0) | 27.0 (25.5 to 28.5) | 29.2 (28.1 to 30.2) |
| littera-humboldt-ft2, `off` | 47.2 (44.7 to 49.5) | 44.9 (42.5 to 47.4) | 46.1 (44.5 to 47.8) | 53.9 (51.7 to 55.7) | 54.6 (52.9 to 56.3) | 52.5 (50.6 to 54.3) | 53.5 (52.4 to 54.6) |
| littera-humboldt-ft2, `upright` (deployed) | 38.9 (36.9 to 40.8) | 37.0 (34.8 to 39.1) | 38.0 (36.6 to 39.5) | 45.5 (43.6 to 47.2) | 45.7 (43.9 to 47.6) | 42.3 (40.4 to 44.0) | 44.3 (43.2 to 45.3) |
| ft2: `upright`'s word edges, `off`'s inner edges | 47.2 (44.7 to 49.7) | 45.2 (42.8 to 47.5) | 46.3 (44.7 to 48.0) | 54.0 (51.9 to 55.9) | 53.5 (51.9 to 55.1) | 52.0 (49.9 to 53.8) | 53.0 (51.9 to 54.2) |
| even inside ft2 `upright`'s words | 29.2 (27.6 to 30.9) | 27.2 (25.6 to 28.6) | 28.3 (27.2 to 29.4) | 35.1 (33.5 to 36.5) | 35.2 (33.7 to 36.6) | 32.4 (30.9 to 34.0) | 34.0 (33.2 to 35.0) |

Letters inside connected words, %:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 14.4 (12.7 to 16.1) | 14.6 (12.7 to 16.7) | 14.5 (13.3 to 15.7) | 16.7 (14.7 to 18.9) | 15.8 (14.1 to 17.4) | 18.3 (16.5 to 20.2) | 16.8 (15.7 to 17.9) |
| aligner-k2, raw-mid | 48.8 (45.1 to 52.4) | 48.5 (44.9 to 52.0) | 48.6 (46.1 to 51.2) | 60.1 (56.4 to 63.4) | 60.4 (57.3 to 63.2) | 56.4 (52.9 to 59.6) | 59.0 (57.0 to 60.8) |
| aligner-k2, snap-mid (deployed) | 38.3 (35.3 to 41.4) | 36.8 (33.9 to 39.8) | 37.6 (35.5 to 39.7) | 49.1 (45.7 to 52.8) | 48.7 (46.1 to 51.3) | 45.0 (42.1 to 47.8) | 47.6 (45.9 to 49.5) |
| k2: snap-mid's word edges, raw-mid's inner edges | 49.3 (45.6 to 52.9) | 48.4 (44.7 to 51.9) | 48.9 (46.4 to 51.3) | 60.2 (56.4 to 63.5) | 59.4 (56.6 to 62.1) | 56.1 (52.8 to 59.6) | 58.5 (56.6 to 60.3) |
| even inside k2 snap-mid's words | 26.2 (23.9 to 28.6) | 25.2 (22.9 to 27.9) | 25.8 (24.2 to 27.5) | 33.8 (30.6 to 37.1) | 33.3 (31.1 to 35.5) | 31.7 (29.3 to 34.2) | 32.9 (31.5 to 34.4) |
| littera-pilot-seg, `upright` | 24.6 (22.4 to 27.0) | 22.1 (19.7 to 24.4) | 23.4 (21.8 to 25.3) | 30.6 (27.2 to 33.9) | 31.1 (28.5 to 33.6) | 26.6 (23.9 to 29.6) | 29.5 (27.8 to 31.0) |
| littera-humboldt-ft2, `off` | 45.8 (42.5 to 49.1) | 44.4 (40.6 to 48.0) | 45.2 (42.7 to 47.6) | 54.4 (50.6 to 57.7) | 55.5 (53.1 to 58.0) | 51.0 (48.2 to 54.1) | 53.8 (52.0 to 55.4) |
| littera-humboldt-ft2, `upright` (deployed) | 37.3 (34.4 to 40.0) | 36.2 (32.9 to 39.3) | 36.8 (34.7 to 38.7) | 45.5 (42.4 to 48.7) | 47.0 (44.1 to 50.2) | 42.0 (39.1 to 45.1) | 45.0 (43.3 to 46.7) |
| ft2: `upright`'s word edges, `off`'s inner edges | 46.1 (42.8 to 49.4) | 44.6 (40.6 to 48.4) | 45.4 (42.8 to 47.9) | 55.1 (51.2 to 58.3) | 55.0 (52.5 to 57.4) | 51.9 (48.9 to 55.0) | 54.0 (52.3 to 55.7) |
| even inside ft2 `upright`'s words | 26.4 (24.0 to 28.7) | 24.6 (22.1 to 27.1) | 25.5 (23.8 to 27.4) | 34.5 (31.5 to 37.5) | 32.6 (30.5 to 34.8) | 31.8 (29.2 to 34.5) | 32.9 (31.5 to 34.3) |

Other letters, %:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 11.5 (10.7 to 12.4) | 11.2 (10.2 to 12.1) | 11.3 (10.8 to 12.0) | 14.8 (13.8 to 15.9) | 13.1 (12.1 to 14.1) | 14.4 (13.7 to 15.2) | 14.1 (13.6 to 14.6) |
| aligner-k2, raw-mid | 49.4 (46.8 to 52.0) | 46.1 (43.5 to 48.5) | 47.8 (46.1 to 49.6) | 58.0 (56.0 to 60.0) | 56.2 (54.4 to 58.1) | 55.1 (52.9 to 57.1) | 56.2 (55.1 to 57.4) |
| aligner-k2, snap-mid (deployed) | 39.3 (37.2 to 41.3) | 38.0 (35.7 to 40.2) | 38.7 (37.2 to 40.2) | 47.5 (45.5 to 49.5) | 48.1 (46.2 to 50.2) | 44.4 (42.3 to 46.5) | 46.4 (45.2 to 47.6) |
| k2: snap-mid's word edges, raw-mid's inner edges | 49.8 (47.3 to 52.3) | 45.8 (43.2 to 48.3) | 47.9 (46.1 to 49.6) | 56.9 (54.8 to 58.9) | 56.4 (54.5 to 58.5) | 55.2 (53.2 to 57.1) | 56.0 (54.9 to 57.2) |
| even inside k2 snap-mid's words | 30.7 (28.9 to 32.5) | 27.8 (26.1 to 29.6) | 29.3 (28.1 to 30.6) | 37.1 (35.2 to 38.8) | 37.1 (35.4 to 38.9) | 33.6 (31.9 to 35.3) | 35.6 (34.7 to 36.7) |
| littera-pilot-seg, `upright` | 25.0 (23.4 to 26.8) | 24.2 (22.6 to 26.0) | 24.7 (23.5 to 25.9) | 30.9 (28.9 to 32.7) | 30.3 (28.6 to 31.9) | 27.1 (25.5 to 28.7) | 29.1 (28.0 to 30.2) |
| littera-humboldt-ft2, `off` | 47.5 (44.8 to 50.0) | 45.1 (42.6 to 47.7) | 46.4 (44.7 to 48.1) | 53.7 (51.6 to 55.6) | 54.2 (52.5 to 56.1) | 52.8 (50.7 to 54.7) | 53.5 (52.4 to 54.7) |
| littera-humboldt-ft2, `upright` (deployed) | 39.3 (37.1 to 41.5) | 37.2 (34.9 to 39.5) | 38.3 (36.7 to 39.9) | 45.4 (43.5 to 47.5) | 45.2 (43.3 to 47.3) | 42.3 (40.3 to 44.0) | 44.1 (43.0 to 45.2) |
| ft2: `upright`'s word edges, `off`'s inner edges | 47.5 (44.9 to 50.1) | 45.3 (42.8 to 47.7) | 46.5 (44.9 to 48.2) | 53.7 (51.5 to 55.8) | 53.0 (51.2 to 54.9) | 52.0 (49.8 to 53.9) | 52.8 (51.6 to 54.0) |
| even inside ft2 `upright`'s words | 30.0 (28.2 to 31.9) | 27.9 (26.2 to 29.5) | 29.0 (27.8 to 30.3) | 35.2 (33.5 to 36.8) | 36.0 (34.4 to 37.6) | 32.5 (30.8 to 34.2) | 34.4 (33.4 to 35.4) |

All letters, difference to `k2-snap-mid` in points (paired):

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | -27.0 (-28.8 to -25.1) | -25.9 (-27.6 to -24.1) | -26.5 (-27.8 to -25.2) | -32.7 (-34.5 to -30.8) | -34.5 (-36.1 to -32.9) | -29.4 (-31.5 to -27.4) | -32.0 (-33.2 to -31.0) |
| aligner-k2, raw-mid | 10.1 (8.7 to 11.5) | 8.8 (7.5 to 10.2) | 9.5 (8.6 to 10.5) | 10.6 (9.3 to 11.9) | 9.0 (7.8 to 10.1) | 10.8 (9.6 to 11.8) | 10.1 (9.4 to 10.8) |
| aligner-k2, snap-mid (deployed) |  |  |  |  |  |  |  |
| k2: snap-mid's word edges, raw-mid's inner edges | 10.6 (9.2 to 11.8) | 8.5 (7.2 to 9.9) | 9.6 (8.7 to 10.5) | 9.7 (8.5 to 11.0) | 8.9 (7.9 to 9.9) | 10.9 (9.8 to 11.8) | 9.9 (9.2 to 10.5) |
| even inside k2 snap-mid's words | -9.4 (-10.7 to -8.0) | -10.5 (-11.8 to -9.1) | -9.9 (-10.8 to -9.0) | -11.5 (-12.8 to -10.1) | -12.1 (-13.4 to -10.9) | -11.2 (-12.5 to -9.9) | -11.6 (-12.4 to -10.9) |
| littera-pilot-seg, `upright` | -14.2 (-15.6 to -12.7) | -13.9 (-15.2 to -12.6) | -14.1 (-15.1 to -13.0) | -17.1 (-18.8 to -15.2) | -17.8 (-19.3 to -16.4) | -17.5 (-19.1 to -15.9) | -17.5 (-18.4 to -16.5) |
| littera-humboldt-ft2, `off` | 8.1 (6.4 to 9.6) | 7.2 (5.6 to 8.9) | 7.6 (6.5 to 8.8) | 6.0 (4.5 to 7.5) | 6.3 (5.0 to 7.6) | 8.0 (6.6 to 9.3) | 6.9 (6.1 to 7.6) |
| littera-humboldt-ft2, `upright` (deployed) | -0.3 (-1.4 to 0.8) | -0.8 (-2.2 to 0.6) | -0.5 (-1.4 to 0.4) | -2.4 (-3.9 to -1.0) | -2.6 (-3.8 to -1.3) | -2.2 (-3.5 to -1.1) | -2.4 (-3.1 to -1.6) |
| ft2: `upright`'s word edges, `off`'s inner edges | 8.1 (6.5 to 9.7) | 7.4 (5.8 to 9.0) | 7.8 (6.7 to 8.9) | 6.2 (4.5 to 7.7) | 5.2 (4.0 to 6.5) | 7.5 (6.1 to 8.7) | 6.4 (5.6 to 7.1) |
| even inside ft2 `upright`'s words | -9.9 (-11.2 to -8.5) | -10.6 (-12.0 to -9.1) | -10.2 (-11.2 to -9.2) | -12.8 (-14.1 to -11.6) | -13.1 (-14.2 to -11.9) | -12.1 (-13.3 to -10.7) | -12.6 (-13.4 to -11.8) |

Letters inside connected words, difference to `k2-snap-mid` in points (paired):

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | -23.9 (-26.7 to -21.2) | -22.2 (-25.3 to -18.7) | -23.1 (-25.2 to -21.0) | -32.5 (-36.0 to -28.9) | -32.9 (-35.8 to -30.2) | -26.8 (-29.8 to -23.8) | -30.8 (-32.7 to -28.9) |
| aligner-k2, raw-mid | 10.4 (7.8 to 13.0) | 11.7 (8.6 to 15.1) | 11.0 (9.0 to 13.0) | 11.0 (8.0 to 13.9) | 11.7 (9.4 to 13.8) | 11.3 (8.8 to 13.7) | 11.4 (9.9 to 12.8) |
| aligner-k2, snap-mid (deployed) |  |  |  |  |  |  |  |
| k2: snap-mid's word edges, raw-mid's inner edges | 11.0 (8.3 to 13.6) | 11.5 (8.6 to 14.9) | 11.2 (9.0 to 13.3) | 11.1 (8.3 to 13.8) | 10.7 (8.7 to 12.7) | 11.1 (8.4 to 13.5) | 10.9 (9.5 to 12.2) |
| even inside k2 snap-mid's words | -12.1 (-14.9 to -9.3) | -11.6 (-14.2 to -8.7) | -11.9 (-13.7 to -9.9) | -15.3 (-18.5 to -12.4) | -15.4 (-17.9 to -12.9) | -13.3 (-16.1 to -10.5) | -14.7 (-16.3 to -13.1) |
| littera-pilot-seg, `upright` | -13.7 (-16.5 to -10.7) | -14.8 (-17.6 to -11.8) | -14.2 (-16.4 to -11.8) | -18.6 (-21.9 to -15.3) | -17.6 (-20.2 to -14.8) | -18.4 (-21.3 to -15.4) | -18.1 (-19.8 to -16.3) |
| littera-humboldt-ft2, `off` | 7.5 (4.4 to 10.4) | 7.5 (4.3 to 11.1) | 7.5 (5.2 to 9.7) | 5.2 (2.3 to 8.2) | 6.9 (4.4 to 9.2) | 6.0 (3.0 to 8.8) | 6.1 (4.6 to 7.7) |
| littera-humboldt-ft2, `upright` (deployed) | -1.0 (-3.5 to 1.5) | -0.7 (-3.4 to 2.1) | -0.9 (-2.8 to 0.9) | -3.6 (-6.9 to -0.2) | -1.7 (-4.2 to 0.9) | -3.1 (-6.2 to 0.0) | -2.6 (-4.3 to -0.9) |
| ft2: `upright`'s word edges, `off`'s inner edges | 7.8 (4.6 to 10.9) | 7.8 (4.3 to 11.3) | 7.8 (5.5 to 10.0) | 6.0 (3.1 to 9.0) | 6.4 (4.0 to 8.5) | 6.8 (4.0 to 9.6) | 6.4 (4.8 to 7.9) |
| even inside ft2 `upright`'s words | -11.9 (-14.9 to -9.0) | -12.3 (-14.9 to -9.6) | -12.1 (-14.2 to -10.2) | -14.7 (-17.3 to -12.1) | -16.1 (-18.4 to -13.5) | -13.2 (-16.2 to -10.2) | -14.8 (-16.5 to -13.2) |

Control, letters named from the span's width alone, %:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 12.9 (12.1 to 13.7) | 12.4 (11.5 to 13.2) | 12.7 (12.1 to 13.2) | 14.2 (13.4 to 15.0) | 12.8 (12.1 to 13.6) | 15.5 (14.9 to 16.1) | 14.2 (13.8 to 14.7) |
| aligner-k2, raw-mid | 12.6 (11.8 to 13.4) | 12.7 (11.7 to 13.7) | 12.6 (12.0 to 13.2) | 15.5 (14.6 to 16.4) | 13.9 (13.1 to 14.8) | 16.3 (15.6 to 17.0) | 15.3 (14.8 to 15.7) |
| aligner-k2, snap-mid (deployed) | 13.0 (12.2 to 13.9) | 12.9 (12.0 to 13.9) | 13.0 (12.4 to 13.6) | 15.6 (14.8 to 16.5) | 13.7 (12.9 to 14.6) | 16.3 (15.5 to 17.0) | 15.2 (14.8 to 15.7) |
| k2: snap-mid's word edges, raw-mid's inner edges | 13.1 (12.3 to 14.0) | 13.3 (12.4 to 14.2) | 13.2 (12.6 to 13.8) | 15.2 (14.3 to 16.2) | 13.9 (13.2 to 14.8) | 16.4 (15.7 to 17.2) | 15.3 (14.8 to 15.8) |
| even inside k2 snap-mid's words | 12.7 (11.9 to 13.5) | 12.4 (11.5 to 13.2) | 12.5 (11.9 to 13.1) | 14.3 (13.6 to 15.2) | 13.3 (12.5 to 14.0) | 15.8 (15.1 to 16.3) | 14.6 (14.1 to 15.0) |
| littera-pilot-seg, `upright` | 13.4 (12.7 to 14.2) | 12.5 (11.6 to 13.4) | 13.0 (12.4 to 13.5) | 14.9 (14.1 to 15.8) | 13.2 (12.3 to 14.1) | 16.4 (15.7 to 17.1) | 14.9 (14.4 to 15.3) |
| littera-humboldt-ft2, `off` | 15.5 (14.6 to 16.5) | 15.5 (14.3 to 16.5) | 15.5 (14.8 to 16.3) | 18.3 (17.4 to 19.2) | 16.4 (15.4 to 17.4) | 18.4 (17.6 to 19.2) | 17.7 (17.2 to 18.2) |
| littera-humboldt-ft2, `upright` (deployed) | 15.0 (14.1 to 15.9) | 14.5 (13.4 to 15.5) | 14.8 (14.1 to 15.5) | 17.8 (16.9 to 18.8) | 16.0 (15.0 to 17.0) | 18.1 (17.3 to 19.0) | 17.3 (16.8 to 17.8) |
| ft2: `upright`'s word edges, `off`'s inner edges | 14.3 (13.4 to 15.2) | 14.3 (13.2 to 15.2) | 14.3 (13.6 to 14.9) | 16.6 (15.8 to 17.6) | 15.2 (14.3 to 16.2) | 17.3 (16.6 to 18.1) | 16.4 (15.9 to 16.9) |
| even inside ft2 `upright`'s words | 12.6 (11.9 to 13.4) | 12.2 (11.2 to 13.1) | 12.4 (11.8 to 13.0) | 14.3 (13.4 to 15.1) | 13.4 (12.6 to 14.2) | 15.6 (15.0 to 16.3) | 14.5 (14.1 to 14.9) |

Control, always the commonest letter, %:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 12.9 (12.1 to 13.7) | 12.4 (11.5 to 13.2) | 12.7 (12.1 to 13.2) | 14.2 (13.4 to 15.0) | 12.8 (12.1 to 13.6) | 15.5 (14.9 to 16.1) | 14.2 (13.8 to 14.7) |
| aligner-k2, raw-mid | 12.9 (12.1 to 13.7) | 12.4 (11.5 to 13.2) | 12.7 (12.1 to 13.2) | 14.2 (13.4 to 15.0) | 12.8 (12.1 to 13.6) | 15.5 (14.9 to 16.1) | 14.2 (13.8 to 14.7) |
| aligner-k2, snap-mid (deployed) | 12.9 (12.1 to 13.7) | 12.4 (11.5 to 13.2) | 12.7 (12.1 to 13.2) | 14.2 (13.4 to 15.0) | 12.8 (12.1 to 13.6) | 15.5 (14.9 to 16.1) | 14.2 (13.8 to 14.7) |
| k2: snap-mid's word edges, raw-mid's inner edges | 12.9 (12.1 to 13.7) | 12.4 (11.5 to 13.2) | 12.7 (12.1 to 13.2) | 14.2 (13.4 to 15.0) | 12.8 (12.1 to 13.6) | 15.5 (14.9 to 16.1) | 14.2 (13.8 to 14.7) |
| even inside k2 snap-mid's words | 12.9 (12.1 to 13.7) | 12.4 (11.5 to 13.2) | 12.7 (12.1 to 13.2) | 14.2 (13.4 to 15.0) | 12.8 (12.1 to 13.6) | 15.5 (14.9 to 16.1) | 14.2 (13.8 to 14.7) |
| littera-pilot-seg, `upright` | 12.9 (12.2 to 13.7) | 12.4 (11.4 to 13.3) | 12.7 (12.1 to 13.3) | 14.2 (13.4 to 15.0) | 12.8 (12.0 to 13.7) | 15.5 (14.9 to 16.2) | 14.2 (13.8 to 14.7) |
| littera-humboldt-ft2, `off` | 12.9 (12.2 to 13.7) | 12.4 (11.4 to 13.3) | 12.7 (12.1 to 13.3) | 14.2 (13.4 to 15.0) | 12.8 (12.0 to 13.7) | 15.5 (14.9 to 16.2) | 14.2 (13.8 to 14.7) |
| littera-humboldt-ft2, `upright` (deployed) | 12.9 (12.2 to 13.7) | 12.4 (11.4 to 13.3) | 12.7 (12.1 to 13.3) | 14.2 (13.4 to 15.0) | 12.8 (12.0 to 13.7) | 15.5 (14.9 to 16.2) | 14.2 (13.8 to 14.7) |
| ft2: `upright`'s word edges, `off`'s inner edges | 12.9 (12.2 to 13.7) | 12.4 (11.4 to 13.3) | 12.7 (12.1 to 13.3) | 14.2 (13.4 to 15.0) | 12.8 (12.0 to 13.7) | 15.5 (14.9 to 16.2) | 14.2 (13.8 to 14.7) |
| even inside ft2 `upright`'s words | 12.9 (12.2 to 13.7) | 12.4 (11.4 to 13.3) | 12.7 (12.1 to 13.3) | 14.2 (13.4 to 15.0) | 12.8 (12.0 to 13.7) | 15.5 (14.9 to 16.2) | 14.2 (13.8 to 14.7) |

Occlusion: tokens whose evidence centre lies inside their span, %:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 45.2 (41.9 to 48.8) | 44.3 (41.0 to 47.5) | 44.8 (42.5 to 47.3) | 54.4 (51.8 to 57.1) | 45.7 (43.0 to 48.5) | 46.2 (43.2 to 49.0) | 48.2 (46.5 to 49.7) |
| aligner-k2, raw-mid | 75.0 (72.1 to 77.6) | 74.4 (72.0 to 76.6) | 74.7 (72.7 to 76.4) | 78.1 (76.0 to 80.0) | 74.5 (72.5 to 76.5) | 75.3 (73.6 to 76.9) | 75.7 (74.6 to 76.7) |
| aligner-k2, snap-mid (deployed) | 72.0 (69.3 to 74.5) | 72.3 (70.1 to 74.4) | 72.2 (70.2 to 73.8) | 74.4 (72.5 to 76.2) | 71.6 (69.7 to 73.4) | 71.8 (70.2 to 73.5) | 72.4 (71.4 to 73.4) |
| k2: snap-mid's word edges, raw-mid's inner edges | 74.2 (71.5 to 76.8) | 74.2 (72.0 to 76.4) | 74.2 (72.3 to 76.0) | 77.7 (75.7 to 79.6) | 73.7 (71.7 to 75.6) | 74.8 (73.1 to 76.4) | 75.2 (74.1 to 76.2) |
| even inside k2 snap-mid's words | 71.0 (68.3 to 73.5) | 70.4 (67.9 to 72.6) | 70.7 (68.6 to 72.3) | 73.0 (71.2 to 74.9) | 69.0 (67.1 to 71.0) | 70.4 (68.7 to 71.9) | 70.6 (69.5 to 71.6) |
| littera-pilot-seg, `upright` | 60.3 (57.3 to 63.1) | 59.3 (55.9 to 62.4) | 59.8 (57.7 to 61.9) | 61.6 (58.8 to 64.2) | 61.5 (59.1 to 63.8) | 59.3 (56.7 to 61.6) | 60.7 (59.2 to 62.1) |
| littera-humboldt-ft2, `off` | 73.8 (71.1 to 76.2) | 73.4 (71.2 to 75.6) | 73.6 (72.0 to 75.3) | 76.2 (74.4 to 78.0) | 74.0 (72.0 to 75.8) | 73.6 (72.0 to 75.1) | 74.4 (73.4 to 75.4) |
| littera-humboldt-ft2, `upright` (deployed) | 73.5 (70.7 to 76.0) | 72.8 (70.3 to 75.0) | 73.2 (71.5 to 74.8) | 74.7 (72.9 to 76.3) | 72.8 (70.9 to 74.8) | 73.3 (71.6 to 75.0) | 73.5 (72.4 to 74.6) |
| ft2: `upright`'s word edges, `off`'s inner edges | 74.3 (71.6 to 76.7) | 73.5 (71.3 to 75.7) | 73.9 (72.1 to 75.6) | 76.5 (74.8 to 78.2) | 74.0 (72.0 to 75.8) | 74.7 (73.0 to 76.2) | 74.9 (73.9 to 75.9) |
| even inside ft2 `upright`'s words | 70.7 (68.1 to 73.0) | 69.7 (67.3 to 72.0) | 70.2 (68.5 to 71.9) | 71.8 (69.9 to 73.6) | 68.9 (67.0 to 70.6) | 70.0 (68.2 to 71.5) | 70.0 (69.0 to 71.1) |

Occlusion: mean distance of the evidence centre from the middle of the span, character widths:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 2.32 (1.89 to 2.82) | 2.41 (1.91 to 2.97) | 2.36 (2.01 to 2.76) | 1.49 (1.24 to 1.84) | 1.75 (1.56 to 1.97) | 1.83 (1.66 to 2.04) | 1.72 (1.59 to 1.85) |
| aligner-k2, raw-mid | 1.38 (0.94 to 2.05) | 1.19 (0.91 to 1.54) | 1.29 (1.01 to 1.71) | 0.90 (0.70 to 1.17) | 0.94 (0.81 to 1.10) | 0.96 (0.84 to 1.08) | 0.93 (0.85 to 1.04) |
| aligner-k2, snap-mid (deployed) | 1.39 (0.94 to 2.06) | 1.20 (0.91 to 1.54) | 1.29 (1.01 to 1.71) | 0.92 (0.72 to 1.18) | 0.96 (0.84 to 1.12) | 0.97 (0.85 to 1.10) | 0.95 (0.87 to 1.05) |
| k2: snap-mid's word edges, raw-mid's inner edges | 1.37 (0.93 to 2.04) | 1.18 (0.90 to 1.53) | 1.28 (0.99 to 1.70) | 0.89 (0.70 to 1.16) | 0.94 (0.82 to 1.10) | 0.94 (0.83 to 1.07) | 0.93 (0.84 to 1.03) |
| even inside k2 snap-mid's words | 1.41 (0.97 to 2.08) | 1.23 (0.95 to 1.57) | 1.32 (1.03 to 1.74) | 0.94 (0.75 to 1.21) | 1.00 (0.88 to 1.17) | 1.01 (0.90 to 1.14) | 0.99 (0.91 to 1.09) |
| littera-pilot-seg, `upright` | 1.68 (1.29 to 2.20) | 1.69 (1.32 to 2.15) | 1.68 (1.40 to 2.01) | 1.24 (0.97 to 1.60) | 1.17 (1.01 to 1.37) | 1.36 (1.18 to 1.54) | 1.26 (1.14 to 1.40) |
| littera-humboldt-ft2, `off` | 1.30 (0.92 to 1.89) | 1.15 (0.88 to 1.49) | 1.23 (0.96 to 1.55) | 0.89 (0.69 to 1.16) | 0.91 (0.79 to 1.08) | 0.94 (0.82 to 1.08) | 0.91 (0.82 to 1.01) |
| littera-humboldt-ft2, `upright` (deployed) | 1.34 (0.96 to 1.92) | 1.17 (0.91 to 1.52) | 1.26 (0.99 to 1.58) | 0.92 (0.72 to 1.18) | 0.95 (0.82 to 1.13) | 0.99 (0.87 to 1.13) | 0.96 (0.86 to 1.05) |
| ft2: `upright`'s word edges, `off`'s inner edges | 1.32 (0.94 to 1.91) | 1.16 (0.90 to 1.51) | 1.24 (0.97 to 1.56) | 0.90 (0.70 to 1.16) | 0.92 (0.80 to 1.10) | 0.96 (0.84 to 1.10) | 0.93 (0.84 to 1.03) |
| even inside ft2 `upright`'s words | 1.36 (0.99 to 1.94) | 1.21 (0.95 to 1.56) | 1.29 (1.02 to 1.61) | 0.95 (0.76 to 1.21) | 0.99 (0.86 to 1.16) | 1.04 (0.92 to 1.18) | 1.00 (0.91 to 1.09) |

Occlusion: mean signed distance (positive: evidence right of the span's middle), character widths:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | -0.57 (-0.90 to -0.27) | -0.09 (-0.51 to 0.37) | -0.34 (-0.57 to -0.07) | -0.07 (-0.24 to 0.10) | -0.55 (-0.78 to -0.32) | -0.65 (-0.86 to -0.45) | -0.46 (-0.59 to -0.34) |
| aligner-k2, raw-mid | -0.00 (-0.40 to 0.61) | -0.57 (-0.88 to -0.34) | -0.28 (-0.53 to 0.11) | -0.29 (-0.51 to -0.04) | -0.40 (-0.56 to -0.27) | -0.20 (-0.30 to -0.10) | -0.30 (-0.38 to -0.21) |
| aligner-k2, snap-mid (deployed) | 0.18 (-0.22 to 0.79) | -0.41 (-0.72 to -0.17) | -0.10 (-0.36 to 0.28) | -0.13 (-0.34 to 0.10) | -0.24 (-0.40 to -0.12) | 0.01 (-0.09 to 0.11) | -0.12 (-0.20 to -0.04) |
| k2: snap-mid's word edges, raw-mid's inner edges | 0.08 (-0.32 to 0.69) | -0.50 (-0.81 to -0.27) | -0.20 (-0.46 to 0.19) | -0.22 (-0.44 to 0.02) | -0.34 (-0.50 to -0.21) | -0.11 (-0.21 to -0.01) | -0.22 (-0.30 to -0.14) |
| even inside k2 snap-mid's words | 0.13 (-0.27 to 0.74) | -0.44 (-0.75 to -0.20) | -0.14 (-0.40 to 0.24) | -0.19 (-0.41 to 0.05) | -0.30 (-0.46 to -0.18) | -0.12 (-0.21 to -0.01) | -0.20 (-0.29 to -0.12) |
| littera-pilot-seg, `upright` | -0.36 (-0.60 to -0.16) | -0.27 (-0.59 to 0.04) | -0.32 (-0.54 to -0.12) | -0.16 (-0.34 to 0.02) | -0.25 (-0.43 to -0.07) | -0.16 (-0.31 to -0.01) | -0.19 (-0.28 to -0.09) |
| littera-humboldt-ft2, `off` | -0.08 (-0.30 to 0.15) | -0.52 (-0.84 to -0.27) | -0.29 (-0.48 to -0.11) | -0.10 (-0.31 to 0.16) | -0.30 (-0.46 to -0.19) | -0.02 (-0.13 to 0.09) | -0.14 (-0.23 to -0.05) |
| littera-humboldt-ft2, `upright` (deployed) | -0.09 (-0.31 to 0.14) | -0.51 (-0.84 to -0.25) | -0.29 (-0.48 to -0.11) | -0.12 (-0.33 to 0.14) | -0.30 (-0.45 to -0.19) | -0.05 (-0.16 to 0.05) | -0.16 (-0.24 to -0.07) |
| ft2: `upright`'s word edges, `off`'s inner edges | -0.09 (-0.31 to 0.14) | -0.52 (-0.85 to -0.26) | -0.29 (-0.49 to -0.11) | -0.11 (-0.32 to 0.15) | -0.30 (-0.45 to -0.19) | -0.04 (-0.14 to 0.07) | -0.15 (-0.23 to -0.06) |
| even inside ft2 `upright`'s words | -0.07 (-0.28 to 0.16) | -0.45 (-0.79 to -0.20) | -0.25 (-0.45 to -0.07) | -0.09 (-0.31 to 0.16) | -0.27 (-0.43 to -0.16) | -0.05 (-0.16 to 0.06) | -0.14 (-0.22 to -0.05) |

Occlusion: tokens with evidence, % of tokens:

| placer | val | test_page | dev | test_bench | test_journal | letters | report |
|---|---:|---:|---:|---:|---:|---:|---:|
| even spacing, whole line | 94.1 (93.0 to 95.0) | 93.7 (92.4 to 95.0) | 93.9 (93.0 to 94.7) | 96.7 (95.9 to 97.5) | 95.3 (94.4 to 96.1) | 95.0 (94.1 to 95.7) | 95.5 (95.0 to 96.0) |
| aligner-k2, raw-mid | 94.1 (93.0 to 95.0) | 93.7 (92.4 to 95.0) | 93.9 (93.0 to 94.7) | 96.7 (95.9 to 97.5) | 95.3 (94.4 to 96.1) | 95.0 (94.1 to 95.7) | 95.5 (95.0 to 96.0) |
| aligner-k2, snap-mid (deployed) | 94.1 (93.0 to 95.0) | 93.7 (92.4 to 95.0) | 93.9 (93.0 to 94.7) | 96.7 (95.9 to 97.5) | 95.3 (94.4 to 96.1) | 95.0 (94.1 to 95.7) | 95.5 (95.0 to 96.0) |
| k2: snap-mid's word edges, raw-mid's inner edges | 94.1 (93.0 to 95.0) | 93.7 (92.4 to 95.0) | 93.9 (93.0 to 94.7) | 96.7 (95.9 to 97.5) | 95.3 (94.4 to 96.1) | 95.0 (94.1 to 95.7) | 95.5 (95.0 to 96.0) |
| even inside k2 snap-mid's words | 94.1 (93.0 to 95.0) | 93.7 (92.4 to 95.0) | 93.9 (93.0 to 94.7) | 96.7 (95.9 to 97.5) | 95.3 (94.4 to 96.1) | 95.0 (94.1 to 95.7) | 95.5 (95.0 to 96.0) |
| littera-pilot-seg, `upright` | 94.1 (92.9 to 95.2) | 93.7 (92.3 to 95.1) | 93.9 (93.0 to 94.8) | 96.7 (95.9 to 97.5) | 95.3 (94.4 to 96.0) | 95.0 (94.2 to 95.8) | 95.5 (95.0 to 96.0) |
| littera-humboldt-ft2, `off` | 94.1 (92.9 to 95.2) | 93.7 (92.3 to 95.1) | 93.9 (93.0 to 94.8) | 96.7 (95.9 to 97.5) | 95.3 (94.4 to 96.0) | 95.0 (94.2 to 95.8) | 95.5 (95.0 to 96.0) |
| littera-humboldt-ft2, `upright` (deployed) | 94.1 (92.9 to 95.2) | 93.7 (92.3 to 95.1) | 93.9 (93.0 to 94.8) | 96.7 (95.9 to 97.5) | 95.3 (94.4 to 96.0) | 95.0 (94.2 to 95.8) | 95.5 (95.0 to 96.0) |
| ft2: `upright`'s word edges, `off`'s inner edges | 94.1 (92.9 to 95.2) | 93.7 (92.3 to 95.1) | 93.9 (93.0 to 94.8) | 96.7 (95.9 to 97.5) | 95.3 (94.4 to 96.0) | 95.0 (94.2 to 95.8) | 95.5 (95.0 to 96.0) |
| even inside ft2 `upright`'s words | 94.1 (92.9 to 95.2) | 93.7 (92.3 to 95.1) | 93.9 (93.0 to 94.8) | 96.7 (95.9 to 97.5) | 95.3 (94.4 to 96.0) | 95.0 (94.2 to 95.8) | 95.5 (95.0 to 96.0) |

## Contact sheets for the blind check

`modal/humboldt.py::sheets --a <placer> --b <placer>` writes `results/r3/sheets/<a>_vs_<b>/sheet_NN.png` (12 tiles
each), `key.json` and an empty `judged.txt` (an existing `judged.txt` is kept). Candidates are sequences of two and
three letters inside one word (letters only) where the two placers' spans differ by more than half a letter at
either end; a letter is the median width of the placed characters of both placers in that line. Lines where either
placer's line fit is below 0.2 are left out. At most two sequences per line, 60 sequences drawn with seed 5, each
shown once per placer, all tiles shuffled. A tile shows the crop from three letters left of the earlier start to
three letters right of the later end, the claimed span in full contrast between red ticks and the rest faded,
labelled with a tile id and the letter sequence only. Two sets are built. I have not judged them:

- `k2-snap-mid_vs_ft2-upright`: 1,485 candidate sequences on the 1,000 lines, 120 tiles on 10 sheets.
- `ft2-upright_vs_mix_ft2-off+ft2-upright`: 980 candidates, 120 tiles. This pair differs only inside words (the
  snapped word edges are the same), so it shows what the stretching of the inner edges does.

## Entry points

```
uv run modal run modal/humboldt.py::place --model /vol/models/<model> --name <placer> --options '<json>' [--halves dev]
uv run modal run modal/humboldt.py::score --names <a,b,c> [--halves dev] [--out results/r3/<file>.json]
uv run modal run modal/humboldt.py::classify --names <a,b,c> [--halves dev] [--out results/r3/<file>.json] [--reuse]
uv run modal run modal/humboldt.py::widths --names <a,b,c> [--halves dev]
uv run modal run modal/humboldt.py::occlusion --names <a,b,c> [--halves dev]
uv run modal run modal/humboldt.py::sheets --a <placer> --b <placer> [--halves dev]
uv run modal run modal/humboldt.py::peek --uids <uid,uid> --names <a,b> --out <folder>
```

`place` writes `/vol/eval_runs/humboldt-1000/<placer>.jsonl` (and `<placer>.meta.json`). `--options` is JSON: `snap`
and any other keyword of `Aligner.load`; any field of `GraphOptions`, at the top level or under `decode`, is merged
into the model's decode options (a field `GraphOptions` does not have raises); `objective` and `geometry` are dicts
merged into the model's. The local `littera` package is uploaded with every run, so new fields in `graph.py` and
`place.py` work without a change here. `score`, `classify`, `widths` and `occlusion` default to the dev half;
`--halves dev,report` adds the report half. `score` and `classify` need the references `k2-snap-mid` and
`ft2-upright` on the volume (`--refs`). `even@<placer>` and `mix:<inner>+<outer>` can be given wherever a placer name
is and need no file. `classify` trains anew for every placer and half (about 4 minutes for a dev half, 8 for a
report half, 4 CPU cores each, up to 10 at once); `--reuse` takes the per-letter results already on the volume, which
is only right while the placement file of that name has not changed. `occlusion` only scores; the evidence is on the
volume (`occlusion/evidence.json`).

## Spend and open points

Spend: `modal billing report --for today` shows 1.09 USD for the app `littera-humboldt` at 21:22 (38 runs), of a
budget of 4 USD. That figure includes agent T's calls of `place` and `score`, which run under the same app name (at
least five runs that I did not start, about 0.15 USD; the report may still lag by a few cents). The larger items: the classifier for eight placers and both
halves 0.46, for the two `mix:` placers 0.13, the occlusion evidence (four shards of 250 lines on A10G, 75 s each)
0.10, the two data builds 0.07, three placements of 1,000 lines 0.11.

What did not work or was not done:

- The occlusion job waited ten minutes for an L4 that did not come; it now takes the first of L4, A10G, T4.
- `score` results made before 21:08 used Otsu's ink threshold and the result file has no `ink_sensitivity` among its
  thresholds. Results before and after are not comparable; rerun `score` for anything scored earlier.
- The first pool and evaluation set (20:43) had no baselines; both were rebuilt at 20:47 (`docs/r3/DATA.md`).
- aligner-k2 was not placed again: its cuts are the October ones from `place-k2.jsonl`. littera with aligner-k2's
  snapping (a row of `docs/HUMBOLDT.md`) was not rebuilt.
- Nobody has judged the contact sheets yet.
- No unit tests were added (`tests/` belongs to agent T this round). Two checks were run by hand: the copy of
  `crop_fragments` gives the same pixels as humboldt-htr's on four lines, and the measures give the expected values
  on a drawn line with one lift and one word gap.
- The classifier was run with two seeds and two folds. The run-to-run noise of one placer was not measured apart from
  the bootstrap over lines; `mix:` and unsnapped placements, which share their inner edges, differ by 0.1 to 0.5 points.
