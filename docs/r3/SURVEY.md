# Round 3, agent S: survey on placing a known text at letter level

Web research, 2026-11-17. A bracketed number is a source opened in this session; "(nc)" means the source could not be
opened. [1], [8], [9], [18] were read as PDF text, the others through an automated page reader: re-check a figure or
quotation against the page before it goes on a slide. Speech results are named as such.

## 1. Methods and their localisation evidence

| family | how a letter gets its span | evidence |
|---|---|---|
| CTC forced alignment | a spike of 1 to 2 frames among blanks; the span is a convention added afterwards | IAM, about 10 frames per character: "the predictions are localized and correspond to 1 or 2 timesteps", an effect of the blank with CTC training, and easier for RNNs because of their unlimited context [1]; formal analysis in [2] (abstract). Speech: CTC "tends to delay token predictions", and phones last 21 ms in its alignment against 82 ms in the hand labels [3]. |
| kraken | trellis from the torchaudio wav2vec2 tutorial: a character runs from its first frame to the next character's first frame, so blanks after a spike go to the letter before [4] | The documentation speaks of "approximate character locations" and of word boxes as the use [5]; no accuracy figure. |
| CTC with label priors | posterior divided by prior^α in training and search, so fewer blanks | Speech, phone / word boundary error in ms: Buckeye 44/58 to 38/43, TIMIT 32/42 to 28/29; the HMM aligner MFA has 30/41 and 17/23. α = 0.3; above it the model "does not converge" [3]. |
| HMM hybrid, segmental | every frame belongs to a letter state or to a gap or silence state | Speech, hand-labelled TIMIT, word boundaries within 25 ms: MFA (GMM-HMM) 72.8 %, WhisperX 52.7 %, MMS (CTC) 43.5 % [7]. Against a GMM reference (no hand labels), a framewise-trained hybrid has 41 ms time-stamp error, models trained by full sum 52 to 106; an alignment without silence shows "a large shift" [8]. One-writer manuscript, word alignment error: 7.20 % with 78 character HMMs, 25.98 % with one model for all characters and one for the blank [9]. Segmental RNNs with latent boundaries beat CTC on handwriting recognition [10] (abstract). Oriflamms aligned medieval charters to character level with manual correction; no accuracy given [11]. |
| attention readers | attention weights per token | "Attention drift" [12] (abstract); behind HMM in [7]. Nothing measured on handwriting. |
| weakly supervised detection | a 2D box or heat map per character | CRAFT: an interim model splits each word; if the count misses the word length (confidence below 0.5) the word is cut evenly. Arabic and Bangla are "difficult to segment", being "written cursively" [13]. DTLR: detector pretrained on synthetic lines, then CTC over boxes sorted by x. No box metric; on CASIA "the bounding boxes degenerate" while the CER keeps improving [14]. A 2026 successor claims "accurate character bounding box prediction" from line transcripts, shown indirectly on "a fully separated formal script" [15]. |
| sprite models | a learned prototype per character at evenly spaced positions, CTC with an empty sprite | Document-specific, so it suits one hand. Scored by reconstruction and CER only: "Such fine-grained annotation is however not available in existing datasets" [16]. The sprite baseline in [15] needed 5 days 2 h to converge. |
| other tools | | Transkribus Text-Image Matching fits text to regions and lines [17]. The Bullinger alignment (2025) uses CTC posteriors as character-level alignment and measures lines only [6]. |

Online handwriting has letter ground truth and a comparison. IAM-OnDB, mIoU on test set F: spatial k-means 91.05, a
BiLSTM boundary predictor fed with CTC spikes 93.72, the authors' query transformer 95.11 (trained on two sets). The
BiLSTM, which is monotone, falls to 49.45 on Vietnamese, where delayed strokes are common [18].

## 2. Architecture and localisation

| question | evidence (speech unless marked) |
|---|---|
| whole-line recurrent or limited context | The best aligner in [3] is a 5 M TDNN-FFN that sees about 13 frames. With TDNN-BLSTM (27 M) and Conformer (85 M) the priors made boundaries worse: these models emitted tokens during silence to escape the prior [3]. A BLSTM alignment "can have further shifting due to the NN encoder" [8]. Handwriting: only the mechanism in [1] and littera's pilot (`local` 85 %, BiLSTM 86 % on synthetic letters). |
| transformers | Behind GMM-HMM as aligners [7]. Online handwriting: a transformer boundary predictor trails the BiLSTM (90.34 against 93.72) [18]. |
| frame stride | Stride 2 (20 ms) best; stride 1 "harder to train" and no better; stride 4 no better [3]. Subsampling improved the CTC alignment [8]. MFA's 10 ms resolution is named as one reason for its lead [7]. Handwriting: none found. |
| 2D or collapsed 1D | No controlled comparison found. The 2D detectors [14][15] output boxes, measured only indirectly and on separated script. |
| recognition accuracy as a guide | "higher ASR accuracy does not always lead to a better alignment quality" [8]. |

## 3. Data with letter-level ground truth

| set | annotation | size | licence, access |
|---|---|---|---|
| IAM-OnDB with Character Queries labels [18] | English pen trajectories; a character per sample point from a recogniser-scored cut search ("remains an approximation") | 221 writers, 13,049 lines (nc) | labels CC BY 4.0, GitHub; IAM-OnDB: non-commercial research, registration (nc) |
| BRUSH [19] | online; a character per point, from an image segmentation model according to [18] | 170 writers, 27,649 samples | non-commercial research; Google Drive |
| DeepWriting [20] | online, extends IAM-OnDB; monotone labels from a private tool, "cannot accommodate cursive writing" [18] | 294 authors | CC BY-NC-SA 4.0 with conditions; direct |
| CharLoc v3 [21] | character boxes, contours, labels on English word images (GoBo: 40 writers, 37k words) | 343 MB | CC BY 4.0, Zenodo; no paper found |
| Fontenay [22] | TEI with character ids and image zones from the Oriflamms software; share of hand-checked letters not stated | 104 charters, 12th c. to 1213 | CC BY 4.0, Zenodo |
| other scripts | Greek letter boxes on papyri [23]; Kuzushiji character boxes [24] | 150 + 34 images; 1,086,326 characters | CC BY-NC 4.0, Zenodo; CC BY-SA 4.0, direct |
| DigiPal / Archetype [25] | letter-form annotations, 11th-century English vernacular minuscule | counts, licence (nc) | |

Two searches for Kurrent or other 18th and 19th-century cursive with letter-level ground truth found nothing. The
nearest test is online handwriting rendered to images: letter extents in connected Latin cursive, with the pen lifts
known. It tests the method with the general model, not the Humboldt model; the non-commercial sets are for evaluation
only (CLAUDE.md).

## 4. littera's feature map and kraken's (from `model.py`, `graph.py`)

It is the same kind of feature map: a 2D CNN on the grey line strip, the height removed, one vector per 4 px column,
BiLSTM layers, a per-frame softmax. The comparison uses littera's copy of kraken's default network (`VGSLBackbone`
with the `kraken` context), not kraken's source; `model.py` loads only littera's own weights. The differences: eight
residual blocks with 3×3 kernels and batch norm against four plain layers with 13 and 9 px wide kernels (theoretical
horizontal receptive field about 109 and 103 px, computed by hand); the height pooled by learned attention over 6 rows
into 256 values against 12 rows stacked into 768 values, which keeps the vertical position; two BiLSTM layers of 256
against three of 200. littera adds a boundary head and the segmental objective, in which every frame belongs to a
letter, a gap or garbage, where kraken has one softmax with a CTC blank.

## 5. Verdict

1. No source shows another family placing letters better than an HMM-style model with explicit segments. Where hand
   labels exist (speech), the order is HMM, CTC with priors, then plain CTC and attention [3][7]. Nobody has published
   the comparison on hand-marked handwritten letters, so agent M's measures are the evidence.
2. Cheapest plausible gain: the limited-context encoder (`context="local"`, checkpoint `pilot-seg-local`) fine-tuned
   on Humboldt, one run like ft2 (about 40 GPU minutes). Support: [3], [1], [8]. The CNN alone already sees about
   109 px, several letters, so the effect may be small.
3. Next: pen-lift gap states with boundary supervision on real ink, as in the brief; a graph option and one or two
   fine-tunes.
4. Not for this round: detection models [14][15] (in [15] a million pretraining iterations and almost two days to
   converge; no loss holds the boxes on real data; shown on separated script), sprite models [16] (five days in [15];
   localisation never measured), label priors (behind HMM, worse with BiLSTM encoders [3]; littera's own attempt read
   9 % of the characters, PILOT.md), attention readers [7][12].
5. Precedent for the ink-anchored idea: according to [1] the CTC blank was meant for the connections and short white
   spaces between characters, and silence states in HMM aligners decide the shift [8]. Anchors as partial supervision
   follow CRAFT's confidence-weighted pseudo-labels [13] and the cut search at stroke boundaries in [18]. The box
   drift in [14] is the failure of ft2 without a boundary loss.
6. Pitfalls. (a) A pen lift is not always a letter boundary (Sayre's paradox [26]; delayed strokes broke the monotone
   predictor in [18]): keep the gap optional and accept an anchor only if the piece count fits, as CRAFT does.
   (b) Ink-free columns need upright strokes; spatial separation fails when "the cursive writing is slanted
   excessively" [18], so look for gaps after slant correction. (c) A long-context encoder can label empty frames as
   letters to avoid a penalty [3]; check this against the ink prior. (d) My inference, without a source: anchored
   letters are the unconnected ones (word edges, capitals), so the connected-run measure has to show the transfer.

## Sources

1. Bluche, Ney, Louradour, Kermorvant, "Framewise and CTC Training of Neural Networks for Handwriting Recognition", ICDAR 2015. https://www-i6.informatik.rwth-aachen.de/publications/downloader.php?id=1025&row=pdf
2. Zeyer, Schlüter, Ney, "Why does CTC result in peaky behavior?", 2021. https://arxiv.org/abs/2105.14849
3. Huang, Zhang, Ni, Sun, Hira, Hwang, Manohar, Pratap, Wiesner, Watanabe, Povey, Khudanpur, "Less Peaky and More Accurate CTC Forced Alignment by Label Priors", ICASSP 2024. https://arxiv.org/html/2406.02560v3
4. kraken, `kraken/align.py`, main branch. https://raw.githubusercontent.com/mittagessen/kraken/main/kraken/align.py
5. kraken 7.0, "Introduction to the Python API". https://kraken.re/7.0/user_guide/api.html
6. Peer, Scius-Bertrand, Fischer, "CTC Transcription Alignment of the Bullinger Letters: Automatic Improvement of Annotation Quality", 2025. https://arxiv.org/html/2508.07904
7. Rousso, Cohen, Keshet, Chodroff, "Tradition or Innovation: A Comparison of Modern ASR Methods for Forced Alignment", Interspeech 2024. https://arxiv.org/html/2406.19363
8. Raissi, Zhou, Berger, Schlüter, Ney, "HMM vs. CTC for Automatic Speech Recognition: Comparison Based on Full-Sum Training from Scratch", SLT 2022. https://arxiv.org/pdf/2210.09951
9. Toselli, Romero, Vidal, "Viterbi Based Alignment between Text Images and their Transcripts", LaTeCH 2007. https://aclanthology.org/W07-0902.pdf
10. Kong, Dyer, Smith, "Segmental Recurrent Neural Networks", ICLR 2016. https://arxiv.org/abs/1511.06018
11. Lavrentiev, Leydier, Stutzmann, "Specifying a TEI-XML Based Format for Aligning Text to Image at Character Level", Balisage 2015. https://balisage.net/Proceedings/vol16/html/Lavrentiev01/BalisageVol16-Lavrentiev01.html
12. Cheng, Bai, Xu, Zheng, Pu, Zhou, "Focusing Attention: Towards Accurate Text Recognition in Natural Images", ICCV 2017. https://arxiv.org/abs/1709.02054
13. Baek, Lee, Han, Yun, Lee, "Character Region Awareness for Text Detection", CVPR 2019. https://ar5iv.labs.arxiv.org/html/1904.01941
14. Baena, Kalleli, Aubry, "General Detection-based Text Line Recognition", NeurIPS 2024. https://arxiv.org/html/2409.17095v2
15. Vlachou Efstathiou, Baena, Stutzmann, Aubry, "Leveraging Morphology for Historical Script Metrological Analysis", arXiv, June 2026. https://arxiv.org/html/2606.09446
16. Siglidis, Gonthier, Gaubil, Monnier, Aubry, "The Learnable Typewriter: A Generative Approach to Text Analysis", 2023. https://ar5iv.labs.arxiv.org/html/2302.01660
17. Transkribus help, "Import existing Transcriptions with Text-Image Matching". https://help.transkribus.org/import-existing-transcriptions-with-text-image-matching
18. Jungo, Wolf, Maksai, Musat, Fischer, "Character Queries: A Transformer-based Approach to On-Line Handwritten Character Segmentation", ICDAR 2023. https://arxiv.org/pdf/2309.03072; labels: https://github.com/jungomi/character-queries
19. Kotani, Tellex, Tompkin, "Generating Handwriting via Decoupled Style Descriptors", ECCV 2020; BRUSH. https://github.com/brownvc/decoupled-style-descriptors
20. Aksan, Pece, Hilliges, "DeepWriting: Making Digital Ink Editable via Deep Generative Modeling", CHI 2018. https://ait.ethz.ch/deepwriting; https://github.com/emreaksan/deepwriting
21. Yasin, Zesch, "CharLoc" v3, Zenodo 2026. https://zenodo.org/records/21894406; GoBo (Gold, van den Boom, Zesch, 2021): https://zenodo.org/records/8085511
22. Stutzmann, "Fontenay Dataset. Original Charters From Fontenay before 1213", Zenodo 2022. https://zenodo.org/records/6507963
23. Marthot-Santaniello, Serbaeva and others, "ICDAR2023 Competition on Detection and Recognition of Greek Letters on Papyri" dataset, Zenodo 2024. https://zenodo.org/records/13825619
24. CODH, Kuzushiji dataset (日本古典籍くずし字データセット). http://codh.rois.ac.jp/char-shape/
25. DigiPal at King's Digital Lab, https://next.kdl.kcl.ac.uk/projects/digipal/; Archetype, https://wiki.digitalclassicist.org/Archetype
26. Wikipedia, "Sayre's paradox" (after Sayre 1973, Pattern Recognition 5(3)). https://en.wikipedia.org/wiki/Sayre%27s_paradox

Not opened, so not used as evidence: Fischer, Frinken, Fornés, Bunke 2011 (HMM transcription alignment of Latin
manuscripts; known through [6]); Bluche, Stutzmann, Kermorvant, DAS 2016 (character segmentation for palaeography; PDF
403, HAL blocked); Stökl Ben Ezra and others, ICFHR 2020 (Dead Sea Scrolls; abstract page only); the IAM-OnDB pages
(expired certificate; the figures above come from a search summary); digipal.eu (timeout); the MMS and WhisperX papers
(through [7]); Sayre 1973 (through [26]).
