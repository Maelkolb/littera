# littera as a verifier of readings (2026-11-20)

ehd-htr's `judge.words` scores each line's candidate texts on the ink with littera-humboldt-ft3 (the whole-line
log-likelihood under the decode graph). On the 119 unseen Humboldt evaluation pages the ll delta of a one-word swap
against the pick tells a swap that brings the line closer to the edition from one that takes it further with AUC 0.785;
the three line readers' scores together reach 0.738. littera is the strongest single judge there.

Tried to make it a better judge: `src/littera/verify.py`, `modal/humboldt_verify.py`. ft3 fine-tuned 2,000 steps (A10G)
on the humboldt_ft pool with its usual loss plus `softplus((margin - (nll(corruption) - nll(truth))) / tau)` over six
corruptions per line drawn from ehd-htr's readers' confusions on test_page (`/vol/verify/confusions.json`: m/n, a/e/o, é/e,
r/n, spaces and dots dropped or added; plus words of the pool that look alike, capitals).

| model | val pairs right | median gap (nats) | val CER | swap AUC (ehd-htr, 29,393 swaps) |
|---|---|---|---|---|
| ft3 (`v0-ft3`, learning rate 0) | 0.974 | 12.1 | 0.120 | 0.785 |
| `v1` (lr 5e-5, weight 1) | 0.974 | 14.9 | 0.121 | 0.788 |
| `v1-strong` (lr 1e-4, weight 3) | 0.975 | 17.1 | 0.125 | 0.790 |

The corruptions are too easy: ft3 already ranks 97.4 % of them right, and the fine-tune widens the margin without ranking
more pairs right; on the readers' real errors it changes nothing that matters. Not adopted; ehd-htr keeps ft3. Released
for reference as `/vol/models/littera-humboldt-v1` and `-v1-strong`. Other decode graphs (no garbage between words,
required spaces, dearer garbage) and crops (grow 0, 6, 12, their mean) move the AUC by at most 0.003. What the ink does not
see is mostly spaces (AUC 0.705) and figures (0.61). Harder negatives would be the readers' own errors on pages with
truth that no evaluation uses.
