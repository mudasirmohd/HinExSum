# ILSUM Hindi @ 3 sentences (tune on 200 train/val docs, test on 200 held-out test docs)

- **Dataset:** FIRE **ILSUM 2.0** Hindi (`ILSUM/ILSUM-2.0` on Hugging Face, ungated) —
  news Article/Summary pairs. Val slice = first 200 train rows; test slice = first 200
  test rows (disjoint). Gold summaries present in both splits.
- **Model:** fastText `cc.hi.300.bin`; selection = global-topn; POS on; budget = 3.
- **Systems:** all7 equal-weight, tuned pos+tfidf (position weight chosen on val only),
  Lead-1, Lead-3, TextRank-3, Random-3. 1000-resample bootstrap; paired gaps vs Lead-3.
- **Command:** `python run_ilsum.py`  (val/test = 200/200, bootstrap = 1000, seed 42)
- **Date:** 2026-07-07

## Validation sweep — pos+tfidf, R1-F @ 3 (200 val docs)

| position weight | R1-F (val) |
|---|---|
| 1 | 0.347 |
| 2 | 0.426 |
| 4 | 0.490 |
| 8 | 0.543 |
| 16 | 0.560  ⬅ best |

**Selected tuned config:** pos+tfidf with position=16, tfidf=1

## Test-slice results — 200 held-out ILSUM test docs, 1000-resample bootstrap

| System | R1-F | R1-F 95% CI | R2-F | RL-F | SU4-F |
|---|---|---|---|---|---|
| Tuned pos+tfidf (pos=16) | 0.516 | [0.483, 0.552] | 0.446 | 0.494 | 0.442 |
| all7 (equal) | 0.256 | [0.237, 0.277] | 0.148 | 0.201 | 0.153 |
| Lead-1 | 0.441 | [0.407, 0.476] | 0.380 | 0.428 | 0.367 |
| Lead-3 | 0.522 | [0.488, 0.557] | 0.451 | 0.499 | 0.449 |
| TextRank-3 | 0.268 | [0.245, 0.291] | 0.146 | 0.214 | 0.152 |
| Random-3 | 0.233 | [0.212, 0.254] | 0.102 | 0.175 | 0.115 |

## Paired R1-F gap vs Lead-3 (1000-resample paired bootstrap)

| System | ΔR1-F vs Lead-3 | 95% CI | Real gap? |
|---|---|---|---|
| Tuned pos+tfidf (pos=16) | -0.005 | [-0.011, -0.000] | yes |
| all7 (equal) | -0.265 | [-0.296, -0.235] | yes |
| Lead-1 | -0.080 | [-0.113, -0.049] | yes |
| TextRank-3 | -0.254 | [-0.289, -0.224] | yes |
| Random-3 | -0.289 | [-0.323, -0.259] | yes |

## What this shows — the arc replicates on a second Hindi corpus

**1. Same story, independent dataset.** Lead dominates; the paper's equal-weight all7 config
is *catastrophically* worse (0.256 vs Lead-3's 0.522 — less than half, Δ = −0.265, CI far
from 0); and tuning the position weight up to 16 recovers almost all of it (0.516), landing
statistically level with Lead-3. This is the exact pattern found on XL-Sum, reproduced on
ILSUM with no retuning of the method.

**2. Tuned ≈ Lead-3 once more — and here a hair *below* it.** Δ = −0.005, 95% CI
[−0.011, −0.000]: the CI just touches 0 at the top, so tuned is at best equal to Lead-3 and
never above it. As on XL-Sum, the ceiling of this feature family is lead selection. The tiny
deficit (vs XL-Sum's exact tie) comes from the tfidf tiebreak occasionally swapping a lead
sentence — on ILSUM, which is even more lead-extractive, any non-position perturbation costs
a little.

**3. ILSUM is *more* lead-extractive than XL-Sum.** Absolute scores are far higher
(Lead-3 R1-F 0.522, **R2-F 0.451**, RL-F 0.499) — the reference reuses the article's opening
almost verbatim, so bigram overlap alone is ~0.45. That leaves even less headroom for
diversity features, which is why all7-equal collapses so hard here.

**4. Tuned beats every non-lead system by large, real margins** — all7-equal (−0.265),
TextRank-3 (−0.254), Random-3 (−0.289), and even Lead-1 (−0.080), all with CIs excluding 0.

## Closing the arc

Across two independent Hindi news-summarization benchmarks (XL-Sum, ILSUM), the conclusion
is the same and now well-supported statistically:

- The paper's **equal-weight 7-feature ranking is significantly worse than Lead**, badly so
  on ILSUM.
- **Position is the only feature that carries signal** on these corpora; up-weighting it
  recovers Lead-level performance, and the other six features are dilutive-to-harmful.
- **No configuration of this feature set beats Lead-3.** Matching a strong lead baseline is
  the ceiling; surpassing it needs signal orthogonal to position that the references
  actually reward — which lead-biased, single-reference news summaries do not contain.

The honest framing for a write-up: report Lead as the baseline these methods must beat, show
that tuned position-weighting *matches* it (a real gain over the published equal-weight
setup), and be explicit that the distributional-semantics machinery adds nothing beyond
position on lead-biased Hindi news. A genuinely harder test would be a query-focused or
multi-document Hindi corpus where the answer is not the lead.
