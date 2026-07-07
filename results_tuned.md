# Weight tuning @ 3 sentences (sweep on 200 val docs, test on 200 held-out docs 201-400)

## Validation sweep — R1-F @ 3 sentences (200 val docs, global-topn)

| Config | R1-F (val) |
|---|---|
| pos+tfidf pos=16 | 0.239  ⬅ best |
| pos+tfidf pos=8 | 0.238 |
| all7 pos=16 | 0.235 |
| pos+tfidf pos=4 | 0.234 |
| all7 pos=8 | 0.227 |
| pos+tfidf pos=2 | 0.223 |
| pos+tfidf pos=1 | 0.211 |
| all7 pos=4 | 0.209 |
| all7 pos=2 | 0.198 |
| all7 pos=1 | 0.191 |

**Selected config:** `pos+tfidf pos=16`  (weights: position=16, tfidf=1)

## Test-slice result — best config, 200 held-out docs, 1000-resample bootstrap

| System | R1-F | R1-F 95% CI | R2-F | RL-F | SU4-F |
|---|---|---|---|---|---|
| Tuned (pos+tfidf pos=16) | 0.231 | [0.221, 0.241] | 0.056 | 0.153 | 0.074 |
| Lead-3 | 0.231 | [0.221, 0.242] | 0.056 | 0.154 | 0.074 |
| TextRank-3 | 0.207 | [0.197, 0.217] | 0.049 | 0.139 | 0.067 |

## Paired R1-F gap vs Lead-3 (test slice, 1000-resample paired bootstrap)

| System | ΔR1-F vs Lead-3 | 95% CI | Real gap? |
|---|---|---|---|
| Tuned (pos+tfidf pos=16) | -0.000 | [-0.002, +0.001] | no |
| TextRank-3 | -0.024 | [-0.034, -0.015] | yes |

## Setup

- **Model:** fastText `cc.hi.300.bin`; POS on; selection = global-topn; budget = 3.
- **Tune/test split:** config selected purely on **200 validation docs**
  (`xlsum_hindi_val.jsonl`), then evaluated **once** on **held-out test docs 201–400**
  (`xlsum_hindi_test.jsonl`, disjoint from the 1–200 analysed earlier). No test peeking.
- **Command:** `python evaluate_xlsum.py --model cc.hi.300.bin --data xlsum_hindi_val.jsonl --limit 200 --tune --test-data xlsum_hindi_test.jsonl --test-skip 200 --test-limit 200 --max-sentences 3 --bootstrap 1000 --seed 42`
- **Date:** 2026-07-07

## What this shows

**1. Tuning recovers all the lost ground — the tuned system now ties Lead-3.**
On held-out data, Tuned = 0.231 vs Lead-3 = 0.231; paired Δ = −0.000 with a very tight
95% CI **[−0.002, +0.001]**. This is a statistical dead heat, not an underpowered "can't
tell." Recall the untuned equal-weight system was 0.185/0.186 — *significantly worse* than
Lead-3. Making position dominate (weight 16) and dropping the other five features closes
that entire gap.

**2. But it does not beat Lead-3 — the ceiling of this feature set is lead selection.**
At position weight 16 the ranking is essentially lead-locked; the tfidf term only breaks
ties among near-lead sentences and nets out to zero effect vs pure lead. So on XL-Sum
Hindi, the best this distributional-semantics feature family can do at 3 sentences is
*replicate* Lead-3. The embedding/cluster/POS machinery adds nothing beyond position here.

**3. The validation sweep is monotonic and consistent with the ablation.**
Within both families, R1-F rises monotonically with position weight (all7: 0.191→0.235;
pos+tfidf: 0.211→0.239), and pos+tfidf ≥ all7 at every matched weight. Every non-position
feature is dilutive; the more you let position dominate, the better — up to the Lead-3
ceiling. The paper's equal-weight config (all7 pos=1) is dead last.

**4. The tuned system does beat the other unsupervised baselines.**
Because Tuned ≈ Lead-3 and Lead-3 > TextRank-3 (Δ = −0.024, CI excludes 0) and > Random-3
(from the ablation), the tuned config is significantly above TextRank-3 and Random-3.

## Bottom line

Weight tuning turns a system that *lost* to Lead-3 into one that *matches* it — a real,
methodologically clean improvement (selected on val, confirmed once on a disjoint test
slice). It does **not** surpass Lead-3: on a lead-biased, single-reference, abstractive
corpus, position is the whole story, and no reweighting of these seven features extracts
signal beyond it.

To actually beat Lead-3 you need signal orthogonal to position that the reference rewards
— e.g. features tuned to XL-Sum's abstractive style, a learned (not hand-swept) weighting,
or a different corpus (ILSUM / multi-reference, longer targets) where lead bias is weaker
and the diversity features can pay off. That is the honest next fork.
