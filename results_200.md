# XL-Sum (Hindi) Evaluation — 200 documents

- **Model:** fastText `cc.hi.300.bin` (Common Crawl, 300-dim)
- **Corpus:** `xlsum_hindi_test.jsonl` (XL-Sum Hindi test split), first 200 docs
- **Command:** `python evaluate_xlsum.py --model cc.hi.300.bin --data xlsum_hindi_test.jsonl --limit 200 --no-pos`
- **Config:** `top_m=5`, `bv_size=100`, POS features disabled (`--no-pos`)
- **Date:** 2026-07-06

Values are averaged ROUGE Precision / Recall / F-score (paper Tables 2–3 style).

## 25% summary length

| Metric | Pr | Rc | Fs |
|---|---|---|---|
| ROUGE-1 | 0.095 | 0.555 | 0.157 |
| ROUGE-2 | 0.026 | 0.148 | 0.042 |
| ROUGE-L | 0.063 | 0.373 | 0.104 |
| ROUGE-SU4 | 0.032 | 0.203 | 0.053 |
| **macro-avg F** | | | **0.089** |

## 50% summary length

| Metric | Pr | Rc | Fs |
|---|---|---|---|
| ROUGE-1 | 0.069 | 0.643 | 0.122 |
| ROUGE-2 | 0.022 | 0.206 | 0.039 |
| ROUGE-L | 0.048 | 0.453 | 0.084 |
| ROUGE-SU4 | 0.026 | 0.264 | 0.046 |
| **macro-avg F** | | | **0.073** |

## Notes

- **Recall ≫ Precision** is expected: XL-Sum references are short (~1 sentence)
  abstractive summaries, while this is an extractive system emitting several
  sentences. The README flags this explicitly — report it, don't hide it.
- Raising the ratio from 25% → 50% trades precision for recall (a longer extract
  covers more reference tokens but adds non-matching text), lowering macro-F.
- POS features were disabled (`--no-pos`); enabling Stanza may shift ranking slightly.

### Raw program output

```
=== Averaged results, 25% summary length (200 docs) ===
Metric            Pr      Rc      Fs
ROUGE-1        0.095   0.555   0.157
ROUGE-2        0.026   0.148   0.042
ROUGE-L        0.063   0.373   0.104
ROUGE-SU4      0.032   0.203   0.053
macro-avg F                    0.089

=== Averaged results, 50% summary length (200 docs) ===
Metric            Pr      Rc      Fs
ROUGE-1        0.069   0.643   0.122
ROUGE-2        0.022   0.206   0.039
ROUGE-L        0.048   0.453   0.084
ROUGE-SU4      0.026   0.264   0.046
macro-avg F                    0.073
```
