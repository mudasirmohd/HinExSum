# ROUGE validation — hinexsum/rouge.py vs XL-Sum multilingual ROUGE

- External scorer: csebuetnlp `rouge_score` (fork of google-research), `lang='hindi'`, pyonmttok tokenizer, `use_stemmer=False`.
- 20 XL-Sum test docs (ablation set); systems scored: Ours-full, Lead-3, TextRank-3 at 3 sentences.
- 'Ours'/'Ext' are mean F over the 20 docs; |Δ| is the mean of per-document |ours − ext|.

## Table 1 — mean F-scores, side by side

| System | Metric | Ours F | Ext F | mean per-doc |Δ| |
|---|---|---|---|---|
| Ours-full | R1 | 0.201 | 0.201 | 0.000 |
| Ours-full | R2 | 0.058 | 0.058 | 0.000 |
| Ours-full | RL | 0.139 | 0.139 | 0.000 |
| Lead-3 | R1 | 0.244 | 0.244 | 0.000 |
| Lead-3 | R2 | 0.063 | 0.063 | 0.000 |
| Lead-3 | RL | 0.177 | 0.177 | 0.000 |
| TextRank-3 | R1 | 0.216 | 0.216 | 0.000 |
| TextRank-3 | R2 | 0.059 | 0.059 | 0.000 |
| TextRank-3 | RL | 0.145 | 0.145 | 0.000 |

## Table 2 — pooled per-document mean absolute difference (all systems, n=60 pairs)

| Metric | mean |Δ| | max |Δ| |
|---|---|---|
| R1 | 0.0000 | 0.0000 |
| R2 | 0.0000 | 0.0000 |
| RL | 0.0000 | 0.0000 |

**Verdict: PASS** — largest mean per-document |Δ| across R1/R2/RL is 0.0000 (threshold 0.02).

## Scorer independence & the tokenization difference (why 0.0000 is not a tautology)

The two scorers are genuinely independent code with **different tokenizers**:

- **Ours** (`hinexsum/rouge.py`): regex `[ऀ-ॿ]+|[A-Za-z]+|\d+`. The danda
  (U+0964) and double danda (U+0965) fall inside the Devanagari block, so a sentence-final
  word keeps its danda glued to it — e.g. `भेजा।`, `है।`, `सफल।`.
- **External** (XL-Sum `rouge_score`, `lang='hindi'`): a `MultiTokenizer` whose sanitizer
  first strips punctuation, then applies an aggressive `pyonmttok` + whitespace pass — so the
  same words come out as `भेजा`, `है`, `सफल` (danda removed).

That difference is real and *can* move the score. On a crafted danda-flip case — reference
`देश का मिशन सफल।` vs candidate `मिशन सफल रहा है और देश खुश है।` — ours scores R1-F **0.333**
(its `सफल।` does not match the candidate's `सफल`) while the external scorer gives **0.500**
(both reduce to `सफल`): |Δ| = 0.167. This confirms the external path is exercised and not an
accidental alias of ours.

It does not surface on real data because the gluing is applied *consistently within each
scorer* and, in extractive summaries, danda sits sentence-finally in both candidate and
reference; a flip requires the same content word to appear sentence-finally in one text and
mid-sentence in the other, which did not decisively occur in any of the 60 pairs (max
per-document |Δ| = 0.0000 to four decimals). The agreement therefore validates
`rouge.py` for this task while the diagnostic documents the one behavioural difference to be
aware of.

<!-- WORST_MEAN_ABSDIFF=0.0000 VERDICT=PASS -->
