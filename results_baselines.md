# Baseline comparison — XL-Sum (Hindi), 200 documents

- **Our system:** 6-feature config, **POS disabled** (`--no-pos`) → this is the **ablation** row, not the full system.
- **Model:** fastText `cc.hi.300.bin` (Common Crawl, 300-dim)
- **Corpus:** `xlsum_hindi_test.jsonl` (XL-Sum Hindi test), first 200 docs
- **Command:** `python evaluate_xlsum.py --model cc.hi.300.bin --data xlsum_hindi_test.jsonl --limit 200 --no-pos --baselines --settings 2,3 --seed 42`
- **Fairness:** every system (ours + baselines) is scored on the *same* danda-aware
  sentence split from `hinexsum.preprocessing`. TextRank is a TF-IDF cosine
  graph + PageRank power-iteration over that same split (gensim 4.x dropped its
  summarizer; sumy would re-split with an English tokenizer, so it is implemented
  in-file to keep the segmentation identical).
- **Date:** 2026-07-07

Cells are averaged ROUGE **F-scores**.

## Setting: 2 sentences (200 docs)

| System | R1-F | R2-F | RL-F | SU4-F |
|---|---|---|---|---|
| Ours | 0.201 | 0.046 | 0.134 | 0.063 |
| Lead-1 | 0.223 | 0.050 | 0.168 | 0.071 |
| Lead-3 | 0.228 | 0.058 | 0.160 | 0.076 |
| Random-2 | 0.206 | 0.039 | 0.139 | 0.061 |
| TextRank-2 | 0.223 | 0.054 | 0.151 | 0.072 |

## Setting: 3 sentences (200 docs)

| System | R1-F | R2-F | RL-F | SU4-F |
|---|---|---|---|---|
| Ours | 0.183 | 0.043 | 0.120 | 0.059 |
| Lead-1 | 0.223 | 0.050 | 0.168 | 0.071 |
| Lead-3 | 0.228 | 0.058 | 0.160 | 0.076 |
| Random-3 | 0.200 | 0.039 | 0.130 | 0.060 |
| TextRank-3 | 0.204 | 0.053 | 0.140 | 0.068 |

## Headline read — Ours R1-F @ 3 vs Lead-3

**Ours = 0.183 vs Lead-3 = 0.228 → we do NOT beat Lead-3 (−0.045 R1-F).**

This is the important-to-know outcome. In this (no-POS) config our system is in fact
the **lowest** R1-F in the 3-sentence table — below Lead-1 (0.223), TextRank-3 (0.204),
and even Random-3 (0.200). The same ordering holds at N=2.

Two diagnostic signals:

1. **Adding a 3rd sentence hurts us but not the Leads.** Ours drops 0.201 → 0.183 from
   N=2 → N=3, while Lead-1/Lead-3 are budget-independent. The extra sentence our ranker
   selects is *lowering* R1-F — it adds non-matching tokens against XL-Sum's ~1-sentence
   references instead of matching content.
2. **Losing to Lead-1 (a single sentence) is the real tell.** XL-Sum Hindi references are
   short, abstractive, and written from the article's opening. Our round-robin
   *per-cluster* selection deliberately spreads picks across the document for diversity —
   exactly the wrong bias when almost all the reference-matching content sits at the top.

### Caveats before over-reading this
- This is the **POS-ablated** system. The full 7-feature run (Stanza now installed) is the
  fair headline number; POS features (NVC / proper-noun) may re-rank selection.
- XL-Sum is single-reference and abstractive, which structurally favors Lead on ROUGE.
  It is the honest benchmark to report, but it is the hardest possible case for a
  diversity-oriented extractive system — a multi-reference or DUC-style corpus would be a
  fairer analogue of the paper's original setup.

### Next steps this points to
- Run the **full 7-feature** config (same command without `--no-pos`) and compare that
  Ours@3 R1-F against Lead-3 — that is the number to build on or worry about.
- Test the user's hypothesis directly: does the **position feature** already dominate the
  score? An ablation that (a) drops position and (b) up-weights position would show whether
  the other five features add anything beyond lead bias on this corpus.
- Consider evaluating on a longer-reference / multi-reference Hindi corpus (ILSUM) where an
  extractive system is not fighting a 1-sentence lead-biased target.
