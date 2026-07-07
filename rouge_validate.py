#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate hinexsum/rouge.py against the XL-Sum authors' multilingual
ROUGE scorer (csebuetnlp fork of google-research rouge_score, lang='hindi',
pyonmttok tokenizer). Scores identical system summaries with both scorers on
20 XL-Sum ablation docs and reports F-score agreement.
"""

import sys

import numpy as np

from hinexsum import (HindiSummarizer, load_embeddings, rouge_scores,
                              split_sentences, preprocess_sentence)
from hinexsum.ranking import compute_features, weighted_scores
from evaluate_xlsum import lead_select, textrank_select, load_docs
from rouge_score import rouge_scorer

MODEL, DATA, N, LIMIT = "cc.hi.300.bin", "xlsum_hindi_test.jsonl", 3, 20
SYSTEMS = ["Ours-full", "Lead-3", "TextRank-3"]
PAIRS = {"R1": ("rouge-1", "rouge1"), "R2": ("rouge-2", "rouge2"),
         "RL": ("rouge-l", "rougeL")}
THRESHOLD = 0.02


def main():
    model = load_embeddings(MODEL, top_m=5, bv_size=100)
    summ = HindiSummarizer(model, use_pos=True)
    docs = load_docs(DATA, "text", "summary", limit=LIMIT, skip=0)
    ext = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                   use_stemmer=False, lang="hindi")

    # per system: metric -> list of (ours_f, ext_f); plus pooled abs-diff
    data = {s: {m: [] for m in PAIRS} for s in SYSTEMS}
    pooled = {m: [] for m in PAIRS}

    for di, (text, ref) in enumerate(docs, 1):
        sents = split_sentences(text)
        n = len(sents)
        toks = [preprocess_sentence(s) for s in sents]
        vecs = [model.sentence_vector(t) for t in toks]
        feats = compute_features(sents, toks, vecs, use_pos=True)
        sc_all = weighted_scores(feats, None)
        cands = {
            "Ours-full": " ".join(sents[i] for i in summ.select_indices(sents, toks, sc_all, N, "cluster-rr")),
            "Lead-3": " ".join(sents[i] for i in lead_select(n, 3)),
            "TextRank-3": " ".join(sents[i] for i in textrank_select(toks, 3)),
        }
        for s in SYSTEMS:
            o = rouge_scores(cands[s], [ref])
            e = ext.score(ref, cands[s])          # (target, prediction)
            for m, (ok, ek) in PAIRS.items():
                of, ef = o[ok]["f"], e[ek].fmeasure
                data[s][m].append((of, ef))
                pooled[m].append(abs(of - ef))
        print(f"  [rouge-val] {di}/{len(docs)}", file=sys.stderr, flush=True)

    # ---- report ----------------------------------------------------------
    print("# ROUGE validation — hinexsum/rouge.py vs XL-Sum multilingual ROUGE\n")
    print(f"- External scorer: csebuetnlp `rouge_score` (fork of google-research), "
          f"`lang='hindi'`, pyonmttok tokenizer, `use_stemmer=False`.")
    print(f"- {LIMIT} XL-Sum test docs (ablation set); systems scored: "
          f"{', '.join(SYSTEMS)} at {N} sentences.")
    print(f"- 'Ours'/'Ext' are mean F over the {LIMIT} docs; |Δ| is the mean of "
          f"per-document |ours − ext|.\n")

    print("## Table 1 — mean F-scores, side by side\n")
    print("| System | Metric | Ours F | Ext F | mean per-doc |Δ| |")
    print("|---|---|---|---|---|")
    for s in SYSTEMS:
        for m in PAIRS:
            arr = np.array(data[s][m])
            of, ef = arr[:, 0].mean(), arr[:, 1].mean()
            ad = np.abs(arr[:, 0] - arr[:, 1]).mean()
            print(f"| {s} | {m} | {of:.3f} | {ef:.3f} | {ad:.3f} |")

    print("\n## Table 2 — pooled per-document mean absolute difference "
          f"(all systems, n={LIMIT * len(SYSTEMS)} pairs)\n")
    print("| Metric | mean |Δ| | max |Δ| |")
    print("|---|---|---|")
    worst = 0.0
    for m in PAIRS:
        md = float(np.mean(pooled[m]))
        mx = float(np.max(pooled[m]))
        worst = max(worst, md)
        print(f"| {m} | {md:.4f} | {mx:.4f} |")

    verdict = "PASS" if worst <= THRESHOLD else "FAIL"
    print(f"\n**Verdict: {verdict}** — largest mean per-document |Δ| across R1/R2/RL "
          f"is {worst:.4f} (threshold {THRESHOLD}).")
    # machine-readable line for the harness
    print(f"\n<!-- WORST_MEAN_ABSDIFF={worst:.4f} VERDICT={verdict} -->")


if __name__ == "__main__":
    main()
