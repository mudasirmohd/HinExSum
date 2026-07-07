#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selection-position analysis on the XL-Sum ablation docs (first 200 test docs).

For each system, records the distribution of *source-article positions* of the
sentences it selects (normalized to [0,1], binned into deciles), plus mean
normalized position. Then prints 3 example documents contrasting what
Ours-minus-position picks against Lead-3 and the reference summary.

Systems mirror results_ablation.md exactly (3-sentence budget, POS on).
"""

import sys

import numpy as np

from hinexsum import (HindiSummarizer, load_embeddings,
                              split_sentences, preprocess_sentence)
from hinexsum.ranking import compute_features, weighted_scores
from hinexsum.summarizer import FEATURE_NAMES
from evaluate_xlsum import lead_select, random_select, textrank_select, load_docs

MODEL, DATA, N, LIMIT, SEED = "cc.hi.300.bin", "xlsum_hindi_test.jsonl", 3, 200, 42
if len(sys.argv) > 1:            # optional: override LIMIT for a smoke test
    LIMIT = int(sys.argv[1])
SYSTEMS = ["Ours-full", "Ours-global-topn", "Ours-position-only",
           "Ours-minus-position", "Lead-3", "TextRank-3", "Random-3"]


def norm_positions(idxs, n):
    return [0.0] * len(idxs) if n <= 1 else [i / (n - 1) for i in idxs]


def clip(s, k=150):
    s = " ".join(s.split())
    return s if len(s) <= k else s[:k] + "…"


def main():
    model = load_embeddings(MODEL, top_m=5, bv_size=100)
    summ = HindiSummarizer(model, use_pos=True)
    docs = load_docs(DATA, "text", "summary", limit=LIMIT, skip=0)

    deciles = {s: [0] * 10 for s in SYSTEMS}
    meanpos = {s: [] for s in SYSTEMS}
    records = []

    mp_w = {f: 1.0 for f in FEATURE_NAMES}; mp_w["position"] = 0.0   # all-minus-position
    po_w = {f: 0.0 for f in FEATURE_NAMES}; po_w["position"] = 1.0   # position only

    for di, (text, ref) in enumerate(docs):
        sents = split_sentences(text)
        n = len(sents)
        if n == 0:
            continue
        token_lists = [preprocess_sentence(s) for s in sents]
        sent_vecs = [model.sentence_vector(t) for t in token_lists]
        feats = compute_features(sents, token_lists, sent_vecs, use_pos=True)
        sc_all = weighted_scores(feats, None)
        sc_mp = weighted_scores(feats, mp_w)
        sc_po = weighted_scores(feats, po_w)

        sel = {
            "Ours-full": summ.select_indices(sents, token_lists, sc_all, N, "cluster-rr"),
            "Ours-global-topn": summ.select_indices(sents, token_lists, sc_all, N, "global-topn"),
            "Ours-position-only": summ.select_indices(sents, token_lists, sc_po, N, "global-topn"),
            "Ours-minus-position": summ.select_indices(sents, token_lists, sc_mp, N, "global-topn"),
            "Lead-3": lead_select(n, 3),
            "TextRank-3": textrank_select(token_lists, 3),
            "Random-3": random_select(n, 3, SEED + di + 1),
        }
        for s in SYSTEMS:
            for p in norm_positions(sel[s], n):
                deciles[s][min(9, int(p * 10))] += 1
                meanpos[s].append(p)
        records.append((di, n, ref, sents, sel))
        if (di + 1) % 25 == 0:
            print(f"  [sel] {di + 1}/{len(docs)}", file=sys.stderr, flush=True)

    # ---- report ----------------------------------------------------------
    print("# Selection-position analysis — XL-Sum Hindi, first 200 test docs "
          "(ablation set)\n")
    print("- Budget: 3 sentences. POS on. Positions normalized to [0,1] "
          "(0 = article start, 1 = article end) and binned into deciles.\n")

    print("## Fraction of selected sentences by source-position decile\n")
    header = "| System | " + " | ".join(f"D{d+1}" for d in range(10)) + " | mean pos |"
    print(header)
    print("|" + "---|" * 12)
    for s in SYSTEMS:
        tot = sum(deciles[s]) or 1
        cells = " | ".join(f"{deciles[s][d] / tot:.2f}" for d in range(10))
        mp = np.mean(meanpos[s]) if meanpos[s] else 0.0
        print(f"| {s} | {cells} | {mp:.3f} |")
    print("\nD1 = first 10% of the article … D10 = last 10%. "
          "Rows sum to ~1.00 across D1–D10.\n")

    # ---- 3 examples: where Ours-minus-position strays furthest from lead --
    def mp_mean(rec):
        _, n, _, _, sel = rec
        ps = norm_positions(sel["Ours-minus-position"], n)
        return np.mean(ps) if ps else 0.0

    examples = sorted((r for r in records if r[1] >= 6),
                      key=mp_mean, reverse=True)[:3]

    print("## Examples — Ours-minus-position vs Lead-3 vs reference\n")
    print("Three documents where dropping the position feature pulls selection "
          "furthest from the lead (largest mean selected position). Sentences "
          "shown as `[source_index] text`.\n")
    for rank, (di, n, ref, sents, sel) in enumerate(examples, 1):
        print(f"### Example {rank} — test doc #{di + 1} ({n} sentences)\n")
        print(f"**Reference summary:** {clip(ref, 300)}\n")
        print(f"**Lead-3** (positions {sel['Lead-3']}):")
        for i in sel["Lead-3"]:
            print(f"- `[{i}]` {clip(sents[i])}")
        mp_idx = sel["Ours-minus-position"]
        mp_norm = ", ".join(f"{i}({i/(n-1):.2f})" for i in mp_idx)
        print(f"\n**Ours-minus-position** (positions {mp_idx}; normalized {mp_norm}):")
        for i in mp_idx:
            print(f"- `[{i}]` {clip(sents[i])}")
        print()


if __name__ == "__main__":
    main()
