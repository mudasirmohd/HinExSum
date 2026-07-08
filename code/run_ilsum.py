#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ILSUM (FIRE) Hindi evaluation with the same harness as XL-Sum.

Systems @ 3 sentences: all7 equal-weight, tuned pos+tfidf (position weight
selected on a validation slice only), Lead-1, Lead-3, TextRank-3, Random-3.
Reports F-scores + 1000-resample bootstrap 95% CIs on R1-F, and paired R1-F
gaps vs Lead-3.

Tuning uses ILSUM train rows (val slice); final evaluation uses ILSUM test
rows (held-out) — the two slices are disjoint.
"""

import json
import sys

import numpy as np

from hinexsum import (HindiSummarizer, load_embeddings,
                              rouge_scores, average_scores,
                              split_sentences, preprocess_sentence)
from hinexsum.ranking import compute_features, weighted_scores
from hinexsum.summarizer import FEATURE_NAMES
from evaluate_xlsum import (lead_select, random_select, textrank_select,
                            _topn_summary, _r1f)

MODEL = "cc.hi.300.bin"
DATA = "ilsum_hindi.jsonl"
N, SEED, BOOT = 3, 42, 1000
VAL_LIMIT, TEST_LIMIT = 200, 200
METRICS = ("rouge-1", "rouge-2", "rouge-l", "rouge-su4")

if len(sys.argv) > 1:            # smoke test: argv = val_limit [test_limit [boot]]
    VAL_LIMIT = int(sys.argv[1])
    TEST_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else VAL_LIMIT
    BOOT = int(sys.argv[3]) if len(sys.argv) > 3 else BOOT


def load_split(path, want_split, limit):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("split") != want_split:
                continue
            if r.get("text") and r.get("summary"):
                out.append((r["text"], r["summary"]))
            if len(out) >= limit:
                break
    return out


def extract(summ, model, docs, tag):
    """(sentences, feats, ref) per doc, POS on, all features."""
    summ.use_pos = True; summ.features = None; summ.weights = None
    out = []
    for i, (text, ref) in enumerate(docs, 1):
        sents = split_sentences(text)
        if not sents:
            out.append(([], {}, ref)); continue
        toks = [preprocess_sentence(s) for s in sents]
        vecs = [model.sentence_vector(t) for t in toks]
        feats = compute_features(sents, toks, vecs, use_pos=True)
        out.append((sents, feats, ref, toks))
        if i % 25 == 0:
            print(f"  [{tag}] {i}/{len(docs)}", file=sys.stderr, flush=True)
    return out


def main():
    model = load_embeddings(MODEL, top_m=5, bv_size=100)
    summ = HindiSummarizer(model, use_pos=True)

    val = load_split(DATA, "train", VAL_LIMIT)
    test = load_split(DATA, "test", TEST_LIMIT)
    print(f"# ILSUM Hindi @ {N} sentences "
          f"(tune on {len(val)} train/val docs, test on {len(test)} held-out test docs)")

    # ---- sweep pos+tfidf position weight on validation -------------------
    print("Extracting validation features ...", file=sys.stderr, flush=True)
    vf = extract(summ, model, val, "val")
    grid = [1, 2, 4, 8, 16]
    sweep = []
    for w in grid:
        wd = {f: 0.0 for f in FEATURE_NAMES}; wd["position"] = float(w); wd["tfidf"] = 1.0
        r1 = []
        for rec in vf:
            sents, feats, ref = rec[0], rec[1], rec[2]
            if not sents:
                r1.append(0.0); continue
            sc = weighted_scores(feats, wd)
            r1.append(rouge_scores(_topn_summary(sents, sc, N), [ref])["rouge-1"]["f"])
        sweep.append((w, wd, float(np.mean(r1))))
    sweep.sort(key=lambda x: -x[2])
    best_w, best_wd, best_val = sweep[0]

    print(f"\n## Validation sweep — pos+tfidf, R1-F @ {N} ({len(val)} val docs)\n")
    print("| position weight | R1-F (val) |")
    print("|---|---|")
    for w, _, r in sorted(sweep, key=lambda x: x[0]):
        star = "  ⬅ best" if w == best_w else ""
        print(f"| {w} | {r:.3f}{star} |")
    print(f"\n**Selected tuned config:** pos+tfidf with position={best_w}, tfidf=1")

    # ---- evaluate all systems on the held-out test slice -----------------
    print("Extracting test features ...", file=sys.stderr, flush=True)
    tf = extract(summ, model, test, "test")
    names = [f"Tuned pos+tfidf (pos={best_w})", "all7 (equal)",
             "Lead-1", "Lead-3", "TextRank-3", "Random-3"]
    scores = {n: [] for n in names}
    for di, rec in enumerate(tf):
        sents, feats, ref = rec[0], rec[1], rec[2]
        toks = rec[3] if len(rec) > 3 else []
        if not sents:
            for nm in names:
                scores[nm].append(rouge_scores("", [ref]))
            continue
        sc_tuned = weighted_scores(feats, best_wd)
        sc_all = weighted_scores(feats, None)
        scores[names[0]].append(rouge_scores(_topn_summary(sents, sc_tuned, N), [ref]))
        scores["all7 (equal)"].append(rouge_scores(_topn_summary(sents, sc_all, N), [ref]))
        scores["Lead-1"].append(rouge_scores(" ".join(sents[i] for i in lead_select(len(sents), 1)), [ref]))
        scores["Lead-3"].append(rouge_scores(" ".join(sents[i] for i in lead_select(len(sents), 3)), [ref]))
        scores["TextRank-3"].append(rouge_scores(" ".join(sents[i] for i in textrank_select(toks, 3)), [ref]))
        scores["Random-3"].append(rouge_scores(" ".join(sents[i] for i in random_select(len(sents), 3, SEED + di + 1)), [ref]))

    ndoc = len(test)
    rng = np.random.default_rng(SEED)
    idxm = rng.integers(0, ndoc, size=(BOOT, ndoc))

    print(f"\n## Test-slice results — {ndoc} held-out ILSUM test docs, "
          f"{BOOT}-resample bootstrap\n")
    print("| System | R1-F | R1-F 95% CI | R2-F | RL-F | SU4-F |")
    print("|---|---|---|---|---|---|")
    for nm in names:
        avg = average_scores(scores[nm])
        arr = _r1f(scores[nm]); bm = arr[idxm].mean(axis=1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        print(f"| {nm} | {avg['rouge-1']['f']:.3f} | [{lo:.3f}, {hi:.3f}] | "
              f"{avg['rouge-2']['f']:.3f} | {avg['rouge-l']['f']:.3f} | "
              f"{avg['rouge-su4']['f']:.3f} |")

    ref_arr = _r1f(scores["Lead-3"])
    print(f"\n## Paired R1-F gap vs Lead-3 ({BOOT}-resample paired bootstrap)\n")
    print("| System | ΔR1-F vs Lead-3 | 95% CI | Real gap? |")
    print("|---|---|---|---|")
    for nm in names:
        if nm == "Lead-3":
            continue
        d = _r1f(scores[nm]) - ref_arr; bm = d[idxm].mean(axis=1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        real = "yes" if (lo > 0 or hi < 0) else "no"
        print(f"| {nm} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {real} |")


if __name__ == "__main__":
    main()
