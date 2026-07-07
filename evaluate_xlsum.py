#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the Hindi summarizer on XL-Sum (Hindi) or any JSONL corpus,
reproducing the style of Tables 2/3 of the paper (Pr / Rc / Fs for
ROUGE-1, -2, -L, -SU4 at 25% and 50% summary lengths).

Getting the data (one-time):
    pip install datasets
    python - <<'PY'
    from datasets import load_dataset
    ds = load_dataset("csebuetnlp/xlsum", "hindi", split="test")
    ds.to_json("xlsum_hindi_test.jsonl")
    PY

Then:
    python evaluate_xlsum.py --model cc.hi.300.bin \
        --data xlsum_hindi_test.jsonl --limit 100

    # exactly N sentences instead of a ratio:
    python evaluate_xlsum.py --model cc.hi.300.bin --data xlsum_hindi_test.jsonl \
        --fixed --max-sentences 3 --limit 200

    # our system vs baselines (Lead-1/Lead-3/Random/TextRank) at N=2 and N=3:
    python evaluate_xlsum.py --model cc.hi.300.bin --data xlsum_hindi_test.jsonl \
        --baselines --settings 2,3 --limit 200

JSONL rows need a "text" field (article) and "summary" field (reference).
"""

import argparse
import json
import random
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from hinexsum import (HindiSummarizer, load_embeddings,
                              rouge_scores, average_scores,
                              split_sentences, preprocess_sentence,
                              FEATURE_NAMES, SELECTIONS)
from hinexsum.ranking import weighted_scores

METRICS = ("rouge-1", "rouge-2", "rouge-l", "rouge-su4")


def parse_weights(spec):
    """'position=8,tfidf=1' -> {'position': 8.0, 'tfidf': 1.0}."""
    if not spec:
        return None
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in FEATURE_NAMES:
            raise SystemExit(f"unknown feature in --weights: {k!r}; "
                             f"valid: {','.join(FEATURE_NAMES)}")
        out[k] = float(v)
    return out


def load_docs(path, text_field, summary_field, limit=0, skip=0):
    """Load (text, summary) pairs from a JSONL corpus, skipping the first
    `skip` rows and returning at most `limit` (0 = all)."""
    docs = []
    with open(path, encoding="utf-8") as fh:
        for j, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if j < skip:
                continue
            row = json.loads(line)
            docs.append((row[text_field], row[summary_field]))
            if limit and len(docs) >= limit:
                break
    return docs


# ---------------------------------------------------------------------------
# Candidate builders (all operate on the *same* danda-aware sentence split so
# every system is scored on an identical segmentation -> a fair comparison)
# ---------------------------------------------------------------------------
def _join(sentences, idxs):
    return " ".join(sentences[i] for i in sorted(idxs))


def lead_select(n_sent, k):
    """Lead-k: the first k sentences (classic, very strong on news)."""
    return list(range(min(k, n_sent)))


def random_select(n_sent, k, seed):
    """k sentences chosen uniformly at random, seeded for reproducibility,
    returned in original document order."""
    if n_sent <= k:
        return list(range(n_sent))
    return sorted(random.Random(seed).sample(range(n_sent), k))


def textrank_select(token_lists, k, d=0.85, iters=60):
    """Unsupervised TextRank over the shared sentence split.

    Builds a TF-IDF cosine-similarity graph between sentences (using the same
    preprocessed tokens the main system uses) and runs PageRank via power
    iteration -- i.e. classic gensim/LexRank-style TextRank, but over
    hinexsum's segmentation so it is directly comparable. (gensim 4.x
    dropped its summarization module and sumy would re-split with its own
    English tokenizer, so we implement the graph here to keep the split fair.)
    """
    n = len(token_lists)
    if n <= k:
        return list(range(n))
    docs = [" ".join(t) for t in token_lists]
    try:
        X = TfidfVectorizer(token_pattern=r"\S+").fit_transform(docs)
    except ValueError:
        return list(range(k))            # empty vocabulary -> fall back to lead
    if X.shape[1] == 0:
        return list(range(k))
    sim = (X @ X.T).toarray().astype(float)
    np.fill_diagonal(sim, 0.0)
    deg = sim.sum(axis=1)
    deg[deg == 0] = 1.0
    M = sim / deg[:, None]               # row-stochastic transition matrix
    scores = np.full(n, 1.0 / n)
    for _ in range(iters):
        scores = (1.0 - d) / n + d * (M.T @ scores)
    top = sorted(range(n), key=lambda i: -scores[i])[:k]
    return sorted(top)


def system_fixed_summary(summarizer, text, n, selection=None):
    """Our system constrained to exactly n sentences (ratio ignored), under the
    given selection strategy. Uses the low-level score/select API so the budget
    is exactly n (no connective expansion) -- an apples-to-apples match to the
    fixed-budget baselines."""
    sentences, token_lists, _, scores = summarizer.score_sentences(text)
    if not sentences:
        return ""
    idxs = summarizer.select_indices(sentences, token_lists, scores, n, selection)
    return " ".join(sentences[i] for i in sorted(idxs))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_prf_table(label, ndocs, avg):
    """Paper Tables 2/3 style: Pr / Rc / Fs for each ROUGE variant."""
    print(f"\n=== Averaged results, {label} ({ndocs} docs) ===")
    print(f"{'Metric':<12}{'Pr':>8}{'Rc':>8}{'Fs':>8}")
    for m in METRICS:
        s = avg[m]
        print(f"{m.upper():<12}{s['p']:>8.3f}{s['r']:>8.3f}{s['f']:>8.3f}")
    macro_f = sum(avg[m]["f"] for m in avg) / len(avg)
    print(f"{'macro-avg F':<12}{'':>16}{macro_f:>8.3f}")


def print_f_table(setting_label, ndocs, system_names, system_scores):
    """Combined table for one setting: rows = systems, columns = ROUGE-1/2/L/SU4
    F-scores (Markdown so the redirected output is a clean .md)."""
    print(f"\n## Setting: {setting_label} ({ndocs} docs)\n")
    print("| System | R1-F | R2-F | RL-F | SU4-F |")
    print("|---|---|---|---|---|")
    for name in system_names:
        avg = average_scores(system_scores[name])
        cells = " | ".join(f"{avg[m]['f']:.3f}" for m in METRICS)
        print(f"| {name} | {cells} |")


# ---------------------------------------------------------------------------
def run_baselines(summarizer, docs, settings, seed):
    """Evaluate our system against Lead-1, Lead-3, Random-N and TextRank-N at
    each sentence budget N in `settings`, one combined F-score table per N."""
    for N in settings:
        names = ["Ours", "Lead-1", "Lead-3", f"Random-{N}", f"TextRank-{N}"]
        scores = {name: [] for name in names}
        for i, (text, ref) in enumerate(docs, 1):
            sents = split_sentences(text)
            toks = [preprocess_sentence(s) for s in sents]
            n = len(sents)

            scores["Ours"].append(
                rouge_scores(system_fixed_summary(summarizer, text, N), [ref]))
            scores["Lead-1"].append(
                rouge_scores(_join(sents, lead_select(n, 1)), [ref]))
            scores["Lead-3"].append(
                rouge_scores(_join(sents, lead_select(n, 3)), [ref]))
            scores[f"Random-{N}"].append(
                rouge_scores(_join(sents, random_select(n, N, seed + i)), [ref]))
            scores[f"TextRank-{N}"].append(
                rouge_scores(_join(sents, textrank_select(toks, N)), [ref]))

            if i % 25 == 0:
                print(f"  [N={N}] {i}/{len(docs)}", file=sys.stderr, flush=True)

        print_f_table(f"{N} sentences", len(docs), names, scores)

    # headline read: does our system beat the Lead-3 news baseline on R1-F?
    print("\n<!-- Headline: compare 'Ours' R1-F at N=3 against 'Lead-3'. -->")


# ---------------------------------------------------------------------------
# Selection x feature ablation, with bootstrap CIs on R1-F
# ---------------------------------------------------------------------------
def _r1f(scores_list):
    """Per-document ROUGE-1 F array from a list of rouge_scores() dicts."""
    return np.array([s["rouge-1"]["f"] for s in scores_list])


def run_ablation(summarizer, docs, n, seed, boot=1000):
    """One combined table at budget n comparing selection strategies and
    feature subsets against the baselines, with 95% bootstrap CIs on R1-F
    (per system, and paired vs Lead-3 so we know which gaps are real)."""
    ALL_MINUS_POS = [f for f in FEATURE_NAMES if f != "position"]
    row_names = ["Ours-full", "Ours-global-topn", "Ours-position-only",
                 "Ours-minus-position", "Lead-3", "TextRank-3", "Random-3"]
    scores = {r: [] for r in row_names}

    for i, (text, ref) in enumerate(docs, 1):
        sents = split_sentences(text)
        toks = [preprocess_sentence(s) for s in sents]
        nsent = len(sents)

        # Config A: all 7 features, POS on. Scored once, reused for the
        # cluster-rr (full system) and global-topn selections.
        summarizer.use_pos = True
        summarizer.features = None
        s_sents, s_toks, _, sc_all = summarizer.score_sentences(text)
        idx_full = summarizer.select_indices(s_sents, s_toks, sc_all, n, "cluster-rr")
        idx_gtn = summarizer.select_indices(s_sents, s_toks, sc_all, n, "global-topn")

        # Config B: all features except position, POS on (POS re-used from the
        # memoized tagger), global-topn.
        summarizer.features = ALL_MINUS_POS
        _, _, _, sc_mp = summarizer.score_sentences(text)
        idx_mp = summarizer.select_indices(s_sents, s_toks, sc_mp, n, "global-topn")

        # Config C: position only, POS off, global-topn (== Lead-n by
        # construction: position is monotone in sentence index).
        summarizer.use_pos = False
        summarizer.features = ["position"]
        _, _, _, sc_pos = summarizer.score_sentences(text)
        idx_po = summarizer.select_indices(s_sents, s_toks, sc_pos, n, "global-topn")

        def j(idx):
            return " ".join(s_sents[k] for k in sorted(idx))

        scores["Ours-full"].append(rouge_scores(j(idx_full), [ref]))
        scores["Ours-global-topn"].append(rouge_scores(j(idx_gtn), [ref]))
        scores["Ours-position-only"].append(rouge_scores(j(idx_po), [ref]))
        scores["Ours-minus-position"].append(rouge_scores(j(idx_mp), [ref]))
        scores["Lead-3"].append(rouge_scores(_join(sents, lead_select(nsent, 3)), [ref]))
        scores["TextRank-3"].append(rouge_scores(_join(sents, textrank_select(toks, 3)), [ref]))
        scores["Random-3"].append(rouge_scores(_join(sents, random_select(nsent, 3, seed + i)), [ref]))

        if i % 25 == 0:
            print(f"  [ablation N={n}] {i}/{len(docs)}", file=sys.stderr, flush=True)

    # bootstrap: one shared resample-index matrix so per-system CIs and paired
    # difference CIs are drawn from the same resamples.
    ndoc = len(docs)
    rng = np.random.default_rng(seed)
    idxm = rng.integers(0, ndoc, size=(boot, ndoc))

    print(f"\n## Ablation @ {n} sentences ({ndoc} docs), {boot}-resample bootstrap\n")
    print("| System | R1-F | R1-F 95% CI | R2-F | RL-F | SU4-F |")
    print("|---|---|---|---|---|---|")
    for name in row_names:
        avg = average_scores(scores[name])
        arr = _r1f(scores[name])
        bm = arr[idxm].mean(axis=1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        print(f"| {name} | {avg['rouge-1']['f']:.3f} | [{lo:.3f}, {hi:.3f}] | "
              f"{avg['rouge-2']['f']:.3f} | {avg['rouge-l']['f']:.3f} | "
              f"{avg['rouge-su4']['f']:.3f} |")

    ref_arr = _r1f(scores["Lead-3"])
    print(f"\n## Paired R1-F gap vs Lead-3 (95% CI, {boot}-resample paired bootstrap)\n")
    print("| System | ΔR1-F vs Lead-3 | 95% CI | Real gap (CI excludes 0)? |")
    print("|---|---|---|---|")
    for name in row_names:
        if name == "Lead-3":
            continue
        d = _r1f(scores[name]) - ref_arr
        bm = d[idxm].mean(axis=1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        real = "yes" if (lo > 0 or hi < 0) else "no"
        print(f"| {name} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {real} |")

    # restore an all-features / POS-on state
    summarizer.use_pos = True
    summarizer.features = None


# ---------------------------------------------------------------------------
# Weight tuning: sweep on validation, evaluate the single best config on a
# held-out test slice with bootstrap CIs.
# ---------------------------------------------------------------------------
def _topn_summary(sentences, scores, budget):
    n = len(sentences)
    if n == 0:
        return ""
    k = min(budget, n)
    idx = sorted(sorted(range(n), key=lambda i: -scores[i])[:k])
    return " ".join(sentences[i] for i in idx)


def _extract_feats(summarizer, docs, tag):
    """Compute (sentences, feature-matrix, ref) once per doc (POS on)."""
    summarizer.use_pos = True
    summarizer.features = None
    summarizer.weights = None
    out = []
    for i, (text, ref) in enumerate(docs, 1):
        sents, feats = summarizer.feature_matrix(text)
        out.append((sents, feats, ref))
        if i % 25 == 0:
            print(f"  [{tag} feats] {i}/{len(docs)}", file=sys.stderr, flush=True)
    return out


def run_tune(summarizer, val_docs, test_docs, budget, seed, boot=1000):
    pos_grid = [1, 2, 4, 8, 16]
    configs = []
    # Family A: all 7 features, others=1, position swept.
    for w in pos_grid:
        configs.append((f"all7 pos={w}", {"position": float(w)}))
    # Family B: position + tfidf only (others=0), position swept, tfidf=1.
    for w in pos_grid:
        wd = {f: 0.0 for f in FEATURE_NAMES}
        wd["position"] = float(w)
        wd["tfidf"] = 1.0
        configs.append((f"pos+tfidf pos={w}", wd))

    # ---- sweep on validation (features computed once per doc) ------------
    print("Extracting validation features ...", file=sys.stderr, flush=True)
    val = _extract_feats(summarizer, val_docs, "val")
    sweep = []
    for name, wd in configs:
        r1 = []
        for sents, feats, ref in val:
            if not sents:
                r1.append(0.0)
                continue
            sc = weighted_scores(feats, wd)
            r1.append(rouge_scores(_topn_summary(sents, sc, budget), [ref])["rouge-1"]["f"])
        sweep.append((name, wd, float(np.mean(r1))))

    sweep.sort(key=lambda r: -r[2])
    best_name, best_wd, best_val_r1 = sweep[0]

    print(f"\n## Validation sweep — R1-F @ {budget} sentences "
          f"({len(val_docs)} val docs, global-topn)\n")
    print("| Config | R1-F (val) |")
    print("|---|---|")
    for name, _, r in sweep:
        star = "  ⬅ best" if name == best_name else ""
        print(f"| {name} | {r:.3f}{star} |")
    wd_str = ", ".join(f"{k}={best_wd[k]:g}" for k in FEATURE_NAMES if best_wd.get(k))
    print(f"\n**Selected config:** `{best_name}`  (weights: {wd_str})")

    # ---- evaluate the single best config ONCE on the test slice ----------
    print("Extracting test-slice features ...", file=sys.stderr, flush=True)
    test = _extract_feats(summarizer, test_docs, "test")

    names = [f"Tuned ({best_name})", "Lead-3", "TextRank-3"]
    scores = {n: [] for n in names}
    for sents, feats, ref in test:
        if not sents:
            for n in names:
                scores[n].append(rouge_scores("", [ref]))
            continue
        sc = weighted_scores(feats, best_wd)
        scores[names[0]].append(rouge_scores(_topn_summary(sents, sc, budget), [ref]))
        scores["Lead-3"].append(rouge_scores(_join(sents, lead_select(len(sents), 3)), [ref]))
        toks = [preprocess_sentence(s) for s in sents]
        scores["TextRank-3"].append(rouge_scores(_join(sents, textrank_select(toks, 3)), [ref]))

    ndoc = len(test_docs)
    rng = np.random.default_rng(seed)
    idxm = rng.integers(0, ndoc, size=(boot, ndoc))

    print(f"\n## Test-slice result — best config, {ndoc} held-out docs, "
          f"{boot}-resample bootstrap\n")
    print("| System | R1-F | R1-F 95% CI | R2-F | RL-F | SU4-F |")
    print("|---|---|---|---|---|---|")
    for name in names:
        avg = average_scores(scores[name])
        arr = _r1f(scores[name])
        bm = arr[idxm].mean(axis=1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        print(f"| {name} | {avg['rouge-1']['f']:.3f} | [{lo:.3f}, {hi:.3f}] | "
              f"{avg['rouge-2']['f']:.3f} | {avg['rouge-l']['f']:.3f} | "
              f"{avg['rouge-su4']['f']:.3f} |")

    ref_arr = _r1f(scores["Lead-3"])
    print(f"\n## Paired R1-F gap vs Lead-3 (test slice, {boot}-resample paired bootstrap)\n")
    print("| System | ΔR1-F vs Lead-3 | 95% CI | Real gap? |")
    print("|---|---|---|---|")
    for name in [names[0], "TextRank-3"]:
        d = _r1f(scores[name]) - ref_arr
        bm = d[idxm].mean(axis=1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        real = "yes" if (lo > 0 or hi < 0) else "no"
        print(f"| {name} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {real} |")

    summarizer.use_pos = True
    summarizer.features = None
    summarizer.weights = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Path to Hindi embeddings (cc.hi.300.bin, IndicFT "
                         ".bin, or word2vec .vec/.txt)")
    ap.add_argument("--data", required=True, help="JSONL corpus path")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--summary-field", default="summary")
    ap.add_argument("--limit", type=int, default=0,
                    help="Evaluate at most N documents (0 = all)")
    ap.add_argument("--skip", type=int, default=0,
                    help="Skip the first N documents of --data")
    ap.add_argument("--ratios", default="0.25,0.5",
                    help="Comma-separated summary length ratios")
    ap.add_argument("--max-sentences", type=int, default=None,
                    help="Cap every summary at this many sentences "
                         "(passed to summarize(max_sentences=...))")
    ap.add_argument("--fixed", action="store_true",
                    help="Ignore --ratios and extract exactly --max-sentences "
                         "sentences per document")
    ap.add_argument("--baselines", action="store_true",
                    help="Compare our system to Lead-1/Lead-3/Random/TextRank "
                         "at each budget in --settings (one F-score table each)")
    ap.add_argument("--settings", default=None,
                    help="Comma-separated sentence budgets for --baselines "
                         "(default: --max-sentences, else 2,3)")
    ap.add_argument("--selection", default="cluster-rr", choices=list(SELECTIONS),
                    help="Sentence selection strategy for our system")
    ap.add_argument("--features", default=None,
                    help="Comma-separated feature subset to activate "
                         f"(default: all). Valid: {','.join(FEATURE_NAMES)}")
    ap.add_argument("--weights", default=None,
                    help="Per-feature weight overrides, e.g. "
                         "'position=8,tfidf=1' (applied on top of --features)")
    ap.add_argument("--tune", action="store_true",
                    help="Sweep position weight on validation (--data) and "
                         "evaluate the best config on a held-out test slice "
                         "(--test-data + --test-skip/--test-limit)")
    ap.add_argument("--test-data", default=None,
                    help="Test corpus for --tune's final evaluation")
    ap.add_argument("--test-skip", type=int, default=200,
                    help="Skip the first N test docs before evaluating (--tune)")
    ap.add_argument("--test-limit", type=int, default=200,
                    help="Number of test docs to evaluate (--tune)")
    ap.add_argument("--ablation", action="store_true",
                    help="Run the selection x feature ablation table with "
                         "bootstrap CIs on R1-F (uses --max-sentences, def 3)")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="Bootstrap resamples for --ablation CIs")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed for the Random-N baseline / bootstrap")
    ap.add_argument("--top-m", type=int, default=5)
    ap.add_argument("--bv-size", type=int, default=100)
    ap.add_argument("--no-pos", action="store_true",
                    help="Skip POS features (faster; no Stanza needed)")
    args = ap.parse_args()

    features = ([f.strip() for f in args.features.split(",")]
                if args.features else None)
    weights = parse_weights(args.weights)

    print(f"Loading embeddings from {args.model} ...", file=sys.stderr, flush=True)
    model = load_embeddings(args.model, top_m=args.top_m, bv_size=args.bv_size)
    summarizer = HindiSummarizer(model, use_pos=not args.no_pos,
                                 features=features, selection=args.selection,
                                 weights=weights)

    docs = load_docs(args.data, args.text_field, args.summary_field,
                     limit=args.limit, skip=args.skip)
    print(f"Loaded {len(docs)} documents"
          f"{f' (skipped {args.skip})' if args.skip else ''}.",
          file=sys.stderr, flush=True)

    # ---- weight-tuning mode (sweep on val, test on held-out slice) -------
    if args.tune:
        if not args.test_data:
            sys.exit("--tune requires --test-data")
        budget = args.max_sentences or 3
        test_docs = load_docs(args.test_data, args.text_field,
                              args.summary_field,
                              limit=args.test_limit, skip=args.test_skip)
        print(f"Loaded {len(test_docs)} test docs "
              f"(skipped {args.test_skip}).", file=sys.stderr, flush=True)
        print(f"# Weight tuning @ {budget} sentences "
              f"(sweep on {len(docs)} val docs, test on {len(test_docs)} "
              f"held-out docs {args.test_skip + 1}-{args.test_skip + len(test_docs)})")
        run_tune(summarizer, docs, test_docs, budget, args.seed,
                 boot=args.bootstrap)
        return

    # ---- selection x feature ablation mode -------------------------------
    if args.ablation:
        n = args.max_sentences or 3
        print(f"# Selection x feature ablation @ {n} sentences")
        run_ablation(summarizer, docs, n, args.seed, boot=args.bootstrap)
        return

    # ---- baseline comparison mode ----------------------------------------
    if args.baselines:
        if args.settings:
            settings = [int(s) for s in args.settings.split(",")]
        elif args.max_sentences:
            settings = [args.max_sentences]
        else:
            settings = [2, 3]
        pos = "7-feature (with POS)" if not args.no_pos else "6-feature (no POS / ablation)"
        print(f"# Baseline comparison (our system = {pos})")
        run_baselines(summarizer, docs, settings, args.seed)
        return

    # ---- fixed-N single-system mode --------------------------------------
    if args.fixed:
        if not args.max_sentences:
            sys.exit("--fixed requires --max-sentences N")
        n = args.max_sentences
        all_scores = []
        for i, (text, ref) in enumerate(docs, 1):
            summary = system_fixed_summary(summarizer, text, n, args.selection)
            all_scores.append(rouge_scores(summary, [ref]))
            if i % 25 == 0:
                print(f"  [fixed {n}] {i}/{len(docs)}", file=sys.stderr, flush=True)
        print_prf_table(f"fixed {n} sentences ({args.selection})",
                        len(docs), average_scores(all_scores))
        return

    # ---- ratio mode (original behaviour) ---------------------------------
    for ratio in [float(r) for r in args.ratios.split(",")]:
        all_scores = []
        for i, (text, ref) in enumerate(docs, 1):
            res = summarizer.summarize(text, ratio=ratio,
                                       max_sentences=args.max_sentences,
                                       selection=args.selection)
            all_scores.append(rouge_scores(res.summary, [ref]))
            if i % 25 == 0:
                print(f"  [{ratio:.0%}] {i}/{len(docs)}", file=sys.stderr, flush=True)
        print_prf_table(f"{ratio:.0%} summary length", len(docs),
                        average_scores(all_scores))


if __name__ == "__main__":
    main()
