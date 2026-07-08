# -*- coding: utf-8 -*-
"""Sentence ranking (Section 3.4 of the paper), adapted to Hindi.

Features per sentence:
  1. Sentence length            |s_i| after preprocessing
  2. Sentence position          s_i^p = 1 - (i - 1)/|S|
  3. TF-IDF                     sum of token TF-IDF scores
  4. Noun/verb phrase count     NVC via POS tags (Stanza 'hi' if available)
  5. Proper-noun count          NNP/PROPN tags (no capitalization in Hindi,
                                so this relies entirely on the tagger)
  6. Aggregate cosine sim.      mean embedding-cosine to all other sentences
  7. Cue phrases                Hindi cue-phrase lexicon bonus

Each feature is min-max normalized over the document; the final score is
the sum of normalized features. Connective handling and redundancy
elimination are implemented in summarizer.py.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_phrases(fname: str) -> List[str]:
    path = os.path.join(_HERE, "data", fname)
    phrases = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                p = line.strip()
                if p and not p.startswith("#"):
                    phrases.append(p)
    return phrases


CUE_PHRASES = _load_phrases("hindi_cue_phrases.txt")
CONNECTIVES = _load_phrases("hindi_connectives.txt")

# ----------------------------------------------------------------------------
# Optional POS tagging via Stanza (replaces Stanford POS tagger of the paper)
# ----------------------------------------------------------------------------
_STANZA_PIPELINE = None


def _get_stanza():
    """Lazily build a Stanza Hindi pipeline; returns None if unavailable."""
    global _STANZA_PIPELINE
    if _STANZA_PIPELINE is not None:
        return _STANZA_PIPELINE or None
    try:
        import stanza
        _STANZA_PIPELINE = stanza.Pipeline(
            "hi", processors="tokenize,pos",
            tokenize_pretokenized=False, verbose=False)
    except Exception:
        _STANZA_PIPELINE = False
    return _STANZA_PIPELINE or None


@lru_cache(maxsize=200000)
def pos_counts(sentence: str) -> Dict[str, int]:
    """Return counts of nouns+verbs (nvc) and proper nouns (propn).

    Memoized on the sentence string: the ablation harness scores the same
    document under several feature configurations, and this keeps Stanza from
    re-tagging identical sentences. Callers must treat the returned dict as
    read-only.

    Uses Stanza's Hindi model when installed (pip install stanza;
    stanza.download('hi')). Falls back to a crude heuristic otherwise:
    NVC ~ number of content tokens, PROPN ~ 0 (feature becomes neutral
    after normalization).
    """
    nlp = _get_stanza()
    if nlp is None:
        from .preprocessing import preprocess_sentence
        return {"nvc": len(preprocess_sentence(sentence, do_stem=False)),
                "propn": 0}
    doc = nlp(sentence)
    nvc = propn = 0
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos in ("NOUN", "VERB", "PROPN"):
                nvc += 1
            if word.upos == "PROPN":
                propn += 1
    return {"nvc": nvc, "propn": propn}


# ----------------------------------------------------------------------------
# TF-IDF over preprocessed tokens
# ----------------------------------------------------------------------------
def tfidf_scores(token_lists: Sequence[Sequence[str]]) -> List[float]:
    """Sentence TF-IDF = sum of token tf-idf, treating each sentence as a
    document (as in the paper's s_i^tf definition)."""
    n = len(token_lists)
    df: Counter = Counter()
    for toks in token_lists:
        df.update(set(toks))
    scores = []
    for toks in token_lists:
        tf = Counter(toks)
        s = 0.0
        for w, c in tf.items():
            idf = math.log((1 + n) / (1 + df[w])) + 1.0
            s += (c / max(len(toks), 1)) * idf
        scores.append(s)
    return scores


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def aggregate_cosine(sent_vecs: Sequence[np.ndarray]) -> List[float]:
    n = len(sent_vecs)
    if n <= 1:
        return [0.0] * n
    out = []
    for i in range(n):
        s = sum(cosine(sent_vecs[i], sent_vecs[j]) for j in range(n) if j != i)
        out.append(s / (n - 1))
    return out


def cue_phrase_score(sentence: str) -> int:
    return sum(1 for p in CUE_PHRASES if p in sentence)


def starts_with_connective(sentence: str) -> bool:
    s = sentence.strip()
    return any(s.startswith(c) for c in CONNECTIVES)


def _minmax(xs: List[float]) -> List[float]:
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


FEATURES = ("length", "position", "tfidf", "nvc", "propn", "cosine", "cue")


def compute_features(sentences: List[str],
                     token_lists: List[List[str]],
                     sent_vecs: List[np.ndarray],
                     use_pos: bool = True) -> Dict[str, List[float]]:
    """Per-feature, min-max-normalized score arrays for each sentence.

    Returned as a dict keyed by FEATURES. Computing this once and then taking
    different weighted sums is what makes a weight sweep cheap (features are the
    expensive part; the weighting is a dot product)."""
    n = len(sentences)
    if n == 0:
        return {}
    length = [float(len(t)) for t in token_lists]
    position = [1.0 - i / n for i in range(n)]
    tfidf = tfidf_scores(token_lists)
    cos = aggregate_cosine(sent_vecs)
    cue = [float(cue_phrase_score(s)) for s in sentences]

    if use_pos:
        pc = [pos_counts(s) for s in sentences]
        nvc = [float(p["nvc"]) for p in pc]
        propn = [float(p["propn"]) for p in pc]
    else:
        nvc = [0.0] * n
        propn = [0.0] * n

    return {
        "length": _minmax(length), "position": _minmax(position),
        "tfidf": _minmax(tfidf), "nvc": _minmax(nvc),
        "propn": _minmax(propn), "cosine": _minmax(cos),
        "cue": _minmax(cue),
    }


def weighted_scores(feats: Dict[str, List[float]],
                    weights: Optional[Dict[str, float]] = None) -> List[float]:
    """Combine a precomputed feature matrix into a per-sentence score."""
    if not feats:
        return []
    w = {f: 1.0 for f in FEATURES}
    if weights:
        w.update(weights)
    n = len(next(iter(feats.values())))
    return [sum(w[f] * feats[f][i] for f in feats) for i in range(n)]


def rank_sentences(sentences: List[str],
                   token_lists: List[List[str]],
                   sent_vecs: List[np.ndarray],
                   use_pos: bool = True,
                   weights: Optional[Dict[str, float]] = None) -> List[float]:
    """Compute the total (summed, normalized) ranking score of each sentence
    in the document. Indices align with `sentences`."""
    if len(sentences) == 0:
        return []
    feats = compute_features(sentences, token_lists, sent_vecs, use_pos)
    return weighted_scores(feats, weights)
