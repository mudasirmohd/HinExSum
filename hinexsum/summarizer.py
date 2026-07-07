# -*- coding: utf-8 -*-
"""Hindi extractive summarizer replicating Mohd, Jan & Shah (2020).

Pipeline (Fig. 1 of the paper):
  1. Preprocess              (preprocessing.py, Hindi-specific)
  2. Big-vector generation   (embeddings.py, Hindi word vectors)
  3. TF-IDF vectorize BVs -> k-means clustering of sentences
  4. Rank sentences per cluster (ranking.py)
  5. Redundancy elimination + connective rule -> extractive summary
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import split_sentences, preprocess_sentence
from .embeddings import SemanticModel
from .ranking import (rank_sentences, starts_with_connective,
                      compute_features, weighted_scores)

# The seven ranking features (Section 3.4). A subset can be activated via the
# `features` argument to run feature ablations (e.g. position only).
FEATURE_NAMES = ("length", "position", "tfidf", "nvc", "propn", "cosine", "cue")

# Sentence-selection strategies.
SELECTION_CLUSTER_RR = "cluster-rr"    # k-means clusters + round-robin (paper)
SELECTION_GLOBAL_TOPN = "global-topn"  # rank all sentences, take top N, no clustering
SELECTIONS = (SELECTION_CLUSTER_RR, SELECTION_GLOBAL_TOPN)


@dataclass
class SummaryResult:
    summary: str
    selected_indices: List[int]
    sentences: List[str]
    scores: List[float]
    clusters: List[int]
    diagnostics: Dict = field(default_factory=dict)


class HindiSummarizer:
    def __init__(self,
                 model: SemanticModel,
                 n_clusters: Optional[int] = None,
                 use_pos: bool = True,
                 redundancy_eps: float = 0.02,
                 random_state: int = 42,
                 features: Optional[Sequence[str]] = None,
                 selection: str = SELECTION_CLUSTER_RR,
                 weights: Optional[Dict[str, float]] = None):
        """
        model          : SemanticModel wrapping Hindi embeddings
        n_clusters     : k for k-means; default = ceil(sqrt(#sentences))
        use_pos        : compute NVC / proper-noun features (needs Stanza
                         for real POS tags; degrades gracefully without it)
        redundancy_eps : two sentences in the same cluster whose total
                         scores differ by less than eps (on normalized
                         scores) are treated as semantically redundant and
                         only one is kept (Section 3.4)
        features       : subset of FEATURE_NAMES to activate (None = all 7).
                         Inactive features get weight 0 in the ranking sum.
        selection      : 'cluster-rr' (paper) or 'global-topn'.
        weights        : optional per-feature weight overrides applied on top of
                         the features mask (e.g. {'position': 8}).
        """
        self.model = model
        self.n_clusters = n_clusters
        self.use_pos = use_pos
        self.redundancy_eps = redundancy_eps
        self.random_state = random_state
        self.features = features
        self.selection = selection
        self.weights = weights

    # ------------------------------------------------------------------
    def _weights(self) -> Optional[Dict[str, float]]:
        """Effective weight dict: start from the features mask (active=1,
        inactive=0; all=1 when features is None), then apply self.weights
        overrides. Returns None only when nothing is customized."""
        if self.features is None and not self.weights:
            return None
        if self.features is None:
            base = {f: 1.0 for f in FEATURE_NAMES}
        else:
            active = set(self.features)
            unknown = active - set(FEATURE_NAMES)
            if unknown:
                raise ValueError(f"unknown feature(s): {sorted(unknown)}; "
                                 f"valid: {FEATURE_NAMES}")
            base = {f: (1.0 if f in active else 0.0) for f in FEATURE_NAMES}
        if self.weights:
            unknown = set(self.weights) - set(FEATURE_NAMES)
            if unknown:
                raise ValueError(f"unknown weighted feature(s): {sorted(unknown)}; "
                                 f"valid: {FEATURE_NAMES}")
            base.update(self.weights)
        return base

    def feature_matrix(self, text: str):
        """Return (sentences, feats_dict) where feats_dict holds the min-max
        normalized per-feature arrays. Lets a weight sweep score many configs
        from a single (expensive) feature computation per document."""
        sentences = split_sentences(text)
        if not sentences:
            return [], {}
        token_lists = [preprocess_sentence(s) for s in sentences]
        sent_vecs = [self.model.sentence_vector(t) for t in token_lists]
        feats = compute_features(sentences, token_lists, sent_vecs,
                                 use_pos=self.use_pos)
        return sentences, feats

    def score_sentences(self, text: str):
        """Run the expensive part: split, preprocess, embed, POS-tag and rank.
        Returns (sentences, token_lists, sent_vecs, scores). Selection-agnostic,
        so one call can feed several selection strategies."""
        sentences = split_sentences(text)
        n = len(sentences)
        if n == 0:
            return [], [], [], []
        token_lists = [preprocess_sentence(s) for s in sentences]
        sent_vecs = [self.model.sentence_vector(t) for t in token_lists]
        scores = rank_sentences(sentences, token_lists, sent_vecs,
                                use_pos=self.use_pos, weights=self._weights())
        return sentences, token_lists, sent_vecs, scores

    # ------------------------------------------------------------------
    def _cluster_rr_select(self, sentences: List[str],
                           token_lists: List[List[str]],
                           scores: List[float], target: int
                           ) -> Tuple[List[int], List[int], int, int]:
        """k-means clustering + redundancy elimination + round-robin selection.
        Returns (selected_indices, labels, k, removed_redundant)."""
        n = len(sentences)
        bv_texts = [self.model.big_vector_text(toks) for toks in token_lists]
        k = self.n_clusters or max(2, min(n - 1, math.ceil(math.sqrt(n))))
        k = min(k, n)
        X = TfidfVectorizer(token_pattern=r"\S+").fit_transform(bv_texts)
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(X)

        # redundancy elimination within clusters: near-equal scores in the
        # same (semantic) cluster => redundant; earlier sentence survives.
        alive = [True] * n
        for c in range(k):
            idxs = sorted((i for i in range(n) if labels[i] == c),
                          key=lambda i: -scores[i])
            for a in range(len(idxs)):
                if not alive[idxs[a]]:
                    continue
                for b in range(a + 1, len(idxs)):
                    if not alive[idxs[b]]:
                        continue
                    if abs(scores[idxs[a]] - scores[idxs[b]]) < self.redundancy_eps:
                        drop = max(idxs[a], idxs[b])  # keep the earlier sentence
                        alive[drop] = False

        # round-robin: best-ranked surviving sentence per cluster, by round.
        per_cluster = {
            c: sorted((i for i in range(n) if labels[i] == c and alive[i]),
                      key=lambda i: -scores[i])
            for c in range(k)
        }
        selected: List[int] = []
        rounds = 0
        while len(selected) < target and rounds < n:
            progressed = False
            order = sorted(per_cluster,
                           key=lambda c: -(scores[per_cluster[c][0]]
                                           if per_cluster[c] else -1e9))
            for c in order:
                if per_cluster[c] and len(selected) < target:
                    selected.append(per_cluster[c].pop(0))
                    progressed = True
            if not progressed:
                break
            rounds += 1
        return selected, list(labels), k, n - sum(alive)

    def select_indices(self, sentences: List[str],
                       token_lists: List[List[str]],
                       scores: List[float], target: int,
                       selection: Optional[str] = None) -> List[int]:
        """Return exactly `target` sentence indices (in document order) under
        the chosen selection strategy. No connective expansion, so the budget
        is exact -- used by the fixed-budget evaluation/ablation harness."""
        selection = selection or self.selection
        n = len(sentences)
        if n <= target:
            return list(range(n))
        if selection == SELECTION_GLOBAL_TOPN:
            top = sorted(range(n), key=lambda i: -scores[i])[:target]
            return sorted(top)
        selected, _, _, _ = self._cluster_rr_select(
            sentences, token_lists, scores, target)
        # guarantee exactly `target` by taking the highest-scoring picks
        selected = sorted(selected, key=lambda i: -scores[i])[:target]
        return sorted(selected)

    # ------------------------------------------------------------------
    def summarize(self, text: str, ratio: float = 0.25,
                  max_sentences: Optional[int] = None,
                  selection: Optional[str] = None) -> SummaryResult:
        selection = selection or self.selection
        sentences, token_lists, sent_vecs, scores = self.score_sentences(text)
        n = len(sentences)
        if n == 0:
            return SummaryResult("", [], [], [], [])
        target = max(1, math.ceil(n * ratio))
        if max_sentences:
            target = min(target, max_sentences)
        if n <= target:
            return SummaryResult(" ".join(sentences), list(range(n)),
                                 sentences, [1.0] * n, [0] * n)

        if selection == SELECTION_GLOBAL_TOPN:
            selected = sorted(range(n), key=lambda i: -scores[i])[:target]
            ordered = sorted(selected)
            summary = " ".join(sentences[i] for i in ordered)
            return SummaryResult(summary, ordered, sentences, scores, [0] * n,
                                 {"k": 0, "target": target,
                                  "selection": selection,
                                  "removed_redundant": 0})

        # cluster-rr (paper pipeline)
        selected, labels, k, removed = self._cluster_rr_select(
            sentences, token_lists, scores, target)

        # connective rule: pull in the preceding sentence
        final = set(selected)
        for i in list(final):
            if i > 0 and starts_with_connective(sentences[i]):
                final.add(i - 1)

        ordered = sorted(final)[: max(target, len(final))]
        summary = " ".join(sentences[i] for i in sorted(ordered))
        return SummaryResult(summary, sorted(ordered), sentences,
                             scores, labels,
                             {"k": k, "target": target,
                              "selection": selection,
                              "removed_redundant": removed})
