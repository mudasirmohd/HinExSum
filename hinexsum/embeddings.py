# -*- coding: utf-8 -*-
"""Distributional semantics: embedding loading and big-vector generation.

Implements Section 3.2 / 3.2.1 of the paper for Hindi:

  BV(s) = phi(w1) ++ phi(w2) ++ ... ++ phi(wk)

where phi(w) returns the top-m most similar words to w in the embedding
space, and the resulting word list is padded / truncated to a fixed size n.

Supported embedding sources (pick ONE and pass its path to `load_embeddings`):

  1. fastText binary (recommended for Hindi - subword info handles rich
     morphology and OOV):        cc.hi.300.bin        (fasttext.cc)
  2. AI4Bharat IndicFT:          indicnlp.ft.hi.300.bin
  3. Any word2vec-format text/binary vectors (.vec / .txt / .bin), e.g.
     cc.hi.300.vec or a skip-gram model you train yourself with gensim
     (methodologically closest to the original paper).

Download (run once, outside this library):
  fastText CC:  https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.bin.gz
  IndicFT:      https://ai4bharat.iitm.ac.in/resources (IndicFT, Hindi)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Sequence

import numpy as np


class SemanticModel:
    """Wraps a gensim KeyedVectors / FastTextKeyedVectors object and
    provides big-vector construction."""

    def __init__(self, kv, top_m: int = 5, bv_size: int = 100,
                 pad_token: str = "<pad>"):
        self.kv = kv
        self.top_m = top_m          # m: similar words concatenated per word
        self.bv_size = bv_size      # n: fixed big-vector length (in words)
        self.pad_token = pad_token

    # -- phi(w): top-m similar words -------------------------------------
    @lru_cache(maxsize=100_000)
    def similar_words(self, word: str) -> tuple:
        try:
            sims = self.kv.most_similar(word, topn=self.top_m)
            return tuple(w for w, _ in sims)
        except KeyError:
            # OOV for pure word2vec models (fastText models rarely hit this)
            return (word,)

    # -- big-vector as a word sequence -----------------------------------
    def big_vector_words(self, tokens: Sequence[str]) -> List[str]:
        """Concatenate top-m similar words of every token, then pad or
        truncate to exactly `bv_size` words."""
        bv: List[str] = []
        for w in tokens:
            bv.append(w)
            bv.extend(self.similar_words(w))
            if len(bv) >= self.bv_size:
                break
        bv = bv[: self.bv_size]
        if len(bv) < self.bv_size:
            bv += [self.pad_token] * (self.bv_size - len(bv))
        return bv

    def big_vector_text(self, tokens: Sequence[str]) -> str:
        """Big-vector rendered as a space-joined pseudo-document, ready for
        TF-IDF vectorization (Section 3.3)."""
        return " ".join(self.big_vector_words(tokens))

    # -- dense sentence vector (used for cosine-similarity feature) ------
    def sentence_vector(self, tokens: Sequence[str]) -> np.ndarray:
        vecs = []
        for w in tokens:
            try:
                vecs.append(self.kv[w])
            except KeyError:
                continue
        if not vecs:
            return np.zeros(self.kv.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0)


def load_embeddings(path: str, top_m: int = 5, bv_size: int = 100) -> SemanticModel:
    """Load Hindi embeddings from `path` and return a SemanticModel.

    Handles: fastText .bin (full model), word2vec .vec/.txt (text),
    word2vec .bin (binary), and gensim-native saves.
    """
    from gensim.models import KeyedVectors

    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        try:
            # fastText binary (cc.hi.300.bin / IndicFT) - keeps subword info
            from gensim.models.fasttext import load_facebook_vectors
            kv = load_facebook_vectors(path)
        except Exception:
            # plain word2vec binary
            kv = KeyedVectors.load_word2vec_format(path, binary=True)
    elif ext in (".vec", ".txt"):
        kv = KeyedVectors.load_word2vec_format(path, binary=False)
    else:
        # gensim-native (KeyedVectors.save / Word2Vec.save)
        try:
            kv = KeyedVectors.load(path)
        except Exception:
            from gensim.models import Word2Vec
            kv = Word2Vec.load(path).wv
    return SemanticModel(kv, top_m=top_m, bv_size=bv_size)
