# -*- coding: utf-8 -*-
"""Big-vector generation (Section 3.2): padding and truncation to a fixed size.

Uses a fake KeyedVectors so the test needs no embedding model on disk."""

import numpy as np

from hinexsum.embeddings import SemanticModel


class FakeKV:
    """Minimal stand-in for a gensim KeyedVectors object."""
    vector_size = 4

    def most_similar(self, word, topn):
        return [(f"{word}~{i}", 1.0) for i in range(topn)]

    def __getitem__(self, word):
        return np.ones(self.vector_size, dtype=np.float32)


def test_big_vector_padding():
    m = SemanticModel(FakeKV(), top_m=2, bv_size=12, pad_token="<pad>")
    bv = m.big_vector_words(["क", "ख"])   # 2 tokens x (1 + 2 similar) = 6 words
    assert len(bv) == 12                    # padded up to bv_size
    assert bv.count("<pad>") == 6
    assert bv[0] == "क"


def test_big_vector_truncation():
    m = SemanticModel(FakeKV(), top_m=5, bv_size=4, pad_token="<pad>")
    bv = m.big_vector_words(["a", "b", "c", "d", "e"])   # far more than bv_size
    assert len(bv) == 4                     # truncated down to bv_size
    assert "<pad>" not in bv


def test_big_vector_text_is_space_joined():
    m = SemanticModel(FakeKV(), top_m=1, bv_size=3)
    txt = m.big_vector_text(["क"])
    assert isinstance(txt, str)
    assert len(txt.split()) == 3
