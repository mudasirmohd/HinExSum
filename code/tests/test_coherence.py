# -*- coding: utf-8 -*-
"""The discourse-coherence rule: a selected sentence that begins with a
connective drags its predecessor into the summary. This is a coherence
heuristic in this implementation; the tests below exercise both the underlying
predicate and the end-to-end behaviour."""

import numpy as np

from hinexsum.ranking import starts_with_connective, CONNECTIVES
from hinexsum.embeddings import SemanticModel
from hinexsum.summarizer import HindiSummarizer


class FakeKV:
    vector_size = 4

    def most_similar(self, word, topn):
        return [(f"{word}~{i}", 1.0) for i in range(topn)]

    def __getitem__(self, word):
        return np.ones(self.vector_size, dtype=np.float32)


def test_connective_lexicon_loaded():
    assert len(CONNECTIVES) > 0
    assert "लेकिन" in CONNECTIVES


def test_starts_with_connective_predicate():
    assert starts_with_connective("लेकिन यह सच है।")
    assert starts_with_connective("परंतु वह नहीं आया।")
    assert not starts_with_connective("यह एक सामान्य वाक्य है।")


def test_coherence_rule_pulls_in_preceding_sentence():
    # Two sentences: index 1 begins with the connective 'लेकिन' and is made to
    # outscore index 0 (longer, contains the cue phrase 'निष्कर्षतः'). With a
    # one-sentence budget the ranker picks sentence 1, and the coherence rule
    # must then add its predecessor (sentence 0).
    model = SemanticModel(FakeKV(), top_m=3, bv_size=20)
    summ = HindiSummarizer(model, use_pos=False, n_clusters=1, redundancy_eps=0.0)
    text = ("भारत एक देश है। "
            "लेकिन निष्कर्षतः यह एक बहुत लंबा और महत्वपूर्ण वाक्य है जो "
            "अधिक जानकारी और अधिक विवरण देता है।")
    res = summ.summarize(text, ratio=0.5)   # target = 1 before the coherence rule
    assert 1 in res.selected_indices          # the connective sentence was selected
    assert 0 in res.selected_indices          # its predecessor was pulled in
    assert res.selected_indices == [0, 1]
