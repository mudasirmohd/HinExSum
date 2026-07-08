# -*- coding: utf-8 -*-
"""Devanagari-safe ROUGE: identity, disjoint, and partial-overlap cases."""

from hinexsum.rouge import rouge_scores

METRICS = ("rouge-1", "rouge-2", "rouge-l")


def test_rouge_identity():
    t = "भारत ने चंद्र मिशन का सफल प्रक्षेपण किया"
    sc = rouge_scores(t, [t])
    for m in METRICS:
        assert abs(sc[m]["f"] - 1.0) < 1e-9
        assert abs(sc[m]["p"] - 1.0) < 1e-9
        assert abs(sc[m]["r"] - 1.0) < 1e-9


def test_rouge_disjoint():
    sc = rouge_scores("एक दो तीन चार", ["पाँच छह सात आठ"])
    for m in METRICS:
        assert sc[m]["f"] == 0.0


def test_rouge_partial():
    # candidate and reference share exactly the unigrams {भारत, ने}
    sc = rouge_scores("भारत ने मिशन भेजा", ["भारत ने चंद्रमा"])
    f1 = sc["rouge-1"]["f"]
    assert 0.0 < f1 < 1.0
    # precision 2/4, recall 2/3 -> F1 ~= 0.571
    assert abs(sc["rouge-1"]["p"] - 0.5) < 1e-9
    assert abs(sc["rouge-1"]["r"] - 2 / 3) < 1e-9
