# -*- coding: utf-8 -*-
"""Devanagari-safe ROUGE: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-SU4.

The standard ROUGE-1.5.5 / many Python ports assume ASCII word patterns and
silently drop Devanagari tokens. This implementation tokenizes on Unicode
word boundaries (Devanagari + Latin + digits) so Hindi n-grams are counted
correctly, and reports precision / recall / F1 against one or more
reference summaries (max over references, as is conventional).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

_TOKEN_RE = re.compile(r"[\u0900-\u097F]+|[A-Za-z]+|\d+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _skip_bigrams(tokens: Sequence[str], max_gap: int = 4) -> Counter:
    """Skip-bigrams with maximum skip distance (SU4 uses max_gap=4),
    plus unigrams (the 'U' in SU)."""
    sb = Counter()
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 2 + max_gap, len(tokens))):
            sb[(tokens[i], tokens[j])] += 1
    for t in tokens:
        sb[("<U>", t)] += 1
    return sb


def _prf(match: int, cand_total: int, ref_total: int) -> Dict[str, float]:
    p = match / cand_total if cand_total else 0.0
    r = match / ref_total if ref_total else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f": f}


def _overlap(c: Counter, r: Counter) -> int:
    return sum((c & r).values())


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            cur[j] = (prev[j - 1] + 1 if a[i - 1] == b[j - 1]
                      else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_scores(candidate: str, references: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Return {'rouge-1'|'rouge-2'|'rouge-l'|'rouge-su4': {'p','r','f'}}.
    Scores are the max over the provided reference summaries."""
    cand = _tokens(candidate)
    out: Dict[str, Dict[str, float]] = {}
    best = {m: {"p": 0.0, "r": 0.0, "f": -1.0}
            for m in ("rouge-1", "rouge-2", "rouge-l", "rouge-su4")}

    c1, c2, csu = _ngrams(cand, 1), _ngrams(cand, 2), _skip_bigrams(cand)

    for ref_text in references:
        ref = _tokens(ref_text)
        r1, r2, rsu = _ngrams(ref, 1), _ngrams(ref, 2), _skip_bigrams(ref)

        cand_scores = {
            "rouge-1": _prf(_overlap(c1, r1), sum(c1.values()), sum(r1.values())),
            "rouge-2": _prf(_overlap(c2, r2), sum(c2.values()), sum(r2.values())),
            "rouge-l": _prf(_lcs_len(cand, ref), len(cand), len(ref)),
            "rouge-su4": _prf(_overlap(csu, rsu), sum(csu.values()), sum(rsu.values())),
        }
        for m, s in cand_scores.items():
            if s["f"] > best[m]["f"]:
                best[m] = s

    for m in best:
        if best[m]["f"] < 0:
            best[m]["f"] = 0.0
        out[m] = best[m]
    return out


def average_scores(list_of_scores: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Macro-average a list of rouge_scores() results."""
    if not list_of_scores:
        return {}
    metrics = list_of_scores[0].keys()
    avg = {m: {k: 0.0 for k in ("p", "r", "f")} for m in metrics}
    for sc in list_of_scores:
        for m in metrics:
            for k in ("p", "r", "f"):
                avg[m][k] += sc[m][k]
    n = len(list_of_scores)
    for m in metrics:
        for k in ("p", "r", "f"):
            avg[m][k] /= n
    return avg
