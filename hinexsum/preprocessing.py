# -*- coding: utf-8 -*-
"""Preprocessing for Hindi text (Devanagari).

Mirrors Section 3.1 of Mohd, Jan & Shah (2020) with the language-specific
steps replaced for Hindi:

  - URL removal
  - Unicode normalization (NFC)  [replaces English lower-casing: Devanagari
    has no case, but Hindi text frequently mixes nukta / matra encodings]
  - Sentence splitting on the purna viram (danda) as well as ? and !
  - Tokenization
  - Hindi stop-word removal
  - Light suffix-stripping stemmer  [replaces CoreNLP lemmatization]

If the Indic NLP Library (`indic-nlp-library`) is installed it is used for
sentence splitting and tokenization; otherwise robust regex fallbacks are
used so the pipeline has no hard external dependency.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Optional Indic NLP Library support
# ----------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when the library is installed
    from indicnlp.tokenize import sentence_tokenize as _indic_sent
    from indicnlp.tokenize import indic_tokenize as _indic_tok

    _HAS_INDICNLP = True
except Exception:  # ImportError or missing resources
    _HAS_INDICNLP = False

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
# Sentence terminators: danda, double danda, ?, !, and '.' (common in Hindi
# news text). '.' is handled carefully to avoid splitting decimals/initials.
_SENT_SPLIT_RE = re.compile(r"(?<=[\u0964\u0965?!])\s+|(?<=[^\d]\.)\s+(?=[\u0900-\u097F\"'A-Z])")
# Token: Devanagari word, Latin word, or number. Punctuation is dropped.
# NOTE: \u0964 (danda) and \u0965 (double danda) sit inside the Devanagari
# block and must be excluded, or they stick to the last word of a sentence.
_TOKEN_RE = re.compile(r"[\u0900-\u0963\u0966-\u097F]+|[A-Za-z]+|\d+(?:\.\d+)?")


def _load_wordlist(fname: str) -> set:
    path = os.path.join(_HERE, "data", fname)
    words = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                w = line.strip()
                if w and not w.startswith("#"):
                    words.add(w)
    return words


HINDI_STOPWORDS = _load_wordlist("hindi_stopwords.txt")

# Suffixes for the light stemmer, longest first (based on Ramanathan &
# Rao's classic lightweight Hindi stemmer suffix inventory).
_HINDI_SUFFIXES = [
    # length 4+
    "ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां",
    "त्वों", "ेंगी", "ेंगे", "ूंगी", "ूंगा", "ाओगी", "ाओगे",
    "ियाँ", "ियों", "ियां", "त्व",
    # length 3
    "ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे",
    "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं",
    "ुओं", "ुएं", "ुआं",
    # length 2
    "कर", "ाओ", "िए", "ाई", "ाए", "नी", "ना", "ते", "ीं", "ती",
    "ता", "ाँ", "ां", "ों", "ें",
    # length 1 (matras)
    "ो", "े", "ू", "ु", "ी", "ि", "ा",
]


def normalize(text: str) -> str:
    """NFC-normalize and clean whitespace; strip URLs."""
    text = unicodedata.normalize("NFC", text)
    text = _URL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split Hindi text into sentences (danda-aware)."""
    text = normalize(text)
    if not text:
        return []
    if _HAS_INDICNLP:
        sents = _indic_sent.sentence_split(text, lang="hi")
    else:
        sents = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s and _DEVANAGARI_RE.search(s)]


def tokenize(sentence: str) -> List[str]:
    """Tokenize a sentence into word tokens (punctuation removed)."""
    if _HAS_INDICNLP:
        toks = _indic_tok.trivial_tokenize(sentence, lang="hi")
        return [t for t in toks if _TOKEN_RE.fullmatch(t)]
    return _TOKEN_RE.findall(sentence)


def stem(word: str) -> str:
    """Light suffix-stripping stemmer for Hindi."""
    if not _DEVANAGARI_RE.search(word):
        return word.lower()
    for suf in _HINDI_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return word


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in HINDI_STOPWORDS]


def preprocess_sentence(sentence: str, do_stem: bool = True) -> List[str]:
    """Full token pipeline for one sentence: tokenize -> stopwords -> stem."""
    toks = remove_stopwords(tokenize(sentence))
    if do_stem:
        toks = [stem(t) for t in toks]
    return toks
