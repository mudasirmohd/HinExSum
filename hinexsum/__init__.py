# -*- coding: utf-8 -*-
"""Hindi extractive summarization via distributional semantics.

Replication of: Mohd, Jan & Shah (2020), "Text document summarization
using word embedding", Expert Systems With Applications 143:112958 —
adapted for Hindi (Devanagari) text.
"""

from .embeddings import SemanticModel, load_embeddings
from .summarizer import (HindiSummarizer, SummaryResult,
                         FEATURE_NAMES, SELECTIONS)
from .rouge import rouge_scores, average_scores
from .preprocessing import split_sentences, preprocess_sentence

__all__ = [
    "SemanticModel", "load_embeddings",
    "HindiSummarizer", "SummaryResult", "FEATURE_NAMES", "SELECTIONS",
    "rouge_scores", "average_scores",
    "split_sentences", "preprocess_sentence",
]
__version__ = "1.0.0"
