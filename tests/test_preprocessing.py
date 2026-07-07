# -*- coding: utf-8 -*-
"""Preprocessing: sentence splitting, tokenizer danda exclusion, stopword
removal, and the light Hindi stemmer."""

from hinexsum.preprocessing import (split_sentences, tokenize,
                                     remove_stopwords, stem, HINDI_STOPWORDS)

DANDA = "।"


def test_split_sentences_danda_question_exclaim():
    text = "यह पहला वाक्य है। क्या यह दूसरा वाक्य है? हाँ यह तीसरा वाक्य है!"
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0].endswith("।")
    assert sents[1].endswith("?")
    assert sents[2].endswith("!")


def test_split_sentences_keeps_only_devanagari_segments():
    # a trailing numeric/punctuation-only fragment carries no Devanagari and is dropped
    sents = split_sentences("यह वाक्य है। 12345 ...")
    assert len(sents) == 1
    assert all(any("ऀ" <= ch <= "ॿ" for ch in s) for s in sents)


def test_tokenizer_excludes_danda():
    assert tokenize("यह वाक्य है।") == ["यह", "वाक्य", "है"]
    # danda glued to a word must not survive as part of any token
    toks = tokenize("भारत। नया")
    assert "भारत" in toks
    assert all(DANDA not in t for t in toks)


def test_remove_stopwords():
    content = "क्रिकेट"
    assert "अपना" in HINDI_STOPWORDS          # sanity: lexicon loaded
    assert content not in HINDI_STOPWORDS      # sanity: content word is not a stopword
    assert remove_stopwords(["अपना", content, "अपने"]) == [content]


def test_stemmer_examples():
    assert stem("किताबें") == "किताब"          # strips the plural suffix ें
    assert stem("India") == "india"             # non-Devanagari token -> lowercased
    assert stem("भारत") == "भारत"               # no strippable suffix -> unchanged
    # the stemmer must never lengthen a Devanagari word
    for w in ["लड़कियों", "जाएगा", "करते", "किताबें"]:
        assert len(stem(w)) <= len(w)
