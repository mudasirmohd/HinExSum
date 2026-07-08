# HinExSum — Hindi Extractive Summariser (Distributional Semantics)

Open reference implementation, adapted to **Hindi**, of

> Mohd, M., Jan, R., & Shah, M. (2020). *Text document summarization using word
> embedding.* **Expert Systems With Applications** 143:112958.
> doi:[10.1016/j.eswa.2019.112958](https://doi.org/10.1016/j.eswa.2019.112958)

Pipeline (Fig. 1 of the paper): preprocessing → big-vector generation via word
embeddings → TF-IDF vectorisation → k-means clustering → per-cluster
feature-based ranking → redundancy elimination + a discourse-coherence rule →
extractive summary. Ships with a **Devanagari-safe ROUGE** module (ROUGE-1/2/L/SU4) and a
**bootstrap evaluation harness** for XL-Sum (Hindi) and ILSUM.

## Install

```bash
pip install -e .                 # HinExSum + core deps (gensim, scikit-learn, numpy)
pip install -e ".[pos,eval]"     # optional: Stanza POS features + datasets for evaluation
```

## Get the Hindi embedding model

HinExSum needs a Hindi word-embedding model. The recommended model is
**fastText Common Crawl** `cc.hi.300.bin` (300-d, handles OOV via subwords).

- **On the Code Ocean capsule** accompanying this software, the model is
  provided under `data/cc.hi.300.bin` — no download required.
- **To run locally**, download it once (≈4.3 GB gzip, ≈6.9 GB unpacked):

  ```bash
  wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.bin.gz
  gunzip cc.hi.300.bin.gz
  ```

  (source: fastText Common Crawl vectors, <https://fasttext.cc/docs/en/crawl-vectors.html>)

Alternatives that also work: AI4Bharat IndicFT (`indicnlp.ft.hi.300.bin`) or any
word2vec `.vec`/`.txt`/`.bin`, or a skip-gram model you train yourself with
gensim (closest to the original paper). Pass whichever file you use via
`--model` / `load_embeddings(path)`.

## Usage

```python
from hinexsum import HindiSummarizer, load_embeddings

model = load_embeddings("cc.hi.300.bin", top_m=5, bv_size=100)   # or data/cc.hi.300.bin
summarizer = HindiSummarizer(model)          # use_pos=False to skip Stanza
result = summarizer.summarize(hindi_text, ratio=0.25)
print(result.summary)
```

Command line (scripts live under `code/`):

```bash
python code/demo.py --model cc.hi.300.bin article.txt --ratio 0.25
python code/demo.py --toy --no-pos           # quick smoke test, trains tiny vectors on the input (testing only)
```

## Evaluation (XL-Sum / ILSUM Hindi)

```bash
python - <<'PY'
from datasets import load_dataset
load_dataset("csebuetnlp/xlsum", "hindi", split="test").to_json("xlsum_hindi_test.jsonl")
PY

python code/evaluate_xlsum.py --model cc.hi.300.bin --data xlsum_hindi_test.jsonl --limit 200
```

`code/evaluate_xlsum.py` reports Pr/Rc/Fs for ROUGE-1/2/L/SU4 and supports fixed-budget
extraction (`--fixed --max-sentences N`), baseline comparison
(`--baselines`), feature/selection ablations (`--ablation`, `--features`,
`--selection`, `--weights`), and validation-tuned evaluation (`--tune`), all with
1000-resample bootstrap confidence intervals. Any JSONL with `text` and
`summary` fields works, so ILSUM (`run_ilsum.py`) or your own data drop in
directly.

## Tests

```bash
pip install -e ".[test]"
pytest code            # or: cd code && pytest
```

## Repository layout (Code Ocean capsule)

```
README.md  LICENSE  CITATION.cff        # project root
pyproject.toml  requirements.txt  environment_setup.md  RELEASE.md
code/                                   # -> /code in the capsule
  run                # executable entrypoint: pytest + demo + sample evaluation -> /results
  hinexsum/
    preprocessing.py # NFC normalisation, danda splitting, stopwords, stemmer
    embeddings.py    # embedding loading + big-vector generation (§3.2)
    ranking.py       # 7 ranking features (§3.4), Stanza POS optional
    summarizer.py    # clustering + selection + redundancy + coherence rule (§3.3–3.4)
    rouge.py         # Devanagari-safe ROUGE-1/2/L/SU4
    data/            # Hindi stopwords, cue phrases, connectives
  demo.py            # summarise a file or the built-in sample
  evaluate_xlsum.py  # evaluation harness (baselines, ablations, tuning, bootstrap)
  run_ilsum.py       # ILSUM (FIRE) Hindi evaluation
  tests/             # pytest unit tests
  data/
    xlsum_hindi_sample.jsonl   # 10-doc sample for the entrypoint (CC BY-NC-SA; see data/README.md)
```

In a Code Ocean capsule the embedding model is provided as a data asset at
`/data/cc.hi.300.bin`, and the entrypoint writes its outputs to `/results`.
Run it with `code/run` (see `environment_setup.md` for the pinned environment).

## Key hyperparameters

- `top_m` (default 5): similar words concatenated per token in the big-vector.
- `bv_size` (default 100): fixed big-vector length (pad/truncate), the paper's *n*.
- `n_clusters`: k for k-means; defaults to ⌈√(#sentences)⌉.
- `redundancy_eps` (0.02): score-difference threshold under which two
  same-cluster sentences are treated as paraphrases and deduplicated.

## Citing

If you use HinExSum, please cite the software (see `CITATION.cff`) and the
underlying method paper (Mohd, Jan & Shah 2020, above).

## Licence

MIT — see `LICENSE`.
