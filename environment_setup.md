# Environment setup (Code Ocean capsule)

Exact, pinned versions used to develop and verify HinExSum v1.0.0. Configure the
capsule's environment with these so the entrypoint (`run`) is reproducible.

## Base

- **Python 3.12.13** (any CPython 3.9–3.12 works; 3.12 is what was tested).
- OS: Linux x86-64. No GPU required.

## Required packages (core + tests)

These are all the entrypoint (`run`) needs — the pytest suite, the demo summary,
and `evaluate_xlsum.py` with baselines:

```
numpy==2.5.1
scipy==1.18.0
scikit-learn==1.9.0
gensim==4.4.0
smart-open==8.0.0
joblib==1.5.3
threadpoolctl==3.6.0
pytest==9.1.1
pluggy==1.6.0
iniconfig==2.3.0
packaging==26.2
```

Install (Code Ocean "Install packages" / postInstall step):

```bash
pip install \
  numpy==2.5.1 scipy==1.18.0 scikit-learn==1.9.0 \
  gensim==4.4.0 smart-open==8.0.0 joblib==1.5.3 threadpoolctl==3.6.0 \
  pytest==9.1.1
```

`pluggy`, `iniconfig` and `packaging` are pulled in automatically by `pytest`;
`smart-open`, `joblib`, `scipy` and `threadpoolctl` by `gensim`/`scikit-learn`.
Pin them explicitly only if you need a fully frozen image.

Alternatively, install the package itself (which declares the core deps):

```bash
pip install -e .            # gensim, scikit-learn, numpy
pip install -e ".[test]"    # + pytest
```

## Optional packages (not needed by the default `run`)

- **`stanza==1.13.0`** — Part-of-speech features (NVC / proper nouns). The
  entrypoint uses `--no-pos`, so Stanza is not required. To enable POS features
  you must also download the Hindi model once: `python -c "import stanza;
  stanza.download('hi')"`.
- **`datasets==2.21.0`** — Only to *download* the full XL-Sum / ILSUM corpora.
  The capsule ships a 10-document sample (`data/xlsum_hindi_sample.jsonl`), so
  `datasets` is not needed for the default run.

## Data assets

- **`cc.hi.300.bin`** (fastText Common Crawl Hindi, ≈6.5 GiB, 6,958,280,158 bytes)
  must be provided as a data asset mounted at **`/data/cc.hi.300.bin`**. It is
  *not* bundled with the code. Download source:
  <https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.bin.gz>
  (gunzip before use). The entrypoint auto-detects it at `/data/cc.hi.300.bin`.

## Entrypoint

Set the capsule's run command to `run` (or `bash run`). It writes `pytest.txt`,
`demo_summary.txt` and `evaluate_sample.md` to `/results`. Paths are overridable
via `MODEL_PATH`, `SAMPLE_DATA`, `RESULTS_DIR` and `PYTHON` environment variables.
