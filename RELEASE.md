# Releasing HinExSum v1.0.0 and minting a Zenodo DOI

This guide tags the `v1.0.0` release and archives it on Zenodo to obtain a
citable DOI for the Software Impacts submission.

## 1. Pre-release checklist

- [ ] `pytest` passes (`pip install -e ".[test]" && pytest`).
- [ ] Version string is `1.0.0` and consistent in: `pyproject.toml`,
      `hinexsum/__init__.py` (`__version__`), and `CITATION.cff` (`version`).
- [ ] `CITATION.cff`: set `date-released` to today's date; fill `repository-code`.
- [ ] `LICENSE` year and author are correct (2026, Dr. Mudasir Mohd).
- [ ] `README.md` install/usage commands run against a clean checkout.
- [ ] Large artefacts (the `cc.hi.300.bin` model, `*.jsonl` corpora) are **not**
      committed — keep them out with a `.gitignore` (they belong in the Code
      Ocean capsule / are downloaded per `README.md`).

Suggested `.gitignore`:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
cc.hi.300.bin
cc.hi.300.bin.gz
*.jsonl
*.log
```

## 2. Initialise git and tag the release

```bash
cd hinexsum-repo
git init
git add .
git commit -m "HinExSum v1.0.0: Hindi extractive summariser (reference implementation)"
git branch -M main
git remote add origin https://github.com/mudasirmohd/HinExSum.git
git push -u origin main

git tag -a v1.0.0 -m "HinExSum v1.0.0"
git push origin v1.0.0
```

## 3. Archive on Zenodo for a DOI

**Option A — GitHub ↔ Zenodo integration (recommended):**

1. Sign in at <https://zenodo.org> with the GitHub account.
2. Zenodo → *Settings → GitHub*, and toggle the `HinExSum` repository **On**.
   (Do this *before* publishing the GitHub release.)
3. On GitHub, publish a **Release** from the `v1.0.0` tag
   (*Releases → Draft a new release → choose `v1.0.0` → Publish*).
4. Zenodo automatically captures the release and issues a DOI. Zenodo mints two
   DOIs: a **concept DOI** (all versions) and a **version DOI** (this release);
   cite the concept DOI for "the software" in general.

**Option B — manual upload:** create a `.zip` of the tagged source
(`git archive --format=zip -o HinExSum-v1.0.0.zip v1.0.0`), then Zenodo →
*New upload*, set *Upload type = Software*, add title/author/licence, and publish.

## 4. After the DOI is issued

- [ ] Add the Zenodo **concept DOI** to `CITATION.cff` (uncomment the `doi:` line
      and/or add an `identifiers:` entry) and, optionally, a DOI badge to `README.md`.
- [ ] Fill the Code Metadata table in `osp_draft.md`: the *Permanent link to
      code/repository* (Zenodo record URL) and the *Permanent link to Reproducible
      Capsule* (Code Ocean capsule DOI/URL).
- [ ] Verify the Zenodo record renders `CITATION.cff` correctly (Zenodo reads it).

## Version bumps for future releases

Update the version in `pyproject.toml`, `hinexsum/__init__.py`, and
`CITATION.cff` together, tag `vX.Y.Z`, and publish a new GitHub release; the
Zenodo integration issues a fresh version DOI automatically.
