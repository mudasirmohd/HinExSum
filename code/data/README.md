# Bundled sample data

## `xlsum_hindi_sample.jsonl`

A **10-document sample** of the Hindi test split of **XL-Sum**, included so the
capsule entrypoint (`../run`) can exercise the evaluation harness without
downloading the full corpus. Each line is a JSON object with `id`, `url`,
`title`, `text` (article) and `summary` (reference) fields.

### Source and licence

The sample is drawn from the XL-Sum dataset
(<https://huggingface.co/datasets/csebuetnlp/xlsum>), which is released under the
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)** licence: <https://creativecommons.org/licenses/by-nc-sa/4.0/>.

This 10-document excerpt is redistributed here under the same CC BY-NC-SA 4.0
terms, for **non-commercial research and reproducibility** only. The underlying
news articles remain the property of their original publishers.

> Note: XL-Sum's non-commercial licence differs from HinExSum's own MIT licence,
> which covers the *software* only, not this bundled data sample.

### Citation

If you use this data, please cite the XL-Sum paper:

```bibtex
@inproceedings{hasan-etal-2021-xl-sum,
  title     = "{XL}-{S}um: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
  author    = "Hasan, Tahmid and Bhattacharjee, Abhik and Islam, Md. Saiful and
               Mubasshir, Kazi and Li, Yuan-Fang and Kang, Yong-Bin and
               Rahman, M. Sohel and Shahriyar, Rifat",
  booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
  year      = "2021",
  publisher = "Association for Computational Linguistics"
}
```

(Please verify the exact page numbers / Anthology ID against the ACL Anthology
when finalising a manuscript.)
