# Frozen LDA and layoff handoff

These outputs and the related `dataset/reddit/layoff_mentions/` and
`dataset/layoffs/layoffs.csv` were approved for Git sharing by the owner of
notebooks 2b/2d on 2026-11-07. Use the frozen files to reproduce the existing
analyses without downloading data or refitting LDA.

Run notebooks from `notebook/`. In 2b, section 9 reads the frozen LDA exports;
in 2d, section 7 reads the frozen archive counts, external records and LDA shares.
Earlier sections of 2b fit LDA and should only be run when a new fit is wanted.
`lda_run.json` records the training parameters and SHA-256 of `comments_clean.csv`.
The six top-level LDA CSV/JSON exports include individual comment weights,
profiles, monthly shares, trend summaries and the K-means/LDA comparison.
`analysis/` contains presentation tables; `figures/` contains their plots.

The missing `functions/lda_topics.py` and `functions/layoff_mentions.py` were
restored from local Python 3.14 bytecode. Their behavior was compared with those
original cached modules on synthetic inputs and all archived match texts.
No existing preprocessing, tokenizer or toxicity metrics were changed.

The original live downloader source could not be recovered.
`scripts/collect_layoff_mentions.py` is a replacement **offline** entry point:
it reconstructs numerator counts from `matches/*.json`, validates exact-match
flags and compares all counts with `monthly_counts.csv`. Monthly population
denominators remain the original archived values; they cannot be independently
reconstructed from keyword matches. This script does not claim to reproduce
live API collection, and the retained `run.json` describes the original run.

```bash
python scripts/collect_layoff_mentions.py
python -m unittest discover -s tests -p 'test_objective1_handoff.py'
```

The external CSV is the original local `swaptr/layoffs-2022` Kaggle mirror
snapshot (local cache version 485). No assertion about its redistribution
license has been added; the existing notebooks document the source.
