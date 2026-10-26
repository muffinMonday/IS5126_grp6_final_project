# IS5126 Group 6: Macro trend analysis and cross-domain moderation on r/jobs

Two linked pipelines (details in `Project Proposal/`):

- **Objective 1, topics:** TF-IDF → PCA (50 dims) → K-means → centroid term profiles → monthly topic trends.
- **Objective 2, toxicity:** pooled labelled datasets → GloVe → Bi-LSTM → zero-shot test on a small hand-labelled r/jobs set (precision, recall, macro F1, PR-AUC).
- **Combined:** apply the classifier to the clustered comments → toxicity rate per topic per month → dashboard.

## Repository layout

```
functions/        shared code (import from here, do not copy it into notebooks)
  preprocessing.py  cleanup_text
  embeddings.py     GloVe download + embedding matrix
  cv.py             kfold_wrapper: k-fold CV, final evaluation, threshold tuning, save/load
  tuning.py         bayes_search: Bayesian hyperparameter search built on kfold_wrapper
notebook/         one notebook per work package, e.g. 2_LSTM_model.ipynb
dataset/          small data files only; see dataset/README.md
glove/            GloVe vectors (not in git, downloaded automatically)
models/           saved search results (not in git)
Project Proposal/ the project specification
```

## Setup

```bash
git clone <repo-url>            # clone OUTSIDE OneDrive: syncing a .git folder can corrupt it
cd <repo>
python -m venv .venv && .venv\Scripts\activate     # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install            # once per clone: commits notebooks without outputs
```

Notebooks add the repo root to the path (`sys.path.append('..')`), so run them from `notebook/`.

Not in git, created locally:
- **GloVe** (`glove/glove.6B.100d.txt`, 347 MB): downloaded on first run of the GloVe cell (needs the 820 MB `glove.6B.zip` once).
- **Datasets:** see `dataset/README.md`.
- **Saved search results** (`models/bayes_search/`, ~33 MB): created by the search cell. Re-run it; do not share it.

## Work packages

| # | Package | Output |
|---|---|---|
| 1 | r/jobs data collection and annotation | comment table + labelled test set |
| 2 | Objective 1: topics | `comment_id, cluster, topic_label` file + trend plots |
| 3 | Pooled training data and baselines | cleaned pooled dataset, baseline scores, evaluation code |
| 4 | Objective 2: Bi-LSTM | `predict_toxicity(texts) -> probabilities`, zero-shot results |
| 5 | Integration, dashboard, report | toxicity rate per topic per month, dashboard |

Agreed interfaces, so work doesn't block:
- **Comment table:** `comment_id, text, created_utc` (CSV or parquet).
- **Topic file:** `comment_id, cluster, topic_label`.
- **Toxicity function:** `predict_toxicity(list_of_texts) -> list_of_probabilities`.
- Until real files exist, develop against small mock files with these columns.

## Git workflow

- `main` is protected: no direct pushes. Work on a branch per task (`feature/<short-name>`) and open a pull request with at least one reviewer.
- Changes to `functions/` always go through a pull request, since everyone depends on it. If you change the behaviour of `cleanup_text`, say so in the pull request: saved models trained on the old cleaning are no longer valid.
- Notebooks: outputs are stripped on commit (nbstripout). Keep each notebook owned by one person to avoid merge conflicts.
- Never commit data, GloVe files, saved models or credentials.
- Small, focused commits with a message that says why, not only what.

## Reproducibility

- Random seeds are fixed (`random_state=42`); keep them so results are comparable between people.
- After editing anything in `functions/`, restart the notebook kernel: a running kernel keeps the old classes.
- A saved search in `models/` is not checked against the data or preprocessing. After changing the data, cleaning, tokenizer or search space, re-run the search.
- The code was developed on pre-release TensorFlow/Keras builds (see `requirements.txt`).
