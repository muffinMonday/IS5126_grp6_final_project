# IS5126 Group 6: Macro trend analysis and cross-domain moderation on r/jobs

## Start here: `final_submission/main_analysis.ipynb`

`final_submission/main_analysis.ipynb` is the combined analysis for the course submission. It reproduces every table and figure in our slides and video in one run, from the saved models and shared data, and it is saved with its outputs. [Set-up steps](#set-up-and-run-main_analysisipynb) are below; `final_submission/README.md` is the short version that travels with the submission.

**The notebooks in `notebook/` are working files.** Each of us used them while building an individual part: topics (`2_topic_discovery`), LDA (`2b_lda_topics`), layoffs (`2d_layoffs_comparison`), pooled data and baselines (`3_pooled_data`), the Bi-LSTM (`4_LSTM_model`) and the model comparison (`4b_LSTM_Logistic_Comparison`). They stay in the repository as the record of how each part was built, and for rerunning one part on its own. The combined output is `main_analysis.ipynb`: if a working notebook and `main_analysis.ipynb` ever disagree, `main_analysis.ipynb` is the version of record.

## Project in brief

Two linked pipelines (details in `Project Proposal/`):

- **Objective 1, topics:** TF-IDF → PCA (100 dims, row-normalised) → K-means (18 clusters) → centroid term profiles → monthly topic trends, cross-checked with LDA and compared with layoffs.fyi.
- **Objective 2, toxicity:** pooled labelled datasets → GloVe → Bi-LSTM → zero-shot test on a small hand-labelled r/jobs set (precision, recall, macro F1, PR-AUC).
- **Combined:** apply the classifier to the clustered comments → toxicity rate per topic per month → dashboard.

## Repository layout

```
functions/        shared code (import from here, do not copy it into notebooks)
  preprocessing.py  cleanup_text
  embeddings.py     GloVe download + embedding matrix
  cv.py             kfold_wrapper: k-fold CV, final evaluation, threshold tuning, save/load
  tuning.py         bayes_search: Bayesian hyperparameter search built on kfold_wrapper
  metrics.py        toxicity_metrics / metrics_by_group: precision, recall, macro F1, PR-AUC
  pooled_data.py    prepare_pooled_data: load, sample, clean, deduplicate and split the four training sources
  baselines.py      run_baselines: TF-IDF + logistic regression baselines, leave-one-source-out check
  diagnostics.py    numbers behind the transfer, triage and cost analysis (slides 30-33)
  reddit_collection.py  OAuth collection, annotation preparation, balanced test-set sampling
scripts/          command-line entry points (collect_reddit.py, run_part3.py for a fresh kernel)
tests/            offline checks: python -m unittest discover -s tests
final_submission/ the submission: main_analysis.ipynb (combined analysis, saved with its outputs), README.md, requirements.txt
notebook/         working notebooks, one per work package, e.g. 4_LSTM_model.ipynb (not the submission)
dataset/          approved shared data and dataset instructions; see dataset/README.md
glove/            GloVe vectors (not in git, downloaded automatically)
models/           final saved models and the tokenizer, loaded by the notebooks (in git, see Git workflow)
outputs/          tables and figures written by main_analysis.ipynb (not in git)
models_retrained/ written by main_analysis.ipynb only when RETRAIN = True (not in git)
Project Proposal/ the project specification
```

## Set up and run `main_analysis.ipynb`

1. **Get the code.** Clone the repository (outside OneDrive: syncing a `.git` folder can corrupt it), or unpack the submission folder. Either way the notebook needs `functions/` next to it or in a folder above it.
2. **Create an environment.** We used Python 3.14 on Windows, CPU only; other versions were not tested.
   ```bash
   python -m venv .venv && .venv\Scripts\activate     # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt                    # or final_submission/requirements.txt (the submission's copy), which also adds `notebook`
   ```
   `requirements.txt` installs `ipykernel`, so VS Code can run the notebook directly. For a browser interface, also run `pip install notebook` and start `jupyter notebook`.
3. **Open and run.** Open `final_submission/main_analysis.ipynb`, choose the kernel of the environment above and run all cells. It does not matter which folder Jupyter was started from: the notebook looks for `functions/` in its own folder and then in the folders above it.
4. **Files are checked, and fetched if missing.** The cell after the settings lists every data and model file the notebook reads, with its SHA-256 hash. In a full clone they are already in git, so nothing is downloaded. In a light package (code only) the cell downloads the missing files, about 96 MB, from <https://github.com/muffinMonday/IS5126_grp6_final_project> at a fixed release tag, so it needs internet access and the repository must be public. A local file whose hash differs is kept and reported, never overwritten; set `OVERWRITE_CHANGED = True` to replace it, or `DOWNLOAD_MISSING = False` to forbid downloads.
5. **What you get.** With the default `RETRAIN = False` nothing is trained: the saved models are loaded and the full run takes about a minute on our laptop (CPU). Tables and figures that the notebook regenerates go to `outputs/`; the notebook never writes into `dataset/` or `models/`.

Settings are in the first code cell (`RETRAIN`, `DOWNLOAD_MISSING`, `OVERWRITE_CHANGED`, `FETCH_PARTS`). `RETRAIN = True` refits every model, writes the new ones to `models_retrained/` and takes hours; it needs GloVe (`glove/glove.6B.100d.txt`, 347 MB, downloaded if missing). Leave it `False` unless you want to retrain.

Some messages are harmless: TensorFlow's oneDNN notice, a Keras line about skipping optimizer variables when it loads a saved model, and scikit-learn's version warning for the K-means model (it was fitted on an older release and still reproduces its assignments exactly).

If a cell fails: `Open this notebook from the project folder` means `functions/` was not found; `ModuleNotFoundError` means the environment is missing a package; a hash mismatch for a downloaded file means the download was corrupted or the repository changed, so run the cell again.

`main_analysis.ipynb` carries a `keep_output` flag in its metadata, so `nbstripout` keeps its saved outputs when it is committed. The working notebooks do not: their outputs are stripped.

## Setup for the working notebooks

```bash
git clone <repo-url>            # clone OUTSIDE OneDrive: syncing a .git folder can corrupt it
cd <repo>
python -m venv .venv && .venv\Scripts\activate     # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install            # once per clone: commits notebooks without outputs
```

To run notebook 3 from the original data, also install [Git LFS](https://git-lfs.com/),
then run `git lfs install` and `git lfs pull` in this repository. The Civil Comments
training ZIP uses LFS (about 273 MiB); extract its `train.csv` before running
notebook 3. The other three original inputs are in ordinary Git. See
`dataset/README.md` for verification and setup details.

The working notebooks add the repo root to the path (`sys.path.append('..')`), so run them from `notebook/`. (`main_analysis.ipynb` finds the project folder itself.)

Not in git, created locally:
- **GloVe** (`glove/glove.6B.100d.txt`, 347 MB): downloaded on first run of the GloVe cell (needs the 820 MB `glove.6B.zip` once).
- **Other datasets and generated outputs:** see `dataset/README.md`; the four approved Part 3 source inputs and the prepared handoff are tracked.

The final saved models are in git, so notebook 4b runs without training: `models/baselines/` and `models/tokenizer/` (written by notebooks 3 and 4), and `models/bayes_search_pooled_balanced/` and `models/bayes_search_pooled_unweighted/` (about 25 MB each, written by the search in notebook 4). Regenerate them by re-running notebooks 3 and 4 when the data, cleaning, tokenizer or search space change. `main_analysis.ipynb` also loads `models/topics_kmeans/` (the frozen K-means pipeline) and `models/topics_lda/` (a refit LDA; no number in the analysis comes from it, see its `manifest.json`).

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

## Work package 2: Ready-to-use topic inputs

`notebook/2_topic_discovery.ipynb` now defaults to the complete
[`dataset/monthly_Topics/kmeans/`](dataset/monthly_Topics/kmeans/README.md)
bundle: raw candidates, cleaned data, sampling metadata, the final PCA100/L2/
k=18 pipeline and reference comparison tables. It reproduces the named analyses
offline from a fresh checkout. Use `requirements-objective1.txt` for a separate
analytical environment. New run outputs and intermediate models remain ignored.

## Work package 3: Ready-to-use training data

The approved `dataset/pooled/pooled_train.csv` (19,187 rows),
`pooled_val.csv` (4,797 rows) and `manifest.json` are tracked for package 4.
Use `text` as input and `label` as target; fit the tokenizer on training data only.
Three compact Part 3 score summaries (`baseline_metrics.csv`, `loso_metrics.csv`,
`transfer_comparison.csv`) are also tracked as reference results. See the
[dataset guide](dataset/README.md#pooled-training-data) for loading, version checks
and the distinction between training inputs and reference outputs.

## Work package 1: Reddit test comments

The committed `dataset/reddit/test_100_labeled.csv` contains 100 real comments
from 58 r/jobs posts, with 50 non-toxic and 50 toxic **hand labels**, ready for
zero-shot evaluation in `notebook/4b_LSTM_Logistic_Comparison.ipynb`. The hand labels are the reference
labels for this test set (an AI draft was corrected by hand before the first commit;
`ai_draft_rationale` keeps the draft's reasons). `collection_manifest.json` records acquisition and sampling
limitations. These two files are an explicitly approved exception to the data
exclusion rule. Other collection artifacts stay local and ignored. See the
[dataset guide](dataset/README.md#shared-hand-labelled-sample-2026-10-28) for usage.

The small test set contains **50 non-toxic (`is_toxic=0`) and 50 toxic
(`is_toxic=1`) r/jobs comments**. These are toxicity labels, not positive/negative
sentiment. Collect a larger candidate pool, obtain contextual AI annotations,
then export exactly 50 comments from each class with seed 42. This is a balanced
zero-shot test set, not a representative sample for estimating monthly toxicity.

From the repository root (only Python's standard library is needed):

```bash
python scripts/collect_reddit.py collect --limit 500
# Prepare AI label/reason JSON following dataset/README.md; use a new output name.
python scripts/collect_reddit.py annotate-ai --input dataset/reddit/candidates.csv --annotations dataset/reddit/ai_annotations_100.json --output dataset/reddit/test_100_ai_labeled_v2.csv
python -m unittest discover -s tests -p 'test_reddit_collection.py'
```

For this academic project, Reddit's current access guidance directs researchers
to Reddit for Researchers (BigQuery); `import-csv` accepts a permitted export.
The API collector requires OAuth credentials and specific permission covering
this use, rather than a developer app alone. See
[the data collection and annotation guide](dataset/README.md#reddit-collection-and-annotation)
for credentials, CSV columns, annotation rules and commands. Generated files
default to `dataset/reddit/`; only the two approved sample files are tracked.
Existing files are never overwritten. This workflow does not train models or
change preprocessing.

## Git workflow

- Pushing directly to `main` is allowed. `git pull` first, and never force-push.
- Changes to `functions/` affect everyone, so ask a teammate to review them. If you change the behaviour of `cleanup_text`, tell the team: saved models trained on the old cleaning are no longer valid.
- Notebooks: outputs are stripped on commit (nbstripout). Keep each notebook owned by one person to avoid merge conflicts.
- Keep data, GloVe files and credentials out of Git. The explicitly
  approved exceptions are the existing YouTube sample, the two Reddit sample
  files and the three Part 3 handoff files listed above. Other raw downloads,
  predictions, error tables and figures stay ignored.
- Saved models are the other exception: commit only the final folders listed under Setup (`baselines`, `tokenizer`, the two `bayes_search_pooled_*` folders, `topics_kmeans` and `topics_lda`). Replace them when the data, cleaning, tokenizer or search space change, and do not add a folder per retrain, because every `.keras` file (about 25 MB) stays in the Git history.
- Small, focused commits with a message that says why, not only what.

## Reproducibility

- Random seeds are fixed (`random_state=42`); keep them so results are comparable between people.
- After editing anything in `functions/`, restart the notebook kernel: a running kernel keeps the old classes.
- A saved search in `models/` is not checked against the data or preprocessing. After changing the data, cleaning, tokenizer or search space, re-run the search and replace the committed models.
- The code was developed on pre-release TensorFlow/Keras builds (see `requirements.txt`).
