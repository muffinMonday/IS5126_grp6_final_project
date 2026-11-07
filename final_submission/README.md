# IS5126 Group 6: toxicity classification and topic discovery on r/jobs

`main_analysis.ipynb` reproduces every table and figure in our video and slides. It has two parts:

- **Part A, topic discovery** (slides 9-24): K-means topics on 10,000 r/jobs comments (January to October 2026), an LDA cross-check, and a comparison with layoffs.fyi.
- **Part B, toxicity classification** (slides 25-34): TF-IDF logistic regression and a GloVe Bi-LSTM, trained on pooled external datasets and tested zero-shot on 100 hand-labelled r/jobs comments.

The notebook is saved with the outputs of the run the slides use, so it can be read without running anything. An appendix at its end maps each slide to the section that produces its numbers.

## What is in this folder

| Item | Purpose |
|---|---|
| `main_analysis.ipynb` | the analysis, saved with its outputs |
| `functions/` | project code the notebook imports (preprocessing, models, tuning, metrics, topic helpers) |
| `scripts/compare_topic_models.py` | stop-word and phrase lists, needed only by the LDA refit (`RETRAIN = True`) |
| `requirements.txt` | Python packages |

The data and the trained models are **not** in this folder, to keep it small. The first cell of the notebook after the settings downloads whatever is missing (74 files, about 96 MB) from our repository, <https://github.com/muffinMonday/IS5126_grp6_final_project>, at a fixed release tag, and checks each file against a SHA-256 hash recorded in the notebook. Nothing is downloaded if the files are already present. If you have the repository folders (`dataset/`, `models/`), copy them next to the notebook and nothing needs to be downloaded.

## How to run

1. We used Python 3.14; other versions were not tested. Create an environment and run `pip install -r requirements.txt`.
2. Start Jupyter from this folder and open `main_analysis.ipynb`. The notebook looks for `functions/` in its own folder and in the folders above it.
3. Run all cells. With the default settings (`RETRAIN = False`) nothing is trained: the saved models are loaded. On our laptop CPU this takes about one minute once the files are downloaded, plus the time to download them.

You can also just read the saved outputs.

## Settings (first code cell)

| Setting | Default | Meaning |
|---|---|---|
| `RETRAIN` | `False` | `False` loads the saved models. `True` refits the logistic regressions, the two Bi-LSTMs, the K-means pipeline and the LDA models and saves them to `models_retrained/` (never over `models/`). It needs `glove/glove.6B.100d.txt` (347 MB; downloaded from Stanford if missing) and takes hours, dominated by the Bayesian searches (30 trials with 5-fold cross-validation for each of four models). A retrained model differs slightly from the one behind the slides. |
| `DOWNLOAD_MISSING` | `True` | fetch missing files from the repository |
| `OVERWRITE_CHANGED` | `False` | by default a local file that differs from the repository is kept and reported, never replaced |
| `FETCH_PARTS` | `{'A', 'B'}` | download only the topic files (`'A'`) or only the toxicity files (`'B'`) |

Everything the notebook computes is written to `outputs/`. It never writes into `dataset/` or `models/`.

## What is and is not reproduced

- Every number in Part B and in the K-means part of Part A is recomputed from the saved models and data. The K-means pipeline is a frozen model whose hash is checked, and it reproduces all 9,770 shipped cluster assignments exactly.
- The LDA and layoff tables are recomputed from the shipped LDA exports and layoff counts, and section A6 compares each regenerated table with the shipped one (14 of 14 match). The LDA *model* in `models/topics_lda/` is a refit that gives different topics from the slides, because refitting LDA under another library version does not give the same solution; no LDA number is taken from it.
- Fit times (slide 33) are not measured: they differ from machine to machine.
- Collecting the Reddit data and pooling the external training datasets are not part of this notebook. The pooled splits and the r/jobs sample are shared files.

## Data notes

- **Reddit comments:** 10,000 r/jobs comments for the topic analysis, and a 100-comment test set (50 toxic, 50 non-toxic) labelled by hand by one team member. They are public comments used for coursework; the test set is a keyword-enriched convenience sample and does not estimate real toxicity rates.
- **layoffs.fyi:** the layoff records are used for one comparison (slide 23).
- **Training data:** four public sources pooled into 19,187 training and 4,797 validation comments. The pooled splits ship with the repository; the raw downloads do not.

## Reproducibility notes

All random seeds are fixed (42 unless stated; the bootstrap uses seed 0). Small numerical differences between machines can come from floating-point order. We ran the notebook once in our repository and once from a folder containing only the code (everything downloaded by the first cell); both runs gave identical tables.
