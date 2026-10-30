# Brief for coding agents

Read this before changing anything. Five people work on this repo in parallel, each with their own agent. Read `README.md` for the project, layout and setup.

## Project in one paragraph
IS5126 group project: toxicity classification of Reddit r/jobs comments plus topic discovery. Objective 1 is unsupervised topics (TF-IDF → PCA → K-means). Objective 2 is a GloVe + Bi-LSTM toxicity classifier trained on pooled external datasets and tested zero-shot on r/jobs. The spec is `Project Proposal/*.pdf`; its metrics are precision, recall, macro F1 and PR-AUC.

## Ground rules
- **Stay in your lane.** Work only on the task you were given and the files it needs. Do not refactor, rename or reformat other people's notebooks or modules.
- **`functions/` is shared.** Prefer adding to it over changing existing behaviour. If you must change existing behaviour (especially `cleanup_text`, the tokenizer-related code, or metric definitions), tell the user explicitly, since it can invalidate other people's results and saved models.
- **Match the surrounding style.** `functions/` uses 2-space indentation and short comments; notebooks use 4-space. Keep comment density similar to the code around.
- **Notebooks:** edit cells with notebook tools, not by hand-editing JSON. Notebook outputs are stripped at commit time (`nbstripout`); do not rely on outputs being in git.
- **Ask before:** downloading large files (GloVe is 820 MB), starting runs that take more than a few minutes (the Bayesian search fits ~180 models), installing or upgrading packages, deleting files, or overwriting `models/`.
- **Never commit** unapproved data, `glove/`, `models/`, `*.keras` or credentials. The approved data exceptions are the existing YouTube sample, the shared Reddit sample and its manifest, and the six approved Part 3 files: `dataset/pooled/pooled_train.csv`, `dataset/pooled/pooled_val.csv`, `dataset/pooled/manifest.json`, `dataset/pooled/baseline_metrics.csv`, `dataset/pooled/loso_metrics.csv`, and `dataset/pooled/transfer_comparison.csv`. Other raw downloads and generated outputs stay ignored. Do not commit or push unless the user asks. Never force-push. Pushing to `main` is allowed when the user asks.
- **Tell the user** about problems you find outside your task instead of silently fixing them.

## Workflow
1. `git pull` before starting work.
2. Make small changes; restart the notebook kernel after editing `functions/` (stale classes are a common source of confusing errors).
3. Verify before saying done: run the cell or a small synthetic test and report what actually ran. State what you did not run.
4. Commit with a message that says why, then push to `main` when the user asks. Changes to `functions/` should be reviewed by a teammate.

## Code map
- `functions/preprocessing.py`: `cleanup_text` (HTML/URL removal, contraction expansion, punctuation → space, keeps `!` and `?`). Output feeds the tokenizer and GloVe lookup.
- `functions/embeddings.py`: `download_glove`, `build_embedding_matrix` (row i = GloVe vector for tokenizer id i; 0 is padding).
- `functions/cv.py`: `kfold_wrapper(model, X_train, y_train, X_val, y_val, ...)`. Takes a compiled, unfitted Keras model, runs stratified k-fold CV on the training set with a fresh clone per fold, then trains a final model on all of `X_train` and scores `X_val`. Exposes `avg_*_metrics`, `std_val_metrics`, `oof_proba`, `hyperparameters`, `tune_threshold`, `evaluate_at`, `save`/`load`.
- `functions/tuning.py`: `bayes_search(build_model, search_space, ...)`. Scikit-optimize ask/tell loop; each candidate is scored with `kfold_wrapper`. `selection='stable'` picks, among near-ties, the lowest fold standard deviation. Search keys `epochs`/`batch_size` go to the wrapper, others to `build_model`.

## Evaluation rules (do not break these)
- **The validation set is never used for training, tuning or choosing the threshold.** Cross-validation runs inside the training split only; the decision threshold is tuned on out-of-fold predictions only (`tune_threshold`) and applied once to validation.
- Known minor leak: the tokenizer is fitted on all comments before the split. Do not make it worse; fix it if asked.
- Inputs at prediction time must go through the same pipeline as training: `cleanup_text` → the same tokenizer → pad/truncate to `max_length`.
- Any change to cleaning, tokenizer, `max_length`, data or the search space means saved models in `models/` are stale: re-run the search.
- Report the proposal's metrics (precision, recall, macro F1, PR-AUC). Hold-out sets are small (about 200 rows), so differences of a few points are within noise; say so rather than over-interpreting.

## Environment notes
- Windows, PowerShell/Git Bash, Python 3.14 on the original machine; paths in notebooks are relative to `notebook/`.
- The original environment uses pre-release TensorFlow/Keras builds. A harmless Keras warning about LSTM masking is filtered in the notebook setup cell; masking itself works.
- `kfold_wrapper` reshapes labels to `(n, 1)` because the Keras `F1Score` metric requires it; keep this.
