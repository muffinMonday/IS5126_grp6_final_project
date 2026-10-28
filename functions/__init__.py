"""Project helpers, split by responsibility:

preprocessing  text cleaning (cleanup_text)
embeddings     GloVe download and embedding matrix
cv             kfold_wrapper: k-fold cross-validation + final evaluation
tuning         bayes_search: Bayesian hyperparameter search built on kfold_wrapper
metrics        toxicity_metrics, metrics_by_group: the proposal's evaluation metrics
pooled_data    prepare_pooled_data: pooled train / validation split of the four sources
baselines      run_baselines: TF-IDF baselines and leave-one-source-out check
"""
