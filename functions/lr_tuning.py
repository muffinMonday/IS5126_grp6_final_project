"""Train-only CV tuning for the two Part 3 logistic regressions."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .baselines import build_tfidf_logreg


def tune_logistic_regressions(texts, labels, c_values=(0.01, 0.1, 1, 10, 100),
                             n_splits=5, random_state=42, n_jobs=1,
                             ngram_ranges=((1, 1), (1, 2)), min_dfs=(1, 2, 5),
                             selection='best'):
  """Return fitted pipelines and CV logs; accepts training data only.

  Both variants use identical stratified folds and AP scoring. Jointly search
  C and TF-IDF settings, selecting the highest mean AP by default. Optional
  'stable' retains the earlier LSTM-style rule. Exact ties keep the first grid
  candidate. TF-IDF is fitted inside each fold. No threshold is tuned.
  """
  texts = np.asarray(texts)
  labels = np.asarray(labels)
  if texts.ndim != 1 or labels.ndim != 1 or len(texts) != len(labels):
    raise ValueError('Expected equally sized one-dimensional texts and labels.')
  if not len(texts) or not all(isinstance(t, str) and t.strip() for t in texts):
    raise ValueError('Training texts must be nonempty strings.')
  if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
    raise ValueError('Training labels must contain both binary classes.')
  if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
    raise ValueError('n_splits must be an integer of at least 2.')
  if np.bincount(labels.astype(int)).min() < n_splits:
    raise ValueError('Each class needs at least n_splits training rows.')
  c_values = np.asarray(c_values, dtype=float)
  if c_values.ndim != 1 or not len(c_values) or not np.isfinite(c_values).all() or (c_values <= 0).any():
    raise ValueError('C values must be finite positive numbers.')
  if selection not in ('best', 'stable'):
    raise ValueError("selection must be 'best' or 'stable'.")
  ngram_ranges = [tuple(value) for value in ngram_ranges]
  min_dfs = list(min_dfs)
  if not ngram_ranges or any(len(v) != 2 or any(not isinstance(x, int) or isinstance(x, bool) for x in v)
                             or not 1 <= v[0] <= v[1] for v in ngram_ranges):
    raise ValueError('ngram_ranges must contain positive ordered integer pairs.')
  if not min_dfs or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in min_dfs):
    raise ValueError('min_dfs must contain positive integer document counts.')

  cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=random_state).split(texts, labels))

  def select_stable(results):
    mean = np.asarray(results['mean_test_score'])
    # GridSearchCV reports population SD; the LSTM wrapper uses sample SD.
    std = np.asarray(results['std_test_score']) * np.sqrt(n_splits / (n_splits - 1))
    best = int(np.argmax(mean))
    eligible = np.flatnonzero(mean >= mean[best] - std[best])
    return int(eligible[np.argmin(std[eligible])])

  grid = {'logisticregression__C': c_values.tolist(),
          'tfidfvectorizer__ngram_range': ngram_ranges,
          'tfidfvectorizer__min_df': min_dfs}
  models, logs, selected, selected_params = {}, [], {}, {}
  for name, weight in [('tfidf_logreg', None), ('tfidf_logreg_balanced', 'balanced')]:
    search = GridSearchCV(
      build_tfidf_logreg(weight, random_state),
      grid, scoring='average_precision',
      cv=cv, refit=select_stable if selection == 'stable' else True,
      n_jobs=n_jobs, error_score='raise',
    ).fit(texts, labels)
    models[name] = search.best_estimator_
    selected[name] = float(search.best_params_['logisticregression__C'])
    selected_params[name] = dict(search.best_params_)
    params = search.cv_results_['params']
    log = pd.DataFrame({
      'model': name, 'C': [p['logisticregression__C'] for p in params],
      'ngram_range': [p['tfidfvectorizer__ngram_range'] for p in params],
      'min_df': [p['tfidfvectorizer__min_df'] for p in params],
      'mean_cv_pr_auc': search.cv_results_['mean_test_score'],
      'std_cv_pr_auc': search.cv_results_['std_test_score'] * np.sqrt(n_splits / (n_splits - 1)),
      'selected': np.arange(len(params)) == search.best_index_,
    })
    for fold in range(n_splits):
      log[f'fold_{fold + 1}_pr_auc'] = search.cv_results_[f'split{fold}_test_score']
    logs.append(log)
  return {'models': models, 'cv_results': pd.concat(logs, ignore_index=True),
          'config': {'C_grid': c_values.tolist(), 'selected_C': selected,
                     'ngram_ranges': ngram_ranges, 'min_dfs': min_dfs,
                     'selected_params': selected_params,
                     'candidates_per_variant': len(c_values) * len(ngram_ranges) * len(min_dfs),
                     'n_splits': n_splits, 'random_state': random_state,
                     'objective': 'average_precision', 'selection': selection,
                     'threshold': 0.5, 'tfidf_fit': 'inside each training fold',
                     'class_weights': {'tfidf_logreg': None, 'tfidf_logreg_balanced': 'balanced'}}}
