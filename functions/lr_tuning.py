"""Train-only CV tuning for the two Part 3 logistic regressions."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .baselines import build_tfidf_logreg


def tune_logistic_regressions(texts, labels, c_values=(0.01, 0.1, 1, 10, 100),
                             n_splits=5, random_state=42, n_jobs=1):
  """Return fitted pipelines and CV logs; accepts training data only.

  Both variants use identical stratified folds and AP scoring. As in the LSTM
  stable selection, prefer the smallest fold SD among candidates within one
  fold SD of the highest mean. Ties retain the first C in the supplied grid.
  TF-IDF is fitted inside each fold. No threshold is tuned.
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

  cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=random_state).split(texts, labels))

  def select_stable(results):
    mean = np.asarray(results['mean_test_score'])
    # GridSearchCV reports population SD; the LSTM wrapper uses sample SD.
    std = np.asarray(results['std_test_score']) * np.sqrt(n_splits / (n_splits - 1))
    best = int(np.argmax(mean))
    eligible = np.flatnonzero(mean >= mean[best] - std[best])
    return int(eligible[np.argmin(std[eligible])])

  models, logs, selected = {}, [], {}
  for name, weight in [('tfidf_logreg', None), ('tfidf_logreg_balanced', 'balanced')]:
    search = GridSearchCV(
      build_tfidf_logreg(weight, random_state),
      {'logisticregression__C': c_values.tolist()}, scoring='average_precision',
      cv=cv, refit=select_stable, n_jobs=n_jobs, error_score='raise',
    ).fit(texts, labels)
    models[name] = search.best_estimator_
    selected[name] = float(search.best_params_['logisticregression__C'])
    log = pd.DataFrame({
      'model': name, 'C': c_values,
      'mean_cv_pr_auc': search.cv_results_['mean_test_score'],
      'std_cv_pr_auc': search.cv_results_['std_test_score'] * np.sqrt(n_splits / (n_splits - 1)),
      'selected': np.arange(len(c_values)) == search.best_index_,
    })
    for fold in range(n_splits):
      log[f'fold_{fold + 1}_pr_auc'] = search.cv_results_[f'split{fold}_test_score']
    logs.append(log)
  return {'models': models, 'cv_results': pd.concat(logs, ignore_index=True),
          'config': {'C_grid': c_values.tolist(), 'selected_C': selected,
                     'n_splits': n_splits, 'random_state': random_state,
                     'objective': 'average_precision', 'selection': 'stable',
                     'threshold': 0.5, 'tfidf_fit': 'inside each training fold',
                     'class_weights': {'tfidf_logreg': None, 'tfidf_logreg_balanced': 'balanced'}}}
