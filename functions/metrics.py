"""Evaluation metrics shared by every model in the project.

The proposal's metrics are precision, recall, macro F1 and PR-AUC. They are
computed the same way as in kfold_wrapper, so baseline, Bi-LSTM and zero-shot
r/jobs results can be compared directly.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
  accuracy_score,
  average_precision_score,
  f1_score,
  precision_score,
  recall_score,
  roc_auc_score,
)


def toxicity_metrics(y_true, proba, threshold=0.5):
  """Metrics for binary toxic (1) / non-toxic (0) predictions.

  `proba` is the predicted probability of toxic; labels are predicted as toxic
  when proba >= threshold. PR-AUC and ROC-AUC do not depend on the threshold.
  `prevalence` (share of toxic rows) is the PR-AUC of a random guess, so read
  PR-AUC against it. With only one class present the AUCs are NaN."""
  y = np.asarray(y_true).astype(int).ravel()
  p = np.asarray(proba, dtype=float).ravel()
  pred = (p >= threshold).astype(int)
  both_classes = len(np.unique(y)) == 2
  return {
    'n': len(y),
    'prevalence': y.mean() if len(y) else np.nan,
    'precision': precision_score(y, pred, zero_division=0),
    'recall': recall_score(y, pred, zero_division=0),
    'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
    'pr_auc': average_precision_score(y, p) if both_classes else np.nan,
    'roc_auc': roc_auc_score(y, p) if both_classes else np.nan,
    'accuracy': accuracy_score(y, pred),
    'flagged': pred.mean() if len(y) else np.nan,
  }


def metrics_by_group(y_true, proba, groups, threshold=0.5):
  """toxicity_metrics for every group (e.g. data source), plus an 'all' row."""
  y = np.asarray(y_true).astype(int).ravel()
  p = np.asarray(proba, dtype=float).ravel()
  g = np.asarray(groups).ravel()

  rows = {'all': toxicity_metrics(y, p, threshold)}
  for name in sorted(set(g)):
    mask = g == name
    rows[name] = toxicity_metrics(y[mask], p[mask], threshold)
  return pd.DataFrame(rows).T.rename_axis('group')


def paired_bootstrap(y_true, proba_a, proba_b, n_boot=1000, seed=0, threshold=0.5, level=0.95):
  """Paired bootstrap of the metric differences between two models (a minus b).

  Resamples the comments with replacement, the same comments for both models, and
  recomputes recall, false positive rate (FPR), macro F1 and PR-AUC each time.
  Returns one row per metric: the `difference` on the full sample, the bootstrap
  `mean`, and the percentile interval `ci_low`/`ci_high` at `level`. If the interval
  includes zero, the sample cannot tell the models apart on that metric. Resamples
  with a single class are skipped. The comments are treated as independent."""
  y = np.asarray(y_true).astype(int).ravel()
  pa = np.asarray(proba_a, dtype=float).ravel()
  pb = np.asarray(proba_b, dtype=float).ravel()
  if not len(y) == len(pa) == len(pb):
    raise ValueError('y_true and both probability arrays must have the same length.')

  def scores(idx):
    out = {}
    for name, p in (('a', pa), ('b', pb)):
      yy, pp = y[idx], p[idx]
      pred = (pp >= threshold).astype(int)
      out[name] = {
        'recall': recall_score(yy, pred, zero_division=0),
        'FPR': ((pred == 1) & (yy == 0)).sum() / (yy == 0).sum(),
        'macro_f1': f1_score(yy, pred, average='macro', zero_division=0),
        'pr_auc': average_precision_score(yy, pp),
      }
    return {k: out['a'][k] - out['b'][k] for k in out['a']}

  observed = scores(np.arange(len(y)))
  rng = np.random.default_rng(seed)
  draws = {k: [] for k in observed}
  for _ in range(n_boot):
    idx = rng.choice(len(y), len(y))
    if len(np.unique(y[idx])) < 2:
      continue
    for k, v in scores(idx).items():
      draws[k].append(v)
  tail = (1 - level) / 2 * 100
  return pd.DataFrame({
    k: {'difference': observed[k], 'mean': np.mean(v),
        'ci_low': np.percentile(v, tail), 'ci_high': np.percentile(v, 100 - tail)}
    for k, v in draws.items()
  }).T.rename_axis('metric')
