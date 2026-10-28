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
