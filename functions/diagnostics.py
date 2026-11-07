"""Numbers behind the transfer, triage and cost slides (30-33).

Everything takes plain arrays or the results tables of notebook 4b, never a model,
so it can be tested without TensorFlow. The results tables have one row per model
and the columns 'recall (TPR)', 'FPR', 'roc_auc' (as built by `results_table` in 4b).
"""

import re
import time

import numpy as np
import pandas as pd

# profanity as used on slide 30: matched on the lower-cased original comment
PROFANITY = re.compile(r'\b(fuck\w*|shit\w*|damn\w*)')


def has_profanity(texts):
  """True for each comment that contains a word starting with fuck, shit or damn."""
  return np.array([bool(PROFANITY.search(str(text).lower())) for text in texts], dtype=bool)


def token_length_summary(n_tokens, max_length=80):
  """Mean, median and the share of comments longer than the Bi-LSTM input length."""
  n_tokens = np.asarray(n_tokens)
  return pd.Series({'comments': len(n_tokens), 'mean': n_tokens.mean(), 'median': np.median(n_tokens),
                    f'over_{max_length}': int((n_tokens > max_length).sum()),
                    f'share_over_{max_length}': float((n_tokens > max_length).mean())})


def flag_breakdown(y, proba_by_model, profane, n_tokens, max_length=80, threshold=0.5):
  """Where each model errs: false alarms on profane / clean non-toxic comments, misses on long / short toxic ones.

  Each cell is 'wrong / group size', e.g. 9 of the 14 non-toxic profane comments flagged.
  """
  y, profane, n_tokens = np.asarray(y).astype(int), np.asarray(profane, bool), np.asarray(n_tokens)
  nontoxic, toxic, long_ = y == 0, y == 1, n_tokens > max_length
  groups = {
    'non-toxic, profane: flagged': (nontoxic & profane, True),
    'non-toxic, no profanity: flagged': (nontoxic & ~profane, True),
    f'toxic, over {max_length} tokens: missed': (toxic & long_, False),
    f'toxic, up to {max_length} tokens: missed': (toxic & ~long_, False),
  }
  rows = {}
  for name, proba in proba_by_model.items():
    flagged = np.asarray(proba) >= threshold
    rows[name] = {label: (int(((flagged if is_flag else ~flagged) & mask).sum()), int(mask.sum()))
                  for label, (mask, is_flag) in groups.items()}
  counts = pd.DataFrame(rows).T
  return counts


def transfer_summary(val_results, other_results):
  """Recall, false positive rate and ROC-AUC on the validation split against another dataset.

  `fpr_ratio` is the other FPR divided by the validation FPR (how many times more false alarms);
  `roc_auc_drop` is validation minus other.
  """
  out = pd.DataFrame({
    'recall_val': val_results['recall (TPR)'], 'recall_other': other_results['recall (TPR)'],
    'fpr_val': val_results['FPR'], 'fpr_other': other_results['FPR'],
    'roc_auc_val': val_results['roc_auc'], 'roc_auc_other': other_results['roc_auc'],
  }).astype(float)
  out['recall_change'] = out['recall_other'] - out['recall_val']
  out['fpr_ratio'] = out['fpr_other'] / out['fpr_val']
  out['roc_auc_drop'] = out['roc_auc_val'] - out['roc_auc_other']
  return out


def triage_at_share(recall, fpr, share=0.05):
  """Precision, share of all comments flagged and share of toxic caught when `share` of comments are toxic.

  Recall and FPR are kept as measured, so only the mix of comments changes.
  """
  recall, fpr = np.asarray(recall, float), np.asarray(fpr, float)
  precision = recall * share / (recall * share + fpr * (1 - share))
  flagged = fpr + share * (recall - fpr)
  return precision, flagged, recall


def triage_table(val_results, other_results, share=0.05, other_name='r/jobs'):
  """`triage_at_share` for each model, from the validation rates and from another dataset's rates."""
  columns = {}
  for label, results in (('validation', val_results), (other_name, other_results)):
    precision, flagged, caught = triage_at_share(results['recall (TPR)'].astype(float), results['FPR'].astype(float), share)
    columns[(label, 'precision')] = precision
    columns[(label, 'queue share')] = flagged
    columns[(label, 'toxic caught')] = caught
  return pd.DataFrame(columns, index=val_results.index)


def cutoff_table(y, proba_by_model, cutoffs=(0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)):
  """Precision, recall and share flagged at each decision cutoff, one row per (model, cutoff)."""
  y = np.asarray(y).astype(int)
  rows = []
  for name, proba in proba_by_model.items():
    proba = np.asarray(proba)
    for cutoff in cutoffs:
      flagged = proba >= cutoff
      tp = int((flagged & (y == 1)).sum())
      rows.append({'model': name, 'cutoff': cutoff, 'precision': tp / max(int(flagged.sum()), 1),
                   'recall': tp / int((y == 1).sum()), 'flagged_share': float(flagged.mean())})
  return pd.DataFrame(rows)


def time_call(function, repeats=3):
  """Best wall-clock time in seconds of `function()` over a few runs (the first run warms caches)."""
  best = float('inf')
  for _ in range(repeats):
    start = time.perf_counter()
    function()
    best = min(best, time.perf_counter() - start)
  return best
