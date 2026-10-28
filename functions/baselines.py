"""Fixed Part 3 baselines and transfer checks on external data only."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from functions.metrics import metrics_by_group, toxicity_metrics


DATA_COLUMNS = ['orig_id', 'source', 'text_raw', 'text', 'label']
SOURCES = {'youtube', 'wikipedia', 'news', 'dota'}


def build_tfidf_logreg(class_weight=None, random_state=42):
  """Build an unfitted pipeline; vocabulary is learned only by fit()."""
  return make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000,
                    sublinear_tf=True),
    LogisticRegression(C=1, solver='lbfgs', max_iter=1000,
                       class_weight=class_weight, random_state=random_state),
  )


def _validate_data(train_df, val_df):
  for name, frame in [('train', train_df), ('validation', val_df)]:
    missing = set(DATA_COLUMNS) - set(frame.columns)
    if missing:
      raise ValueError(f'{name} is missing columns: {sorted(missing)}')
    if frame.empty or frame[DATA_COLUMNS].isna().any().any():
      raise ValueError(f'{name} must contain nonmissing data.')
    for col in DATA_COLUMNS[:-1]:
      if not frame[col].map(lambda value: isinstance(value, str)).all():
        raise ValueError(f'{name}.{col} must contain strings.')
      if frame[col].str.strip().eq('').any():
        raise ValueError(f'{name}.{col} must not contain blank values.')
    if not frame['label'].isin([0, 1]).all():
      raise ValueError(f'{name} labels must be binary 0/1.')
    if set(frame['source']) != SOURCES:
      raise ValueError(f'{name} must contain exactly the four external sources.')
    if frame.duplicated(['source', 'orig_id']).any() or frame['text'].duplicated().any():
      raise ValueError(f'{name} has duplicate source IDs or cleaned texts.')
    if not frame.groupby('source')['label'].nunique().eq(2).all():
      raise ValueError(f'{name} must contain both classes in every source.')
  train_ids = set(zip(train_df['source'], train_df['orig_id']))
  val_ids = set(zip(val_df['source'], val_df['orig_id']))
  if train_ids & val_ids or set(train_df['text']) & set(val_df['text']):
    raise ValueError('Training and validation source IDs/texts must not overlap.')


def _prediction_table(frame, proba, model, threshold):
  p = np.asarray(proba, dtype=float).ravel()
  if len(p) != len(frame) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
    raise ValueError('Predicted probabilities must be finite values in [0, 1], one per row.')
  rows = frame[DATA_COLUMNS].reset_index(drop=True).copy()
  rows['model'] = model
  rows['proba'] = p
  rows['predicted_label'] = (p >= threshold).astype(int)
  actual = rows['label'].to_numpy(dtype=int)
  pred = rows['predicted_label'].to_numpy()
  rows['error_type'] = np.select(
    [(actual == 0) & (pred == 0), (actual == 1) & (pred == 1),
     (actual == 0) & (pred == 1)], ['TN', 'TP', 'FP'], default='FN',
  )
  return rows


def run_baselines(train_df, val_df, threshold=0.5, random_state=42):
  """Score three prespecified baselines and four leave-one-source-out checks.

  All fits use train_df only. A transfer model excludes its target source from
  training and scores only that source's shared validation subset. All results
  are reported; no model or threshold is selected from these scores.
  """
  _validate_data(train_df, val_df)
  try:
    threshold = float(threshold)
  except (ValueError, TypeError):
    raise ValueError('threshold must be a finite number in [0, 1].') from None
  if not np.isfinite(threshold) or not 0 <= threshold <= 1:
    raise ValueError('threshold must be a finite number in [0, 1].')

  y_train = train_df['label'].to_numpy(dtype=int)
  y_val = val_df['label'].to_numpy(dtype=int)
  probabilities = {'majority': np.full(len(val_df), y_train.mean())}
  for name, weight in [('tfidf_logreg', None), ('tfidf_logreg_balanced', 'balanced')]:
    model = build_tfidf_logreg(weight, random_state)
    model.fit(train_df['text'], y_train)
    probabilities[name] = model.predict_proba(val_df['text'])[:, 1]

  metrics, predictions = [], []
  for name, proba in probabilities.items():
    rows = _prediction_table(val_df, proba, name, threshold)
    table = metrics_by_group(y_val, rows['proba'], val_df['source'], threshold).reset_index()
    table.insert(0, 'model', name)
    table['threshold'] = threshold
    metrics.append(table)
    predictions.append(rows)

  loso_metrics, loso_predictions = [], []
  for source in sorted(SOURCES):
    fitting = train_df[train_df['source'] != source]
    held_out = val_df[val_df['source'] == source]
    model = build_tfidf_logreg('balanced', random_state)
    model.fit(fitting['text'], fitting['label'])
    proba = model.predict_proba(held_out['text'])[:, 1]
    rows = _prediction_table(held_out, proba, 'tfidf_logreg_balanced_loso', threshold)
    rows['held_out_source'] = source
    loso_predictions.append(rows)
    loso_metrics.append({
      'held_out_source': source,
      'train_sources': ','.join(sorted(set(fitting['source']))),
      'n_train': len(fitting),
      **toxicity_metrics(held_out['label'], rows['proba'], threshold),
      'threshold': threshold,
    })

  config = {
    'random_state': int(random_state), 'threshold': threshold,
    'tfidf': {'ngram_range': [1, 2], 'min_df': 2, 'max_features': 50_000,
              'sublinear_tf': True},
    'logistic_regression': {'C': 1, 'solver': 'lbfgs', 'max_iter': 1000},
    'baseline_class_weights': {'tfidf_logreg': None, 'tfidf_logreg_balanced': 'balanced'},
    'majority_probability': float(y_train.mean()),
    'loso_class_weight': 'balanced',
    'loso_design': 'other-source training rows -> target-source shared validation rows',
    'precision_recall_positive_label': 1,
    'f1_average': 'macro', 'pr_auc_definition': 'average_precision_score (AP)',
    'selection': 'none; all fixed baselines and transfer checks are reported',
  }
  return {
    'metrics': pd.concat(metrics, ignore_index=True),
    'predictions': pd.concat(predictions, ignore_index=True),
    'loso_metrics': pd.DataFrame(loso_metrics),
    'loso_predictions': pd.concat(loso_predictions, ignore_index=True),
    'config': config,
  }
