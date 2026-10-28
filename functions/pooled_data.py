"""Prepare the four external toxicity sources without reading Reddit data."""

import hashlib
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from .preprocessing import cleanup_text


POOLED_COLUMNS = ['orig_id', 'source', 'text_raw', 'text', 'label']
JIGSAW_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
SOURCE_FILES = {
  'youtube': ('youtoxic_english_1000.csv', 'CommentId', 'Text', ['IsToxic']),
  'wikipedia': ('jigsaw-toxic-comment-classification-challenge/train.csv.zip',
                'id', 'comment_text', JIGSAW_LABELS),
  'news': ('jigsaw-unintended-bias-in-toxicity-classification/train.csv',
           'id', 'comment_text', ['target']),
  'dota': ('gosu-ai-english-dota-chat/tagged-data.csv', 'Unnamed: 0', 'text', ['target']),
}
LABEL_MAPPING = {
  'youtube': 'IsToxic True -> 1; False -> 0',
  'wikipedia': 'Any of toxic,severe_toxic,obscene,threat,insult,identity_hate == 1 -> 1',
  'news': 'target >= 0.5 -> 1; target < 0.5 -> 0',
  'dota': 'target in {1,2} -> 1; target == 0 -> 0',
}


def _sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def _map_labels(frame, source, label_columns):
  if source == 'youtube':
    values = frame['IsToxic'].str.strip().str.casefold()
    if not values.isin(['true', 'false']).all():
      raise ValueError(f'{source}: IsToxic must contain only True or False, without blanks.')
    return values.eq('true').astype('int64')

  values = frame[label_columns].apply(pd.to_numeric, errors='coerce')
  if not np.isfinite(values.to_numpy(dtype=float)).all():
    raise ValueError(f'{source}: toxicity labels must be finite numbers, without blanks.')
  if source == 'wikipedia':
    if not values.isin([0, 1]).all().all():
      raise ValueError('wikipedia: all six toxicity labels must be 0 or 1.')
    return values.max(axis=1).astype('int64')
  if source == 'news':
    if not values['target'].between(0, 1).all():
      raise ValueError('news: target must be between 0 and 1 inclusive.')
    return values['target'].ge(0.5).astype('int64')
  if not values['target'].isin([0, 1, 2]).all():
    raise ValueError('dota: target must be one of 0, 1, 2.')
  return values['target'].isin([1, 2]).astype('int64')


def _load_sample(path, source, cap, seed):
  _, id_column, text_column, label_columns = SOURCE_FILES[source]
  columns = [id_column, text_column] + label_columns
  header = pd.read_csv(path, nrows=0, keep_default_na=False)
  missing = set(columns) - set(header.columns)
  if missing:
    raise ValueError(f'{source}: missing required columns {sorted(missing)} in {path}.')

  rng = np.random.default_rng(seed)
  counts = np.zeros(2, dtype=np.int64)
  pools = [None, None]
  seen_ids = set()
  # Keep only the cap smallest random priorities per class, not 1.8M news texts.
  with pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False,
                   chunksize=100_000) as chunks:
    for frame in chunks:
      ids = frame[id_column].str.strip()
      if ids.eq('').any():
        raise ValueError(f'{source}: original IDs must not be empty.')
      chunk_ids = set(ids)
      if len(chunk_ids) != len(ids) or seen_ids.intersection(chunk_ids):
        raise ValueError(f'{source}: duplicate original IDs must be resolved before sampling.')
      seen_ids.update(chunk_ids)
      labels = _map_labels(frame, source, label_columns)
      chunk = pd.DataFrame({
        'orig_id': ids, 'source': source, 'text_raw': frame[text_column],
        'label': labels, '_priority': rng.random(len(frame)),
      })
      for label in (0, 1):
        part = chunk[chunk['label'] == label]
        counts[label] += len(part)
        pools[label] = pd.concat([pools[label], part], ignore_index=True)
        pools[label] = pools[label].nsmallest(cap, '_priority')

  total = int(counts.sum())
  if not total or (counts == 0).any():
    raise ValueError(f'{source}: both binary classes are required; counts are {counts.tolist()}.')
  n = min(total, cap)
  n_toxic = round(n * int(counts[1]) / total)
  quotas = [n - n_toxic, n_toxic]
  if min(quotas) < 2:
    raise ValueError(f'{source}: sample cap leaves too few rows per class for stratification.')
  sampled = pd.concat([pools[y].head(quotas[y]) for y in (0, 1)], ignore_index=True)
  sampled = sampled.drop(columns='_priority')
  return sampled, {'rows': total, 'non_toxic': int(counts[0]), 'toxic': int(counts[1])}


def _summary(frame, stage):
  rows = []
  for source in SOURCE_FILES:
    group = frame[frame['source'] == source]
    toxic = int(group['label'].sum())
    n = len(group)
    rows.append({'source': source, 'stage': stage, 'rows': n,
                 'non_toxic': n - toxic, 'toxic': toxic,
                 'toxic_share': toxic / n if n else 0.0})
  return rows


def _check_strata(frame, name, minimum=1):
  strata = frame.groupby(['source', 'label']).size()
  for source in SOURCE_FILES:
    for label in (0, 1):
      if strata.get((source, label), 0) < minimum:
        raise ValueError(f'{name}: source {source}, label {label} has too few rows '
                         'for the requested stratified train/validation split.')


def prepare_pooled_data(data_dir, cap_per_source=10000, test_size=0.2, random_state=42):
  """Return fixed train/validation frames and audit tables; never write files.

  Sampling preserves each source's class share (rounded to the nearest row).
  Deduplication applies to the selected pool before splitting. Conflicting binary
  labels are removed; matching labels retain the first (source, orig_id) pair.
  """
  if isinstance(cap_per_source, bool) or not isinstance(cap_per_source, int) or cap_per_source < 2:
    raise ValueError('cap_per_source must be an integer of at least 2.')
  if isinstance(test_size, bool) or not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
    raise ValueError('test_size must be a fraction strictly between 0 and 1.')
  if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
    raise ValueError('random_state must be a non-negative integer.')

  data_dir = Path(data_dir)
  for source, (relative, _, _, _) in SOURCE_FILES.items():
    if not (data_dir / relative).is_file():
      raise FileNotFoundError(f'{source}: required source is missing: {data_dir / relative}')

  parts, summaries, inputs = [], [], {}
  for source, (relative, _, _, _) in SOURCE_FILES.items():
    sampled, raw = _load_sample(data_dir / relative, source, cap_per_source, random_state)
    parts.append(sampled)
    summaries.append({'source': source, 'stage': 'raw', **raw,
                      'toxic_share': raw['toxic'] / raw['rows']})
    inputs[source] = {'path': relative, 'sha256': _sha256(data_dir / relative), **raw}

  sampled = pd.concat(parts, ignore_index=True)
  summaries.extend(_summary(sampled, 'sampled'))
  sampled['text'] = sampled['text_raw'].str.lower().apply(cleanup_text)
  sampled = sampled[POOLED_COLUMNS].sort_values(['source', 'orig_id'], kind='stable')
  removed = []
  empty = sampled['text'].eq('')
  removed.append(sampled.loc[empty].assign(reason='empty_text'))
  cleaned = sampled.loc[~empty].copy()
  summaries.extend(_summary(cleaned, 'cleaned'))

  label_counts = cleaned.groupby('text')['label'].nunique()
  conflicts = cleaned['text'].isin(label_counts[label_counts > 1].index)
  removed.append(cleaned.loc[conflicts].assign(reason='conflicting_label'))
  consistent = cleaned.loc[~conflicts]
  duplicates = consistent['text'].duplicated(keep='first')
  removed.append(consistent.loc[duplicates].assign(reason='duplicate_text'))
  pooled = consistent.loc[~duplicates].reset_index(drop=True)
  summaries.extend(_summary(pooled, 'deduplicated'))
  _check_strata(pooled, 'deduplicated pool', minimum=2)

  n_val = math.ceil(len(pooled) * test_size)
  if min(n_val, len(pooled) - n_val) < 2 * len(SOURCE_FILES):
    raise ValueError('Split sizes are too small to retain all source/label strata.')
  strata = pooled['source'] + '_' + pooled['label'].astype(str)
  try:
    train, val = train_test_split(pooled, test_size=test_size, stratify=strata,
                                  random_state=random_state)
  except ValueError as exc:
    raise ValueError(f'Cannot stratify the pooled data: {exc}') from exc
  train, val = train.reset_index(drop=True), val.reset_index(drop=True)
  _check_strata(train, 'train')
  _check_strata(val, 'validation')
  if set(train['text']) & set(val['text']):
    raise AssertionError('Cleaned text leaked across the train/validation split.')
  train_ids = set(zip(train['source'], train['orig_id']))
  if train_ids & set(zip(val['source'], val['orig_id'])):
    raise AssertionError('Source IDs leaked across the train/validation split.')
  summaries.extend(_summary(train, 'train'))
  summaries.extend(_summary(val, 'val'))

  manifest = {
    'config': {'cap_per_source': cap_per_source, 'test_size': test_size,
               'random_state': random_state, 'class_resampling': None},
    'inputs': inputs,
    'label_mapping': LABEL_MAPPING.copy(),
    'cleaning': {'pipeline': 'str.lower -> functions.preprocessing.cleanup_text',
                 'sha256': _sha256(Path(__file__).with_name('preprocessing.py'))},
    'preparation_code_sha256': _sha256(__file__),
    'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
    'deduplication': 'Selected pool only; matching labels keep first sorted source/orig_id; '
                     'all conflicting labels are removed; no refill.',
    'rows': {'train': len(train), 'val': len(val)},
    'split_sha256': {
      name: hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
      for name, frame in [('train', train), ('val', val)]
    },
  }
  audit = {
    'summary': pd.DataFrame(summaries, columns=[
      'source', 'stage', 'rows', 'non_toxic', 'toxic', 'toxic_share']),
    'removed': pd.concat(removed, ignore_index=True)[POOLED_COLUMNS + ['reason']],
    'manifest': manifest,
  }
  return train, val, audit
