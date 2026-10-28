"""Small external-source fixtures for the Part 3 data contract."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from functions.pooled_data import (
  JIGSAW_LABELS, POOLED_COLUMNS, SOURCE_FILES, _load_sample, prepare_pooled_data,
)


class PooledDataTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.frames = {}
    for source in SOURCE_FILES:
      rows = []
      for label in (0, 1):
        for number in range(20):
          identity = label * 20 + number
          text = f'{source} class {label} unique message {number}'
          if source == 'youtube':
            row = {'CommentId': f'y{identity:03}', 'Text': text, 'IsToxic': bool(label)}
          elif source == 'wikipedia':
            row = {'id': f'w{identity:03}', 'comment_text': text,
                   **dict.fromkeys(JIGSAW_LABELS, 0)}
            row['obscene'] = label
          elif source == 'news':
            row = {'id': f'n{identity:03}', 'comment_text': text,
                   'target': 0.5 if label else 0.499}
          else:
            row = {'Unnamed: 0': str(identity), 'text': text,
                   'target': 1 + number % 2 if label else 0}
          rows.append(row)
      self.frames[source] = pd.DataFrame(rows)
    self.frames['youtube'].loc[0, 'Text'] = 'None'
    self.write_sources()

  def write_sources(self):
    for source, frame in self.frames.items():
      path = self.root / SOURCE_FILES[source][0]
      path.parent.mkdir(parents=True, exist_ok=True)
      compression = {'method': 'zip', 'archive_name': 'train.csv'} if source == 'wikipedia' else None
      frame.to_csv(path, index=False, compression=compression)

  def prepare(self, **kwargs):
    return prepare_pooled_data(self.root, cap_per_source=100, **kwargs)

  def test_mappings_and_id_preservation(self):
    train, val, audit = self.prepare()
    pooled = pd.concat([train, val])
    self.assertEqual(list(train.columns), POOLED_COLUMNS)
    self.assertEqual(len(pooled), 160)
    for source in SOURCE_FILES:
      self.assertEqual(pooled.loc[pooled.source == source, 'label'].value_counts().to_dict(),
                       {0: 20, 1: 20})
    none = pooled[pooled.text_raw == 'None'].iloc[0]
    self.assertEqual(none['text'], 'none')
    dota = pooled[pooled.source == 'dota']
    self.assertEqual(set(dota.orig_id), set(str(i) for i in range(40)))
    self.assertTrue(all(isinstance(value, str) for value in pooled.orig_id))
    self.assertEqual(set(audit['summary'].stage),
                     {'raw', 'sampled', 'cleaned', 'deduplicated', 'train', 'val'})
    self.assertEqual(set(audit['manifest']['inputs']), set(SOURCE_FILES))

  def test_duplicates_conflicts_and_empty_text(self):
    self.frames['dota'].loc[0, 'text'] = 'Shared duplicate'
    self.frames['youtube'].loc[1, 'Text'] = 'SHARED DUPLICATE'
    self.frames['wikipedia'].loc[0, 'comment_text'] = 'Conflicting text'
    self.frames['news'].loc[20, 'comment_text'] = 'conflicting text'
    self.frames['wikipedia'].loc[1, 'comment_text'] = '<b></b> https://example.com'
    self.write_sources()
    train, val, audit = self.prepare()
    pooled = pd.concat([train, val])
    self.assertEqual(len(pooled), 156)
    self.assertEqual(pooled[pooled.text == 'shared duplicate'].source.tolist(), ['dota'])
    self.assertNotIn('conflicting text', set(pooled.text))
    self.assertNotIn('', set(pooled.text))
    self.assertEqual(audit['removed'].reason.value_counts().to_dict(),
                     {'conflicting_label': 2, 'empty_text': 1, 'duplicate_text': 1})

  def test_sampling_and_split_are_reproducible_and_disjoint(self):
    first = prepare_pooled_data(self.root, cap_per_source=20)
    second = prepare_pooled_data(self.root, cap_per_source=20)
    for a, b in zip(first[:2], second[:2]):
      pd.testing.assert_frame_equal(a, b)
    train, val, audit = first
    self.assertEqual(len(train) + len(val), 80)
    sampled = audit['summary'].query('stage == "sampled"')
    self.assertTrue(sampled.rows.eq(20).all())
    self.assertTrue(sampled.toxic.eq(10).all())
    self.assertFalse(set(train.text) & set(val.text))
    self.assertFalse(set(zip(train.source, train.orig_id)) & set(zip(val.source, val.orig_id)))
    self.assertEqual(audit['manifest']['split_sha256'], second[2]['manifest']['split_sha256'])
    for frame in [train, val]:
      self.assertEqual(len(frame.groupby(['source', 'label'])), 8)

  def test_streaming_imbalanced_sampling_matches_global_priorities(self):
    labels = np.array([0] * 108 + [1] * 12)
    frame = pd.DataFrame({
      'id': [f'n{i:03}' for i in range(120)],
      'comment_text': [f'message {i}' for i in range(120)],
      'target': np.where(labels == 1, '0.5', '0.1'),
    })

    class Chunks:
      def __enter__(self):
        return iter([frame.iloc[i:i + 40].copy() for i in range(0, 120, 40)])

      def __exit__(self, *args):
        return False

    def read_chunks(path, **kwargs):
      return frame.iloc[:0] if kwargs.get('nrows') == 0 else Chunks()

    with patch('functions.pooled_data.pd.read_csv', side_effect=read_chunks):
      sampled, counts = _load_sample('synthetic.csv', 'news', 20, 42)

    priorities = np.random.default_rng(42).random(len(frame))
    expected_ids = []
    for label, quota in [(0, 18), (1, 2)]:
      indices = np.flatnonzero(labels == label)
      selected = indices[np.argsort(priorities[indices])[:quota]]
      expected_ids.extend(frame.iloc[selected].id.tolist())
    self.assertEqual(sampled.orig_id.tolist(), expected_ids)
    self.assertEqual(sampled.label.value_counts().to_dict(), {0: 18, 1: 2})
    self.assertAlmostEqual(sampled.label.mean(), labels.mean())
    self.assertEqual(counts, {'rows': 120, 'non_toxic': 108, 'toxic': 12})

  def test_missing_source_fails_before_partial_preparation(self):
    with tempfile.TemporaryDirectory() as empty:
      with self.assertRaisesRegex(FileNotFoundError, 'required source is missing'):
        prepare_pooled_data(empty)

  def test_missing_column_rejected(self):
    self.frames['dota'] = self.frames['dota'].drop(columns='target')
    self.write_sources()
    with self.assertRaisesRegex(ValueError, 'missing required columns'):
      self.prepare()

  def test_invalid_and_blank_labels_rejected(self):
    mutations = [('youtube', 'IsToxic', ''), ('youtube', 'IsToxic', 'yes'),
                 ('wikipedia', 'toxic', -1), ('news', 'target', 1.01),
                 ('news', 'target', ''), ('dota', 'target', 3)]
    for source, column, value in mutations:
      with self.subTest(source=source, value=value):
        original = self.frames[source].copy()
        self.frames[source][column] = self.frames[source][column].astype(object)
        self.frames[source].loc[0, column] = value
        self.write_sources()
        with self.assertRaises(ValueError):
          self.prepare()
        self.frames[source] = original
    self.write_sources()

  def test_duplicate_and_blank_ids_rejected(self):
    for value in ['', 'y001']:
      with self.subTest(value=value):
        self.frames['youtube'].loc[0, 'CommentId'] = value
        self.write_sources()
        with self.assertRaisesRegex(ValueError, 'IDs'):
          self.prepare()

  def test_small_class_or_split_rejected(self):
    self.frames['dota'] = self.frames['dota'].iloc[:21]
    self.write_sources()
    with self.assertRaisesRegex(ValueError, 'too few'):
      self.prepare()
    self.frames['dota'] = self.frames['dota'].iloc[:0].copy()
    self.write_sources()
    with self.assertRaisesRegex(ValueError, 'both binary classes'):
      self.prepare()

  def test_invalid_configuration(self):
    for config in [{'cap_per_source': 0}, {'test_size': 0}, {'test_size': 1},
                   {'random_state': -1}, {'random_state': True}]:
      with self.subTest(config=config):
        with self.assertRaises(ValueError):
          prepare_pooled_data(self.root, **config)


if __name__ == '__main__':
  unittest.main()
