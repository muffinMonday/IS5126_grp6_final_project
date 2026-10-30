"""Synthetic, offline checks for the monthly time-stratified collection entry."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from functions.topic_data import ArchiveUnavailable
from scripts.collect_monthly_topics import collect


class FakeArchive:
  def __init__(self, **kwargs):
    self.requests = 0

  def search(self, start, end):
    self.requests += 1
    return [{'id': f'c{self.requests}x{i}', 'body': 'Useful advice about a job interview',
             'created_utc': start + (end - start) * (i + 1) // 21,
             'subreddit': 'jobs', 'link_id': 't3_post', 'parent_id': 't3_post',
             'author': 'synthetic-test-author'} for i in range(20)]


class MonthlyTopicCollectionTests(unittest.TestCase):
  def test_collect_clean_sample_and_checkpoint_provenance(self):
    with tempfile.TemporaryDirectory() as tmp:
      work = Path(tmp) / 'run'
      with patch('scripts.collect_monthly_topics.ArcticShiftClient', FakeArchive), contextlib.redirect_stdout(io.StringIO()):
        result = collect(work, per_month=4, strata=2, as_of='2026-02-01T00:00:00Z')
      self.assertEqual(result, 0)
      sample = pd.read_csv(work / 'prepared/comments_clean.csv')
      self.assertEqual(sample.groupby('sampling_stratum').size().to_dict(), {0: 2, 1: 2})
      self.assertTrue(sample['comment_id'].is_unique)
      self.assertNotIn('author', sample.columns)
      manifest = json.loads((work / 'prepared/sampling_manifest.json').read_text(encoding='utf-8'))
      self.assertEqual(manifest['strata_per_month'], 2)
      self.assertEqual(manifest['counts']['selected'], 4)
      self.assertEqual(len(list((work / 'source_cache').glob('*.json'))), 2)
      with self.assertRaises(FileExistsError):
        collect(work, per_month=4, strata=2)

  def test_source_failure_preserves_failure_instead_of_fabricating_samples(self):
    with tempfile.TemporaryDirectory() as tmp:
      work = Path(tmp) / 'failed'
      with patch('scripts.collect_monthly_topics.ArcticShiftClient') as client, contextlib.redirect_stdout(io.StringIO()):
        client.return_value.search.side_effect = ArchiveUnavailable('HTTP 429; wait for reset')
        client.return_value.requests = 1
        result = collect(work, per_month=4, strata=2, as_of='2026-02-01T00:00:00Z')
      self.assertEqual(result, 2)
      status = json.loads((work / 'acquisition_status.json').read_text(encoding='utf-8'))
      self.assertEqual(status['status'], 'source_error')
      self.assertEqual(status['completed_strata'], 0)
      self.assertFalse((work / 'prepared').exists())


if __name__ == '__main__':
  unittest.main()
