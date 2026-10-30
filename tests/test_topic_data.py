"""Small synthetic checks for time coverage, sampling and topic cleaning."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from functions.topic_data import (
  ArchiveUnavailable, clean_topic_comments, export_topic_run,
  fetch_archive_window, month_windows, sample_months,
)


AS_OF = '2026-10-28T06:00:00+00:00'


def comment(index, month=1, **overrides):
  row = {'comment_id': f'c{index}', 'text': 'The job interview was helpful!',
         'created_utc': int(pd.Timestamp(f'2026-{month:02d}-15', tz='UTC').timestamp()),
         'subreddit': 'jobs', 'link_id': 't3_post'}
  row.update(overrides)
  return row


class TopicDataTests(unittest.TestCase):
  def test_windows_utc_current_partial_and_no_future_months(self):
    windows = month_windows(2026, AS_OF)
    self.assertEqual(len(windows), 10)
    self.assertEqual(windows[-1]['month'], '2026-10')
    self.assertTrue(windows[-1]['partial_month'])
    self.assertFalse(any(w['partial_month'] for w in windows[:-1]))
    self.assertEqual(len(month_windows(2026, '2026-10-01T00:00:00Z')), 9)
    with self.assertRaises(ValueError):
      month_windows(2026, '2026-10-28')

  def test_cleaning_preserves_raw_and_same_text_different_ids(self):
    raw = 'I can\'t find &amp; keep a job! https://example.com'
    eligible, excluded, audit = clean_topic_comments(
      [comment(1, text=raw), comment(2, text=raw), comment(1, text=raw)], as_of=AS_OF)
    self.assertEqual(eligible['text'].tolist(), [raw, raw])
    self.assertTrue(eligible['duplicate_text'].all())
    self.assertIn('can not', eligible.iloc[0]['text_topic'])
    self.assertNotIn('https', eligible.iloc[0]['text_topic'])
    self.assertEqual(excluded['exclusion_reason'].tolist(), ['duplicate_comment_id'])
    self.assertEqual(audit['n_rows'].sum(), 3)

  def test_invalid_deleted_bot_short_language_and_time_reasons(self):
    rows = [comment(1, text=None), comment(2, text='[removed]'),
            comment(3, author='AutoModerator'), comment(4, text='Thanks!'),
            comment(5, language='fr'), comment(6, created_utc=1700000000000),
            comment(7, created_utc=int(pd.Timestamp('2027-01-01', tz='UTC').timestamp())),
            comment(8, subreddit='other'), comment(9, month=2)]
    eligible, excluded, _ = clean_topic_comments(rows, as_of=AS_OF)
    self.assertEqual(eligible['comment_id'].tolist(), ['c9'])
    self.assertEqual(set(excluded['exclusion_reason']),
                     {'empty_or_nonstring_text', 'deleted_or_removed', 'known_bot',
                      'too_short', 'known_non_english', 'invalid_timestamp',
                      'at_or_after_cutoff', 'wrong_subreddit'})

  def test_conflicting_same_id_is_not_silently_overwritten(self):
    with self.assertRaisesRegex(ValueError, 'Conflicting snapshots'):
      clean_topic_comments([comment(1), comment(1, text='Another version of this comment')], as_of=AS_OF)

  def test_boundaries_and_no_missing_string_tokens(self):
    rows = [comment(1, created_utc=int(pd.Timestamp('2026-01-01', tz='UTC').timestamp())),
            comment(2, created_utc=int(pd.Timestamp(AS_OF).timestamp())),
            comment(3, text='https://example.com'), comment(4, text=float('nan'))]
    eligible, excluded, _ = clean_topic_comments(rows, as_of=AS_OF)
    self.assertEqual(eligible['comment_id'].tolist(), ['c1'])
    self.assertEqual(len(excluded), 3)

  def test_sampling_reproducible_no_replacement_and_explicit_shortfalls(self):
    rows = [comment(i) for i in range(20)] + [comment(100 + i, month=2) for i in range(3)]
    eligible, _, _ = clean_topic_comments(rows, as_of=AS_OF)
    sample, coverage = sample_months(eligible, as_of=AS_OF, per_month=5)
    reordered, _ = sample_months(eligible.iloc[::-1], as_of=AS_OF, per_month=5)
    self.assertEqual(sample['comment_id'].tolist(), reordered['comment_id'].tolist())
    self.assertTrue(sample['comment_id'].is_unique)
    self.assertEqual(coverage['selected'].tolist()[:3], [5, 3, 0])
    self.assertEqual(coverage['shortfall'].tolist()[:3], [0, 2, 5])
    self.assertTrue(coverage['source_window_complete'].isna().all())

  def test_archive_splits_saturated_time_windows(self):
    client = Mock()
    client.search.side_effect = [[{}] * 100,
                                 [{'created_utc': 10, 'subreddit': 'jobs'}],
                                 [{'created_utc': 15, 'subreddit': 'jobs'}]]
    rows = fetch_archive_window(client, 10, 20)
    self.assertEqual(len(rows), 2)
    self.assertEqual(client.search.call_args_list[1].args, (10, 15))
    self.assertEqual(client.search.call_args_list[2].args, (15, 20))

  def test_time_strata_keep_equal_quotas_and_report_local_shortfalls(self):
    rows = [comment(i, sampling_stratum=0) for i in range(10)]
    rows += [comment(100, sampling_stratum=1)]
    eligible, _, _ = clean_topic_comments(rows, as_of=AS_OF)
    sample, coverage = sample_months(eligible, as_of=AS_OF, per_month=6, strata_per_month=2)
    self.assertEqual(sample.groupby('sampling_stratum').size().to_dict(), {0: 3, 1: 1})
    self.assertEqual(coverage.iloc[0]['selected'], 4)
    self.assertEqual(coverage.iloc[0]['shortfall'], 2)

  def test_archive_rejects_truncation_and_outside_window(self):
    client = Mock()
    client.search.return_value = [{}] * 100
    with self.assertRaises(ArchiveUnavailable):
      fetch_archive_window(client, 10, 11)
    client.search.return_value = [{'created_utc': 20, 'subreddit': 'jobs'}]
    with self.assertRaises(ArchiveUnavailable):
      fetch_archive_window(client, 10, 20)

  def test_export_reconciles_counts_and_never_overwrites(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp) / 'run'
      sample, coverage, manifest = export_topic_run(
        output, [comment(1), comment(2, text='[deleted]')], 2026, AS_OF,
        {'method': 'synthetic_test'}, per_month=5)
      self.assertEqual(manifest['counts'], {'raw': 2, 'eligible': 1, 'excluded': 1, 'selected': 1})
      self.assertTrue((output / 'sampling_manifest.json').exists())
      self.assertEqual(len(coverage), 10)
      self.assertEqual(len(sample), 1)
      with self.assertRaises(FileExistsError):
        export_topic_run(output, [comment(1)], 2026, AS_OF, {})


if __name__ == '__main__':
  unittest.main()
