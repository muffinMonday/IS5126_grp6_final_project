"""Check monthly denominators and sensitivity on known topic assignments."""

import unittest

import numpy as np
import pandas as pd

from functions.topic_reporting import topic_trend_diagnostics


class TopicReportingTests(unittest.TestCase):
  def test_counts_use_all_topics_and_post_weights_reduce_thread_dominance(self):
    frame = pd.DataFrame({'comment_id': list('abcdef'), 'month': ['2026-01'] * 4 + ['2026-07'] * 2,
                          'cluster': [9, 9, 9, 17, 9, 17],
                          'link_id': ['post_a', 'post_a', 'post_a', 'post_b', 'post_c', 'post_d'],
                          'sampling_stratum': [0, 0, 0, 1, 0, 1]})
    coverage = pd.DataFrame({'month': ['2026-01', '2026-07'], 'partial_month': [False, False]})
    counts, shares, monthly, summary = topic_trend_diagnostics(frame, coverage, ['2026-01'], ['2026-07'])
    row = monthly.loc[monthly.month.eq('2026-01') & monthly.cluster.eq(9)].iloc[0]
    self.assertEqual(row.n_assigned, 4)
    self.assertEqual(row.topic_share, 0.75)
    self.assertEqual(row.equal_post_share, 0.5)
    self.assertEqual(row.minimum_share, 0)
    self.assertEqual(row.maximum_share, 1)
    self.assertEqual(summary.set_index('cluster').loc[9, 'change_pp'], -25)
    self.assertTrue(np.allclose(shares.sum(axis=1), 1))
    self.assertEqual(counts.shape, (2, 2))

  def test_partial_periods_and_missing_identifiers_do_not_fabricate_comparisons(self):
    frame = pd.DataFrame({'comment_id': ['a', 'b', 'c', 'd'],
                          'month': ['2026-01', '2026-01', '2026-07', '2026-07'],
                          'cluster': [9, 17, 9, 17]})
    coverage = pd.DataFrame({'month': ['2026-01', '2026-07'], 'partial_month': [False, True]})
    _, _, monthly, summary = topic_trend_diagnostics(frame, coverage, ['2026-01'], ['2026-07'])
    self.assertTrue(summary.last3_share.isna().all())
    self.assertTrue(summary.change_pp.isna().all())
    self.assertTrue(monthly.equal_post_share.isna().all())
    self.assertTrue(monthly.minimum_share.isna().all())
    self.assertFalse(summary.both_periods_available.any())


if __name__ == '__main__':
  unittest.main()
