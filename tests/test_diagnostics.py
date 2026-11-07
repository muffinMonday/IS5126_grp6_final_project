"""Hand-checkable tests of the slide 30-33 helpers in functions/diagnostics.py."""
import unittest

import numpy as np
import pandas as pd

from functions.diagnostics import (cutoff_table, flag_breakdown, has_profanity, time_call,
                                   token_length_summary, transfer_summary, triage_at_share, triage_table)


class DiagnosticsTests(unittest.TestCase):
  def test_profanity_matches_word_starts_on_lowercased_text(self):
    flags = has_profanity(['What the FUCK', 'shitty boss', 'Damn it', 'class action', 'fuck-all', 'refuckulous'])
    self.assertEqual(flags.tolist(), [True, True, True, False, True, False])  # \b: a word must start with it

  def test_token_length_summary(self):
    summary = token_length_summary([10, 20, 80, 81, 200], max_length=80)
    self.assertEqual(summary['comments'], 5)
    self.assertEqual(summary['mean'], 78.2)
    self.assertEqual(summary['median'], 80)
    self.assertEqual(summary['over_80'], 2)       # 80 itself is not cut
    self.assertEqual(summary['share_over_80'], 0.4)

  def test_flag_breakdown_counts_each_group_against_its_size(self):
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    profane = np.array([True, True, False, False, False, False, False, False])
    n_tokens = np.array([5, 5, 5, 5, 100, 100, 10, 10])
    proba = {'m': np.array([0.9, 0.1, 0.6, 0.2, 0.9, 0.2, 0.4, 0.5])}  # exactly 0.5 counts as flagged
    table = flag_breakdown(y, proba, profane, n_tokens, max_length=80)
    row = table.loc['m']
    self.assertEqual(row['non-toxic, profane: flagged'], (1, 2))
    self.assertEqual(row['non-toxic, no profanity: flagged'], (1, 2))
    self.assertEqual(row['toxic, over 80 tokens: missed'], (1, 2))
    self.assertEqual(row['toxic, up to 80 tokens: missed'], (1, 2))

  def test_transfer_summary(self):
    val = pd.DataFrame({'recall (TPR)': [0.5], 'FPR': [0.02], 'roc_auc': [0.9]}, index=['m'])
    other = pd.DataFrame({'recall (TPR)': [0.4], 'FPR': [0.20], 'roc_auc': [0.75]}, index=['m'])
    out = transfer_summary(val, other).loc['m']
    self.assertAlmostEqual(out['recall_change'], -0.1)
    self.assertAlmostEqual(out['fpr_ratio'], 10.0)
    self.assertAlmostEqual(out['roc_auc_drop'], 0.15)

  def test_triage_at_share_matches_a_worked_example(self):
    precision, flagged, caught = triage_at_share(0.5, 0.1, 0.05)
    self.assertAlmostEqual(float(precision), 0.025 / (0.025 + 0.095))   # 0.2083
    self.assertAlmostEqual(float(flagged), 0.025 + 0.095)               # 12% of all comments reach the queue
    self.assertAlmostEqual(float(caught), 0.5)
    # at 50% toxic, precision = recall / (recall + FPR)
    self.assertAlmostEqual(float(triage_at_share(0.6, 0.2, 0.5)[0]), 0.6 / 0.8)

  def test_triage_table_has_one_row_per_model_and_both_datasets(self):
    val = pd.DataFrame({'recall (TPR)': [0.5, 0.7], 'FPR': [0.02, 0.05], 'roc_auc': [0.9, 0.9]}, index=['a', 'b'])
    other = pd.DataFrame({'recall (TPR)': [0.4, 0.6], 'FPR': [0.2, 0.3], 'roc_auc': [0.8, 0.8]}, index=['a', 'b'])
    table = triage_table(val, other, share=0.05)
    self.assertEqual(list(table.index), ['a', 'b'])
    self.assertEqual(set(table.columns.get_level_values(0)), {'validation', 'r/jobs'})
    self.assertAlmostEqual(table.loc['a', ('validation', 'precision')], 0.5 * 0.05 / (0.5 * 0.05 + 0.02 * 0.95))

  def test_cutoff_table_precision_recall_and_flagged_share(self):
    y = np.array([0, 0, 1, 1, 1])
    proba = {'m': np.array([0.95, 0.2, 0.99, 0.85, 0.4])}
    table = cutoff_table(y, proba, cutoffs=(0.5, 0.9, 0.995)).set_index('cutoff')
    self.assertAlmostEqual(table.loc[0.5, 'precision'], 2 / 3)
    self.assertAlmostEqual(table.loc[0.5, 'recall'], 2 / 3)
    self.assertAlmostEqual(table.loc[0.9, 'precision'], 1 / 2)
    self.assertAlmostEqual(table.loc[0.9, 'recall'], 1 / 3)
    self.assertEqual(table.loc[0.995, 'flagged_share'], 0.0)
    self.assertEqual(table.loc[0.995, 'precision'], 0.0)   # nothing flagged: no division by zero

  def test_time_call_returns_the_best_of_several_runs(self):
    calls = []
    seconds = time_call(lambda: calls.append(1), repeats=3)
    self.assertEqual(len(calls), 3)
    self.assertGreaterEqual(seconds, 0.0)


if __name__ == '__main__':
  unittest.main()
