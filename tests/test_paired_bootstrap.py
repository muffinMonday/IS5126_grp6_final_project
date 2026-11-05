"""Checks of the paired bootstrap helper and of the cell in notebook 4b that uses it."""
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from functions.metrics import paired_bootstrap


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebook/4b_LSTM_Logistic_Comparison.ipynb'


class PairedBootstrapTests(unittest.TestCase):
  def setUp(self):
    rng = np.random.default_rng(1)
    self.y = (rng.random(300) < 0.3).astype(int)
    self.good = np.clip(self.y * 0.6 + rng.random(300) * 0.4, 0, 1)   # separates the classes
    self.noise = rng.random(300)                                      # does not

  def test_identical_models_have_no_difference(self):
    table = paired_bootstrap(self.y, self.good, self.good, n_boot=50)
    self.assertEqual(list(table.index), ['recall', 'FPR', 'macro_f1', 'pr_auc'])
    self.assertTrue((table[['difference', 'mean', 'ci_low', 'ci_high']] == 0).all().all())

  def test_better_model_has_a_positive_interval_and_swapping_negates_it(self):
    ab = paired_bootstrap(self.y, self.good, self.noise, n_boot=200)
    ba = paired_bootstrap(self.y, self.noise, self.good, n_boot=200)
    self.assertGreater(ab.loc['pr_auc', 'ci_low'], 0)
    self.assertTrue((ab.loc['pr_auc', 'ci_low'] <= ab.loc['pr_auc', 'difference'] <= ab.loc['pr_auc', 'ci_high']))
    np.testing.assert_allclose(ab['difference'], -ba['difference'])
    np.testing.assert_allclose(ab['ci_low'], -ba['ci_high'])

  def test_difference_is_the_full_sample_metric_gap_and_the_seed_fixes_the_result(self):
    from sklearn.metrics import average_precision_score, recall_score
    table = paired_bootstrap(self.y, self.good, self.noise, n_boot=20, seed=3, threshold=0.4)
    again = paired_bootstrap(self.y, self.good, self.noise, n_boot=20, seed=3, threshold=0.4)
    other = paired_bootstrap(self.y, self.good, self.noise, n_boot=20, seed=4, threshold=0.4)
    pd.testing.assert_frame_equal(table, again)
    self.assertFalse(table['ci_low'].equals(other['ci_low']))
    self.assertAlmostEqual(table.loc['pr_auc', 'difference'],
                           average_precision_score(self.y, self.good) - average_precision_score(self.y, self.noise))
    self.assertAlmostEqual(table.loc['recall', 'difference'],
                           recall_score(self.y, self.good >= 0.4) - recall_score(self.y, self.noise >= 0.4))

  def test_length_mismatch_is_rejected(self):
    with self.assertRaises(ValueError):
      paired_bootstrap(self.y, self.good[:-1], self.noise)

  def test_notebook_cell_reports_points_and_whether_zero_is_inside(self):
    cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    source = next(''.join(c['source']) for c in cells if c['id'] == 'bootstrap')
    probas = {'LSTM (standard)': self.good, 'LSTM (balanced)': self.noise, 'LR (balanced)': self.good * 0.9}
    shown = []
    ns = {'pd': pd, 'paired_bootstrap': paired_bootstrap, 'THRESHOLD': 0.5, 'display': shown.append,
          'y_val': self.y, 'val_proba': probas, 'y_reddit': self.y[:100], 'reddit_proba': {k: v[:100] for k, v in probas.items()}}
    with redirect_stdout(StringIO()):
      exec(compile(source.replace("'n_boot': 1000", "'n_boot': 30"), 'bootstrap', 'exec'), ns)
    points = ns['bootstrap_points']
    self.assertEqual(len(points), 16)  # 2 datasets x 2 comparisons x 4 metrics
    self.assertEqual(points.index.names, ['dataset', 'comparison', 'metric'])
    inside = (ns['bootstrap']['ci_low'] <= 0) & (ns['bootstrap']['ci_high'] >= 0)
    self.assertTrue((points['includes zero'] == inside).all())
    row = ns['bootstrap'].loc[('validation', 'LSTM (standard) minus LR (balanced)', 'pr_auc')]
    self.assertAlmostEqual(points.loc[('validation', 'LSTM (standard) minus LR (balanced)', 'pr_auc'), 'observed difference on the full set'],
                           round(row['difference'] * 100, 1))


if __name__ == '__main__':
  unittest.main()
