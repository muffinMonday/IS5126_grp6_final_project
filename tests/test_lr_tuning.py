"""Small checks of fold isolation, selection and the notebook's opt-in path."""
import unittest
from unittest.mock import patch
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from functions.lr_tuning import tune_logistic_regressions


class LRTuningTests(unittest.TestCase):
  def setUp(self):
    self.texts = np.array([f'row{i} ' + ('rude attack' if i % 2 else 'kind helpful')
                           for i in range(24)])
    self.labels = np.arange(24) % 2

  def test_each_fold_fits_its_own_vocabulary_and_both_variants_share_folds(self):
    fitted = []
    original = TfidfVectorizer.fit_transform

    def record(vectorizer, documents, y=None):
      fitted.append(tuple(documents))
      return original(vectorizer, documents, y)

    with patch.object(TfidfVectorizer, 'fit_transform', record):
      result = tune_logistic_regressions(self.texts, self.labels, c_values=[0.1, 1], n_splits=3)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=42).split(self.texts, self.labels))
    expected = [tuple(self.texts[tr]) for tr, _ in folds] * 2 + [tuple(self.texts)]
    self.assertEqual(fitted, expected * 2)
    for name, model in result['models'].items():
      lr = model.named_steps['logisticregression']
      self.assertEqual(lr.class_weight, 'balanced' if name.endswith('balanced') else None)
      log = result['cv_results'].query('model == @name')
      best = log.loc[log.mean_cv_pr_auc.idxmax()]
      eligible = log[log.mean_cv_pr_auc >= best.mean_cv_pr_auc - best.std_cv_pr_auc]
      chosen = log[log.selected].iloc[0]
      self.assertAlmostEqual(chosen.std_cv_pr_auc, eligible.std_cv_pr_auc.min())
      self.assertEqual(lr.C, chosen.C)
      self.assertEqual(model.predict_proba(['unseen validationonly']).shape, (1, 2))
      self.assertNotIn('validationonly', model.named_steps['tfidfvectorizer'].vocabulary_)

  def test_invalid_inputs(self):
    for grid in [[], [0], [float('nan')]]:
      with self.assertRaises(ValueError):
        tune_logistic_regressions(self.texts, self.labels, c_values=grid)
    with self.assertRaises(ValueError):
      tune_logistic_regressions(self.texts, self.labels, n_splits=13)

  def test_notebook_defaults_do_not_train_or_save(self):
    path = Path(__file__).resolve().parents[1] / 'notebook/3_pooled_data.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    cell = next(c for c in notebook.cells if c.id == 'lr-tuning-run')
    with patch('functions.lr_tuning.tune_logistic_regressions') as tune:
      exec(compile(cell.source, str(path), 'exec'), {})
      tune.assert_not_called()

  def test_notebook_enabled_path_scores_holdout_without_saving(self):
    path = Path(__file__).resolve().parents[1] / 'notebook/3_pooled_data.ipynb'
    notebook = nbformat.read(path, as_version=4)
    cell = next(c for c in notebook.cells if c.id == 'lr-tuning-run')
    train = pd.DataFrame({'text': self.texts, 'label': self.labels, 'source': 'synthetic'})
    val = pd.DataFrame({'text': ['validationonly kind helpful', 'validationonly rude attack'],
                        'label': [0, 1], 'source': 'synthetic'})
    namespace = {'train_df': train, 'val_df': val, 'pd': pd,
                 'RANDOM_STATE': 42, 'display': lambda *args: None}
    with patch('joblib.dump') as save:
      exec(compile(cell.source.replace('RUN_LR_TUNING = False', 'RUN_LR_TUNING = True'),
                   str(path), 'exec'), namespace)
      save.assert_not_called()
    self.assertEqual(len(namespace['tuned_lr']['models']), 2)
    self.assertEqual(len(namespace['tuned_lr_metrics']), 4)
    self.assertTrue(namespace['tuned_lr_metrics']['threshold'].eq(0.5).all())
    for model in namespace['tuned_lr']['models'].values():
      self.assertNotIn('validationonly', model.named_steps['tfidfvectorizer'].vocabulary_)


if __name__ == '__main__':
  unittest.main()
