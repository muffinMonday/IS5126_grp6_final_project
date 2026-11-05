"""Small checks of fold isolation, selection and the notebook's opt-in path."""
import unittest
from unittest.mock import patch
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from functions.lr_tuning import bayes_tune_logistic_regressions, tune_logistic_regressions
from functions.tuning import bayes_search


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
    expected = [tuple(self.texts[tr]) for tr, _ in folds] * 12 + [tuple(self.texts)]
    self.assertEqual(fitted, expected * 2)
    for name, model in result['models'].items():
      lr = model.named_steps['logisticregression']
      self.assertEqual(lr.class_weight, 'balanced' if name.endswith('balanced') else None)
      log = result['cv_results'].query('model == @name')
      best = log.loc[log.mean_cv_pr_auc.idxmax()]
      chosen = log[log.selected].iloc[0]
      self.assertAlmostEqual(chosen.mean_cv_pr_auc, best.mean_cv_pr_auc)
      self.assertEqual(lr.C, chosen.C)
      tfidf = model.named_steps['tfidfvectorizer']
      self.assertEqual(tfidf.ngram_range, chosen.ngram_range)
      self.assertEqual(tfidf.min_df, chosen.min_df)
      self.assertEqual(len(log), 12)
      self.assertEqual(result['config']['selected_params'][name], {
        'logisticregression__C': lr.C, 'tfidfvectorizer__ngram_range': tfidf.ngram_range,
        'tfidfvectorizer__min_df': tfidf.min_df})
      self.assertEqual(model.predict_proba(['unseen validationonly']).shape, (1, 2))
      self.assertNotIn('validationonly', model.named_steps['tfidfvectorizer'].vocabulary_)

  def test_invalid_inputs(self):
    for grid in [[], [0], [float('nan')]]:
      with self.assertRaises(ValueError):
        tune_logistic_regressions(self.texts, self.labels, c_values=grid)
    with self.assertRaises(ValueError):
      tune_logistic_regressions(self.texts, self.labels, n_splits=13)
    for kwargs in [{'ngram_ranges': []}, {'min_dfs': [0]}, {'selection': 'holdout'}]:
      with self.assertRaises(ValueError):
        tune_logistic_regressions(self.texts, self.labels, **kwargs)

  def test_stable_selection_remains_available(self):
    result = tune_logistic_regressions(self.texts, self.labels, c_values=[0.1, 1],
                                      ngram_ranges=[(1, 2)], min_dfs=[2], selection='stable')
    for _, log in result['cv_results'].groupby('model'):
      best = log.loc[log.mean_cv_pr_auc.idxmax()]
      eligible = log[log.mean_cv_pr_auc >= best.mean_cv_pr_auc - best.std_cv_pr_auc]
      self.assertAlmostEqual(log[log.selected].iloc[0].std_cv_pr_auc, eligible.std_cv_pr_auc.min())

  def test_bayes_search_logs_trials_and_uses_the_lstm_selection_rule(self):
    result = bayes_tune_logistic_regressions(self.texts, self.labels, n_iter=5, n_initial_points=2, n_splits=3)
    self.assertEqual(result['config']['candidates_per_variant'], 5)
    for name, model in result['models'].items():
      log = result['cv_results'].query('model == @name').reset_index(drop=True)
      self.assertEqual(len(log), 5)
      self.assertEqual(log.selected.sum(), 1)
      folds = log[[f'fold_{i}_pr_auc' for i in (1, 2, 3)]].to_numpy()
      np.testing.assert_allclose(log.mean_cv_pr_auc, folds.mean(axis=1))
      np.testing.assert_allclose(log.std_cv_pr_auc, folds.std(axis=1, ddof=1))
      # same rule as bayes_search, applied to the same columns
      stub = bayes_search.__new__(bayes_search)
      stub.objective, stub.selection, stub.stability_tolerance = 'pr_auc', 'stable', 1.0
      stub.results = log.rename(columns={'mean_cv_pr_auc': 'avg_val_pr_auc', 'std_cv_pr_auc': 'std_val_pr_auc'})
      self.assertEqual(log.index[log.selected][0], stub._select_trial())
      chosen = log[log.selected].iloc[0]
      self.assertEqual(model.named_steps['logisticregression'].C, chosen.C)
      self.assertEqual(model.named_steps['tfidfvectorizer'].min_df, chosen.min_df)
      self.assertNotIn('validationonly', model.named_steps['tfidfvectorizer'].vocabulary_)

  def test_bayes_search_invalid_inputs(self):
    for kwargs in [{'selection': 'holdout'}, {'n_initial_points': 0}, {'c_range': (1, 0.1)},
                   {'min_df_range': (0, 3)}, {'ngram_ranges': []}, {'n_splits': 13}]:
      with self.assertRaises(ValueError):
        bayes_tune_logistic_regressions(self.texts, self.labels, n_iter=3, **kwargs)

  def test_notebook_validation_section_scores_saved_models_and_rejects_untuned_ones(self):
    import json, os, tempfile, joblib, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, f1_score,
                                 precision_recall_curve, precision_score, recall_score)
    from functions.baselines import build_tfidf_logreg
    path = Path(__file__).resolve().parents[1] / 'notebook/3_pooled_data.ipynb'
    cells = {c.id: c.source for c in nbformat.read(path, as_version=4).cells}
    val = pd.DataFrame({'text': ['validationonly kind helpful', 'validationonly rude attack'] * 5,
                        'label': [0, 1] * 5, 'source': 'synthetic'})
    manifest = {'split_sha256': {'train': 'abc'}}
    shown = []
    with tempfile.TemporaryDirectory() as tmp:
      base = os.path.join(tmp, 'baselines')
      os.makedirs(base)
      for name, weight in [('tfidf_logreg', None), ('tfidf_logreg_balanced', 'balanced')]:
        joblib.dump(build_tfidf_logreg(weight, 42).fit(self.texts, self.labels), os.path.join(base, f'{name}.joblib'))
      def write(config):
        with open(os.path.join(base, 'baselines_manifest.json'), 'w') as f:
          json.dump({'train_split_sha256': 'abc', 'config': config}, f)
      load = cells['lr-val-load'].replace("'../models/baselines'", repr(base))
      namespace = {'os': os, 'json': json, 'joblib': joblib, 'pd': pd, 'np': np, 'plt': plt,
                   'manifest': manifest, 'val_df': val, 'THRESHOLD': 0.5, 'display': shown.append,
                   'ConfusionMatrixDisplay': ConfusionMatrixDisplay, 'average_precision_score': average_precision_score,
                   'f1_score': f1_score, 'precision_recall_curve': precision_recall_curve,
                   'precision_score': precision_score, 'recall_score': recall_score}
      write({'search': 'grid'})
      with self.assertRaisesRegex(ValueError, 'not the Bayesian'):
        exec(compile(load, 'load', 'exec'), namespace)
      write({'search': 'bayes'})
      exec(compile(load, 'load', 'exec'), namespace)
      with patch.object(plt, 'show'):
        for cell in ('lr-val-metrics', 'lr-val-pr', 'lr-val-cutoffs'):
          exec(compile(cells[cell], cell, 'exec'), namespace)
      plt.close('all')
      manifest['split_sha256']['train'] = 'different'
      with self.assertRaisesRegex(ValueError, 'different training split'):
        exec(compile(load, 'load', 'exec'), namespace)
    metrics = namespace['val_metrics']
    self.assertEqual(list(metrics.index), ['LR (standard)', 'LR (balanced)'])
    self.assertTrue(metrics['threshold'].eq(0.5).all())
    self.assertTrue(metrics['n_comments'].eq(10).all())
    self.assertEqual(len(shown), 3)  # metrics table and one cutoff table per model

  def test_notebook_defaults_do_not_train_or_save(self):
    path = Path(__file__).resolve().parents[1] / 'notebook/3_pooled_data.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    cell = next(c for c in notebook.cells if c.id == 'lr-tuning-run')
    with patch('functions.lr_tuning.bayes_tune_logistic_regressions') as tune:
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
      source = cell.source.replace('RUN_LR_TUNING = False', 'RUN_LR_TUNING = True')
      source = source.replace("'n_iter': 30, 'n_initial_points': 10", "'n_iter': 4, 'n_initial_points': 2")
      exec(compile(source, str(path), 'exec'), namespace)
      save.assert_not_called()
    self.assertEqual(len(namespace['tuned_lr']['models']), 2)
    self.assertEqual(len(namespace['tuned_lr']['cv_results']), 8)
    self.assertEqual(namespace['tuned_lr']['config']['selection'], 'stable')
    self.assertEqual(namespace['tuned_lr']['config']['search'], 'bayes')
    self.assertEqual(len(namespace['tuned_lr_metrics']), 4)
    self.assertTrue(namespace['tuned_lr_metrics']['threshold'].eq(0.5).all())
    for model in namespace['tuned_lr']['models'].values():
      self.assertNotIn('validationonly', model.named_steps['tfidfvectorizer'].vocabulary_)


if __name__ == '__main__':
  unittest.main()
